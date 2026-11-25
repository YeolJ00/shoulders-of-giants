"""E1 -- THE held-out test: do parameters assigned to an UNSEEN real object make it move like
the real thing?

Every friction and density number so far was measured on real Physics 101 objects and
transferred by MATERIAL. That transfer has never been tested against an object the
pipeline did not see. Here each real object is held out in turn; its parameters are
predicted without any of its own trials, and its REAL tracked slides are the judge.

Friction predictors for held-out object O (20 deg ramp):
  oracle         O's own OTHER trials (leave one trial out) -- upper bound
  material       median over OTHER objects of O's true material   (leave one object out)
  pipeline       material named by an LLM from O's PHOTO, then median over other objects of
                 that material -- what we would do for an authored asset
  generic 0.40   the value LLMs implied for every object (f14)
  global         median over all other objects, any material (a dataset-wide prior)
Scores:
  trajectory     predicted s(t) = s0 + a(mu) (t-t0)^2 / 2 against the real tracked s(t) on the
                 ramp segment; ONLY t0 and s0 are aligned, never the acceleration; RMS in mm
                 (Newton reproduces this law to <1 mm on these clips, scripts/p4_sim_loop.py)
  outcome        slides / stays at 10 and 20 deg, for EVERY object incl. never-slide ones
Mass: density of the material from the other objects (or areal density for the hollow
cardboard cubes) x O's real volume, vs O's measured mass; baseline = an LLM's direct guess
from the photo and the volume.

    C scripts/e1_heldout.py classify      (LLM, cached)
    W scripts/e1_heldout.py score
"""
import argparse
import base64
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
P101 = REPO / "data/phys101"
G = 9.81
MATS = ["cardboard", "dough", "foam", "h_rubber", "h_wood", "m_coin", "m_pole", "p_block",
        "p_doll", "p_ring", "p_toy", "porcelain", "rubber", "w_block", "w_pole"]
DESC = {"cardboard": "cardboard (box)", "dough": "dough / modelling clay", "foam": "foam",
        "h_rubber": "hollow rubber", "h_wood": "hollow wood", "m_coin": "metal coin",
        "m_pole": "metal pole/rod", "p_block": "plastic block", "p_doll": "plastic doll",
        "p_ring": "plastic ring", "p_toy": "plastic toy", "porcelain": "porcelain / ceramic",
        "rubber": "solid rubber", "w_block": "wooden block", "w_pole": "wooden pole/rod"}
LLM = "claude-opus-5-5"
CACHE = REPO / "outputs/scene/e1_llm.json"
OUT = REPO / "docs/results/e1_heldout.json"


def objects_table():
    import zipfile
    z = zipfile.ZipFile(P101 / "phys101_v1.0.zip")
    P = lambda f: {l.split()[0]: float(l.split()[1]) for l in z.read(f).decode().split("\n")
                   if l.strip()}
    return P("phys101/objects/mass"), P("phys101/objects/vol")


def do_classify(args):
    from openai import OpenAI
    cli = OpenAI(api_key=open(os.path.expanduser("~/.config/letsur/key")).read().strip(),
                 base_url="https://gw.letsur.ai/v1")
    mass, vol = objects_table()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    opts = "; ".join(f"{k} = {v}" for k, v in DESC.items())
    for obj in sorted(mass):
        if obj in cache and cache[obj].get("material"):
            continue
        img = base64.b64encode((P101 / f"phys101/objects/{obj}.JPG").read_bytes()).decode()
        prompt = (f"The photo shows one small everyday object (volume {vol[obj]:.0f} ml). "
                  f"(1) Which material class is it? Choose exactly one of: {opts}. "
                  f"(2) Estimate its mass in grams. Reply with ONLY a JSON object "
                  f'{{"material": "<class key>", "mass_g": <number>}}.')
        txt = None
        for _ in range(3):
            try:
                r = cli.chat.completions.create(model=LLM, max_tokens=300, messages=[{
                    "role": "user", "content": [{"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]}])
                txt = r.choices[0].message.content
                import re
                d = json.loads(re.findall(r"\{[^{}]*\}", txt)[-1])
                cache[obj] = {"material": d["material"], "mass_g": float(d["mass_g"]), "raw": txt}
                break
            except Exception as e:
                cache[obj] = {"error": str(e)[:200], "raw": txt}; time.sleep(3)
        print(f"  {obj:<14} -> {cache[obj].get('material')}  {cache[obj].get('mass_g')} g", flush=True)
        CACHE.write_text(json.dumps(cache, indent=1))
    return 0


def do_classify_mu(args):
    """FAIR LLM baseline: a direct friction estimate for THIS surface, from the photo.
    (f14's ~0.40 was asked about a wooden plank; the Physics 101 ramp is a smooth white
    laminated board, so comparing against it would be unfair to the LLM.)"""
    from openai import OpenAI
    import re
    cli = OpenAI(api_key=open(os.path.expanduser("~/.config/letsur/key")).read().strip(),
                 base_url="https://gw.letsur.ai/v1")
    mass, vol = objects_table()
    cache = json.loads(CACHE.read_text())
    for obj in sorted(mass):
        if cache.get(obj, {}).get("mu") is not None:
            continue
        img = base64.b64encode((P101 / f"phys101/objects/{obj}.JPG").read_bytes()).decode()
        prompt = (f"The photo shows one small everyday object (volume {vol[obj]:.0f} ml, mass "
                  f"{mass[obj]:.0f} g). It is placed on a smooth, white, laminated board "
                  f"(whiteboard-like surface) tilted at 20 degrees and released. Estimate the "
                  f"KINETIC friction coefficient between this object and that board. Reply "
                  f'with ONLY a JSON object {{"mu": <number>}}.')
        for _ in range(3):
            try:
                r = cli.chat.completions.create(model=LLM, max_tokens=200, messages=[{
                    "role": "user", "content": [{"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]}])
                cache[obj]["mu"] = float(json.loads(re.findall(r"\{[^{}]*\}", r.choices[0].message.content)[-1])["mu"])
                break
            except Exception as e:
                time.sleep(3)
        print(f"  {obj:<14} mu {cache[obj].get('mu')}", flush=True)
        CACHE.write_text(json.dumps(cache, indent=1))
    return 0


def trials():
    """Per real trial: object, material, angle, outcome, mu, s(t), fps, ramp segment."""
    mat = lambda o: o.rsplit("_", 1)[0]
    rows = []
    for r in json.loads((REPO / "docs/results/p1_ramp_friction.json").read_text()):
        o, st, _ = r["clip"].split("/")
        ang = 20 if st.startswith("20") else 10
        if ang == 20 and r.get("fit"):
            rows.append(dict(obj=o, mat=mat(o), ang=20, outcome="slides", mu=r["fit"]["mu"],
                             s=r["s"], fps=r["fps"], start=r["start"], peak=r["peak"]))
        elif ang == 10:
            rows.append(dict(obj=o, mat=mat(o), ang=10, outcome="stays"))   # checked by eye, p1
    for r in json.loads((REPO / "docs/results/p2_ramp_all.json").read_text()):
        o, st, _ = r["clip"].split("/")
        ang = 20 if st.startswith("20") else 10
        oc = r["outcome"]
        if oc == "slides" and r.get("impossible"):
            continue
        if oc in ("slides", "stays", "rolls"):
            d = dict(obj=o, mat=mat(o), ang=ang, outcome=oc)
            if oc == "slides":
                d.update(mu=r["mu"], s=r["s"], fps=r["fps"], start=r["start"], peak=r["peak"])
            rows.append(d)
    return rows


def traj_rms(row, mu, ang=20.0):
    """RMS (mm) of the real ramp-segment s(t) against s0 + a(mu) (t-t0)^2/2, t0 and s0 fitted."""
    s = np.array(row["s"]); t = np.arange(len(s)) / row["fps"]
    seg = slice(max(row["start"] - 1, 0), row["peak"] + 1)
    ts, ss = t[seg], s[seg]
    al = np.radians(ang)
    a = max(G * (np.sin(al) - mu * np.cos(al)), 0.0)
    best = np.inf
    for t0 in np.linspace(ts[0] - 0.3, ts[0] + 0.05, 71):
        tau = np.clip(ts - t0, 0, None)
        pred = 0.5 * a * tau ** 2
        s0 = np.mean(ss - pred)
        best = min(best, float(np.sqrt(np.mean((ss - s0 - pred) ** 2))))
    return 1000 * best


def do_score(args):
    mass, vol = objects_table()
    llm = json.loads(CACHE.read_text())
    R = trials()
    slides20 = [r for r in R if r["ang"] == 20 and r["outcome"] == "slides"]
    obj_mu = collections.defaultdict(list)
    for r in slides20:
        obj_mu[r["obj"]].append(r["mu"])
    obj_med = {o: float(np.median(v)) for o, v in obj_mu.items()}
    mat_of = lambda o: o.rsplit("_", 1)[0]

    def mat_mu(m, exclude):
        v = [obj_med[o] for o in obj_med if mat_of(o) == m and o != exclude]
        return float(np.median(v)) if v else None

    # --- friction: trajectory error per held-out object
    preds = {"oracle": [], "material": [], "pipeline": [], "LLM direct mu": [],
             "generic 0.40": [], "global": []}
    per_obj = {}
    for o in sorted(obj_mu):
        rs = [r for r in slides20 if r["obj"] == o]
        lm = llm.get(o, {}).get("material")
        others = [obj_med[x] for x in obj_med if x != o]
        cand = {"material": mat_mu(mat_of(o), o),
                "pipeline": mat_mu(lm, o) if lm in MATS else None,
                "LLM direct mu": llm.get(o, {}).get("mu"),
                "generic 0.40": 0.40, "global": float(np.median(others))}
        rec = {"material_true": mat_of(o), "material_llm": lm, "mu_own": obj_med[o]}
        for i, r in enumerate(rs):
            loto = [x["mu"] for j, x in enumerate(rs) if j != i]
            mus = {"oracle": float(np.median(loto)) if loto else None, **cand}
            for k, mu in mus.items():
                if mu is not None:
                    preds[k].append((o, traj_rms(r, mu), abs(mu - r["mu"])))
        rec.update({k: v for k, v in cand.items()})
        per_obj[o] = rec
    print("FRICTION -- held-out real objects, 20 deg ramp; predicted slide vs the REAL tracked slide")
    print(f"  {'predictor':<14}{'trials':>7}{'objects':>8}{'traj RMS mm (median)':>22}{'p90':>8}{'|d mu| median':>15}")
    res = {"friction": {}}
    for k, v in preds.items():
        e = np.array([x[1] for x in v]); dm = np.array([x[2] for x in v])
        res["friction"][k] = {"n_trials": len(v), "n_objects": len({x[0] for x in v}),
                              "rms_mm_median": float(np.median(e)), "rms_mm_p90": float(np.percentile(e, 90)),
                              "dmu_median": float(np.median(dm))}
        print(f"  {k:<14}{len(v):>7}{len({x[0] for x in v}):>8}{np.median(e):>22.1f}{np.percentile(e, 90):>8.1f}{np.median(dm):>15.3f}")
    # --- outcome at 10 and 20 deg (stays / slides), every object incl. never-slide materials
    print("\nOUTCOME -- does the predicted mu say slide / stay correctly? (per object, majority of its trials)")
    oc = collections.defaultdict(lambda: collections.Counter())
    for r in R:
        if r["outcome"] in ("slides", "stays"):
            oc[(r["obj"], r["ang"])][r["outcome"]] += 1
    res["outcome"] = {}
    for k in ("material", "pipeline", "LLM direct mu", "generic 0.40", "global"):
        hit = n = 0
        for (o, ang), c in oc.items():
            truth = c.most_common(1)[0][0]
            m_true, m_llm = mat_of(o), llm.get(o, {}).get("material")
            if k == "material":
                mu = mat_mu(m_true, o)
            elif k == "pipeline":
                mu = mat_mu(m_llm, o) if m_llm in MATS else None
            elif k == "generic 0.40":
                mu = 0.40
            elif k == "LLM direct mu":
                mu = llm.get(o, {}).get("mu")
                if mu is None:
                    continue
            else:
                mu = float(np.median([obj_med[x] for x in obj_med if x != o]))
            if mu is None:
                # no OTHER object of that material ever slid: the material is a never-slider
                pred = "stays"
            else:
                pred = "slides" if mu < np.tan(np.radians(ang)) else "stays"
            hit += pred == truth; n += 1
        res["outcome"][k] = {"acc": hit / n, "n": n}
        print(f"  {k:<14} {100*hit/n:5.1f}%  of {n} object x angle cases")
    # --- material classification from the photo
    acc = np.mean([llm.get(o, {}).get("material") == mat_of(o) for o in mass])
    conf = collections.Counter((mat_of(o), llm.get(o, {}).get("material")) for o in mass
                               if llm.get(o, {}).get("material") != mat_of(o))
    print(f"\nLLM material from photo: {100*acc:.1f}% of {len(mass)} objects; confusions: "
          + ", ".join(f"{a}->{b} x{c}" for (a, b), c in conf.most_common(6)))
    res["llm_material_acc"] = float(acc)
    # --- mass
    print("\nMASS -- held-out object's mass from material density x its real volume")
    errs = {"material density (volumetric)": [], "build-aware (areal for cardboard)": [],
            "LLM direct guess": []}
    for o in mass:
        m = mat_of(o)
        dens = [mass[x] / vol[x] for x in mass if mat_of(x) == m and x != o]
        if not dens:
            continue
        pv = float(np.median(dens)) * vol[o]
        errs["material density (volumetric)"].append(abs(np.log(pv / mass[o])))
        if m == "cardboard":
            ar = [mass[x] / (6 * vol[x] ** (2 / 3)) for x in mass if mat_of(x) == m and x != o]
            pb = float(np.median(ar)) * 6 * vol[o] ** (2 / 3)
        else:
            pb = pv
        errs["build-aware (areal for cardboard)"].append(abs(np.log(pb / mass[o])))
        g = llm.get(o, {}).get("mass_g")
        if g:
            errs["LLM direct guess"].append(abs(np.log(max(g, 1e-3) / mass[o])))
    res["mass"] = {}
    for k, v in errs.items():
        v = np.array(v)
        res["mass"][k] = {"n": len(v), "median_factor": float(np.exp(np.median(v))),
                          "within_25pct": float(np.mean(v < np.log(1.25)))}
        print(f"  {k:<36} n={len(v):>3}  median error x{np.exp(np.median(v)):.2f}   within 25%: {100*np.mean(v < np.log(1.25)):.0f}%")
    res["per_object"] = per_obj
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("classify").set_defaults(f=do_classify)
    sub.add_parser("classify_mu").set_defaults(f=do_classify_mu)
    sub.add_parser("score").set_defaults(f=do_score)
    a = ap.parse_args()
    sys.exit(a.f(a))
