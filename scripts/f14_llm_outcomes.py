"""F14 -- RULE 1 FOR TEXT SUPERVISION: do LLMs predict the outcome of a designed probe?

Video models failed as supervision (judge, surprise, FPT, MYRIAD, Wan). The remaining
large-model channel is language: commonsense about objects ("a mug does not tip when
nudged"). Before any such statement is turned into a constraint on theta, the LLM's
predictions must be shown to be right where the answer is KNOWN.

GROUND TRUTH IS ANALYTIC, NOT THE SIMULATOR. Two simulator artefacts would grade correct
real-world answers as wrong: no static friction (at 15 deg every asset creeps 3-10 cm where
a real one stays put) and a tumble flag that fires on nearly every bowl shove. Instead,
textbook rigid-body criteria on each mesh's geometry:

  incline at theta:  stays    if tan(theta) < mu_eff  and tan(theta) < b/h
                     slides   if tan(theta) > mu_eff  and mu_eff < b/h
                     topples  otherwise
  shove on a level table at v:  tips over if mu_eff > b/h, else slides v^2 / (2 mu_eff g)

  b = horizontal distance from the centre of mass to the leading (downhill) base edge,
  h = centre-of-mass height. Centre of mass: area-weighted shell for thin-walled hollow
  objects (bowl, pot, teapot, sorter, box), voxel-filled volume for solid ones (books,
  shark) -- the real objects' build, not the simulator's vertex mean.

A question is SCORED only if its outcome is the same for every mu_eff in [0.30, 0.60]
(plausible pair friction of these materials on wood). Others are AMBIGUOUS: the LLM's
answer there is what it would supervise, with nothing to grade it against.

Models via the Letsur gateway (OpenAI-compatible; key in ~/.config/letsur/key, never in
this file). Each gets the scene image plus a close-up crop, the object's dimensions, and
the scenario; it returns probabilities over the options as JSON.

    W scripts/f14_llm_outcomes.py build
    C scripts/f14_llm_outcomes.py ask [--models a,b,c]
    W scripts/f14_llm_outcomes.py score
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

MU_EFF = (0.30, 0.60)
TILTS = (10, 15, 20, 25, 30, 35, 40, 45)
SPEEDS = (0.5, 1.0, 2.0, 3.0)                 # m/s
BINS = ((0, 5), (5, 20), (20, 80), (80, 1e9))   # cm
HOLLOW = {"bowl", "pot", "teapot", "sorter", "box"}
EXCLUDE = {"shark"}           # rests on a fin tip as authored; no well-defined base
NAMES = {"box": "cardboard box", "sorter": "blue plastic desk file sorter",
         "bowl": "wooden bowl", "pot": "brass pot", "teapot": "white porcelain teapot",
         "books": "set of hardcover encyclopedia books", "shark": "toy great white shark"}
Q = Path("outputs/scene/f14_questions.json")
A = Path("outputs/scene/f14_answers.json")
RES = Path("docs/results/f14_llm_outcomes.json")
MODELS = ("claude-opus-5-5", "gpt-6-astra", "gemini-3.1-pro-preview")


def geometry(tag, asset):
    import o13_tilted as T
    from src.data.assets import load_asset
    cat, _, nm = asset.rpartition("/")
    m = load_asset(cat, nm)
    m = m.copy(); m.apply_scale(T.scale_for(asset))
    V = np.asarray(m.vertices)
    if tag in HOLLOW:
        c = np.asarray(m.triangles_center); a = np.asarray(m.area_faces)
        com = (c * a[:, None]).sum(0) / a.sum()
        kind = "shell (area-weighted)"
    else:
        # surface -> voxels by dense surface sampling, then exterior flood fill (numpy
        # only: neither env has scipy, which trimesh's voxelizer needs)
        pitch = float(np.ptp(V, 0).max()) / 80
        import trimesh
        pts = np.asarray(trimesh.sample.sample_surface(m, 400000, seed=0)[0])
        lo = V.min(0) - pitch
        ijk = np.floor((pts - lo) / pitch).astype(int)
        shape = ijk.max(0) + 2
        shell = np.zeros(shape, bool); shell[tuple(ijk.T)] = True
        solid = _fill(shell)
        com = lo + (np.argwhere(solid) + 0.5).mean(0) * pitch
        kind = "solid (voxel-filled)"
    z0 = V[:, 2].min()
    # contact patch = vertices within 5 mm of the lowest point (settled contact). At 2 mm the
    # shark touches on a fin tip alone, behind its centre of mass -- an authored pose it
    # could not rest in; it is EXCLUDED from scoring rather than given an invented support.
    sup = V[V[:, 2] < z0 + 0.005]
    b = float(sup[:, 0].max() - com[0])            # to the +x (downhill / leading) edge
    h = float(com[2] - z0)
    vm = V.mean(0)
    return {"com": com.tolist(), "kind": kind, "b": b, "h": h, "b_over_h": b / h,
            "ext_cm": (np.ptp(V, 0) * 100).round(1).tolist(),
            "vertex_mean_h": float(vm[2] - z0), "vertex_mean_b_over_h":
            float((sup[:, 0].max() - vm[0]) / (vm[2] - z0))}


def _fill(shell):
    """Solid = everything not reachable from outside (exterior flood fill, 6-connected).
    Written here because the warp env has no scipy."""
    from collections import deque
    S = np.pad(shell.astype(bool), 1)
    out = np.zeros_like(S)
    dq = deque([(0, 0, 0)]); out[0, 0, 0] = True
    nb = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while dq:
        x, y, z = dq.popleft()
        for dx, dy, dz in nb:
            a, b, c = x + dx, y + dy, z + dz
            if 0 <= a < S.shape[0] and 0 <= b < S.shape[1] and 0 <= c < S.shape[2] \
                    and not out[a, b, c] and not S[a, b, c]:
                out[a, b, c] = True; dq.append((a, b, c))
    return (~out)[1:-1, 1:-1, 1:-1]


def incline(tilt, mu, boh):
    t = np.tan(np.radians(tilt))
    if t < mu and t < boh:
        return "stays"
    if t > mu and mu < boh:
        return "slides"
    return "topples"


def shove(v, mu, boh):
    if mu > boh:
        return "tips over"
    d = 100 * v * v / (2 * mu * 9.81)
    lo, hi = next((a, b) for a, b in BINS if a <= d < b)
    return f"{lo}-{hi if hi < 1e8 else 'more'} cm" if hi < 1e8 else f"more than {lo} cm"


def shove_opts():
    return ["less than 5 cm", "5-20 cm", "20-80 cm", "more than 80 cm", "tips over"]


def _shove_label(v, mu, boh):
    if mu > boh:
        return "tips over"
    d = 100 * v * v / (2 * mu * 9.81)
    return ("less than 5 cm" if d < 5 else "5-20 cm" if d < 20 else
            "20-80 cm" if d < 80 else "more than 80 cm")


def do_build(args):
    import o13_tilted as T
    mus = np.linspace(*MU_EFF, 31)
    qs, geo = [], {}
    for tag, asset in T.ASSETS:
        if tag in EXCLUDE:
            continue
        g = geometry(tag, asset); geo[tag] = g
        print(f"  {tag:<7} {g['kind']:<22} h={100*g['h']:5.1f} cm  b={100*g['b']:5.1f} cm  "
              f"b/h={g['b_over_h']:.2f}   (sim vertex-mean b/h {g['vertex_mean_b_over_h']:.2f})")
        for tl in TILTS:
            outs = [incline(tl, mu, g["b_over_h"]) for mu in mus]
            qs.append({"id": f"{tag}_incline{tl}", "tag": tag, "kind": "incline", "tilt": tl,
                       "options": ["stays", "slides", "topples"],
                       "dist": {o: outs.count(o) / len(outs) for o in set(outs)},
                       "robust": len(set(outs)) == 1, "truth": outs[0] if len(set(outs)) == 1
                       else None})
        for v in SPEEDS:
            outs = [_shove_label(v, mu, g["b_over_h"]) for mu in mus]
            qs.append({"id": f"{tag}_shove{v}", "tag": tag, "kind": "shove", "v": v,
                       "options": shove_opts(),
                       "dist": {o: outs.count(o) / len(outs) for o in set(outs)},
                       "robust": len(set(outs)) == 1, "truth": outs[0] if len(set(outs)) == 1
                       else None})
    Q.write_text(json.dumps({"geometry": geo, "questions": qs}, indent=1))
    n = sum(q["robust"] for q in qs)
    print(f"\n{len(qs)} questions, {n} with an unambiguous answer over mu_eff in {MU_EFF}")
    from collections import Counter
    print("  robust truths:", dict(Counter(q["truth"] for q in qs if q["robust"])))
    return 0


def _img_b64(path, crop=None, scale=1):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop).resize(((crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale))
    b = io.BytesIO(); im.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def _images(tag):
    """Scene (object resting on the level table) + a 3x close-up of the object."""
    import f4_fpt_joint as F
    from src.render.crop import project_points
    d = Path(f"outputs/scene/f7_{tag}")
    spec = json.loads((d / "spec.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    uv, ok = project_points(F.cam_at(0), np.asarray(meta[spec[0]["key"]][0], float))
    uv = uv[ok]
    x0, y0 = (uv.min(0) - 25).astype(int); x1, y1 = (uv.max(0) + 25).astype(int)
    p = d / "sim_s0@a" / "f0000.png"
    return _img_b64(p), _img_b64(p, (max(x0, 0), max(y0, 0), x1, y1), 3)


def _prompt(q, g):
    obj = NAMES[q["tag"]]
    dims = " x ".join(f"{x:.0f}" for x in g["ext_cm"])
    head = (f"The images show a {obj} (about {dims} cm, width x depth x height) resting on "
            f"its base on a wooden table; the second image is a close-up. Assume it is an "
            f"ordinary real object of this kind.\n\n")
    if q["kind"] == "incline":
        sc = (f"Scenario: the object is placed, resting on its base exactly as shown, on a "
              f"plain wooden plank tilted at {q['tilt']} degrees, and released from rest. "
              f"What happens in the next second?\n"
              f"- stays: it remains where it is\n- slides: it slides down the plank without "
              f"tipping over\n- topples: it tips over / tumbles")
    else:
        sc = (f"Scenario: on the level wooden table, the object is given a quick horizontal "
              f"shove so that it starts moving at {q['v']} m/s along the table. What happens?"
              f"\n- less than 5 cm / 5-20 cm / 20-80 cm / more than 80 cm: it slides that far "
              f"before stopping, staying upright\n- tips over: it tips over instead")
    return (head + sc + "\n\nReason about the physics briefly, then end with ONLY a JSON object "
            f"giving a probability for each option, keys exactly {q['options']}, summing to 1.")


def do_ask(args):
    from openai import OpenAI
    cli = OpenAI(api_key=open(os.path.expanduser("~/.config/letsur/key")).read().strip(),
                 base_url="https://gw.letsur.ai/v1")
    D = json.loads(Q.read_text())
    models = args.models.split(",") if args.models else list(MODELS)
    imgs = {}
    for m in models:
        Am = A.with_name(f"f14_answers_{m}.json")
        ans = json.loads(Am.read_text()) if Am.exists() else {}
        ans.setdefault(m, {})
        for q in D["questions"]:
            if q["id"] in ans[m] and ans[m][q["id"]].get("probs"):
                continue
            if q["tag"] not in imgs:
                imgs[q["tag"]] = _images(q["tag"])
            full, crop = imgs[q["tag"]]
            msg = [{"role": "user", "content": [
                {"type": "text", "text": _prompt(q, D["geometry"][q["tag"]])},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + full}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + crop}}]}]
            kw = {"max_completion_tokens": 6000} if m.startswith(("gpt", "o")) else \
                 {"max_tokens": 6000}
            txt, probs = None, None
            for attempt in range(3):
                try:
                    r = cli.chat.completions.create(model=m, messages=msg, **kw)
                    txt = r.choices[0].message.content or ""
                    probs = _parse(txt, q["options"])
                    if probs:
                        break
                except Exception as e:
                    txt = f"ERROR {e}"; time.sleep(5)
            ans[m][q["id"]] = {"probs": probs, "text": txt[-1500:] if txt else txt}
            print(f"  {m:<24}{q['id']:<20}{'ok' if probs else 'PARSE FAIL'}", flush=True)
            Am.write_text(json.dumps(ans, indent=1))
    return 0


def _parse(txt, options):
    import re
    for blk in reversed(re.findall(r"\{[^{}]*\}", txt or "")):
        try:
            d = json.loads(blk)
        except Exception:
            continue
        p = {o: float(d.get(o, 0.0)) for o in options}
        s = sum(p.values())
        if s > 0:
            return {o: v / s for o, v in p.items()}
    return None


def do_score(args):
    D = json.loads(Q.read_text())
    ans = {}
    for f in sorted(A.parent.glob("f14_answers_*.json")):
        ans.update(json.loads(f.read_text()))
    qs = {q["id"]: q for q in D["questions"]}
    rob = [q for q in qs.values() if q["robust"]]
    res = {}
    print(f"{len(rob)} unambiguous questions (of {len(qs)}).")
    # baselines: chance, and always-the-most-common-truth per scenario kind
    from collections import Counter
    maj = {k: Counter(q["truth"] for q in rob if q["kind"] == k).most_common(1)[0][0]
           for k in ("incline", "shove")}
    base_acc = np.mean([q["truth"] == maj[q["kind"]] for q in rob])
    chance = np.mean([1 / len(q["options"]) for q in rob])
    print(f"  chance {100*chance:.0f}%   majority-per-kind {100*base_acc:.0f}%")
    print(f"\n{'model':<26}{'n':>4}{'acc':>7}{'incline':>9}{'shove':>7}{'logloss':>9}"
          f"{'p(truth)':>10}")
    for m, am in ans.items():
        rows = [(q, am[q["id"]]["probs"]) for q in rob if am.get(q["id"], {}).get("probs")]
        if not rows:
            continue
        acc = [max(p, key=p.get) == q["truth"] for q, p in rows]
        ll = [-np.log(max(p.get(q["truth"], 0), 1e-3)) for q, p in rows]
        pt = [p.get(q["truth"], 0) for q, p in rows]
        ai = [a for (q, _), a in zip(rows, acc) if q["kind"] == "incline"]
        as_ = [a for (q, _), a in zip(rows, acc) if q["kind"] == "shove"]
        res[m] = {"n": len(rows), "acc": float(np.mean(acc)), "acc_incline": float(np.mean(ai)),
                  "acc_shove": float(np.mean(as_)), "logloss": float(np.mean(ll)),
                  "p_truth": float(np.mean(pt)),
                  "wrong": [q["id"] + f": truth {q['truth']}, said {max(p, key=p.get)}"
                            for (q, p), a in zip(rows, acc) if not a]}
        print(f"{m:<26}{len(rows):>4}{100*np.mean(acc):>6.0f}%{100*np.mean(ai):>8.0f}%"
              f"{100*np.mean(as_):>6.0f}%{np.mean(ll):>9.2f}{np.mean(pt):>10.2f}")
    for m in res:
        print(f"\n{m} wrong on:"); [print("   ", w) for w in res[m]["wrong"]]
    RES.write_text(json.dumps({"baselines": {"chance": chance, "majority": base_acc},
                               "models": res}, indent=1, default=float))
    print("wrote", RES)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(f=do_build)
    s = sub.add_parser("ask"); s.set_defaults(f=do_ask); s.add_argument("--models", default="")
    sub.add_parser("score").set_defaults(f=do_score)
    a = ap.parse_args()
    sys.exit(a.f(a))
