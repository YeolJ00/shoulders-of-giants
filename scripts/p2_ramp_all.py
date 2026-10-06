"""P2 -- ramp friction for ALL Physics 101 materials, with slide / roll / stay classified.

Extends p1 (cardboard cubes) to the other 14 materials. Two things p1 could ignore:

  ROLLING.  Poles, rings and some toys roll. A rolling cylinder accelerates at ~2/3 g sin a,
  which the sliding formula turns into a plausible-looking mu ~0.12 -- a wrong number that
  looks right. Each clip's object tracks are fitted with a 2-D rigid transform between the
  start and the end of the ramp segment; rotation > 20 deg => ROLL/TUMBLE, excluded from mu.
  STAYING.  When the object does not move, p1's "furthest-travelling points" latched onto the
  hand and produced impossible negative mu. Here a clip counts as a SLIDE only if a cluster
  of tracks travels > 8 cm along the ramp direction (measured on the cardboard clips); else
  STAYS -- read as a bound, mu_static > tan(alpha).

Scale: one camera scale for every clip, 1661 px/m, measured in p1 from nine cubes' own
edges (1633-1724, +-3%). At 20 deg mu moves only 0.008 per 10% scale error. The nominal ramp
angle (+-1 deg -> +-0.02 in mu) remains the dominant uncertainty, as for cardboard.

Sampling: up to 5 trials per object at 20 deg, 2 at 10 deg.

    C scripts/p2_ramp_all.py run
    W scripts/p2_ramp_all.py summary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO / "scripts"))
import p1_ramp_friction as P1

SCALE = 1661.0                      # px per metre, p1
RAMP_DIR_DEG = 16.0                 # image direction of down-ramp motion, p1 (+-12 deg)
MIN_SLIDE_M = 0.08
ROLL_DEG = 20.0
N20, N10 = 5, 2
OUT = REPO / "docs/results/p2_ramp_all.json"


def rigid_rot_deg(A, B):
    """Rotation of the 2-D rigid transform best mapping point set A onto B (Kabsch)."""
    a, b = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = (U @ Vt).T
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def classify(clip, fr, fps, tr):
    T = len(fr)
    disp = tr[-1] - tr[0]
    dist = np.linalg.norm(disp, axis=1)
    ang = np.degrees(np.arctan2(disp[:, 1], disp[:, 0]))
    ok = (np.abs(ang - RAMP_DIR_DEG) < 12) & (dist > MIN_SLIDE_M * SCALE)
    alpha = np.radians(20.0 if "/20_" in clip else 10.0)
    if ok.sum() < 3:
        return {"clip": clip, "outcome": "stays", "n_pts": int(ok.sum()),
                "bound": f"mu_static > {np.tan(alpha):.3f}"}
    sel = np.flatnonzero(ok)
    path = np.median(tr[:, sel], 1)
    d = np.median(disp[sel], 0); d /= np.linalg.norm(d)
    s = ((path - path[0]) @ d) / SCALE
    t = np.arange(T) / fps
    v = np.gradient(s, t)
    start = int(np.argmax(s > 0.005)); peak = int(np.argmax(v))
    res = {"clip": clip, "n_pts": int(len(sel)), "s": s.tolist(), "fps": fps,
           "start": start, "peak": peak}
    if peak - start < 4 or len(sel) < 3:
        res["outcome"] = "unclear"; return res
    rot = rigid_rot_deg(tr[start, sel], tr[peak, sel])
    res["rot_deg"] = rot
    if abs(rot) > ROLL_DEG:
        res["outcome"] = "rolls"; return res
    best = None
    for t0 in np.linspace(t[max(start - 6, 0)], t[start], 25):
        tt = t[start - 1:peak + 1] - t0
        m = tt > 0
        A = np.stack([np.ones(m.sum()), 0.5 * tt[m] ** 2], 1)
        coef, *_ = np.linalg.lstsq(A, s[start - 1:peak + 1][m], rcond=None)
        r = np.sqrt(np.mean((A @ coef - s[start - 1:peak + 1][m]) ** 2))
        if best is None or r < best[0]:
            best = (r, float(t0), float(coef[0]), float(coef[1]))
    rms, t0, s0, a = best
    mu = float(np.tan(alpha) - a / (P1.G * np.cos(alpha)))
    res.update({"outcome": "slides", "a": a, "rms_m": rms, "mu": mu,
                "impossible": bool(mu < 0)})   # faster than frictionless => not a slide
    return res


def do_run(args):
    clips = []
    for obj in sorted(p.name for p in P1.ROOT.iterdir() if p.is_dir()):
        if obj.startswith("cardboard"):
            continue                               # done in p1
        for ang, n in (("20", N20), ("10", N10)):
            cs = sorted(str(p.parent.relative_to(P1.ROOT))
                        for p in P1.ROOT.glob(f"{obj}/{ang}_0*/*/Camera_1.mp4"))
            clips += cs[:: max(1, len(cs) // n)][:n]
    print(f"{len(clips)} clips", flush=True)
    res = []
    for i, c in enumerate(clips):
        try:
            fr, fps = P1.load(c)
            tr, _ = P1.track(fr)
            r = classify(c, fr, fps, tr)
        except Exception as e:
            r = {"clip": c, "outcome": "error", "error": str(e)}
        res.append(r)
        print(f"  [{i+1}/{len(clips)}] {c:<26} {r['outcome']}"
              + (f"  mu {r['mu']:.3f} rot {r['rot_deg']:.0f}" if r["outcome"] == "slides" else "")
              + (f"  rot {r['rot_deg']:.0f}" if r["outcome"] == "rolls" else ""), flush=True)
        if (i + 1) % 25 == 0:
            OUT.write_text(json.dumps(res))
    OUT.write_text(json.dumps(res))
    return 0


def do_restays(args):
    """Re-track every clip classified 'stays' on a 2.5x denser grid (19 px spacing): small or
    thin objects (hollow-wood plates, toys, coins) fell between the 48 px grid points and
    defaulted to 'stays' -- checked by eye on h_wood_03 and p_toy_02, which both slid."""
    R = json.loads(OUT.read_text())
    idx = [i for i, r in enumerate(R) if r["outcome"] == "stays"
           or (r["outcome"] == "error" and r.get("retracked"))]
    print(f"re-tracking {len(idx)} 'stays'/errored clips at grid 100 (70 on OOM)", flush=True)
    for n, i in enumerate(idx):
        c = R[i]["clip"]
        try:
            fr, fps = P1.load(c)
            try:
                tr, _ = P1.track(fr, grid=100)
            except Exception as e:
                if "out of memory" not in str(e):
                    raise
                import torch; torch.cuda.empty_cache()
                tr, _ = P1.track(fr, grid=70)
            r = classify(c, fr, fps, tr)
        except Exception as e:
            r = {"clip": c, "outcome": "error", "error": str(e)}
        r["retracked"] = True
        changed = r["outcome"] != "stays"
        R[i] = r
        print(f"  [{n+1}/{len(idx)}] {c:<26} {r['outcome']}{'   <-- CHANGED' if changed else ''}",
              flush=True)
        if (n + 1) % 25 == 0:
            OUT.write_text(json.dumps(R))
    OUT.write_text(json.dumps(R))
    return 0


def do_summary(args):
    import collections
    R = json.loads(OUT.read_text())
    C = json.loads((REPO / "docs/results/p1_ramp_friction.json").read_text())
    by = collections.defaultdict(lambda: {"20": [], "10": []})
    for r in R:
        mat = r["clip"].split("/")[0].rsplit("_", 1)[0]
        by[mat]["20" if "/20_" in r["clip"] else "10"].append(r)
    print(f"{'material':<11}{'20deg: slide/roll/stay/other':>30}{'mu median':>11}{'IQR':>15}"
          f"{'  10deg stays':>13}")
    out = {}
    for mat, d in sorted(by.items()):
        c20 = collections.Counter(r["outcome"] for r in d["20"])
        mus = [r["mu"] for r in d["20"] if r["outcome"] == "slides" and not r["impossible"]]
        imp = sum(1 for r in d["20"] if r.get("impossible"))
        st10 = sum(1 for r in d["10"] if r["outcome"] == "stays")
        out[mat] = {"n20": len(d["20"]), "slides": c20["slides"], "rolls": c20["rolls"],
                    "stays": c20["stays"], "impossible": imp, "mu": mus,
                    "mu_median": float(np.median(mus)) if mus else None,
                    "stays10": st10, "n10": len(d["10"])}
        cnt = f"{c20['slides']}/{c20['rolls']}/{c20['stays']}/{c20['unclear']+c20['error']+imp}"
        mtxt = (f"{np.median(mus):>11.3f}{np.percentile(mus, 25):>8.3f}-{np.percentile(mus, 75):<6.3f}"
                if mus else f"{'-':>11}{'':>15}")
        print(f"{mat:<11}{cnt:>30}{mtxt}{st10:>6}/{len(d['10'])}")
    cm = [r["fit"]["mu"] for r in C if "/20_" in r["clip"] and r.get("fit")]
    print(f"{'cardboard':<11}{'(p1, 93 trials)':>30}{np.median(cm):>11.3f}"
          f"{np.percentile(cm, 25):>8.3f}-{np.percentile(cm, 75):<6.3f}")
    (REPO / "docs/results/p2_ramp_summary.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(f=do_run)
    sub.add_parser("summary").set_defaults(f=do_summary)
    sub.add_parser("restays").set_defaults(f=do_restays)
    a = ap.parse_args()
    sys.exit(a.f(a))
