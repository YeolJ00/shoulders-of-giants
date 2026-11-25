"""L2 -- the cloth teacher against PGND's robot cloth, scored on what the cameras saw (lsim.real.replay, lsim.real.pgnd).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_real_cloth_pgnd_teacher.py --cloth cali2 [--fit_windows 8 --gens 10]

Claim 3's comparison: the teacher with its best material for a cloth -- fitted on that cloth's training recordings --
against the held-out recording. From frame 7 of a window (the grid fitted to the observed first cloud, tracked 7
frames), the held patches following the robot's path, the teacher's material (w, Y, bending length, mu within the
dataset's prior) is searched by cross-entropy (32 worlds a generation) for the smallest mean distance from the
observed clouds at frames 60 and 120 to the simulated surface, over `fit_windows` training windows spread through
the cloth's training recordings; the best material and the prior's centre then run on every test window (and the fit
windows), scored the same way as scripts/l2_real_cloth_pgnd_eval.py.

Writes docs/results/l2_real_cloth_pgnd_teacher_<cloth>.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
T0 = 7
CENTRE = {"w": 0.16, "Y": 3e3, "c": 0.027, "mu": 0.35}


def score(ep, Xw):
    """Per world, the cloud errors (m) at the window's observed frames after T0: (W, n_frames)."""
    fr = [f for f in ep.cloud_frames if f > T0]
    return np.array([[ep.cloud_error(Xw[w, f - T0], f) for f in fr] for w in range(len(Xw))]), fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloth", required=True, choices=("cloth6", "cali2"))
    ap.add_argument("--fit_windows", type=int, default=8)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from lsim.real import replay
    from lsim.real.pgnd import load_pgnd
    train = [e for e in load_pgnd("train") if e.meta["case"].startswith(a.cloth) and 120 in e.clouds]
    test = [e for e in load_pgnd("test") if e.meta["case"].startswith(a.cloth) and 60 in e.clouds]
    fit = [train[i] for i in np.linspace(0, len(train) - 1, a.fit_windows).astype(int)]
    print(f"[{a.cloth}] fit on {[e.meta['case'] for e in fit]}; test {len(test)} windows", flush=True)
    rng = np.random.default_rng(a.seed)
    lo = np.log([replay.RANGE[k][0] for k in ("w", "Y", "c", "mu")])
    hi = np.log([replay.RANGE[k][1] for k in ("w", "Y", "c", "mu")])
    mean, sd = 0.5 * (lo + hi), 0.35 * (hi - lo)
    best, best_s, hist = None, np.inf, []
    t0 = time.time()
    for gen in range(a.gens):
        Z = mean + sd * rng.standard_normal((a.pop, 4))
        mats = [replay.decode(z) for z in Z]
        S = []
        for ep in fit:
            Xw = replay.simulate(ep, mats, T0, ep.n_frames - 1 - T0)
            s, _ = score(ep, Xw)
            ok = np.isfinite(Xw).all(axis=(1, 2, 3)) & (np.abs(Xw).max(axis=(1, 2, 3)) < 10.0)
            S.append(np.where(ok, s.mean(1), np.inf))
        sc = np.mean(S, 0)
        order = np.argsort(sc)
        el = np.array([replay.encode(mats[i]) for i in order[:a.elite]])
        mean, sd = el.mean(0), np.maximum(el.std(0), 0.02 * (hi - lo))
        if sc[order[0]] < best_s:
            best, best_s = mats[order[0]], float(sc[order[0]])
        hist.append({"gen": gen, "best_mm": 1e3 * float(sc[order[0]]), "elite_mm": 1e3 * float(sc[order[:a.elite]].mean()),
                     "best": mats[order[0]], "centre": replay.decode(mean)})
        print(f"  gen {gen}: best {1e3 * sc[order[0]]:.1f} mm (elite {1e3 * sc[order[:a.elite]].mean():.1f}) at "
              f"{ {k: round(v, 4) for k, v in mats[order[0]].items()} } [{time.time() - t0:.0f} s]", flush=True)
    rows = {}
    for ep in fit + test:
        Xw = replay.simulate(ep, [best, CENTRE], T0, ep.n_frames - 1 - T0)
        s, fr = score(ep, Xw)
        rows[ep.meta["case"]] = {"split": ep.meta["split"], "fit": ep in fit,
                                 "best": {str(f): 1e3 * float(v) for f, v in zip(fr, s[0])},
                                 "centre": {str(f): 1e3 * float(v) for f, v in zip(fr, s[1])}}
        print(f"  {ep.meta['case']:18s} {ep.meta['split']:5s} best " + " ".join(f"{f}:{1e3 * v:6.1f}" for f, v in zip(fr, s[0]))
              + " | centre " + " ".join(f"{f}:{1e3 * v:6.1f}" for f, v in zip(fr, s[1])), flush=True)
    summary = {}
    for k in ("best", "centre"):
        for f in ("60", "120"):
            v = [r[k][f] for r in rows.values() if r["split"] == "test" and f in r[k]]
            if v:
                summary[f"test_{k}@{f}"] = {"mean_of_medians_mm": float(np.mean(v)), "median_mm": float(np.median(v)), "n": len(v)}
    print(json.dumps(summary, indent=1), flush=True)
    (REPO / f"docs/results/l2_real_cloth_pgnd_teacher_{a.cloth}.json").write_text(json.dumps(
        {"cloth": a.cloth, "material": best, "fit_mm": 1e3 * best_s, "fit_windows": [e.meta["case"] for e in fit],
         "history": hist, "summary": summary, "windows": rows, "seconds": time.time() - t0}, indent=1))


if __name__ == "__main__":
    main()
