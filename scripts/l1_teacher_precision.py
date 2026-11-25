"""The resolution floor: how precisely must the 3 input frames be known to predict the next one? (2026-11-21)

Every fully trained cloth network stalls at ~0.63 of constant velocity's one-frame error on moving free cloth, at one
material as at many (the one-material fine-tune with the slot head apart: 0.628 vs 0.632), while the teacher restarted
from the same 3 frame positions reaches 0.25. Hypothesis (stiffness): the teacher's frame-average forces are sums of
large cancelling edge forces set by strains of ~1e-4 -- micrometres on a centimetre edge -- so the one-frame map needs the
inputs resolved far below anything the network's features carry. This measures that need directly: the teacher
restarted from X[t] with back2 velocity, as l1_teacher_from_inputs / l1_teacher_blind, its own material, but each of the
3 input frames (x_{t-2}, x_{t-1}, x_t) jittered per vertex by iid Gaussian noise of sigma per axis. For each sigma, one
draw is what a predictor that reads its inputs to +-sigma and acts on them as exact would score; the mean over K draws,
extrapolated to K -> inf (squared error linear in 1/K), approximates E[x_{t+1} | inputs known to +-sigma], the best such a
predictor could do. Where that floor crosses ~0.63 says what input resolution the networks behave as if they had.

Same episodes and instants as l1_teacher_sensitivity / l1_teacher_blind (seed 11, 64 held-out-object worlds, 16 instants).

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_teacher_precision.py
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from l1_teacher_from_inputs import FPS, held_at, on_surface

REPO = Path(__file__).resolve().parents[1]
KS = (1, 2, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_t", type=int, default=16)
    ap.add_argument("--sigmas_um", default="1,3,10,30,100", help="input jitter per axis, micrometres")
    ap.add_argument("--per_batch", type=int, default=4, help="variants per run_batch call (worlds x this)")
    ap.add_argument("--material", default=None, help="w,Y,c,mu: every episode at this material (c1m: 0.16,3000,0.027,0.35)")
    ap.add_argument("--tag", default="", help="results file suffix")
    a = ap.parse_args()
    import warp as wp
    from lsim.data import library
    from lsim.data.cloth import run_batch, sample_spec
    wp.init()
    _, test = library.object_split("gso")
    rng = np.random.default_rng(a.seed)
    specs = [sample_spec(rng, test, seed=a.seed * 10000 + k) for k in range(a.worlds)]
    if a.material:
        from lsim.teacher.cloth_teacher import G
        mw, mY, mc, mmu = (float(v) for v in a.material.split(","))
        specs = [replace(sp, cloth=replace(sp.cloth, w=mw, Y=mY, B=mw * G * mc ** 3, mu=mmu)) for sp in specs]
    sig = [float(s) * 1e-6 for s in a.sigmas_um.split(",")]
    t0 = time.time()
    res = run_batch(specs, seed=a.seed, record=4.0, log=lambda m: None, record_v=True)
    X, settle, path, events, worlds = res["x"].astype(np.float64), res["settle"], res["path"], res["events"], res["worlds"]
    sub0 = res["substeps"]
    del res
    print(f"recorded {a.worlds} episodes x {X.shape[0] - 1} frames, {sub0} substeps per frame, in {time.time() - t0:.0f} s",
          flush=True)
    W, K = a.worlds, max(KS)
    variants = [None] + [(s, k) for s in range(len(sig)) for k in range(K)]   # None: exact inputs; (sigma, draw)
    F = X.shape[0] - 1
    ts = np.unique(np.linspace(8, F - 2, a.n_t).round().astype(int))
    nrng = np.random.default_rng(a.seed + 7)
    names = ["exact"] + [f"s{s}" for s in range(len(sig))]
    err = {"cv": {"surface": [], "off": []}, "exact": {"surface": [], "off": []}}
    for s in range(len(sig)):
        for k in KS:
            err[(s, k)] = {"surface": [], "off": []}
    sq = {(s, k): {"surface": [], "off": []} for s in range(len(sig)) for k in KS}
    for t in ts:
        T = settle + t
        replay, sel = [], []
        for w in range(W):
            p0, n = worlds[w]["particles"]
            h, ev = held_at(events[w], T, n)
            replay.append({"path": path[T:T + 2, w], "events": ev})
            moving = np.linalg.norm(X[t, p0:p0 + n] - X[t - 1, p0:p0 + n], axis=-1) * FPS > 0.02
            sel.append(moving & ~h)
        tot = sum(wd["particles"][1] for wd in worlds)
        pred = {}
        for b0 in range(0, len(variants), a.per_batch):
            B = variants[b0:b0 + a.per_batch]
            bspecs, start = [], []
            for v in B:
                if v is None:
                    x2, x1, x0 = X[t - 2], X[t - 1], X[t]
                else:
                    s = sig[v[0]]
                    x2, x1, x0 = (X[t - j] + s * nrng.standard_normal(X[t].shape) for j in (2, 1, 0))
                vel = (3 * x0 - 4 * x1 + x2) * FPS / 2                 # back2, from the (jittered) frames
                for w in range(W):
                    p0, n = worlds[w]["particles"]
                    bspecs.append(specs[w])
                    start.append({"x": x0[p0:p0 + n], "v": vel[p0:p0 + n]})
            r = run_batch(bspecs, seed=a.seed, record=1.0 / FPS, log=lambda m: None, replay=replay * len(B), settle=0,
                          start=start, substeps=sub0)
            Y = r["x"][1].astype(np.float64)
            del r
            for j, v in enumerate(B):
                pred[v] = Y[j * tot:(j + 1) * tot]
        for w in range(W):
            p0, n = worlds[w]["particles"]
            sl = sel[w]
            if not sl.any():
                continue
            tgt = X[t + 1, p0:p0 + n][sl]
            surf = on_surface(X[t, p0:p0 + n][sl], specs[w])
            cv = np.linalg.norm(2 * X[t, p0:p0 + n][sl] - X[t - 1, p0:p0 + n][sl] - tgt, axis=-1)
            err["cv"]["surface"].append(cv[surf]); err["cv"]["off"].append(cv[~surf])
            e = np.linalg.norm(pred[None][p0:p0 + n][sl] - tgt, axis=-1)
            err["exact"]["surface"].append(e[surf]); err["exact"]["off"].append(e[~surf])
            for s in range(len(sig)):
                for k in KS:
                    m = np.mean([pred[(s, d)][p0:p0 + n][sl] for d in range(k)], axis=0)
                    d2 = np.sum((m - tgt) ** 2, axis=-1)
                    err[(s, k)]["surface"].append(np.sqrt(d2[surf])); err[(s, k)]["off"].append(np.sqrt(d2[~surf]))
                    sq[(s, k)]["surface"].append(d2[surf]); sq[(s, k)]["off"].append(d2[~surf])
        off = lambda d: np.concatenate(d["off"][-W:])
        cvm = off(err["cv"]).mean()
        print(f"  t {t:3d} (off surfaces): exact {off(err['exact']).mean() / cvm:.3f}  " +
              "  ".join(f"{1e6 * sig[s]:g}um 1:{off(err[(s, 1)]).mean() / cvm:.3f} {K}:{off(err[(s, K)]).mean() / cvm:.3f}"
                        for s in range(len(sig))) + f"   (constant velocity {1e3 * cvm:.3f} mm)", flush=True)
    out = {"worlds": W, "frames": [int(t) for t in ts], "substeps": int(sub0), "K": list(KS), "material": a.material,
           "sigmas_um": [1e6 * s for s in sig], "classes": {}}
    for c in ("all", "off", "surface"):
        pick = (lambda d: np.concatenate(d["surface"] + d["off"])) if c == "all" else (lambda d, c=c: np.concatenate(d[c]))
        cv = pick(err["cv"])
        cv_rms = np.sqrt((cv ** 2).mean())
        e = pick(err["exact"])
        row = {"n": int(len(cv)), "cv_mm": 1e3 * float(cv.mean()), "exact": {"ratio": float(e.mean() / cv.mean()),
               "rms_ratio": float(np.sqrt((e ** 2).mean()) / cv_rms)}, "sigma": []}
        for s in range(len(sig)):
            r = {"sigma_um": 1e6 * sig[s]}
            for k in KS:
                ek = pick(err[(s, k)])
                r[f"K{k}"] = {"ratio": float(ek.mean() / cv.mean()), "rms_ratio": float(np.sqrt((ek ** 2).mean()) / cv_rms)}
            ms = np.array([pick(sq[(s, k)]).mean() for k in KS])
            b, a_ = np.polyfit(1.0 / np.array(KS), ms, 1)
            rms_inf = float(np.sqrt(max(a_, 0.0)) / cv_rms)
            rK = r[f"K{max(KS)}"]
            r["K_inf"] = {"rms_ratio": rms_inf, "ratio": rK["ratio"] * rms_inf / rK["rms_ratio"]}
            row["sigma"].append(r)
        out["classes"][c] = row
        print(f"{c:8s} n {row['n']:7d}  constant velocity {row['cv_mm']:.3f} mm  exact {row['exact']['ratio']:.3f}  " +
              "  ".join(f"{r['sigma_um']:g}um one draw {r['K1']['ratio']:.3f} floor {r['K_inf']['ratio']:.3f}"
                        for r in row["sigma"]), flush=True)
    json.dump(out, open(REPO / f"docs/results/l1_teacher_precision{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
