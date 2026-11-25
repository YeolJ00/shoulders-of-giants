"""The material-blind floor: how well can ANY predictor do one frame ahead from 3 frame positions without the material?
(2026-11-20)

Fully trained, every cloth network stalls at ~0.65-0.67 of constant velocity's one-frame error on moving free cloth:
twice the training buys -0.02, 8x the parameters nothing, the stencil input nothing, and training windows score like
held-out ones. l1_teacher_sensitivity found the one-frame map in the air is mostly material (the teacher from 3 frames
0.24 with its own material, 1.00 with another episode's) and that the networks barely respond to the material they are
given. A predictor that cannot use the material does best by predicting the next frame averaged over the material
prior; this measures that average's error directly: the teacher restarted from 3 frame positions (back2 velocity, the
model's information) with K materials of other episodes (draws from the prior), x_{t+1} averaged over the K, scored
against the recorded frame. The mean over K draws carries an extra sigma^2 / K; the floor is extrapolated to K -> inf
from K = 1 / 3 / 7 / 15 (the squared error is linear in 1 / K).

Same episodes and instants as l1_teacher_sensitivity (seed 11, 64 held-out-object worlds, 16 instants).

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_teacher_blind.py
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from l1_teacher_from_inputs import FPS, held_at, on_surface, velocities

REPO = Path(__file__).resolve().parents[1]
KS = (1, 3, 7, 15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_t", type=int, default=16)
    ap.add_argument("--per_batch", type=int, default=4, help="variants per run_batch call (worlds x this)")
    ap.add_argument("--material", default=None, help="w,Y,c,mu: every episode at this material (as l1_generate_cloth "
                    "--material, c1m: 0.16,3000,0.027,0.35); the donors then come from a separate draw of the prior")
    ap.add_argument("--tag", default="", help="results file suffix")
    a = ap.parse_args()
    import warp as wp
    from lsim.data import library
    from lsim.data.cloth import run_batch, sample_spec
    wp.init()
    _, test = library.object_split("gso")
    rng = np.random.default_rng(a.seed)
    specs = [sample_spec(rng, test, seed=a.seed * 10000 + k) for k in range(a.worlds)]
    mats = [sp.cloth for sp in specs]                                   # the donors' materials: the batch's own
    if a.material:
        from lsim.teacher.cloth_teacher import G
        mw, mY, mc, mmu = (float(v) for v in a.material.split(","))
        specs = [replace(sp, cloth=replace(sp.cloth, w=mw, Y=mY, B=mw * G * mc ** 3, mu=mmu)) for sp in specs]
        r2 = np.random.default_rng(a.seed + 1)
        mats = [sample_spec(r2, test, seed=(a.seed + 1) * 10000 + k).cloth for k in range(a.worlds)]
    t0 = time.time()
    res = run_batch(specs, seed=a.seed, record=4.0, log=lambda m: None, record_v=True)
    X, V, settle, path, events, worlds = res["x"].astype(np.float64), res["v"].astype(np.float64), res["settle"], \
        res["path"], res["events"], res["worlds"]
    sub0 = res["substeps"]
    del res
    if a.material:                                                      # the donors may need more (converged in dt:
        from lsim.teacher.cloth_teacher import stable_substeps          # twice the substeps moves it 0.015)
        sub0 = max(sub0, stable_substeps(mats))
    print(f"recorded {a.worlds} episodes x {X.shape[0] - 1} frames, {sub0} substeps per frame, in {time.time() - t0:.0f} s",
          flush=True)
    W, K = a.worlds, max(KS)
    donors = [[(w + 1 + k * (W - 1) // K) % W for k in range(K)] for w in range(W)]   # K other episodes, spread out
    assert all(w not in d for w, d in enumerate(donors))
    variants = [None] + list(range(K))                                  # None: its own material; k: donor k's
    F = X.shape[0] - 1
    ts = np.unique(np.linspace(8, F - 2, a.n_t).round().astype(int))
    names = ["own"] + [f"blind K{k}" for k in KS]
    err = {v: {"surface": [], "off": []} for v in names + ["cv"]}
    sq = {k: {"surface": [], "off": []} for k in KS}                    # squared errors, for the K -> inf fit
    for t in ts:
        T = settle + t
        vel = velocities(X, V, t)["back2"]
        replay, sel = [], []
        for w in range(W):
            p0, n = worlds[w]["particles"]
            h, ev = held_at(events[w], T, n)
            replay.append({"path": path[T:T + 2, w], "events": ev})
            moving = np.linalg.norm(X[t, p0:p0 + n] - X[t - 1, p0:p0 + n], axis=-1) * FPS > 0.02
            sel.append(moving & ~h)
        tot = sum(wd["particles"][1] for wd in worlds)
        pred = {v: np.zeros((tot, 3)) for v in range(-1, K)}            # -1: own material
        for b0 in range(0, len(variants), a.per_batch):
            B = variants[b0:b0 + a.per_batch]
            bspecs, start = [], []
            for v in B:
                for w in range(W):
                    p0, n = worlds[w]["particles"]
                    c = specs[w].cloth
                    if v is not None:
                        o = mats[donors[w][v]]
                        c = replace(c, w=o.w, Y=o.Y, nu=o.nu, B=o.B, mu=o.mu)
                    bspecs.append(replace(specs[w], cloth=c))
                    start.append({"x": X[t, p0:p0 + n], "v": vel[p0:p0 + n]})
            r = run_batch(bspecs, seed=a.seed, record=1.0 / FPS, log=lambda m: None, replay=replay * len(B), settle=0,
                          start=start, substeps=sub0)
            Y = r["x"][1].astype(np.float64)
            del r
            for j, v in enumerate(B):
                pred[-1 if v is None else v] = Y[j * tot:(j + 1) * tot]
        for w in range(W):
            p0, n = worlds[w]["particles"]
            s = sel[w]
            if not s.any():
                continue
            tgt = X[t + 1, p0:p0 + n][s]
            surf = on_surface(X[t, p0:p0 + n][s], specs[w])
            cv = np.linalg.norm(2 * X[t, p0:p0 + n][s] - X[t - 1, p0:p0 + n][s] - tgt, axis=-1)
            err["cv"]["surface"].append(cv[surf]); err["cv"]["off"].append(cv[~surf])
            e = np.linalg.norm(pred[-1][p0:p0 + n][s] - tgt, axis=-1)
            err["own"]["surface"].append(e[surf]); err["own"]["off"].append(e[~surf])
            for k in KS:
                m = np.mean([pred[d][p0:p0 + n][s] for d in range(k)], axis=0)
                d2 = np.sum((m - tgt) ** 2, axis=-1)
                e = np.sqrt(d2)
                err[f"blind K{k}"]["surface"].append(e[surf]); err[f"blind K{k}"]["off"].append(e[~surf])
                sq[k]["surface"].append(d2[surf]); sq[k]["off"].append(d2[~surf])
        cat = lambda d: np.concatenate([np.concatenate(d[c][-W:]) for c in ("surface", "off") if d[c][-W:]])
        cvm = cat(err["cv"]).mean()
        print(f"  t {t:3d}: " + "  ".join(f"{v} {cat(err[v]).mean() / cvm:.3f}" for v in names) +
              f"   (constant velocity {1e3 * cvm:.3f} mm)", flush=True)
    out = {"worlds": W, "frames": [int(t) for t in ts], "substeps": int(sub0), "K": list(KS), "material": a.material,
           "classes": {}}
    for c in ("all", "off", "surface"):
        pick = (lambda d: np.concatenate(d["surface"] + d["off"])) if c == "all" else (lambda d, c=c: np.concatenate(d[c]))
        cv = pick(err["cv"])
        row = {"n": int(len(cv)), "cv_mm": 1e3 * float(cv.mean()), "cv_rms_mm": 1e3 * float(np.sqrt((cv ** 2).mean()))}
        for v in names:
            e = pick(err[v])
            row[v] = {"mm": 1e3 * float(e.mean()), "ratio": float(e.mean() / cv.mean()),
                      "rms_ratio": float(np.sqrt((e ** 2).mean() / (cv ** 2).mean()))}
        # mean squared error is a + b / K: extrapolate the RMS ratio to K -> inf; scale the mean-distance ratio alike
        ms = np.array([pick(sq[k]).mean() for k in KS])
        b, a_ = np.polyfit(1.0 / np.array(KS), ms, 1)
        rms_inf = float(np.sqrt(max(a_, 0.0)) / np.sqrt((cv ** 2).mean()))
        r15 = row[f"blind K{max(KS)}"]
        row["blind K_inf"] = {"rms_ratio": rms_inf, "ratio": r15["ratio"] * rms_inf / r15["rms_ratio"]}
        out["classes"][c] = row
        print(f"{c:8s} n {row['n']:7d}  constant velocity {row['cv_mm']:.3f} mm  " +
              "  ".join(f"{v} {row[v]['ratio']:.3f}" for v in names + ["blind K_inf"]) +
              f"   (rms: own {row['own']['rms_ratio']:.3f}, K_inf {rms_inf:.3f})", flush=True)
    json.dump(out, open(REPO / f"docs/results/l1_teacher_blind{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
