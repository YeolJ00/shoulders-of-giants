"""L2 -- the cloth model against its teacher over seconds, in its own distribution (c1_test).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_v3/last.pt --tag v3

64 held-out episodes (seeded), each from frame 7 alone to the end (up to 113 frames, 3.8 s), all four material slots
hidden, the gripper on its recorded path: the median free-vertex distance from the teacher at 1 s, 2 s and the end,
by whether the cloth is held most of the time. v3: 34 / 127 / 200 mm (end median 115): a model judged only at 1 s.

Whether the cloth stays a cloth (2026-11-02: distance alone cannot tell a towel in the wrong place from a towel
collapsed to a scrap), per episode at 5 s, 10 s and the end: the mesh area over the teacher's at the same frame (a
cloth compressed or stretched in plane), its radius of gyration over the teacher's (a crumpled or collapsed cloth:
folds keep the area but not the extent) and the mean |edge strain| of model and teacher.

A generative model (--noise_dim checkpoints): --samples rollouts per episode, each with its own fixed noise (seeded
from --seed, the sample and the episode); the headline numbers are sample 0's (one sampled future, as the model is
used), beside each one's spread: the mean vertex distance between samples 0 and 1 at the same times -- calibrated
when it matches the error. --zero_noise: the model's deterministic mode instead.

Writes docs/results/l2_cloth_long_<tag>.json.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
TIMES = (("1s", 29), ("2s", 59), ("5s", 149), ("10s", 299))         # frame indices of the rollout


def shape_stats(X, ep):
    """Mesh area, radius of gyration and mean |edge strain| of cloth positions X (V, 3) of episode ep."""
    t, e = ep.tris, ep.edges
    area = 0.5 * np.linalg.norm(np.cross(X[t[:, 1]] - X[t[:, 0]], X[t[:, 2]] - X[t[:, 0]]), axis=1).sum()
    rg = np.sqrt(((X - X.mean(0)) ** 2).sum(-1).mean())
    L0 = np.linalg.norm(ep.rest[e[:, 0]] - ep.rest[e[:, 1]], axis=1)
    strain = np.abs(np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1) / L0 - 1).mean()
    return float(area), float(rg), float(strain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a checkpoint, or 'layer' (no cloth forces) / 'still' (frame 7 held)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--data", default="c1_test", help="dataset under data/datasets (c1_long_test: 20 s episodes)")
    ap.add_argument("--frames", type=int, default=113, help="most frames rolled out from frame 7")
    ap.add_argument("--samples", type=int, default=1, help="generative models: rollouts per episode (independent noise)")
    ap.add_argument("--seed", type=int, default=0, help="generative models: the noise seed")
    ap.add_argument("--zero_noise", action="store_true", help="generative models: no noise (deterministic mode)")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from l2_real_cloth_pgnd_eval import Zero, load_model
    from lsim.model.cloth_qwen import set_noise
    from lsim.train.cloth_train import rollout
    from lsim.train.cloth_windows import load_cloth_shard
    m = Zero() if a.ckpt in ("layer", "still") else load_model(REPO / a.ckpt, "cuda")
    gen = bool(getattr(m, "noise_dim", 0))
    S = a.samples if gen and not a.zero_noise else 1
    eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz")))[:16] for e in load_cloth_shard(p[:-4])]
    rows = []
    for i, e in enumerate([eps[i] for i in np.random.default_rng(0).choice(len(eps), min(a.n, len(eps)), replace=False)]):
        T0, n = 7, min(a.frames, e.n_frames - 8)
        idx = {k: min(f, n - 1) for k, f in TIMES}
        idx["end"] = n - 1
        free = ~e.held[T0 + 1:T0 + 1 + n]
        preds = []
        for s in range(S):
            if gen:
                set_noise(m, seed=a.seed * 1_000_003 + 1000 * s + i, zero=a.zero_noise)
            with torch.no_grad():
                P, Tr, _ = rollout(m, e, T0, n, dev="cuda", hidden=("density", "stretch", "bend", "mu"))
            if a.ckpt == "still":                                        # the cloth stays where it is at frame 8
                P = np.broadcast_to(np.asarray(Tr[0]), np.shape(Tr)).copy()
            preds.append(P)
        smp = []
        for P in preds:
            err = np.array([np.median(np.linalg.norm(P[k] - Tr[k], axis=-1)[free[k]]) for k in range(n)])
            mean_err = np.array([np.linalg.norm(P[k] - Tr[k], axis=-1)[free[k]].mean() for k in range(n)])
            r = {"e30": 1e3 * float(err[idx["1s"]]), "e60": 1e3 * float(err[idx["2s"]]),
                 "e150": 1e3 * float(err[idx["5s"]]), "e300": 1e3 * float(err[idx["10s"]]), "end": 1e3 * float(err[-1])}
            r["mean_mm"] = {k: 1e3 * float(mean_err[f]) for k, f in idx.items()}
            for k in ("5s", "10s", "end"):
                am, gm, sm = shape_stats(P[idx[k]], e)
                at, gt, st = shape_stats(Tr[idx[k]], e)
                r[f"area_ratio_{k}"], r[f"rg_ratio_{k}"] = am / at, gm / gt
                r[f"strain_pct_{k}"], r[f"teacher_strain_pct_{k}"] = 100 * sm, 100 * st
            smp.append(r)
        row = {"held_frac": float(e.held[T0:T0 + n].any(1).mean()), "n": int(n), **smp[0]}
        if S > 1:
            row["samples"] = smp
            row["spread_mm"] = {k: 1e3 * float(np.linalg.norm(preds[0][f] - preds[1][f], axis=-1).mean()) for k, f in idx.items()}
        rows.append(row)
        print(f"[{i:2d}] end {row['end']:6.1f} mm  area {row['area_ratio_end']:.2f}  rg {row['rg_ratio_end']:.2f}  strain "
              f"{row['strain_pct_end']:.2f}% (teacher {row['teacher_strain_pct_end']:.3f}%)"
              + (f"  spread {row['spread_mm']['end']:.0f} vs error {row['mean_mm']['end']:.0f} mm" if S > 1 else ""), flush=True)
    summ = {}
    for lab, sel in (("held", lambda r: r["held_frac"] > 0.5), ("not_held", lambda r: r["held_frac"] <= 0.5), ("all", lambda r: True)):
        R = [r for r in rows if sel(r)]
        if not R:
            continue
        summ[lab] = {"n": len(R), "1s": float(np.mean([r["e30"] for r in R])), "2s": float(np.mean([r["e60"] for r in R])),
                     "5s": float(np.mean([r["e150"] for r in R])), "10s": float(np.mean([r["e300"] for r in R])),
                     "end": float(np.mean([r["end"] for r in R])), "end_median": float(np.median([r["end"] for r in R]))}
        print(f"{lab:9s} n={len(R):3d}  1 s {summ[lab]['1s']:6.1f}  2 s {summ[lab]['2s']:6.1f}  5 s {summ[lab]['5s']:6.1f}  10 s {summ[lab]['10s']:6.1f}  end {summ[lab]['end']:6.1f} mm "
              f"(median {summ[lab]['end_median']:5.1f})", flush=True)
    sh = {}
    for k in ("5s", "10s", "end"):                                       # median over episodes (ratios: a few blow-ups)
        sh[k] = {q: float(np.median([r[f"{q}_{k}"] for r in rows])) for q in ("area_ratio", "rg_ratio", "strain_pct", "teacher_strain_pct")}
        sh[k]["mean_mm"] = float(np.mean([r["mean_mm"][k] for r in rows]))
        if S > 1:
            sh[k]["spread_mm"] = float(np.mean([r["spread_mm"][k] for r in rows]))
            # all samples' errors (sample 0 is the headline): how much the score depends on the draw
            sh[k]["mean_mm_per_sample"] = [float(np.mean([r["samples"][s]["mean_mm"][k] for r in rows])) for s in range(S)]
        print(f"shape {k:4s}  median area ratio {sh[k]['area_ratio']:.3f}  rg ratio {sh[k]['rg_ratio']:.3f}  strain "
              f"{sh[k]['strain_pct']:.3f}% (teacher {sh[k]['teacher_strain_pct']:.3f}%)  mean distance {sh[k]['mean_mm']:.1f} mm"
              + (f"  sample spread {sh[k]['spread_mm']:.1f} mm  per sample {['%.1f' % x for x in sh[k]['mean_mm_per_sample']]}" if S > 1 else ""),
              flush=True)
    (REPO / f"docs/results/l2_cloth_long_{a.tag}.json").write_text(json.dumps(
        {"ckpt": a.ckpt, "data": a.data, "samples": S, "seed": a.seed, "zero_noise": a.zero_noise, "summary": summ,
         "shape": sh, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
