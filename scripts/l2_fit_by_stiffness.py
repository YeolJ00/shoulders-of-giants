"""Does the one-step fit depend on how stiff the cloth is (2026-11-10)?

Hypothesis: the teacher's frame-average force is a sum of large cancelling edge forces set by strains the inputs
barely resolve, so stiffer cloth (stretch stiffness Y over areal weight w) is harder to fit. Scores one checkpoint on
held-out windows ONE AT A TIME (clean history, every slot given: l2_fit_curve's setting) and relates each window's
position and shape error to its cloth's Y, w, Y / w, with the window's motion (the teacher's mean acceleration)
as the covariate that otherwise dominates.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_fit_by_stiffness.py --ckpt outputs/lsim/cloth_q64/step23999.pt
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def partial(x, y, z):
    """Spearman correlation of x and y with z regressed out (all as ranks)."""
    rx, ry, rz = rank(x), rank(y), rank(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="c1_test")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--tag", default="q64")
    a = ap.parse_args()
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import _windows, step_loss
    from lsim.train.cloth_windows import collate_cloth, decode_slots, load_shard
    dev = "cuda"
    eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz"))) for e in load_shard(p[:-4])]
    wins = _windows(eps, a.n, 8, 8, np.random.default_rng(3))
    m = load_model(a.ckpt, dev)
    rows = []
    for i, w in enumerate(wins):
        b = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
        _, info = step_loss(m, b, np.random.default_rng(i), 8, dev, sig=0.0)
        x, P = b["pos"], 8
        fr = (b["is_cloth"][:, None] & ~b["held"])[:, P - 1]
        acc = (x[fr, P] - 2 * x[fr, P - 1] + x[fr, P - 2]).norm(dim=-1).mean().item() * 1e3      # mm / frame^2
        E = b["edges"]
        L0 = (b["rest"][E[:, 0]] - b["rest"][E[:, 1]]).norm(dim=-1) * b["size"][b["sample_of"][E[:, 0]]]
        ln = (x[E[:, 0], P - 1:P + 8] - x[E[:, 1], P - 1:P + 8]).norm(dim=-1)
        wY = decode_slots(b["params"][0].cpu().numpy())
        rows.append(dict(pos_mm=info["clean_mean_mm"], shape_pct=info["strain_err_pct"], acc=acc,
                         teacher_edge_pct=100 * ((ln[:, 1:] - ln[:, :-1]).abs() / L0[:, None]).mean().item(),
                         w=float(wY[0]), Y=float(wY[1]), c=float(wY[2]), mu=float(wY[3])))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    ok = np.isfinite(R["pos_mm"])
    R = {k: v[ok] for k, v in R.items()}
    R["Y_over_w"] = R["Y"] / R["w"]
    out = {"n": int(ok.sum()), "corr": {}, "tertiles": {}}
    print(f"{out['n']} windows; Spearman of the error with each quantity (plain / with the window's motion regressed out)")
    for q in ("acc", "Y", "w", "Y_over_w", "c", "mu", "teacher_edge_pct"):
        for e in ("pos_mm", "shape_pct"):
            plain = float(np.corrcoef(rank(R[q]), rank(R[e]))[0, 1])
            par = partial(R[q], R[e], R["acc"]) if q != "acc" else float("nan")
            out["corr"][f"{e}~{q}"] = (plain, par)
        c = out["corr"]
        print(f"  {q:17s} pos {c[f'pos_mm~{q}'][0]:+.2f} / {c[f'pos_mm~{q}'][1]:+.2f}   shape {c[f'shape_pct~{q}'][0]:+.2f} / {c[f'shape_pct~{q}'][1]:+.2f}")
    for q in ("Y", "Y_over_w"):
        cut = np.quantile(R[q], [1 / 3, 2 / 3])
        g = np.digitize(R[q], cut)
        out["tertiles"][q] = []
        for t in range(3):
            s = g == t
            row = dict(lo=float(R[q][s].min()), hi=float(R[q][s].max()), pos_mm=float(np.median(R["pos_mm"][s])),
                       shape_pct=float(np.median(R["shape_pct"][s])), acc=float(np.median(R["acc"][s])),
                       teacher_edge_pct=float(np.median(R["teacher_edge_pct"][s])))
            out["tertiles"][q].append(row)
            print(f"  {q} {row['lo']:9.0f}-{row['hi']:9.0f}: median pos {row['pos_mm']:.3f} mm  shape {row['shape_pct']:.3f} %  "
                  f"(motion {row['acc']:.2f} mm/frame^2, teacher edge change {row['teacher_edge_pct']:.4f} %)")
    out["rows"] = {k: v.tolist() for k, v in R.items()}
    json.dump(out, open(REPO / f"docs/results/l2_fit_by_stiffness_{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
