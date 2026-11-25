"""L2 -- how much the cloth model stretches (phase 3).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_cloth_strain.py --ckpt outputs/lsim/cloth_v3/last.pt --tag v3

The teacher's towels are close to inextensible (stretch modulus 300-30,000 N/m: a hanging towel strains 1e-4 to
1e-2), and at 1 cm between vertices a neighbour 0.1 mm off is already 1% strain. From held-out moving starts (> 3 mm
mean motion over 10 frames, c1_test), with the true 8 frames before, the model rolls out 10 frames; per frame ahead,
the edge strain |L / L0 - 1| of the free cloth's mesh edges in the teacher and in the model, and how much of the
relative error of an edge's two ends is its length (the rest is its direction).

Writes docs/results/l2_cloth_strain_<tag>.json.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from lsim.model.cloth_net import build_cloth_model
from lsim.train.cloth_train import rollout
from lsim.train.cloth_windows import load_cloth_shard

REPO = Path(__file__).resolve().parents[1]
FRAMES = (1, 5, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/datasets/c1_test")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--n", type=int, default=32, help="moving starts")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fp32", action="store_true", help="run the blocks in fp32 (the model trained in bf16)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = [e for p in sorted(glob.glob(f"{a.data}/shard_*.npz"))[:a.shards] for e in load_cloth_shard(p[:-4])]
    ck = torch.load(a.ckpt, map_location=dev)
    A = ck["args"]
    m = build_cloth_model(A, bf16=not a.fp32).to(dev).eval()
    m.load_state_dict(ck["model"])
    rng = np.random.default_rng(7)                                      # the same starts for every checkpoint
    T = max(FRAMES)
    st = {k: {f: [] for f in FRAMES} for k in ("teacher", "model")}
    len_err, rel_err, vert_err = ({f: [] for f in FRAMES} for _ in range(3))
    held = got = 0
    while got < a.n:
        e = eps[rng.integers(len(eps))]
        t0 = int(rng.integers(8, e.n_frames - T - 1))
        x = e.x(np.arange(t0, t0 + T + 1))
        if np.linalg.norm(x[-1] - x[0], axis=-1).mean() < 3e-3:
            continue
        got += 1
        held += bool(e.held[t0].any())
        with torch.no_grad():
            pred, tr, _ = rollout(m, e, t0, T, dev=dev)
        E = e.edges
        L0 = np.linalg.norm(e.rest[E[:, 0]] - e.rest[E[:, 1]], axis=-1)
        free = ~e.held[t0]
        fe = free[E[:, 0]] & free[E[:, 1]]
        for f in FRAMES:
            for k, X in (("teacher", tr[f - 1]), ("model", pred[f - 1])):
                L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=-1)
                st[k][f].append(np.abs(L / L0 - 1.0)[fe])
            Lt = np.linalg.norm(tr[f - 1][E[:, 0]] - tr[f - 1][E[:, 1]], axis=-1)
            Lm = np.linalg.norm(pred[f - 1][E[:, 0]] - pred[f - 1][E[:, 1]], axis=-1)
            de = pred[f - 1] - tr[f - 1]
            len_err[f].append(np.abs(Lm - Lt)[fe])
            rel_err[f].append(np.linalg.norm(de[E[:, 0]] - de[E[:, 1]], axis=-1)[fe])
            vert_err[f].append(float(np.linalg.norm(de, axis=-1)[free].mean()))
    res = {"ckpt": a.ckpt, "n": a.n, "held_at_start": held, "frames": {}}
    print(f"{a.n} moving starts ({held} held at t0); edge strain |L / L0 - 1| of the free cloth:")
    for f in FRAMES:
        row = {}
        for k in ("teacher", "model"):
            s = np.concatenate(st[k][f])
            row[k] = {"median": float(np.median(s)), "p90": float(np.percentile(s, 90)), "p99": float(np.percentile(s, 99)),
                      "max": float(s.max()), "share_over_1pct": float(np.mean(s > 0.01))}
            print(f"  frame {f:2d} {k:8s} median {row[k]['median']:.2e}  p90 {row[k]['p90']:.2e}  p99 {row[k]['p99']:.2e}  "
                  f"share > 1%: {row[k]['share_over_1pct']:.3f}", flush=True)
        le, pe = np.concatenate(len_err[f]), np.concatenate(rel_err[f])
        row["edge_rel_err_mm_median"] = 1e3 * float(np.median(pe))
        row["edge_len_err_mm_median"] = 1e3 * float(np.median(le))
        row["len_share_rms"] = float(np.sqrt((le ** 2).mean() / max((pe ** 2).mean(), 1e-20)))
        row["vertex_err_mm_median"] = 1e3 * float(np.median(vert_err[f]))
        print(f"  frame {f:2d} an edge's two ends: relative error {row['edge_rel_err_mm_median']:.3f} mm, its length "
              f"{row['edge_len_err_mm_median']:.3f} mm (rms share {row['len_share_rms']:.2f}); mean vertex error "
              f"{row['vertex_err_mm_median']:.2f} mm", flush=True)
        res["frames"][str(f)] = row
    tag = a.tag or Path(a.ckpt).parent.name
    (REPO / f"docs/results/l2_cloth_strain_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
