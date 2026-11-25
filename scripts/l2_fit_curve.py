"""Train / held-out fit of cloth-family checkpoints on the same footing (2026-11-06).

The trainer logs its training error on noised inputs and its held-out error on clean ones, so the two are not
comparable. This scores each checkpoint on fixed windows from TRAINING shards and from HELD-OUT shards alike: clean
history (no noise), every slot given as in validation, the 8 next frames each predicted from the teacher's own history
(teacher-forced one-step predictions). Two numbers per window:
  pos_mm    mean free-vertex distance to the teacher's next frame (mm), first step
  shape_pct mean over mesh edges of |predicted length - teacher's length| / rest length (%), all 8 steps: how wrong
            the shape is, independent of where the whole cloth is (a rigid shift scores 0)
and the teacher's own per-step change of edge length for scale.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_fit_curve.py --run cloth_q64 --train c1 --test c1_test
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def windows(ds, n, seed, episodes=0):
    from lsim.train.cloth_train import _windows
    from lsim.train.cloth_windows import load_shard
    eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{ds}/shard_*.npz")))[:4] for e in load_shard(p[:-4])]
    if episodes:                                                     # an overfit run's own episodes (cloth_train --episodes)
        eps = eps[:episodes]
    return _windows(eps, n, 8, 8, np.random.default_rng(seed))


@torch.no_grad()
def score(model, wins, dev):
    from lsim.train.cloth_train import step_loss
    from lsim.train.cloth_windows import collate_cloth
    pos, shape, teach = [], [], []
    for i in range(0, len(wins), 4):
        b = collate_cloth(wins[i:i + 4], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
        _, info = step_loss(model, b, np.random.default_rng(i), 8, dev, sig=0.0)
        pos.append(info["clean_mean_mm"]); shape.append(info["strain_err_pct"])
        E, P = b["edges"], 8                                            # the teacher's own edge change per step
        x = b["pos"]
        L0 = (b["rest"][E[:, 0]] - b["rest"][E[:, 1]]).norm(dim=-1) * b["size"][b["sample_of"][E[:, 0]]]
        ln = (x[E[:, 0], P - 1:P + 8] - x[E[:, 1], P - 1:P + 8]).norm(dim=-1)
        teach.append(100 * ((ln[:, 1:] - ln[:, :-1]).abs() / L0[:, None]).mean().item())
    return float(np.mean(pos)), float(np.mean(shape)), float(np.mean(teach))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/lsim/<run>/step*.pt")
    ap.add_argument("--train", required=True, help="training datasets, comma-separated (scored separately)")
    ap.add_argument("--test", required=True, help="held-out datasets, same order")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=0, help="training windows from the first n episodes only (overfit runs)")
    a = ap.parse_args()
    from l2_real_cloth_pgnd_eval import load_model
    dev = "cuda"
    sets = {}
    for tr, te in zip(a.train.split(","), a.test.split(",")):
        sets[tr] = (windows(tr, a.n, 1, a.episodes), windows(te, a.n, 2))
    ck = sorted(glob.glob(str(REPO / f"outputs/lsim/{a.run}/step*.pt")), key=lambda p: int(re.findall(r"step(\d+)", p)[-1]))
    out = []
    for p in ck:
        step = int(re.findall(r"step(\d+)", p)[-1])
        m = load_model(p, dev)
        row = {"step": step}
        for k, (wt, wv) in sets.items():
            row[k] = {"train": score(m, wt, dev), "test": score(m, wv, dev)}
            print(f"{a.run} step {step:6d} {k:4s}  pos mm train {row[k]['train'][0]:.3f} test {row[k]['test'][0]:.3f}  "
                  f"shape % train {row[k]['train'][1]:.3f} test {row[k]['test'][1]:.3f}  (teacher edge change "
                  f"{row[k]['test'][2]:.4f} %)", flush=True)
        out.append(row)
        del m; torch.cuda.empty_cache()
    (REPO / "docs/results").mkdir(exist_ok=True)
    json.dump(out, open(REPO / f"docs/results/l2_fit_curve_{a.run}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
