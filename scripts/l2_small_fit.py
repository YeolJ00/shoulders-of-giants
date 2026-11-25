"""Can the network fit a handful of teacher episodes at all (2026-11-22; the user: "can we even fit the teacher on small
data?")?

A run trained on the first N episodes of c1's first shard (cloth_train --shards 1 --episodes N, no noise, no
pushforward), scored on those same episodes and on held-out ones, every checkpoint:
  one frame   error / constant velocity's on moving (> 2 cm/s) unheld cloth, clean history, every slot given: --n
              windows of the training episodes, and --n of c1_test's first 4 shards
and the last checkpoint rolled out alone on the training episodes it memorised (the first min(N, --n_roll); from
frame 7 to the end, the material given -- it was given in training): median free-vertex distance from the teacher at
1 s, 2 s and the end, mean |edge strain|, beside the cloth left where it was at frame 7. The rollout rebuilds its
window every frame from the model's own frames (replay, below), as a training window is built: cloth_train.rollout's
one window over the whole rollout carries the static points near the cloth's whole sweep (1,166 tokens on c1's first
episode against a training window's 878), which the fully trained network does not notice (12 held-out episodes,
the two ways level) and a network that memorised 1-16 episodes does (its first frame 2.6 mm instead of 0.6); that
rollout's numbers are kept beside it ("whole_*").
A network that can represent and be optimised to the teacher's map drives the training ratio towards the teacher's
own floor (restarted from 3 frames: 0.25; from its full state ~0.01) and replays its episodes.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_small_fit.py --run cloth_fit_1 --episodes 1

Writes docs/results/l2_small_fit_<run>.json.
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


@torch.no_grad()
def one_frame(m, wins, dev, P=8):
    from lsim.train.cloth_train import step_loss
    from lsim.train.cloth_windows import collate_cloth
    em = ec = 0.0
    n = 0
    for i, w in enumerate(wins):
        b = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
        parts = {}
        step_loss(m, b, np.random.default_rng(i), 8, dev, sig=0.0, parts=parts)
        x, fr, pos = parts["x"][:, 0], parts["free"][:, 0], b["pos"]
        tgt, cv = pos[:, P], 2 * pos[:, P - 1] - pos[:, P - 2]
        sel = fr & ((pos[:, P - 1] - pos[:, P - 2]).norm(dim=-1) * 30 > 0.02)
        em += float((x - tgt)[sel].norm(dim=-1).sum()); ec += float((cv - tgt)[sel].norm(dim=-1).sum()); n += int(sel.sum())
    return {"ratio": em / max(ec, 1e-12), "model_mm": 1e3 * em / max(n, 1), "cv_mm": 1e3 * ec / max(n, 1), "n": n}


class _Replayed:
    """The episode with its cloth frames replaced by X (the model's own, as they come)."""

    def __init__(self, ep, X):
        self._ep, self._X = ep, X

    def __getattr__(self, k):
        return getattr(self._ep, k)

    def x(self, fr):
        return self._X[np.asarray(fr)]


def replay(m, e, T0, n, dev):
    """Roll the model out n frames from frame T0, one frame per window: each window is built (make_cloth_window, as in
    training) from the model's own last 8 frames and the gripper's recorded path. Returns (pred, true), (n, Vc, 3)."""
    from lsim.train.cloth_train import rollout
    X = np.asarray(e.x(list(range(e.n_frames)))).copy()
    true = X[T0 + 1:T0 + 1 + n].copy()
    v = _Replayed(e, X)
    for t in range(T0, T0 + n):
        X[t + 1] = rollout(m, v, t, 1, dev=dev)[0][0]
    return X[T0 + 1:T0 + 1 + n], true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/lsim/<run>/step*.pt")
    ap.add_argument("--tag", default="", help="suffix of the result file (a re-score beside the first)")
    ap.add_argument("--episodes", type=int, required=True, help="the run's --episodes")
    ap.add_argument("--n", type=int, default=160, help="one-frame windows per set")
    ap.add_argument("--n_roll", type=int, default=16)
    a = ap.parse_args()
    from l2_cloth_long import shape_stats
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import _windows, rollout
    from lsim.train.cloth_windows import load_shard
    dev = "cuda"
    tr = load_shard(sorted(glob.glob(str(REPO / "data/datasets/c1/shard_*.npz")))[0][:-4])[:a.episodes]
    te = [e for p in sorted(glob.glob(str(REPO / "data/datasets/c1_test/shard_*.npz")))[:4] for e in load_shard(p[:-4])]
    wt, wv = _windows(tr, a.n, 8, 8, np.random.default_rng(1)), _windows(te, a.n, 8, 8, np.random.default_rng(2))
    cks = sorted(glob.glob(str(REPO / f"outputs/lsim/{a.run}/step*.pt")), key=lambda p: int(re.findall(r"step(\d+)", p)[-1]))
    out = {"run": a.run, "episodes": a.episodes, "curve": [], "rollouts": []}
    m = None
    for p in cks:
        m = load_model(p, dev)
        row = {"step": int(re.findall(r"step(\d+)", p)[-1]), "train": one_frame(m, wt, dev), "held_out": one_frame(m, wv, dev)}
        out["curve"].append(row)
        print(f"{a.run} step {row['step']:6d}  one frame, error / constant velocity:  its training episodes "
              f"{row['train']['ratio']:.3f} ({row['train']['model_mm']:.3f} vs {row['train']['cv_mm']:.3f} mm)   held-out "
              f"{row['held_out']['ratio']:.3f}", flush=True)
    for j, e in enumerate(tr[:a.n_roll]):                                # the last checkpoint on its own episodes
        T0, n = 7, min(113, e.n_frames - 8)
        free = ~e.held[T0 + 1:T0 + 1 + n]
        with torch.no_grad():
            Pw, Tr, _ = rollout(m, e, T0, n, dev=dev)                     # one window over the whole rollout
            Pr, _ = replay(m, e, T0, n, dev)                             # a window per frame, as in training
        still = np.broadcast_to(np.asarray(e.x([T0]))[0], np.shape(Tr))
        d = lambda X, k: 1e3 * float(np.median(np.linalg.norm(X[k] - Tr[k], axis=-1)[free[k]])) if free[k].any() else float("nan")
        r = {"episode": j, "init": e.wd["init"], "families": [f["family"] for f in e.wd["families"]]}
        for lab, k in (("1s", min(29, n - 1)), ("2s", min(59, n - 1)), ("end", n - 1)):
            r[lab], r[f"still_{lab}"], r[f"whole_{lab}"] = d(Pr, k), d(still, k), d(Pw, k)
        r["strain_pct"], r["teacher_strain_pct"] = 100 * shape_stats(Pr[-1], e)[2], 100 * shape_stats(Tr[-1], e)[2]
        out["rollouts"].append(r)
        print(f"  rollout of training episode {j} ({r['init']}, {'+'.join(r['families'])}): 1 s {r['1s']:.1f}  2 s {r['2s']:.1f}  end "
              f"{r['end']:.1f} mm   (left still: {r['still_1s']:.0f} / {r['still_2s']:.0f} / {r['still_end']:.0f}; one window over "
              f"the rollout: {r['whole_1s']:.0f} / {r['whole_2s']:.0f} / {r['whole_end']:.0f})   strain "
              f"{r['strain_pct']:.2f}% (teacher {r['teacher_strain_pct']:.3f}%)", flush=True)
    (REPO / f"docs/results/l2_small_fit_{a.run}{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
