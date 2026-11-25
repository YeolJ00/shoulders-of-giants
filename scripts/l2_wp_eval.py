"""WorldParticle's predictions on the scorer's windows and rollouts (scripts/l2_wp_data.py export --plan writes them;
scripts/l2_wp_data.py score turns these predictions into the l2_coverage_score JSON). Run in the WorldParticle env:

    ATTN_IMPL=flash_attn CUDA_VISIBLE_DEVICES=<gpu> /home/nas5/jooyeolyun/envs/worldparticle/bin/python \
        scripts/l2_wp_eval.py --ckpt outputs/lsim/wp/<run>/last.pt --out outputs/lsim/wp/<run>/pred_last.npz

One frame: from the recorded frame t (positions, velocity from t - 1 -> t), one model step, the cloth's predicted
positions at t + 1. Rollouts: from frame 7 alone to the end (up to 113 frames), the gripper and the held vertices on
their recorded path (driven particles, as in training).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from l2_wp_train import Episode, build, step  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", default="outputs/lsim/wp/data/plan.json")
    ap.add_argument("--data_dir", default="outputs/lsim/wp/data/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="")
    ap.add_argument("--n_win", type=int, default=0, help="first n windows per family only (smoke tests)")
    ap.add_argument("--n_roll", type=int, default=-1, help="first n rollouts per family only (smoke tests)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(REPO / a.ckpt, map_location=dev)
    model = build(ck["cfg"], dev)
    model.model.load_state_dict(ck["model"])
    model.eval()
    plan = json.loads((REPO / a.plan).read_text())
    cache = {}

    def ep(eid):
        if eid not in cache:
            cache.clear()
            cache[eid] = Episode(REPO / a.data_dir / f"{eid}.npz", dev)
        return cache[eid]

    pred = {}
    t0 = time.time()
    fams = a.families.split(",") if a.families else list(plan)
    for fam in fams:
        pl = plan[fam]
        idx = sorted(range(len(pl["win_ep"])), key=lambda i: pl["win_ep"][i])     # one load per episode
        if a.n_win:
            idx = [i for i in idx if i < a.n_win]
        for i in idx:
            e = ep(pl["win_ep"][i])
            t = int(pl["win_t"][i])
            pos, vel = e.state(t)
            _, _, p, _ = step(model, e, pos, vel, t)
            pred[f"{fam}_win{i:03d}"] = p[:e.Vc].cpu().numpy()
        roll = pl["roll_ep"] if a.n_roll < 0 else pl["roll_ep"][:a.n_roll]
        for eid in roll:
            e = ep(eid)
            T0, n = 7, min(113, e.T - 8)
            pos, vel = e.state(T0)
            xs = []
            for k in range(n):
                pos, vel, _, _ = step(model, e, pos, vel, T0 + k)
                xs.append(pos[:e.Vc].cpu().numpy())
            pred[f"{fam}_roll_{eid}"] = np.stack(xs)
        print(f"[wp eval] {fam}: {len(idx)} windows, {len(roll)} rollouts, {time.time() - t0:.0f} s", flush=True)
    np.savez(REPO / a.out, **pred)
    print(f"[wp eval] {REPO / a.out}")


if __name__ == "__main__":
    main()
