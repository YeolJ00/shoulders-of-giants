"""The small-fit test, rigid track (2026-11-22; the cloth one is l2_small_fit.py): can the rigid model fit and replay
the episodes it trained on? Phase 2's unit check (l2_overfit.py, October) stopped at 5 mm after 8 frames and 14-24 mm
after 30 on its one episode and was never pushed further; the cloth network now replays its one episode to under a
millimetre. A checkpoint written by l2_overfit.py --out (the first N episodes of a shard with dynamic bodies), scored
on those same episodes:
  one frame   from the true past (rollout H = 1 from every 4th frame): mean vertex distance of each dynamic rigid body
              that moves (> 0.1 mm a frame), over the same for constant velocity (vertices 2 x_t - x_{t-1})
  replay      alone from frame 8 to the end (model.rollout): per dynamic rigid body the mean vertex distance at 1 s,
              2 s and the end, beside the body left where it was at frame 8

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/rigid_fit_1.pt --episodes 1
    ... --ckpt outputs/lsim/rigid_pilot128_attn.pt --episodes 32 --shard 1 --tag held      (episodes it never trained on)
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from lsim.data.episode import rotate
from lsim.model.net import build_model
from lsim.model.rollout import rollout
from lsim.train.windows import PARAMS, load_shard

REPO = Path(__file__).resolve().parents[1]


def verts(e, q):
    """World vertices of each dynamic rigid body at poses q (n_bodies, 7): list of (n, 3)."""
    return [rotate(q[i, 3:], b.verts.astype(float)) + q[i, :3] for i, b in enumerate(e.bodies)
            if b.role == "dynamic" and b.type == "rigid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/datasets/v0")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", type=int, default=0, help="which shard the episodes come from (l2_overfit.py trains on shard 0: another one is held out)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev)
    model = build_model(ck["args"], len(PARAMS)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    shard = sorted(glob.glob(f"{a.data}/shard_*.npz"))[a.shard][:-4]
    eps = [e for e in load_shard(shard) if e.bodies and any(b.role == "dynamic" for b in e.bodies)][:a.episodes]
    ids = lambda e: [b.body for b in e.bodies]
    out = {"ckpt": a.ckpt, "episodes": []}
    num = den = 0.0
    for k, e in enumerate(eps):
        T = e.q.shape[0]
        em = ec = 0.0
        for t0 in range(8, T - 1, 4):
            pred, true, _ = rollout(model, e, t0, 1)
            now, prev = verts(e, e.q[t0][ids(e)]), verts(e, e.q[t0 - 1][ids(e)])
            for vp, vt, vn, vv in zip(verts(e, pred[0]), verts(e, true[0]), now, prev):
                if np.linalg.norm(vt - vn, axis=1).mean() < 1e-4:
                    continue
                em += np.linalg.norm(vp - vt, axis=1).mean(); ec += np.linalg.norm(2 * vn - vv - vt, axis=1).mean()
        num += em; den += ec
        H = T - 1 - 8
        pred, true, _ = rollout(model, e, 8, H)
        start = verts(e, e.q[8][ids(e)])
        rows = []
        for j, (vs,) in enumerate(zip(start)):
            row = {"model": [], "still": []}
            for h in (29, 59, H - 1):
                vp, vt = verts(e, pred[h])[j], verts(e, true[h])[j]
                row["model"].append(1e3 * float(np.linalg.norm(vp - vt, axis=1).mean()))
                row["still"].append(1e3 * float(np.linalg.norm(vs - vt, axis=1).mean()))
            rows.append(row)
        M, S = np.array([r["model"] for r in rows]), np.array([r["still"] for r in rows])
        moved = S[:, 2] > 5.0
        print(f"episode {k}: one frame {em / max(ec, 1e-12):.3f} of constant velocity   replay of {len(rows)} bodies, mean vertex "
              f"distance at 1 s / 2 s / end ({H / 30:.1f} s): median {np.median(M, 0).round(1).tolist()}  worst {M.max(0).round(1).tolist()} mm"
              f"   (left where they were: median {np.median(S, 0).round(1).tolist()}  worst {S.max(0).round(1).tolist()})", flush=True)
        for j, r in enumerate(rows):
            print(f"    body {j}: {'  '.join(f'{x:7.1f}' for x in r['model'])}   still {'  '.join(f'{x:7.1f}' for x in r['still'])}"
                  f"{'' if moved[j] else '   (the teacher left it in place)'}")
        out["episodes"].append({"one_frame_ratio": em / max(ec, 1e-12), "bodies": rows})
    out["one_frame_ratio"] = num / max(den, 1e-12)
    print(f"all {len(eps)} episodes: one frame {out['one_frame_ratio']:.3f} of constant velocity")
    name = Path(a.ckpt).stem + (f"_{a.tag}" if a.tag else "")
    (REPO / f"docs/results/l2_small_fit_rigid_{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
