"""Hinged parts (drawers, doors, lids: dynamic bodies that are not free rigid bodies), which l2_small_fit_rigid.py and
the gates do not score (2026-11-24): one frame from every 4th frame over constant velocity (moving parts), and the
replay from frame 8 at 1 s / 2 s / end beside the part left where it was, split by whether the teacher moves the part.
On the 128 episodes of v0's second shard (52 parts, 16 moved): the plain transformer follows the moved ones (11 mm at
the end; v1 9, B1 86 = not at all) and lets shut ones drift (27 mm; v1 0.1, B1 0.9); nobody beats constant velocity
over one frame (1.15-1.20; v1 4.2).

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_rigid_hinged.py <ckpt> <shard index> <n episodes>
"""
import glob, sys
import numpy as np, torch
from lsim.data.episode import rotate
from lsim.model.net import build_model
from lsim.model.rollout import rollout
from lsim.train.windows import PARAMS, load_shard
dev = torch.device("cuda")
ck = torch.load(sys.argv[1], map_location=dev)
model = build_model(ck["args"], len(PARAMS)).to(dev); model.load_state_dict(ck["model"]); model.eval()
shard = sorted(glob.glob("data/datasets/v0/shard_*.npz"))[int(sys.argv[2])][:-4]
eps = [e for e in load_shard(shard) if e.bodies and any(b.role == "dynamic" for b in e.bodies)][:int(sys.argv[3])]
num = den = 0.0; nmov = 0
M, S = [], []
for k, e in enumerate(eps):
    sel = [i for i, b in enumerate(e.bodies) if b.role == "dynamic" and b.type != "rigid"]
    if not sel:
        continue
    ids = [b.body for b in e.bodies]
    V = lambda q, i: rotate(q[3:], e.bodies[i].verts.astype(float)) + q[:3]
    T = e.q.shape[0]
    for t0 in range(8, T - 1, 4):
        with torch.no_grad():
            pred, true, _ = rollout(model, e, t0, 1)
        for i in sel:
            vt, vn, vp = V(true[0][i], i), V(e.q[t0, ids[i]], i), V(e.q[t0 - 1, ids[i]], i)
            if np.linalg.norm(vt - vn, axis=1).mean() < 1e-4:
                continue
            num += np.linalg.norm(V(pred[0][i], i) - vt, axis=1).mean(); den += np.linalg.norm(2 * vn - vp - vt, axis=1).mean(); nmov += 1
    H = T - 1 - 8
    with torch.no_grad():
        pred, true, _ = rollout(model, e, 8, H)
    for i in sel:
        vs = V(e.q[8, ids[i]], i)
        M.append([1e3 * np.linalg.norm(V(pred[h][i], i) - V(true[h][i], i), axis=1).mean() for h in (29, 59, H - 1)])
        S.append([1e3 * np.linalg.norm(vs - V(true[h][i], i), axis=1).mean() for h in (29, 59, H - 1)])
M, S = np.array(M), np.array(S)
mv = S[:, 2] > 5.0
print(f"{sys.argv[1]}: {len(M)} hinged parts in {len(eps)} episodes; one frame {num / max(den, 1e-12):.3f} of constant velocity over {nmov} moving part-frames")
print(f"   replay, all parts, 1 s / 2 s / end: median {np.median(M, 0).round(1).tolist()}  p90 {np.percentile(M, 90, 0).round(1).tolist()}   (left in place: median {np.median(S, 0).round(1).tolist()}  p90 {np.percentile(S, 90, 0).round(1).tolist()})")
if mv.sum():
    print(f"   parts the teacher moves by the end ({mv.sum()}): median {np.median(M[mv], 0).round(1).tolist()}   left in place {np.median(S[mv], 0).round(1).tolist()}")
if (~mv).sum():
    print(f"   parts the teacher leaves shut ({(~mv).sum()}): median {np.median(M[~mv], 0).round(1).tolist()}  p90 {np.percentile(M[~mv], 90, 0).round(1).tolist()}")
