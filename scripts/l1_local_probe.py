"""How simple is the one-frame rule at one material (2026-11-19)? A model-free learnability probe.

The 64M network trained at one material (c1m) predicts moving free cloth's next frame at 0.90 of constant velocity's
error, the network trained on all materials 0.85 on the same windows: fixing the material does not let it fit. Here
small direct regressors are fitted on c1m and scored on c1m_test, per moving unheld interior vertex away from the
flat surfaces (and with no held vertex in its stencil), from the same information the teacher restarted from 3 frame
positions has inside a (2R + 1)^2 grid stencil:
  inputs   the stencil's positions at t, t-1, t-2 relative to the centre at t, and gravity's direction
  target   the centre's x_{t+1} - 2 x_t + x_{t-1} (constant velocity's error)
  frames   world (as the network sees them) or the vertex's own frame (tangents along the grid's two directions and
           the normal, at t)
  models   ridge regression (linear), and a 2-layer MLP on the largest stencil
Error / constant velocity's on the held-out episodes, the network's 0.85-0.90 and the teacher's 0.24 beside it.

    CUDA_VISIBLE_DEVICES="" $L scripts/l1_local_probe.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0
TABLE_Z, FLOOR_Z = 0.0, -0.75


def samples(eps, R, per_ep, rng):
    """Over episodes, up to per_ep random (frame, vertex) picks per episode whose (2R + 1)^2 stencil is inside the
    cloth, holds no held vertex and none near the table or floor: stencils (n, 3 frames, 2R+1, 2R+1, 3) relative to
    the centre at t (world), the centre's local axes at t (n, 3, 3; rows), and its acceleration (n, 3; world)."""
    from scipy.ndimage import maximum_filter
    from lsim.data.scene import TABLE_HALF
    S, Ms, As = [], [], []
    for e in eps:
        nx, ny = int(e.c["nx"]), int(e.c["ny"])
        F = e.n_frames
        X = e.x(np.arange(F)).astype(np.float64).reshape(F, ny + 1, nx + 1, 3)
        H = e.held.reshape(F, ny + 1, nx + 1)
        low = ((np.abs(X[..., 2] - TABLE_Z) < 0.01) & (np.abs(X[..., 0]) <= TABLE_HALF[0] + 0.01) &
               (np.abs(X[..., 1]) <= TABLE_HALF[1] + 0.01)) | (X[..., 2] < FLOOR_Z + 0.01)
        bad = H | low
        bad[1:] |= H[:-1]
        bad = maximum_filter(bad.astype(np.uint8), size=(1, 2 * R + 1, 2 * R + 1), mode="constant", cval=1) > 0
        mv = np.zeros_like(bad)
        mv[1:] = np.linalg.norm(X[1:] - X[:-1], axis=-1) * FPS > 0.02
        ok = mv & ~bad
        ok[:3] = False
        ok[F - 1:] = False
        t, j, i = np.nonzero(ok)
        if not len(t):
            continue
        k = rng.choice(len(t), min(per_ep, len(t)), replace=False)
        t, j, i = t[k], j[k], i[k]
        d = np.arange(-R, R + 1)
        st = X[(t[:, None, None, None] + np.arange(-2, 1)[None, :, None, None]),
               (j[:, None, None, None] + d[None, None, :, None]),
               (i[:, None, None, None] + d[None, None, None, :])]           # (n, 3, 2R+1, 2R+1, 3)
        c = X[t, j, i]
        S.append((st - c[:, None, None, None]).astype(np.float32))
        As.append(X[t + 1, j, i] - 2 * c + X[t - 1, j, i])
        u = X[t, j, i + 1] - X[t, j, i - 1]
        v = X[t, j + 1, i] - X[t, j - 1, i]
        e1 = u / np.linalg.norm(u, axis=-1, keepdims=True)
        e2 = v - (v * e1).sum(-1, keepdims=True) * e1
        e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
        Ms.append(np.stack([e1, e2, np.cross(e1, e2)], 1))
    return np.concatenate(S), np.concatenate(Ms), np.concatenate(As)


def features(S, M, A, R, local):
    """Crop the stencils to radius R; in world or local axes: (features, target)."""
    Rm = (S.shape[2] - 1) // 2
    st = S[:, :, Rm - R:Rm + R + 1, Rm - R:Rm + R + 1].astype(np.float64)
    g = np.broadcast_to(np.array([0.0, 0.0, -1.0]), A.shape)
    if local:
        st = np.einsum("nab,n...b->n...a", M, st)
        A = np.einsum("nab,nb->na", M, A)
        g = np.einsum("nab,b->na", M, np.array([0.0, 0.0, -1.0]))
    return np.c_[st.reshape(len(st), -1), g].astype(np.float32), A


def ridge(Xtr, Ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))].astype(np.float64)
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ Ytr)
    return np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ W


def mlp(Xtr, Ytr, Xev, epochs=30, width=512, seed=0):
    """A 2-layer MLP fitted on (Xtr, Ytr); predictions for each array in Xev."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    ys = float(np.sqrt((Ytr ** 2).mean()))
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(Ytr / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], width), torch.nn.GELU(), torch.nn.Linear(width, width),
                              torch.nn.GELU(), torch.nn.Linear(width, 3))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((len(xt) + 1023) // 1024))
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 1024):
            k = perm[b:b + 1024]
            loss = ((net(xt[k]) - yt[k]) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return [net(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy().astype(np.float64) * ys for X in Xev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="c1m")
    ap.add_argument("--test", default="c1m_test")
    ap.add_argument("--train_shards", type=int, default=8)
    ap.add_argument("--per_ep", type=int, default=400)
    ap.add_argument("--radii", default="1,2,3,4")
    ap.add_argument("--mlp_epochs", type=int, default=30)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from lsim.train.cloth_windows import load_shard
    tr = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.train}/shard_*.npz")))[:a.train_shards]
          for e in load_shard(p[:-4])]
    te = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.test}/shard_*.npz"))) for e in load_shard(p[:-4])]
    print(f"{len(tr)} train / {len(te)} test episodes", flush=True)
    out = {"train": a.train, "test": a.test, "rows": []}
    radii = [int(r) for r in a.radii.split(",")]
    t0 = time.time()
    tr_s = samples(tr, max(radii), a.per_ep, np.random.default_rng(0))
    te_s = samples(te, max(radii), a.per_ep, np.random.default_rng(1))
    cv = float(np.linalg.norm(te_s[2], axis=-1).mean())
    out.update(n_train=len(tr_s[0]), n_test=len(te_s[0]), cv_mm=1e3 * cv)
    print(f"{len(tr_s[0])} / {len(te_s[0])} samples (same vertices for every R), constant velocity {1e3 * cv:.3f} mm, "
          f"{time.time() - t0:.0f} s", flush=True)
    for local in (False, True):
        fr = "local" if local else "world"
        for R in radii:
            Xtr, Ytr = features(*tr_s, R, local)
            Xte, Yte = features(*te_s, R, local)
            best = min(float(np.linalg.norm(ridge(Xtr, Ytr, Xte, lam) - Yte, axis=-1).mean() / cv)
                       for lam in (1e-6, 1e-4, 1e-2))
            out["rows"].append({"frame": fr, "R": R, "model": "linear", "ratio": best})
            print(f"  {fr:5s} R {R} linear: {best:.3f}", flush=True)
            if R == radii[-1] and a.mlp_epochs:
                t1 = time.time()
                n = len(Xte)
                P, Ptr = mlp(Xtr, Ytr, [Xte, Xtr[:n]], a.mlp_epochs)
                r = float(np.linalg.norm(P - Yte, axis=-1).mean() / cv)
                rtr = float(np.linalg.norm(Ptr - Ytr[:n], axis=-1).mean() / np.linalg.norm(Ytr[:n], axis=-1).mean())
                out["rows"].append({"frame": fr, "R": R, "model": "mlp", "ratio": r, "train_ratio": rtr})
                print(f"  {fr:5s} R {R} MLP: {r:.3f} held out, {rtr:.3f} on its training samples ({time.time() - t1:.0f} s)",
                      flush=True)
    json.dump(out, open(REPO / f"docs/results/l1_local_probe_{a.train}{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
