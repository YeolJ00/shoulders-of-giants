"""Does the local one-frame rule get better with more episodes (2026-11-20)? l1_local_probe's learning curve, on c1.

l1_local_probe fitted a small MLP to each moving free vertex's 9x9 grid stencil (3 frames, the vertex's own axes) on
one material: 0.71 of constant velocity's error held out but 0.52 on its own training samples -- it overfits, so the
local rule may be data-limited where the 64M network (train = test error) is not. Here the same kind of regressor is
fitted on nested subsets of c1's 20k training episodes (all materials; the episode's normalised material given as 4
inputs, as the teacher has it) and scored on c1_test, with early stopping on held-aside training episodes:
  curve      R 4, own axes, material given; 1/4/16/64/all of the training shards
  at all     R 2 / 4 / 6; R 4 without the material; R 4 in world axes; R 4 with a wider, deeper MLP
All on one vertex set (stencils clear at R 6). Phase 1 (CPU, spawn pool) extracts the samples to --cache; phase 2
(one GPU) fits. Error / constant velocity's on the held-out episodes; the teacher from 3 frames reaches 0.24.

    CUDA_VISIBLE_DEVICES="" $L scripts/l1_local_probe_scale.py --extract
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_local_probe_scale.py --fit
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
RMAX = 6                                  # the vertex set: stencils clear at this radius
STORE_R = 6                               # the stencil radius stored (--store_R; a cache's own is in its rest npz)


def _extract_shard(args):
    """One shard's samples (worker; spawn): stencils at RMAX, axes, accelerations and the material per sample."""
    path, k, per_ep, store_r = args
    from l1_local_probe import samples
    from lsim.train.cloth_windows import cloth_slots, load_shard
    eps = load_shard(path[:-4])
    rng = np.random.default_rng(1000 + k)
    out = []
    for e in eps:
        try:
            S, M, A = samples([e], RMAX, per_ep, rng)
        except ValueError:                                     # no usable vertex in this episode
            continue
        S = S[:, :, RMAX - store_r:RMAX + store_r + 1, RMAX - store_r:RMAX + store_r + 1]
        out.append((np.ascontiguousarray(S), M.astype(np.float32), A.astype(np.float32), np.repeat(cloth_slots(e.c)[None], len(S), 0)))
    if not out:
        return None
    return [np.concatenate(z) for z in zip(*out)] + [len(out)]


def extract(a):
    from multiprocessing import get_context
    tr = sorted(glob.glob(str(REPO / "data/datasets/c1/shard_*.npz")))
    te = sorted(glob.glob(str(REPO / "data/datasets/c1_test/shard_*.npz")))
    perm = np.random.default_rng(0).permutation(len(tr))
    tr = [tr[i] for i in perm]
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        for name, shards in (("test", te), ("train", tr)):
            res = pool.map(_extract_shard, [(p, k, a.per_ep, a.store_R) for k, p in enumerate(shards)], chunksize=1)
            keep = [(k, r) for k, r in enumerate(res) if r is not None]
            S, M, A, C = (np.concatenate([r[i] for _, r in keep]) for i in range(4))
            shard = np.concatenate([np.full(len(r[0]), k, np.int32) for k, r in keep])
            n_ep = {int(k): int(r[4]) for k, r in keep}
            np.save(Path(a.cache) / f"{name}_S.npy", S)
            np.savez(Path(a.cache) / f"{name}_rest.npz", M=M, A=A, C=C, shard=shard, store_R=a.store_R,
                     n_ep=np.array([n_ep.get(k, 0) for k in range(len(shards))]))
            print(f"{name}: {len(shards)} shards, {sum(n_ep.values())} episodes, {len(S)} samples, "
                  f"{time.time() - t0:.0f} s", flush=True)
            del S, M, A, C, res


def features(S, M, A, C, R, local, material, chunk=100000):
    """Crop to radius R, rotate into the vertex's own axes (local), append gravity and the material: (X, Y) float32 on
    the GPU, built in chunks (the full-radius stencils of all c1 do not fit twice)."""
    import torch
    dev = torch.device("cuda")
    n, w = len(S), 3 * (2 * R + 1) ** 2 * 3 + 3 + (4 if material else 0)
    X, Y = torch.empty(n, w, device=dev), torch.empty(n, 3, device=dev)
    gw = torch.tensor([0.0, 0.0, -1.0], device=dev)
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        Rs = (S.shape[2] - 1) // 2                                       # the radius stored
        st = torch.from_numpy(np.ascontiguousarray(S[i:j, :, Rs - R:Rs + R + 1, Rs - R:Rs + R + 1])).to(dev)
        Mt, At = torch.from_numpy(M[i:j]).to(dev), torch.from_numpy(A[i:j]).to(dev)
        g = gw.expand(j - i, 3)
        if local:
            st = torch.einsum("nab,ntijb->ntija", Mt, st)
            At = torch.einsum("nab,nb->na", Mt, At)
            g = torch.einsum("nab,b->na", Mt, gw)
        parts = [st.reshape(j - i, -1), g] + ([torch.from_numpy(C[i:j]).to(dev)] if material else [])
        X[i:j] = torch.cat(parts, 1)
        Y[i:j] = At
    return X, Y


def train_noise(sig, R, K=8):
    """The cloth trainer's chunk noise on a batch of stencils (raw X, Y): each sample is one of the K steps of a window,
    k uniform in 0..K-1; the frames after the clean past carry a per-vertex random walk of sig m per frame scaled by
    tau = sigmoid(N(0, 1)) (step_loss, --noise vertex), so step k sees it on its last min(k, 3) input frames, and its
    target is the clean next frame from the noised ones. Isotropic, so drawn in the vertex's own axes."""
    import torch

    def apply(X, Y):
        n, S = len(X), (2 * R + 1) ** 2
        k = torch.randint(0, K, (n, 1, 1), device=X.device).float()
        tau = torch.sigmoid(torch.randn(n, 1, 1, device=X.device))
        w2 = torch.randn(n, S, 3, device=X.device) * sig * (k - 2).clamp_min(0).sqrt()     # frame t-2: k-2 steps
        w1 = w2 + torch.randn(n, S, 3, device=X.device) * sig * (k >= 2).float()           # t-1: k-1
        w0 = w1 + torch.randn(n, S, 3, device=X.device) * sig * (k >= 1).float()           # t: k
        W = torch.stack([w2, w1, w0], 1) * tau[:, None]                                    # (n, 3 frames, S, 3)
        c = R * (2 * R + 1) + R
        W = W - W[:, 2:3, c:c + 1]                                      # the stencil is relative to the centre at t
        Xn = X.clone()
        Xn[:, :3 * S * 3] += W.reshape(n, -1)
        Wc = torch.stack([w2, w1, w0], 1)[:, :, c] * tau                # the centre's own noise per frame (n, 3, 3)
        return Xn, Y - 2 * Wc[:, 2] + Wc[:, 1]
    return apply


def fit(Xtr, Ytr, Xva, Yva, Xev, width, depth, steps, batch=2048, lr=1e-3, seed=0, noise=None):
    """An MLP fitted on (Xtr, Ytr) with the val-best state kept; predictions for each tensor in Xev. noise: a function
    (X, Y) -> (X, Y) applied to each raw training batch."""
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    ys = Ytr.pow(2).mean().sqrt()
    layers, w_in = [], Xtr.shape[1]
    for _ in range(depth):
        layers += [torch.nn.Linear(w_in, width), torch.nn.GELU()]
        w_in = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(w_in, 3)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)

    def pred(X):
        with torch.no_grad():
            return torch.cat([net((X[i:i + 65536] - mu) / sd) for i in range(0, len(X), 65536)]) * ys

    best, best_state, best_step = float("inf"), None, 0
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for s in range(steps):
        k = torch.randint(0, len(Xtr), (batch,), device="cuda", generator=gen)
        Xb, Yb = (Xtr[k], Ytr[k]) if noise is None else noise(Xtr[k], Ytr[k])
        loss = (net((Xb - mu) / sd) - Yb / ys).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (s + 1) % 1000 == 0 or s + 1 == steps:
            v = float((pred(Xva) - Yva).norm(dim=-1).mean())
            if v < best:
                best, best_step = v, s + 1
                best_state = {n: p.detach().clone() for n, p in net.state_dict().items()}
    net.load_state_dict(best_state)
    return [pred(X) for X in Xev], best_step


def fit_all(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False                  # the stencil's sub-mm offsets
    c = Path(a.cache)
    S_tr, S_te = np.load(c / "train_S.npy", mmap_mode="r"), np.load(c / "test_S.npy")
    rtr, rte = dict(np.load(c / "train_rest.npz")), dict(np.load(c / "test_rest.npz"))
    n_ep = rtr["n_ep"]
    n_sh = len(n_ep)
    va_sh = np.arange(n_sh - a.val_shards, n_sh)                    # held-aside training shards: early stopping
    va = np.isin(rtr["shard"], va_sh)
    cv = float(np.linalg.norm(rte["A"], axis=-1).mean())
    out = {"cv_mm": 1e3 * cv, "n_test": int(len(S_te)), "val_episodes": int(n_ep[va_sh].sum()), "rows": []}
    print(f"test {len(S_te)} samples, constant velocity {1e3 * cv:.3f} mm; val {int(va.sum())} samples", flush=True)
    curve = [int(s) for s in a.curve.split(",")]
    runs = [dict(shards=s, R=4, local=True, material=True, width=a.width, depth=a.depth) for s in curve]
    full = curve[-1]
    runs += [dict(shards=full, R=2, local=True, material=True, width=a.width, depth=a.depth),
             dict(shards=full, R=6, local=True, material=True, width=a.width, depth=a.depth),
             dict(shards=full, R=4, local=True, material=False, width=a.width, depth=a.depth),
             dict(shards=full, R=4, local=False, material=True, width=a.width, depth=a.depth),
             dict(shards=full, R=4, local=True, material=True, width=2 * a.width, depth=a.depth + 1)]
    for r in runs:
        t0 = time.time()
        sel = np.isin(rtr["shard"], np.arange(min(r["shards"], n_sh - a.val_shards)))
        idx = np.nonzero(sel)[0]
        vidx = np.nonzero(va)[0]
        f = lambda S, rest, i: features(S[i], rest["M"][i], rest["A"][i], rest["C"][i], r["R"], r["local"],
                                        r["material"])
        Xtr, Ytr = f(S_tr, rtr, idx)
        Xva, Yva = f(S_tr, rtr, vidx)
        Xte, Yte = f(S_te, rte, slice(None))
        n = min(len(Xte), len(Xtr))
        steps = a.steps if r["shards"] >= 16 else a.steps // 2
        (Pte, Ptr), best_step = fit(Xtr, Ytr, Xva, Yva, [Xte, Xtr[:n]], r["width"], r["depth"], steps)
        row = dict(r, episodes=int(n_ep[:min(r["shards"], n_sh - a.val_shards)].sum()), n_train=int(len(idx)),
                   ratio=float((Pte - Yte).norm(dim=-1).mean()) / cv,
                   train_ratio=float((Ptr - Ytr[:n]).norm(dim=-1).mean() / Ytr[:n].norm(dim=-1).mean()),
                   best_step=best_step, steps=steps)
        out["rows"].append(row)
        print(f"  {row['episodes']:6d} episodes ({row['n_train']} samples)  R {r['R']}  "
              f"{'own' if r['local'] else 'world'} axes  material {'given' if r['material'] else 'hidden'}  "
              f"MLP {r['width']}x{r['depth']}: held out {row['ratio']:.3f}, train {row['train_ratio']:.3f} "
              f"(best at {best_step}/{steps}, {time.time() - t0:.0f} s)", flush=True)
        del Xtr, Ytr, Xva, Yva, Xte, Yte, Pte, Ptr
        torch.cuda.empty_cache()
        json.dump(out, open(REPO / "docs/results/l1_local_probe_scale.json", "w"), indent=1)


def material_from_motion(a):
    """Can a vertex's 3-frame stencil (R 4, own axes) tell the material? An MLP regresses the 4 normalised slots from it
    (no material input); R^2 per slot on c1_test. If high, the material given adds nothing a local rule needs."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    c = Path(a.cache)
    S_tr, S_te = np.load(c / "train_S.npy", mmap_mode="r"), np.load(c / "test_S.npy")
    rtr, rte = dict(np.load(c / "train_rest.npz")), dict(np.load(c / "test_rest.npz"))
    n_sh = len(rtr["n_ep"])
    va = np.isin(rtr["shard"], np.arange(n_sh - a.val_shards, n_sh))
    idx, vidx = np.nonzero(~va)[0], np.nonzero(va)[0]
    f = lambda S, rest, i: features(S[i], rest["M"][i], rest["A"][i], rest["C"][i], 4, True, False)[0]
    Xtr, Xva, Xte = f(S_tr, rtr, idx), f(S_tr, rtr, vidx), f(S_te, rte, slice(None))
    Ctr, Cva, Cte = (torch.from_numpy(r["C"][i]).cuda() for r, i in ((rtr, idx), (rtr, vidx), (rte, slice(None))))
    out = {"slots": ["density", "stretch", "bend", "mu"], "rows": []}
    for k in range(4):
        y = lambda C: C[:, k:k + 1].expand(-1, 3).contiguous()         # fit() wants 3 outputs: the slot thrice
        (P,), best = fit(Xtr, y(Ctr), Xva, y(Cva), [Xte], a.width, a.depth, a.steps // 2)
        p, t = P[:, 0], Cte[:, k]
        r2 = 1 - float(((p - t) ** 2).mean() / ((t - t.mean()) ** 2).mean())
        out["rows"].append({"slot": out["slots"][k], "r2_vertex": r2, "best_step": best})
        print(f"  {out['slots'][k]:8s} from one vertex's 3-frame stencil: R^2 {r2:.3f} (best at {best})", flush=True)
    json.dump(out, open(REPO / "docs/results/l1_local_probe_material.json", "w"), indent=1)


def noise_test(a):
    """Does the trainer's chunk noise keep the local rule from being learned? The all-episode R 4 fit (own axes,
    material given) trained with that noise on its inputs (sig 1e-3, the recipe's; and 1e-4), scored on clean history
    as the network is (its first step)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    c = Path(a.cache)
    S_tr, S_te = np.load(c / "train_S.npy", mmap_mode="r"), np.load(c / "test_S.npy")
    rtr, rte = dict(np.load(c / "train_rest.npz")), dict(np.load(c / "test_rest.npz"))
    n_sh = len(rtr["n_ep"])
    va = np.isin(rtr["shard"], np.arange(n_sh - a.val_shards, n_sh))
    idx, vidx = np.nonzero(~va)[0], np.nonzero(va)[0]
    f = lambda S, rest, i: features(S[i], rest["M"][i], rest["A"][i], rest["C"][i], 4, True, True)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = f(S_tr, rtr, idx), f(S_tr, rtr, vidx), f(S_te, rte, slice(None))
    cv = float(Yte.norm(dim=-1).mean())
    out = {"cv_mm": 1e3 * cv, "rows": []}
    for sig in (1e-3, 1e-4, 1e-5):
        t0 = time.time()
        (P,), best = fit(Xtr, Ytr, Xva, Yva, [Xte], a.width, a.depth, a.steps, noise=train_noise(sig, 4))
        r = float((P - Yte).norm(dim=-1).mean()) / cv
        out["rows"].append({"sig": sig, "ratio": r, "best_step": best})
        print(f"  trained with the chunk noise at {1e3 * sig:g} mm per frame: held out (clean history) {r:.3f} "
              f"(best at {best}, {time.time() - t0:.0f} s)", flush=True)
        json.dump(out, open(REPO / "docs/results/l1_local_probe_noise.json", "w"), indent=1)


def per_ep_test(a):
    """Episodes or samples? The R 4 fit (own axes, material given) on --cache's denser draw (--per_ep 450 per episode)
    against a third of it (every third sample of each episode: ~150, as the curve), at 64 shards and at all."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    c = Path(a.cache)
    S_tr, S_te = np.load(c / "train_S.npy", mmap_mode="r"), np.load(c / "test_S.npy")
    rtr, rte = dict(np.load(c / "train_rest.npz")), dict(np.load(c / "test_rest.npz"))
    n_ep, n_sh = rtr["n_ep"], len(rtr["n_ep"])
    va = np.isin(rtr["shard"], np.arange(n_sh - a.val_shards, n_sh))
    vidx = np.nonzero(va)[0][::3]
    f = lambda S, rest, i: features(S[i], rest["M"][i], rest["A"][i], rest["C"][i], 4, True, True)
    Xva, Yva = f(S_tr, rtr, vidx)
    Xte, Yte = f(S_te, rte, slice(None))
    cv = float(Yte.norm(dim=-1).mean())
    out = {"cv_mm": 1e3 * cv, "n_test": int(len(Yte)), "rows": []}
    # samples of one episode are contiguous (extraction order); a third = every third one of the whole array
    n = len(rtr["shard"])
    third = np.arange(n) % 3 == 0
    for shards in (64, n_sh - a.val_shards):
        for frac, keep in (("all", np.ones(n, bool)), ("a third", third)):
            t0 = time.time()
            idx = np.nonzero(np.isin(rtr["shard"], np.arange(min(shards, n_sh - a.val_shards))) & keep)[0]
            Xtr, Ytr = f(S_tr, rtr, idx)
            (P,), best = fit(Xtr, Ytr, Xva, Yva, [Xte], a.width, a.depth, a.steps)
            r = float((P - Yte).norm(dim=-1).mean()) / cv
            row = {"shards": int(shards), "episodes": int(n_ep[:min(shards, n_sh - a.val_shards)].sum()),
                   "samples": frac, "n_train": int(len(idx)), "ratio": r, "best_step": best}
            out["rows"].append(row)
            print(f"  {row['episodes']:6d} episodes, {frac:7s} of the draw ({len(idx)} samples): held out {r:.3f} "
                  f"(best at {best}, {time.time() - t0:.0f} s)", flush=True)
            del Xtr, Ytr, P
            torch.cuda.empty_cache()
            json.dump(out, open(REPO / "docs/results/l1_local_probe_per_ep.json", "w"), indent=1)


def long_test(a):
    """Is the dense all-episode fit (per_ep_test's best, 0.662, best at 32k of 40k steps) limited by training length?
    The same fit for --steps, at width 1024 x 3 and 2048 x 4."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    c = Path(a.cache)
    S_tr, S_te = np.load(c / "train_S.npy", mmap_mode="r"), np.load(c / "test_S.npy")
    rtr, rte = dict(np.load(c / "train_rest.npz")), dict(np.load(c / "test_rest.npz"))
    n_sh = len(rtr["n_ep"])
    va = np.isin(rtr["shard"], np.arange(n_sh - a.val_shards, n_sh))
    f = lambda S, rest, i: features(S[i], rest["M"][i], rest["A"][i], rest["C"][i], 4, True, True)
    Xva, Yva = f(S_tr, rtr, np.nonzero(va)[0][::3])
    Xte, Yte = f(S_te, rte, slice(None))
    Xtr, Ytr = f(S_tr, rtr, np.nonzero(~va)[0])
    cv = float(Yte.norm(dim=-1).mean())
    out = {"cv_mm": 1e3 * cv, "steps": a.steps, "rows": []}
    for width, depth in ((1024, 3), (2048, 4)):
        t0 = time.time()
        n = min(len(Xte), len(Xtr))
        (P, Ptr), best = fit(Xtr, Ytr, Xva, Yva, [Xte, Xtr[:n]], width, depth, a.steps)
        r = float((P - Yte).norm(dim=-1).mean()) / cv
        rtr_ = float((Ptr - Ytr[:n]).norm(dim=-1).mean() / Ytr[:n].norm(dim=-1).mean())
        out["rows"].append({"width": width, "depth": depth, "ratio": r, "train_ratio": rtr_, "best_step": best})
        print(f"  MLP {width}x{depth}, {a.steps} steps: held out {r:.3f}, train {rtr_:.3f} (best at {best}, "
              f"{time.time() - t0:.0f} s)", flush=True)
        json.dump(out, open(REPO / "docs/results/l1_local_probe_long.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--material_from_motion", action="store_true")
    ap.add_argument("--noise_test", action="store_true")
    ap.add_argument("--per_ep_test", action="store_true")
    ap.add_argument("--long_test", action="store_true")
    ap.add_argument("--store_R", type=int, default=6, help="stencil radius kept in the cache (the vertex set is RMAX's)")
    ap.add_argument("--cache", default=str(REPO / "outputs/lsim/probe_scale"))
    ap.add_argument("--per_ep", type=int, default=150)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val_shards", type=int, default=5)
    ap.add_argument("--curve", default="1,4,16,64,308")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40000)
    a = ap.parse_args()
    if a.extract:
        extract(a)
    if a.fit:
        fit_all(a)
    if a.material_from_motion:
        material_from_motion(a)
    if a.noise_test:
        noise_test(a)
    if a.per_ep_test:
        per_ep_test(a)
    if a.long_test:
        long_test(a)


if __name__ == "__main__":
    main()
