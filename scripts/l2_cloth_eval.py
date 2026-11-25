"""L2 -- the cloth model on 30-frame rollouts (phase 3).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_cloth_eval.py --ckpt outputs/lsim/cloth_v0/last.pt --data data/datasets/c1_test

From a random frame of a random episode, with the true 8 frames before it, the model rolls the cloth out frame by
frame (lsim.train.cloth_train.rollout: the gripper, the held vertices and the static scene follow the recording).
Error: mean distance of the free cloth vertices to the teacher's, per frame ahead; resting cloth (moving < 1 mm
over the 30 frames in the teacher) and moving cloth apart, as the rigid drift diagnostic does. Beside it:

  default   the physics layer with the network's forces at zero (gravity, air drag, Coulomb contact, the grasp)
  constvel  every free vertex keeps its last velocity
  noise     the teacher against itself: two runs from the same state drift apart by chaos (cloth identifiability
            gate, docs/results/l1_cloth_gate.json: median 0.03 mm at 2 frames, 0.5 mm at 10, 5 mm at 30)

--ensemble M (G5): per start, M rollouts with hidden slots sampled from the slot posterior and the teacher's own
per-frame noise (--chaos, m) on the free vertices; spread (rms over vertices of the ensemble's spread, with the
(M + 1) / M finite-ensemble factor) against skill (rms error of the ensemble mean), as GenCast checks its ensembles:
calibrated at a ratio of 0.7-1.3.

Writes docs/results/l2_cloth_eval_<tag>.json.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--data", default="data/datasets/c1_test")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--skip", type=int, default=0, help="skip the first n shards (those a run trained on)")
    ap.add_argument("--episodes", type=int, default=0, help="only the first n episodes (overfit checks)")
    ap.add_argument("--n", type=int, default=48, help="rollouts")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--hidden", default="", help="slots hidden, comma-separated (density,stretch,bend,mu)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--chaos", default="2e-5", help="m per frame on free vertices in ensemble members, or 'unc': the model's own")
    ap.add_argument("--layer", type=int, default=0, help="run the network under this exact layer (0: the one it trained with)")
    a = ap.parse_args()
    from lsim.model.cloth_net import build_cloth_model
    from lsim.train.cloth_train import rollout
    from lsim.train.cloth_windows import load_cloth_shard
    dev = "cuda"
    eps = []
    for p in sorted(glob.glob(f"{a.data}/shard_*.npz"))[a.skip:a.skip + a.shards]:
        eps += load_cloth_shard(p[:-4])
    if a.episodes:
        eps = eps[:a.episodes]
    hidden = tuple(h for h in a.hidden.split(",") if h)

    class Zero(torch.nn.Module):                                      # the physics layer alone: the model's (layer 2 at
        layer = 2                                                     # least), or --layer's without a checkpoint

        def forward(self, b):
            N, Fw, _ = b["pos"].shape
            z = torch.zeros(N, Fw, 3, device=b["pos"].device)
            return z, z, torch.zeros(int(b["n_bodies"]), 4, device=b["pos"].device), None
    models = {"default": Zero()}
    models["default"].layer = max(2, a.layer)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev)
        m = build_cloth_model(ck["args"]).to(dev).eval()
        m.load_state_dict(ck["model"])
        models = {"model": m, **models}
        models["default"].layer = max(2, m.layer)
    rng = np.random.default_rng(0)
    T = a.frames
    errs = {k: [] for k in list(models) + ["constvel"]}
    moving = []
    for i in range(a.n):
        e = eps[rng.integers(len(eps))]
        t0 = int(rng.integers(8, e.n_frames - T - 1))
        true = e.x(np.arange(t0, t0 + T + 1))                        # frames t0 .. t0 + T (t0 = now)
        free = ~e.held[t0:t0 + T]                                     # (T, V): free in the step into frame t0 + k + 1
        moving.append(float(np.linalg.norm(true[-1] - true[0], axis=-1).mean()) > 1e-3)
        for k, m in models.items():
            with torch.no_grad():
                pred, tr, _ = rollout(m, e, t0, T, dev=dev, hidden=hidden)   # frames t0 + 1 .. t0 + T
            d = np.linalg.norm(pred - tr, axis=-1)                    # (T, V)
            errs[k].append([float(d[j][free[j]].mean()) if free[j].any() else 0.0 for j in range(T)])
        cv = true[0][None] + np.arange(1, T + 1)[:, None, None] * (true[0] - e.x([t0 - 1])[0])[None]
        d = np.linalg.norm(cv - true[1:T + 1], axis=-1)
        errs["constvel"].append([float(d[j][free[j]].mean()) for j in range(T)])
    if a.ensemble and a.ckpt:                                         # G5: spread against skill
        ratio, sp_all, sk_all = [], [], []
        rng = np.random.default_rng(1)
        for i in range(a.n):
            e = eps[rng.integers(len(eps))]
            t0 = int(rng.integers(8, e.n_frames - T - 1))
            runs = []
            for k in range(a.ensemble):
                with torch.no_grad():
                    chaos = a.chaos if a.chaos == "unc" else float(a.chaos)
                    pred, tr, _ = rollout(models["model"], e, t0, T, dev=dev, hidden=hidden, sample=True, chaos=chaos, seed=k + 1)
                runs.append(pred)
            runs = np.stack(runs)                                     # (M, T, V, 3)
            free = ~e.held[t0:t0 + T]
            mean = runs.mean(0)
            var = runs.var(0, ddof=1).sum(-1)                         # (T, V)
            se = ((mean - tr) ** 2).sum(-1)
            sp = np.array([np.sqrt(var[j][free[j]].mean() * (a.ensemble + 1) / a.ensemble) for j in range(T)])
            sk = np.array([np.sqrt(se[j][free[j]].mean()) for j in range(T)])
            sp_all.append(sp); sk_all.append(sk)
        sp_all, sk_all = 1e3 * np.array(sp_all), 1e3 * np.array(sk_all)
        res_g5 = {f"f{j + 1}": {"spread_mm": float(np.sqrt((sp_all[:, j] ** 2).mean())), "skill_mm": float(np.sqrt((sk_all[:, j] ** 2).mean())),
                                "ratio": float(np.sqrt((sp_all[:, j] ** 2).mean() / max((sk_all[:, j] ** 2).mean(), 1e-12)))}
                  for j in (0, 4, 9, 29) if j < T}
        print("G5 " + " | ".join(f"{f}: spread {r['spread_mm']:.2f} mm, skill {r['skill_mm']:.2f} mm, ratio {r['ratio']:.2f}" for f, r in res_g5.items()), flush=True)
    moving = np.array(moving)
    res = {"n": a.n, "frames": T, "data": a.data, "ckpt": a.ckpt, "hidden": list(hidden), "moving_fraction": float(moving.mean()),
           "layer": int(models["model"].layer) if a.ckpt else int(models["default"].layer)}
    for k, v in errs.items():
        v = 1e3 * np.array(v)
        res[k] = {}
        for sel, name in ((np.ones(len(v), bool), "all"), (~moving, "resting"), (moving, "moving")):
            if sel.any():
                res[k][name] = {f"f{j + 1}": [float(np.median(v[sel, j])), float(np.percentile(v[sel, j], 90))] for j in (0, 4, 9, 29) if j < T}
        print(f"{k:9s} " + " | ".join(f"{name}: " + ", ".join(f"{f} {m:.2f} (p90 {p:.1f})" for f, (m, p) in r.items()) for name, r in res[k].items()),
              flush=True)
    if a.ensemble and a.ckpt:
        res["g5"] = dict(res_g5, members=a.ensemble, chaos_m=a.chaos)
    tag = a.tag or (Path(a.ckpt).parent.name if a.ckpt else "default")
    (REPO / f"docs/results/l2_cloth_eval_{tag}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
