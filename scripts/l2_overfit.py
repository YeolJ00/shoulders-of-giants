"""L2 -- Phase 2 unit check: the model must overfit one episode, then 100 (DIRECTION.md).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1 --steps 1500 [--plain] [--mode onestep|chunk|ar]

Same training step as lsim.train.train (diffusion forcing, slots, physics layer), on a fixed
small set of episodes. Every 250 steps, on 16 FIXED windows of those episodes: the chunk
position RMSE of a single prediction at chunk noise tau in {0, 0.2, 0.5, 1}, and of the 10-step
sampler (lsim.model.sampler) -- first and last chunk frame -- for the chunk ablation, and for
every mode the frame-by-frame rollout from the true past ("ar").
"""
import argparse
import glob
import time

import numpy as np
import torch

from lsim.model.net import Simulator
from lsim.model.sampler import sample_chunk
from lsim.train.train import collate, step_loss, step_loss_ar, step_loss_chunk
from lsim.train.windows import PARAMS, load_shard, make_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/v0")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--plain", action="store_true", help="no augmentation, every slot given, all past frames, no noise")
    ap.add_argument("--bidir", action="store_true", help="time attention sees the whole window, not only earlier frames")
    ap.add_argument("--mode", default="onestep", choices=("onestep", "chunk", "ar"),
                    help="training loss: step_loss (default), step_loss_chunk or step_loss_ar")
    ap.add_argument("--out", default="")
    ap.add_argument("--pair_norm", action="store_true", help="contact forces per body pair (Simulator pair_norm)")
    ap.add_argument("--support", action="store_true", help="default support on static contacts (Simulator support)")
    ap.add_argument("--coulomb", action="store_true", help="default Coulomb contact on static surfaces")
    ap.add_argument("--pf", type=int, default=0, help="pushforward: up to this many own-rollout frames (step_loss pf)")
    ap.add_argument("--l1", action="store_true", help="position loss on distances, not squared (step_loss l1)")
    ap.add_argument("--mix", action="store_true", help="position loss: distance + squared distance (step_loss mix)")
    ap.add_argument("--noise", default="0.01,0.05", help="chunk noise per frame: m, rad (step_loss sig_p, sig_r)")
    ap.add_argument("--hist_noise", default="0,0", help="lo,hi m/frame: unflagged history noise (step_loss sig_h)")
    ap.add_argument("--step_weight", type=float, default=0.0, help="m: noise-normalised step weights (step_loss w_sig)")
    ap.add_argument("--d", type=int, default=128, help="network width (the trainer's default)")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--direct", action="store_true", help="baseline B2: accelerations directly, no physics layer")
    ap.add_argument("--r_contact", type=float, default=0.03, help="m: contact-edge radius (Simulator r_contact; windows built to match)")
    ap.add_argument("--grip_force", action="store_true", help="learned force on gripper contact edges too (Simulator grip_force)")
    ap.add_argument("--trunk", default="graph", choices=("graph", "attn"), help="attn: the cloth network's blocks, whole-scene attention (Simulator trunk)")
    ap.add_argument("--aug", type=int, default=-1, choices=(-1, 0, 1), help="the random yaw and shift per window: 1 on (also with --plain), 0 off (also without it); -1 = as --plain sets it")
    ap.add_argument("--p_hidden", type=float, default=-1.0, help="probability a parameter-slot group is hidden (collate; -1 = as --plain sets it)")
    ap.add_argument("--p_hide_grip", type=float, default=-1.0, help="probability a window hides the gripper's force (make_window; -1 = as --plain sets it)")
    ap.add_argument("--all_past", type=int, default=-1, choices=(-1, 0, 1), help="0: a random shorter past per window (collate); -1 = as --plain sets it")
    ap.add_argument("--p_hist_noise", type=float, default=-1.0, help="probability of flagged history noise (collate; -1 = as --plain sets it)")
    ap.add_argument("--touch_cv", action="store_true", help="no default on a rigid body the gripper or another moving body is near: zero output = constant velocity there (Simulator touch_cv)")
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--p_start", type=float, default=0.0, help="probability a training window starts at frame 0: its whole "
                    "past is frame 0 repeated (the scene held still, then let go -- how l2_phys101.py starts the real cube); "
                    "otherwise starts are uniform from frame 1 as before. 0 draws no extra random numbers")
    ap.add_argument("--init", default="", help="start from this checkpoint's weights (new optimiser and schedule; the "
                    "network flags must match the checkpoint's)")
    ap.add_argument("--K", type=int, default=8, help="chunk frames per window (8 past frames always); --pf can reach K")
    ap.add_argument("--save_every", type=int, default=0, help="also save <out stem>_s<step>.pt every this many steps")
    a = ap.parse_args()
    assert a.pf <= a.K, f"--pf {a.pf} needs --K >= {a.pf}"
    noise = dict(zip(("sig_p", "sig_r"), (float(x) for x in a.noise.split(","))),
                 sig_h=tuple(float(x) for x in a.hist_noise.split(",")), w_sig=a.step_weight, pf=a.pf, l1=a.l1, mix=a.mix)
    ck = dict(p_hidden=0.0, p_hist_noise=0.0, augment=False, all_past=True) if a.plain else {}
    if a.aug >= 0:
        ck["augment"] = bool(a.aug)
    if a.p_hidden >= 0:
        ck["p_hidden"] = a.p_hidden
    if a.all_past >= 0:
        ck["all_past"] = bool(a.all_past)
    if a.p_hist_noise >= 0:
        ck["p_hist_noise"] = a.p_hist_noise
    wk = dict(near=a.r_contact, **(dict(p_hide_grip=0.0) if a.plain else {}))
    if a.p_hide_grip >= 0:
        wk["p_hide_grip"] = a.p_hide_grip
    dev = torch.device("cuda")
    shards = sorted(glob.glob(f"{a.data}/shard_*.npz"))
    eps = []
    for sh in shards[:max(1, -(-a.episodes // 128))]:               # 128 episodes a shard: the first ones, in order
        eps += [e for e in load_shard(sh[:-4]) if e.bodies and any(b.role == "dynamic" for b in e.bodies)]
    eps = eps[:a.episodes]
    print(f"{len(eps)} episodes from the first {max(1, -(-a.episodes // 128))} shard(s)", flush=True)
    model = Simulator(a.d, a.layers, n_params=len(PARAMS), causal=not a.bidir, direct=a.direct, pair_norm=a.pair_norm,
                      support="coulomb" if a.coulomb else a.support, r_contact=a.r_contact, grip_force=a.grip_force,
                      trunk=a.trunk, n_heads=a.n_heads, touch_cv=a.touch_cv).to(dev)
    wk["near"] = model.near
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev)["model"])
        print(f"weights from {a.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps + 1, pct_start=0.05)
    rng = np.random.default_rng(0)
    P, K = 8, a.K
    erng = np.random.default_rng(123)
    evals = []
    for i in range(16):
        e = eps[i % len(eps)]
        w = make_window(e, int(erng.integers(1, e.q.shape[0] - K)), P, K, np.random.default_rng(i), **wk)
        evals.append(collate([w], dev, np.random.default_rng(1000 + i), **ck))

    def evaluate():
        model.eval()
        out = {}
        with torch.no_grad():
            r = [step_loss_ar(model, b, np.random.default_rng(j), K, dev)[1] for j, b in enumerate(evals)]
            out["ar"] = (np.median([x["rmse_first_mm"] for x in r]), np.median([x["rmse_last_mm"] for x in r]))
            for tau in ((0.0, 0.2, 0.5, 1.0) if a.mode == "chunk" else ()):
                r = [step_loss_chunk(model, b, np.random.default_rng(j), K, dev, tau_fixed=tau)[1] for j, b in enumerate(evals)]
                out[f"tau{tau:g}"] = (np.median([x["rmse_first_mm"] for x in r]), np.median([x["rmse_last_mm"] for x in r]))
            g = torch.Generator(device=dev).manual_seed(0)
            r = []
            for b in (evals if a.mode == "chunk" else []):
                x0, first, _ = sample_chunk(model, b, K, steps=10, generator=g)
                dyn = (b["role"] == 2) & (b["type"] == 0)
                err = ((x0 - b["pos"][:, P:P + K]) ** 2).sum(-1)[dyn]
                r.append((1000 * err[:, 0].mean().sqrt().item(), 1000 * err[:, -1].mean().sqrt().item()))
            if r:
                out["sampler"] = tuple(np.median(np.array(r), 0))
        model.train()
        return "  ".join(f"{k} {v[0]:.1f}/{v[1]:.1f}" for k, v in out.items())

    t0 = time.time()
    for step in range(a.steps + 1):
        wins = [make_window(e, 0 if (a.p_start > 0 and rng.random() < a.p_start) else int(rng.integers(1, e.q.shape[0] - K)),
                            P, K, rng, **wk)
                for e in (eps[int(i)] for i in rng.integers(len(eps), size=a.batch))]
        b = collate(wins, dev, rng, **ck)
        loss, info = (step_loss(model, b, rng, K, dev, **noise) if a.mode == "onestep" else
                      {"chunk": step_loss_chunk, "ar": step_loss_ar}[a.mode](model, b, rng, K, dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0:
            print(f"step {step:5d}  loss {loss.item():.3g}  | first/last-frame rmse mm: {evaluate()}  {time.time()-t0:.0f} s", flush=True)
        if a.out and a.save_every and step and step % a.save_every == 0 and step < a.steps:
            torch.save({"model": model.state_dict(), "args": vars(a)}, a.out[:-3] + f"_s{step}.pt")
    if a.out:
        torch.save({"model": model.state_dict(), "args": vars(a)}, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
