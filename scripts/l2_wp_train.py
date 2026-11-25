"""Train WorldParticle (arXiv 2605.15305; data/external/WorldParticle, which ships the model and inference but no
trainer) on our cloth episodes (scripts/l2_wp_data.py export), as the paper describes its training: autoregressive
rollouts of W = 5 steps with an MSE loss on positions and velocities, AdamW at 1e-4 with a linear warmup then cosine
decay, batch 1, bf16. Run in its own env (torch 2.2 + open3d 0.19; never the lsim env):

    ATTN_IMPL=flash_attn CUDA_VISIBLE_DEVICES=<gpu> /home/nas5/jooyeolyun/envs/worldparticle/bin/python \
        scripts/l2_wp_train.py --data outputs/lsim/wp/data/train --out outputs/lsim/wp/<run> --hours 7.2

Adapting it to our scenes (2026-11-24), all of it in the model's own terms:
  - particles: the cloth's vertices and the gripper's 110 surface points; the gripper's points and the held cloth
    vertices are DRIVEN -- after every step their positions and velocities are overwritten with the recorded ones,
    which is how the released code treats driven particles ('force position override', its PhysTwin setting);
  - per-particle features (the model's `feats`, 9 channels): is-gripper, is-held, the driven particles' next
    displacement (cm; zero for free cloth: the same information our network gets as the gripper's next move and the
    held vertices' given positions), and the cloth's four material slots (zero on the gripper);
  - obstacles (the model's `box`, static per episode): table, floor, plank and object points with normal + log mu;
  - topology: the cloth mesh's neighbours (CSR), the model's topology branch, as its 'cloth_on_ball' data type;
  - physics step: gravity along -z at 30 fps (its explicit predictor), learned position and velocity corrections
    (velhead) -- the published configuration otherwise (config.yaml: 384-wide continuous convolutions, 6 merge
    layers of 16 heads, 8 decoder layers of 12 heads).
The loss is taken on free cloth vertices only, positions and velocities x the time step, both in mm. Training stops at
a wall-clock budget (--hours), with the cosine schedule laid over that budget: the comparison is at equal GPU-hours.
"""
import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data/external/WorldParticle"))
from trajectory_model import TrajectoryModel  # noqa: E402

DT = 1.0 / 30.0
N_FEATS = 9


def config(particle_radius=0.005):
    return {
        "device": "cuda", "gravity": [0.0, 0.0, -9.81], "timestep": DT, "particle_radius": particle_radius,
        "kernel_size": [4, 4, 4], "cconv_embedding_dim": 384, "radius_scale": 1.5,
        "use_topology_conv": True, "topology_search_radius": particle_radius, "topology_blend_alpha": None,
        "slot_compression_ratio": 4, "slot_attn_iters": 6, "slot_attn_heads": 16, "k_partite": 2,
        "particle_position_rope_dim": 48, "num_decoder_layers": 8, "decoder_attn_heads": 12,
        "decoder_attn_ffn_hidden_dim": 512, "decoder_attn_dropout": 0.1, "output_hidden_dim": 512, "output_layers": 5,
        "verbose": False, "data": "cloth_on_ball", "velhead": True,
        "resolved_other_feats_channels": N_FEATS, "resolved_obstacle_feats_channels": 4,
    }


class Episode:
    def __init__(self, path, dev):
        z = np.load(path)
        self.name = Path(path).stem
        self.x = torch.from_numpy(z["x"]).to(dev)                      # (T, N, 3)
        self.T, self.N = self.x.shape[0], self.x.shape[1]
        self.Vc = int(z["n_cloth"])
        held = torch.zeros(self.T, self.N, dtype=torch.bool, device=dev)
        held[:, :self.Vc] = torch.from_numpy(z["held"]).to(dev)
        self.held = held
        self.grip = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.grip[self.Vc:] = True
        self.slots = torch.from_numpy(z["slots"]).to(dev)
        self.box = torch.from_numpy(z["box"]).to(dev)
        self.box_feats = torch.from_numpy(z["box_feats"]).to(dev)
        self.nbr_index = torch.from_numpy(z["nbr_index"]).to(dev)
        self.nbr_splits = torch.from_numpy(z["nbr_splits"]).to(dev)

    def driven(self, t):
        return self.grip | self.held[t]

    def feats(self, t):
        f = torch.zeros(self.N, N_FEATS, device=self.x.device)
        d = self.driven(t)
        f[:, 0] = self.grip.float()
        f[:, 1] = self.held[t].float()
        nxt = self.x[min(t + 1, self.T - 1)] - self.x[t]
        f[:, 2:5] = torch.where(d[:, None], nxt / 0.01, torch.zeros_like(nxt))
        f[:self.Vc, 5:9] = self.slots
        return f

    def state(self, t):
        return self.x[t], (self.x[t] - self.x[max(t - 1, 0)]) / DT


def step(model, ep, pos, vel, t):
    """One model step from frame t, then the driven particles put on their recorded frame t + 1."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=os.environ.get("WP_FP32") != "1"):   # WP_FP32=1 needs ATTN_IMPL=sdpa
        p, v = model(pos, vel, ep.feats(t), ep.box, ep.box_feats,
                     topology_neighbors_index=ep.nbr_index, topology_neighbors_row_splits=ep.nbr_splits)
    p, v = p.float(), v.float()
    d = ep.driven(t)[:, None]
    tp, tv = ep.state(min(t + 1, ep.T - 1))
    return torch.where(d, tp, p), torch.where(d, tv, v), p, v


def build(cfg, dev):
    model = TrajectoryModel(dict(cfg)).to(dev)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=7.2)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many steps (smoke tests)")
    ap.add_argument("--particle_radius", type=float, default=0.005)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(REPO / a.data / "*.npz")))
    t0 = time.time()
    eps = [Episode(f, dev) for f in files]
    print(f"{len(eps)} episodes loaded in {time.time() - t0:.0f} s; particles {min(e.N for e in eps)}-{max(e.N for e in eps)}", flush=True)
    cfg = config(a.particle_radius)
    model = build(cfg, dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{n_par / 1e6:.1f}M parameters; attention backend {os.environ.get('ATTN_IMPL', 'flash_attn')}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    budget = a.hours * 3600.0
    W = a.window
    start = time.time()
    step_i, hist, last_ckpt = 0, [], start
    while True:
        el = time.time() - start
        if el > budget or (a.max_steps and step_i >= a.max_steps):
            break
        lr = a.lr * min(1.0, (step_i + 1) / a.warmup) * 0.5 * (1.0 + math.cos(math.pi * min(el / budget, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        ep = eps[int(rng.integers(len(eps)))]
        t = int(rng.integers(1, ep.T - 1 - W))
        pos, vel = ep.state(t)
        loss, e_pos = 0.0, []
        model.train()
        for k in range(W):
            pos, vel, p_raw, v_raw = step(model, ep, pos, vel, t + k)
            tp, tv = ep.state(t + k + 1)
            free = ~ep.driven(t + k)
            dp = (p_raw - tp)[free] * 1e3                                 # mm
            dv = (v_raw - tv)[free] * DT * 1e3                            # mm per frame
            loss = loss + ((dp ** 2).sum(-1).mean() + (dv ** 2).sum(-1).mean()) / W
            e_pos.append(dp.norm(dim=-1).mean().item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append((loss.item(), e_pos[0], e_pos[-1], float(gn)))
        step_i += 1
        if step_i % 200 == 0 or step_i == 1:
            h = np.array(hist[-200:])
            print(f"step {step_i:6d}  loss {h[:, 0].mean():.3e}  first / last step error {h[:, 1].mean():.2f} / {h[:, 2].mean():.2f} mm"
                  f"  grad {h[:, 3].mean():.2f}  lr {lr:.2e}  {time.time() - start:.0f} s  {(time.time() - start) / step_i:.3f} s/step", flush=True)
        if time.time() - last_ckpt > 3600:
            torch.save({"model": model.model.state_dict(), "cfg": cfg, "step": step_i}, out / "last.pt")
            last_ckpt = time.time()
    torch.save({"model": model.model.state_dict(), "cfg": cfg, "step": step_i}, out / "last.pt")
    print(f"done: {step_i} steps in {(time.time() - start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
