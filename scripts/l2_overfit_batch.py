"""Can the cloth network fit a handful of windows at all (2026-11-10)?

The 100-episode runs saw each of their ~16k windows about twice, so they could not show memorisation. This is the
strict version: n FIXED windows (no noise, no augmentation, no pushforward, the same slots given every time), repeated
for --steps steps at a constant learning rate. A network and objective that can fit the teacher drive the one-step
error towards the teacher's own noise (~0.05 mm); a floor here is the architecture's, the objective's or the optimiser's.
  --loss full   the trainer's loss as it is (uncertainty head, slot posterior, slots hidden as in training)
  --loss pos    position + edge terms only (no uncertainty head, every slot given)

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_overfit_batch.py --loss pos --tag pos
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
    ap.add_argument("--loss", default="full", choices=("full", "pos"))
    ap.add_argument("--n", type=int, default=16, help="fixed windows (batches of 4)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip (0: none)")
    ap.add_argument("--edge", type=float, default=1.0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--arch", default="qwen", choices=("qwen", "graph"), help="graph: the message-passing network (d 256, 8 layers)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=4, help="windows per batch")
    ap.add_argument("--episode", type=int, default=-1, help="every window from this one episode (default: the first 100)")
    ap.add_argument("--mix", type=int, default=1, help="1: the trainer's distance + squared position loss; 0: squared only")
    ap.add_argument("--sched", default="const", choices=("const", "cos"), help="cos: cosine decay to 0 after the warm-up")
    ap.add_argument("--aug", default="", choices=("", "both", "yaw", "shift"), help="re-collate every step with the "
                    "trainer's augmentation (a random yaw and shift per window; or one of the two); the windows stay the same")
    ap.add_argument("--side_apart", action="store_true", help="--loss full: the trainer's --side_apart (slot head off the "
                    "trunk, side heads clipped apart)")
    ap.add_argument("--hide_past", action="store_true", help="re-collate every step with the trainer's random past (some "
                    "given frames absent)")
    ap.add_argument("--own", action="store_true", help="cloth tokens in their own frames (cloth_train --own)")
    ap.add_argument("--own_up", action="store_true", help="--own + the world-z parts and a z force (cloth_train --own_up)")
    ap.add_argument("--direct", action="store_true", help="no exact layer: the network gives each vertex's whole acceleration")
    ap.add_argument("--wide", action="store_true", help="qwen at d 1024, 8 layers (192M) instead of d 512, 16 layers")
    a = ap.parse_args()
    from lsim.model.cloth_net import build_cloth_model
    from lsim.train.cloth_train import _windows, step_loss
    from lsim.train.cloth_windows import collate_cloth, load_shard
    dev = "cuda"
    torch.manual_seed(0)
    eps = [e for p in sorted(glob.glob(str(REPO / "data/datasets/c1/shard_*.npz")))[:2] for e in load_shard(p[:-4])][:100]
    if a.episode >= 0:
        eps = [eps[a.episode]]
        print(f"episode {a.episode}: {eps[0].wd['init']}, {[f['family'] for f in eps[0].wd['families']]}, "
              f"{len(eps[0].rest)} vertices", flush=True)
    wins = _windows(eps, a.n, 8, 8, np.random.default_rng(1))
    full = a.loss == "full"
    A = dict(arch="qwen", d=512, layers=16, heads=8, kv_heads=4, mlp=1536, prob_slots=full, unc=full, accel=True, layer=2,
             fp32=a.fp32)
    if a.arch == "graph":
        A.update(arch="graph", d=256, layers=8, no_global=False)
    if a.wide:
        A.update(d=1024, layers=8, mlp=3072)
    A["direct"] = a.direct
    A["own"] = a.own or a.own_up
    A["own_up"] = a.own_up
    A["side_apart"] = a.side_apart
    model = build_cloth_model(A).to(dev)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    batches = [collate_cloth(wins[i:i + a.bs], dev, np.random.default_rng(i), augment=False, all_past=True,
                             **({} if full else {"p_hidden": 0.0})) for i in range(0, a.n, a.bs)]
    cv = []                                                             # constant velocity on the same vertices
    for b in batches:                                                   # (clean_mean_mm's: free cloth, first step)
        x, fr = b["pos"], b["is_cloth"] & ~b["held"][:, 7]
        cv.append(1e3 * (2 * x[fr, 7] - x[fr, 6] - x[fr, 8]).norm(dim=-1).mean().item())
    print(f"constant velocity on these windows: {np.mean(cv):.4f} mm (by batch {[round(c, 3) for c in cv]})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * (
        0.5 * (1 + math.cos(math.pi * s / a.steps)) if a.sched == "cos" else 1.0))
    rng = np.random.default_rng(0)
    log = []
    for step in range(a.steps):
        b = batches[step % len(batches)]
        if a.aug or a.hide_past:                                        # the same windows, the trainer's view of them
            i0 = (step % len(batches)) * a.bs
            b = collate_cloth(wins[i0:i0 + a.bs], dev, np.random.default_rng(10_000 + step), augment=bool(a.aug),
                              aug_yaw=a.aug != "shift", aug_shift=a.aug != "yaw",
                              all_past=not a.hide_past, **({} if full else {"p_hidden": 0.0}))
        opt.zero_grad(set_to_none=True)
        loss, info = step_loss(model, b, rng, 8, dev, sig=0.0, pf=0, edge_w=a.edge, mix=bool(a.mix))
        loss.backward()
        if a.side_apart:
            side = lambda n: n.startswith(("unc.", "param."))
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if side(n)], 1.0)
            gn = torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if not side(n)], a.clip)
        else:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else 1e30)
        if torch.isfinite(gn):
            opt.step()
        sched.step()
        log.append({"step": step, "pos_mm": info["clean_mean_mm"], "shape_pct": info["strain_err_pct"], "grad": float(gn)})
        if step % 25 == 0 or step == a.steps - 1:
            m = lambda k: float(np.mean([r[k] for r in log[-len(batches):]]))
            print(f"step {step:5d}  pos mm {m('pos_mm'):.4f}  / constant velocity {m('pos_mm') / np.mean(cv):.3f}  shape % "
                  f"{m('shape_pct'):.4f}  grad norm {m('grad'):.3g}  loss {loss.item():.4g}", flush=True)
    json.dump({"args": vars(a), "cv_mm": cv, "log": log}, open(REPO / f"docs/results/l2_overfit_batch_{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
