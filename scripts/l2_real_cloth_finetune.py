"""L2 -- claim 3: fine-tune the cloth model on PGND's real training recordings, scored by what the cameras saw.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_real_cloth_finetune.py --init outputs/lsim/cloth_v3/last.pt \
        --out outputs/lsim/cloth_r1 [--steps 1000 --lr 1e-4 --trunc 10 --teacher_w 1]

Every step: one real training window (lsim.real.pgnd, the box set left out; the test recordings never seen), the model
rolled out alone from frame 7 to frame 120 with all material slots hidden (as scripts/l2_real_cloth_pgnd_eval.py),
the loss the mean distance (cm) from the observed clouds at frames 60 and 120 to the predicted surface
(lsim.train.cloth_real.cloud_loss) plus back_w times the mean distance from the predicted vertices back to the cloud
(mesh_to_cloud_loss: without it the one-sided loss was gamed, the cloth blown up to 17,500x its area), gradients through the `trunc` frames before each (the rollout's gradients explode
beyond ~10 frames). With teacher_w > 0, plus that weight times the teacher loss on a batch of c1 windows with the
init checkpoint's own recipe (lsim.train.cloth_train.step_loss), so the model is not free to forget the teacher.
Checkpoints every 250 steps (stepN.pt) and last.pt, loadable by l2_real_cloth_pgnd_eval.py.
"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
T0, T_END = 7, 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--trunc", type=int, default=10)
    ap.add_argument("--teacher_w", type=float, default=1.0)
    ap.add_argument("--back_w", type=float, default=0.5, help="weight of the mesh-to-cloud distance (0: the gamed one-sided loss)")
    ap.add_argument("--strain_w", type=float, default=1.0, help="weight of the rollout's mean |edge strain| (in %%)")
    ap.add_argument("--back_clip", type=float, default=0.0, help="clip (m) of the mesh-to-cloud distance (0: none; 0.05 let r2/r3 blow up)")
    ap.add_argument("--n_real", type=int, default=0, help="use a random subset of this many real training windows (claim 3's curve; 0: all)")
    ap.add_argument("--motion_pow", type=float, default=0.0,
                    help="draw windows with probability ~ motion^pow (motion: median distance of the frame-120 cloud from the frame-0 one; "
                         "r5's gap to the teacher is 8.5 mm on the more moving half of the windows, 2.3 on the rest)")
    ap.add_argument("--data", default="data/datasets/c1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.real.pgnd import load_pgnd
    from lsim.train.cloth_real import cloud_loss, mesh_to_cloud_loss, rollout_grad, strain_loss
    from lsim.train.cloth_train import _load, _windows, collate_cloth, step_loss
    dev = "cuda"
    ck = torch.load(REPO / a.init, map_location=dev)
    A = ck["args"]
    model = load_model(REPO / a.init, dev).train()
    real = [e for e in load_pgnd("train") if 60 in e.clouds and 120 in e.clouds]
    if a.n_real:
        real = [real[i] for i in sorted(np.random.default_rng(1000 + a.n_real).choice(len(real), a.n_real, replace=False))]
    print(f"{len(real)} real training windows ({sorted({e.meta['case'].rsplit('_p', 1)[0] for e in real})})", flush=True)
    p_draw = None
    if a.motion_pow:
        from scipy.spatial import cKDTree
        mot = np.array([np.median(cKDTree(e.clouds[0]).query(e.clouds[120])[0]) for e in real])
        p_draw = mot ** a.motion_pow / (mot ** a.motion_pow).sum()
        print(f"motion {1e3 * np.percentile(mot, [10, 50, 90]).round(4)} mm; drawn with p ~ motion^{a.motion_pow}", flush=True)
    teacher = _load(sorted(glob.glob(f"{REPO / a.data}/shard_*.npz")), set()) if a.teacher_w > 0 else []
    if teacher:
        print(f"{len(teacher)} teacher episodes", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    args = dict(A, init=a.init, real_steps=a.steps, real_lr=a.lr, trunc=a.trunc, teacher_w=a.teacher_w, back_w=a.back_w, strain_w=a.strain_w, back_clip=a.back_clip, n_real=a.n_real, motion_pow=a.motion_pow)
    targets = [60 - T0 - 1, T_END - T0 - 1]
    t0, hist = time.time(), []
    for step in range(a.steps):
        ep = real[rng.choice(len(real), p=p_draw)]
        opt.zero_grad(set_to_none=True)
        X = rollout_grad(model, ep, T0, T_END - T0, dev=dev, trunc=a.trunc, targets=targets)
        l60, d60 = cloud_loss(X[targets[0]], ep.clouds[60], ep.tris)
        l120, d120 = cloud_loss(X[targets[1]], ep.clouds[120], ep.tris)
        bc = a.back_clip or None
        b60, e60 = mesh_to_cloud_loss(X[targets[0]], ep.clouds[60], bc)
        b120, e120 = mesh_to_cloud_loss(X[targets[1]], ep.clouds[120], bc)
        l_real = (l60 + l120 + a.back_w * (b60 + b120)) / 2 / 1e-2
        l_str = strain_loss(X, ep.rest, ep.tris)
        loss = l_real + a.strain_w * l_str / 1e-2
        info = {}
        if teacher:
            wins = _windows(teacher, A.get("batch", 4), 8, 8, rng)
            batch = collate_cloth(wins, dev, rng)
            l_t, info = step_loss(model, batch, rng, 8, dev, sig=A.get("sig", 1e-3), pf=A.get("pf", 0), edge_w=A.get("edge", 0.0),
                                  noise=A.get("noise", "vertex"), sig_rot=A.get("sig_rot", 3e-3), sig_white=A.get("sig_white", 2e-5))
            loss = loss + a.teacher_w * l_t
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append((1e3 * float(d60.median()), 1e3 * float(d120.median()), 1e3 * float(e60.median()), 1e3 * float(e120.median())))
        if step % 10 == 0:
            h = np.array(hist[-50:])
            print(f"step {step:5d}  {ep.meta['case']:18s} real median 2 s {hist[-1][0]:5.1f} 4 s {hist[-1][1]:5.1f} mm  "
                  f"(last 50: {h[:, 0].mean():5.1f} / {h[:, 1].mean():5.1f}; back {h[:, 2].mean():5.1f} / {h[:, 3].mean():5.1f})  l_real {l_real.item():.3f}  strain {100 * l_str.item():.2f}%  "
                  + "  ".join(f"{k} {v:.3g}" for k, v in info.items() if k in ("pos_rmse_mm", "clean_mean_mm", "strain_err_pct"))
                  + f"  grad {gn.item():.2g}  lr {sched.get_last_lr()[0]:.2g}  {time.time() - t0:.0f} s", flush=True)
        if (step + 1) % 250 == 0 or step == a.steps - 1:
            state = {"model": model.state_dict(), "args": args, "step": step}
            torch.save(state, out / f"step{step + 1}.pt")
            torch.save(state, out / "last.pt")


if __name__ == "__main__":
    main()
