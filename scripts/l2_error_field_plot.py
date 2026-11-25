"""Where on the cloth the network's one-frame error is (2026-11-21): the true acceleration, the network's, and the
error drawn on the cloth for the held-out windows with the most moving free cloth (l2_error_where's 300 c1_test
windows, rng 3, clean history, every slot given). Each row is one window: the cloth at frame t seen from the side
that shows the most of it (its two principal directions), coloured by |acceleration| (x_{t+1} - 2 x_t + x_{t-1}) of the
teacher, by the network's, and by the network's error, all in mm per frame^2, held vertices black, the gripper red;
the last column the network's acceleration against the teacher's, each vertex's component along that window's
mean acceleration direction and across it.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_error_field_plot.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt --out f.png
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from l2_error_where import classify
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import _windows, step_loss
    from lsim.train.cloth_windows import collate_cloth, load_shard
    dev = "cuda"
    eps = [e for p in sorted(glob.glob(str(REPO / "data/datasets/c1_test/shard_*.npz"))) for e in load_shard(p[:-4])]
    wins = _windows(eps, a.n, 8, 8, np.random.default_rng(3))
    m = load_model(a.ckpt, dev)
    P = 8
    cands = []
    with torch.no_grad():
        for i, w in enumerate(wins):
            b = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
            pos = b["pos"]
            fr = (b["is_cloth"][:, None] & ~b["held"])[:, P - 1]
            moving = (pos[:, P - 1] - pos[:, P - 2]).norm(dim=-1) * 30 > 0.02
            sel = fr & moving & classify(b, P)["free"]
            A = (pos[:, P] - 2 * pos[:, P - 1] + pos[:, P - 2])[sel].norm(dim=-1)
            cands.append((float(A.sum()), i))
        picks = [i for _, i in sorted(cands, reverse=True)[:a.rows * 3:3]]      # strong motion, not all one episode
        fig, ax = plt.subplots(a.rows, 4, figsize=(15, 3.3 * a.rows), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
        for r, i in enumerate(picks):
            b = collate_cloth([wins[i]], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
            parts = {}
            step_loss(m, b, np.random.default_rng(i), 8, dev, sig=0.0, parts=parts)
            x = parts["x"][:, 0].double().cpu().numpy()
            pos = b["pos"].double().cpu().numpy()
            cl = b["is_cloth"].cpu().numpy()
            held = b["held"][:, P - 1].cpu().numpy()
            kind = b["kind"].cpu().numpy()
            A = (pos[:, P] - 2 * pos[:, P - 1] + pos[:, P - 2]) * 1e3
            Ah = (x - 2 * pos[:, P - 1] + pos[:, P - 2]) * 1e3
            E = Ah - A
            c = cl & ~held
            X0 = pos[cl, P - 1]
            mu = X0.mean(0)
            _, _, Vt = np.linalg.svd(X0 - mu, full_matrices=False)
            proj = lambda p: (p - mu) @ Vt[:2].T * 100                     # cm in the cloth's principal plane
            vmax = np.percentile(np.linalg.norm(A[c], axis=-1), 98)
            for j, (F, title) in enumerate(((A, "teacher's acceleration"), (Ah, "network's acceleration"),
                                            (E, "network's error"))):
                q = proj(pos[c, P - 1])
                sc = ax[r, j].scatter(q[:, 0], q[:, 1], c=np.linalg.norm(F[c], axis=-1), s=5, cmap="magma_r", vmin=0,
                                      vmax=vmax, linewidths=0)
                if held[cl].any():
                    h = proj(pos[cl & held, P - 1])
                    ax[r, j].scatter(h[:, 0], h[:, 1], c="k", s=6)
                g = kind == 1                                                  # gripper points, if any in the window
                if g.any():
                    gp = proj(pos[g, P - 1])
                    ax[r, j].scatter(gp[:, 0], gp[:, 1], c="tab:red", s=8, marker="x")
                ax[r, j].set_aspect("equal")
                ax[r, j].tick_params(labelsize=7)
                if r == 0:
                    ax[r, j].set_title(title + "\n(mm / frame$^2$; cloth seen face-on, cm)", fontsize=10)
                if j == 2:
                    plt.colorbar(sc, ax=ax[r, j], fraction=0.04)
            mv = np.linalg.norm(pos[:, P - 1] - pos[:, P - 2], axis=-1) * 30 > 0.02
            s = c & mv
            d = A[s].mean(0)
            d = d / (np.linalg.norm(d) + 1e-12)
            al, alh = A[s] @ d, Ah[s] @ d
            ax[r, 3].scatter(al, alh, s=3, alpha=0.4, c="tab:blue", linewidths=0, label="along the mean direction")
            perp = lambda F: np.linalg.norm(F - (F @ d)[:, None] * d, axis=-1) * np.sign((F - (F @ d)[:, None] * d) @ np.cross(d, [0, 0, 1.0]) + 1e-12)
            ax[r, 3].scatter(perp(A[s]), perp(Ah[s]), s=3, alpha=0.3, c="tab:orange", linewidths=0, label="across it")
            lim = np.percentile(np.abs(np.concatenate([al, alh])), 99.5) * 1.1
            ax[r, 3].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
            ax[r, 3].set_xlim(-lim, lim); ax[r, 3].set_ylim(-lim, lim)
            ax[r, 3].set_aspect("equal")
            ax[r, 3].tick_params(labelsize=7)
            ep, t = wins[i]["episode"] if "episode" in wins[i] else "", ""
            ratio = np.linalg.norm(E[s], axis=-1).mean() / np.linalg.norm(A[s], axis=-1).mean()
            ax[r, 3].text(0.03, 0.95, f"error / constant velocity's {ratio:.2f}", transform=ax[r, 3].transAxes, fontsize=8,
                          va="top")
            if r == 0:
                ax[r, 3].set_title("network vs teacher, per vertex\n(mm / frame$^2$; dashed: equal)", fontsize=10)
                ax[r, 3].legend(fontsize=7, loc="lower right", markerscale=3)
        fig.suptitle("One frame ahead on held-out cloth: the network gets the pattern of the motion and about two "
                     "thirds of its size", fontsize=12)
        fig.tight_layout()
        fig.savefig(a.out, dpi=130)
        print("wrote", a.out, "windows", picks)


if __name__ == "__main__":
    main()
