"""L2 -- PGND's held-out robot cloth windows drawn: each prediction over what the cameras saw, 2 s and 4 s on.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_render_real_cloth_pgnd.py --model "64M=outputs/lsim/cloth_q64/step23999.pt" \
        --model "64M + real=outputs/lsim/cloth_rq64/step1000.pt" --tag q64 [--n 8]

PGND released no camera images, only the fused depth clouds, so the cameras' view is that cloud (grey points, observed
at frames 0, 60 and 120). Columns: the start (frame 0: the cloud and the grid fitted to it), then each prediction at
2 s (frame 60) and 4 s (frame 120) as a shaded mesh over that frame's cloud -- the teacher with its material fitted to
the cloth's training recordings (scripts/l2_real_cloth_pgnd_teacher.py), the exact layer alone (no cloth forces) and
each model (material slots hidden), all from frame 7 as scripts/l2_real_cloth_pgnd_eval.py scores them; the grippers
are the black points. Each panel's corner: the median cloud-to-surface distance (the eval's score). Windows: the test
windows with both clouds, spread over how far the cloth moved.

Writes outputs/lsim/render_pgnd/<tag>/<case>.png and index.json (cases, scores).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
T0 = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="label=checkpoint (repeatable)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import cKDTree
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from l2_real_cloth_pgnd_eval import Zero, load_model
    from lsim.real import replay
    from lsim.real.pgnd import load_pgnd
    from lsim.train.cloth_train import rollout
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mats = {c: json.loads((REPO / f"docs/results/l2_real_cloth_pgnd_teacher_{c}.json").read_text())["material"]
            for c in ("cloth6", "cali2")}
    models = {"no cloth forces": Zero()}
    for s in a.model:
        k, p = s.split("=", 1)
        models[k] = load_model(REPO / p, dev)
    eps = [e for e in load_pgnd("test") if 60 in e.clouds and 120 in e.clouds]
    mot = np.array([np.median(cKDTree(e.clouds[0]).query(e.clouds[120])[0]) for e in eps])
    pick = np.argsort(mot)[np.linspace(0, len(eps) - 1, min(a.n, len(eps))).astype(int)]
    out = REPO / f"outputs/lsim/render_pgnd/{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for i in pick[::-1]:                                                # most motion first
        ep = eps[i]
        T = ep.n_frames
        preds = {"teacher (fitted)": replay.simulate(ep, [mats[ep.meta["case"].split("_")[0]]], T0, T - 1 - T0)[0][1:]}
        for k, m in models.items():
            with torch.no_grad():
                preds[k] = rollout(m, ep, T0, T - 1 - T0, dev=dev, hidden=("density", "stretch", "bend", "mu"))[0]
        cols = ["start"] + list(preds)
        allp = np.concatenate([ep.clouds[f] for f in (0, 60, 120)])
        lo, hi = np.percentile(allp, 1, 0) - 0.03, np.percentile(allp, 99, 0) + 0.03
        c, r = 0.5 * (lo + hi), 0.5 * (hi - lo).max()
        fig = plt.figure(figsize=(3.2 * len(cols), 6.4))
        scores = {}
        for row, f in enumerate((60, 120)):
            for j, k in enumerate(cols):
                ax = fig.add_subplot(2, len(cols), row * len(cols) + j + 1, projection="3d")
                fr = 0 if k == "start" else f
                X = ep._x[0] if k == "start" else preds[k][f - T0 - 1]
                cl = ep.clouds[fr]
                cl = cl[np.random.default_rng(0).choice(len(cl), min(2500, len(cl)), replace=False)]
                ax.scatter(*cl.T, s=0.6, c="0.55", alpha=0.5, depthshade=False)
                ok = np.isfinite(X).all() and np.abs(X - c).max() < 5 * r
                if ok:
                    ax.add_collection3d(Poly3DCollection(X[ep.tris], facecolors="tab:orange" if k != "teacher (fitted)" else "tab:blue",
                                                         edgecolors=(0, 0, 0, 0), alpha=0.55, shade=True))
                H = ep.hand[fr]
                ax.scatter(*H[::4].T, s=1.5, c="k", depthshade=False)
                e = 1e3 * ep.cloud_error(X, fr) if ok else float("nan")
                if k != "start":
                    scores.setdefault(k, {})[str(f)] = e
                ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
                ax.view_init(elev=35, azim=-60)
                ax.set_axis_off()
                if row == 0 or k == "start":
                    ax.set_title(k if k != "start" else "start (frame 0)", fontsize=10)
                ax.text2D(0.03, 0.03, ("2 s" if f == 60 else "4 s") + (f"  {e:.0f} mm" if k != "start" else ""),
                          transform=ax.transAxes, fontsize=9)
        fig.suptitle(f"{ep.meta['case']} -- grey: what the cameras saw; mesh: the prediction; black: grippers", fontsize=10)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0, hspace=0.05)
        fig.savefig(out / f"{ep.meta['case']}.png", dpi=110)
        plt.close(fig)
        index.append({"case": ep.meta["case"], "motion_mm": 1e3 * float(mot[i]), "scores": scores})
        print(ep.meta["case"], f"motion {1e3 * mot[i]:.0f} mm", {k: {f: round(v, 1) for f, v in s.items()} for k, s in scores.items()}, flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
