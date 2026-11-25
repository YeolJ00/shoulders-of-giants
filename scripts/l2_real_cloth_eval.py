"""L2 -- the cloth model on real cloth (converted by scripts/l1_real_cloth.py; lsim.real.cloth).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_real_cloth_eval.py --ckpt outputs/lsim/cloth_v3/last.pt --tag v3

From starts every --every frames inside each recording's trusted frames (its fit's median residual under 6 mm),
with the 8 fitted frames before, the model rolls the cloth out for --frames frames; the held patch follows the
recording, the material is hidden (all four slots: nobody measured it), the grasp force is empty. Beside it, the
exact layer alone, constant velocity, and the fitted mesh itself (the floor: how far the reconstruction already is
from what the cameras saw). Two errors per frame ahead:

  track   median distance of the visible tracked points that have moved over 1 cm since the start (the moving
          cloth: over every point the cloth lying still on the table dominates) to their embedded positions on
          the predicted mesh: against what the cameras saw (the fitted mesh's own value is the floor; the
          median, as drifted tracks would dominate a mean)
  mesh    mean distance of the free vertices to the fitted mesh's (the synthetic evaluations' error)

Writes docs/results/l2_real_cloth_eval_<tag>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
H = (1, 5, 10, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--source", default="phystwin")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from lsim.model.cloth_net import build_cloth_model
    from lsim.real.cloth import load_real_cloth
    from lsim.train.cloth_train import rollout
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = load_real_cloth(a.source)
    hidden = ("density", "stretch", "bend", "mu")

    class Zero(torch.nn.Module):
        layer = 2

        def forward(self, b):
            N, Fw, _ = b["pos"].shape
            z = torch.zeros(N, Fw, 3, device=b["pos"].device)
            return z, z, torch.zeros(int(b["n_bodies"]), 4, device=b["pos"].device), None
    models = {"default": Zero()}
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev)
        A = ck["args"]
        m = build_cloth_model(A).to(dev).eval()
        m.load_state_dict(ck["model"])
        models = {"model": m, **models}
        models["default"].layer = max(2, m.layer)
    names = list(models) + ["constvel", "fit"]
    rows = []
    for e in eps:
        ok = np.flatnonzero(e.frame_ok)
        last = int(ok[np.argmax(np.diff(np.r_[ok, 10 ** 9]) > 1)])     # the end of the first run of trusted frames
        T = min(a.frames, last - 8)
        if T < 10:
            continue
        for t0 in range(8, last - T + 1, a.every):
            fr = np.arange(t0 + 1, t0 + T + 1)
            true = e.x(np.arange(t0, t0 + T + 1))
            free = ~e.held[t0:t0 + T]
            preds = {}
            for k, mdl in models.items():
                with torch.no_grad():
                    preds[k] = rollout(mdl, e, t0, T, dev=dev, hidden=hidden)[0]
            preds["constvel"] = true[0][None] + np.arange(1, T + 1)[:, None, None] * (true[0] - e.x([t0 - 1])[0])[None]
            preds["fit"] = true[1:]
            row = {"episode": e.name, "t0": t0, "T": T, "moved_mm": float(1e3 * np.linalg.norm(true[-1] - true[0], axis=-1).mean())}
            for k in names:
                tr = e.track_error(preds[k], fr, moved_from=t0)
                d = np.linalg.norm(preds[k] - true[1:], axis=-1)
                row[k] = {"track": [float(v) for v in tr],
                          "mesh": [float(d[j][free[j]].mean()) if free[j].any() else 0.0 for j in range(T)]}
            rows.append(row)
            print(f"{e.name:30s} t0 {t0:3d} moved {row['moved_mm']:6.1f} mm | " + " | ".join(
                f"{k} {1e3 * row[k]['track'][min(9, T - 1)]:6.1f}" for k in names) + "  (track error at 10 frames, mm)", flush=True)
    res = {"ckpt": a.ckpt, "source": a.source, "starts": len(rows), "rows": rows, "summary": {}}
    print(f"\n{len(rows)} starts on {len({r['episode'] for r in rows})} recordings; median (p90) error, mm:")
    for metric in ("track", "mesh"):
        for k in names:
            cells = {}
            for h in H:
                v = [r[k][metric][h - 1] for r in rows if r["T"] >= h]
                if v:
                    cells[f"f{h}"] = [float(1e3 * np.median(v)), float(1e3 * np.percentile(v, 90)), len(v)]
            res["summary"][f"{metric}/{k}"] = cells
            print(f"  {metric:5s} {k:9s} " + "  ".join(f"f{h} {c[0]:6.1f} ({c[1]:5.1f})" for h, c in zip(H, cells.values())))
    (REPO / f"docs/results/l2_real_cloth_eval_{a.tag or 'default'}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
