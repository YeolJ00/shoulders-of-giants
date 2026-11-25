"""L2 -- the model and its baselines on PGND's robot cloth windows, scored on what the cameras saw (lsim.real.pgnd).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_v3/last.pt --tag v3 [--split test]

Per converted window (scripts/l1_real_cloth_pgnd.py): given frames 0-7 (the grid fitted to the observed first cloud,
then tracked through PGND's carried points), every prediction runs alone to frame 120 with both held patches moved
by the robot's recorded path, and is scored by the distance from the observed cloud at frames 60 and 120 (the next
windows' first clouds: 2 s and 4 s on) to the predicted surface (median over the cloud's points; one-sided -- a fold's
hidden layer is not penalised). Predictions: the model (all four material slots hidden); the exact layer alone
(gravity, the table, the held patches; no cloth forces); the cloth left where it was at frame 7 (only the held
patches move). Beside the score, its blind side: the median distance from the
predicted vertices back to the cloud (back<f>), the median edge strain (%) and the area over the rest area (a cloth
blown up to cover the cloud scores well one way only: a fine-tune on the one-sided loss reached 17,500x its area).
Reference: PGND's own carried points at frame 60 (their velocity field integrated from the first
cloud; the targets PGND trains on), scored the same way against the cloud (to the nearest point). The box set is
left out (its box is not modelled).

Writes docs/results/l2_real_cloth_pgnd_<tag>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
T0 = 7


class Zero(torch.nn.Module):
    """No cloth forces: the exact layer alone."""
    layer = 2

    def forward(self, b):
        N, Fw, _ = b["pos"].shape
        z = torch.zeros(N, Fw, 3, device=b["pos"].device)
        return z, z, torch.zeros(int(b["n_bodies"]), 4, device=b["pos"].device), None


def load_model(path, dev):
    from lsim.model.cloth_net import build_cloth_model
    ck = torch.load(path, map_location=dev)
    A = ck["args"]
    m = build_cloth_model(A).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="baselines")
    ap.add_argument("--split", default="test", choices=("test", "train", "all"))
    ap.add_argument("--every", type=int, default=1, help="every n-th window")
    a = ap.parse_args()
    from scipy.spatial import cKDTree
    from lsim.real.pgnd import load_pgnd
    from lsim.train.cloth_train import rollout
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {"layer": Zero()}
    if a.ckpt:
        models = {"model": load_model(REPO / a.ckpt, dev), **models}
    eps = load_pgnd(None if a.split == "all" else a.split)[:: a.every]
    out = {}
    for ep in eps:
        T = ep.n_frames
        frames = [f for f in ep.cloud_frames if f > T0]
        preds = {}
        for k, mdl in models.items():
            with torch.no_grad():
                preds[k] = rollout(mdl, ep, T0, T - 1 - T0, dev=dev, hidden=("density", "stretch", "bend", "mu"))[0]
        still = np.broadcast_to(ep.x([T0])[0], (T - 1 - T0,) + ep._x.shape[1:]).copy()
        hv = ep.held[0]
        still[:, hv] = ep.x(np.arange(T0 + 1, T))[:, hv]
        preds["still"] = still
        row = {}
        E = ep.edges
        L0 = np.linalg.norm(ep.rest[E[:, 0]] - ep.rest[E[:, 1]], axis=1)
        tr = ep.tris
        area = lambda X: 0.5 * np.linalg.norm(np.cross(X[tr[:, 1]] - X[tr[:, 0]], X[tr[:, 2]] - X[tr[:, 0]]), axis=1).sum()
        A0 = area(ep.rest)
        for k, P in preds.items():
            row[k] = {str(f): 1e3 * ep.cloud_error(P[f - T0 - 1], f) for f in frames}
            for f in frames:                                            # the score's blind side: a cloth blown up to
                X = P[f - T0 - 1]                                       # cover the cloud scores well one way only
                row[k][f"back{f}"] = float(1e3 * np.median(cKDTree(ep.clouds[f]).query(X)[0]))
                row[k][f"strain{f}"] = float(100 * np.median(np.abs(np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) / L0 - 1)))
                row[k][f"area{f}"] = float(area(X) / A0)
            row[k]["finite"] = bool(np.isfinite(P).all() and np.abs(P).max() < 10.0)
        if 60 in ep.clouds:
            row["carried"] = {"60": float(1e3 * np.median(cKDTree(ep.carried[60]).query(ep.clouds[60])[0]))}
        row["start"] = {"0": 1e3 * ep.cloud_error(ep._x[0], 0)}
        row["meta"] = {"split": ep.meta["split"], "recording": ep.meta["case"].rsplit("_p", 1)[0]}
        out[ep.meta["case"]] = row
        print(f"{ep.meta['case']:18s} " + " | ".join(f"{k} " + " ".join(f"{f}:{v:6.1f}" for f, v in r.items() if f.isdigit())
                                                  for k, r in row.items() if k != "meta"), flush=True)
    keys = [k for k in next(iter(out.values())) if k not in ("meta", "start")] if out else []
    summary = {}
    for k in keys:
        for f in ("60", "120", "back60", "back120", "strain60", "strain120", "area60", "area120"):
            v = [r[k][f] for r in out.values() if f in r.get(k, {})]
            if v:
                summary[f"{k}@{f}"] = {"mean_of_medians_mm": float(np.mean(v)), "median_mm": float(np.median(v)), "n": len(v)}
    print(json.dumps(summary, indent=1))
    (REPO / f"docs/results/l2_real_cloth_pgnd_{a.tag}.json").write_text(json.dumps({"ckpt": a.ckpt, "split": a.split, "t0": T0,
                                                                                    "summary": summary, "windows": out}, indent=1))


if __name__ == "__main__":
    main()
