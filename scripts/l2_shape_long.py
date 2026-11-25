"""Whether a rolled-out body keeps its shape, per material, over whole held-out episodes (2026-11-03: the invariant every
learned model failed): given frames 0-7, then alone to the end with the material hidden, the gripper on its path.

  cloth  mesh area over the teacher's at the same frame, mean |edge strain|
  rope   length (sum of segment lengths) over its rest length, largest segment strain
  soft   volume over the teacher's, share of tets inverted, smallest tet volume ratio, mean |edge strain|

beside the mean free-vertex distance from the teacher at 1 s and at the end. Per episode and the median.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_shape_long.py --ckpt outputs/lsim/cloth_m64/step23999.pt --tag m64
    LSIM_RESTORE=1 ... --tag m64_P            # the restoration at inference
Writes docs/results/l2_shape_long_<tag>.json.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
HIDE = ("density", "stretch", "bend", "mu")


def tet_vol(X, T):
    a = X[:, T[:, 0]]
    return np.einsum("fti,fti->ft", np.cross(X[:, T[:, 1]] - a, X[:, T[:, 2]] - a), X[:, T[:, 3]] - a) / 6.0


def shape(kind, X, Y, ep):
    """Shape numbers of predicted positions X and the teacher's Y (frames, V, 3) at the last frame."""
    E = ep.edges
    L0 = np.linalg.norm(ep.rest[E[:, 0]] - ep.rest[E[:, 1]], axis=-1)
    s = np.linalg.norm(X[-1, E[:, 0]] - X[-1, E[:, 1]], axis=-1) / L0 - 1
    out = {"strain_mean_abs": float(np.abs(s).mean()), "strain_max": float(np.abs(s).max())}
    if kind == "cloth":
        t = ep.tris
        ar = lambda Z: 0.5 * np.linalg.norm(np.cross(Z[t[:, 1]] - Z[t[:, 0]], Z[t[:, 2]] - Z[t[:, 0]]), axis=1).sum()
        out["area_ratio"] = float(ar(X[-1]) / ar(Y[-1]))
    elif kind == "rope":
        out["length_ratio"] = float(np.linalg.norm(X[-1, E[:, 0]] - X[-1, E[:, 1]], axis=-1).sum() / L0.sum())
        out["length_ratio_teacher"] = float(np.linalg.norm(Y[-1, E[:, 0]] - Y[-1, E[:, 1]], axis=-1).sum() / L0.sum())
    else:
        T = ep.tets
        V0 = tet_vol(ep.rest[None].astype(np.float64), T)[0]
        V = tet_vol(X[-1:].astype(np.float64), T)[0]
        Vt = tet_vol(Y[-1:].astype(np.float64), T)[0]
        out["volume_ratio"] = float(V.sum() / Vt.sum())
        out["inverted_frac"] = float((V * np.sign(V0) <= 0).mean())
        out["tet_ratio_min"] = float((V / V0).min())
        # the worst over the rollout too: inverted share at its largest
        Va = tet_vol(X.astype(np.float64), T)
        out["inverted_frac_max"] = float((Va * np.sign(V0)[None] <= 0).mean(1).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="r1_test,s1_test,c1_test")
    ap.add_argument("--n", type=int, default=24, help="episodes per dataset (seeded)")
    a = ap.parse_args()
    from lsim.model.cloth_net import build_cloth_model
    from lsim.train.cloth_train import rollout
    from lsim.train.cloth_windows import load_shard
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = build_cloth_model(ck["args"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    print(f"{a.ckpt}: restoration {'on' if m.restore is not None else 'off'}", flush=True)
    res = {"ckpt": a.ckpt, "restore": m.restore, "datasets": {}}
    for d in a.data.split(","):
        eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{d}/shard_*.npz")))[:2] for e in load_shard(p[:-4])]
        kind = getattr(eps[0], "material", "cloth")
        rng = np.random.default_rng(0)
        rows = []
        for k in rng.choice(len(eps), min(a.n, len(eps)), replace=False):
            e = eps[int(k)]
            n = e.n_frames - 8
            with torch.no_grad():
                X, Y, _ = rollout(m, e, 7, n, dev=dev, hidden=HIDE)
            free = ~e.held[7:7 + n]
            dist = np.linalg.norm(X - Y, axis=-1)
            err = [float(dist[j][free[j]].mean()) if free[j].any() else 0.0 for j in range(n)]
            r = {"episode": int(k), "err_1s_mm": 1e3 * err[min(29, n - 1)], "err_end_mm": 1e3 * err[-1], **shape(kind, X, Y, e)}
            rows.append(r)
            print(f"[{kind}] ep {int(k):3d}  1 s {r['err_1s_mm']:7.1f}  end {r['err_end_mm']:7.1f} mm  " +
                  "  ".join(f"{key} {v:.3f}" for key, v in r.items() if key not in ("episode", "err_1s_mm", "err_end_mm")), flush=True)
        med = {key: float(np.median([r[key] for r in rows])) for key in rows[0] if key != "episode"}
        res["datasets"][d] = {"kind": kind, "median": med, "episodes": rows}
        print(f"== {d} ({kind}, {len(rows)} episodes) median: " + "  ".join(f"{key} {v:.3f}" for key, v in med.items()), flush=True)
    out = REPO / f"docs/results/l2_shape_long_{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
