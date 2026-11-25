"""How far the teacher's instantaneous cloth velocity is from what the 30 fps positions show (2026-11-19).

On record_v shards (l1_generate_cloth --record_v: the teacher's instantaneous velocities at each recorded frame), per
moving unheld vertex-frame (central-difference speed > 5 cm/s): |v_t - (x_{t+1} - x_{t-1}) 15| over the speed, and
per mesh edge (both ends unheld) the instantaneous relative velocity against the central-difference one -- a
vertex-scale vibration inside the frame shows as a relative velocity far above what the positions move. By Young's
modulus tertile. Model-free, CPU.

    CUDA_VISIBLE_DEVICES="" $L scripts/l1_velocity_alias.py --data c1v_test
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="c1v_test")
    a = ap.parse_args()
    from lsim.train.cloth_windows import load_shard
    eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz"))) for e in load_shard(p[:-4])]
    rows = []
    for e in eps:
        X = e.x(np.arange(e.n_frames)).astype(np.float64)
        V = e.cv.astype(np.float64)[1:-1]
        cen = (X[2:] - X[:-2]) * 15.0
        free = ~e.held[1:-1] & ~e.held[:-2]
        sp = np.linalg.norm(cen, axis=-1)
        m = free & (sp > 0.05)
        E = e.edges
        fe = free[:, E[:, 0]] & free[:, E[:, 1]] & (m[:, E[:, 0]] | m[:, E[:, 1]])
        rv = np.linalg.norm(V[:, E[:, 1]] - V[:, E[:, 0]], axis=-1)[fe]
        rc = np.linalg.norm(cen[:, E[:, 1]] - cen[:, E[:, 0]], axis=-1)[fe]
        if not m.any():
            continue
        rows.append(dict(Y=float(e.c["Y"]), n=int(m.sum()), rel=float(np.median(np.linalg.norm(V - cen, axis=-1)[m] / sp[m])),
                         speed=float(np.median(sp[m])), edge_true=float(np.median(rv)), edge_cen=float(np.median(rc))))
    Y = np.array([r["Y"] for r in rows])
    cuts = np.quantile(Y, [1 / 3, 2 / 3])
    out = {"episodes": len(rows)}
    for name, sel in (("all", np.ones(len(rows), bool)), ("soft", Y <= cuts[0]), ("mid", (Y > cuts[0]) & (Y <= cuts[1])),
                      ("stiff", Y > cuts[1])):
        R = [r for r, s in zip(rows, sel) if s]
        out[name] = {k: float(np.median([r[k] for r in R])) for k in ("rel", "speed", "edge_true", "edge_cen", "Y")}
        o = out[name]
        print(f"{name:5s} ({len(R)} episodes, Y ~{o['Y']:.3g}): |v - central| / speed {o['rel']:.2f} (speed {o['speed']:.3f} m/s); "
              f"edge-relative velocity: instantaneous {o['edge_true']:.4f}, from positions {o['edge_cen']:.4f} m/s "
              f"({o['edge_true'] / max(o['edge_cen'], 1e-9):.1f}x)")
    json.dump(out, open(REPO / f"docs/results/l1_velocity_alias_{a.data}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
