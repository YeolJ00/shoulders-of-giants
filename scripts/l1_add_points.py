"""L1 -- add surface-point learner geometry (lsim.data.assets.surface_points) to processed assets
in place, without redoing remeshing or convex decomposition.

    $L scripts/l1_add_points.py [--workers 16]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data/assets/proc/gso"
SRC = REPO / "data/assets/gso"


def work(f):
    from lsim.data.assets import load_mesh, shell_props, surface_points
    d = dict(np.load(f))
    if "points_v" in d:
        return f.stem, len(d["points_v"]), 0.0
    raw = load_mesh(SRC / f.stem / "meshes" / "model.obj")
    com = raw.center_mass if raw.is_watertight else shell_props(raw)[0]   # the frame process() used
    raw.apply_translation(-com)
    pts, nrm, e, area = surface_points(raw)
    gap = float(pts[:, 2].min() - raw.vertices[:, 2].min())
    d.update(points_v=pts, points_n=nrm, points_edges=e, points_area=area)
    np.savez_compressed(f, **d)
    return f.stem, len(pts), gap


def main():
    files = sorted(PROC.glob("*.npz"))
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    with Pool(workers) as pool:
        out = pool.map(work, files)
    n = np.array([o[1] for o in out]); g = np.array([o[2] for o in out])
    print(f"{len(out)} assets: points median {np.median(n):.0f}, max {n.max()}; bottom gap median {1000*np.median(g):.2f} mm, "
          f"p95 {1000*np.percentile(g, 95):.2f} mm, max {1000*g.max():.1f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
