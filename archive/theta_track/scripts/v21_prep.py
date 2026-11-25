"""Precompute everything the 2.1 scorer needs, so it never imports our `src`.

Meta's vjepa2 repo has its own top-level `src` package, which collides with ours. Rather
than juggling sys.modules, the 2.1 scorer runs standalone and gets its inputs from here:
clip paths, pair metadata, and the object's spatial patches per temporal slot at the 24x24
grid that 384/patch-16 produces.

    /home/jooyeolyun/anaconda3/envs/warp/bin/python scripts/v21_prep.py [LAB] [GRID]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

LAB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/scene/v1")
GRID = int(sys.argv[2]) if len(sys.argv) > 2 else 24
SIZE, TUBELET, DEPTH = 384, 2, 32
PATCH = SIZE // GRID


def main():
    import v1_killtest as V
    from t0_tracking import pulled_camera
    from src.surprise.regions import box_to_spatial_tokens

    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    meta = json.loads((LAB / "meta.json").read_text())
    cam = pulled_camera(V.CAM_PULL)
    boxes = V.shared_boxes(cam, meta, spec, 544, 448)
    out = {}
    for key, r in spec.items():
        clip = LAB / "clips" / f"{key}.mp4"
        if not clip.exists():
            continue
        x0, y0, x1, y1 = boxes[key]
        sc = SIZE / float(x1 - x0)
        pts = np.asarray(meta[key], float)
        per_slot = []
        for s in range(DEPTH):
            f0 = s * TUBELET
            sel = pts[min(f0, len(pts) - 1): min(f0 + TUBELET, len(pts))]
            uv = [cam.project(p)[0][cam.project(p)[1] > 0] for p in sel]
            uv = np.concatenate([a for a in uv if len(a)]) if any(len(a) for a in uv) else None
            if uv is None:
                per_slot.append(list(range(GRID * GRID))); continue
            b = [(uv[:, 0].min() - x0) * sc, (uv[:, 1].min() - y0) * sc,
                 (uv[:, 0].max() - x0) * sc, (uv[:, 1].max() - y0) * sc]
            sp = box_to_spatial_tokens(b, (1.0, 0.0, 0.0), grid=GRID, patch=PATCH, dilate=1)
            per_slot.append([int(v) for v in (sp if len(sp) else range(GRID * GRID))])
        out[key] = {**r, "clip": str(clip), "object_patches": per_slot,
                    "median_patches": int(np.median([len(p) for p in per_slot]))}
    dst = Path("docs/results") / f"v21_input_{LAB.name}.json"
    dst.write_text(json.dumps({"grid": GRID, "size": SIZE, "clips": out}))
    frac = 100 * np.median([v["median_patches"] for v in out.values()]) / (GRID * GRID)
    print(f"wrote {dst}  ({len(out)} clips; object covers ~{frac:.0f}% of the {GRID}x{GRID} grid)")


if __name__ == "__main__":
    main()
