"""Compose several independent probe rollouts into ONE render, then cut them apart.

WHY. Measured at 544x448/48 samples, a frame costs ~1.43 s of fixed work (CPU-side scene
update, denoise setup, file write) against ~0.58 s that scales with pixels, and only ~0.16 s
of that is actually sampling -- halving samples buys 6%. So the lever is FEWER RENDER PASSES,
not cheaper ones: N cells in one frame cost `1.43 + 0.58 N`. Measured on a 4-cell prototype:
0.70 s per cell-frame against 1.24 s single-cell, a 1.77x speedup.

This is a RENDER-SIDE composition only. Each cell is a separate simulation that never
interacted with the others, so no physics changes and no contact pair is introduced.

LAYOUT. Cells are offset along the CAMERA'S RIGHT vector, not along world +x. The probe
views look diagonally across the table, so a world-x offset moves cells toward and away from
the camera: a first attempt put the four cells at image y 417/442/468/500 with visibly
different apparent sizes, and two of them overlapped horizontally. Offsetting along camera
right keeps every cell at the same distance, so they differ only by a translation across the
image.

REMAINING CAVEAT. Even equidistant, a cell at the edge is seen at a different angle than one
at the centre -- perspective is not translation-invariant. So a cell is pinned to an ASSET
for the whole experiment, never to a theta: per-asset calibration then absorbs the cell's
fixed geometry, which is exactly the regime the per-asset readouts already work in. Putting
different theta of one asset in different cells would make cell position a confound.
"""
import json
from pathlib import Path

import numpy as np


def camera_right(view="a"):
    """Unit vector across the image, horizontal in world terms."""
    from src.render.views import VIEWS
    v = VIEWS[view]
    d = np.asarray(v["target"], float) - np.asarray(v["eye"], float)
    d /= np.linalg.norm(d)
    r = np.cross(d, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(r)
    if n < 1e-9:                      # straight-down view: any horizontal axis will do
        return np.array([1.0, 0.0, 0.0])
    return r / n


def cell_offsets(n, pitch, view="a"):
    """n offsets, centred on the origin, spaced `pitch` metres along camera right."""
    r = camera_right(view)
    k = (np.arange(n) - (n - 1) / 2.0) * float(pitch)
    return [r * x for x in k]


def compose(items, out_lab, view="a", pitch=0.42, tilt_deg=0.0, key="grid"):
    """items = [(lab, scene_key, cell_name), ...] -> one lab holding a single grid scene.

    Every item contributes one object, offset into its own cell. The number of frames is
    the shortest sequence, so a diverged or short rollout truncates the grid rather than
    corrupting it.
    """
    offs = cell_offsets(len(items), pitch, view)
    objs, poses = {}, {}
    for (lab, k, name), off in zip(items, offs):
        src = Path(lab)
        sp = json.loads((src / "scene_poses.json").read_text())
        cfg = json.loads((src / "scene.json").read_text())
        oname = next(iter(sp[k]["objects"]))
        seq = sp[k]["objects"][oname]
        poses[name] = [{"loc": [float(p["loc"][i] + off[i]) for i in range(3)],
                        "mat": p["mat"]} for p in seq]
        o = cfg["objects"][oname]
        objs[name] = {**o, "pos": [float(o["pos"][i] + off[i]) for i in range(3)]}
    nf = min(len(v) for v in poses.values())
    poses = {k2: v[:nf] for k2, v in poses.items()}
    d = Path(out_lab); d.mkdir(parents=True, exist_ok=True)
    (d / "scene_poses.json").write_text(json.dumps(
        {key: {"tilt_deg": tilt_deg, "objects": poses}}))
    (d / "scene.json").write_text(json.dumps({"ground_z": 0.706, "objects": objs}))
    return {"cells": list(poses), "frames": nf, "offsets": [o.tolist() for o in offs]}
