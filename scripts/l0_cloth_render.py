"""Cloth gate scenes you can watch: each solver configuration simulates the same cloth, then Blender
(scripts/blender_render.py) draws the panels side by side.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l0_cloth_render.py export [scene ...]   # drape swing cantilever
    CUDA_VISIBLE_DEVICES=<gpu> $B --background --python scripts/blender_render.py -- outputs/lsim/render_cloth/<scene>
    $L scripts/l0_cloth_render.py compose [scene ...]

Coordinates: the gate's table top (z = 0.75, edge at x = 0) becomes the render table's (z = 0, edge at
x = 0.6); the swing hangs from a rail beside the table, the cantilever sits on a ledge on the table.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/lsim/render_cloth"
FPS = 30


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# the configurations compared: (label, solver, overrides); the same cloth in physical units for all
CONFIGS = {
    "vbd": ("VBD, Newton's default cloth solver: 20 substeps x 10 iterations", "vbd", {}),
    "style3d": ("Style3D, converged: 10 substeps x 4 x 60 PCG iterations", "style3d", {}),
    "explicit": ("Explicit, 2,000 substeps a frame: the cloth teacher", "semi", {"self_contact": 1}),
    "explicit_noself": ("Explicit without self-contact", "semi", {}),
}
ACCENT = {"vbd": (150, 156, 162, 255), "style3d": (166, 110, 40, 255), "explicit": (31, 92, 70, 255),
          "explicit_noself": (191, 54, 12, 255)}


def scenes(g):
    C = g.CLOTH

    def drape(b, solver, cloth):
        return g._towel_on_table()(b, solver, cloth)

    def swing(b, solver, cloth):
        ids = g.add_sheet(b, solver, cloth, 30, 30, (0.0, -0.15, 1.0))
        x = np.array([b.particle_q[i] for i in ids])
        return {"fixed": ids[x[:, 0] < 1e-6], "ids": ids}

    c = (C["B"] / (C["w"] * g.G)) ** (1.0 / 3.0)
    l = round(2.0 * c / C["h"]) * C["h"]

    def cantilever(b, solver, cloth):
        nx = int(round((l + 0.03) / cloth["h"]))
        ids = g.add_sheet(b, solver, cloth, nx, 3, (-0.03, -0.015, 1.0))
        x = np.array([b.particle_q[i] for i in ids])
        return {"fixed": ids[x[:, 0] < 1e-6], "ids": ids}

    def fold(b, solver, cloth):
        b.add_ground_plane()
        g.static_box(b, (0.0, 0.0, g.TOP / 2), (0.5, 0.5, g.TOP / 2), cloth["mu"])
        return {"ids": g.add_sheet(b, solver, cloth, 30, 30, (-0.40, -0.15, g.TOP + cloth["r"]))}

    def fold_script(sim, f):                       # the gripper: up and across, across and down, let go
        if f == 5:
            x = sim.x()
            sim.meta["corner"] = int(np.argmin(np.abs(x[:, 0] - x[:, 0].max()) + np.abs(x[:, 1] - x[:, 1].max())))
            sim.grasp([sim.meta["corner"]], (-0.25, -0.25, 0.20))
        elif f == 23:
            sim.grasp([sim.meta["corner"]], (-0.25, -0.25, -0.19))
        elif f == 41:
            sim.kin, sim.graph = None, None

    three = ["vbd", "style3d", "explicit"]
    return {
        # name: (scene, grid nx, ny, drag, seconds, render offset, camera, static boxes, sources, script)
        "drape": (drape, 30, 30, 1.0, 4.0, (0.6, 0.0, -0.75),
                  {"eye": [1.30, -0.80, 0.42], "target": [0.52, 0.0, -0.06], "fov": 34.0}, [], three, None),
        "swing": (swing, 30, 30, 0.0, 4.0, (0.62, 0.0, -0.60),
                  {"eye": [1.42, -0.92, 0.46], "target": [0.70, 0.0, 0.26], "fov": 38.0},
                  [{"boxes": [[[0.62, 0.0, 0.40], [0.006, 0.17, 0.006]]], "material": "gripper"}], three, None),
        "cantilever": (cantilever, int(round((l + 0.03) / C["h"])), 3, 4.0, 3.0, (0.30, 0.0, -0.90),
                       {"eye": [0.34, -0.42, 0.13], "target": [0.33, 0.0, 0.085], "fov": 26.0},
                       [{"boxes": [[[0.235, 0.0, 0.0485], [0.065, 0.03, 0.0485]]], "material": "furniture"}], three, None),
        "fold": (fold, 30, 30, 1.0, 2.9, (0.6, 0.0, -0.75),
                 {"eye": [0.62, -0.62, 0.55], "target": [0.35, 0.0, 0.0], "fov": 40.0}, [],
                 ["explicit_noself", "explicit"], fold_script),
    }


def export(names):
    import warp as wp
    g = _module("cloth_gate", REPO / "scripts/l0_cloth_gate.py")
    wp.init()
    sc = scenes(g)
    for name in names:
        fn, nx, ny, drag, seconds, off, cam, static, sources, script = sc[name]
        v, t = g.grid(nx, ny, g.CLOTH["h"])
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        poses, info = {}, {}
        for key in sources:
            label, solver, over = CONFIGS[key]
            sim = g.Sim(solver, fn, g.CLOTH, drag=drag, **over)
            ids = sim.meta["ids"]
            xs = [sim.x()[ids]]
            grip = [np.array([0.0, 0.0, 9.0])]             # the gripper waits out of view until it grasps
            for f in range(int(seconds * FPS)):
                if script is not None:
                    script(sim, f)
                sim.frame()
                xs.append(sim.x()[ids])
                if sim.kin is not None:                    # just above the grasped corner
                    grip.append(sim.x()[int(sim.kin[0].numpy()[0])] + np.array([0.0, 0.0, 0.004]))
                else:                                      # let go: it lifts away at 0.3 m/s
                    grip.append(grip[-1] + np.array([0.0, 0.0, 0.3 / FPS]) if grip[-1][2] < 5.0 else grip[-1])
            xs = np.array(xs) + np.array(off)
            poses[f"{key}_cloth0"] = xs.astype(np.float32)
            poses[key] = np.zeros((len(xs), 2, 7))
            poses[key][:, 1, :3] = np.array(grip) + np.array(off)
            poses[key][:, :, 6] = 1.0
            sp = np.linalg.norm(sim.v()[ids], axis=1)
            info[key] = {"max_speed_end_m_s": float(sp.max()), "sim_s_per_wall_s": sim.rate()}
            print(f"[export] {name} {key}: {len(xs)} frames, max speed at end {sp.max():.4f} m/s, {sim.rate():.2f} sim-s/wall-s", flush=True)
        clip = {"kind": "cloth", "scene": name, "fps": FPS, "frames": len(xs), "sources": sources,
                "labels": {k: CONFIGS[k][0] for k in sources}, "info": info, "camera": cam, "static": static,
                "ghost": [], "bodies": [{"kind": "cloth", "label": "cloth", "rest": (np.array(v) + np.array(off)).tolist(),
                                         "tris": np.array(t).reshape(-1, 3).tolist(), "uv": np.array(v)[:, :2].tolist()}]}
        if script is not None:                             # a parallel-jaw gripper: palm and two fingers
            clip["bodies"].append({"kind": "boxes", "label": "gripper", "material": "gripper",
                                   "boxes": [[[0.0, 0.0, 0.034], [0.012, 0.032, 0.006]],
                                             [[0.0, 0.013, 0.014], [0.008, 0.004, 0.014]],
                                             [[0.0, -0.013, 0.014], [0.008, 0.004, 0.014]]]})
        (d / "clip.json").write_text(json.dumps(clip))
        np.savez(d / "poses.npz", **poses)


def compose(names):
    r = _module("l2_render", REPO / "scripts/l2_render.py")
    from PIL import Image
    accent = ACCENT
    for name in names:
        d = OUT / name
        clip = json.loads((d / "clip.json").read_text())
        out = d / "composite"
        out.mkdir(exist_ok=True)
        for f in range(clip["frames"]):
            tiles = []
            for s in clip["sources"]:
                im = Image.open(d / s / f"f{f:04d}.png").convert("RGB")
                r._label(im, clip["labels"][s], [(f"t = {f / clip['fps']:4.2f} s", r.WHITE)], accent[s])
                tiles.append(im)
            canvas = Image.new("RGB", (sum(x.width for x in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (255, 255, 255))
            x = 0
            for tl in tiles:
                canvas.paste(tl, (x, 0))
                x += tl.width + 4
            canvas.save(out / f"c{f:04d}.png")
        r.encode(out, OUT / f"{name}.mp4", clip["fps"])
        Image.open(out / f"c{clip['frames'] - 1:04d}.png").save(OUT / f"{name}_last.jpg", quality=88)
        print(f"[compose] {OUT / f'{name}.mp4'}", flush=True)


if __name__ == "__main__":
    what, names = sys.argv[1], sys.argv[2:] or ["drape", "swing", "cantilever", "fold"]
    export(names) if what == "export" else compose(names)
