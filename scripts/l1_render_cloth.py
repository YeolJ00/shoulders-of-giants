"""Watch generated cloth episodes: a shard's episodes as Blender panels in a grid video.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l1_render_cloth.py export <shard path without suffix> <episode> [episode ...]
    CUDA_VISIBLE_DEVICES=<gpu> $B --background --python scripts/blender_render.py -- outputs/lsim/render_c1/<shard>_e<k>
    $L scripts/l1_render_cloth.py compose <shard path without suffix> <episode> [episode ...]

Each episode is its own clip (scripts/blender_render.py): the cloth, the gripper, the scanned objects with
their textures on the table. The composite tiles them 2 x 2 with what each one is.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/lsim/render_c1"
sys.path.insert(0, str(REPO / "scripts"))


def load(shard):
    shard = Path(shard)
    npz = np.load(shard.with_suffix(".npz"))
    meta = json.loads(shard.with_suffix(".json").read_text())
    return npz, meta


def export(shard, eps):
    from l2_render import obstacle_mesh
    from lsim.teacher.cloth_teacher import grid
    npz, meta = load(shard)
    off = npz["cloth_off"]
    for e in eps:
        wd = meta["worlds"][e]
        c = wd["cloth"]
        x = npz["cloth_d"][:, off[e]:off[e + 1]].astype(np.float32) + npz["cloth_c"][:, e:e + 1]
        v, t = grid(c["nx"], c["ny"], c["h"])
        grip = npz["grip_q"][:, e]
        bodies = [{"kind": "cloth", "label": "cloth", "rest": x[0].tolist(), "tris": t.tolist(), "uv": v[:, :2].tolist(),
                   "color": [0.30, 0.45, 0.70] if c["w"] < 0.15 else ([0.80, 0.42, 0.30] if c["w"] < 0.3 else [0.45, 0.55, 0.35])},
                  {"kind": "boxes", "label": "gripper", "material": "gripper",
                   "boxes": [[[0.0, 0.0, 0.03], [0.05, 0.02, 0.015]], [[-0.0078, 0.0, -0.02], [0.0075, 0.015, 0.035]],
                             [[0.0078, 0.0, -0.02], [0.0075, 0.015, 0.035]]]}]
        poses = np.zeros((x.shape[0], 2 + len(wd["obstacles"]), 7), np.float32)
        poses[:, :, 6] = 1.0
        poses[:, 1] = grip
        for k, o in enumerate(wd["obstacles"]):
            bodies.append(obstacle_mesh(o["name"]))
            poses[:, 2 + k, :3] = o["pos"]
            poses[:, 2 + k, 3:] = o["quat"]
        static = []
        if wd.get("ramp"):                                   # the plank the cloth slides on
            from lsim.data.scene import RAMP, ramp_frame
            from scipy.spatial.transform import Rotation
            o_, ax, lat, n = ramp_frame(wd["ramp"])
            pc = o_ + 0.5 * RAMP["L"] * ax + 0.5 * RAMP["T"] * n
            static.append({"boxes": [[[0.0, 0.0, 0.0], [0.5 * RAMP["L"], 0.5 * RAMP["W"], 0.5 * RAMP["T"]]]], "material": "furniture",
                           "pos": pc.tolist(), "quat": Rotation.from_matrix(np.stack([ax, lat, n], 1)).as_quat().tolist()})
        lo = np.minimum(x.reshape(-1, 3).min(0), np.array([-0.3, -0.25, 0.0]))
        hi = np.maximum(x.reshape(-1, 3).max(0), np.array([0.3, 0.25, 0.1]))
        lo, hi = np.maximum(lo, [-0.75, -0.65, -0.3]), np.minimum(hi, [0.75, 0.65, 0.6])
        ctr = 0.5 * (lo + hi)
        r = max(0.35, 0.62 * float(np.linalg.norm(hi - lo)))
        az, el = np.radians(-60.0), np.radians(36.0)
        eye = ctr + r / np.tan(np.radians(19.0)) * 0.55 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        fams = ", ".join(p["family"].replace("_", " ") for p in wd["families"]) or "none"
        label = (f"{c['nx']} x {c['ny']} cm, {1000 * c['w']:.0f} g/m2, Y {c['Y']:,.0f} N/m, bends over "
                 f"{100 * c['bending_length']:.1f} cm, mu {c['mu']:.2f}")
        story = f"starts {wd['init']}: {fams}"
        if wd.get("ramp"):
            story += f" | plank {np.degrees(wd['ramp']['angle']):.0f} deg"
        d = OUT / f"{Path(shard).name}_e{e}"
        d.mkdir(parents=True, exist_ok=True)
        clip = {"kind": "cloth", "fps": 30, "frames": int(x.shape[0]), "sources": ["teacher"], "labels": {"teacher": label},
                "story": story,
                "camera": {"eye": eye.tolist(), "target": ctr.tolist(), "fov": 38.0}, "static": static, "ghost": [], "bodies": bodies}
        (d / "clip.json").write_text(json.dumps(clip))
        np.savez(d / "poses.npz", teacher=poses, teacher_cloth0=x)
        print(f"[export] {d.name}: {label} | {story}", flush=True)


def compose(shard, eps, fps=30):
    r = __import__("l2_render")
    from PIL import Image
    dirs = [OUT / f"{Path(shard).name}_e{e}" for e in eps]
    clips = [json.loads((d / "clip.json").read_text()) for d in dirs]
    n = min(c["frames"] for c in clips)
    out = OUT / f"{Path(shard).name}_grid"
    out.mkdir(parents=True, exist_ok=True)
    for f in range(n):
        tiles = []
        for d, c in zip(dirs, clips):
            im = Image.open(d / "teacher" / f"f{f:04d}.png").convert("RGB")
            r._label(im, c["labels"]["teacher"], [(f"t = {f / fps:4.2f} s   {c['story']}", r.WHITE)], (31, 92, 70, 255))
            tiles.append(im)
        w, h = tiles[0].size
        cols = 2
        rows = (len(tiles) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (255, 255, 255))
        for k, tl in enumerate(tiles):
            canvas.paste(tl, ((k % cols) * (w + 4), (k // cols) * (h + 4)))
        canvas.save(out / f"c{f:04d}.png")
    r.encode(out, OUT / f"{Path(shard).name}_grid.mp4", fps)
    Image.open(out / f"c{n - 1:04d}.png").save(OUT / f"{Path(shard).name}_grid_last.jpg", quality=88)
    print(f"[compose] {OUT / (Path(shard).name + '_grid.mp4')}", flush=True)


if __name__ == "__main__":
    what, shard, eps = sys.argv[1], sys.argv[2], [int(e) for e in sys.argv[3:]]
    export(shard, eps) if what == "export" else compose(shard, eps)
