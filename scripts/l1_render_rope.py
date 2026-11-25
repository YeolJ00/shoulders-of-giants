"""Watch generated rope episodes: each episode a Blender clip (the rope as a tube of its radius, the gripper, the
scanned objects and the plank on the table), an mp4 per episode with what it is, and one contact sheet.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_render_rope.py all <shard path without suffix>:<episode> [...] [--size 640x480 --samples 48]

`all` = export (clip.json + poses.npz per episode), Blender (scripts/blender_render.py), compose (labelled mp4 per
episode, a 2-column grid mp4 and a contact sheet of 5 instants per episode). Output: outputs/lsim/render_rope/.
Ropes thinner than 1.2 mm are drawn at 1.2 mm (said in the label) so a thread stays visible; one rope colour for all.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/lsim/render_rope"
BLENDER = "/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender"
DRAW_MIN_R = 0.0012
sys.path.insert(0, str(REPO / "scripts"))


def load(shard):
    shard = Path(shard)
    return np.load(shard.with_suffix(".npz")), json.loads(shard.with_suffix(".json").read_text())


def clip_dir(shard, e):
    return OUT / f"{Path(shard).parent.name}_{Path(shard).name}_e{e}"


def describe(wd):
    r = wd["rope"]
    mat = wd.get("material", {})
    cat = mat.get("category", "").replace("_", " ")
    cap = " (capped)" if mat.get("capped") else ""
    label = (f"{cat}: {100 * r['length']:.0f} cm, {1e3 * r['lam']:.1f} g/m, EA {r['EA']:,.0f} N{cap}, EI {r['EI']:.1e} N m2 "
             f"(bends over {100 * r['bending_length']:.1f} cm), mu {r['mu']:.2f}, r {1e3 * r['r']:.1f} mm"
             + (" (drawn 1.2)" if r["r"] < DRAW_MIN_R else ""))
    fams = ", ".join(p["family"].replace("_", " ") + (" (skipped)" if "skipped" in p else "") for p in wd["families"]) or "none"
    st = wd["init"] + (f" ({wd['start']['kind']})" if wd.get("start", {}).get("kind") else "")
    story = f"starts {st}: {fams}"
    if wd.get("ramp"):
        story += f" | plank {np.degrees(wd['ramp']['angle']):.0f} deg"
    return label, story


def camera(x, grip):
    """Eye and target for a rope clip: the box around the rope (x: (F, n, 3)) and the gripper while it works near the
    table."""
    low = grip[:, 2] - 0.055 < 0.3                                 # the gripper while it works near the table
    pts = np.concatenate([x.reshape(-1, 3), grip[low, :3]]) if low.any() else x.reshape(-1, 3)
    pts = pts[pts[:, 2] > -0.3] if (pts[:, 2] > -0.3).any() else pts            # frame the table, not the floor
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    ctr = 0.5 * (lo + hi)
    ctr[2] = 0.5 * (max(lo[2], -0.05) + min(hi[2], 0.35))
    span = float(np.linalg.norm(np.maximum(hi - lo, [0.25, 0.25, 0.1])))      # the box's diagonal, at least 25 cm
    az, el = np.radians(-60.0), np.radians(35.0)
    dist = 0.62 * span / np.tan(np.radians(19.0))                  # the whole box inside a 38 deg view, with margin
    eye = ctr + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return eye, ctr


def export(shard, e):
    from l2_render import obstacle_mesh
    npz, meta = load(shard)
    off = npz["cloth_off"]
    wd = meta["worlds"][e]
    r = wd["rope"]
    x = npz["cloth_d"][:, off[e]:off[e + 1]].astype(np.float32) + npz["cloth_c"][:, e:e + 1]
    grip = npz["grip_q"][:, e]
    bodies = [{"kind": "rope", "label": "rope", "rest": x[0].tolist(), "radius": max(float(r["r"]), DRAW_MIN_R)},
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
    if wd.get("ramp"):
        from lsim.data.scene import RAMP, ramp_frame
        from scipy.spatial.transform import Rotation
        o_, ax, lat, n = ramp_frame(wd["ramp"])
        pc = o_ + 0.5 * RAMP["L"] * ax + 0.5 * RAMP["T"] * n
        static.append({"boxes": [[[0.0, 0.0, 0.0], [0.5 * RAMP["L"], 0.5 * RAMP["W"], 0.5 * RAMP["T"]]]], "material": "furniture",
                       "pos": pc.tolist(), "quat": Rotation.from_matrix(np.stack([ax, lat, n], 1)).as_quat().tolist()})
    eye, ctr = camera(x, grip)
    label, story = describe(wd)
    d = clip_dir(shard, e)
    d.mkdir(parents=True, exist_ok=True)
    clip = {"kind": "rope", "fps": 30, "frames": int(x.shape[0]), "sources": ["teacher"], "labels": {"teacher": label},
            "story": story, "camera": {"eye": eye.tolist(), "target": ctr.tolist(), "fov": 38.0}, "static": static,
            "ghost": [], "bodies": bodies}
    (d / "clip.json").write_text(json.dumps(clip))
    np.savez(d / "poses.npz", teacher=poses, teacher_rope0=x)
    print(f"[export] {d.name}: {label} | {story}", flush=True)
    return d


def blender(d, size, samples):
    w, h = size
    subprocess.run([BLENDER, "--background", "--python", str(REPO / "scripts/blender_render.py"), "--", str(d), str(w), str(h),
                    str(samples)], check=True, stdout=subprocess.DEVNULL, env=dict(os.environ))
    print(f"[blender] {d.name}", flush=True)


def compose(dirs):
    r = __import__("l2_render")
    from PIL import Image
    clips = [json.loads((d / "clip.json").read_text()) for d in dirs]
    accent = (191, 110, 31, 255)
    labelled = []
    for d, c in zip(dirs, clips):                                  # one mp4 per episode
        out = d / "composite"
        out.mkdir(exist_ok=True)
        for f in range(c["frames"]):
            im = Image.open(d / "teacher" / f"f{f:04d}.png").convert("RGB")
            r._label(im, c["labels"]["teacher"], [(f"t = {f / c['fps']:4.2f} s   {c['story']}", r.WHITE)], accent)
            im.save(out / f"c{f:04d}.png")
        r.encode(out, OUT / f"{d.name}.mp4", c["fps"])
        labelled.append(out)
        print(f"[compose] {OUT / (d.name + '.mp4')}", flush=True)
    n = min(c["frames"] for c in clips)                            # a 2-column grid of all of them
    grid = OUT / "grid"
    grid.mkdir(exist_ok=True)
    w, h = Image.open(labelled[0] / "c0000.png").size
    cols = 2
    rows = (len(dirs) + cols - 1) // cols
    for f in range(n):
        canvas = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (255, 255, 255))
        for k, L in enumerate(labelled):
            canvas.paste(Image.open(L / f"c{f:04d}.png"), ((k % cols) * (w + 4), (k // cols) * (h + 4)))
        canvas.save(grid / f"c{f:04d}.png")
    r.encode(grid, OUT / "rope_pilot_grid.mp4", clips[0]["fps"])
    times = np.linspace(0, n - 1, 7).astype(int)                   # contact sheet: 7 instants per episode
    tw, th = w // 2, h // 2
    sheet = Image.new("RGB", (len(times) * tw + (len(times) - 1) * 3, len(dirs) * th + (len(dirs) - 1) * 3), (255, 255, 255))
    for k, L in enumerate(labelled):
        for j, f in enumerate(times):
            sheet.paste(Image.open(L / f"c{f:04d}.png").resize((tw, th)), (j * (tw + 3), k * (th + 3)))
    sheet.save(OUT / "rope_pilot_contact_sheet.jpg", quality=90)
    print(f"[compose] {OUT / 'rope_pilot_grid.mp4'}, {OUT / 'rope_pilot_contact_sheet.jpg'}", flush=True)


def main():
    args = sys.argv[1:]
    size, samples = (640, 480), 48
    if "--size" in args:
        k = args.index("--size")
        size = tuple(int(v) for v in args[k + 1].split("x"))
        args = args[:k] + args[k + 2:]
    if "--samples" in args:
        k = args.index("--samples")
        samples = int(args[k + 1])
        args = args[:k] + args[k + 2:]
    what, items = args[0], [(a.rsplit(":", 1)[0], int(a.rsplit(":", 1)[1])) for a in args[1:]]
    dirs = [clip_dir(s, e) for s, e in items]
    if what in ("export", "all"):
        dirs = [export(s, e) for s, e in items]
    if what in ("blender", "all"):
        for d in dirs:
            blender(d, size, samples)
    if what in ("compose", "all"):
        compose(dirs)


if __name__ == "__main__":
    main()
