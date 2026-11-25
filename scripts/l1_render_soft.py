"""Watch generated soft-body episodes: a shard's episodes as Blender clips, each an mp4, plus a contact sheet.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
    $L scripts/l1_render_soft.py export <shard path without suffix> <episode> [episode ...]
    CUDA_VISIBLE_DEVICES=<gpu> $B --background --python scripts/blender_render.py -- outputs/lsim/render_soft/<shard>_e<k>
    $L scripts/l1_render_soft.py compose <shard path without suffix> <episode> [episode ...]

Each clip (scripts/blender_render.py): the body's surface over all its tet vertices in its category's fixed look, the
gripper's palm and pads as the boxes the teacher collides, the scanned objects with their textures, the plank, on the
table. compose labels every frame (material, start, families) and encodes <shard>_e<k>.mp4;
    $L scripts/l1_render_soft.py sheet <clip dir name> [...]
tiles 6 frames of each clip into contact_sheet.jpg.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/lsim/render_soft"
sys.path.insert(0, str(REPO / "scripts"))


def load(shard):
    shard = Path(shard)
    return np.load(shard.with_suffix(".npz")), json.loads(shard.with_suffix(".json").read_text())


def _label(wd):
    s = wd["soft"]
    fams = ", ".join(p["family"].replace("_", " ") for p in wd["families"]) or "no action"
    title = (f"{s['params'].get('category', '?').replace('_', ' ')} {s['shape']} {100 * max(s['extent']):.0f} cm, "
             f"{1000 * s['mass']:.0f} g | rho {s['rho']:.0f}, E {s['E']:.2g} Pa, nu {s['nu']:.2f}, mu {s['mu']:.2f}, "
             f"zeta {s['zeta']:.2f}")
    return title, f"starts {wd['init'].replace('_', ' ')}: {fams}"


def gripper_bodies(grip, gw):
    """The soft teacher's gripper as Blender bodies (palm, two pads: the boxes it collides, rounded by its margin) and
    their poses per frame (F, 3, 7): the pads along the hand's local x at +-(w / 2 + half) of its recorded opening."""
    from lsim.teacher.soft_teacher import FINGER_HALF, PALM_HALF, R_ROUND
    grip, gw = np.asarray(grip, np.float64), np.asarray(gw, np.float64)
    bodies = [{"kind": "boxes", "label": "palm", "material": "gripper", "bevel": R_ROUND,
               "boxes": [[[0.0, 0.0, 2 * FINGER_HALF[2] + PALM_HALF[2]], list(PALM_HALF)]]},
              {"kind": "boxes", "label": "finger_l", "material": "gripper", "bevel": R_ROUND,
               "boxes": [[[0.0, 0.0, FINGER_HALF[2]], list(FINGER_HALF)]]},
              {"kind": "boxes", "label": "finger_r", "material": "gripper", "bevel": R_ROUND,
               "boxes": [[[0.0, 0.0, FINGER_HALF[2]], list(FINGER_HALF)]]}]
    poses = np.zeros((len(grip), 3, 7), np.float32)
    poses[:, 0] = grip
    for k in range(len(grip)):
        q = grip[k, 3:]
        ax = np.array([1 - 2 * (q[1] ** 2 + q[2] ** 2), 2 * (q[0] * q[1] + q[2] * q[3]), 2 * (q[0] * q[2] - q[1] * q[3])])
        for j, sg in ((1, -1.0), (2, 1.0)):
            poses[k, j, :3] = grip[k, :3] + sg * (gw[k] / 2 + FINGER_HALF[0]) * ax
            poses[k, j, 3:] = q
    return bodies, poses


def camera(x, pads):
    """Eye and target for a soft clip: the box around what moves on the table (x: (F, V, 3)) and the pads (pads: (F, 2,
    7) poses) while they are near the body (not their travel)."""
    from lsim.teacher.soft_teacher import FINGER_HALF
    pts = x.reshape(-1, 3)
    pts = pts[(pts[:, 2] > -0.1)] if (pts[:, 2] > -0.1).any() else pts
    cb = x.mean(1)
    near = (np.linalg.norm(pads[:, 0, :2] - cb[:, :2], axis=1) < 0.2) & (pads[:, 0, 2] < 0.25)
    if near.any():
        p = np.concatenate([pads[near, j, :3] + np.array([0.0, 0.0, dz]) for j in (0, 1) for dz in (0.0, 2 * FINGER_HALF[2])])
        pts = np.concatenate([pts, p])
    lo, hi = pts.min(0), pts.max(0)
    lo, hi = np.minimum(lo, x[0].mean(0) - 0.12), np.maximum(hi, x[0].mean(0) + 0.12)
    lo[2], hi[2] = max(lo[2], -0.05), max(min(hi[2], 0.35), 0.15)
    lo, hi = np.maximum(lo, [-0.75, -0.65, -0.1]), np.minimum(hi, [0.75, 0.65, 0.4])
    ctr = 0.5 * (lo + hi)
    ctr[2] = 0.35 * hi[2]
    r = max(0.28, 0.6 * float(np.linalg.norm(hi - lo)))
    az, el = np.radians(-60.0), np.radians(28.0)
    eye = ctr + r / np.tan(np.radians(19.0)) * 0.6 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return eye, ctr


def export(shard, eps):
    from l2_render import obstacle_mesh
    npz, meta = load(shard)
    off, toff = npz["cloth_off"], npz["tri_off"]
    for e in eps:
        wd = meta["worlds"][e]
        x = npz["cloth_d"][:, off[e]:off[e + 1]].astype(np.float32) + npz["cloth_c"][:, e:e + 1]
        tris = npz["soft_tris"][toff[e]:toff[e + 1]]
        grip, gw = npz["grip_q"][:, e].astype(np.float64), npz["grip_w"][:, e].astype(np.float64)
        F = x.shape[0]
        bodies = [{"kind": "soft", "label": "soft", "rest": x[0].tolist(), "tris": tris.tolist(),
                   "look": wd["soft"]["params"].get("category", "wide")}]
        gb, gp = gripper_bodies(grip, gw)
        bodies += gb
        poses = np.zeros((F, 4 + len(wd["obstacles"]), 7), np.float32)
        poses[:, :, 6] = 1.0
        poses[:, 1:4] = gp
        for k, o in enumerate(wd["obstacles"]):
            bodies.append(obstacle_mesh(o["name"]))
            poses[:, 4 + k, :3] = o["pos"]
            poses[:, 4 + k, 3:] = o["quat"]
        static = []
        if wd.get("ramp"):
            from lsim.data.scene import RAMP, ramp_frame
            from scipy.spatial.transform import Rotation
            o_, ax, lat, n = ramp_frame(wd["ramp"])
            pc = o_ + 0.5 * RAMP["L"] * ax + 0.5 * RAMP["T"] * n
            static.append({"boxes": [[[0.0, 0.0, 0.0], [0.5 * RAMP["L"], 0.5 * RAMP["W"], 0.5 * RAMP["T"]]]], "material": "furniture",
                           "pos": pc.tolist(), "quat": Rotation.from_matrix(np.stack([ax, lat, n], 1)).as_quat().tolist()})
        eye, ctr = camera(x, poses[:, 2:4])
        title, story = _label(wd)
        d = OUT / f"{Path(shard).name}_e{e}"
        d.mkdir(parents=True, exist_ok=True)
        clip = {"kind": "soft", "fps": 30, "frames": int(F), "sources": ["teacher"], "labels": {"teacher": title}, "story": story,
                "camera": {"eye": eye.tolist(), "target": ctr.tolist(), "fov": 38.0}, "static": static, "ghost": [], "bodies": bodies}
        (d / "clip.json").write_text(json.dumps(clip))
        np.savez(d / "poses.npz", teacher=poses, teacher_soft0=x)
        print(f"[export] {d.name}: {title} | {story}", flush=True)


def compose(shard, eps, fps=30):
    r = __import__("l2_render")
    from PIL import Image
    for e in eps:
        d = OUT / f"{Path(shard).name}_e{e}"
        c = json.loads((d / "clip.json").read_text())
        out = d / "composite"
        out.mkdir(exist_ok=True)
        for f in range(c["frames"]):
            im = Image.open(d / "teacher" / f"f{f:04d}.png").convert("RGB")
            r._label(im, c["labels"]["teacher"], [(f"t = {f / fps:4.2f} s   {c['story']}", r.WHITE)], (31, 92, 70, 255))
            im.save(out / f"c{f:04d}.png")
        r.encode(out, OUT / f"{d.name}.mp4", fps)
        print(f"[compose] {OUT / (d.name + '.mp4')}", flush=True)


def sheet(clips, n=6):
    """clips: rendered clip directory names under OUT (e.g. shard_00003_e12); 6 frames of each in a row."""
    from PIL import Image
    rows = []
    for name in clips:
        frames = sorted((OUT / name / "composite").glob("c*.png"))
        pick = [frames[int(round(k * (len(frames) - 1) / (n - 1)))] for k in range(n)]
        ims = [Image.open(p).convert("RGB") for p in pick]
        w, h = ims[0].size
        rows.append([im.resize((w // 2, h // 2)) for im in ims])
    w, h = rows[0][0].size
    canvas = Image.new("RGB", (n * w + (n - 1) * 3, len(rows) * h + (len(rows) - 1) * 3), (255, 255, 255))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            canvas.paste(im, (j * (w + 3), i * (h + 3)))
    canvas.save(OUT / "contact_sheet.jpg", quality=88)
    print(f"[sheet] {OUT / 'contact_sheet.jpg'}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        sheet(sys.argv[2:])
    else:
        what, shard, eps = sys.argv[1], sys.argv[2], [int(e) for e in sys.argv[3:]]
        {"export": export, "compose": compose}[what](shard, eps)
