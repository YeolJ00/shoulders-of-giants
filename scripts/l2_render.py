"""L2 -- rollouts you can watch: the teacher's episode beside the models' free rollouts, rendered.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python; B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
    $L scripts/l2_render.py export [n_episodes=24]   # trajectories -> outputs/lsim/render/<clip>/{clip.json, poses.npz}
    $L scripts/l2_render.py hinge <drawer|lid>       # the first test episode that opens one, wherever it is
    $B --background --python scripts/blender_render.py -- outputs/lsim/render/<clip>    # panels -> PNG frames
    $L scripts/l2_render.py compose                  # panels side by side, labelled -> outputs/lsim/render/<clip>.mp4

Scene clips (held-out objects, v0_test): each model is given frames 0-7 and then runs alone to the end of
the episode (5.7 s) while the gripper follows its recorded path. Panels: the teacher (Kamino), model v0,
model v1; the model panels carry the teacher's objects as ghosts. The error under a panel is the drift
diagnostic's (scripts/l2_drift.py): mean vertex distance from the teacher, averaged over the rigid
objects. Which episodes are shown is fixed by rule, not by eye: the first n test episodes are all
rolled out, and the clips are the episodes at v1's median and 90th-percentile final error plus the first
test episodes that open a drawer and a lid (searched past the first n when needed).
Phys101 clip: a real cardboard cube's slide (the footage, side view) beside each model's slide of the
same cube at the friction that fits it best (scripts/l2_phys101.py's protocol)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from lsim.data import assets
from lsim.data.scene import ramp_frame
from lsim.model.net import build_model
from lsim.model.rollout import rollout
from lsim.real.phys101 import align_err, objects, ramp_segment, ramp_tracks
from lsim.real.synthetic import ramp_cube
from lsim.train import windows as W
from lsim.train.windows import PARAMS, load_shard

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/lsim/render"
CKPTS = {"v0": "outputs/lsim/train_v0/last.pt", "v1": "outputs/lsim/train_v1/last.pt"}
LABELS = {"teacher": "Teacher (Kamino): ground truth", "v0": "Model v0: support contact",
          "v1": "Model v1: Coulomb contact", "real": "Real footage (Physics 101)"}
FFMPEG = "/home/jooyeolyun/anaconda3/envs/svg/bin/ffmpeg"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FPS = 30.0
P = 8


def vert_err(b, p, t):
    """Mean vertex distance between two poses (..., 7) of body b (scripts/l2_drift.py)."""
    v = b.verts.astype(float)
    return np.linalg.norm(W.rotate(p[3:], v) + p[:3] - W.rotate(t[3:], v) - t[:3], axis=-1).mean()


def boxes_of(fn, *args):
    """The (centre, half extents) boxes a windows.py mesh builder is made of: it hands them to
    _boxes_points, which is watched here, so the render shows exactly the geometry the model sees."""
    seen, real = [], W._boxes_points
    W._boxes_points = lambda boxes, *a, **k: seen.append(boxes) or real(boxes, *a, **k)
    try:
        getattr(fn, "__wrapped__", fn)(*args)
    finally:
        W._boxes_points = real
    return [[list(map(float, c)), list(map(float, h))] for c, h in seen[0]]


def gso_mesh(name):
    """The scan with its texture, and the offset that puts it in the body frame (origin at the
    uniform-solid centre of mass, exactly as lsim.data.assets.process placed it)."""
    path = REPO / f"data/assets/gso/{name}/meshes/model.obj"
    raw = assets.load_mesh(path)
    com = raw.center_mass if raw.is_watertight else assets.shell_props(raw)[0]
    col = np.load(REPO / f"data/assets/proc/gso/{name}.npz")["collide_v"]       # same frame? (rule 6)
    v = np.asarray(raw.vertices) - com
    gap = max(np.abs(v.min(0) - col.min(0)).max(), np.abs(v.max(0) - col.max(0)).max())
    assert gap < 2e-3, f"{name}: processed mesh {1000 * gap:.1f} mm off the scan"
    return {"kind": "mesh", "obj": str(path), "texture": str(path.parents[1] / "materials/textures/texture.png"),
            "com": [float(c) for c in com]}


def obstacle_mesh(name):
    """What blender_render.py draws for an obstacle: a GSO scan (gso_mesh), or a SimReady asset as its own textured
    USD (kind "usd": a flattened copy, scripts/l1_render_simready.flat_usd, placed in the body frame by rest_T)."""
    from lsim.data import library
    if library.source_of(name) == "gso":
        return gso_mesh(name)
    from l1_render_simready import flat_usd
    from lsim.data import simready
    row = next(r for r in simready.manifest() if r["name"] == name)
    flat = flat_usd(row, REPO / "outputs/lsim/render_simready/flat")
    T = np.load(simready.PROC / f"{name}.npz")["rest_T"]
    return {"kind": "usd", "usd": str(flat), "rest_T": np.asarray(T, float).tolist()}


def camera_scene(pts, az=-60.0, el=34.0, fov=38.0):
    """Look at the box around the points (object centres +- their size, the gripper over the table)
    from the front right, where the gripper comes from."""
    pts = pts if len(pts) else np.array([[-0.2, -0.2, 0.0], [0.2, 0.2, 0.1]])
    lo, hi = pts.min(0), pts.max(0)
    c = 0.5 * (lo + hi)
    c[2] = 0.04
    r = max(0.5 * np.linalg.norm(hi[:2] - lo[:2]), 0.3)
    d = r / np.tan(np.radians(fov / 2)) * 1.05
    a, e = np.radians(az), np.radians(el)
    eye = c + d * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return {"eye": eye.tolist(), "target": c.tolist(), "fov": fov}


def load_models(dev):
    out = {}
    for k, p in CKPTS.items():
        ck = torch.load(REPO / p, map_location=dev)
        m = build_model(ck.get("args", {}), len(PARAMS)).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        out[k] = m
    return out


def scene_rollouts(ep, models, t0=P - 1):
    """Poses (frames, bodies, 7) per source and the per-frame error of each model (mm)."""
    ids = [b.body for b in ep.bodies]
    F = ep.q.shape[0]
    poses = {"teacher": ep.q[:, ids].astype(np.float32)}
    for k, m in models.items():
        pred, _, _ = rollout(m, ep, t0, F - 1 - t0)
        q = poses["teacher"].copy()
        q[t0 + 1:] = pred
        poses[k] = q
    rigid = [i for i, b in enumerate(ep.bodies) if b.role == "dynamic" and b.type == "rigid"]
    err = {k: [1000 * float(np.mean([vert_err(ep.bodies[i], poses[k][f, i], poses["teacher"][f, i]) for i in rigid]))
               for f in range(F)] for k in models}
    return poses, err


def write_scene_clip(name, ep, poses, err, why):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    bodies, ghost = [], []
    for i, b in enumerate(ep.bodies):
        if b.type == "rigid" and b.role == "dynamic":
            bodies.append(dict(gso_mesh(b.name), label=b.name, material="scan"))
            ghost.append(i)
        elif b.name == "gripper":
            bodies.append({"kind": "boxes", "boxes": boxes_of(W.gripper_mesh), "label": "gripper", "material": "gripper"})
        else:
            bodies.append({"kind": "boxes", "boxes": boxes_of(W.hinge_mesh, b.name), "label": b.name, "material": "furniture"})
            ghost.append(i)
    static = []
    h = ep.wd.get("hinge")
    if h:
        static.append({"boxes": boxes_of(W.cabinet_points, h["kind"], *map(float, h["xy"])), "material": "furniture"})
    tq = poses["teacher"]
    size = np.array([b.params["size"] / 2 if b.type == "rigid" and b.role == "dynamic" else 0.15 for b in ep.bodies])
    pts = np.concatenate([tq[:, i, :3] + sgn * size[i] * np.array([1.0, 1.0, 0.0]) for i in ghost for sgn in (-1, 1)])
    grip = [i for i, b in enumerate(ep.bodies) if b.name == "gripper"]
    pts = np.concatenate([pts, tq[:, grip[0], :3]]) if grip else pts
    pts = pts[(np.abs(pts[:, 0]) < 0.6) & (np.abs(pts[:, 1]) < 0.5) & (pts[:, 2] > -0.05)]   # on the table
    if h:
        pts = np.concatenate([pts, [[*h["xy"], 0.1]]])
    clip = {"kind": "scene", "fps": FPS, "frames": int(tq.shape[0]), "given": P, "camera": camera_scene(pts),
            "bodies": bodies, "ghost": ghost, "static": static, "sources": ["teacher", *err.keys()],
            "labels": LABELS, "err_mm": err, "why": why,
            "episode": {"shard": ep.shard, "world": ep.w, "objects": [b.name for b in ep.bodies if b.type == "rigid" and b.role == "dynamic"],
                        "hinge": h["kind"] if h else None, "families": [f["family"] for f in ep.wd["families"]]}}
    (d / "clip.json").write_text(json.dumps(clip, indent=1))
    np.savez_compressed(d / "poses.npz", **poses)


def phys101_clip(models, trial_name):
    """The first fitted 20 deg trial unless named: each model's best-fitting slide of that cube."""
    objs, R = objects(), ramp_tracks("cardboard")
    t = next(r for r in R if r["clip"] == trial_name)
    cube = trial_name.split("/")[0]
    mass_g, vol = objs[cube]
    edge = vol ** (1 / 3) / 100.0
    rt, rs = ramp_segment(t)
    MUS = np.round(np.arange(0.15, 0.351, 0.02), 3)
    H = 36
    poses, fit, ramp = {}, {}, None
    for k, m in models.items():
        best = None
        for mu in MUS:
            ep, ramp = ramp_cube(edge, mass_g / 1000.0, float(mu), 20.0, 0.42, P + H)
            pred, _, _ = rollout(m, ep, P - 1, H)
            _, ax, _, _ = ramp_frame(ramp)
            q = ep.q[:P + H].astype(np.float32).copy()                          # frames 0 .. P+H-1
            q[P:] = pred
            curve = (q[P - 1:P + H, 0, :3] - q[P - 1, 0, :3]) @ (-ax)          # released at frame P-1
            e, t0, s0 = align_err(rt, rs, curve, FPS)
            if best is None or e < best[0]:
                best = (e, t0, s0, float(mu), q, curve)
        e, t0, s0, mu, q, curve = best
        poses[k] = q
        fit[k] = {"rms_mm": 1000 * e, "t0": t0, "s0": s0, "mu": mu, "curve_m": curve.tolist()}
        print(f"[phys101] {trial_name} {k}: mu {mu}, rms {1000 * e:.2f} mm", flush=True)
    o, ax, lat, n = ramp_frame(ramp)
    R_ = np.stack([ax, lat, n], 1)
    from scipy.spatial.transform import Rotation
    plank = {"boxes": [[[0.5 * W.RAMP["L"], 0.0, 0.5 * W.RAMP["T"]], [0.5 * W.RAMP["L"], 0.5 * W.RAMP["W"], 0.5 * W.RAMP["T"]]]],
             "pos": o.tolist(), "quat": Rotation.from_matrix(R_).as_quat().tolist(), "material": "board"}
    mid = o + 0.5 * W.RAMP["L"] * ax
    target = mid + np.array([0.0, 0.0, 0.02])
    cam = {"eye": (target + np.array([0.0, 0.95, 0.0])).tolist(), "target": target.tolist(), "fov": 36.0}
    name = "phys101_" + trial_name.replace("/", "_")
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    clip = {"kind": "phys101", "fps": FPS, "frames": P + H, "given": P, "camera": cam,
            "bodies": [{"kind": "boxes", "boxes": [[[0, 0, 0], [edge / 2] * 3]], "label": "cube", "material": "cardboard"}],
            "ghost": [], "static": [plank], "sources": list(models), "labels": LABELS,
            "trial": trial_name, "video": str(REPO / "data/phys101/phys101/scenarios/ramp" / trial_name / "Camera_1.mp4"),
            "video_fps": t["fps"], "real_t": rt.tolist(), "real_s": rs.tolist(), "edge_m": edge, "mass_g": mass_g, "fit": fit}
    (d / "clip.json").write_text(json.dumps(clip, indent=1))
    np.savez_compressed(d / "poses.npz", **{k: v[:, :1] for k, v in poses.items()})


def export(n_eps):
    dev = torch.device("cuda")
    models = load_models(dev)
    phys101_clip(models, "cardboard_01/20_01/01")
    shards = sorted((REPO / "data/datasets/v0_test").glob("shard_*.npz"))
    eps = []
    for s in shards:
        for e in load_shard(str(s)[:-4]):
            e.shard = s.stem
            eps.append(e)
            if len(eps) == n_eps:
                break
        if len(eps) == n_eps:
            break
    rows = []
    for e in eps:
        poses, err = scene_rollouts(e, models)
        fam = [f["family"] for f in e.wd["families"]]
        rows.append({"shard": e.shard, "world": e.w, "hinge": (e.wd.get("hinge") or {}).get("kind"), "families": fam,
                     **{f"{k}_1s_mm": err[k][P - 1 + 30] for k in err}, **{f"{k}_end_mm": err[k][-1] for k in err}})
        print(f"[scene] {e.shard} w{e.w:3d} {rows[-1]['hinge'] or '-':6s} " + " ".join(
            f"{k} 1 s {err[k][P - 1 + 30]:7.1f} end {err[k][-1]:7.1f} mm" for k in err), flush=True)
        e.poses, e.err = poses, err
    (OUT / "scene_errors.json").write_text(json.dumps(rows, indent=1))
    end = np.array([r["v1_end_mm"] for r in rows])
    order = np.argsort(end)
    pick = {"median": int(order[len(order) // 2]), "p90": int(order[int(np.ceil(0.9 * len(order))) - 1])}
    for kind in ("drawer", "lid"):
        i = next((i for i, r in enumerate(rows) if r["hinge"] == kind and "hinge" in r["families"]), None)
        if i is not None:
            pick[kind] = i
    why = {"median": "v1's median final error among the first %d test episodes" % len(rows),
           "p90": "v1's 90th-percentile final error among the first %d test episodes" % len(rows),
           "drawer": "the first test episode whose gripper opens a drawer", "lid": "the first test episode whose gripper opens a lid"}
    for k, i in pick.items():
        e = eps[i]
        write_scene_clip(f"scene_{k}", e, e.poses, e.err, why[k])
        print(f"[pick] {k}: {e.shard} world {e.w} (v0 end {rows[i]['v0_end_mm']:.1f} mm, v1 end {rows[i]['v1_end_mm']:.1f} mm)", flush=True)


# ------------------------------------------------------------------------------------------------
# compose: panels side by side, labelled, encoded
# ------------------------------------------------------------------------------------------------
def _font(size, bold=False):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def _label(img, title, lines, accent):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    d.rectangle([0, 0, w, 34], fill=(20, 20, 22, 190))
    d.rectangle([0, 0, 6, 34], fill=accent)
    d.text((14, 7), title, font=_font(17, True), fill=(255, 255, 255, 255))
    y = h - 8 - 24 * len(lines)
    if lines:
        d.rectangle([0, y - 6, w, h], fill=(20, 20, 22, 170))
    for txt, col in lines:
        d.text((14, y), txt, font=_font(18), fill=col)
        y += 24


ACCENT = {"teacher": (31, 92, 70, 255), "v0": (150, 156, 162, 255), "v1": (107, 63, 160, 255), "real": (191, 54, 12, 255)}   # the pages: green = exact physics, violet = what the model fills
WHITE, AMBER = (255, 255, 255, 255), (255, 193, 7, 255)


def encode(frames_dir, out, fps):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "c%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "slow", "-movflags", "+faststart", str(out)],
                   check=True)


def compose_scene(d):
    from PIL import Image
    clip = json.loads((d / "clip.json").read_text())
    out = d / "composite"
    out.mkdir(exist_ok=True)
    for f in range(clip["frames"]):
        tiles = []
        for s in clip["sources"]:
            im = Image.open(d / s / f"f{f:04d}.png").convert("RGB")
            t = f / clip["fps"]
            if s == "teacher":
                lines = [(f"t = {t:4.2f} s", WHITE)]
            elif f < clip["given"]:
                lines = [(f"t = {t:4.2f} s   given (true history)", WHITE)]
            else:
                lines = [(f"t = {t:4.2f} s   running alone", WHITE), (f"error {clip['err_mm'][s][f]:6.1f} mm", AMBER)]
            if "extra" in clip and s in clip["extra"] and (s == "teacher" or f >= clip["given"]):
                lines += [(t, WHITE) for t in clip["extra"][s][f].split("\n")]   # e.g. a rope's length, a soft body's volume
            _label(im, clip["labels"][s], lines, ACCENT[s])
            tiles.append(im)
        canvas = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (255, 255, 255))
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0)); x += t.width + 4
        canvas.save(out / f"c{f:04d}.png")
    encode(out, OUT / f"{d.name}.mp4", clip["fps"])
    Image.open(out / f"c{clip['frames'] - 1:04d}.png").save(OUT / f"{d.name}_last.jpg", quality=88)


def compose_phys101(d, slow=0.5, crop=(0, 60, 1040, 840)):
    """Real time on one clock: footage frame round(t * fps); a model's frame from its release-time
    alignment (frame given-1 is the release, earlier times hold the start pose)."""
    from PIL import Image
    import tempfile
    clip = json.loads((d / "clip.json").read_text())
    rt = np.asarray(clip["real_t"])
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", clip["video"], str(Path(tmp) / "r%04d.png")], check=True)
        real = sorted(Path(tmp).glob("r*.png"))
        out = d / "composite"
        out.mkdir(exist_ok=True)
        t_rel = min(f["t0"] for f in clip["fit"].values())                     # the earliest release
        ts = np.arange(max(t_rel - 0.2, 0.0), rt[-1] + 1e-6, 1 / FPS)          # until the real cube leaves the board
        for j, t in enumerate(ts):
            ri = min(int(round(t * clip["video_fps"])), len(real) - 1)
            im = Image.open(real[ri]).convert("RGB").crop(crop)
            tiles = []
            first = Image.open(d / clip["sources"][0] / "f0000.png")
            im = im.resize(first.size)
            s_real = float(np.interp(t, rt, np.asarray(clip["real_s"]) - clip["real_s"][0])) if t >= rt[0] else 0.0   # held at the board's end
            _label(im, LABELS["real"], [(f"t = {t:4.2f} s", WHITE), (f"travelled {100 * s_real:5.1f} cm", AMBER)], ACCENT["real"])
            tiles.append(im)
            for s in clip["sources"]:
                ft = clip["fit"][s]
                k = int(np.clip(round((t - ft["t0"]) * FPS), 0, clip["frames"] - clip["given"]))
                fi = clip["given"] - 1 + k
                tr = float(np.interp(max(t - ft["t0"], 0.0), np.arange(len(ft["curve_m"])) / FPS, ft["curve_m"]))
                si = Image.open(d / s / f"f{fi:04d}.png").convert("RGB")
                _label(si, clip["labels"][s], [(f"friction {ft['mu']:.2f} (best fit)", WHITE),
                                               (f"travelled {100 * tr:5.1f} cm   fit {ft['rms_mm']:.1f} mm", AMBER)], ACCENT[s])
                tiles.append(si)
            canvas = Image.new("RGB", (sum(x.width for x in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (255, 255, 255))
            x = 0
            for tl in tiles:
                canvas.paste(tl, (x, 0)); x += tl.width + 4
            canvas.save(out / f"c{j:04d}.png")
    encode(out, OUT / f"{d.name}.mp4", FPS * slow)
    Image.open(out / f"c{len(ts) - 1:04d}.png").save(OUT / f"{d.name}_last.jpg", quality=88)


def compose():
    for d in sorted(p for p in OUT.iterdir() if p.is_dir() and (p / "clip.json").exists()):
        kind = json.loads((d / "clip.json").read_text())["kind"]
        (compose_phys101 if kind == "phys101" else compose_scene)(d)
        print(f"[compose] {d.name}.mp4", flush=True)


def hinge_clip(kind):
    """The first test episode (shard order) whose gripper opens a hinged part of this kind."""
    models = load_models(torch.device("cuda"))
    for s in sorted((REPO / "data/datasets/v0_test").glob("shard_*.npz")):
        for e in load_shard(str(s)[:-4]):
            if (e.wd.get("hinge") or {}).get("kind") == kind and any(f["family"] == "hinge" for f in e.wd["families"]):
                e.shard = s.stem
                poses, err = scene_rollouts(e, models)
                write_scene_clip(f"scene_{kind}", e, poses, err, f"the first test episode whose gripper opens a {kind}")
                print(f"[pick] {kind}: {e.shard} world {e.w} (v0 end {err['v0'][-1]:.1f} mm, v1 end {err['v1'][-1]:.1f} mm)", flush=True)
                return


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "export":
        export(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    elif sys.argv[1] == "hinge":
        hinge_clip(sys.argv[2])
    else:
        compose()
