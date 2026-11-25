"""Watch the deformable model: a teacher episode beside the model's own rollout of it, in Blender -- cloth, rope or
soft body (the shard's kind), any cloth checkpoint (graph, qwen, qwen3d).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python; B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_render_cloth.py export <ckpt> <shard path without suffix> <episode> <t0> [frames] [tag]
    CUDA_VISIBLE_DEVICES=<gpu> $B --background --python scripts/blender_render.py -- outputs/lsim/render_cloth_model/<name>
    $L scripts/l2_render_cloth.py compose outputs/lsim/render_cloth_model/<name>

frames 0 (or "end") = to the episode's last frame. Options (anywhere after the subcommand): --out <dir> (default
outputs/lsim/render_cloth_model), --hidden (all four material slots hidden: the model reads the material from the 8
given frames; default: given, as before), --replay (export: the window rebuilt every frame from the model's own
frames, l2_small_fit.replay -- for a network that memorised a few episodes). `sheet <clip dir> [...]` tiles 6 instants of each composed clip into
<out>/contact_sheet.jpg and writes <out>/metrics.json from the clips' metrics.

The model is given the teacher's 8 frames up to t0, then runs alone (scripts/l2_cloth_eval.py's rollout: the gripper,
the vertices it holds and the static scene follow the recording); both panels show the true history first. The error
line is the mean distance of the free (not held) vertices to the teacher's. Under it: a rope's length, a cloth's area
or a soft body's volume (signed tet volume; and the share of its tets inverted) over the start's, for both panels.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/lsim/render_cloth_model"
sys.path.insert(0, str(REPO / "scripts"))
GIVEN = 8
SLOTS = ("density", "stretch", "bend", "mu")             # cloth_windows.CLOTH_PARAMS: the four slot groups, any material
BOUND_LO, BOUND_HI = np.array([-0.75, -0.65, -0.3]), np.array([0.75, 0.65, 0.6])


def _ramp_static(wd):
    if not wd.get("ramp"):
        return []
    from lsim.data.scene import RAMP, ramp_frame
    from scipy.spatial.transform import Rotation
    o_, ax, lat, n = ramp_frame(wd["ramp"])
    pc = o_ + 0.5 * RAMP["L"] * ax + 0.5 * RAMP["T"] * n
    return [{"boxes": [[[0.0, 0.0, 0.0], [0.5 * RAMP["L"], 0.5 * RAMP["W"], 0.5 * RAMP["T"]]]], "material": "furniture",
             "pos": pc.tolist(), "quat": Rotation.from_matrix(np.stack([ax, lat, n], 1)).as_quat().tolist()}]


CLOTH_GRIPPER = [[[0.0, 0.0, 0.03], [0.05, 0.02, 0.015]], [[-0.0078, 0.0, -0.02], [0.0075, 0.015, 0.035]],
                 [[0.0078, 0.0, -0.02], [0.0075, 0.015, 0.035]]]


def _measure(ep, x):
    """Per frame: rope length (m), cloth area (m^2), or soft volume (m^3) and the share of tets inverted."""
    if ep.material == "rope":
        return np.linalg.norm(np.diff(x, axis=1), axis=-1).sum(1), None
    if ep.material == "cloth":
        t = ep.tris
        return 0.5 * np.linalg.norm(np.cross(x[:, t[:, 1]] - x[:, t[:, 0]], x[:, t[:, 2]] - x[:, t[:, 0]]), axis=-1).sum(1), None
    T = ep.se.tets
    a, b, c, d = (x[:, T[:, k]] for k in range(4))
    v = np.einsum("fti,fti->ft", np.cross(b - a, c - a), d - a) / 6.0
    s0 = np.sign(v[0])                                               # the rest orientation of each tet
    return (v * s0).sum(1), ((v * s0) < 0).mean(1)


def export(ckpt, shard, e, t0, frames=60, tag="", out=None, hidden=False, replay=False):
    import torch
    from l2_render import obstacle_mesh
    from lsim.model.cloth_net import build_cloth_model
    from lsim.train.cloth_train import rollout
    from lsim.train.cloth_windows import load_shard
    out = Path(out) if out else OUT
    meta = json.loads(Path(shard).with_suffix(".json").read_text())
    ep = load_shard(shard, only_ok=False)[e]
    assert ep.w == e
    kind = ep.material
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    a = ck["args"]
    m = build_cloth_model(a).cuda().eval()
    m.load_state_dict(ck["model"])
    frames = ep.n_frames - 1 - t0 if frames in (0, "end") else int(frames)
    with torch.no_grad():
        if replay:                                                   # --replay: a window per frame, as in training
            from l2_small_fit import replay as _replay                # (l2_small_fit.py: a network that memorised a
            pred, true = _replay(m, ep, t0, frames, "cuda")           # few episodes needs it; material given)
        else:
            pred, true, _ = rollout(m, ep, t0, frames, dev="cuda", hidden=SLOTS if hidden else ())   # frames t0 + 1 .. t0 + frames
    hist = ep.x(np.arange(t0 - GIVEN + 1, t0 + 1))                    # the given frames, t0 - 7 .. t0
    teach = np.concatenate([hist, true])
    model = np.concatenate([hist, pred])
    free = ~ep.held[t0 - GIVEN + 1:t0 + frames + 1]                   # held in the step into the frame: not scored
    err = [0.0] * GIVEN + [float(1e3 * np.linalg.norm(model[f] - teach[f], axis=-1)[free[f - 1]].mean()) for f in range(GIVEN, len(teach))]
    wd = meta["worlds"][e]
    fr = np.arange(t0 - GIVEN + 1, t0 + frames + 1)
    n_obs = len(wd["obstacles"])
    # what each kind draws: the body, the gripper as the boxes its teacher collides, the obstacles, the plank
    if kind == "cloth":
        from lsim.teacher.cloth_teacher import grid
        c = wd["cloth"]
        v, t = grid(c["nx"], c["ny"], c["h"])
        bodies = [{"kind": "cloth", "label": "cloth", "rest": teach[0].tolist(), "tris": t.tolist(), "uv": v[:, :2].tolist(),
                   "color": [0.30, 0.45, 0.70] if c["w"] < 0.15 else ([0.80, 0.42, 0.30] if c["w"] < 0.3 else [0.45, 0.55, 0.35])},
                  {"kind": "boxes", "label": "gripper", "material": "gripper", "boxes": CLOTH_GRIPPER}]
        n_grip = 1
        grip_poses = ep.grip[fr][:, None]
        title = (f"{c['nx']} x {c['ny']} cm, {1000 * c['w']:.0f} g/m2, Y {c['Y']:,.0f} N/m, bends over "
                 f"{100 * c['bending_length']:.1f} cm, mu {c['mu']:.2f}")
        fams = ", ".join(p["family"].replace("_", " ") for p in wd["families"]) or "none"
        story = f"starts {wd['init']}: {fams} | {title}"
    elif kind == "rope":
        import l1_render_rope as R
        bodies = [{"kind": "rope", "label": "rope", "rest": teach[0].tolist(), "radius": max(ep.radius, R.DRAW_MIN_R)},
                  {"kind": "boxes", "label": "gripper", "material": "gripper", "boxes": CLOTH_GRIPPER}]
        n_grip = 1
        grip_poses = ep.grip[fr][:, None]
        title, story = R.describe(wd)
    else:
        import l1_render_soft as S
        bodies = [{"kind": "soft", "label": "soft", "rest": teach[0].tolist(), "tris": ep.surf_tris.tolist(),
                   "look": wd["soft"]["params"].get("category", "wide")}]
        gb, grip_poses = S.gripper_bodies(ep.grip[fr], ep.grip_w[fr])
        bodies += gb
        n_grip = 3
        title, story = S._label(wd)
    poses = np.zeros((len(fr), 1 + n_grip + n_obs, 7), np.float32)
    poses[:, :, 6] = 1.0
    poses[:, 1:1 + n_grip] = grip_poses
    for k, o in enumerate(wd["obstacles"]):
        bodies.append(obstacle_mesh(o["name"]))
        poses[:, 1 + n_grip + k, :3] = o["pos"]
        poses[:, 1 + n_grip + k, 3:] = o["quat"]
    # the camera: the kind's own framing over the teacher and the model while the model is within 10 cm of it (a
    # diverging model leaves the frame instead of shrinking the scene)
    near = np.asarray(err) < 100.0
    both = np.concatenate([teach, np.where(near[:, None, None], np.clip(model, BOUND_LO, BOUND_HI), teach)], axis=1)
    if kind == "rope":
        eye, ctr = R.camera(both, ep.grip[fr])
    elif kind == "soft":
        eye, ctr = S.camera(both, grip_poses[:, 1:3])
    else:
        pts = both.reshape(-1, 3)
        lo = np.maximum(np.minimum(pts.min(0), [-0.3, -0.25, 0.0]), BOUND_LO)
        hi = np.minimum(np.maximum(pts.max(0), [0.3, 0.25, 0.1]), BOUND_HI)
        ctr = 0.5 * (lo + hi)
        r = max(0.35, 0.62 * float(np.linalg.norm(hi - lo)))
        az, el = np.radians(-60.0), np.radians(36.0)
        eye = ctr + r / np.tan(np.radians(19.0)) * 0.55 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    # the shape measures, over frame t0's (the last given)
    mt, it = _measure(ep, teach.astype(np.float64))
    mm, im = _measure(ep, model.astype(np.float64))
    word = {"rope": "length", "cloth": "area", "soft": "volume"}[kind]
    rt, rm = mt / mt[GIVEN - 1], mm / mt[GIVEN - 1]
    extra = {"teacher": [f"{story}\n{word} {rt[f]:.3f} x" for f in range(len(fr))],
             "model": [f"{word} {rm[f]:.3f} x" + (f"   tets inverted {100 * im[f]:.1f}%" if im is not None else "") for f in range(len(fr))]}
    f1 = min(GIVEN - 1 + 30, len(err) - 1)
    metrics = {"kind": kind, "shard": str(shard), "episode": e, "t0": t0, "frames_alone": frames, "hidden": hidden,
               "err_1s_mm": err[f1], "err_end_mm": err[-1], "err_median_alone_mm": float(np.median(err[GIVEN:])),
               f"{word}_end_teacher": float(rt[-1]), f"{word}_end_model": float(rm[-1]),
               f"{word}_model_range": [float(rm[GIVEN:].min()), float(rm[GIVEN:].max())]}
    if im is not None:
        metrics["tets_inverted_end_model"] = float(im[-1])
        metrics["tets_inverted_max_model"] = float(im[GIVEN:].max())
    name = ("" if kind == "cloth" else f"{Path(shard).parent.name}_") + f"{Path(shard).name}_e{e}_t{t0}" + (f"_{tag}" if tag else "")
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    clip = {"kind": kind, "fps": 30, "frames": len(fr), "given": GIVEN, "sources": ["teacher", "model"],
            "labels": {"teacher": "teacher (Newton, 1,000+ substeps a frame)" if kind == "cloth" else f"teacher: {title}",
                       "model": f"learned model, alone after {GIVEN} frames" + (", material hidden" if hidden else "")},
            "story": story, "err_mm": {"model": err}, "extra": extra, "metrics": metrics,
            "camera": {"eye": np.asarray(eye).tolist(), "target": np.asarray(ctr).tolist(), "fov": 38.0},
            "static": _ramp_static(wd), "ghost": [], "bodies": bodies}
    (d / "clip.json").write_text(json.dumps(clip))
    np.savez(d / "poses.npz", teacher=poses, model=poses, **{f"teacher_{kind}0": teach.astype(np.float32),
                                                            f"model_{kind}0": model.astype(np.float32)})
    print(f"[export] {d}: {kind}, error at 1 s / end alone {err[f1]:.1f} / {err[-1]:.1f} mm; {word} at the end teacher "
          f"{rt[-1]:.3f} x, model {rm[-1]:.3f} x" + (f", model tets inverted {100 * im[-1]:.1f}%" if im is not None else ""),
          flush=True)
    return d


def compose(d, out=None):
    import l2_render as r
    r.OUT = Path(out) if out else OUT
    r.ACCENT["model"] = (107, 63, 160, 255)
    r.compose_scene(Path(d))
    print(f"[compose] {r.OUT / (Path(d).name + '.mp4')}", flush=True)


def sheet(dirs, out=None, n=6):
    """6 instants of each composed clip in a row, half size; and the clips' metrics in one json."""
    from PIL import Image
    out = Path(out) if out else OUT
    rows, mets = [], []
    for d in map(Path, dirs):
        fs = sorted((d / "composite").glob("c*.png"))
        rows.append([Image.open(fs[int(round(k * (len(fs) - 1) / (n - 1)))]).convert("RGB") for k in range(n)])
        mets.append(dict(json.loads((d / "clip.json").read_text())["metrics"], clip=d.name))
    w, h = rows[0][0].width // 2, rows[0][0].height // 2
    canvas = Image.new("RGB", (n * w + (n - 1) * 3, len(rows) * h + (len(rows) - 1) * 3), (255, 255, 255))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            canvas.paste(im.resize((w, h)), (j * (w + 3), i * (h + 3)))
    canvas.save(out / "contact_sheet.jpg", quality=88)
    (out / "metrics.json").write_text(json.dumps(mets, indent=1))
    print(f"[sheet] {out / 'contact_sheet.jpg'}, {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    out, hidden = None, False
    if "--out" in args:
        k = args.index("--out")
        out = args[k + 1]
        args = args[:k] + args[k + 2:]
    if "--hidden" in args:
        args.remove("--hidden")
        hidden = True
    replay = "--replay" in args
    if replay:
        args.remove("--replay")
    if args[0] == "export":
        fr = args[5] if len(args) > 5 else 60
        export(args[1], args[2], int(args[3]), int(args[4]), fr if fr in ("end",) else int(fr), args[6] if len(args) > 6 else "",
               out=out, hidden=hidden, replay=replay)
    elif args[0] == "sheet":
        sheet(args[1:], out=out)
    else:
        compose(args[1], out=out)
