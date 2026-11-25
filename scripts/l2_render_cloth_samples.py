"""Watch a generative cloth model: the teacher beside two sampled futures of the same start, in Blender.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python; B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_render_cloth_samples.py export <ckpt> <shard path without suffix> <episode> <t0> [frames] [tag] [seeds]
    CUDA_VISIBLE_DEVICES=<gpu> $B --background --python scripts/blender_render.py -- outputs/lsim/render_cloth_model/<name>
    $L scripts/l2_render_cloth_samples.py compose outputs/lsim/render_cloth_model/<name>

As scripts/l2_render_cloth.py (whose scene building this copies; that script is left as it is), with two model panels:
the same checkpoint rolled out twice from the same 8 true frames, each with its own fixed noise (seeds, default
"1,2"; lsim.model.cloth_qwen.set_noise). A deterministic checkpoint gives two identical panels.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/lsim/render_cloth_model"
sys.path.insert(0, str(REPO / "scripts"))
GIVEN = 8
SOURCES = ("sample_a", "sample_b")


def export(ckpt, shard, e, t0, frames=60, tag="", seeds="1,2"):
    import torch
    from l1_render_cloth import load
    from l2_render import gso_mesh
    from lsim.model.cloth_net import build_cloth_model
    from lsim.model.cloth_qwen import set_noise
    from lsim.teacher.cloth_teacher import grid
    from lsim.train.cloth_train import rollout
    from lsim.train.cloth_windows import load_cloth_shard
    npz, meta = load(shard)
    ep = load_cloth_shard(shard, only_ok=False)[e]
    ck = torch.load(ckpt, map_location="cuda")
    m = build_cloth_model(ck["args"]).cuda().eval()
    m.load_state_dict(ck["model"])
    hist = ep.x(np.arange(t0 - GIVEN + 1, t0 + 1))                    # the given frames, t0 - 7 .. t0
    free = ~ep.held[t0 - GIVEN + 1:t0 + frames + 1]
    runs, err = {}, {}
    for src, seed in zip(SOURCES, [int(s) for s in seeds.split(",")]):
        set_noise(m, seed=seed)
        with torch.no_grad():
            pred, true, _ = rollout(m, ep, t0, frames, dev="cuda")   # frames t0 + 1 .. t0 + frames
        teach = np.concatenate([hist, true])
        runs[src] = np.concatenate([hist, pred])
        err[src] = [0.0] * GIVEN + [float(1e3 * np.linalg.norm(runs[src][f] - teach[f], axis=-1)[free[f - 1]].mean())
                                    for f in range(GIVEN, len(teach))]
    wd = meta["worlds"][e]
    c = wd["cloth"]
    v, t = grid(c["nx"], c["ny"], c["h"])
    fr = np.arange(t0 - GIVEN + 1, t0 + frames + 1)
    grip = npz["grip_q"][fr, e]
    bodies = [{"kind": "cloth", "label": "cloth", "rest": teach[0].tolist(), "tris": t.tolist(), "uv": v[:, :2].tolist(),
               "color": [0.30, 0.45, 0.70] if c["w"] < 0.15 else ([0.80, 0.42, 0.30] if c["w"] < 0.3 else [0.45, 0.55, 0.35])},
              {"kind": "boxes", "label": "gripper", "material": "gripper",
               "boxes": [[[0.0, 0.0, 0.03], [0.05, 0.02, 0.015]], [[-0.0078, 0.0, -0.02], [0.0075, 0.015, 0.035]],
                         [[0.0078, 0.0, -0.02], [0.0075, 0.015, 0.035]]]}]
    poses = np.zeros((len(fr), 2 + len(wd["obstacles"]), 7), np.float32)
    poses[:, :, 6] = 1.0
    poses[:, 1] = grip
    for k, o in enumerate(wd["obstacles"]):
        bodies.append(gso_mesh(o["name"]))
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
    pts = np.concatenate([teach.reshape(-1, 3)] + [r.reshape(-1, 3) for r in runs.values()])
    lo = np.maximum(np.minimum(pts.min(0), [-0.3, -0.25, 0.0]), [-0.75, -0.65, -0.3])
    hi = np.minimum(np.maximum(pts.max(0), [0.3, 0.25, 0.1]), [0.75, 0.65, 0.6])
    ctr = 0.5 * (lo + hi)
    r = max(0.35, 0.62 * float(np.linalg.norm(hi - lo)))
    az, el = np.radians(-60.0), np.radians(36.0)
    eye = ctr + r / np.tan(np.radians(19.0)) * 0.55 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fams = ", ".join(p["family"].replace("_", " ") for p in wd["families"]) or "none"
    mat = (f"{c['nx']} x {c['ny']} cm, {1000 * c['w']:.0f} g/m2, Y {c['Y']:,.0f} N/m, bends over {100 * c['bending_length']:.1f} cm, "
           f"mu {c['mu']:.2f}")
    name = f"{Path(shard).name}_e{e}_t{t0}" + (f"_{tag}" if tag else "")
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    clip = {"kind": "cloth", "fps": 30, "frames": len(fr), "given": GIVEN, "sources": ["teacher", *SOURCES],
            "labels": {"teacher": "teacher (Newton, 1,000+ substeps a frame)",
                       "sample_a": f"learned model, sample 1, alone after {GIVEN} frames",
                       "sample_b": f"learned model, sample 2, alone after {GIVEN} frames"},
            "story": f"starts {wd['init']}: {fams} | {mat}", "err_mm": err,
            "camera": {"eye": eye.tolist(), "target": ctr.tolist(), "fov": 38.0}, "static": static, "ghost": [], "bodies": bodies}
    (d / "clip.json").write_text(json.dumps(clip))
    np.savez(d / "poses.npz", teacher=poses, teacher_cloth0=teach.astype(np.float32),
             **{s: poses for s in SOURCES}, **{f"{s}_cloth0": runs[s].astype(np.float32) for s in SOURCES})
    for s in SOURCES:
        print(f"[export] {d} {s}: error at 10 / 30 / {frames} frames alone {err[s][GIVEN + 9]:.1f} / "
              f"{err[s][min(GIVEN + 29, len(err[s]) - 1)]:.1f} / {err[s][-1]:.1f} mm", flush=True)
    print(f"[export] samples apart at the end: {1e3 * np.linalg.norm(runs[SOURCES[0]][-1] - runs[SOURCES[1]][-1], axis=-1).mean():.1f} mm",
          flush=True)


def compose(d):
    import l2_render as r
    r.OUT = OUT
    r.ACCENT["sample_a"] = (107, 63, 160, 255)
    r.ACCENT["sample_b"] = (150, 90, 190, 255)
    r.compose_scene(Path(d))
    print(f"[compose] {OUT / (Path(d).name + '.mp4')}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 60,
               sys.argv[7] if len(sys.argv) > 7 else "", sys.argv[8] if len(sys.argv) > 8 else "1,2")
    else:
        compose(sys.argv[2])
