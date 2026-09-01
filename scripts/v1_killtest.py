"""V1 -- the kill test. Does surprise depend on theta?

This is rule 1 applied to a supervision signal. The judge failed it: rho = -0.02 with
density, -0.07 with damping. Surprise gets exactly the same test, and the same verdict
either way -- a flat result is written up as a flat result and V2 is not built.

Staged, because the stages need different interpreters (the house pattern, cf.
scripts/j2r_calibration.py):

  sim     warp env      rollouts -> LAB/scene_poses.json + scene.json + meta.json
  render  blender       LAB/sim_<key>@<view>/f####.png
  clip    warp env      PNG -> one square mp4 per scene, cropped around the object
  score   cosmos env    every clip -> docs/results/v1_surprise.json
  report  warp env      plots + the verdict

  W=/home/jooyeolyun/anaconda3/envs/warp/bin/python
  C=/home/jooyeolyun/anaconda3/envs/cosmos/bin/python
  B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
  G=$($W src/gpu.py | tail -1 | sed 's/.*-> //')
  CUDA_VISIBLE_DEVICES=$G $W scripts/v1_killtest.py sim
  CUDA_VISIBLE_DEVICES=$G LAB=outputs/scene/v1 $B -b --python scripts/blender_render_scene.py
  $W scripts/v1_killtest.py clip
  CUDA_VISIBLE_DEVICES=$G $C scripts/v1_killtest.py score

DESIGN, all of it forced by V0:

* ONE object, one parameter at a time. Magnitudes are confounded ACROSS objects and
  clean WITHIN one -- the same rebound observable gives rho -0.186 across 14 objects
  and -0.995 within one. V0 reproduced that confound in the surprise signal itself
  (per-clip rho +0.95 / +0.83, pooled -0.01), so every comparison here holds the
  object, the framing and the excitation fixed and moves only theta.

* Motion is maximised. V0 showed that slowing a clip destroys the harness's ability
  to discriminate it at all. So: a centre drop rather than the edge drop the damping
  probe uses (the object stays in frame and bounces hard), and a square crop around
  the trajectory rather than the full table.

* Two controls, and they are not optional.
  - GROSS VIOLATIONS must fire. Built by corrupting the reference pose track before
    rendering, so appearance is identical to the legal clips and only the motion
    differs. If these do not separate, the harness is broken and no theta curve
    means anything.
  - The MASS SWEEP must stay flat. A body slides v^2/(2 mu g), independent of mass:
    that is the `collide` lesson, reused here as an instrument check. Surprise moving
    with mass in a mass-invariant scene is surprise reading render noise.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO))

LAB = Path(os.environ.get("LAB", "outputs/scene/v1"))
FLOOR = Path("outputs/scene/v1_floor")
RESULTS = Path("docs/results/v1_surprise.json")

GZ = 0.706
FPS, SUBSTEPS, NF = 24.0, 60, 64
TABLE = (0.79, 0.49)
OBJ, SCALE, MASS = "rigid/wooden_bowl_01", 0.62, 0.40
VIEW = "a"

# CAM_PULL must be given to BOTH renderers -- Blender's camera and the Python one the crop
# box is projected through. They were out of step in the first pilot and the box stopped
# covering the object. 0.65 is the closest pull that still contains the whole drop: at 0.45
# only 91.5% of the object's projected points stay inside the frame, and a clipped object
# reads as occlusion rather than as physics.
#
# Note what zooming does NOT buy. Cropping to the trajectory normalises the zoom away --
# the box grows in exact proportion, so motion-in-crop is fixed by the object's size
# relative to the TRAJECTORY EXTENT, not by the camera. The lever that works is a shorter
# trajectory, which is why the drop height matters and the camera distance mostly does not.
CAM_PULL = 0.65
CROP_PAD = 0.18

# An 8 cm drop, not the 30 cm one the damping probe uses. Measured on a pilot: at 30 cm
# the fall dominates the crop box and the clip reads motion 0.5-1.3; at 8 cm the same
# sweep reads 1.5-4.1. The bounces are what damping controls, and they should fill the
# clip rather than share it with a long fall.
BOUNCE_Z = 0.08
PIVOT = np.array([-0.05, -0.06, GZ])

# Sweeps: 9 points, log-spaced, geometric centre = the reference. theta_ref is "the
# value used in the reference render", NOT a handbook number (rule 2).
CD = np.geomspace(400.0, 12800.0, 9)
CD_REF = float(np.sqrt(400.0 * 12800.0))
# Every one of these four numbers is pinned by FRAMING, checked by projecting the whole
# trajectory through the render camera before spending any Cycles time:
#   * at 40 deg the mu = 0.1 case slides 2.5 m off a 1.58 m table;
#   * at 30 deg from x = -0.60 the bowl STARTS half outside the left edge (uv x = -50);
#   * at 26 deg the table's down-slope end swings below the bottom of the frame, so the
#     low-mu cases slide out of view -- it is the TABLE's rotation that runs out of frame
#     first, not the object's travel;
#   * at mu = 0.5 the bowl travels 0.4 cm in the whole clip, which is a static clip, and
#     V0 showed the harness cannot read those.
# 24 deg from x = -0.42 over mu 0.10-0.35 keeps every theta in frame (>= 99.7% of the
# object's projected points) with a 30x range in slide distance, 56.8 cm down to 1.9 cm.
MU = np.geomspace(0.10, 0.35, 9)
MU_REF = float(np.sqrt(0.10 * 0.35))
MASSES = np.geomspace(0.05, 1.60, 5)          # the null control
TILT0, TILT1 = 0.0, 24.0
TILT_X0 = -0.42

VIOLATIONS = ("teleport", "through", "vanish", "freeze", "rewind")
N_FLOOR = 5


# ------------------------------------------------------------------------ sim helpers

def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz]


def quat_y(deg):
    h = np.radians(deg) / 2.0
    return [0.0, float(np.sin(h)), 0.0, float(np.cos(h))]


def build(cd=3200.0, mu=0.35, mass=MASS, pos=(0., 0., GZ + BOUNCE_Z), vel=(0., 0., 0.),
          rot=None, gseq=None):
    from src.sim.mesh_scene import MeshProbeScene
    s = MeshProbeScene([OBJ], [list(pos)], [list(vel)], rot0=None if rot is None else [rot],
                       masses=[float(mass)], ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                       n_steps=NF * SUBSTEPS, k=2500.0, cd=[float(cd)], mu=[float(mu)],
                       mesh_scale=[SCALE], faces=900, ground_mu=0.45, ground_cd=3000.0,
                       table=TABLE)
    s.calibrate_stiffness()
    if gseq is not None:
        s.gravity_seq = gseq
    return s


def track(s, tilt_seq=None, n_pts=48):
    """Per-frame poses plus a handful of world points, for the crop box.

    The crop must follow the object through the RENDERED frame, so for a tilt scene the
    points are carried into the tilted-table frame exactly as blender_render_scene.py
    rotates the table -- about PIVOT, by that frame's own angle.
    """
    s.rollout()
    P, Q = s.positions(SUBSTEPS), s.rotations(SUBSTEPS)
    if not (np.isfinite(P).all() and np.isfinite(Q).all()):
        return None
    # rule 3: an explicit solver reaches 1e5 m long before it reaches inf
    if np.abs(P).max() > 50.0:
        return None
    # s.meshes holds wp.Mesh; s.local holds the COM-centred vertices the solver uses,
    # which is exactly the body-frame point set the crop box wants.
    V = s.local.numpy().astype(float)
    idx = np.linspace(0, len(V) - 1, min(n_pts, len(V))).astype(int)
    C = s.coms[0]
    seq, pts = [], []
    for t in range(len(P)):
        R = quat_to_mat(Q[t, 0])
        loc = P[t, 0] - R @ C
        w = (R @ V[idx].T).T + P[t, 0]
        if tilt_seq is not None:
            th = np.radians(float(tilt_seq[min(t, len(tilt_seq) - 1)]))
            Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
            w = PIVOT + (Ry @ (w - PIVOT).T).T
        seq.append({"loc": [float(v) for v in loc],
                    "mat": [[float(v) for v in r] for r in R]})
        pts.append(w.tolist())
    return seq, pts


# ------------------------------------------------------------------------------- sim

def corrupt(seq, kind):
    """Author a gross violation on top of a legal pose track. Same asset, same
    lighting, same framing -- only the motion is impossible."""
    out = [dict(loc=list(p["loc"]), mat=[list(r) for r in p["mat"]]) for p in seq]
    n = len(out)
    if kind == "teleport":                       # 15 cm jump, mid-clip, no transit
        for t in range(n // 2, n):
            out[t]["loc"][0] += 0.15
    elif kind == "through":                      # sinks through a solid table
        for t in range(n // 3, n):
            out[t]["loc"][2] -= 0.30 * (t - n // 3) / (n - n // 3)
    elif kind == "vanish":                       # gone for 8 frames, then back
        for t in range(n // 2, n // 2 + 8):
            out[t]["loc"][1] += 4.0
    elif kind == "freeze":                       # stops dead in mid-air
        for t in range(n // 3, n):
            out[t] = dict(loc=list(out[n // 3]["loc"]), mat=[list(r) for r in out[n // 3]["mat"]])
    elif kind == "rewind":                       # bounces higher each time
        out = out[::-1]
    else:
        raise ValueError(kind)
    return out


def do_sim(args):
    import warp as wp
    from src.sim.tilt_probe import ramp_gravity_seq

    LAB.mkdir(parents=True, exist_ok=True)
    wp.init()
    scenes, meta, spec = {}, {}, []

    def add(key, seq, pts, tilt=0.0, **info):
        scenes[key] = {"tilt_deg": tilt, "objects": {"subject": seq}}
        meta[key] = pts
        spec.append({"key": key, **info})

    with wp.ScopedDevice("cuda:0"):
        ramp = ramp_gravity_seq(TILT0, TILT1, NF, SUBSTEPS, settle_frames=6)
        tilt_seq = list(np.linspace(TILT0, TILT1, NF))

        if "bounce" in args.probes:
            print("bounce: dropped 30 cm in the middle of the table, cd swept")
            for cd in CD:
                r = track(build(cd=cd))
                if r is None:
                    print(f"  cd={cd:8.0f}  DIVERGED"); continue
                add(f"bounce_cd{cd:07.1f}", *r, probe="bounce", param="cd",
                    value=float(cd), ref=CD_REF)
                z = np.array([p["loc"][2] for p in r[0]])
                print(f"  cd={cd:8.0f}  rebound {100*(z[int(np.argmin(z)):].max()-z.min()):5.2f} cm")

        if "settle" in args.probes:
            print("settle: tipped onto its dome and released, cd swept")
            for cd in CD:
                r = track(build(cd=cd, pos=(0., 0., GZ + 0.075),
                                rot=quat_mul(quat_y(180.0), quat_y(22.0))))
                if r is None:
                    print(f"  cd={cd:8.0f}  DIVERGED"); continue
                add(f"settle_cd{cd:07.1f}", *r, probe="settle", param="cd",
                    value=float(cd), ref=CD_REF)
                print(f"  cd={cd:8.0f}  ok")

        if "tilt" in args.probes:
            print(f"tilt: continuous ramp 0->{TILT1:.0f} deg, mu swept")
            for mu in MU:
                r = track(build(mu=mu, pos=(TILT_X0, 0.0, GZ + 0.02), gseq=ramp),
                          tilt_seq=tilt_seq)
                if r is None:
                    print(f"  mu={mu:5.3f}  DIVERGED"); continue
                add(f"tilt_mu{mu:05.3f}", *r, tilt=tilt_seq, probe="tilt", param="mu",
                    value=float(mu), ref=MU_REF)
                dx = r[0][-1]["loc"][0] - r[0][0]["loc"][0]
                print(f"  mu={mu:5.3f}  slid {100*dx:6.2f} cm")

            print("mass null control: same ramp, mass swept (slip is mass-independent)")
            for mk in MASSES:
                r = track(build(mu=MU_REF, mass=mk, pos=(TILT_X0, 0.0, GZ + 0.02), gseq=ramp),
                          tilt_seq=tilt_seq)
                if r is None:
                    print(f"  m={mk:5.3f}  DIVERGED"); continue
                add(f"null_m{mk:05.3f}", *r, tilt=tilt_seq, probe="null", param="mass",
                    value=float(mk), ref=float(np.sqrt(MASSES[0] * MASSES[-1])))
                dx = r[0][-1]["loc"][0] - r[0][0]["loc"][0]
                print(f"  m={mk:5.3f}  slid {100*dx:6.2f} cm")

        if "control" in args.probes:
            print("violation controls, authored on the reference bounce track")
            ref = track(build(cd=CD_REF))
            for kind in VIOLATIONS:
                add(f"viol_{kind}", corrupt(ref[0], kind), ref[1],
                    probe="violation", param=kind, value=0.0, ref=0.0)
                print(f"  {kind}")

    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene.json").write_text(json.dumps({
        "ground_z": GZ,
        "objects": {"subject": {"asset": OBJ, "scale": SCALE, "rot_z": 0.0,
                                "pos": [0.0, 0.0, GZ + 0.30]}}}, indent=1))
    (LAB / "meta.json").write_text(json.dumps(meta))
    (LAB / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"\nwrote {LAB}/scene_poses.json  ({len(scenes)} scenes)")

    # the noise floor gets its own LABs: identical physics, one nuisance changed
    # the floor is measured on the REFERENCE clip, not on an extreme of the sweep
    bkeys = [r for r in spec if r["probe"] == "bounce"]
    ref_key = min(bkeys, key=lambda r: abs(np.log(r["value"] / CD_REF)))["key"] if bkeys \
        else next(iter(scenes))
    for tag in [f"seed{i}" for i in range(N_FLOOR)] + [f"jit{i}" for i in range(N_FLOOR)]:
        d = FLOOR / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / "scene_poses.json").write_text(json.dumps({ref_key: scenes[ref_key]}))
        (d / "scene.json").write_text((LAB / "scene.json").read_text())
        (d / "meta.json").write_text(json.dumps({ref_key: meta[ref_key]}))
    print(f"wrote {2 * N_FLOOR} noise-floor labs under {FLOOR} (reference scene {ref_key})")
    return 0


# ------------------------------------------------------------------------------ clip

def do_clip(args):
    """PNG -> one square mp4 per scene, cropped around the object's whole trajectory.

    Square because the model's grid is square and letterboxing spends a fifth of the
    tokens on black bars; tight because V0 showed discrimination follows motion, and a
    crop that follows the trajectory magnifies it. Static across the clip -- a box that
    tracks the object would remove the very translation being measured.
    """
    import imageio.v2 as imageio
    from src.render.crop import crop_clip
    from src.render.views import VIEWS

    labs = [LAB] + sorted(p for p in FLOOR.glob("*") if p.is_dir()) if args.floor else [LAB]
    cam = pulled_camera()
    W, H = VIEWS[VIEW]["width"], VIEWS[VIEW]["height"]
    # boxes come from the MAIN lab for every lab, so a floor re-render of the reference
    # scene is cropped exactly as the reference itself was
    boxes = shared_boxes(cam, json.loads((LAB / "meta.json").read_text()),
                         spec_of(LAB), W, H)
    n_out = 0
    for lab in labs:
        meta = json.loads((lab / "meta.json").read_text())
        (lab / "clips").mkdir(exist_ok=True)
        for key, pts in meta.items():
            d = lab / f"sim_{key}@{VIEW}"
            # positions(SUBSTEPS) yields NF+1 samples (it includes t=0); take the first
            # NF so the clip needs no temporal resampling before the model sees it.
            ps = sorted(d.glob("f*.png"))[:NF]
            if len(ps) < NF:
                print(f"  {lab.name}/{key}: {len(ps)}/{NF} frames, skipped"); continue
            frames = np.stack([imageio.imread(p)[..., :3] for p in ps])
            box = boxes.get(key, boxes["__floor__"])
            c = crop_clip(frames, box)
            dst = lab / "clips" / f"{key}.mp4"
            w = imageio.get_writer(str(dst), fps=int(FPS), codec="libx264", quality=9,
                                   macro_block_size=1)
            for f in c:
                w.append_data(f)
            w.close()
            mo = float(np.abs(np.diff(c.astype(np.float32), axis=0)).mean())
            print(f"  {lab.name:>12}/{key:<22} box {box[2]-box[0]:>3}px  motion {mo:5.2f}")
            n_out += 1
    print(f"\n{n_out} clips")
    return 0


def spec_of(lab):
    f = lab / "spec.json"
    return {r["key"]: r for r in json.loads(f.read_text())} if f.exists() else {}


# Which probe's box a set of clips must borrow. Violations are authored ON the reference
# bounce track and the floor labs are re-renders of it, so all three have to be framed
# identically or the comparison measures the crop. The mass null runs the tilt ramp at the
# reference mu, so it shares the tilt box for the same reason.
BOX_FROM = {"violation": "bounce", "floor": "bounce", "null": "tilt"}


def shared_boxes(cam, meta, spec, W, H):
    """ONE crop box per probe, taken from that probe's REFERENCE theta.

    This is not a tidiness point, it is the difference between a measurement and an
    artefact. A box computed per clip grows with whatever the object did, so a high-mu
    clip that barely moves would be framed tightly and a low-mu clip that slid half a
    metre would be framed wide -- and surprise would then be reading the framing, which
    varies with theta, rather than the motion. Every clip of one probe gets the identical
    box, and a theta whose motion leaves that box leaves it, exactly as a fixed camera
    would show.
    """
    by_probe = {}
    for key, r in spec.items():
        by_probe.setdefault(r["probe"], []).append(key)

    per_probe = {}
    for probe, keys in by_probe.items():
        if probe in BOX_FROM:
            continue
        ref_key = min(keys, key=lambda k: abs(np.log(max(spec[k]["value"], 1e-12)
                                                     / max(spec[k]["ref"], 1e-12))))
        per_probe[probe] = square_box(cam, np.asarray(meta[ref_key], float), W, H,
                                      pad=CROP_PAD)
        print(f"  probe {probe:<10} box {per_probe[probe]}  from {ref_key}")
    for probe, src in BOX_FROM.items():
        if src in per_probe:
            per_probe[probe] = per_probe[src]
            print(f"  probe {probe:<10} box {per_probe[probe]}  borrowed from {src}")

    out = {k: per_probe[r["probe"]] for k, r in spec.items() if r["probe"] in per_probe}
    out["__floor__"] = per_probe.get("floor")
    return out


def pulled_camera():
    """The same camera Blender uses, CAM_PULL included."""
    from src.render.camera import Camera
    from src.render.views import VIEWS
    vc = dict(VIEWS[VIEW])
    eye, tgt = np.asarray(vc["eye"], float), np.asarray(vc["target"], float)
    vc["eye"] = (tgt + (eye - tgt) * CAM_PULL).tolist()
    return Camera(vc)


def square_box(cam, pts, W, H, pad=CROP_PAD, min_px=160):
    """Square crop covering the object over the whole clip, padded, clamped to frame."""
    from src.render.crop import project_points
    uv, ok = project_points(cam, pts.reshape(-1, 3))
    uv = uv[ok]
    x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(max(x1 - x0, y1 - y0) * (1 + 2 * pad), min_px)
    side = min(side, min(W, H))
    cx = min(max(cx, side / 2), W - side / 2)
    cy = min(max(cy, side / 2), H - side / 2)
    s = int(side) // 2 * 2
    return (int(cx - s / 2), int(cy - s / 2), int(cx - s / 2) + s, int(cy - s / 2) + s)


# ----------------------------------------------------------------------------- score

def do_score(args):
    """Score every clip, and score a frame-shuffled copy of the SAME clip beside it.

    The shuffled copy is the answer to the confound this sweep would otherwise have. A
    theta that damps harder also MOVES less (motion 4.05 -> 1.54 across the cd sweep), and
    V0 showed surprise tracks motion within a scene -- so a monotone surprise-vs-theta
    curve could be reading "how much moved" rather than "how implausible it was".

    Shuffling the frames of one clip keeps its appearance, its object, its framing and
    roughly its motion budget, and destroys only the temporal coherence. So
    surprise(real) / surprise(shuffled) is normalised by that clip's own difficulty: it
    asks how much the ORDER of the frames helped, which is the only part of the reading
    that can be about physics. Costs one extra encode per clip.
    """
    from src.surprise import clips as C
    from src.surprise.cache import ScoreCache
    from src.surprise.vjepa import SurpriseModel

    m = SurpriseModel(device="cuda:0")
    cache = ScoreCache({**m.config(), "control": "shuffled_self"})
    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    rows = []
    labs = [LAB] + sorted(p for p in FLOOR.glob("*") if p.is_dir())
    for lab in labs:
        for clip in sorted((lab / "clips").glob("*.mp4")):
            key = clip.stem
            hit = cache.get(clip)
            if hit is None:
                frames, _ = C.prepare(clip)
                sc = m.score(frames)
                sc["motion"] = C.motion(frames)
                rng = np.random.default_rng(0)          # same permutation for every clip
                shuf = frames[rng.permutation(len(frames))]
                sc["shuffled"] = m.score(shuf, profile=None)["causal"]["mean"]
                sc["motion_shuffled"] = C.motion(shuf)
                hit = cache.put(clip, sc)
            info = dict(spec.get(key, {"key": key, "probe": "floor", "param": lab.name,
                                       "value": 0.0, "ref": 0.0}))
            if lab is not LAB:
                info = dict(info, probe="floor", param=lab.name)
            rows.append({**info, "lab": lab.name, "clip": str(clip),
                         "surprise": sc_mean(hit), "motion": hit["motion"],
                         "shuffled": hit["shuffled"],
                         "ratio": sc_mean(hit) / hit["shuffled"],
                         "curve": hit["causal"]["curve"]})
            print(f"  {lab.name:>12}/{key:<22} surprise {sc_mean(hit):.5f}"
                  f"  shuffled {hit['shuffled']:.5f}"
                  f"  ratio {rows[-1]['ratio']:.4f}  motion {hit['motion']:5.2f}", flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"config": m.config(), "rows": rows}, indent=1))
    print(f"\nwrote {RESULTS}  ({len(rows)} clips)")
    return 0


def sc_mean(hit):
    return float(hit["causal"]["mean"])


MODES = {"sim": do_sim, "clip": do_clip, "score": do_score}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--probes", default="bounce,settle,tilt,control")
    ap.add_argument("--floor", action="store_true", default=True)
    a = ap.parse_args()
    a.probes = a.probes.split(",")
    raise SystemExit(MODES[a.mode](a))
