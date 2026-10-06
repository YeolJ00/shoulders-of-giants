"""G-A -- does the classifier's decision boundary sit where the PHYSICS threshold sits?

The real kill test. G-0 showed the head is confident on our renders and that the class
split tracks mu (rho -0.900 / -0.933 on two objects). Monotone is not enough: a signal that
moves in the right direction but flips in the wrong place is still not a measurement.

THE THRESHOLD IS COMPUTABLE HERE. Contact coefficients combine as a geometric mean, so a
body slips at atan(sqrt(mu_body * mu_table)). With mu_table = 0.45 and a ramp that reaches
TILT1 degrees, the object slides iff

        mu  <  tan^2(TILT1) / mu_table          -> 0.440 at TILT1 = 24 deg

The V-track's mu sweep stopped at 0.35, entirely below that, so every clip slid and the
observed flip at mu ~ 0.25 cannot have been the physics boundary. This sweep spans it.

Two references are reported, not one: the analytic threshold above, and the simulator's own
measured slide-distance curve (where travel collapses to zero). They should agree; if they
do not, the analytic form is wrong for this scene and the measured one governs.

PASS  |log(mu_boundary_classifier) - log(mu*)| <= 0.35  (a factor of 1.4), and the class
      split is monotone in mu.
FAIL  boundary does not track threshold => recovery is a probe-head-only fine-tune on our
      labelled renders, encoder frozen.

Staged like every other pipeline here (different interpreters):
    W=.../envs/warp/bin/python ; C=.../envs/cosmos/bin/python
    CUDA_VISIBLE_DEVICES=$G LAB=outputs/scene/ga $W scripts/o2_boundary.py sim
    CUDA_VISIBLE_DEVICES=$G LAB=outputs/scene/ga $B -b --python scripts/blender_render_scene.py
    LAB=outputs/scene/ga $W scripts/o2_boundary.py clip
    CUDA_VISIBLE_DEVICES=$G $C scripts/o2_boundary.py score
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

LAB = Path(os.environ.get("LAB", "outputs/scene/ga"))
OUT = Path("docs/results/o2_boundary.json")

# spans the analytic threshold at 0.440 with points on both sides
MU = np.geomspace(0.10, 1.20, 13)
SLIDE = "Lifting a surface with [something] on it until it starts sliding down"
NOSLIDE = "Lifting a surface with [something] on it but not enough for it to slide down"


def mu_star(tilt_deg, mu_table=0.45):
    """Slip requires atan(sqrt(mu*mu_table)) < tilt, i.e. mu < tan^2(tilt)/mu_table."""
    return float(np.tan(np.radians(tilt_deg)) ** 2 / mu_table)


def do_sim(args):
    import warp as wp
    from src.sim.tilt_probe import ramp_gravity_seq
    import v1_killtest as V

    LAB.mkdir(parents=True, exist_ok=True)
    wp.init()
    scenes, meta, spec = {}, {}, []
    tilt_seq = list(np.linspace(V.TILT0, V.TILT1, V.NF))
    print(f"analytic threshold at {V.TILT1:.0f} deg ramp: mu* = {mu_star(V.TILT1):.3f}")
    with wp.ScopedDevice("cuda:0"):
        ramp = ramp_gravity_seq(V.TILT0, V.TILT1, V.NF, V.SUBSTEPS, settle_frames=6)
        for mu in MU:
            r = V.track(V.build(mu=float(mu), pos=(V.TILT_X0, 0.0, V.GZ + 0.002),
                                gseq=ramp), tilt_seq=tilt_seq)
            if r is None:
                print(f"  mu={mu:5.3f}  DIVERGED"); continue
            key = f"ga_mu{mu:05.3f}"
            scenes[key] = {"tilt_deg": tilt_seq, "objects": {"subject": r[0]}}
            meta[key] = r[1]
            dx = 100 * (r[0][-1]["flat"][0] - r[0][0]["flat"][0])
            spec.append({"key": key, "probe": "tilt", "param": "mu",
                         "value": float(mu), "ref": float(mu_star(V.TILT1)),
                         "slide_cm": dx})
            print(f"  mu={mu:5.3f}  slid {dx:6.2f} cm")
    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene.json").write_text((Path("outputs/scene/v1") / "scene.json").read_text())
    (LAB / "meta.json").write_text(json.dumps(meta))
    (LAB / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote {LAB}/scene_poses.json ({len(scenes)} scenes)")
    return 0


def union_box(cam, meta, W, H, pad=0.10):
    """ONE box containing every trajectory in the sweep.

    v1_killtest anchors the shared box on a probe's REFERENCE theta, which is right when
    the reference is mid-sweep. Here the reference is mu* = 0.441, a clip that slides
    4.7 cm, so the box came out 210 px and the mu = 0.10 clip -- which slides 65 cm -- had
    only 85.5% of its projected points inside. The object slid out of frame exactly where
    the signal should have been strongest, and the margin inverted.

    A union box costs object size but makes framing genuinely identical across theta,
    which is the property the comparison depends on.
    """
    from src.render.crop import project_points
    pts = np.concatenate([np.asarray(v, float).reshape(-1, 3) for v in meta.values()])
    uv, ok = project_points(cam, pts)
    uv = uv[ok]
    x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(max(x1 - x0, y1 - y0) * (1 + 2 * pad), min(W, H))
    cx = min(max(cx, side / 2), W - side / 2)
    cy = min(max(cy, side / 2), H - side / 2)
    s = int(side) // 2 * 2
    return (int(cx - s / 2), int(cy - s / 2), int(cx - s / 2) + s, int(cy - s / 2) + s)


def do_clip(args):
    import imageio.v2 as imageio
    from src.render.crop import crop_clip, project_points
    from src.render.views import VIEWS
    import v1_killtest as V
    from t0_tracking import pulled_camera

    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    meta = json.loads((LAB / "meta.json").read_text())
    # CLIP_NF: how many frames a scene must have to be clipped. Defaults to the tilt rig's
    # 64, but the balance rig runs 56 (probe_balance.NF), and a hard 64 silently skipped
    # every one of its scenes. The scoring head resamples to its own 64 either way.
    nf = int(os.environ.get("CLIP_NF", V.NF))
    for view in (v for v in VIEWS if (LAB / f"sim_{next(iter(spec))}@{v}").exists()):
        cam = pulled_camera(float(os.environ.get("CAM_PULL", V.CAM_PULL)), view=view)
        W, H = VIEWS[view]["width"], VIEWS[view]["height"]
        box = union_box(cam, meta, W, H)
        # CROP_BOX forces one box across LABS. Each lab's own union box is computed from
        # its own trajectories, so ga came out 342px and ga3 448px at a different offset:
        # framing then varied across the very conditions being compared, which is the
        # confound this project already banned for framing varying with theta.
        if os.environ.get("CROP_BOX"):
            box = tuple(int(x) for x in os.environ["CROP_BOX"].split(","))
        dst = LAB / ("clips" if view == "a" else f"clips_{view}")
        dst.mkdir(parents=True, exist_ok=True)
        worst = 1.0
        for key in sorted(spec):
            ps = sorted((LAB / f"sim_{key}@{view}").glob("f*.png"))[:nf]
            if len(ps) < nf:
                continue
            uv, ok = project_points(cam, np.asarray(meta[key], float).reshape(-1, 3))
            uv = uv[ok]
            worst = min(worst, float(((uv[:, 0] >= box[0]) & (uv[:, 0] <= box[2]) &
                                      (uv[:, 1] >= box[1]) & (uv[:, 1] <= box[3])).mean()))
            frames = np.stack([imageio.imread(p)[..., :3] for p in ps])
            w = imageio.get_writer(str(dst / f"{key}.mp4"), fps=int(V.FPS),
                                   codec="libx264", quality=9, macro_block_size=1)
            for f in crop_clip(frames, box):
                w.append_data(f)
            w.close()
        print(f"  view {view}: box {box} ({box[2]-box[0]}px), "
              f"worst containment {100*worst:.1f}%", flush=True)
    return 0


def do_score(args):
    from src.outcome.head import OutcomeHead
    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    m = OutcomeHead(device="cuda:0")
    rows = []
    for key in sorted(spec, key=lambda k: spec[k]["value"]):
        clip = LAB / "clips" / f"{key}.mp4"
        if not clip.exists():
            continue
        p = m.score_clip(clip)
        li = {n: i for i, n in enumerate(m.labels)}
        a, b = float(p[li[SLIDE]]), float(p[li[NOSLIDE]])
        rows.append({**spec[key], "p_slide": a, "p_noslide": b, "margin": a - b,
                     "top1": m.top(p, 1)[0][0]})
        print(f"  mu={spec[key]['value']:5.3f}  slide {spec[key]['slide_cm']:6.2f}cm  "
              f"p_slide {a:.3f}  p_noslide {b:.3f}  margin {a-b:+.3f}", flush=True)

    v = np.array([r["value"] for r in rows]); mg = np.array([r["margin"] for r in rows])
    sl = np.array([r["slide_cm"] for r in rows])
    # classifier boundary: where the margin crosses zero (log-linear interpolation)
    cross = None
    for i in range(len(v) - 1):
        if mg[i] > 0 >= mg[i + 1]:
            f = mg[i] / (mg[i] - mg[i + 1])
            cross = float(np.exp(np.log(v[i]) + f * (np.log(v[i + 1]) - np.log(v[i]))))
            break
    star = float(rows[0]["ref"])
    # the simulator's own boundary: where travel collapses (< 1 cm)
    sim_b = next((float(v[i]) for i in range(len(v)) if sl[i] < 1.0), None)
    ra = np.argsort(np.argsort(v)).astype(float); rb = np.argsort(np.argsort(mg)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    rho = float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

    print(f"\nclassifier boundary (margin crosses 0) : mu = "
          f"{'n/a' if cross is None else f'{cross:.3f}'}")
    print(f"analytic threshold  tan^2(tilt)/mu_table : mu* = {star:.3f}")
    print(f"simulator boundary  (travel < 1 cm)      : mu = "
          f"{'n/a' if sim_b is None else f'{sim_b:.3f}'}")
    print(f"monotonicity  rho(mu, margin)            : {rho:+.3f}")
    ok = cross is not None and abs(np.log(cross) - np.log(star)) <= 0.35 and rho <= -0.7
    if cross:
        print(f"|log offset| = {abs(np.log(cross)-np.log(star)):.3f}   (pass <= 0.35)")
    print(f"\nG-A {'PASS' if ok else 'FAIL'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": m.model_id, "rows": rows, "boundary": cross,
                               "mu_star": star, "sim_boundary": sim_b, "rho": rho,
                               "verdict": "PASS" if ok else "FAIL"}, indent=1))
    print(f"wrote {OUT}")
    return 0 if ok else 1


MODES = {"sim": do_sim, "clip": do_clip, "score": do_score}
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--floor", action="store_true", default=False)
    a = ap.parse_args()
    raise SystemExit(MODES[a.mode](a))
