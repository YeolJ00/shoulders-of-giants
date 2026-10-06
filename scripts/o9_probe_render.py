"""Render + score the two NEW probes, so the classifier reads them the way it reads tilt.

tilt->slide is already through the whole pipeline (o4_joint, o6_blind: blind mu to 3% on the
pot). The sensitivity matrix says two more probes carry parameters that tilt cannot:

    tip     ramp to 60 deg, the object topples   -> com   (6.67x; mass and cd EXACTLY 1.00x)
    bounce  dropped, it rebounds                 -> cd    (5.73x; mu-immune at 1.08x)

Each runs at ITS OWN validated solver set, which is not a detail: bounce's rebound spans
5.47x over cd at pen_frac 5e-3 and 1.03x at 1e-4. Rendering bounce on the tilt rig's set
would render a probe that no longer responds to the parameter it is for.

Framing follows the rules the tilt work established the hard way -- ONE crop box per probe
forced across the whole sweep (CROP_BOX), the clip-time camera pull matching the render-time
one, and per-clip containment checked so a clip whose object leaves frame is dropped rather
than scored.

    PROBE=tip    LAB=outputs/scene/p_tip    W scripts/o9_probe_render.py sim
    ... blender ...
    PROBE=tip    LAB=outputs/scene/p_tip    W scripts/o9_probe_render.py clip
    CUDA_VISIBLE_DEVICES=$G C scripts/o4_joint.py score --lab outputs/scene/p_tip
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

PROBE = os.environ.get("PROBE", "tip")
LAB = Path(os.environ.get("LAB", f"outputs/scene/p_{PROBE}"))
SCENE_SRC = Path(os.environ.get("SCENE_SRC", "outputs/scene/v1"))
CROP = os.environ.get("CROP_BOX", "")

# tip sweeps the centre of mass; bounce sweeps contact damping.
# The CoM sweep stops at +/-0.018 m. Past ~0.015 the vase is already past tipping at rest
# and topples immediately whichever way the offset points, so the observable goes EVEN in
# com and the sign is lost (4.85 deg at -0.030 against 4.81 at +0.030). Inside this range it
# is signed and monotone (10.58 / 22.17 / 15.17 deg at -0.015 / 0 / +0.015).
COMS = ([float(x) for x in os.environ["COMS"].split(",")] if os.environ.get("COMS")
        else [-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018])
# Several ramp angles, for the same reason the tilt probe needs several: one condition
# cannot identify a parameter, and the topple angle at one ramp end is a single number.
TIP_ENDS = ([float(x) for x in os.environ["TIP_ENDS"].split(",")]
            if os.environ.get("TIP_ENDS") else None)
CDS = [400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0, 25600.0]
# 36 deg, not 60. The vase topples by ~22 deg, so the rest of a 60 deg ramp adds no
# information and swings the object out of frame -- worst containment was 64.6%. At 36 deg
# with CAM_PULL 2.20 it is 100%. Framing is fixed per probe and reported, so this probe
# using a wider pull than the tilt probe is fine; what is not fine is a box that varies
# within a sweep.
TIP_END = 36.0
# The tip probe needs an object that TOPPLES. A bowl is 0.31 m wide and 0.09 m tall, so it
# tips at ~72 deg and simply slides instead -- at mu = 10 it still slid 71 cm and never
# gripped, so the "topple angle" measured a sliding bowl tumbling. The vase is 0.20 x 0.40 m
# and tips at ~34 deg. mu is set above the slip angle so it tips rather than slides.
TIP_MU = 1.30


def do_sim(args):
    import warp as wp
    import v1_killtest as V
    from src.sim.tilt_probe import ramp_gravity_seq
    LAB.mkdir(parents=True, exist_ok=True)
    wp.init()
    scenes, meta, spec = {}, {}, []
    with wp.ScopedDevice("cuda:0"):
        if PROBE == "tip":
            for END in (TIP_ENDS or [TIP_END]):
                seq = list(np.linspace(0.0, END, V.NF))
                g = ramp_gravity_seq(0.0, END, V.NF, V.SUBSTEPS, settle_frames=6)
                print(f"  ramp to {END:.0f} deg")
                for c in COMS:
                    r = V.track(V.build(mu=TIP_MU, pos=(0.0, 0.0, V.GZ + 0.002), gseq=g,
                                        com=None if c == 0 else [float(c), 0.0, 0.0]),
                                tilt_seq=seq)
                    if r is None:
                        print(f"    com={c:+.4f} DIVERGED"); continue
                    key = f"tip_t{int(END):02d}_com{c:+.4f}"
                    scenes[key] = {"tilt_deg": seq, "objects": {"subject": r[0]}}
                    meta[key] = r[1]
                    spec.append({"key": key, "probe": "tip", "param": "com",
                                 "value": float(c), "ref": 0.0, "tilt": float(END),
                                 "slide_cm": 100 * (r[0][-1]["flat"][0]
                                                    - r[0][0]["flat"][0])})
                    print(f"    com={c:+.4f}  ok")
        elif PROBE == "bounce":
            import probe_bounce_settle as PB
            for cd in CDS:
                s = PB.build(cd, V.NF, pos=(0.0, 0.0, PB.GZ + 0.22))
                r = V.track(s, stride=PB.SUBSTEPS)
                if r is None:
                    print(f"  cd={cd:.0f} DIVERGED"); continue
                z = np.array([p["loc"][2] for p in r[0]])
                d = np.diff(z)
                lo = next((i for i in range(1, len(d)) if d[i - 1] < 0 <= d[i]), None)
                up = next((i for i in range(lo + 1, len(d)) if d[i] < 0), len(z) - 1) if lo else 0
                reb = 1000.0 * (z[up] - z[lo]) if lo else 0.0
                key = f"bnc_cd{cd:07.0f}"
                scenes[key] = {"objects": {"subject": r[0]}}
                meta[key] = r[1]
                spec.append({"key": key, "probe": "bounce", "param": "cd", "value": float(cd),
                             "ref": 3200.0, "tilt": 0.0, "slide_cm": float(reb)})
                print(f"  cd={cd:7.0f}  rebound {reb:6.2f} mm")
        else:
            sys.exit(f"unknown PROBE {PROBE}")
    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene.json").write_text((SCENE_SRC / "scene.json").read_text())
    (LAB / "meta.json").write_text(json.dumps(meta))
    (LAB / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote {len(scenes)} scenes to {LAB}")
    return 0


def do_clip(args):
    if CROP:
        os.environ["CROP_BOX"] = CROP
    os.environ.setdefault("CAM_PULL", "1.20")  # TIP overrides this to 2.20
    sys.argv = [sys.argv[0], "clip"]
    import o2_boundary
    o2_boundary.LAB = LAB
    return o2_boundary.do_clip(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.set_defaults(f=do_sim)
    s = sub.add_parser("clip"); s.set_defaults(f=do_clip)
    a = ap.parse_args()
    sys.exit(a.f(a))
