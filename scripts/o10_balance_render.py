"""Render the balance probe, so mass can be read off pixels like the other three.

Mass is the one parameter with an excellent simulator instrument and no reading from video:
the pan balance recovers it to +/-3% and is EXACTLY immune to mu and cd (o7_sensitivity,
both 1.00x), but it had never been rendered. It is also the one parameter that provably
cannot be read from how an object slides -- that is what killed the collide probe.

The balance is a THRESHOLD instrument and is rendered as one. Its settled beam angle
saturates at the +/-12 degree stops, so a magnitude readout returns -12 for a payload both
far lighter AND far heavier than the reference; what carries the information is WHICH WAY
the beam tipped, against a reference mass we set. So:

    for each subject mass, sweep the reference; the flip point is the subject's mass.

The reference block's volume tracks its mass, so it looks like what it weighs -- the scale
is visible in the frame rather than being a hidden condition.

Generalises balance_render_prep.py, which exported three fixed cases. Two conventions from
it are kept because both were bought with a failure:
  * the SUBJECT points at its own asset, never a re-exported copy -- trimesh's OBJ writer
    drops the material and a re-exported apple renders as a plain white blob, which is a
    different material than the one being asked about (rule 4).
  * poses are written mesh-origin, undoing the COM-centred body frame; with com_offset now
    able to move that origin, the same subtraction still applies because MeshProbeScene
    reports the shifted centre in s.coms.

    W scripts/o10_balance_render.py sim
    ... blender (LAB=outputs/scene/p_bal) ...
    W scripts/o10_balance_render.py clip
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

BTAG = os.environ.get("BTAG", "apple")
LAB = Path(os.environ.get("LAB", f"outputs/scene/p_bal_{BTAG}"))
CROP = os.environ.get("CROP_BOX", "")

# Subject masses to recover. The apple's true mass is 0.182 kg; these are what the rig is
# told the payload weighs, and what the flip point should reveal.
SUBJ = [float(x) for x in os.environ.get("SUBJ", "0.10,0.18,0.30").split(",")]
# References swept per subject. Log-spaced so the flip lands inside the sweep for all three.
REFS = [float(x) for x in os.environ.get(
    "REFS", "0.055,0.075,0.10,0.13,0.155,0.185,0.22,0.26,0.31,0.38,0.46,0.55").split(",")]


def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def do_sim(args):
    import warp as wp
    import probe_balance as PB
    from balance_render_prep import export
    LAB.mkdir(parents=True, exist_ok=True)
    wp.init()
    a_beam = export(PB.beam_with_pans(), "_bal_beam")
    a_stand = export(PB.stand_mesh(), "_bal_stand")
    a_subj = PB.SUBJECT[0] + "/" + PB.SUBJECT[1]
    scenes, objs, spec, meta = {}, {}, [], {}
    with wp.ScopedDevice("cuda:0"):
        for ms in SUBJ:
            for mr in REFS:
                s = PB.build(mr, m_subject=ms)
                s.rollout()
                P, Q = s.positions(PB.SUBSTEPS), s.rotations(PB.SUBSTEPS)
                if not (np.isfinite(P).all() and np.isfinite(Q).all()):
                    print(f"  ms={ms:.3f} mr={mr:.3f}  DIVERGED"); continue
                ang = np.degrees(2.0 * np.arcsin(np.clip(Q[:, 0, 1], -1.0, 1.0)))
                a = PB.settled(ang)
                tag = f"s{ms:05.3f}_r{mr:05.3f}"
                key = f"bal_{tag}"
                BODIES = [("beam", a_beam), ("stand", a_stand), ("obj", a_subj),
                          ("ref", export(PB.ref_mesh(mr), f"_bal_ref_{tag}"))]
                scenes[key] = {"tilt_deg": 0.0, "objects": {}}
                pts = []
                for i, (nm, asset) in enumerate(BODIES):
                    C = s.coms[i]
                    seq = []
                    for t in range(len(P)):
                        R = quat_to_mat(Q[t, i])
                        seq.append({"loc": [float(v) for v in (P[t, i] - R @ C)],
                                    "mat": [[float(v) for v in r] for r in R]})
                    nk = f"{tag}_{nm}"
                    scenes[key]["objects"][nk] = seq
                    objs[nk] = {"asset": asset,
                                "scale": PB.SUBJECT[2] if nm == "obj" else 1.0,
                                "rot_z": 0.0, "pos": [float(v) for v in P[0, i]]}
                    if nm in ("beam", "obj", "ref"):        # what the crop must contain
                        pts += [list(map(float, P[t, i])) for t in range(len(P))]
                meta[key] = pts
                spec.append({"key": key, "probe": "balance", "param": "mass",
                             "value": float(ms), "ref": float(mr), "tilt": 0.0,
                             "slide_cm": float(a)})       # the beam angle, in degrees
                print(f"  subject {ms:.3f} kg vs ref {mr:.3f} kg -> beam {a:+6.2f} deg "
                      f"({'subject down' if a > 0 else 'ref down'})", flush=True)
    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene.json").write_text(json.dumps({"ground_z": PB.GZ, "objects": objs}))
    (LAB / "meta.json").write_text(json.dumps(meta))
    (LAB / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote {len(scenes)} scenes, {len(objs)} bodies to {LAB}")
    return 0


def do_clip(args):
    if CROP:
        os.environ["CROP_BOX"] = CROP
    os.environ.setdefault("CAM_PULL", "1.20")
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
