"""NOTE 2026-10-06: the POT version of this probe is superseded by scripts/p9_pot_topple.py
(with its true shell centre of mass the pot slides before it can tip at this grip).

Centre of mass on a TILTED ramp: how long the object survives before it goes over.

The CoM probe used to ramp gravity and read the ANGLE at which the body toppled. On a fixed
tilt there is no angle to read, so the observable changes: the body is released on an already
inclined plane and the observable is the TIME IT SURVIVES before its axis departs the plane
normal by 25 degrees.

The obvious alternative -- the body's final orientation -- does not work. Once a body is
over it tumbles, and the end state is chaotic: at 24 deg the final angle ran 4.5 / 82.9 /
158.2 / 34.7 / 164.6 / 66.3 / 151.7 degrees across a monotone CoM sweep, which is noise.
Time-to-topple over the same sweep is 2.04 / 0.54 / 0.38 / 0.29 / 0.25 / 0.21 / 0.21 s --
monotone at every tilt, with the threshold moving the right way as the tilt rises.

mu is held high (1.30) so the body topples rather than sliding out of the experiment, and
the sim reports whether it slid or left the table anyway.

    W scripts/o14_com.py sim
    ... blender, CAM_PULL from outputs/scene/_c_box.json ...
    W scripts/o14_com.py clip
    CUDA_VISIBLE_DEVICES=$G C scripts/o4_joint.py score --lab outputs/scene/c_vase
    W scripts/o14_com.py fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

GZ = 0.706
# 32 frames, not 64. The observable is complete by 0.54 s; the remaining 1.3 s is the body
# tumbling and rolling off the table, which the crop box then has to cover -- over 64 frames
# no camera pull contained every clip (best 55.8%). At 32 frames the box closes and the clip
# is mostly the event. The cost is the stable end: anything surviving past 1.33 s reads as
# "never toppled" and saturates. The scoring head resamples to its own 64 frames either way.
FPS, NF, SUBSTEPS = 24.0, 32, 420
TAG = os.environ.get("CTAG", "vase")
ASSET = os.environ.get("CASSET", "rigid/ceramic_vase_01")
SCALE = float(os.environ.get("CSCALE", "0.55"))
TILTS = tuple(float(x) for x in os.environ.get("CTILTS", "16,20,24,28").split(","))
MASS, MU, CD = 0.40, 1.30, 3200.0
COMS = np.linspace(-0.018, 0.018, 9)
TOPPLE_DEG = 25.0
LAB = Path(f"outputs/scene/c_{TAG}")
BOX = Path(f"outputs/scene/_c_box_{TAG}.json")
OUT = Path(f"docs/results/o14_com_{TAG}.json")


def q2m(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def do_sim(args):
    import warp as wp
    from src.sim.mesh_scene import MeshProbeScene
    wp.init()
    LAB.mkdir(parents=True, exist_ok=True)
    scenes, meta, spec = {}, {}, []
    with wp.ScopedDevice("cuda:0"):
        for tilt in TILTS:
            for c in COMS:
                ko = dict(masses=[MASS], ground_z=GZ, dt=1.0 / (FPS * SUBSTEPS),
                          n_steps=NF * SUBSTEPS, k=2500.0, cd=[CD], mu=[MU],
                          mesh_scale=[SCALE], faces=900, ground_mu=0.45,
                          ground_cd=3000.0, table=(0.79, 0.49), tilt_deg=tilt)
                if abs(c) > 1e-9:
                    ko["com_offset"] = [[float(c), 0.0, 0.0]]
                s = MeshProbeScene([ASSET], [[0., 0., 0.]], [[0., 0., 0.]], **ko)
                from src.sim.rigs import apply_tilt_rig
                apply_tilt_rig(s)          # stiffness + restitution (alpha), src/sim/rigs.py
                s.seat_on_plane(along=0.0)
                s.rollout()
                P, Q = s.positions(SUBSTEPS), s.rotations(SUBSTEPS)
                if not (np.isfinite(P).all() and np.isfinite(Q).all()
                        and np.abs(P).max() < 50):
                    print(f"   t{tilt:.0f} com{1000*c:+.0f} DIVERGED"); continue
                n, tv = s._gn, s._gt
                ang = np.array([np.degrees(np.arccos(np.clip(q2m(q)[:, 2] @ n, -1, 1)))
                                for q in Q[:, 0]])
                over = np.where(ang > TOPPLE_DEG)[0]
                t_top = float(over[0] / FPS) if len(over) else float(NF / FPS)
                never = not len(over)
                d = P[:, 0] - s._gp
                # Only a departure BEFORE the topple invalidates the reading. This probe's
                # observable is complete at 0.17-0.54 s; the body then tumbles and rolls off,
                # which is aftermath, not a different event. Flagging any departure marked
                # all 36 clips bad and left the fit with nothing.
                cut = int(over[0]) if len(over) else len(P)
                off_table = bool((np.abs(d[:cut + 1] @ tv) > 0.79).any())
                key = f"t{int(tilt):02d}_c{1000*c:+06.1f}"
                V = s.local.numpy().astype(float)
                idx = np.linspace(0, len(V) - 1, min(48, len(V))).astype(int)
                seq, pts = [], []
                for t2 in range(len(P)):
                    R = q2m(Q[t2, 0])
                    seq.append({"loc": [float(v) for v in (P[t2, 0] - R @ s.coms[0])],
                                "mat": [[float(v) for v in r] for r in R]})
                    pts.append(((R @ V[idx].T).T + P[t2, 0]).tolist())
                scenes[key] = {"tilt_deg": tilt, "objects": {"subject": seq}}
                meta[key] = pts
                spec.append({"key": key, "probe": "topple_fixed", "param": "com",
                             "value": float(c), "ref": 0.0, "tilt": float(tilt),
                             "slide_cm": t_top, "t_topple_s": t_top, "never": never,
                             "off_table": off_table})
                print(f"   t{tilt:.0f} com{1000*c:+6.1f}mm  survives {t_top:5.2f} s"
                      f"{'  NEVER TOPPLED' if never else ''}"
                      f"{'  OFF TABLE' if off_table else ''}", flush=True)
    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene.json").write_text(json.dumps(
        {"ground_z": GZ, "objects": {"subject": {"asset": ASSET, "scale": SCALE,
                                                 "rot_z": 0.0,
                                                 "pos": [0.0, 0.0, GZ + 0.30]}}}))
    (LAB / "meta.json").write_text(json.dumps(meta))
    (LAB / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote {len(scenes)} scenes to {LAB}")
    _choose_box()
    return 0


def _choose_box():
    """One box over the INFORMATIVE clips: a clip that never toppled is saturated, and one
    that left the table is a different event -- neither should size the framing."""
    import o2_boundary as O
    from src.render.crop import project_points
    from src.render.views import VIEWS
    from t0_tracking import pulled_camera
    W, H = VIEWS["a"]["width"], VIEWS["a"]["height"]
    spec = json.loads((LAB / "spec.json").read_text())
    meta = json.loads((LAB / "meta.json").read_text())
    good = {r["key"] for r in spec if not r["off_table"]}
    m = {k: v for k, v in meta.items() if k in good}
    for pull in (1.2, 1.6, 2.0, 2.5, 3.0, 3.6):
        cam = pulled_camera(pull, view="a")
        b = O.union_box(cam, m, W, H, pad=0.12)
        worst = 1.0
        for v in m.values():
            uv, o = project_points(cam, np.asarray(v, float).reshape(-1, 3))
            uv = uv[o]
            worst = min(worst, float(((uv[:, 0] >= b[0]) & (uv[:, 0] <= b[2]) &
                                      (uv[:, 1] >= b[1]) & (uv[:, 1] <= b[3])).mean()))
        print(f"  pull {pull:.1f}: box {b[2]-b[0]}px worst containment {100*worst:.1f}%")
        if worst >= 0.99:
            BOX.write_text(json.dumps({"pull": pull, "box": list(b)}))
            print(f"  chose pull {pull:.1f} -> {BOX}")
            return
    # BEST EFFORT. A round asset rolls after it topples, so no pull contains every clip
    # (the pot's best was 71.7%). Rather than abandon the probe, take the pull that leaves
    # the most clips FULLY contained and let the containment check drop the rest -- a
    # partially framed clip is not a measurement and must not be scored either way.
    best = None
    for pull in (2.5, 3.0, 3.6, 4.4, 5.2):
        cam = pulled_camera(pull, view="a")
        b = O.union_box(cam, m, W, H, pad=0.12)
        n = 0
        for v in m.values():
            uv, o = project_points(cam, np.asarray(v, float).reshape(-1, 3))
            uv = uv[o]
            if float(((uv[:, 0] >= b[0]) & (uv[:, 0] <= b[2]) &
                      (uv[:, 1] >= b[1]) & (uv[:, 1] <= b[3])).mean()) >= 0.99:
                n += 1
        if best is None or n > best[0]:
            best = (n, pull, list(b))
    BOX.write_text(json.dumps({"pull": best[1], "box": best[2], "contained": best[0],
                               "of": len(m)}))
    print(f"  no pull contains every clip; best effort pull {best[1]:.1f} keeps "
          f"{best[0]}/{len(m)} -> {BOX}")


def do_clip(args):
    cfg = json.loads(BOX.read_text())
    os.environ["CROP_BOX"] = ",".join(str(v) for v in cfg["box"])
    os.environ["CAM_PULL"] = str(cfg["pull"])
    sys.argv = [sys.argv[0], "clip"]
    import o2_boundary
    o2_boundary.LAB = LAB
    return o2_boundary.do_clip(args)


def _sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float("nan") if d == 0 else float((ra * rb).sum() / d)


def do_fit(args):
    import glob
    import o4_joint as O
    cfg = json.loads(BOX.read_text())
    probs = {}
    for f in glob.glob("outputs/outcome/cache/*.json"):
        d = json.load(open(f))
        probs[str(Path(d["clip"]).resolve())] = np.asarray(d["probs"], np.float64)
    con = O.containment(LAB, pull=cfg["pull"])
    rows = []
    for r in json.loads((LAB / "spec.json").read_text()):
        if r["off_table"]:
            continue
        c = (LAB / "clips" / f"{r['key']}.mp4").resolve()
        p = probs.get(str(c))
        if p is None or con.get(r["key"], 0) < O.MIN_CONTAIN:
            continue
        rows.append({**r, "p": p})
    if not rows:
        sys.exit("no scored clips")
    print(f"{len(rows)} clips over {len({r['tilt'] for r in rows})} tilts, "
          f"{len({round(r['value'],4) for r in rows})} CoM offsets "
          f"({sum(r['never'] for r in rows)} never toppled)")
    X = np.log(np.array([r["p"] for r in rows]) + 1e-8)
    com = np.array([r["value"] for r in rows])

    # hold out a whole CoM offset: nothing in the calibration has seen it
    hats = {}
    for c0 in sorted({round(c, 4) for c in com}):
        h = np.isclose(com, c0, atol=1e-5)
        hats[c0] = [float(v) for v in O.ridge(X[~h], com[~h])(X[h])]
    print()
    print(f"  {'true CoM':>9} {'read (per tilt)':>34} {'mean':>8} {'err':>8}")
    e = []
    for c0 in sorted(hats):
        m = float(np.mean(hats[c0]))
        e.append(abs(m - c0))
        st = " ".join(f"{1000*v:+6.1f}" for v in hats[c0])
        print(f"  {1000*c0:>7.1f}mm {st:>34} {1000*m:>+7.1f} {1000*(m-c0):>+7.1f}mm")
    ea = [abs(v - c0) for c0 in hats for v in hats[c0]]
    res = {"n": len(rows), "median_err_mm": float(1000 * np.median(e)),
           "single_err_mm": float(1000 * np.median(ea)),
           "guess_zero_mm": float(1000 * np.median(np.abs(sorted(hats)))),
           "rho": _sp([np.mean(hats[c]) for c in sorted(hats)], sorted(hats)),
           "rows": {str(k): v for k, v in hats.items()}}
    print(f"\n  one tilt alone     : median |err| {1000*np.median(ea):.1f} mm  (n={len(ea)})")
    print(f"  tilts averaged     : median |err| {1000*np.median(e):.1f} mm  (n={len(e)})")
    print(f"  always guessing 0  : median |err| {res['guess_zero_mm']:.1f} mm")
    print(f"  rho(read, true)    : {res['rho']:+.3f}")
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.set_defaults(f=do_sim)
    s = sub.add_parser("clip"); s.set_defaults(f=do_clip)
    s = sub.add_parser("fit"); s.set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
