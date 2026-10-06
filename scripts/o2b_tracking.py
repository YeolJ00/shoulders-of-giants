"""G-A2 -- does the classifier's boundary MOVE when the physics threshold moves?

G-A as first specified asks whether the decision boundary sits at mu* = tan^2(tilt)/mu_table.
That may be the wrong question. Everything below mu* slides, but at mu = 0.35 it slides only
6 cm, and a person watching a bowl shift 6 cm on a tilting table would reasonably say it did
not slide. So the classifier's boundary can legitimately sit BELOW mu* -- that offset is the
perceptual calibration the simulator cannot know, which is the entire premise of this track.
Failing the head for supplying exactly what we asked it for would be a mistake.

The test that survives a constant perceptual offset is whether the boundary TRACKS. Changing
the ramp angle moves mu* analytically over a 3x range:

        18 deg -> 0.235        24 deg -> 0.441        30 deg -> 0.741

If the classifier's flip follows, it is reading the physical condition. If the flip sits at
the same mu regardless of ramp angle, it is reading a fixed appearance cue (how far the
object moved on screen, say) and the agreement at any single angle was a coincidence.

PASS  boundary is monotone in tilt across the three angles AND spans >= 1.6x
      (against the analytic 3.15x, allowing a compressed but real response)
FAIL  boundary flat in tilt => fixed appearance cue, not physics

    LAB=outputs/scene/ga2 W scripts/o2b_tracking.py sim
    ... render ...
    LAB=outputs/scene/ga2 W scripts/o2b_tracking.py clip
    CUDA_VISIBLE_DEVICES=$G C scripts/o2b_tracking.py score
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

LAB = Path(os.environ.get("LAB", "outputs/scene/ga2"))
OUT = Path(os.environ.get("OUT", "docs/results/o2b_tracking.json"))
import os as _os
# CONFIRMATION RUN. The 18/24/30 sweep showed the margin's MINIMUM landing within 15% of
# tan^2(tilt)/mu_table at all three angles (0.221 vs 0.235, 0.435 vs 0.441, 0.855 vs 0.741),
# spanning 3.87x against an analytic 3.16x. But the minimum was chosen as the readout AFTER
# seeing the curves, and a post-hoc readout has looked good and then failed three times
# already in this project. So: two INTERPOLATED angles, predictions fixed before rendering.
#
#     tilt 21 deg -> mu* = 0.328        tilt 27 deg -> mu* = 0.577
#
# Neither extrapolates, and the mu grid has a sample within 6% of each prediction.
TILTS = tuple(float(x) for x in _os.environ.get("TILTS", "18,24,30").split(","))
MU = (np.array([float(x) for x in _os.environ["MU"].split(",")])
      if _os.environ.get("MU") else np.geomspace(0.10, 1.20, 11))
SLIDE = "Lifting a surface with [something] on it until it starts sliding down"
NOSLIDE = "Lifting a surface with [something] on it but not enough for it to slide down"


def mu_star(t, mu_table=0.45):
    return float(np.tan(np.radians(t)) ** 2 / mu_table)


def do_sim(args):
    import warp as wp
    from src.sim.tilt_probe import ramp_gravity_seq
    import v1_killtest as V
    LAB.mkdir(parents=True, exist_ok=True)
    wp.init()
    scenes, meta, spec = {}, {}, []
    with wp.ScopedDevice("cuda:0"):
        for t in TILTS:
            ramp = ramp_gravity_seq(0.0, t, V.NF, V.SUBSTEPS, settle_frames=6)
            seq = list(np.linspace(0.0, t, V.NF))
            print(f"tilt {t:.0f} deg   analytic mu* = {mu_star(t):.3f}")
            for mu in MU:
                r = V.track(V.build(mu=float(mu), pos=(V.TILT_X0, 0.0, V.GZ + 0.002),
                                    gseq=ramp), tilt_seq=seq)
                if r is None:
                    print(f"   mu={mu:5.3f} DIVERGED"); continue
                key = f"t{int(t):02d}_mu{mu:05.3f}"
                scenes[key] = {"tilt_deg": seq, "objects": {"subject": r[0]}}
                meta[key] = r[1]
                dx = 100 * (r[0][-1]["flat"][0] - r[0][0]["flat"][0])
                spec.append({"key": key, "probe": "tilt", "param": "mu", "value": float(mu),
                             "ref": mu_star(t), "tilt": float(t), "slide_cm": dx})
                print(f"   mu={mu:5.3f}  slid {dx:6.2f} cm")
    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene.json").write_text(
        Path(os.environ.get("SCENE_SRC", "outputs/scene/v1")).joinpath(
            "scene.json").read_text())
    (LAB / "meta.json").write_text(json.dumps(meta))
    (LAB / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote {len(scenes)} scenes")
    return 0


def do_clip(args):
    sys.argv = [sys.argv[0], "clip"]
    import o2_boundary
    o2_boundary.LAB = LAB
    return o2_boundary.do_clip(args)


def do_score(args):
    from src.outcome.head import OutcomeHead
    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    m = OutcomeHead(device="cuda:0")
    li = {n: i for i, n in enumerate(m.labels)}
    res, rows = {}, []
    for t in TILTS:
        ks = sorted([k for k, s in spec.items() if s["tilt"] == t],
                    key=lambda k: spec[k]["value"])
        v, mg = [], []
        print(f"\ntilt {t:.0f} deg   analytic mu* = {mu_star(t):.3f}")
        for k in ks:
            c = LAB / "clips" / f"{k}.mp4"
            if not c.exists():
                continue
            p = m.score_clip(c)
            a, b = float(p[li[SLIDE]]), float(p[li[NOSLIDE]])
            v.append(spec[k]["value"]); mg.append(a - b)
            rows.append({**spec[k], "p_slide": a, "p_noslide": b, "margin": a - b})
            print(f"   mu={spec[k]['value']:5.3f} slide {spec[k]['slide_cm']:6.2f}cm "
                  f"margin {a-b:+.3f}", flush=True)
        v, mg = np.array(v), np.array(mg)
        cross = None
        for i in range(len(v) - 1):
            if mg[i] > 0 >= mg[i + 1]:
                f = mg[i] / (mg[i] - mg[i + 1])
                cross = float(np.exp(np.log(v[i]) + f * (np.log(v[i+1]) - np.log(v[i])))); break
        res[t] = {"boundary": cross, "mu_star": mu_star(t)}
        print(f"   -> classifier boundary {'n/a' if cross is None else f'{cross:.3f}'}"
              f"   analytic {mu_star(t):.3f}")

    b = [res[t]["boundary"] for t in TILTS]
    s = [res[t]["mu_star"] for t in TILTS]
    ok = all(x is not None for x in b)
    if ok:
        mono = all(b[i] < b[i + 1] for i in range(len(b) - 1))
        span = max(b) / min(b)
        print(f"\nboundaries : {[round(x,3) for x in b]}   span {span:.2f}x")
        print(f"analytic   : {[round(x,3) for x in s]}   span {max(s)/min(s):.2f}x")
        print(f"monotone in tilt: {mono}")
        ok = mono and span >= 1.6
    else:
        print("\nboundary not found at every tilt angle")
    print(f"\nG-A2 {'PASS' if ok else 'FAIL'}  (pass: monotone and span >= 1.6x)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": m.model_id, "per_tilt": {str(k): v for k, v in res.items()},
                               "rows": rows, "verdict": "PASS" if ok else "FAIL"}, indent=1))
    print(f"wrote {OUT}")
    return 0 if ok else 1


MODES = {"sim": do_sim, "clip": do_clip, "score": do_score}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=list(MODES))
    raise SystemExit(MODES[ap.parse_args().mode](None))
