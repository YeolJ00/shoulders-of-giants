"""Render QA: does the RENDER put the object where the SIMULATOR says it is?

No vision model. The pose-frame bug -- poses handed to the renderer in the simulator's level
frame while the renderer tilted the table -- left the object riding up to 11.5 cm off the
ramp for four probes and several hundred rendered clips, and was caught by a human watching a
video. This check would have flagged it on frame 1.

The test is a geometric identity, not a measurement. Whenever the simulator has the body in
contact with the plane, its lowest vertex must lie ON that plane. Both quantities come from
the same poses the renderer is given, so any disagreement means the pose handed over does not
describe the scene that will be drawn.

    W scripts/f3_contact_check.py outputs/scene/t_sorter [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

PIVOT = np.array([-0.05, -0.06, 0.706])


def check(lab):
    """Verify the pose handed to the RENDERER is the simulator pose, correctly transformed.

    An earlier version tested "did the object drift off the plane", which flags every
    legitimate departure -- an object sliding off the table edge (3 m), a bowl mid-bounce
    (307 mm), a vase toppling. Useless as a bug detector.

    This is an exact identity instead. track() stores both representations: "flat" is the
    simulator's position, where the ground is level and gravity is what tilts, and "loc" is
    what the renderer is given, where the TABLE tilts. They must satisfy

        loc == PIVOT + Ry(tilt) @ (flat - PIVOT)

    The pose-frame bug was exactly this identity failing -- "loc" was handed over as the
    untransformed simulator pose -- and it left objects riding up to 11.5 cm off the ramp
    across several hundred clips until someone watched a video. No tolerance to tune: either
    the transform was applied or it was not.

    A lab whose poses predate the fix has no "flat" key and cannot be verified; that is
    reported rather than passed.
    """
    lab = Path(lab)
    sp = lab / "scene_poses.json"
    if not sp.exists():
        return None
    scenes = json.loads(sp.read_text())
    worst, where, n, nover = 0.0, None, 0, 0
    for key, e in scenes.items():
        tilt = e.get("tilt_deg", 0.0)
        seq = tilt if isinstance(tilt, list) else None
        for name, poses in e["objects"].items():
            for t, p in enumerate(poses):
                if "flat" not in p:
                    return {"lab": str(lab), "frames": 0, "err_mm": None,
                            "where": "no 'flat' key -- rendered before the fix",
                            "pass": None}
                th = np.radians(float(seq[min(t, len(seq) - 1)]) if seq else float(tilt))
                Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                               [-np.sin(th), 0, np.cos(th)]])
                want = PIVOT + Ry @ (np.asarray(p["flat"], float) - PIVOT)
                d = float(np.linalg.norm(np.asarray(p["loc"], float) - want))
                n += 1
                if d > worst:
                    worst, where = d, f"{key}/{name}@{t}"
    return {"lab": str(lab), "frames": n, "err_mm": 1000 * worst, "where": where,
            "pass": bool(1000 * worst < 1e-3)}


if __name__ == "__main__":
    labs = sys.argv[1:] or sorted(str(p.parent) for p in
                                  Path("outputs/scene").glob("*/scene_poses.json"))
    print(f"{'lab':<26}{'frames':>8}{'pose error':>13}{'verdict':>11}   note")
    bad = unk = 0
    for l in labs:
        r = check(l)
        if r is None:
            continue
        v = "OK" if r["pass"] else ("UNVERIFIED" if r["pass"] is None else "FAIL")
        bad += r["pass"] is False; unk += r["pass"] is None
        e = "--" if r["err_mm"] is None else f"{r['err_mm']:.4f}mm"
        print(f"{Path(r['lab']).name:<26}{r['frames']:>8}{e:>13}{v:>11}   "
              f"{r['where'] if not r['pass'] else ''}")
    print(f"\n{bad} labs FAIL the transform identity, {unk} predate the fix and "
          f"cannot be verified")
