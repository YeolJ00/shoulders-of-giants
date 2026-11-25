"""L2 -- the trained model against real footage: Physics 101 cardboard slides (20 deg board).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l2_phys101.py --ckpt outputs/lsim/train_v0/last.pt

Exactly the teacher gate's protocol (scripts/l0_teacher_gate.py, check_phys101) with the learned
model in place of the teacher: each real cube (edge from its measured volume, mass measured)
starts at rest on a 20 deg plank, friction swept (object and plank alike, so the pair takes that
value), rolled out frame by frame; every real trial is aligned in release time and offset only,
and the friction with the smallest error wins. Kamino did this at 0.69 mm median RMS.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lsim.data.scene import ramp_frame
from lsim.model.net import build_model
from lsim.model.rollout import rollout
from lsim.real.phys101 import align_err, objects, ramp_segment, ramp_tracks
from lsim.real.synthetic import ramp_cube
from lsim.train.windows import PARAMS

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="docs/results/l2_phys101.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev)
    args = ck["args"]
    model = build_model(args, len(PARAMS)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    objs, R = objects(), ramp_tracks("cardboard")
    MUS = np.round(np.arange(0.15, 0.351, 0.02), 3)
    P, H = 8, 36
    out = {}
    for cube in sorted({r["clip"].split("/")[0] for r in R}):
        mass_g, vol_cm3 = objs[cube]
        edge = vol_cm3 ** (1 / 3) / 100.0
        curves = {}
        for mu in MUS:
            ep, ramp = ramp_cube(edge, mass_g / 1000.0, float(mu), 20.0, 0.42, P + H)
            pred, _, _ = rollout(model, ep, P - 1, H)
            _, ax, _, _ = ramp_frame(ramp)
            traj = np.concatenate([ep.q[P - 1, :1, :3], pred[:, 0, :3]])  # released at "now"
            curves[float(mu)] = (traj - traj[0]) @ (-ax)                     # metres travelled down the slope
        trials = [t for t in R if t["clip"].startswith(cube + "/20") and t.get("fit")]
        per = []
        for t in trials:
            rt, rs = ramp_segment(t)
            errs = np.array([align_err(rt, rs, curves[float(u)], FPS)[0] for u in MUS])
            i = int(np.argmin(errs))
            per.append({"clip": t["clip"], "mu": float(MUS[i]), "rms_mm": 1000 * float(errs[i])})
        out[cube] = {"edge_cm": 100 * edge, "mass_g": mass_g, "n": len(per),
                     "mu_model": float(np.median([p["mu"] for p in per])) if per else None,
                     "rms_model_mm": float(np.median([p["rms_mm"] for p in per])) if per else None, "trials": per}
        print(f"phys101 {cube}: mu {out[cube]['mu_model']}, rms {out[cube]['rms_model_mm']:.2f} mm over {len(per)} trials", flush=True)
    allr = [p["rms_mm"] for o in out.values() for p in o["trials"]]
    res = {"ckpt": a.ckpt, "median_rms_mm": float(np.median(allr)), "p90_rms_mm": float(np.percentile(allr, 90)),
           "trials": len(allr), "teacher_kamino_median_rms_mm": 0.69, "cubes": out}
    print(f"ALL {len(allr)} trials: median RMS {res['median_rms_mm']:.2f} mm, p90 {res['p90_rms_mm']:.2f} mm (Kamino 0.69 mm)", flush=True)
    (REPO / a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
