"""L2 -- how well the model reproduces its teacher on PGND's real windows (separates imitation from the real gap).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_real_cloth_pgnd_imitation.py --ckpt outputs/lsim/cloth_v3/last.pt --tag v3

On every held-out window (scripts/l2_real_cloth_pgnd_eval.py's), from frame 7 with the held patches on the robot's
path: the teacher with its fitted material for that cloth (scripts/l2_real_cloth_pgnd_teacher.py) against the model
given that material (all four slots) and with it hidden. Per frame 30 / 60 / 120: the median vertex distance between
the model's cloth and the teacher's, and each one's distance from the observed cloud where there is one.

Writes docs/results/l2_real_cloth_pgnd_imitation_<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
T0 = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / "scripts"))
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.real import replay
    from lsim.real.pgnd import load_pgnd
    from lsim.train.cloth_train import rollout
    mats = {c: json.loads((REPO / f"docs/results/l2_real_cloth_pgnd_teacher_{c}.json").read_text())["material"]
            for c in ("cloth6", "cali2")}
    model = load_model(REPO / a.ckpt, "cuda")
    out = {}
    for ep in load_pgnd("test")[:: a.every]:
        if 120 not in ep.clouds:
            continue
        m = mats[ep.meta["case"].split("_")[0]]
        Xt = replay.simulate(ep, [m], T0, ep.n_frames - 1 - T0, log=lambda *x: None)[0]   # frame T0 first
        ep.c.update({"w": m["w"], "Y": m["Y"], "bending_length": m["c"], "mu": m["mu"]})
        row = {}
        with torch.no_grad():
            for k, hid in (("given", ()), ("hidden", ("density", "stretch", "bend", "mu"))):
                P = rollout(model, ep, T0, ep.n_frames - 1 - T0, dev="cuda", hidden=hid)[0]   # frame T0 + 1 first
                row[k] = {str(f): 1e3 * float(np.median(np.linalg.norm(P[f - T0 - 1] - Xt[f - T0], axis=-1)))
                          for f in (30, 60, 120)}
                row[k].update({f"cloud{f}": 1e3 * ep.cloud_error(P[f - T0 - 1], f) for f in (60, 120)})
        row["teacher"] = {f"cloud{f}": 1e3 * ep.cloud_error(Xt[f - T0], f) for f in (60, 120)}
        out[ep.meta["case"]] = row
        print(f"{ep.meta['case']:18s} model-vs-teacher given " + " ".join(f"{f}:{row['given'][f]:5.1f}" for f in ("30", "60", "120"))
              + " hidden " + " ".join(f"{f}:{row['hidden'][f]:5.1f}" for f in ("30", "60", "120"))
              + f" | to cloud 60/120: teacher {row['teacher']['cloud60']:5.1f}/{row['teacher']['cloud120']:5.1f}"
              f" model {row['given']['cloud60']:5.1f}/{row['given']['cloud120']:5.1f}", flush=True)
    summ = {f"{k}@{f}": float(np.mean([r[k][f] for r in out.values()])) for k in ("given", "hidden")
            for f in ("30", "60", "120", "cloud60", "cloud120")}
    summ.update({f"teacher@{f}": float(np.mean([r["teacher"][f] for r in out.values()])) for f in ("cloud60", "cloud120")})
    print(json.dumps(summ, indent=1), flush=True)
    (REPO / f"docs/results/l2_real_cloth_pgnd_imitation_{a.tag}.json").write_text(json.dumps(
        {"ckpt": a.ckpt, "materials": mats, "summary": summ, "windows": out}, indent=1))


if __name__ == "__main__":
    main()
