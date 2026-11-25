"""L2 -- G4 for cloth: does the model read a cloth's material from how it moves?

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_cloth_slots.py --ckpt outputs/lsim/cloth_v1/last.pt --data data/datasets/c1_test

Held-out windows with all four slots hidden (areal density, stretch modulus, bending length, friction) and the 8 given
frames full; the slot head's prediction against the truth, in each slot's normalised log units (lsim.train.cloth_windows:
1 unit = the slot's scale, e.g. x/ e^0.6 = 1.8 in density). Scored against the prior's mean (predicting the centre of
the training distribution), per slot and per what the cloth is doing (the frame's motion), and the posterior's
calibration: the share of truths within its 1-sigma and 2-sigma intervals (68% / 95% if calibrated). The rigid models
never left the prior on this test (DIRECTION.md, G4); the cloth identifiability gate says each slot changes the motion in
65-85% of actions, so a model that reads it should beat the prior there.
Writes docs/results/l2_cloth_slots_<tag>.json.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/datasets/c1_test")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--n", type=int, default=256, help="windows")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from lsim.model.cloth_net import build_cloth_model
    from lsim.train.cloth_train import kin_next
    from lsim.train.cloth_windows import CLOTH_PARAMS, C_SCALE, collate_cloth, load_cloth_shard, make_cloth_window
    dev = "cuda"
    eps = [e for p in sorted(glob.glob(f"{a.data}/shard_*.npz"))[a.skip:a.skip + a.shards] for e in load_cloth_shard(p[:-4])]
    ck = torch.load(a.ckpt, map_location=dev)
    args = ck["args"]
    m = build_cloth_model(args).to(dev).eval()
    m.load_state_dict(ck["model"])
    rng = np.random.default_rng(0)
    rows = []
    for i in range(a.n):
        e = eps[rng.integers(len(eps))]
        t = int(rng.integers(8, e.n_frames - 1))
        w = make_cloth_window(e, t, 8, 1, rng)
        b = collate_cloth([w], dev, rng, p_hidden=1.0, augment=False, all_past=True, hidden=CLOTH_PARAMS)
        P = 8
        xw = b["pos"][:, :P]
        prev = torch.cat([xw[:, :1] - b["vel"][:, :1] / 30.0, xw[:, :-1]], 1)
        bj = dict(b, pos=xw, vel=(xw - prev) * 30.0, absent=b["absent"][:, :P], f_ext=b["f_ext"][:, :P], f_state=b["f_state"][:, :P],
                  held=b["held"][:, :P], tau=torch.zeros(P, device=dev), frame_given=torch.ones(P, device=dev), n_past=P)
        bj["kin_next"] = kin_next(bj, b["pos"][:, :P + 1])
        with torch.no_grad():
            _, _, pr, _ = m(bj)
        sd = torch.exp(0.5 * m.slot_logvar)[0].cpu().numpy() if m.prob_slots else np.full(4, np.nan)
        x = e.x(np.arange(t - 7, t + 1))
        motion = float(np.linalg.norm(x[-1] - x[0], axis=-1).mean())                 # over the 8 given frames
        fam = [f["family"] for f in e.wd["families"] if f["start"] - e.meta["settle_frames"] <= t]
        rows.append({"pred": pr[0].cpu().numpy().tolist(), "true": b["params"][0].cpu().numpy().tolist(), "sd": sd.tolist(),
                     "motion_mm": 1e3 * motion, "held": bool(e.held[t].any()), "family": fam[-1] if fam else "settle",
                     "init": e.wd["init"]})
    P_, T_, S_ = (np.array([r[k] for r in rows]) for k in ("pred", "true", "sd"))
    moving = np.array([r["motion_mm"] for r in rows]) > 5.0
    res = {"n": len(rows), "ckpt": a.ckpt, "moving_fraction": float(moving.mean()), "slots": {}}
    for k, name in enumerate(CLOTH_PARAMS):
        err, prior = np.abs(P_[:, k] - T_[:, k]), np.abs(T_[:, k])                    # the prior's mean is 0
        z = np.abs(P_[:, k] - T_[:, k]) / S_[:, k] if m.prob_slots else np.full(len(rows), np.nan)
        row = {"model_abs_err": float(np.median(err)), "prior_abs_err": float(np.median(prior)),
               "model_moving": float(np.median(err[moving])) if moving.any() else None,
               "prior_moving": float(np.median(prior[moving])) if moving.any() else None,
               "model_still": float(np.median(err[~moving])) if (~moving).any() else None,
               "corr": float(np.corrcoef(P_[:, k], T_[:, k])[0, 1]), "factor_model": float(np.exp(np.median(err) * C_SCALE[k])),
               "factor_prior": float(np.exp(np.median(prior) * C_SCALE[k])),
               "within_1sd": float(np.mean(z < 1)), "within_2sd": float(np.mean(z < 2))}
        res["slots"][name] = row
        print(f"{name:8s}: model off by x/ {row['factor_model']:.2f} (median), the prior's mean x/ {row['factor_prior']:.2f}; "
              f"moving cloth {row['model_moving']:.2f} vs prior {row['prior_moving']:.2f} units, still {row['model_still']:.2f}; "
              f"corr {row['corr']:.2f}; posterior holds the truth within 1 / 2 sd: {row['within_1sd']:.2f} / {row['within_2sd']:.2f}",
              flush=True)
    tag = a.tag or Path(a.ckpt).parent.name
    (REPO / f"docs/results/l2_cloth_slots_{tag}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
