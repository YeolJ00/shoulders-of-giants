"""L2 -- evaluation of a trained model on held-out objects (DIRECTION.md gates G1, G2, G4).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l2_eval.py --ckpt outputs/lsim/train_v0/last.pt --data data/datasets/v0_test
    $L scripts/l2_eval.py --ckpt cv ...      (B0: constant velocity, the reference for resting bodies)

G1  rollouts of 30 frames (1 s) from the true past, every slot given: vertex error at 1 / 8 / 30
    frames (mean over a body's vertices, median and p90 over bodies), penetration below the
    table top.
G2  rollouts from frame 8 to the episode's end (5.7 s) -- every slot given; every parameter
    hidden; the gripper's force hidden (the hand case) -- error at
    +1/+8/+30/end (compare B3, scripts/l2_teacher_baseline.py), fraction inside a spatial bound
    (2 m of the table centre, above the floor), drift of bodies the teacher kept at rest.
G4  in-context identification: every parameter slot hidden, the parameter head's error per slot
    against the number of past frames present (0-7), split by whether the object moved.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from lsim.data.episode import rotate
from lsim.model.net import build_model
from lsim.model.rollout import rollout
from lsim.train.train import GROUP_NAMES, collate
from lsim.train.windows import CENTRE, PARAMS, SCALE, load_shard, make_window

REPO = Path(__file__).resolve().parents[1]


def body_errors(ep, pred, true, frames):
    """Mean vertex distance per dynamic rigid body at the given rollout frames (0-based)."""
    out = []
    for i, b in enumerate(ep.bodies):
        if b.role != "dynamic" or b.type != "rigid":
            continue
        v = b.verts.astype(float)
        out.append([np.linalg.norm(rotate(pred[h, i, 3:], v) + pred[h, i, :3] - rotate(true[h, i, 3:], v) - true[h, i, :3],
                                   axis=1).mean() for h in frames])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/datasets/v0_test")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.ckpt == "cv":                      # B0: constant velocity, made rigid per body (B2 with zero output)
        model = build_model({"direct": True}, len(PARAMS)).to(dev).eval()
    else:
        ck = torch.load(a.ckpt, map_location=dev)
        model = build_model(ck["args"], len(PARAMS)).to(dev)
        model.load_state_dict(ck["model"]); model.eval()
    eps = [e for s in sorted(glob.glob(f"{a.data}/shard_*.npz")) for e in load_shard(s[:-4])
           if any(b.role == "dynamic" and b.type == "rigid" for b in e.bodies)][:a.episodes]
    rng = np.random.default_rng(0)
    res = {"ckpt": a.ckpt, "data": a.data, "episodes": len(eps)}

    # G1: 30-frame rollouts from three start frames per episode
    errs, pen = [], []
    for e in eps:
        for t0 in (30, 90, 140):
            pred, true, _ = rollout(model, e, t0, 30)
            errs += body_errors(e, pred, true, (0, 7, 29))
            for i, b in enumerate(e.bodies):
                if b.role == "dynamic" and b.type == "rigid":
                    z = np.stack([(rotate(pred[h, i, 3:], b.verts.astype(float)) + pred[h, i, :3])[:, 2] for h in range(30)])
                    on_table = np.abs(np.stack([pred[h, i, :2] for h in range(30)])).max() < 0.5
                    if on_table:
                        pen.append(float(max(0.0, -z.min())))
    E = 1000 * np.array(errs)
    res["G1"] = {"bodies": len(E), "median_mm": np.median(E, 0).round(2).tolist(), "p90_mm": np.percentile(E, 90, 0).round(1).tolist(),
                 "frames": [1, 8, 30], "penetration_median_mm": 1000 * float(np.median(pen)), "penetration_p90_mm": 1000 * float(np.percentile(pen, 90))}
    print(f"G1 30-frame rollouts ({len(E)} bodies): error at 1/8/30 frames median {res['G1']['median_mm']} mm, "
          f"p90 {res['G1']['p90_mm']} mm; penetration median {res['G1']['penetration_median_mm']:.2f} mm", flush=True)

    # G2: rollouts from frame 8 to the end of the episode, slots given and all hidden; the errors
    # at +1/+8/+30/end compare with B3 (scripts/l2_teacher_baseline.py, same horizons)
    for tag, hidden, grip_hidden in (("given", (), False), ("params_hidden", GROUP_NAMES, False),
                                     ("grip_force_hidden", (), True)):
        inside, rest_drift, errs = [], [], []
        for e in eps[:32]:
            t0 = 8
            H = e.q.shape[0] - 1 - t0
            pred, true, _ = rollout(model, e, t0, H, hidden=hidden, grip_hidden=grip_hidden)
            errs += body_errors(e, pred, true, (0, 7, 29, H - 1))
            for i, b in enumerate(e.bodies):
                if b.role != "dynamic" or b.type != "rigid":
                    continue
                inside.append(bool(np.abs(pred[:, i, :2]).max() < 2.0 and pred[:, i, 2].min() > -0.80))
                if np.linalg.norm(true[-1, i, :3] - true[0, i, :3]) < 1e-3:          # the teacher kept it at rest
                    rest_drift.append(float(np.linalg.norm(pred[-1, i, :3] - pred[0, i, :3])))
        E = 1000 * np.array(errs)
        res[f"G2_{tag}"] = {"bodies": len(inside), "inside_bound": float(np.mean(inside)), "horizon_frames": H,
                            "error_median_mm": np.median(E, 0).round(2).tolist(), "error_p90_mm": np.percentile(E, 90, 0).round(1).tolist(),
                            "rest_drift_median_mm": 1000 * float(np.median(rest_drift)) if rest_drift else None,
                            "rest_drift_p90_mm": 1000 * float(np.percentile(rest_drift, 90)) if rest_drift else None}
        print(f"G2 [{tag}] rollouts from frame 8 ({len(inside)} bodies, {H} frames): error at +1/+8/+30/end median "
              f"{res[f'G2_{tag}']['error_median_mm']} mm; inside bound {res[f'G2_{tag}']['inside_bound']:.3f}; resting drift "
              f"median {res[f'G2_{tag}']['rest_drift_median_mm']} mm", flush=True)

    # G4: parameter inference with every slot hidden, against past frames present
    rows, labels = [], []
    for e in eps:
        for t in rng.integers(9, e.q.shape[0] - 9, 4):
            w = make_window(e, int(t), 8, 0, rng, p_hide_grip=0.0)
            moved = {i: float(np.linalg.norm(e.q[t, b.body, :3] - e.q[t - 7, b.body, :3])) for i, b in enumerate(e.bodies)}
            for n_prev in (0, 1, 3, 7):
                b = collate([w], dev, np.random.default_rng(int(t)), p_hist_noise=0.0, augment=False, all_past=False,
                            hidden=GROUP_NAMES)
                b["absent"][:] = False
                b["absent"][:, :8 - 1 - n_prev] = True
                Fw = b["pos"].shape[1]
                with torch.no_grad():
                    _, _, pred = model(dict(b, tau=torch.zeros(Fw, device=dev), frame_given=torch.ones(Fw, device=dev)),
                                       lambda pr: b["body_mass"])
                pred = pred.cpu().numpy(); true = b["params"].cpu().numpy(); has = b["param_has"].cpu().numpy()
                for i, bd in enumerate(e.bodies):
                    if bd.role != "dynamic" or bd.type != "rigid":
                        continue
                    rows.append((n_prev, moved[i] > 0.01, *(np.abs(pred[i] - true[i]) * SCALE)[:9]))
                    if n_prev == 0:
                        labels.append(true[i, :9] * SCALE[:9] + CENTRE[:9])
    R = np.array(rows, float)
    Lb = np.array(labels)                                                    # the prior: always guess the mean
    prior = np.median(np.abs(Lb - Lb.mean(0)), 0)
    g4 = {"prior_mean_guess": {"n": len(Lb), "mass_abs_log": float(prior[0]), "mu_abs_log": float(prior[1]),
                               "e_abs_logit": float(prior[2]), "com_abs_frac": float(prior[3:6].mean())}}
    for n_prev in (0, 1, 3, 7):
        for mv in (0, 1):
            sel = (R[:, 0] == n_prev) & (R[:, 1] == mv)
            if sel.any():
                g4[f"past{n_prev}_{'moving' if mv else 'still'}"] = {
                    "n": int(sel.sum()), "mass_abs_log": float(np.median(R[sel, 2])), "mu_abs_log": float(np.median(R[sel, 3])),
                    "e_abs_logit": float(np.median(R[sel, 4])), "com_abs_frac": float(np.median(R[sel, 5:8].mean(1)))}
    res["G4"] = g4
    for k, v in g4.items():
        print(f"G4 {k:16s} n {v['n']:4d}: |log m| {v['mass_abs_log']:.2f}  |log mu| {v['mu_abs_log']:.2f}  "
              f"|logit e| {v['e_abs_logit']:.2f}  |com|/size {v['com_abs_frac']:.3f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
