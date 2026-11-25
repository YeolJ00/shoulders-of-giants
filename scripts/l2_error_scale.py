"""At what spatial scale is the network's one-frame error? (2026-11-21)

The fully trained cloth networks leave ~0.63 of constant velocity's first-step error on moving free cloth, where the
teacher from the same 3 frame positions leaves 0.25 and still 0.30 with every input vertex jittered by 100 um
(l1_teacher_precision): the map does not need finer inputs than the network has. This asks what the error looks like:
on each cloth, the error field e_i = x_hat_{t+1} - x_{t+1} over its unheld vertices is averaged over the vertices
within R of each vertex in rest coordinates (R = 1, 2, 4, 8 cm, and the whole cloth); the share of the squared error
left after removing that local mean is the part rougher than R. Rough error (vertex to vertex: stretch, wrinkles) and
smooth error (whole regions moving wrong: a swing, the pull of the gripper through the cloth) call for different
remedies. The same for constant velocity's error (minus the teacher's acceleration) as the reference field.
Same 300 c1_test windows as l2_error_where (rng 3), clean history, every slot given.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_error_scale.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt --tag q64_cont
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RS = (0.01, 0.02, 0.04, 0.08, 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="c1_test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    from l2_error_where import classify
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import _windows, step_loss
    from lsim.train.cloth_windows import collate_cloth, load_shard
    dev = "cuda"
    eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz"))) for e in load_shard(p[:-4])]
    wins = _windows(eps, a.n, 8, 8, np.random.default_rng(3))
    m = load_model(a.ckpt, dev)
    P = 8
    tot = {f: {"all": 0.0, **{R: 0.0 for R in RS}} for f in ("model", "cv")}
    mean_mm = {"model": [], "cv": []}
    CS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0)
    gain = {"aa": 0.0, "pa": 0.0, "pp": 0.0, "dist": {c: [] for c in CS}}   # predicted vs true acceleration
    with torch.no_grad():
        for i, w in enumerate(wins):
            b = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
            parts = {}
            step_loss(m, b, np.random.default_rng(i), 8, dev, sig=0.0, parts=parts)
            x, fr = parts["x"][:, 0].double(), parts["free"][:, 0]
            pos = b["pos"].double()
            tgt, cv = pos[:, P], 2 * pos[:, P - 1] - pos[:, P - 2]
            moving = (pos[:, P - 1] - pos[:, P - 2]).norm(dim=-1) * 30 > 0.02
            sel = fr & moving & classify(b, P)["free"]
            if not sel.any():
                continue
            A, Ah = (tgt - cv)[sel], (x - cv)[sel]                       # true, predicted second differences
            gain["aa"] += float((A * A).sum()); gain["pa"] += float((Ah * A).sum()); gain["pp"] += float((Ah * Ah).sum())
            for c in CS:                                                 # the prediction's acceleration scaled by c
                gain["dist"][c].append((c * Ah - A).norm(dim=-1).cpu().numpy())
            uv = b["rest"][:, :2].double() * b["size"][b["sample_of"]][:, None].double()      # m, per cloth
            so = b["sample_of"]
            for name, pred in (("model", x), ("cv", cv)):
                e = pred - tgt
                mean_mm[name].append(1e3 * e[sel].norm(dim=-1).cpu().numpy())
                tot[name]["all"] += float((e[sel] ** 2).sum())
                for s in torch.unique(so[sel]):
                    on = fr & (so == s)                                  # the field over this cloth's unheld vertices
                    q = sel & (so == s)
                    d = torch.cdist(uv[q], uv[on])                       # (n_sel, n_on)
                    for R in RS:
                        W = (d <= R).double()
                        loc = W @ e[on] / W.sum(1, keepdim=True)
                        tot[name][R] += float(((e[q] - loc) ** 2).sum())
            if i % 50 == 0:
                print(f"  {i} windows", flush=True)
    out = {"ckpt": a.ckpt, "data": a.data, "n": a.n}
    for name in ("model", "cv"):
        mm = np.concatenate(mean_mm[name])
        out[name] = {"mean_mm": float(mm.mean()), "rms_mm": float(np.sqrt((mm ** 2).mean())),
                     "rough_share": {f"{100 * R:g}cm" if R < 1 else "cloth": tot[name][R] / tot[name]["all"] for R in RS}}
        print(f"{name:5s} mean {out[name]['mean_mm']:.3f} mm (rms {out[name]['rms_mm']:.3f}): share of the squared error "
              "rougher than R: " + "  ".join(f"{k} {v:.3f}" for k, v in out[name]["rough_share"].items()), flush=True)
    # the part of the model's error smoother than R, against constant velocity's whole error
    for R in RS:
        k = f"{100 * R:g}cm" if R < 1 else "cloth"
        sm = tot["model"]["all"] * (1 - out["model"]["rough_share"][k])
        out.setdefault("model_smooth_over_cv", {})[k] = float(np.sqrt(max(sm, 0) / tot["cv"]["all"]))
    print("model error smoother than R, rms / constant velocity's rms: " +
          "  ".join(f"{k} {v:.3f}" for k, v in out["model_smooth_over_cv"].items()))
    cvd = np.concatenate(mean_mm["cv"]) / 1e3
    out["gain"] = {"beta": gain["pa"] / gain["aa"], "c_star": gain["pa"] / gain["pp"],
                   "corr": gain["pa"] / np.sqrt(gain["aa"] * gain["pp"]),
                   "ratio_by_c": {str(c): float(np.concatenate(gain["dist"][c]).mean() / cvd.mean()) for c in CS}}
    g = out["gain"]
    print(f"predicted vs true acceleration: regression of predicted on true {g['beta']:.3f}, best rescale of the "
          f"prediction {g['c_star']:.3f} (squared error), correlation {g['corr']:.3f}; mean error / constant velocity's "
          "with the predicted acceleration scaled by c: " + "  ".join(f"{c} {v:.3f}" for c, v in g["ratio_by_c"].items()))
    json.dump(out, open(REPO / f"docs/results/l2_error_scale_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
