"""Where on the cloth is the one-step error (2026-11-19)?

The trained model leaves ~64% of the frame-to-frame acceleration unexplained (error / constant velocity's), and the
frame rate explains only part of it (l2_frame_rate_fit). This splits the first-step error of moving free vertices on
held-out windows (clean history, every slot given) by what the vertex is doing at the current frame:
  table    on a flat surface (within 3 mm of a table / floor / plank rectangle)
  object   within 5 mm of a scanned object's surface points
  self     within 1 cm of its own cloth where the rest distance is > 4 cm (folded onto itself)
  grip     within 4 cm (rest) of a held vertex
  free     none of these
Per class: share of vertices, model and constant-velocity error, their ratio, share of the total squared error.
On record_v data (the teacher's instantaneous velocities in the shards) also two references from the true velocity v_t:
x_t + v_t dt ("true_cv") and that plus free fall's g dt^2 / 2 ("true_ballistic"), as ratios to constant velocity.
--material: the same windows again with the material hidden and with the next window's material (another episode's) given;
per class the error ratio of each and how far the prediction moves from the true-material one (/ constant velocity's
error) -- set against l1_teacher_sensitivity (how far the teacher moves under the same change).

    CUDA_VISIBLE_DEVICES=<gpu or ""> $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_q64/step23999.pt
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
CLASSES = ("table", "object", "self", "grip", "free")


def classify(b, P):
    pos, so = b["pos"][:, P - 1], b["sample_of"]
    role, cl = b["role"], b["is_cloth"]
    N = pos.shape[0]
    on = {k: torch.zeros(N, dtype=torch.bool, device=pos.device) for k in CLASSES}
    PL, ps = b["planes"], b["plane_sample"]
    for i in range(PL.shape[0]):
        o, n, u, v, a, bb = PL[i, 0:3], PL[i, 3:6], PL[i, 6:9], PL[i, 9:12], PL[i, 12], PL[i, 13]
        d = pos - o
        inside = ((d * u).sum(-1).abs() <= a) & ((d * v).sum(-1).abs() <= bb)
        h = (d * n).sum(-1)
        on["table"] |= cl & (so == ps[i]) & inside & (h < 0.003 + b["radius"][so])
    st = role == 0                                                     # lsim.train.windows.ROLE: static 0
    if st.any():
        same = so[cl][:, None] == so[st][None]
        dmin = torch.cdist(pos[cl], pos[st]).masked_fill(~same, 1e9).min(1).values
        on["object"][cl] = dmin < 0.005
    rest = b["rest"] * b["size"][so][:, None]
    idx = torch.nonzero(cl).squeeze(1)
    for s in so[cl].unique():
        I = idx[so[idx] == s]
        dx = torch.cdist(pos[I], pos[I])
        dr = torch.cdist(rest[I], rest[I])
        on["self"][I] = ((dx < 0.01) & (dr > 0.04)).any(1)
        H = b["held"][I, P - 1]
        if H.any():
            on["grip"][I] = dr[:, H].min(1).values < 0.04
    on["free"] = cl & ~(on["table"] | on["object"] | on["self"] | on["grip"])
    return on


def probe_mask(b, P, R=4, h=0.01):
    """l1_local_probe's vertices: cloth vertices whose (2R + 1)^2 grid stencil lies inside the cloth, holds no vertex
    held over the steps into and out of frame P - 1, and none within 1 cm of the table or floor at P - 1."""
    from scipy.ndimage import maximum_filter
    from lsim.data.scene import TABLE_HALF
    N = b["pos"].shape[0]
    Vc = int(b["is_cloth"].sum())
    rm = (b["rest"][:Vc, :2] * b["size"][0]).cpu().numpy().astype(np.float64)
    ij = np.round((rm - rm.min(0)) / h).astype(int)
    nx, ny = ij[:, 0].max(), ij[:, 1].max()
    x = b["pos"][:Vc, P - 1].cpu().numpy()
    low = ((np.abs(x[:, 2]) < 0.01) & (np.abs(x[:, 0]) <= TABLE_HALF[0] + 0.01) & (np.abs(x[:, 1]) <= TABLE_HALF[1] + 0.01)) \
        | (x[:, 2] < -0.75 + 0.01)
    bad = np.zeros((ny + 1, nx + 1), np.uint8)
    hb = (b["held"][:Vc, P - 1] | b["held"][:Vc, P - 2]).cpu().numpy() | low
    bad[ij[:, 1], ij[:, 0]] = hb
    bad = maximum_filter(bad, size=2 * R + 1, mode="constant", cval=1) > 0
    out = torch.zeros(N, dtype=torch.bool, device=b["pos"].device)
    out[:Vc] = torch.from_numpy(~bad[ij[:, 1], ij[:, 0]]).to(out.device)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="c1_test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tag", default="q64")
    ap.add_argument("--material", action="store_true", help="also score with the material hidden and with another "
                    "window's material given; report each and how far the prediction moves (the model's response)")
    ap.add_argument("--probe_R", type=int, default=4, help="the probe subset's stencil radius (l1_local_probe_scale: 6)")
    a = ap.parse_args()
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import _windows, step_loss
    from lsim.train.cloth_windows import CLOTH_PARAMS, collate_cloth, load_shard
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz"))) for e in load_shard(p[:-4])]
    wins = _windows(eps, a.n, 8, 8, np.random.default_rng(3))
    m = load_model(a.ckpt, dev)
    if dev == "cpu":
        m.bf16 = False
    P = 8
    acc = {k: [[], [], [], []] for k in CLASSES}
    per_window = {k: [] for k in CLASSES}                              # (sum model error, sum cv error, n) per window
    mat = {v: {k: [[], []] for k in CLASSES} for v in ("hidden", "swap")}   # --material: (error, response) per vertex
    probe = [[], []]                                                    # l1_local_probe's vertices: (model, cv) error
    for i, w in enumerate(wins):
        b = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
        parts = {}
        step_loss(m, b, np.random.default_rng(i), 8, dev, sig=0.0, parts=parts)
        x, fr = parts["x"][:, 0], parts["free"][:, 0]
        pos = b["pos"]
        tgt, cv = pos[:, P], 2 * pos[:, P - 1] - pos[:, P - 2]
        alt = {}
        if a.material:                                                  # the same window, the material hidden / another
            bh = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True,
                               hidden=CLOTH_PARAMS)                     # window's material given in its place
            j = next(((i + d) % len(wins) for d in range(1, len(wins))   # the next window of another material
                      if not np.allclose(wins[(i + d) % len(wins)]["params"], w["params"])), i)
            bs = dict(b, params=torch.from_numpy(np.asarray(wins[j]["params"])).float()[None].to(dev))
            for v, bb in (("hidden", bh), ("swap", bs)):
                pp = {}
                step_loss(m, bb, np.random.default_rng(i), 8, dev, sig=0.0, parts=pp)
                alt[v] = pp["x"][:, 0]
        if "vtrue" in b:
            tcv = pos[:, P - 1] + b["vtrue"][:, P - 1] / 30.0
            tbal = tcv - torch.tensor([0.0, 0.0, 0.5 * 9.81 / 900.0], device=dev)
        moving = (pos[:, P - 1] - pos[:, P - 2]).norm(dim=-1) * 30 > 0.02
        on = classify(b, P)
        sel = fr & moving
        pm = sel & probe_mask(b, P, R=a.probe_R)
        probe[0].append((x[pm] - tgt[pm]).norm(dim=-1).cpu().numpy())
        probe[1].append((cv[pm] - tgt[pm]).norm(dim=-1).cpu().numpy())
        for k in CLASSES:
            s = sel & on[k]
            acc[k][0].append((x[s] - tgt[s]).norm(dim=-1).cpu().numpy())
            acc[k][1].append((cv[s] - tgt[s]).norm(dim=-1).cpu().numpy())
            per_window[k].append([float(acc[k][0][-1].sum()), float(acc[k][1][-1].sum()), int(s.sum())])
            for v, xa in alt.items():
                mat[v][k][0].append((xa[s] - tgt[s]).norm(dim=-1).cpu().numpy())
                mat[v][k][1].append((xa[s] - x[s]).norm(dim=-1).cpu().numpy())
            if "vtrue" in b:
                acc[k][2].append((tcv[s] - tgt[s]).norm(dim=-1).cpu().numpy())
                acc[k][3].append((tbal[s] - tgt[s]).norm(dim=-1).cpu().numpy())
        if i % 50 == 0:
            print(f"  {i} windows", flush=True)
    tot = sum((np.concatenate(acc[k][0]) ** 2).sum() for k in CLASSES)
    ntot = sum(len(np.concatenate(acc[k][0])) for k in CLASSES)
    out = {}
    print(f"{ntot} moving free vertex-steps in {a.n} held-out windows")
    for k in CLASSES:
        e, c = np.concatenate(acc[k][0]), np.concatenate(acc[k][1])
        if not len(e):
            continue
        out[k] = dict(share=len(e) / ntot, model_mm=float(1e3 * e.mean()), cv_mm=float(1e3 * c.mean()), ratio=float(e.mean() / c.mean()),
                      err2_share=float((e ** 2).sum() / tot))
        if acc[k][2]:
            out[k]["true_cv_ratio"] = float(np.concatenate(acc[k][2]).mean() / c.mean())
            out[k]["true_ballistic_ratio"] = float(np.concatenate(acc[k][3]).mean() / c.mean())
        r = out[k]
        print(f"  {k:7s} {100 * r['share']:5.1f}% of vertices   model {r['model_mm']:.3f} mm   constant velocity "
              f"{r['cv_mm']:.3f} mm   ratio {r['ratio']:.2f}   {100 * r['err2_share']:5.1f}% of the squared error"
              + (f"   true-velocity refs {r['true_cv_ratio']:.2f} / {r['true_ballistic_ratio']:.2f}" if "true_cv_ratio" in r else ""))
        if a.material:
            r["material"] = {v: dict(ratio=float(np.concatenate(mat[v][k][0]).mean() / c.mean()),
                                     response=float(np.concatenate(mat[v][k][1]).mean() / c.mean())) for v in mat}
            print(f"           material hidden: ratio {r['material']['hidden']['ratio']:.3f}, prediction moves "
                  f"{r['material']['hidden']['response']:.3f}; another's given: ratio {r['material']['swap']['ratio']:.3f}, "
                  f"moves {r['material']['swap']['response']:.3f} (of constant velocity's error)")
    pe, pc = np.concatenate(probe[0]), np.concatenate(probe[1])
    out["probe"] = dict(R=a.probe_R, n=int(len(pe)), model_mm=float(1e3 * pe.mean()), cv_mm=float(1e3 * pc.mean()), ratio=float(pe.mean() / pc.mean()))
    print(f"  l1_local_probe's vertices (interior, {2 * a.probe_R + 1}x{2 * a.probe_R + 1} stencil clear of held vertices and the table / floor): "
          f"{len(pe)}, model {1e3 * pe.mean():.3f} mm, constant velocity {1e3 * pc.mean():.3f} mm, ratio {pe.mean() / pc.mean():.3f}")
    out["per_window"] = per_window                                    # for paired comparisons (same --data, --n: same windows)
    json.dump(out, open(REPO / f"docs/results/l2_error_where_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
