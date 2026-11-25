"""Which motions and which materials carry the network's one-frame error? (2026-11-21)

The fully trained cloth networks leave ~0.63 of constant velocity's first-step error on moving free cloth; the error is
calibrated shrinkage (predicted acceleration regressed on the true 0.645, best rescale 0.98) at the motion's own spatial
scale, and on the most-moving windows it misses bands of acceleration that sweep across the cloth
(l2_error_scale, l2_error_field_plot). Hypothesis (waves): the part not learned is wave propagation -- bending waves
(dispersive, omega = sqrt(B / w) k^2: a 5 cm wave in a typical towel has a period of ~2 frames) and drag-driven
flutter. If so, the shortfall should grow with the bending wave coefficient sqrt(B / w) and be largest in the
families that excite waves (shake, the drop after a lift, the fling of a fold), smallest in slow drags and slides.

Per window (the same windows as l2_error_where, rng 3; more of them), the summed first-step error and constant
velocity's on moving free cloth (unheld, off table / objects / self-contact / the 4 cm around the grasp; clean history,
every slot given), grouped by the family running at that frame, its phase (before the grasp, held, after the release),
the start, and terciles of each material parameter, cloth size and speed; the ratio per group with a bootstrap
interval over windows.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_error_by_kind.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt --tag q64_cont
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def phase_family(ep, t):
    """The family running at recorded frame t and its phase (pre: before its grasp, held, post: after the release)."""
    T = t + int(ep.meta.get("settle_frames", 0))
    fams = [f for f in ep.wd.get("families", []) if f["start"] <= T]
    fam = fams[-1]["family"] if fams else "none"
    ev = [e for e in ep.wd.get("events", []) if e[0] <= T]
    last = ev[-1][1] if ev else None
    ph = "held" if last == "grasp" else ("post" if last == "release" else "pre")
    return fam, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="c1_test")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    from l2_error_where import classify
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import step_loss
    from lsim.train.cloth_windows import collate_cloth, load_shard, make_cloth_window
    dev = "cuda"
    eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz"))) for e in load_shard(p[:-4])]
    rng = np.random.default_rng(3)                                       # cloth_train._windows' draws, kept
    m = load_model(a.ckpt, dev)
    P, K = 8, 8
    rows = []
    with torch.no_grad():
        for i in range(a.n):
            ei = int(rng.integers(len(eps)))
            e = eps[ei]
            t = int(rng.integers(1, e.n_frames - K))
            w = make_cloth_window(e, t, P, K, rng)
            b = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
            parts = {}
            step_loss(m, b, np.random.default_rng(i), K, dev, sig=0.0, parts=parts)
            x, fr = parts["x"][:, 0].double(), parts["free"][:, 0]
            pos = b["pos"].double()
            tgt, cv = pos[:, P], 2 * pos[:, P - 1] - pos[:, P - 2]
            spd = (pos[:, P - 1] - pos[:, P - 2]).norm(dim=-1) * 30
            sel = fr & (spd > 0.02) & classify(b, P)["free"]
            n = int(sel.sum())
            if n == 0:
                continue
            fam, ph = phase_family(e, t)
            c = e.c
            rows.append({"i": i, "ep": ei, "t": t, "n": n, "fam": fam, "phase": ph, "init": e.wd.get("init", "?"),
                         "err": float((x - tgt)[sel].norm(dim=-1).sum()), "cv": float((cv - tgt)[sel].norm(dim=-1).sum()),
                         "w": float(c["w"]), "Y": float(c["Y"]), "B": float(c["B"]), "mu": float(c["mu"]),
                         "lb": float(c["bending_length"]), "wave": float(np.sqrt(c["B"] / c["w"])),
                         "size": float(max(c["nx"], c["ny"]) * c["h"]), "speed": float(spd[sel].mean())})
            if i % 250 == 0:
                print(f"  {i} windows, {len(rows)} with moving free cloth", flush=True)
    brng = np.random.default_rng(0)
    E = np.array([r["err"] for r in rows]); C = np.array([r["cv"] for r in rows])

    def group(mask):
        idx = np.nonzero(mask)[0]
        if len(idx) < 5:
            return None
        bs = [E[j].sum() / C[j].sum() for j in (brng.choice(idx, len(idx)) for _ in range(500))]
        return {"windows": int(len(idx)), "vertices": int(sum(rows[j]["n"] for j in idx)),
                "ratio": float(E[idx].sum() / C[idx].sum()), "lo": float(np.percentile(bs, 2.5)),
                "hi": float(np.percentile(bs, 97.5)), "cv_mm": float(1e3 * C[idx].sum() / sum(rows[j]["n"] for j in idx))}

    out = {"ckpt": a.ckpt, "data": a.data, "n": a.n, "all": group(np.ones(len(rows), bool)), "by": {}}
    for key in ("fam", "phase", "init"):
        vals = np.array([r[key] for r in rows])
        out["by"][key] = {v: group(vals == v) for v in sorted(set(vals))}
    fp = np.array([f"{r['fam']}/{r['phase']}" for r in rows])
    out["by"]["fam_phase"] = {v: group(fp == v) for v in sorted(set(fp))}
    for key in ("wave", "lb", "w", "Y", "B", "mu", "size", "speed"):
        v = np.array([r[key] for r in rows])
        q = np.percentile(v, [100 / 3, 200 / 3])
        out["by"][key] = {f"low (<{q[0]:.3g})": group(v < q[0]), f"mid": group((v >= q[0]) & (v < q[1])),
                          f"high (>={q[1]:.3g})": group(v >= q[1])}
        # per-window ratio against the parameter, weighted by constant velocity's error (rank correlation)
        from scipy.stats import spearmanr
        rr = E / np.maximum(C, 1e-12)
        out["by"][key]["spearman_window_ratio"] = float(spearmanr(v, rr).statistic)
    out["rows"] = rows
    a_ = out["all"]
    print(f"all: {a_['windows']} windows, ratio {a_['ratio']:.3f} [{a_['lo']:.3f}, {a_['hi']:.3f}]")
    for key, d in out["by"].items():
        print(f"-- {key}")
        for v, g in d.items():
            if isinstance(g, dict):
                print(f"   {v:28s} {g['windows']:4d} win {g['vertices']:7d} vtx  cv {g['cv_mm']:.3f} mm  ratio "
                      f"{g['ratio']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}]")
            elif g is not None:
                print(f"   {v:28s} {g:+.3f}")
    json.dump(out, open(REPO / f"docs/results/l2_error_by_kind_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
