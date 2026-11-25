"""Is the network's missing acceleration across the cloth or along it? (2026-11-21)

The one-frame error on moving free cloth is calibrated shrinkage at the acceleration field's own spatial scale
(l2_error_scale). Two candidate mechanisms predict different directions: bending and drag waves (a sheet flexing,
fluttering) act along the cloth's normal; tension -- the near-inextensible sheet's constraint force, set over the whole
cloth within a frame (a stretch wave crosses ~10 m per frame) -- acts in its plane. Each vertex's true acceleration
(x_{t+1} minus constant velocity), the network's, and the error are split into the component along the vertex normal
at frame t (area-weighted over its triangles) and the in-plane rest; per component the share of constant velocity's
squared error, the regression of the predicted on the true, the correlation and error / constant velocity. Also
apart for the cloth's border (within 1.5 cm of its rest outline) and its interior. l2_error_where's 300 c1_test
windows (rng 3), clean history, every slot given.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_error_normal.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt --tag q64_cont
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def vertex_normals(x, tris, N):
    a, b, c = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
    fn = torch.cross(b - a, c - a, dim=-1)                               # |fn| = 2 x area: area-weighted
    n = torch.zeros(N, 3, dtype=x.dtype, device=x.device)
    for k in range(3):
        n.index_add_(0, tris[:, k], fn)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


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
    keys = [(r, c) for r in ("all", "border", "interior") for c in ("normal", "plane")]
    acc = {k: {"aa": 0.0, "pa": 0.0, "pp": 0.0, "ee": 0.0, "e_abs": 0.0, "a_abs": 0.0, "n": 0} for k in keys}
    tot_aa = {r: 0.0 for r in ("all", "border", "interior")}
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
            nrm = vertex_normals(pos[:, P - 1], b["tris"], pos.shape[0])
            uv = b["rest"][:, :2].double()                               # rest coordinates / size, centred
            sz = b["size"][b["sample_of"]].double()
            cl = b["is_cloth"]
            lo, hi = uv[cl].min(0).values, uv[cl].max(0).values
            dist_edge = torch.minimum(uv - lo, hi - uv).min(-1).values * sz   # m from the rest outline (a rectangle)
            border = dist_edge < 0.015
            A, Ah = tgt - cv, x - cv
            n = nrm
            comp = {"normal": lambda F: (F * n).sum(-1, keepdim=True) * n,
                    "plane": lambda F: F - (F * n).sum(-1, keepdim=True) * n}
            for r, mask in (("all", sel), ("border", sel & border), ("interior", sel & ~border)):
                if not mask.any():
                    continue
                tot_aa[r] += float((A[mask] ** 2).sum())
                for c, f in comp.items():
                    Ac, Pc = f(A)[mask], f(Ah)[mask]
                    E = Pc - Ac
                    d = acc[(r, c)]
                    d["aa"] += float((Ac * Ac).sum()); d["pa"] += float((Pc * Ac).sum()); d["pp"] += float((Pc * Pc).sum())
                    d["ee"] += float((E * E).sum()); d["e_abs"] += float(E.norm(dim=-1).sum())
                    d["a_abs"] += float(Ac.norm(dim=-1).sum()); d["n"] += int(mask.sum())
            if i % 100 == 0:
                print(f"  {i} windows", flush=True)
    out = {"ckpt": a.ckpt, "data": a.data, "n": a.n, "parts": {}}
    for (r, c), d in acc.items():
        if d["n"] == 0:
            continue
        out["parts"][f"{r}/{c}"] = {"vertices": d["n"], "share_of_cv_sq": d["aa"] / tot_aa[r],
                                    "share_of_error_sq": d["ee"] / sum(acc[(r, cc)]["ee"] for cc in ("normal", "plane")),
                                    "beta": d["pa"] / d["aa"], "corr": d["pa"] / np.sqrt(d["aa"] * d["pp"]),
                                    "rms_ratio": float(np.sqrt(d["ee"] / d["aa"])),
                                    "cv_rms_mm": 1e3 * float(np.sqrt(d["aa"] / d["n"]))}
        q = out["parts"][f"{r}/{c}"]
        print(f"{r:9s} {c:7s} {d['n']:7d} vtx  share of constant velocity's sq. error {q['share_of_cv_sq']:.3f}, of the "
              f"network's {q['share_of_error_sq']:.3f}; predicted on true {q['beta']:.3f}, corr {q['corr']:.3f}, rms error / "
              f"constant velocity's {q['rms_ratio']:.3f} (cv rms {q['cv_rms_mm']:.3f} mm)")
    json.dump(out, open(REPO / f"docs/results/l2_error_normal_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
