"""The teachers' own edge strain and tet volume, per material: the evidence for the tolerance bands of the exact
layer's edge / volume restoration (lsim.model.cloth_net.restore_shape, 2026-11-03).

For every sampled episode and every `--every`-th frame: each mesh edge's strain L / L0 - 1 (cloth mesh edges, rope
segments, soft-body tet edges) and, for soft bodies, each tet's volume ratio V / V0 (signed: < 0 is inverted).
Reported as percentiles of stretch and compression apart, over all edges x frames, plus per category (rope by its
stretch stiffness EA, soft by its category), and the per-episode worst edge (what a band must not cut).

    CUDA_VISIBLE_DEVICES="" $L scripts/l2_strain_bands.py --out docs/results/l2_strain_bands.json
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")                    # CPU only: the loaders import lsim.teacher
import argparse
import glob
import json

import numpy as np

from lsim.train.cloth_windows import load_shard

Q = (0.5, 0.9, 0.99, 0.999, 0.9999)


def pct(a):
    a = np.asarray(a)
    if a.size == 0:
        return {}
    out = {f"p{100 * q:g}": float(np.quantile(a, q)) for q in Q}
    out["max"] = float(a.max())
    out["n"] = int(a.size)
    return out


def summary(strain):
    s = np.asarray(strain)
    return {"stretch": pct(np.clip(s, 0, None)), "compression": pct(np.clip(-s, 0, None)), "abs": pct(np.abs(s)),
            "frac_abs_over": {f"{t:g}": float((np.abs(s) > t).mean()) for t in (1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 0.1, 0.2, 0.3, 0.4)}}


def edge_strain(ep, frames):
    x = ep.x(frames).astype(np.float64)
    E = ep.edges
    L0 = np.linalg.norm(ep.rest[E[:, 0]] - ep.rest[E[:, 1]], axis=-1).astype(np.float64)
    L = np.linalg.norm(x[:, E[:, 0]] - x[:, E[:, 1]], axis=-1)
    return L / L0 - 1.0                                              # (frames, edges)


def tet_volume(x, T):
    a, b, c, d = (x[..., T[:, k], :] for k in range(4))
    return np.einsum("...i,...i->...", np.cross(b - a, c - a), d - a) / 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloth", default="data/datasets/c1,data/datasets/c1_long")
    ap.add_argument("--rope", default="data/datasets/r1,data/datasets/r1_long")
    ap.add_argument("--soft", default="data/datasets/s1")
    ap.add_argument("--shards", type=int, default=12, help="shards per dataset, spread over it")
    ap.add_argument("--eps", type=int, default=16, help="episodes per shard")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--out", default="docs/results/l2_strain_bands.json")
    a = ap.parse_args()
    res = {}
    for kind, dirs in (("cloth", a.cloth), ("rope", a.rope), ("soft", a.soft)):
        allst, worst, groups, vols, vworst = [], [], {}, [], []
        for d in dirs.split(","):
            ps = sorted(p for p in glob.glob(f"{d}/shard_*.npz") if os.path.exists(p[:-4] + ".json"))
            if not ps:
                continue
            pick = [ps[int(i)] for i in np.linspace(0, len(ps) - 1, min(a.shards, len(ps))).round()]
            for p in dict.fromkeys(pick):
                eps = load_shard(p[:-4])
                for ep in eps[:: max(1, len(eps) // a.eps)][: a.eps]:
                    fr = np.arange(0, ep.n_frames, a.every)
                    s = edge_strain(ep, fr)
                    allst.append(s.ravel().astype(np.float32))
                    worst.append(float(np.abs(s).max()))
                    if kind == "rope":
                        g = f"EA<{10 ** np.ceil(np.log10(ep.c['EA'])):g}"
                    elif kind == "soft":
                        g = str(ep.c.get("params", {}).get("category", ep.c.get("category", "?")))
                    else:
                        g = f"Y<{10 ** np.ceil(np.log10(ep.c['Y'])):g}"
                    groups.setdefault(g, []).append(s.ravel().astype(np.float32))
                    if kind == "soft":
                        T = ep.se.tets.astype(np.int64)
                        V0 = tet_volume(ep.rest.astype(np.float64), T)
                        V = tet_volume(ep.x(fr).astype(np.float64), T) / V0
                        vols.append(V.ravel().astype(np.float32))
                        vworst.append(float(np.abs(V - 1).max()))
                print(f"[{kind}] {p}: {sum(len(x) for x in allst):,} edge samples", flush=True)
        if not allst:
            continue
        s = np.concatenate(allst)
        r = {"all": summary(s), "episodes": len(worst), "episode_worst_abs": pct(worst),
             "groups": {g: summary(np.concatenate(v)) for g, v in sorted(groups.items())}}
        if vols:
            v = np.concatenate(vols)
            r["volume_ratio"] = {"below_1": pct(np.clip(1 - v, 0, None)), "above_1": pct(np.clip(v - 1, 0, None)),
                                 "frac_below": {f"{t:g}": float((v < 1 - t).mean()) for t in (0.1, 0.2, 0.35)},
                                 "frac_above": {f"{t:g}": float((v > 1 + t).mean()) for t in (0.02, 0.05, 0.1)},
                                 "inverted_frac": float((v <= 0).mean()), "episode_worst_abs": pct(vworst)}
        res[kind] = r
        A = r["all"]
        print(f"== {kind}: {r['episodes']} episodes; |strain| p50 {A['abs']['p50']:.2e} p99 {A['abs']['p99']:.2e} "
              f"p99.9 {A['abs']['p99.9']:.2e} p99.99 {A['abs']['p99.99']:.2e} max {A['abs']['max']:.2e}; stretch p99.9 "
              f"{A['stretch']['p99.9']:.2e} compression p99.9 {A['compression']['p99.9']:.2e}", flush=True)
        for g, gs in r["groups"].items():
            print(f"   {g:14s} |strain| p99 {gs['abs']['p99']:.2e} p99.9 {gs['abs']['p99.9']:.2e} max {gs['abs']['max']:.2e}"
                  f"  stretch p99.9 {gs['stretch']['p99.9']:.2e} compr p99.9 {gs['compression']['p99.9']:.2e}", flush=True)
        if vols:
            vr = r["volume_ratio"]
            print(f"   volume: below 1 by p99 {vr['below_1']['p99']:.2e} p99.9 {vr['below_1']['p99.9']:.2e} max "
                  f"{vr['below_1']['max']:.2e}; above 1 by p99 {vr['above_1']['p99']:.2e} p99.9 {vr['above_1']['p99.9']:.2e} "
                  f"max {vr['above_1']['max']:.2e}; inverted {vr['inverted_frac']:.2e}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
