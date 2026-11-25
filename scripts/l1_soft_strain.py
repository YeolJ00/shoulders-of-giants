"""How much the soft bodies of a set of shards visibly deform: per episode the largest principal strain (p99 over
tets, max over frames; lsim.data.soft.strain_p99), summarised per category and per action family.

    CUDA_VISIBLE_DEVICES="" $L scripts/l1_soft_strain.py <shard dir> [--out json]
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    from lsim.data.soft import deformation
    d = Path(sys.argv[1])
    rows = []
    for js in sorted(d.glob("shard_*.json")):
        m = json.loads(js.read_text())
        z = np.load(js.with_suffix(".npz"))
        off, toff = z["cloth_off"], z["tet_off"]
        rest = z["soft_rest"].astype(np.float64)
        for w, wd in enumerate(m["worlds"]):
            T = z["soft_tets"][toff[w]:toff[w + 1]].astype(int)
            r = rest[off[w]:off[w + 1]]
            Dm = np.linalg.inv(np.stack([r[T[:, 1]] - r[T[:, 0]], r[T[:, 2]] - r[T[:, 0]], r[T[:, 3]] - r[T[:, 0]]], -1))
            x = z["cloth_d"][:, off[w]:off[w + 1]].astype(np.float64) + z["cloth_c"][:, w][:, None]
            q = wd["qc"]
            dfm = {k: q[k] for k in ("strain_p99", "strain_p50", "shape_dev")} if "shape_dev" in q else deformation(x, T, Dm, r)
            rows.append({"cat": wd["soft"]["params"].get("category", "?"), "fams": q["families"] or ["none"], **dfm,
                         "inv": q["inverted"], "init": wd["init"]})
    soft = {"sponge", "memory_foam", "plush", "gel", "closed_foam", "bread"}

    def summ(rs):
        o = {"n": len(rs)}
        for k, thr in (("strain_p99", 0.15), ("strain_p50", 0.05), ("shape_dev", 0.05)):
            v = np.asarray([r[k] for r in rs])
            o[k] = {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)), f"frac_gt_{thr}": float((v > thr).mean())}
        return o
    out = {"all": summ(rows), "category": {}, "family": {}, "family_soft": {}}
    for c in sorted({r["cat"] for r in rows}):
        out["category"][c] = summ([r for r in rows if r["cat"] == c])
    for f in sorted({f for r in rows for f in r["fams"]}):
        out["family"][f] = summ([r for r in rows if f in r["fams"]])
        v = [r for r in rows if f in r["fams"] and r["cat"] in ("sponge", "memory_foam", "plush", "gel")]
        if v:
            out["family_soft"][f] = summ(v)
    out["inverted_frac"] = float(np.mean([r["inv"] for r in rows]))
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    for k in ("category", "family", "family_soft"):
        for c, v in out[k].items():
            print(f"{k:11s} {c:12s} n {v['n']:3d} | strain p99: p50 {v['strain_p99']['p50']:.3f} >15% {v['strain_p99']['frac_gt_0.15']:.2f}"
                  f" | median-tet strain p50 {v['strain_p50']['p50']:.3f} >5% {v['strain_p50']['frac_gt_0.05']:.2f}"
                  f" | shape dev p50 {v['shape_dev']['p50']:.3f} p90 {v['shape_dev']['p90']:.3f} >5% {v['shape_dev']['frac_gt_0.05']:.2f}")


if __name__ == "__main__":
    main()
