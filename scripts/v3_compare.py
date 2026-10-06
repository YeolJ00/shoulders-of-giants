"""Every scoring run side by side, against the RIGHT reference row.

The reference is IntPhys 2, not IntPhys 1. IntPhys 1 (coloured shapes, plain backgrounds)
is where 96.7-100% comes from; IntPhys 2 (realistic objects, complex backgrounds) reports
most models at CHANCE. Our scenes are the second regime, so "at chance" is the published
state of the art here, not a failure of the harness.

Runs are labelled by whether they used the leaky HF gather or a masked context encode, so
a withdrawn number can never be quoted next to a corrected one without the difference
being visible.

    /home/jooyeolyun/anaconda3/envs/warp/bin/python scripts/v3_compare.py
"""
import json
from math import sqrt
from pathlib import Path

import numpy as np

R = Path("docs/results")
RUNS = [
    ("V-JEPA 2 ViT-L",   "v1_protocol_vitl_f64.json",  "max", "HF gather -- LEAKY"),
    ("V-JEPA 2.1 ViT-g", "v21_v1.json",                "max", "HF-style gather -- LEAKY"),
    ("V-JEPA 2 ViT-L",   "v3_intphys2_v1.json",        "max", "masked context, IntPhys2"),
    ("V-JEPA 2.1 ViT-g", "v21fix_v1.json",             "max", "masked context"),
]
SUST = {"freeze", "through", "vanish"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def acc(rows, stat, sel):
    V = [(k, r) for k, r in rows.items()
         if r.get("probe") == "violation" and r.get("pair_with") in rows and sel(r, rows)]
    d = [100 * (r[stat] - rows[r["pair_with"]][stat]) / rows[r["pair_with"]][stat]
         for _, r in V if stat in r and stat in rows[r["pair_with"]]]
    if not d:
        return None
    ok = sum(x > 0 for x in d); lo, hi = wilson(ok, len(d))
    return ok, len(d), 100 * ok / len(d), 100 * lo, 100 * hi, float(np.median(d))


def main():
    print("REFERENCE (IntPhys 2, arXiv 2506.09849): most models at CHANCE (50%) on "
          "realistic scenes;\n            humans near-perfect. The 96.7-100% figure is "
          "IntPhys 1 -- coloured shapes on plain backgrounds.\n")
    cuts = [("all five kinds", lambda r, R_: True),
            ("sustained only", lambda r, R_: r["param"] in SUST),
            ("teleport", lambda r, R_: r["param"] == "teleport")]
    for model, fn, stat, note in RUNS:
        f = R / fn
        if not f.exists():
            print(f"{model:<18} {note:<26} (not yet written)"); continue
        rows = json.loads(f.read_text())["rows"]
        print(f"{model:<18} {note}")
        for nm, sel in cuts:
            a = acc(rows, stat, sel)
            if a is None:
                continue
            ok, n, pct, lo, hi, med = a
            flag = ("above chance" if lo > 50 else
                    "BELOW chance" if hi < 50 else "at chance")
            print(f"    {nm:<18}{ok:>3}/{n:<3} = {pct:5.1f}%  CI [{lo:4.1f},{hi:5.1f}]"
                  f"  median gap {med:+6.2f}%   {flag}")
        print()


if __name__ == "__main__":
    main()
