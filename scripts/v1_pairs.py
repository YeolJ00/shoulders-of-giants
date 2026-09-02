"""Paired accuracy — the published statistical design, applied to our clips.

Why paired at all: their own raw_surprises show the effect is ~0.4% within a scene
(ViT-L on IntPhys O1-O3: +0.29% to +0.61%) and reverses to -1.77% when pooled ACROSS
scenes. 98% accuracy comes from never crossing a scene boundary, not from a large signal.
Our render noise floor is 0.29% of value, so a 9-point correlation was never going to
resolve an effect of that size. This scores matched pairs instead and reports binomial
confidence intervals.

Three tiers, by how defensible the ground truth is:
  1 violations  categorically impossible          -- unambiguous
  2 gravity     wrong but internally consistent   -- unambiguous, and the bridge to our
                                                     regime (nothing teleports)
  3 theta       our regime                        -- NO ground truth; theta_ref is an
                                                     assumed anchor, reported as such

    /home/jooyeolyun/anaconda3/envs/warp/bin/python scripts/v1_pairs.py [protocol.json]
"""
import json
import sys
from math import sqrt
from pathlib import Path


def wilson(k, n, z=1.96):
    """Binomial CI. With n=5 pairs almost nothing is distinguishable from chance, which
    is exactly why the violation tier now runs on 6 source clips instead of 1."""
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def report(name, pairs, rows, stat, note=""):
    """pairs: list of (implausible_key, plausible_key)."""
    ok, deltas = 0, []
    for bad, good in pairs:
        if bad not in rows or good not in rows:
            continue
        d = rows[bad][stat] - rows[good][stat]
        deltas.append(100 * d / rows[good][stat])
        ok += d > 0
    n = len(deltas)
    if not n:
        return
    lo, hi = wilson(ok, n)
    med = sorted(deltas)[n // 2]
    flag = "" if lo > 0.5 else ("  <- includes chance" if hi > 0.5 else "  <- BELOW chance")
    print(f"  {name:<12} {stat:<5} {ok:>3}/{n:<3} = {100*ok/n:5.1f}%"
          f"  95% CI [{100*lo:4.1f}, {100*hi:4.1f}]   median gap {med:+.2f}%{flag}")
    if note:
        print(f"       {note}")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("docs/results/v1_protocol_vjepa2-vitl-fpc64-256.json")
    if not src.exists():
        src = Path("docs/results/v1_protocol.json")
    blob = json.loads(src.read_text())
    rows = blob["rows"]
    print(f"{src}   {len(rows)} clips\n")
    print("REFERENCE, from their raw_surprises (ViT-L, IntPhys O1-O3):")
    print("  paired accuracy 96.7-100%,  within-scene gap +0.29% to +0.61%")
    print("  the SAME data pooled across scenes: -1.77%, i.e. the wrong sign\n")

    tiers = {"1 violations (impossible)": [], "2 gravity (wrong, legal)": [],
             "3 theta (no ground truth)": []}
    for k, r in rows.items():
        if r["probe"] == "violation" and r.get("pair_with"):
            tiers["1 violations (impossible)"].append((k, r["pair_with"]))
        elif r["probe"] == "gravity" and abs(r["value"] - r["ref"]) > 1e-9:
            ref = next((j for j, q in rows.items()
                        if q["probe"] == "gravity" and q.get("pair_kind") == r.get("pair_kind")
                        and abs(q["value"] - q["ref"]) < 1e-9), None)
            if ref:
                tiers["2 gravity (wrong, legal)"].append((k, ref))
        elif r["probe"] in ("bounce", "settle", "tilt") and abs(r["value"] - r["ref"]) > 1e-9:
            ref = next((j for j, q in rows.items()
                        if q["probe"] == r["probe"] and abs(q["value"] - q["ref"]) < 1e-9), None)
            if ref:
                tiers["3 theta (no ground truth)"].append((k, ref))

    notes = {"3 theta (no ground truth)":
             "theta_ref is the geometric centre of the sweep, not a measured truth: this "
             "row tests consistency with an assumption, not correctness."}
    for name, pairs in tiers.items():
        if not pairs:
            continue
        print(f"{name}   ({len(pairs)} pairs)")
        for stat in ("max", "mean"):
            report("", pairs, rows, stat)
        if name in notes:
            print(f"       {notes[name]}")
        print()

    # the null control, as a paired test: mass cannot change the pixels, so this should
    # sit at chance no matter what the other tiers do
    null = [(k, next((j for j, q in rows.items()
                      if q["probe"] == "null" and abs(q["value"] - q["ref"]) < 1e-9), None))
            for k, r in rows.items()
            if r["probe"] == "null" and abs(r["value"] - r["ref"]) > 1e-9]
    null = [(a, b) for a, b in null if b]
    if null:
        print("NULL CONTROL (mass; must sit at chance)")
        for stat in ("max", "mean"):
            report("mass", null, rows, stat)


if __name__ == "__main__":
    main()
