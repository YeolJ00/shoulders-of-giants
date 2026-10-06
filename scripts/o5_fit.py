"""O3 -- the actual fit. Recover mu for one object from its own clips.

The readout work (o4_joint) established that the head READS the outcome. This asks the
question that matters: does an objective built from the head's output have a minimum at the
true theta, and does a search find it?

The objective fits NOTHING and calibrates NOTHING. For a candidate theta we simulate,
render and score, and compare the candidate's 174-way distribution against the observation's
at the same ramp angle:

    L(mu_c) = sum_over_conditions  d( P_obs(tilt) , P_cand(tilt, mu_c) )

No ridge, no ground-truth slide, no per-object calibration -- so unlike the o4_joint
inversion this is something that could actually run against a measurement whose answer we
do not know.

WHAT THIS TESTS AND WHAT IT DOES NOT.  Observations and candidates both come from our own
render pipeline, so this measures the optimisation -- is the surface unimodal, does the
minimum sit on the truth -- and NOT the domain gap. The 3.4x across-object offset in
o4_joint is the standing evidence that the domain gap is real and unmeasured.

LEAK CONTROL.  Observation and candidate at the same (mu, tilt) are the SAME CLIP and score
distance exactly zero. The target's mu is therefore removed from the candidate grid
entirely, and the minimum is refined by a parabola through its two neighbours in log mu --
which is what a real fit on a finite grid does anyway.

METRIC SELECTION.  Four distances are reported, all of them, always. Choosing the metric
after seeing which one wins is the post-hoc selection that has produced four results in this
project that then failed confirmation, so the metric is fixed before the comparison is read:
SYMKL is the pre-registered primary because it is the only one of the four that is a proper
divergence between distributions and needs no scale choice.

    W scripts/o5_fit.py surface
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import o4_joint as O                                            # noqa: E402

OUT = Path("docs/results/o5_fit.json")
PRIMARY = "symkl"
EPS = 1e-8


def symkl(p, q):
    p = p + EPS; q = q + EPS
    p = p / p.sum(); q = q / q.sum()
    return float((p * np.log(p / q)).sum() + (q * np.log(q / p)).sum())


def l2log(p, q):
    return float(np.linalg.norm(np.log(p + EPS) - np.log(q + EPS)))


def coslog(p, q):
    a, b = np.log(p + EPS), np.log(q + EPS)
    a = a - a.mean(); b = b - b.mean()
    return float(1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def tv(p, q):
    return float(0.5 * np.abs(p - q).sum())


METRICS = {"symkl": symkl, "l2log": l2log, "coslog": coslog, "tv": tv}


def refine(mus, loss):
    """Parabola in log mu through the grid minimum and its two neighbours."""
    i = int(np.argmin(loss))
    x = np.log(mus)
    if i == 0 or i == len(mus) - 1:
        return float(mus[i])
    a, b, c = loss[i - 1], loss[i], loss[i + 1]
    d = a - 2 * b + c
    if d <= 0:
        return float(mus[i])
    off = 0.5 * (a - c) / d
    off = float(np.clip(off, -1.0, 1.0))
    step = x[i + 1] - x[i] if off > 0 else x[i] - x[i - 1]
    return float(np.exp(x[i] + off * abs(step)))


def do_surface(args):
    rows = O.load()
    res = {}
    for obj in ("bowl", "pot"):
        R = [r for r in rows if r["obj"] == obj]
        if len(R) < 10:
            continue
        # index: (tilt, mu) -> distribution
        P = {(r["tilt"], round(r["mu"], 4)): r["p"] for r in R}
        tilts = sorted(set(t for t, _ in P))
        mus = sorted(set(m for _, m in P))
        print(f"\n{'='*70}\n{obj}: {len(R)} clips, {len(mus)} frictions x {len(tilts)} ramp angles")

        out = {m: {"joint": {}, "single": {}} for m in METRICS}
        for name, dist in METRICS.items():
            ej, e1 = [], []
            tab = []
            for mt in mus:
                cond = [t for t in tilts if (t, mt) in P]
                if len(cond) < 2:
                    continue
                cand = [mc for mc in mus if mc != mt]
                # joint: sum the distance over every condition the target was seen at
                L, Lc = [], defaultdict(list)
                for mc in cand:
                    sh = [t for t in cond if (t, mc) in P]
                    if not sh:
                        L.append(np.nan); continue
                    d = [dist(P[(t, mt)], P[(t, mc)]) for t in sh]
                    L.append(float(np.mean(d)))
                    for t, dv in zip(sh, d):
                        Lc[t].append((mc, dv))
                L = np.array(L)
                ok = ~np.isnan(L)
                if ok.sum() < 4:
                    continue
                hat = refine(np.array(cand)[ok], L[ok])
                ej.append(abs(np.log10(hat / mt)))
                per = {}
                for t, v in Lc.items():
                    if len(v) < 4:
                        continue
                    mm = np.array([x[0] for x in v]); dd = np.array([x[1] for x in v])
                    h1 = refine(mm, dd)
                    per[t] = h1
                    e1.append(abs(np.log10(h1 / mt)))
                tab.append({"mu": mt, "joint": hat, "per_condition": per})
            if ej:
                out[name]["joint"] = {"median_pct": float(100 * (10 ** np.median(ej) - 1)),
                                      "n": len(ej), "rows": tab}
                out[name]["single"] = {"median_pct": float(100 * (10 ** np.median(e1) - 1)),
                                       "n": len(e1)}
        print(f"\n  {'metric':<8} {'one condition':>15} {'conditions joined':>19}")
        for name in METRICS:
            if not out[name]["joint"]:
                continue
            s, j = out[name]["single"], out[name]["joint"]
            star = "  <- pre-registered primary" if name == PRIMARY else ""
            print(f"  {name:<8} {s['median_pct']:>13.0f}% {j['median_pct']:>17.0f}%"
                  f"   (n={j['n']}){star}")
        print(f"  {'prior':<8} {86:>13}% {86:>17}%")

        if out[PRIMARY]["joint"]:
            print(f"\n  {PRIMARY} per-target recovery:")
            print(f"  {'true mu':>8} {'fitted':>8} {'err':>7}   per-condition")
            for r in out[PRIMARY]["joint"]["rows"]:
                pc = " ".join(f"{t:.0f}d:{v:.3f}" for t, v in sorted(r["per_condition"].items()))
                print(f"  {r['mu']:>8.3f} {r['joint']:>8.3f} "
                      f"{100*(r['joint']/r['mu']-1):>+6.0f}%   {pc}")
        res[obj] = out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("surface"); s.set_defaults(f=do_surface)
    a = ap.parse_args()
    sys.exit(a.f(a))
