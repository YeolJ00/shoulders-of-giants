"""O3 -- blind fit. Recover a friction that appears nowhere in the calibration.

o4_joint holds out a mu that is still ON the calibration grid, so the forward model has
samples either side of the answer and the readout has seen clips at neighbouring frictions
from the same sweep. This is the harder and more honest version: mu = 0.175, 0.245, 0.390
sit between grid points, were simulated and rendered separately, and nothing in the
calibration has seen them.

The fit uses only things that exist for an asset whose parameters are unknown:

  readout   ridge over the 174 SSv2 log-probabilities, calibrated on THIS object's own
            sweep against the SIMULATOR's slide -- no external ground truth is needed,
            because we know exactly how far our own rollout moved.
  forward   slide(mu | tilt), also from the simulator.
  data      the blind clips at several ramp angles, which are conditions we set.

  mu_hat = argmin over the conditions of  || readout(clip_t) - log slide_sim(mu, t) ||

Reported with an uncertainty propagated from the readout's own residual scatter through
the local leverage |d log slide / d log mu|, which is the honest error bar: a friction the
experiment cannot move is reported as uncertain rather than as a number.

    W scripts/o6_blind.py fit
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

OUT = Path("docs/results/o6_blind.json")
BLIND = {"bowl": "outputs/scene/blind_bowl", "pot": "outputs/scene/blind_pot"}


def load_blind(obj):
    import glob
    probs = {}
    for f in glob.glob("outputs/outcome/cache/*.json"):
        d = json.load(open(f))
        probs[str(Path(d["clip"]).resolve())] = np.asarray(d["probs"], np.float64)
    lab = Path(BLIND[obj])
    con = O.containment(lab)
    out = []
    for r in json.loads((lab / "spec.json").read_text()):
        c = (lab / "clips" / f"{r['key']}.mp4").resolve()
        p = probs.get(str(c))
        if p is None or con.get(r["key"], 0) < O.MIN_CONTAIN:
            continue
        out.append({"obj": obj, "mu": float(r["value"]), "tilt": float(r["tilt"]),
                    "slide": float(r["slide_cm"]), "p": p, "key": r["key"]})
    return out


def full_forward(cal):
    """slide(mu | object, tilt) over the WHOLE sweep, not truncated to a monotone prefix.

    o4_joint.forward keeps only the longest strictly-decreasing prefix because it INVERTS
    the curve one condition at a time, and inversion needs monotonicity. Fitting mu against
    every condition at once only ever evaluates the curve forwards, so the truncation --
    which threw away the pot above mu ~ 0.35, exactly where it is hardest to identify --
    is not needed here.
    """
    f = defaultdict(list)
    for r in cal:
        f[(r["obj"], r["tilt"])].append((r["mu"], r["slide"]))
    out = {}
    for k, v in f.items():
        a = np.array(sorted(v))
        out[k] = (np.log10(a[:, 0]), np.log10(np.maximum(a[:, 1], 1e-2)))
    return out


def predict(fwd, obj, tilt, logmu):
    m, s = fwd[(obj, tilt)]
    return np.interp(logmu, m, s)


GRID = np.log10(np.geomspace(0.05, 1.50, 400))


def fit_mu(fwd, obj, obs):
    """argmin over mu of the squared gap between what we READ and what the SIM predicts.

    obs = [(tilt, log10 slide read from the clip), ...]. A condition where slide barely
    changes with mu contributes a flat term and therefore little information, so leverage
    weighting happens implicitly and no leverage has to be estimated at a noisy point.
    """
    L = np.zeros_like(GRID)
    for t, y in obs:
        if (obj, t) in fwd:
            L += (predict(fwd, obj, t, GRID) - y) ** 2
    i = int(np.argmin(L))
    # local curvature of the fit gives the leverage that sets the error bar
    g = np.gradient(np.stack([predict(fwd, obj, t, GRID) for t, _ in obs
                              if (obj, t) in fwd]), GRID, axis=1)[:, i]
    return float(10 ** GRID[i]), float(np.sqrt((g ** 2).sum()))


def do_fit(args):
    cal = O.load()
    fwd = full_forward(cal)
    res = {}
    for obj in ("bowl", "pot"):
        C = [r for r in cal if r["obj"] == obj]
        B = load_blind(obj)
        if not B:
            print(f"{obj}: no blind clips scored yet"); continue
        Xc = np.log(np.array([r["p"] for r in C]) + 1e-8)
        yc = np.log10(np.maximum([r["slide"] for r in C], 1e-2))
        rd = O.ridge(Xc, yc)
        # The error bar must use the LEAVE-ONE-OUT residual, not the in-sample one. A ridge
        # with 174 features on 33 points fits its own training data far too well: the pot's
        # in-sample sd is 0.033 against a LOO 0.057, and error bars built on the former came
        # out too tight -- the bowl's truth fell inside only 1 of 3.
        loo = np.empty(len(yc))
        for j in range(len(yc)):
            k = np.ones(len(yc), bool); k[j] = False
            loo[j] = O.ridge(Xc[k], yc[k])(Xc[j][None])[0]
        sd = float((yc - loo).std())

        print(f"\n{'='*72}\n{obj}: calibrated on {len(C)} clips "
              f"({len(set(r['mu'] for r in C))} frictions), readout sd {sd:.3f}")
        by = defaultdict(list)
        for r in B:
            by[round(r["mu"], 4)].append(r)
        print(f"  {'true mu':>8} {'fitted':>8} {'err':>7} {'+/-':>7}   per-condition")
        rows = []
        for mu0 in sorted(by):
            obs, hats = [], []
            for r in sorted(by[mu0], key=lambda z: z["tilt"]):
                if (obj, r["tilt"]) not in fwd:
                    continue
                y = float(rd(np.log(r["p"] + 1e-8)[None])[0])
                obs.append((r["tilt"], y))
                h, _ = fit_mu(fwd, obj, [(r["tilt"], y)])   # same fit, one condition
                hats.append((r["tilt"], h))
            if not obs:
                continue
            joint, lv = fit_mu(fwd, obj, obs)
            unc = sd / lv if lv > 0 else np.nan
            pc = " ".join(f"{t:.0f}d:{h:.3f}" for t, h in hats)
            print(f"  {mu0:>8.3f} {joint:>8.3f} {100*(joint/mu0-1):>+6.0f}% "
                  f"{100*(10**unc-1):>6.0f}%   {pc}")
            rows.append({"mu_true": mu0, "mu_hat": joint, "n_cond": len(hats),
                         "err_pct": 100*(joint/mu0-1), "unc_pct": float(100*(10**unc-1)),
                         "per_condition": {str(t): h for t, h in hats}})
        e = [abs(np.log10(r["mu_hat"]/r["mu_true"])) for r in rows]
        e1 = [abs(np.log10(h/r["mu_true"])) for r in rows for h in r["per_condition"].values()]
        print(f"    one condition alone : median {100*(10**np.median(e1)-1):.0f}%  (n={len(e1)})")
        print(f"    conditions joined   : median {100*(10**np.median(e)-1):.0f}%  (n={len(e)})")
        print(f"    flat prior          : ~86%")
        inside = sum(abs(np.log10(r["mu_hat"]/r["mu_true"])) <= np.log10(1+r["unc_pct"]/100)
                     for r in rows)
        print(f"    truth inside the reported error bar: {inside}/{len(rows)}")
        res[obj] = {"readout_sd": sd, "rows": rows,
                    "single_median_pct": float(100*(10**np.median(e1)-1)),
                    "joint_median_pct": float(100*(10**np.median(e)-1)),
                    "inside_bar": int(inside), "n": len(rows)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit"); s.set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
