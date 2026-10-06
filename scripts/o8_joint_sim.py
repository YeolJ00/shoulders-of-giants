"""Joint fit of the whole theta vector on SIMULATOR observables.

theta = mass . com . mu . cd, recovered together from four probes at once. Observations
come from the simulator itself, so this tests IDENTIFIABILITY and the fit machinery -- can
these four probes separate these four parameters at all -- and deliberately not model error.
If the fit cannot recover theta when the forward model is exactly right, no readout will
rescue it, and there is no point rendering anything.

Probes, and why these four (o7_sensitivity.py, docs/results/o7_sensitivity.json):

  balance   flip reference mass   mass 4.94x, mu 1.00x, cd 1.00x -- the mass gauge, exactly
                                  immune to mu and cd, confounded only by com (3.35x)
  tip       topple angle          com 6.67x, mass and cd EXACTLY 1.00x -- the clean one
  tilt      slide distance        mu 13.27x, com 2.07x
  bounce    first rebound         cd 5.73x, mass 4.05x, mu 1.08x

Spin is excluded: mass moves it more than cd does (3.89x vs 3.19x), and under the tilt rig's
stiffer contact it reads friction instead. Settle is excluded: nothing moves it past 1.29x.

The matrix is nearly triangular -- mass and com from balance+tip, then mu, then cd -- but it
is fit JOINTLY rather than in that order, because balance's com confound and tilt's com
confound both couple back, and a sequential fit would have to iterate anyway.

Each probe runs at ITS OWN validated solver set. Which parameter a probe reads depends on
the contact calibration: bounce's rebound spans 5.47x over cd at pen_frac 5e-3 and 1.03x at
1e-4, where the contact is 50x stiffer and cd stops mattering.

    CUDA_VISIBLE_DEVICES=0 W scripts/o8_joint_sim.py fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

OUT = Path(os.environ.get("OUT", "docs/results/o8_joint_sim.json"))

# name, lower, upper, log-scaled?  com is signed, so it is fit linearly.
PARS = [("mass", 0.06, 0.55, True), ("com", -0.030, 0.030, False),
        ("mu", 0.10, 1.20, True), ("cd", 400.0, 25600.0, True)]
NAMES = [p[0] for p in PARS]


def to_x(th):
    return np.array([np.log(th[n]) if lg else th[n] for n, _, _, lg in PARS])


def to_th(x):
    return {n: (float(np.exp(np.clip(v, np.log(lo), np.log(hi)))) if lg
                else float(np.clip(v, lo, hi)))
            for v, (n, lo, hi, lg) in zip(x, PARS)}


# ------------------------------------------------------------------ observables
BAL_LO, BAL_HI, BAL_ITERS = 0.04, 0.80, 6


def obs_balance(th):
    """Reference mass at which the beam flips, by BISECTION -- 6 sims, not a 7-point sweep."""
    import probe_balance as PBAL
    com = None if th["com"] == 0 else [float(th["com"]), 0.0, 0.0]

    def ang(mr):
        s = PBAL.build(mr, m_subject=th["mass"], com=com)
        s.rollout()
        Q = s.rotations(PBAL.SUBSTEPS)[:, 0]
        if not np.isfinite(Q).all():
            return None
        return PBAL.settled(np.degrees(2.0 * np.arcsin(np.clip(Q[:, 1], -1.0, 1.0))))

    lo, hi = BAL_LO, BAL_HI
    a_lo = ang(lo)
    if a_lo is None or a_lo <= 0:            # subject lighter than the lightest reference
        return lo
    for _ in range(BAL_ITERS):
        mid = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
        a = ang(mid)
        if a is None:
            return None
        if a > 0:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (np.log(lo) + np.log(hi))))


def obs_tip(th):
    import o7_sensitivity as S
    return S.tip_angle(th)


def _tilt_at(th, deg):
    import v1_killtest as V
    from src.sim.tilt_probe import ramp_gravity_seq
    g = ramp_gravity_seq(0.0, deg, V.NF, V.SUBSTEPS, settle_frames=6)
    r = V.track(V.build(pos=(V.TILT_X0, 0.0, V.GZ + 0.002), gseq=g,
                        mass=th["mass"], mu=th["mu"], cd=th["cd"],
                        com=None if th["com"] == 0 else [float(th["com"]), 0.0, 0.0]),
                tilt_seq=list(np.linspace(0.0, deg, V.NF)))
    return None if r is None else 100.0 * (r[0][-1]["loc"][0] - r[0][0]["loc"][0])


TILTS = (24.0, 30.0, 36.0)


def obs_tilt24(th):
    return _tilt_at(th, 24.0)


def obs_tilt30(th):
    return _tilt_at(th, 30.0)


def obs_tilt36(th):
    return _tilt_at(th, 36.0)


def obs_bounce(th):
    import o7_sensitivity as S
    return S.bounce_h(th)


# observable -> (fn, is it compared in log space?)
# THREE ramp angles, not one. A single 24 deg ramp cannot identify mu across the range:
# above mu* = tan^2(24)/0.45 = 0.441 the bowl barely moves, so mu = 0.55 and mu = 0.35 give
# almost the same slide (6.0 vs 6.3 cm) and the fit could not tell one target from its own
# starting point -- its residual began at 0.39. Steeper ramps put mu* at 0.74 and 1.17 and
# restore the leverage. This is the O-track result applied to the joint fit: identification
# comes from the SET of known conditions, never from one.
OBS = {"balance": (obs_balance, True), "tip": (obs_tip, False),
       "tilt24": (obs_tilt24, True), "tilt30": (obs_tilt30, True),
       "tilt36": (obs_tilt36, True), "bounce": (obs_bounce, True)}


def observe(th):
    out = {}
    for k, (fn, _) in OBS.items():
        try:
            out[k] = fn(th)
        except Exception:
            out[k] = None
    return out


def resid(o, target):
    """Residual vector. Log for the three magnitudes, degrees/10 for the topple angle.

    A missing or diverged observable contributes 0 rather than a NaN, so one failed probe
    degrades the fit instead of destroying it.
    """
    r = []
    for k, (_, lg) in OBS.items():
        a, b = o.get(k), target.get(k)
        if a is None or b is None or (lg and (a <= 0 or b <= 0)):
            r.append(0.0)
        elif lg:
            r.append(float(np.log(a) - np.log(b)))
        else:
            r.append(float(a - b) / 10.0)
    return np.array(r)


def jacobian(x0, target, h=0.25):
    """Central differences in the fit's own coordinates. 2 evals per parameter."""
    J = np.zeros((len(OBS), len(PARS)))
    for j, (n, lo, hi, lg) in enumerate(PARS):
        step = h if lg else 0.010
        xp, xm = x0.copy(), x0.copy()
        xp[j] += step; xm[j] -= step
        rp = resid(observe(to_th(xp)), target)
        rm = resid(observe(to_th(xm)), target)
        J[:, j] = (rp - rm) / (2 * step)
        print(f"    d/d{n:<5} " + " ".join(f"{v:+7.3f}" for v in J[:, j]), flush=True)
    return J


def fit(target, x0, n_iter=10, lam=1e-2):
    """Levenberg-Marquardt with ADAPTIVE damping, one Jacobian reused.

    Fixed damping plus four backtracks gave up on the first step for one of two targets and
    returned the starting point unchanged -- which looks like an unidentifiable system and
    is really an optimiser that will not shorten its stride. On a rejected step lambda goes
    up tenfold and the step is retried; on an accepted one it comes back down.
    """
    x = to_x(to_th(x0.copy()))
    r = resid(observe(to_th(x)), target)
    print(f"  start  {fmt(to_th(x))}  |r| = {np.linalg.norm(r):.4f}", flush=True)
    print("  jacobian:", flush=True)
    J = jacobian(x, target)
    best = (float(np.linalg.norm(r)), x.copy())
    for it in range(n_iter):
        ok = False
        print(f"    r = " + " ".join(f"{v:+.3f}" for v in r), flush=True)
        for _ in range(8):                       # raise damping until a step helps
            A = J.T @ J + lam * np.eye(len(PARS))
            try:
                dx = -np.linalg.solve(A, J.T @ r)
            except np.linalg.LinAlgError:
                break
            # PROJECT the trial point into the bounds and keep the projected one. to_th
            # clips, so an unprojected iterate can sit far outside the box while to_th keeps
            # returning the same clipped theta -- the residual then stops responding to dx
            # at all and the fit freezes at a corner, which is what happened here.
            # TRUST REGION. The tip column of J is ~40 while the others are 1-15, so an
            # undamped step is almost entirely com and leaves the other three parameters
            # where they started. Capping the step keeps every parameter moving.
            m = float(np.max(np.abs(dx)))
            if m > 0.35:
                dx = dx * (0.35 / m)
            xn = to_x(to_th(x + dx))
            rn = resid(observe(to_th(xn)), target)
            print(f"      lam {lam:>9.3g} dx " + " ".join(f"{v:+.3f}" for v in dx)
                  + f"  -> |r| {np.linalg.norm(rn):.4f}", flush=True)
            if np.linalg.norm(rn) < np.linalg.norm(r):
                lam = max(lam * 0.3, 1e-6); ok = True
                break
            lam *= 10.0
        if not ok:
            print(f"  iter {it}: no improving step (lambda {lam:.3g})", flush=True)
            break
        step, x, r = 1.0, xn, rn
        if np.linalg.norm(r) < best[0]:
            best = (float(np.linalg.norm(r)), x.copy())
        print(f"  iter {it}  {fmt(to_th(x))}  |r| = {np.linalg.norm(r):.4f}", flush=True)
        if np.linalg.norm(step * dx) < 1e-3:
            break
    return to_th(best[1]), best[0]


def fmt(th):
    return (f"mass {th['mass']:.3f}  com {th['com']:+.4f}  "
            f"mu {th['mu']:.3f}  cd {th['cd']:7.0f}")


TARGETS = [{"mass": 0.24, "com": 0.012, "mu": 0.22, "cd": 6000.0},
           {"mass": 0.13, "com": -0.008, "mu": 0.55, "cd": 1500.0}]
START = {"mass": 0.18, "com": 0.0, "mu": 0.35, "cd": 3200.0}


def do_fit(args):
    import warp as wp
    wp.init()
    res = []
    with wp.ScopedDevice("cuda:0"):
        only = int(os.environ.get("TARGET", "-1"))
        for i, tt in enumerate(TARGETS):
            if only >= 0 and i != only:
                continue
            print(f"\n{'='*74}\nTARGET {i}: {fmt(tt)}", flush=True)
            tgt = observe(tt)
            print(f"  observed: " + "  ".join(
                f"{k}={'X' if v is None else format(v,'.4g')}" for k, v in tgt.items()),
                flush=True)
            th, rn = fit(tgt, to_x(START))
            err = {n: (100 * (th[n] / tt[n] - 1) if tt[n] else th[n] - tt[n])
                   for n in NAMES}
            print(f"\n  TRUE      {fmt(tt)}")
            print(f"  FITTED    {fmt(th)}")
            print(f"  error     mass {err['mass']:+.0f}%  com {1000*(th['com']-tt['com']):+.1f} mm  "
                  f"mu {err['mu']:+.0f}%  cd {err['cd']:+.0f}%")
            res.append({"true": tt, "fit": th, "resid": rn, "err_pct": err,
                        "observed": {k: (None if v is None else float(v))
                                     for k, v in tgt.items()}})
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
