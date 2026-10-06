"""Which probe reads which parameter -- the whole theta vector at once.

Rule 1 says a probe must be shown to depend on its parameter before it is trusted, and the
collide probe is the standing example of what happens otherwise: a body shoved at fixed
velocity slides v^2/(2 mu g), so its stopping distance is provably mass-INDEPENDENT and no
tuning could ever have recovered mass from it.

Rule 1b says the test has to cover the whole set. So this sweeps EVERY parameter through
EVERY probe and reports the response, giving a sensitivity matrix rather than four separate
one-parameter claims. The off-diagonal entries are the point: a probe that responds to two
parameters cannot separate them alone, and two probes whose responses are proportional are
one probe wearing two names.

theta = mass (kg) . com (m, horizontal offset of the centre of mass) . mu . cd

Probes, each reading one number off the SIMULATOR (no renders, no network here):

  tilt_slide   ramp to 24 deg, how far it slides                -> mu
  tip_angle    ramp to 60 deg, the angle at which it topples    -> com
  spin_decay   spun about z, 1/e decay time of |omega|          -> cd, mu
  bounce_h     dropped, peak rebound height                     -> cd
  settle_rate  set down tilted, decay rate of the rocking       -> cd

Reported per cell: Spearman rho over the sweep, and the RANGE the observable covers as a
ratio. rho says the response is monotone; range says whether it is big enough to read
through a readout with finite precision. A cell with rho = 1.0 and a range of 1.02 is
useless, and only reporting rho would hide that.

    W scripts/o7_sensitivity.py run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

OUT = Path("docs/results/o7_sensitivity.json")

# Centre of the sweep. Each parameter is swept about this point with the others held here.
BASE = {"mass": 0.18, "com": 0.0, "mu": 0.35, "cd": 3200.0}
SWEEP = {"mass": [0.08, 0.12, 0.18, 0.27, 0.40],
         "com":  [-0.030, -0.015, 0.0, 0.015, 0.030],
         "mu":   [0.12, 0.20, 0.35, 0.60, 1.00],
         "cd":   [400.0, 1130.0, 3200.0, 9050.0, 25600.0]}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float("nan") if d == 0 else float((ra * rb).sum() / d)


def _kw(th):
    return {"mass": th["mass"], "mu": th["mu"], "cd": th["cd"],
            "com": None if th["com"] == 0 else [float(th["com"]), 0.0, 0.0]}


def tilt_slide(th):
    """Ramp to 24 deg. Slide distance -- the probe already validated in the O-track."""
    import v1_killtest as V
    from src.sim.tilt_probe import ramp_gravity_seq
    g = ramp_gravity_seq(0.0, 24.0, V.NF, V.SUBSTEPS, settle_frames=6)
    r = V.track(V.build(pos=(V.TILT_X0, 0.0, V.GZ + 0.002), gseq=g, **_kw(th)),
                tilt_seq=list(np.linspace(0.0, 24.0, V.NF)))
    if r is None:
        return None
    return 100.0 * (r[0][-1]["flat"][0] - r[0][0]["flat"][0])


def tip_angle(th):
    """Ramp to 60 deg. The ramp angle at which the body's own axis departs from the ramp's.

    A slide leaves the body upright relative to the ramp; a topple does not.

    The simulator TILTS GRAVITY, not the table: `ramp_gravity_seq` keeps the ground plane
    flat and rotates g, and the renderer rotates the table to match. So in simulator
    coordinates the ramp normal is (0, 0, 1), and an object riding flat on the ramp has its
    body +z there. Comparing against a ROTATED normal measured roughly the ramp angle itself
    plus the body's own tilt -- and with the sign wrong it measured their sum, 39.6 degrees
    at a 27 degree ramp against a true 22.2. Every clip of a bowl that was merely sliding
    then flagged as toppling, including one that moved 3 cm.
    """
    import v1_killtest as V
    from src.sim.tilt_probe import ramp_gravity_seq
    N = 60.0
    g = ramp_gravity_seq(0.0, N, V.NF, V.SUBSTEPS, settle_frames=6)
    r = V.track(V.build(pos=(V.TILT_X0, 0.0, V.GZ + 0.002), gseq=g, **_kw(th)),
                tilt_seq=list(np.linspace(0.0, N, V.NF)))
    if r is None:
        return None
    seq = np.linspace(0.0, N, V.NF)
    prev = 0.0                                               # see the note in the docstring
    for i, p in enumerate(r[0]):
        bz = np.asarray(p["mat_flat"], float)[:, 2]   # body +z in world; track() gives "mat"
        d = np.degrees(np.arccos(np.clip(bz[2], -1, 1)))     # vs (0,0,1), the ramp normal
        if d > 25.0:
            # Interpolate the crossing between frames. Returning seq[i] quantises the
            # observable to the frame step (~0.95 deg), which makes it a staircase: the
            # sweep returned 11.43 for three different CoM offsets, and a gradient fit
            # cannot take a step on a staircase.
            if i == 0:
                return float(seq[0])
            f = (25.0 - prev) / max(d - prev, 1e-9)
            return float(seq[i - 1] + f * (seq[i] - seq[i - 1]))
        prev = d
    return float(N)


def _bkw(th):
    return {"mass": th["mass"], "mu": th["mu"],
            "com": None if th["com"] == 0 else [float(th["com"]), 0.0, 0.0]}


def spin_decay(th):
    """Spun about z on a flat table. Exponential decay rate of |omega|, per second.

    A threshold crossing quantises to the frame rate -- it returned exactly 0.375 s for
    every cd, which reads as a dead probe and is really a dead readout. Fitting the
    exponential uses the whole curve.
    """
    import probe_bounce_settle as PB
    s = PB.build(th["cd"], 48, pos=(0.0, 0.0, PB.GZ + 0.002), ang=(0.0, 0.0, 12.0),
                 **_bkw(th))
    s.rollout()
    Q = s.rotations(PB.SUBSTEPS)
    if not np.isfinite(Q).all():
        return None
    w = np.array([2.0 * np.arccos(np.clip(abs(float(Q[i - 1][0] @ Q[i][0])), -1, 1)) * PB.FPS
                  for i in range(1, len(Q))])
    k = w > 1e-3
    if k.sum() < 6:
        return None
    t = np.arange(len(w))[k] / PB.FPS
    return float(-np.polyfit(t, np.log(w[k]), 1)[0])


def bounce_h(th):
    """Dropped from 22 cm in the CENTRE of the table. First rebound, in mm.

    The earlier version took the rebound after the GLOBAL minimum of z, but the bowl often
    rolls off the table later in the window, so the global minimum was the floor and every
    rebound measured 0. The first contact is the one restitution acts at.
    """
    import probe_bounce_settle as PB
    s = PB.build(th["cd"], 48, pos=(0.0, 0.0, PB.GZ + 0.22), **_bkw(th))
    s.rollout()
    P = s.positions(PB.SUBSTEPS)
    if not np.isfinite(P).all() or np.abs(P).max() > 50:
        return None
    z = P[:, 0, 2]
    d = np.diff(z)
    lo = next((i for i in range(1, len(d)) if d[i - 1] < 0 <= d[i]), None)
    if lo is None:
        return None
    up = next((i for i in range(lo + 1, len(d)) if d[i] < 0), len(z) - 1)
    return float(1000.0 * (z[up] - z[lo]))


def _settle_rate(th):
    """Rocked on its dome; decay rate of the rocking angle, per second.

    Upside down on the curved outside, which is the only configuration in which the bowl
    oscillates at all -- on its flat foot it falls back and stops, and the probe then
    measures the window length.
    """
    import probe_bounce_settle as PB
    s = PB.build(th["cd"], 120, pos=(0.0, 0.0, PB.GZ + 0.075),
                 rot=PB.quat_mul(PB.quat_y(180.0), PB.quat_y(22.0)), **_bkw(th))
    s.rollout()
    Q = s.rotations(PB.SUBSTEPS)[:, 0]
    if not np.isfinite(Q).all():
        return None
    a = np.degrees(2.0 * np.arcsin(np.clip(Q[:, 1], -1.0, 1.0)))
    a = a - a.mean()
    env = np.abs(a)
    half = len(env) // 2
    e0, e1 = env[:half].max(), env[half:].max()
    return float(np.log((e0 + 1e-9) / (e1 + 1e-9)) / (half / PB.FPS))


BAL_REFS = [0.06, 0.09, 0.13, 0.19, 0.28, 0.41, 0.60]


def balance_ang(th):
    """Pan balance. The observable is the REFERENCE MASS AT WHICH THE BEAM FLIPS.

    This is a threshold probe and has to be read as one. Read as a magnitude -- the settled
    angle against one fixed reference -- it saturates at its +/-12 deg stops and returns
    -12 for a payload both far lighter AND far heavier than the reference, which looks like
    a probe that cannot tell heavy from light. It can: swept properly the sign flips between
    0.18 and 0.22 kg against a true 0.182 kg.

    The rig is also calibrated for payloads near 0.2 kg; at 1.6 kg the payload leaves the
    pan mid-run (the beam tips the right way first, then collapses to the stop), so the
    bracket is only meaningful inside the references swept here.
    """
    import probe_balance as PBAL
    com = None if th["com"] == 0 else [float(th["com"]), 0.0, 0.0]
    prev_r, prev_a = None, None
    for mr in BAL_REFS:
        s = PBAL.build(mr, m_subject=th["mass"], com=com)
        s.rollout()
        Q = s.rotations(PBAL.SUBSTEPS)[:, 0]
        if not np.isfinite(Q).all():
            return None
        a = PBAL.settled(np.degrees(2.0 * np.arcsin(np.clip(Q[:, 1], -1.0, 1.0))))
        if prev_a is not None and prev_a > 0 >= a:
            f = prev_a / (prev_a - a)             # interpolate the crossing in log mass
            return float(np.exp(np.log(prev_r) + f * (np.log(mr) - np.log(prev_r))))
        prev_r, prev_a = mr, a
    return float(BAL_REFS[-1] if (prev_a or 0) > 0 else BAL_REFS[0])


PROBES = {"balance_ang": balance_ang, "tilt_slide": tilt_slide, "tip_angle": tip_angle,
          "spin_decay": spin_decay, "bounce_h": bounce_h, "settle_rate": _settle_rate}
READS = {"balance_ang": "mass", "tilt_slide": "mu", "tip_angle": "com", "spin_decay": "cd",
         "bounce_h": "cd", "settle_rate": "cd"}


def do_run(args):
    import warp as wp
    wp.init()
    res = {}
    with wp.ScopedDevice("cuda:0"):
        for pname, fn in PROBES.items():
            res[pname] = {}
            print(f"\n{pname}  (meant to read {READS[pname]})")
            for par, vals in SWEEP.items():
                obs = []
                for v in vals:
                    th = dict(BASE); th[par] = v
                    try:
                        obs.append(fn(th))
                    except Exception as e:
                        print(f"    {par}={v}: ERROR {type(e).__name__}: {e}")
                        obs.append(None)
                o = np.array([np.nan if x is None else x for x in obs], float)
                ok = np.isfinite(o)
                rho = spearman(vals, o)
                rng = (float(np.nanmax(np.abs(o)) / max(np.nanmin(np.abs(o)), 1e-9))
                       if ok.sum() >= 2 else float("nan"))
                res[pname][par] = {"vals": list(vals), "obs": o.tolist(),
                                   "rho": rho, "range": rng}
                mark = " <- its own parameter" if par == READS[pname] else ""
                print(f"    {par:>5}: rho {rho:+.3f}  range {rng:7.2f}x  "
                      f"{['%.4g' % x if np.isfinite(x) else 'X' for x in o]}{mark}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run"); s.set_defaults(f=do_run)
    a = ap.parse_args()
    sys.exit(a.f(a))
