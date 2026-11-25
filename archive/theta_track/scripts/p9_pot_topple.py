"""P9 -- redesign the POT topple probe for its true (shell) centre of mass.

The o14 pot probe broke when the centre of mass stopped being the vertex mean: with its
real shell build the pot's base leverage is b/h ~ 1.09, so it can only tip once
tan(tilt) > 1.09 (~47 deg) -- but at the probe's friction (body mu 1.30, mu_eff = sqrt(1.30
x 0.45) = 0.76) it starts SLIDING at ~37 deg, ten degrees earlier. Sliding wins; 23 of 36
conditions never toppled. The old vertex-mean centre of mass sat too high (b/h 0.74), which
let it tip inside the sliding window -- the probe worked for the wrong reason.

Redesign, both halves needed: toppling must beat sliding, i.e. mu_eff > b/h -> body mu >
1.09^2 / 0.45 ~ 2.6; use body mu 4.0 (mu_eff ~ 1.34). Then tilts steep enough to tip.

SIM-SIDE GATE FIRST (the step skipped before renders once already): sweep tilt 38..58 deg x
centre-of-mass offset +-18 mm (9 values) on the tilt rig (src/sim/rigs.py, incl. restitution
alpha), time-to-topple over 1.33 s as o14 reads it. Choose the tilts where time-to-topple is
monotone in the offset and NOT saturated (never toppled / instant). Then RULE 1 on the
chosen design: the observable must move with the centre of mass and NOT with mass,
restitution (alpha) or friction above the grip threshold.

    W scripts/p9_pot_topple.py sweep
    W scripts/p9_pot_topple.py immunity --tilts a,b,c,d
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
ASSET, SCALE = "rigid/brass_pot_01", 0.629          # as in the o14 pot lab
MU_BODY = 4.0
COMS = np.linspace(-0.018, 0.018, 9)
TILTS = np.arange(34, 51, 2)
OUT = REPO / "docs/results/p9_pot_topple.json"


def topple_time(tilt, com, mu=MU_BODY, mass=0.40, alpha=None):
    import o14_com as C
    from src.sim.mesh_scene import MeshProbeScene
    from src.sim.rigs import TILT_RIG
    ko = dict(masses=[mass], ground_z=C.GZ, dt=1.0 / (C.FPS * C.SUBSTEPS),
              n_steps=C.NF * C.SUBSTEPS, k=2500.0, cd=[C.CD], mu=[mu], mesh_scale=[SCALE],
              faces=900, ground_mu=0.45, table=(0.79, 0.49), tilt_deg=float(tilt))
    if abs(com) > 1e-9:
        ko["com_offset"] = [[float(com), 0.0, 0.0]]
    s = MeshProbeScene([ASSET], [[0., 0., 0.]], [[0., 0., 0.]], **ko)
    s.set_restitution_alpha(TILT_RIG["alpha"] if alpha is None else alpha)
    s.calibrate_stiffness(pen_frac=TILT_RIG["pen_frac"])
    s.seat_on_plane(along=0.0)
    s.rollout()
    # read every 20 substeps (2 ms), not every frame: with the video readout retired the
    # probe is read off the simulator, and the 1/24 s frame grid flattened every condition
    # above 44 deg to 0.17-0.21 s
    st = 20
    P, Q = s.positions(st), s.rotations(st)
    ang = np.array([np.degrees(np.arccos(np.clip(C.q2m(q)[:, 2] @ s._gn, -1, 1))) for q in Q[:, 0]])
    over = np.where(ang > C.TOPPLE_DEG)[0]
    dt_read = st / (C.FPS * C.SUBSTEPS)
    t = float(over[0] * dt_read) if len(over) else float(C.NF / C.FPS)
    cut = int(over[0]) if len(over) else len(P)
    slid = float(abs(((P[:cut, 0] - P[0, 0]) @ s._gt)[-1])) * 100 if cut > 1 else 0.0
    return t, bool(not len(over)), slid


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)) - (len(a) - 1) / 2, np.argsort(np.argsort(b)) - (len(b) - 1) / 2
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def do_sweep(args):
    import warp as wp
    wp.init()
    res = {}
    print(f"time-to-topple (s; '-' = never within 1.33 s), body mu {MU_BODY}, CoM offset mm:")
    print(f"{'tilt':>5} " + "".join(f"{1000*c:>7.1f}" for c in COMS) + "     rho   range  never  slid before topple (cm, max)")
    with wp.ScopedDevice("cuda:0"):
        for tl in TILTS:
            row = [topple_time(tl, c) for c in COMS]
            ts = [r[0] for r in row]; nev = sum(r[1] for r in row)
            fin = [t for t, n, _ in row if not n]
            rng = (max(fin) / min(fin)) if len(fin) >= 2 else float("nan")
            res[int(tl)] = {"t": ts, "never": [r[1] for r in row], "slid_cm": [r[2] for r in row],
                            "rho": sp(COMS, ts), "range": rng, "n_never": int(nev)}
            print(f"{tl:>5} " + "".join(f"{'-':>7}" if n else f"{t:>7.3f}" for t, n, _ in row)
                  + f"   {sp(COMS, ts):+.2f}  {rng:5.2f}x  {nev:>3}   {max(r[2] for r in row):.1f}", flush=True)
    OUT.write_text(json.dumps({"sweep": res}, indent=1))
    return 0


def do_immunity(args):
    """Rule 1 on the chosen design: CoM must move the observable; mass, alpha, mu must not."""
    import warp as wp
    wp.init()
    tilts = [float(x) for x in args.tilts.split(",")]
    D = json.loads(OUT.read_text())
    out = {}
    with wp.ScopedDevice("cuda:0"):
        for name, kw_list in (("com", [dict(com=c) for c in COMS]),
                              ("mass", [dict(com=0.0, mass=m) for m in (0.1, 0.2, 0.4, 0.8, 1.6)]),
                              ("alpha", [dict(com=0.0, alpha=a) for a in (0.05, 0.2, 0.4, 0.8, 1.2)]),
                              ("mu", [dict(com=0.0, mu=m) for m in (3.0, 4.0, 6.0, 9.0, 15.0)])):
            rng = []
            for tl in tilts:
                ts = [topple_time(tl, **kw)[0] for kw in kw_list]
                rng.append(max(ts) / max(min(ts), 1e-6))
            out[name] = {"range_per_tilt": rng, "median_range": float(np.median(rng))}
            print(f"  {name:<6} time-to-topple range over its sweep, per tilt: "
                  + " ".join(f"{r:.2f}x" for r in rng) + f"   median {np.median(rng):.2f}x", flush=True)
    D["design"] = {"tilts": tilts, "mu_body": MU_BODY, "immunity": out}
    OUT.write_text(json.dumps(D, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sweep").set_defaults(f=do_sweep)
    s = sub.add_parser("immunity"); s.set_defaults(f=do_immunity); s.add_argument("--tilts", required=True)
    a = ap.parse_args()
    sys.exit(a.f(a))
