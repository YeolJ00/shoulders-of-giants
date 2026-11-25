"""P4 -- CLOSE THE LOOP: fit friction THROUGH OUR SIMULATOR to real footage.

p1 read cardboard friction off real Physics 101 ramp clips with the analytic law
a = g (sin a - mu cos a). Here the same real trajectories are fitted by Newton itself:
each real cube (edge from its measured volume, mass from the dataset) on a 20 deg plane,
released from rest, mu swept; each real trial is aligned to the simulated curve in release
time and offset; the mu with the smallest error over the ramp segment wins.

Two questions:
  1. does the simulator-fitted mu agree with the analytic mu (0.20-0.28 per cube)? A
     sliding block is the case where they must -- disagreement would be a simulator defect;
  2. does the simulated trajectory reproduce the real one as well as the parabola did
     (0.3-3 mm RMS)? That is the first time our simulator is checked against reality.

Contact: pair friction mu is set directly (body mu = ground mu = mu; the solver combines by
geometric mean). Stiffness from calibrate_stiffness, as for every probe.

    W scripts/p4_sim_loop.py run
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

FPS, SUB, HORIZON = 30.0, 336, 1.2            # 30 fps like the real clips, dt ~ 1e-4 s
MUS = np.round(np.arange(0.10, 0.401, 0.01), 3)
TILT = 20.0
OUT = REPO / "docs/results/p4_sim_loop.json"


def cube_mesh(edge):
    import trimesh
    m = trimesh.creation.box(extents=(edge, edge, edge))
    for _ in range(3):                         # vertices on faces/edges, not only corners
        m = m.subdivide()
    return m


def simulate(edge, mass, mu):
    import warp as wp
    from src.sim.mesh_scene import MeshProbeScene
    nf = int(HORIZON * FPS) + 1
    s = MeshProbeScene([cube_mesh(edge)], [[0., 0., 0.]], [[0., 0., 0.]], masses=[mass],
                       ground_z=0.706, dt=1.0 / (FPS * SUB), n_steps=nf * SUB, k=2500.0,
                       cd=[3200.0], mu=[float(mu)], ground_mu=float(mu), ground_cd=3000.0,
                       table=(2.0, 1.0), tilt_deg=TILT)
    from src.sim.rigs import apply_tilt_rig
    apply_tilt_rig(s)              # stiffness + restitution (alpha), src/sim/rigs.py
    s.seat_on_plane(along=-0.5)
    s.rollout()
    P = s.positions(SUB)
    return (P[:, 0] - P[0, 0]) @ s._gt          # along-ramp displacement per frame, m


def align_err(real_t, real_s, sim_s):
    """min over release time t0 (and offset s0, closed form) of RMS(real - (s0 + sim(t-t0)))."""
    tg = np.arange(len(sim_s)) / FPS
    best = None
    for t0 in np.linspace(real_t[0] - 0.25, real_t[0] + 0.05, 61):
        tau = real_t - t0
        if tau.min() < 0:
            continue
        sim = np.interp(tau, tg, sim_s)
        s0 = float(np.mean(real_s - sim))
        e = float(np.sqrt(np.mean((real_s - s0 - sim) ** 2)))
        if best is None or e < best[0]:
            best = (e, float(t0), s0)
    return best


def run(args=None):
    import warp as wp
    import zipfile
    wp.init()
    z = zipfile.ZipFile(REPO / "data/phys101/phys101_v1.0.zip")
    P = lambda f: {l.split()[0]: float(l.split()[1]) for l in z.read(f).decode().split("\n")
                   if l.strip()}
    mass, vol = P("phys101/objects/mass"), P("phys101/objects/vol")
    R = json.loads((REPO / "docs/results/p1_ramp_friction.json").read_text())
    out = {}
    with wp.ScopedDevice("cuda:0"):
        for cube in sorted({r["clip"].split("/")[0] for r in R}):
            edge = vol[cube] ** (1 / 3) / 100.0
            m = mass[cube] / 1000.0
            trials = [r for r in R if r["clip"].startswith(cube + "/20") and r.get("fit")]
            curves = {float(mu): simulate(edge, m, mu) for mu in MUS}
            per = []
            for r in trials:
                t = np.arange(len(r["s"])) / r["fps"]
                seg = slice(max(r["start"] - 1, 0), r["peak"] + 1)
                rt, rs = t[seg], np.array(r["s"])[seg]
                errs = {mu: align_err(rt, rs, c)[0] for mu, c in curves.items()}
                mus = np.array(sorted(errs)); es = np.array([errs[u] for u in mus])
                i = int(np.argmin(es))
                # parabolic refinement of the minimum on the grid
                if 0 < i < len(mus) - 1:
                    a, b, c = np.polyfit(mus[i - 1:i + 2], es[i - 1:i + 2], 2)
                    mu_hat = float(-b / (2 * a)) if a > 0 else float(mus[i])
                else:
                    mu_hat = float(mus[i])
                per.append({"clip": r["clip"], "mu_sim": mu_hat, "rms_sim_mm": 1000 * es[i],
                            "mu_analytic": r["fit"]["mu"], "rms_parabola_mm": 1000 * r["fit"]["rms_m"],
                            "edge_mm": rs.shape[0]})
            ms = np.array([p["mu_sim"] for p in per]); ma = np.array([p["mu_analytic"] for p in per])
            out[cube] = {"edge_cm": edge * 100, "mass_g": mass[cube], "n": len(per),
                         "mu_sim_median": float(np.median(ms)), "mu_analytic_median": float(np.median(ma)),
                         "median_abs_diff": float(np.median(np.abs(ms - ma))),
                         "rms_sim_mm": float(np.median([p["rms_sim_mm"] for p in per])),
                         "rms_parabola_mm": float(np.median([p["rms_parabola_mm"] for p in per])),
                         "trials": per,
                         "sim_curve_at_fit": curves[float(MUS[np.argmin(np.abs(MUS - np.median(ms)))])].tolist()}
            o = out[cube]
            print(f"  {cube:<14} edge {o['edge_cm']:4.1f} cm  mass {o['mass_g']:6.1f} g   "
                  f"mu sim {o['mu_sim_median']:.3f}  analytic {o['mu_analytic_median']:.3f}  "
                  f"|diff| {o['median_abs_diff']:.3f}   rms sim {o['rms_sim_mm']:.1f} mm  "
                  f"parabola {o['rms_parabola_mm']:.1f} mm", flush=True)
            OUT.write_text(json.dumps(out))
    allm = np.array([p["mu_sim"] for o in out.values() for p in o["trials"]])
    alla = np.array([p["mu_analytic"] for o in out.values() for p in o["trials"]])
    print(f"\nALL {len(allm)} trials: mu sim median {np.median(allm):.3f}, analytic {np.median(alla):.3f}, "
          f"median |diff| {np.median(np.abs(allm-alla)):.3f}, corr {np.corrcoef(allm, alla)[0,1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
