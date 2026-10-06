"""P8 -- restitution set by Hunt-Crossley alpha must not depend on rig, stiffness or mass.

p7 showed raw cd fails both ways: on the tilt rig (pen_frac 1e-4) no cd gives e < 0.77, and
on any rig one cd gives bodies of different mass different e. MeshProbeScene.
set_restitution_alpha(alpha) derives cd = 1.5 alpha k / HC per body. Theory: e ~ 1 - alpha v.
Test: drop a solid 7 cm cube 0.20 m (v_impact ~ 1.98 m/s) on BOTH rigs at 0.1 and 0.4 kg,
sweep alpha; the four curves must coincide. Then a second drop height checks the velocity
dependence the model implies (real materials show it too, but weakly).

    W scripts/p8_alpha.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO))
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]
RIGS = {"bounce rig": (5e-3, 60), "tilt rig": (1e-4, 420)}
OUT = REPO / "docs/results/p8_alpha.json"


def drop(alpha, pen, sub, mass, h, fps=24.0, T=0.6):
    import trimesh
    from src.sim.mesh_scene import MeshProbeScene
    cube = trimesh.creation.box(extents=(0.07, 0.07, 0.07))
    for _ in range(3):
        cube = cube.subdivide()
    nf = int(T * fps)
    s = MeshProbeScene([cube], [[0., 0., 0.]], [[0., 0., 0.]], masses=[mass], ground_z=0.706,
                       dt=1.0 / (fps * sub), n_steps=nf * sub, k=2500.0, cd=[1.0], mu=[0.5],
                       ground_mu=0.5, table=(2., 1.), mass_model="solid")
    s.set_restitution_alpha(alpha)
    s.calibrate_stiffness(pen_frac=pen)          # after: checks the re-application path
    s.seat_on_plane(along=0.0, clearance=h)
    s.rollout()
    P = s.positions(1)
    z = P[:, 0, 2]; vz = np.gradient(z, 1.0 / (fps * sub))
    k = int(np.argmin(vz))
    return max(float(vz[k:].max()), 0.0) / -float(vz[k]), -float(vz[k])


def main():
    import warp as wp
    wp.init()
    res = {}
    with wp.ScopedDevice("cuda:0"):
        for h in (0.20, 0.05):
            for rig, (pen, sub) in RIGS.items():
                for m in (0.1, 0.4):
                    es, v = [], None
                    for a in ALPHAS:
                        e, v = drop(a, pen, sub, m, h)
                        es.append(e)
                    res[f"h={h} {rig} {m}kg"] = {"alpha": ALPHAS, "e": es, "v_impact": v}
                    print(f"  h={h:.2f} m (v {v:.2f})  {rig:<10} {m} kg: "
                          + " ".join(f"{a}:{e:.2f}" for a, e in zip(ALPHAS, es)), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
