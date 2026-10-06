"""P7 -- the simulator's half of restitution: contact damping cd -> coefficient of restitution.

Real footage can give restitution e (bounce velocity ratio); the simulator needs a contact
damping cd. In a penalty contact the mapping is NOT a material constant -- it depends on the
stiffness setting and the time step (project notes: the bounce probe spans 5.47x over cd at
pen_frac 5e-3 and only 1.03x at 1e-4). So e(cd) is measured here per solver setting:

  a solid 7 cm cube (and a 0.4 kg version) dropped flat from 0.20 m onto a level plane;
  e = normal velocity just after the first impact / just before it, read every substep;
  cd swept 50 .. 30000; two settings:
      bounce rig  pen_frac 5e-3, 24 fps x 60 substeps   (probe_bounce_settle)
      tilt rig    pen_frac 1e-4, 24 fps x 420 substeps  (o13 / o14 / p4)

Output: one curve per setting and mass, so a measured e can be turned into cd for the
probe that will use it -- and, where the curve is flat, the statement that this setting
cannot represent restitution at all.

    W scripts/p7_cd_restitution.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
CDS = np.geomspace(50, 30000, 14)
SETTINGS = {"bounce rig (pen 5e-3, 60 sub)": (5e-3, 60), "tilt rig (pen 1e-4, 420 sub)": (1e-4, 420)}
OUT = REPO / "docs/results/p7_cd_restitution.json"


def drop(cd, pen, sub, mass, h=0.20, fps=24.0, T=0.6):
    import trimesh
    from src.sim.mesh_scene import MeshProbeScene
    cube = trimesh.creation.box(extents=(0.07, 0.07, 0.07))
    for _ in range(3):
        cube = cube.subdivide()
    nf = int(T * fps)
    s = MeshProbeScene([cube], [[0., 0., 0.]], [[0., 0., 0.]], masses=[mass], ground_z=0.706,
                       dt=1.0 / (fps * sub), n_steps=nf * sub, k=2500.0, cd=[float(cd)], mu=[0.5],
                       ground_mu=0.5, ground_cd=float(cd), table=(2., 1.), mass_model="solid")
    s.calibrate_stiffness(pen_frac=pen)
    s.seat_on_plane(along=0.0, clearance=h)
    s.rollout()
    P = s.positions(1)                       # every substep
    z = P[:, 0, 2]; dt = 1.0 / (fps * sub)
    vz = np.gradient(z, dt)
    k = int(np.argmin(vz))                   # fastest descent = just before impact
    after = vz[k:]
    up = float(after.max())
    return max(up, 0.0) / -float(vz[k]), float(-vz[k])


def main():
    import warp as wp
    wp.init()
    res = {}
    with wp.ScopedDevice("cuda:0"):
        for name, (pen, sub) in SETTINGS.items():
            for mass in (0.1, 0.4):
                es = []
                for cd in CDS:
                    e, vin = drop(cd, pen, sub, mass)
                    es.append(e)
                res[f"{name}, {mass} kg"] = {"cd": CDS.tolist(), "e": es}
                print(f"  {name:<30} {mass} kg: " + " ".join(f"{cd:.0f}:{e:.2f}" for cd, e in zip(CDS, es)),
                      flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
