"""G0d -- the pair-contact adjoint in isolation: two cubes in free space, no gravity, no plane.

B moves toward A at 1 m/s; loss = A's displacement after the collision. Two parameters:
  cd_B  (enters the pair force directly through sqrt(cd_A cd_B))  -- head-on hit, loss along x
  mu_B  (pair friction only, since there is no plane)             -- glancing hit, loss along y
Tape vs central finite differences. If these pass, the long-horizon sliding train (g0b/g0c)
is the hard case; if they fail, the pair kernel's adjoint itself is broken.

    W scripts/g0d_grad_free.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import warp as wp
import trimesh
from src.sim.mesh_scene import MeshProbeScene
import g0b_grad_pair as GB

FPS, SUB, T = 24.0, 100, 0.2


@wp.kernel
def axis_loss(pos: wp.array(dtype=wp.vec3), pos0: wp.array(dtype=wp.vec3), bi: int, ax: int,
              loss: wp.array(dtype=float)):
    loss[0] = pos[bi][ax] - pos0[bi][ax]


class FreePair(GB.DiffPairScene):
    def axis_disp(self, bi, ax):
        wp.launch(axis_loss, 1, inputs=[self.pos[self.n_steps], self.pos[0], bi, ax], outputs=[self.loss])
        return float(self.loss.numpy()[0])


def make(mu_b, cd_b, offset_y, grad=False, pen=5e-3):
    cubes = []
    for _ in range(2):
        c = trimesh.creation.box(extents=(0.07, 0.07, 0.07))
        for _ in range(3):
            c = c.subdivide()
        cubes.append(c)
    nf = int(T * FPS)
    cls = FreePair if grad else MeshProbeScene
    s = cls(cubes, [[0., 0., 0.], [-0.10, offset_y, 0.]], [[0., 0., 0.], [1.0, 0., 0.]],
            masses=[0.05, 0.05], ground_z=-10.0, dt=1.0 / (FPS * SUB), n_steps=nf * SUB,
            k=2500.0, cd=[1.0, float(cd_b)], mu=[0.5, float(mu_b)], gravity=(0., 0., 0.),
            table=(2., 1.), mass_model="solid")
    s.calibrate_stiffness(pen_frac=pen)
    s.cd.assign(np.array([1.0 * s.k.numpy()[0] * 0.0, float(cd_b)], np.float32)) if False else None
    s.cd.assign(np.array([float(cd_b), float(cd_b)], np.float32))
    if grad:
        s.enable_grad()
    return s


def run(mu_b, cd_b, offset_y, bi, ax):
    s = make(mu_b, cd_b, offset_y); s.rollout(); P = s.positions(1)
    return float(P[-1, bi, ax] - P[0, bi, ax])


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        # head-on: parameter cd (damping) of the pair, loss = A's x displacement
        cd0 = 200.0; h = 10.0
        fd = (run(0.5, cd0 + h, 0.0, 0, 0) - run(0.5, cd0 - h, 0.0, 0, 0)) / (2 * h)
        s = make(0.5, cd0, 0.0, grad=True); tape = wp.Tape()
        with tape:
            s.rollout(); val = s.axis_disp(0, 0)
        tape.backward(loss=s.loss); g = s.cd.grad.numpy()
        print(f"HEAD-ON  A moved {100*val:.2f} cm in {T} s.  d(A x)/d(cd): tape {g[0]+g[1]:+.3e} (sum over both bodies' cd, since both were set to cd0)   FD {fd:+.3e}")
        # glancing: offset in y, parameter mu_B (pair friction only), loss = A's y displacement
        mu0 = 0.5; h = 0.02
        for off in (0.03, 0.05):
            fd = (run(mu0 + h, cd0, off, 0, 1) - run(mu0 - h, cd0, off, 0, 1)) / (2 * h)
            s = make(mu0, cd0, off, grad=True); tape = wp.Tape()
            with tape:
                s.rollout(); val = s.axis_disp(0, 1)
            tape.backward(loss=s.loss); g = s.mu.grad.numpy()
            print(f"GLANCING offset {100*off:.0f} cm: A moved y {100*val:.2f} cm.  d(A y)/d(mu_B): tape {g[1]:+.4f}   FD {fd:+.4f}   (d/d mu_A tape {g[0]:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
