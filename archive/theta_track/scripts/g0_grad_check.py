"""G0 -- do gradients flow through the VALIDATED exact-mesh contact simulator?

The project's gradient demonstration (diff_collide_6dof_recover.py) was on the older
sphere-cover solver. MeshProbeScene -- the simulator that reproduces real footage to
0.4-1.0 mm -- allocates its force / torque / world-vertex buffers ONCE and reuses them every
step, which is correct for a forward rollout and wrong for a Warp tape (the adjoint of a
nonlinear kernel needs that step's forward values, which the next step has overwritten).

This script makes a tape-safe variant with per-step buffers and checks d(slide)/d(mu) on a
cube sliding down a 20 deg plane against central finite differences. The analytic law gives
d(slide)/d(mu) = -1/2 g cos(a) t^2 = -2.30 m per unit mu at t = 0.5 s (plus seating effects).

    W scripts/g0_grad_check.py
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO))

import warp as wp
from src.sim.mesh_scene import MeshProbeScene
from src.sim.mesh_contact import ground_contact_mesh, world_verts
from src.sim.diff_collide_6dof import integrate_6dof

FPS, SUB, T = 24.0, 100, 0.5
TILT = 20.0


@wp.kernel
def slide_loss(pos: wp.array(dtype=wp.vec3), pos0: wp.array(dtype=wp.vec3), gt: wp.vec3,
               loss: wp.array(dtype=float)):
    loss[0] = wp.dot(pos[0] - pos0[0], gt)


class DiffMeshScene(MeshProbeScene):
    """MeshProbeScene with per-step force / torque / world buffers and requires_grad on the
    state, so a wp.Tape over rollout() is valid. No hinges / links (not needed here)."""

    def enable_grad(self):
        rg = True
        n = self.n_steps
        self.mu = wp.array(self.mu.numpy(), dtype=float, requires_grad=rg)
        self.k = wp.array(self.k.numpy(), dtype=float, requires_grad=rg)
        self.cd = wp.array(self.cd.numpy(), dtype=float, requires_grad=rg)
        self.pos = [wp.zeros(self.N, dtype=wp.vec3, requires_grad=rg) for _ in range(n + 1)]
        self.rot = [wp.zeros(self.N, dtype=wp.quat, requires_grad=rg) for _ in range(n + 1)]
        self.vlin = [wp.zeros(self.N, dtype=wp.vec3, requires_grad=rg) for _ in range(n + 1)]
        self.vang = [wp.zeros(self.N, dtype=wp.vec3, requires_grad=rg) for _ in range(n + 1)]
        self.forces = [wp.zeros(self.N, dtype=wp.vec3, requires_grad=rg) for _ in range(n)]
        self.torques = [wp.zeros(self.N, dtype=wp.vec3, requires_grad=rg) for _ in range(n)]
        self.worlds = [wp.zeros(self.nV, dtype=wp.vec3, requires_grad=rg) for _ in range(n)]
        self.loss = wp.zeros(1, dtype=float, requires_grad=rg)
        return self

    def rollout(self):
        self.pos[0].assign(self.pos0); self.rot[0].assign(self.rot0)
        self.vlin[0].assign(self.vlin0); self.vang[0].assign(self.vang0)
        gn = wp.vec3(*[float(v) for v in self._gn]); gt = wp.vec3(*[float(v) for v in self._gt])
        gp = wp.vec3(*[float(v) for v in self._gp])
        for t in range(self.n_steps):
            wp.launch(world_verts, self.nV, inputs=[self.pos[t], self.rot[t], self.local, self.body],
                      outputs=[self.worlds[t]])
            wp.launch(ground_contact_mesh, self.nV,
                      inputs=[self.worlds[t], self.body, self.pos[t], self.vlin[t], self.vang[t],
                              self.ground_z, self.k, self.cd, self.mu, self._gmu, self._gcd,
                              self.weight, self.half_w, self.half_d, gn, gt, gp],
                      outputs=[self.forces[t], self.torques[t]])
            wp.launch(integrate_6dof, self.N,
                      inputs=[self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                              self.forces[t], self.torques[t], self.mass, self.inv_mass,
                              self.G, self.Ginv, self.gravity, self.dt],
                      outputs=[self.pos[t + 1], self.rot[t + 1], self.vlin[t + 1], self.vang[t + 1]])

    def slide(self):
        wp.launch(slide_loss, 1, inputs=[self.pos[self.n_steps], self.pos[0],
                                         wp.vec3(*[float(v) for v in self._gt])],
                  outputs=[self.loss])
        return float(self.loss.numpy()[0])


GROUND_MU = 0.25     # held FIXED: the tape differentiates w.r.t. the BODY's mu only, and the
                     # pair rule sqrt(mu_b * mu_g) halves the sensitivity at mu_b = mu_g -- a
                     # first version varied both in the finite difference and read "half"


def make(mu, grad=False):
    import trimesh
    cube = trimesh.creation.box(extents=(0.07, 0.07, 0.07))
    for _ in range(3):
        cube = cube.subdivide()
    nf = int(T * FPS)
    cls = DiffMeshScene if grad else MeshProbeScene
    s = cls([cube], [[0., 0., 0.]], [[0., 0., 0.]], masses=[0.0318], ground_z=0.706,
            dt=1.0 / (FPS * SUB), n_steps=nf * SUB, k=2500.0, cd=[1.0], mu=[float(mu)],
            ground_mu=GROUND_MU, table=(2., 1.), tilt_deg=TILT, mass_model="solid")
    s.set_restitution_alpha(0.4)
    s.calibrate_stiffness(pen_frac=1e-4)
    s.seat_on_plane(along=-0.3)
    if grad:
        s.enable_grad()
    return s


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        mu0 = 0.25
        # finite differences on the ORIGINAL scene (reference behaviour)
        def slide_ref(mu):
            s = make(mu); s.rollout()
            P = s.positions(1); return float((P[-1, 0] - P[0, 0]) @ s._gt)
        h = 0.01
        fd = (slide_ref(mu0 + h) - slide_ref(mu0 - h)) / (2 * h)
        s_ref = slide_ref(mu0)
        # tape on the differentiable variant
        s = make(mu0, grad=True)
        t0 = time.time()
        tape = wp.Tape()
        with tape:
            s.rollout()
            val = s.slide()
        tape.backward(loss=s.loss)
        g = float(s.mu.grad.numpy()[0])
        dt_ = time.time() - t0
        # analytic: slide = 1/2 g (sin a - mu_eff cos a) t^2, mu_eff = sqrt(mu_b mu_g)
        th = -0.5 * 9.81 * np.cos(np.radians(TILT)) * T ** 2 * 0.5 * np.sqrt(GROUND_MU / mu0)
        print(f"slide at {T} s: tape-variant {100*val:.2f} cm, original {100*s_ref:.2f} cm "
              f"(identical rollout? {abs(val - s_ref) < 1e-6})")
        print(f"d(slide)/d(mu_body): tape {g:+.4f}   central FD (h={h}) {fd:+.4f}   "
              f"analytic (release-aligned, no seating) {th:+.4f}")
        print(f"tape forward+backward {dt_:.1f} s for {s.n_steps} steps, {s.nV} vertices")
        ok = abs(g - fd) < 0.05 * max(abs(fd), 1e-3)
        print("GRADIENT CHECK:", "PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
