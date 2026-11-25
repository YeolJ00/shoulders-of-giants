"""G0b -- gradients through BODY-BODY mesh contact (wp.mesh_query_point_sign_normal).

g0 showed the tape reproduces MeshProbeScene exactly and d(slide)/d(mu) matches finite
differences through plane contact. Multi-object scenes need gradients through the pair
kernel, which queries one body's wp.Mesh in the other's local frame. Scene: two cubes on a
20 deg plane, B uphill of A; B slides into A and pushes it. Loss = A's final along-slope
displacement; parameter = B's friction. Tape vs central finite differences.

    W scripts/g0b_grad_pair.py
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import warp as wp
from src.sim.mesh_scene import MeshProbeScene
from src.sim.mesh_contact import ground_contact_mesh, pair_contact_mesh, world_verts
from src.sim.diff_collide_6dof import integrate_6dof
import g0_grad_check as G

FPS, SUB, T, TILT = 24.0, 100, 0.6, 20.0


@wp.kernel
def disp_loss(pos: wp.array(dtype=wp.vec3), pos0: wp.array(dtype=wp.vec3), gt: wp.vec3,
              bi: int, loss: wp.array(dtype=float)):
    loss[0] = wp.dot(pos[bi] - pos0[bi], gt)


class DiffPairScene(G.DiffMeshScene):
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
            wp.launch(pair_contact_mesh, self.nV * self.N,
                      inputs=[self.worlds[t], self.body, self.mesh_ids, self.N,
                              self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                              self.k, self.cd, self.mu, self.weight, self.query_dist],
                      outputs=[self.forces[t], self.torques[t]])
            wp.launch(integrate_6dof, self.N,
                      inputs=[self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                              self.forces[t], self.torques[t], self.mass, self.inv_mass,
                              self.G, self.Ginv, self.gravity, self.dt],
                      outputs=[self.pos[t + 1], self.rot[t + 1], self.vlin[t + 1], self.vang[t + 1]])

    def disp(self, bi):
        wp.launch(disp_loss, 1, inputs=[self.pos[self.n_steps], self.pos[0],
                                        wp.vec3(*[float(v) for v in self._gt]), bi],
                  outputs=[self.loss])
        return float(self.loss.numpy()[0])


def make(mu_b, grad=False, pen=1e-4, horizon=None):
    import trimesh
    cubes = []
    for _ in range(2):
        c = trimesh.creation.box(extents=(0.07, 0.07, 0.07))
        for _ in range(3):
            c = c.subdivide()
        cubes.append(c)
    nf = int((horizon or T) * FPS)
    cls = DiffPairScene if grad else MeshProbeScene
    s = cls(cubes, [[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]],
            masses=[0.0318, 0.0318], ground_z=0.706, dt=1.0 / (FPS * SUB), n_steps=nf * SUB,
            k=2500.0, cd=[1.0, 1.0], mu=[0.9, float(mu_b)], ground_mu=0.3, table=(2., 1.),
            tilt_deg=TILT, mass_model="solid")
    s.set_restitution_alpha(0.4)
    s.calibrate_stiffness(pen_frac=pen)
    s.seat_on_plane(bi=0, along=0.0)        # A: grippy, downhill
    s.seat_on_plane(bi=1, along=-0.20)      # B: slippery, 20 cm uphill, slides into A
    if grad:
        s.enable_grad()
    return s


def diag():
    """Is the mismatch the tape, forward nondeterminism, or stiff impulsive contact?"""
    wp.init()
    mu_b = 0.15
    with wp.ScopedDevice("cuda:0"):
        for pen, hz in ((1e-4, 0.6), (1e-4, 0.45), (5e-3, 0.6), (5e-3, 0.45)):
            def dispA(m):
                s = make(m, pen=pen, horizon=hz); s.rollout()
                P = s.positions(1); return float((P[-1, 0] - P[0, 0]) @ s._gt)
            r1, r2 = dispA(mu_b), dispA(mu_b)
            fds = {h: (dispA(mu_b + h) - dispA(mu_b - h)) / (2 * h) for h in (0.002, 0.005, 0.02)}
            s = make(mu_b, grad=True, pen=pen, horizon=hz)
            tape = wp.Tape()
            with tape:
                s.rollout(); val = s.disp(0)
            tape.backward(loss=s.loss)
            g = s.mu.grad.numpy()
            print(f"pen_frac {pen:g}, horizon {hz} s: A moved {100*r1:.2f} cm | forward repeat diff "
                  f"{1e3*abs(r1-r2):.3f} mm, tape-vs-orig {1e3*abs(val-r1):.3f} mm | FD h=.002/.005/.02: "
                  f"{fds[0.002]:+.3f} / {fds[0.005]:+.3f} / {fds[0.02]:+.3f} | tape d/dmu_B {g[1]:+.3f}  d/dmu_A {g[0]:+.3g}",
                  flush=True)


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        mu_b = 0.15
        def disp_ref(m):
            s = make(m); s.rollout()
            P = s.positions(1); return float((P[-1, 0] - P[0, 0]) @ s._gt), float((P[-1, 1] - P[0, 1]) @ s._gt)
        a0, b0 = disp_ref(mu_b)
        h = 0.005
        fd = (disp_ref(mu_b + h)[0] - disp_ref(mu_b - h)[0]) / (2 * h)
        s = make(mu_b, grad=True)
        t0 = time.time()
        tape = wp.Tape()
        with tape:
            s.rollout()
            val = s.disp(0)
        tape.backward(loss=s.loss)
        g = s.mu.grad.numpy()
        dt_ = time.time() - t0
        print(f"A (grippy) moved {100*a0:.2f} cm, B (slippery) moved {100*b0:.2f} cm in {T} s "
              f"-> B hit A: {'yes' if a0 > 0.005 else 'NO (redesign)'}; tape rollout identical: {abs(val - a0) < 1e-6}")
        print(f"d(A displacement)/d(mu_B): tape {g[1]:+.4f}   central FD (h={h}) {fd:+.4f}")
        print(f"d(A displacement)/d(mu_A): tape {g[0]:+.4f}   (not FD-checked)")
        print(f"tape forward+backward {dt_:.1f} s for {s.n_steps} steps, {s.nV} vertices, 2 bodies")
        ok = abs(g[1] - fd) < 0.1 * max(abs(fd), 1e-3)
        print("PAIR-CONTACT GRADIENT CHECK:", "PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(diag() if "--diag" in sys.argv else main())
