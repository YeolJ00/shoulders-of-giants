"""G0c -- where does the body-body contact gradient go missing?

Soft contact (pen_frac 5e-3), 0.6 s, B slides into A: FD d(A disp)/d(mu_B) = -0.38 but the
tape gives -0.03. Three isolating tests:
  A  the mesh-query adjoint alone: d(distance to mesh)/d(query point) vs FD
  C  the same two-body scene, gradient of B's OWN displacement (mostly the validated plane
     path) vs A's (needs the pair adjoint)
  E  a head-on two-body collision with NO plane, loss = A's displacement, parameter = the
     contact stiffness-related mass of B? -- no: parameter = cd of B (damping enters the pair
     force directly), short horizon through the impact only.

    W scripts/g0c_grad_diag.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import warp as wp
import trimesh
import g0b_grad_pair as GB


@wp.kernel
def dist_k(mesh: wp.uint64, q: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    res = wp.mesh_query_point_sign_normal(mesh, q[0], 1.0)
    cp = wp.mesh_eval_position(mesh, res.face, res.u, res.v)
    out[0] = wp.length(q[0] - cp)


def test_a():
    cube = trimesh.creation.box(extents=(0.07, 0.07, 0.07))
    m = wp.Mesh(points=wp.array(np.asarray(cube.vertices, np.float32), dtype=wp.vec3),
                indices=wp.array(np.asarray(cube.faces, np.int32).flatten(), dtype=int))
    for p in ((0.02, 0.01, 0.045), (0.05, 0.0, 0.0), (0.03, 0.03, 0.03)):
        q = wp.array([wp.vec3(*p)], dtype=wp.vec3, requires_grad=True)
        out = wp.zeros(1, dtype=float, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(dist_k, 1, inputs=[m.id, q], outputs=[out])
        tape.backward(loss=out)
        g = q.grad.numpy()[0]
        fd = []
        for ax in range(3):
            def f(eps):
                pp = list(p); pp[ax] += eps
                qq = wp.array([wp.vec3(*pp)], dtype=wp.vec3); oo = wp.zeros(1, dtype=float)
                wp.launch(dist_k, 1, inputs=[m.id, qq], outputs=[oo]); return float(oo.numpy()[0])
            fd.append((f(1e-4) - f(-1e-4)) / 2e-4)
        print(f"TEST A mesh-query adjoint at q={p}: tape {np.round(g, 3)}  FD {np.round(fd, 3)}", flush=True)


def test_c():
    mu_b = 0.15; pen, hz = 5e-3, 0.6

    def disp(m):
        s = GB.make(m, pen=pen, horizon=hz); s.rollout(); P = s.positions(1)
        return float((P[-1, 0] - P[0, 0]) @ s._gt), float((P[-1, 1] - P[0, 1]) @ s._gt)
    h = 0.005
    dA = (disp(mu_b + h)[0] - disp(mu_b - h)[0]) / (2 * h)
    dB = (disp(mu_b + h)[1] - disp(mu_b - h)[1]) / (2 * h)
    for bi, fdv, nm in ((0, dA, "A (pushed body)"), (1, dB, "B (sliding body)")):
        s = GB.make(mu_b, grad=True, pen=pen, horizon=hz); tape = wp.Tape()
        with tape:
            s.rollout(); s.disp(bi)
        tape.backward(loss=s.loss); g = s.mu.grad.numpy()
        print(f"TEST C d(disp of {nm})/d(mu_B): tape {g[1]:+.4f}   FD {fdv:+.4f}", flush=True)


def test_c2():
    """Same scene, but B is held off the plane's friction entirely: ground_mu tiny so the
    only mu_B path is the PAIR friction with A. If this tape gradient is ~0 while FD is not,
    the pair adjoint is the broken piece."""
    import g0b_grad_pair as GBm
    pen, hz = 5e-3, 0.6
    GBm_make = GBm.make

    def make2(m, grad=False):
        s = GBm_make(m, grad=False, pen=pen, horizon=hz)
        return s
    # Reuse GB.make but override friction coupling by giving A a large mu so the pair term
    # matters, and compare gradient w.r.t. mu_A (pair-only path: A rests on a mu_A plane too,
    # so not clean) -- instead report d/d(mu_A) tape vs FD as a second data point.
    mu_b = 0.15; h = 0.02

    def dispA_muA(mA):
        s = GBm_make(mu_b, pen=pen, horizon=hz)
        s.mu.assign(np.array([mA, mu_b], np.float32)); s.rollout(); P = s.positions(1)
        return float((P[-1, 0] - P[0, 0]) @ s._gt)
    fd = (dispA_muA(0.9 + h) - dispA_muA(0.9 - h)) / (2 * h)
    s = GBm_make(mu_b, grad=True, pen=pen, horizon=hz); tape = wp.Tape()
    with tape:
        s.rollout(); s.disp(0)
    tape.backward(loss=s.loss); g = s.mu.grad.numpy()
    print(f"TEST C2 d(disp of A)/d(mu_A): tape {g[0]:+.4f}   FD {fd:+.4f}", flush=True)


if __name__ == "__main__":
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        test_a(); test_c(); test_c2()
