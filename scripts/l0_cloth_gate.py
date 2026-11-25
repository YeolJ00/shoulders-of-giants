"""L0 -- cloth teacher gate (T0-cloth): which Newton cloth solver, if any, is fit to distil from?

A cloth dataset inherits every defect of the solver that generated it, so the cloth teacher is
checked against invariants and closed forms before any cloth episode is generated (CLAUDE.md
rule 2). The cloth is given in physical units -- areal density w (kg/m2), stretch modulus Y (N/m),
Poisson ratio nu, bending stiffness B (N m), friction mu, air drag c (1/s) -- and each solver gets
the same cloth through its own parameters. Checks:

  stretch    strip 5 x 30 cm pinned along its top edge, hanging: extension against the closed
             form w g L^2 / (2 Y), from a soft sheet (Y = 20 N/m) to woven cotton (1e4 N/m); 5%
  swing      sheet 30 x 30 cm pinned along one edge, released horizontal, no drag: kinetic plus
             potential energy never rises above its start (1% of the energy released); with
             drag c = 2/s it is at rest after 5 s (no vertex moves 5 mm in the next second)
  freefall   sheet falling free for 0.5 s: centroid acceleration g within 0.1%, shape kept
  airfall    with air (rho Cd 1.44 kg/m3) the same sheet falls face-down at sqrt(2 w g / rho Cd) (5%)
  slide      sheet 10 x 10 cm on a 20 deg plane: a = g (sin - mu cos) within 2% at mu 0.2;
             drift < 1 mm/s at mu 0.5 (above tan 20)
  bend       Peirce cantilever: a 3 cm strip clamped on a ledge, overhang l = 2 (B / w g)^(1/3):
             tip deflection against the heavy elastica for the same B (10%), unchanged at half the
             mesh size and at 8 B with twice the overhang (B is a physical quantity)
  drape      towel 30 x 30 cm dropped on a table, 30% (9 cm) over the edge: at rest after 4 s (no vertex
             moves 5 mm in the next second), no creep on the table (< 1 mm/s), no penetration (> -1 mm),
             and it hangs (lowest point > 4.5 cm below the top: the elastica puts it ~8 cm down)
  throughput 64 worlds of the drape: simulated seconds per wall second

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l0_cloth_gate.py [check ...] [--solver vbd,style3d,xpbd] [--key value ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l0_cloth_gate.json"

if "CUDA_VISIBLE_DEVICES" not in os.environ:          # pick an idle card before CUDA starts
    from lsim.gpu import wait_for_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(wait_for_gpu(min_free_gib=8.0, max_util=20))

import warp as wp                                     # noqa: E402
import newton                                         # noqa: E402
import newton._src.solvers.style3d.solver_style3d as _s3   # noqa: E402
from newton.solvers import style3d                    # noqa: E402
from newton._src.geometry.kernels import triangle_closest_point_barycentric   # noqa: E402

G = 9.81
FPS = 30
CLOTH = dict(w=0.2, Y=1.0e4, nu=0.3, B=3.0e-5, mu=0.6, h=0.01, r=0.003)     # a cotton-like towel
SOLVERS = {"vbd": dict(substeps=20, iters=10, feps=1.0e-2),
           "style3d": dict(substeps=10, iters=4, lin=60, predictor=1, feps=1.0e-2),   # its cheapest converged setting
           "xpbd": dict(substeps=20, iters=10, feps=1.0e-2),
           # explicit: no iterations to converge, only a step small enough to be stable (one vertex 19 mg
           # on a 1e4 N/m membrane: dt < 4e-5 s); contacts refreshed every 50 substeps (1.7 ms)
           "semi": dict(substeps=2000, iters=1, feps=1.0e-2, collide_every=50)}
SELF = dict(thick=0.002, ke=1.0e3, mu=0.5)            # cloth self-contact: thickness (m), penalty (N/m), friction
# explicit-solver material: Newton's membrane damping (off: it damps rotation too), our strain-rate damping (N s/m),
# bending damping, the friction gain below the Coulomb cap (N s/m; < 2 m / dt to stay stable), air density x drag
# coefficient (kg/m3: 1.44 in air, 0 in the checks)
SEMI = dict(tri_kd=0.0, obj_kd=1.0e-3, edge_kd=0.0, kf=1.0, rho_cd=0.0)
CONTACT = dict(ke=1.0e3, kd=0.0)                      # particle-shape penalty: a 19 mg vertex sinks 0.2 um; damping
                                                      # (on approach only) made explicit contacts chatter and creep


@wp.kernel
def _velocity_undamped(dt: float, prev_pos: wp.array[wp.vec3], pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
    i = wp.tid()
    vel[i] = (pos[i] - prev_pos[i]) / dt


_S3_VELOCITY = _s3.update_velocity                    # Style3D multiplies every velocity by 0.998 per step:
                                                      # a damping set by the step size, not by the cloth
ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))


@wp.kernel
def _init_step_inertial(dt: float, gravity: wp.array[wp.vec3], particle_world: wp.array[wp.int32],
                        f_ext: wp.array[wp.vec3], v_curr: wp.array[wp.vec3], x_curr: wp.array[wp.vec3],
                        x_prev: wp.array[wp.vec3], pd_diags: wp.array[float], particle_masses: wp.array[float],
                        particle_flags: wp.array[wp.int32], x_inertia: wp.array[wp.vec3],
                        static_A_diags: wp.array[float], dx: wp.array[wp.vec3]):
    """Style3D's step start with the inertial prediction (gravity included) as the first guess; its
    own guess, v dt, leaves the uniform weight to a few Jacobi-PCG iterations, which barely move the
    lowest mode: a free sheet fell at 1.3% of g."""
    tid = wp.tid()
    x_last = x_curr[tid]
    x_prev[tid] = x_last
    if (particle_flags[tid] & ACTIVE) == 0:
        x_inertia[tid] = x_last
        static_A_diags[tid] = 0.0
        dx[tid] = wp.vec3(0.0)
    else:
        mass = particle_masses[tid]
        static_A_diags[tid] = pd_diags[tid] + mass / (dt * dt)
        xi = x_last + v_curr[tid] * dt + (gravity[particle_world[tid]] + f_ext[tid] / mass) * (dt * dt)
        x_inertia[tid] = xi
        dx[tid] = xi - x_last


_S3_INIT = _s3.init_step_kernel


@wp.kernel
def _drag(scale: float, vel: wp.array[wp.vec3]):
    i = wp.tid()
    vel[i] = vel[i] * scale


@wp.kernel
def _self_candidates(mesh: wp.uint64, x: wp.array[wp.vec3], tri: wp.array2d[int], query: float,
                     cand: wp.array2d[int], count: wp.array[int]):
    """Once per contact refresh: the triangles (not its own) within the query distance of each vertex."""
    i = wp.tid()
    xi = x[i]
    q = wp.mesh_query_aabb(mesh, xi - wp.vec3(query), xi + wp.vec3(query))
    face = int(0)
    k = int(0)
    while wp.mesh_query_aabb_next(q, face):
        ia = tri[face, 0]
        ib = tri[face, 1]
        ic = tri[face, 2]
        if ia != i and ib != i and ic != i and k < cand.shape[1]:
            w = triangle_closest_point_barycentric(x[ia], x[ib], x[ic], xi)
            if wp.length(xi - (w[0] * x[ia] + w[1] * x[ib] + w[2] * x[ic])) < query:
                cand[i, k] = face
                k += 1
    count[i] = k


@wp.kernel
def _self_contact(x: wp.array[wp.vec3], v: wp.array[wp.vec3], tri: wp.array2d[int], cand: wp.array2d[int],
                  count: wp.array[int], thick: float, ke: float, kf: float, mu: float, f: wp.array[wp.vec3]):
    """Vertex-triangle repulsion inside the cloth (Bridson-style): a vertex closer than the cloth thickness
    to a triangle it is not part of is pushed off along the separating direction, the triangle's corners
    take the opposite force by barycentric weight, and friction is capped at mu times the push."""
    i = wp.tid()
    xi = x[i]
    for k in range(count[i]):
        face = cand[i, k]
        ia = tri[face, 0]
        ib = tri[face, 1]
        ic = tri[face, 2]
        w = triangle_closest_point_barycentric(x[ia], x[ib], x[ic], xi)
        n = xi - (w[0] * x[ia] + w[1] * x[ib] + w[2] * x[ic])
        d = wp.length(n)
        if d < thick and d > 1.0e-9:
            n = n / d
            fn = ke * (thick - d)
            vr = v[i] - (w[0] * v[ia] + w[1] * v[ib] + w[2] * v[ic])
            vt = vr - n * wp.dot(n, vr)
            vs = wp.length(vt)
            F = n * fn
            if vs > 1.0e-12:
                F = F - vt / vs * wp.min(kf * vs, mu * fn)
            wp.atomic_add(f, i, F)
            wp.atomic_add(f, ia, -w[0] * F)
            wp.atomic_add(f, ib, -w[1] * F)
            wp.atomic_add(f, ic, -w[2] * F)


@wp.kernel
def _membrane_damping(x: wp.array[wp.vec3], v: wp.array[wp.vec3], tri: wp.array2d[int], pose: wp.array[wp.mat22],
                      kd: float, f: wp.array[wp.vec3]):
    """Damping of the membrane's strain rate C' = F'^T F + F^T F' (Rayleigh dissipation kd A |C'|^2 / 2): blind
    to rigid motion, so a swinging or falling sheet is not braked. Newton's explicit triangle damps F' itself,
    which includes rotation: a sheet released from horizontal took seconds to swing down."""
    t = wp.tid()
    i0 = tri[t, 0]
    i1 = tri[t, 1]
    i2 = tri[t, 2]
    D = pose[t]
    area = 1.0 / (2.0 * wp.determinant(D))
    g1 = wp.vec2(D[0, 0], D[0, 1])
    g2 = wp.vec2(D[1, 0], D[1, 1])
    g0 = -g1 - g2
    x10 = x[i1] - x[i0]
    x20 = x[i2] - x[i0]
    v10 = v[i1] - v[i0]
    v20 = v[i2] - v[i0]
    Fa = x10 * D[0, 0] + x20 * D[1, 0]                 # columns of F and F'
    Fb = x10 * D[0, 1] + x20 * D[1, 1]
    Va = v10 * D[0, 0] + v20 * D[1, 0]
    Vb = v10 * D[0, 1] + v20 * D[1, 1]
    caa = 2.0 * wp.dot(Va, Fa)
    cab = wp.dot(Va, Fb) + wp.dot(Fa, Vb)
    cbb = 2.0 * wp.dot(Vb, Fb)
    s = -2.0 * kd * area
    # f_i = -2 kd A F C' g_i
    wp.atomic_add(f, i0, s * (Fa * (caa * g0[0] + cab * g0[1]) + Fb * (cab * g0[0] + cbb * g0[1])))
    wp.atomic_add(f, i1, s * (Fa * (caa * g1[0] + cab * g1[1]) + Fb * (cab * g1[0] + cbb * g1[1])))
    wp.atomic_add(f, i2, s * (Fa * (caa * g2[0] + cab * g2[1]) + Fb * (cab * g2[0] + cbb * g2[1])))


@wp.kernel
def _kinematic(ids: wp.array[int], x0: wp.array[wp.vec3], vel: wp.array[wp.vec3], clock: wp.array[float],
               x: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    """Grasped vertices follow the gripper: x = x0 + vel t."""
    k = wp.tid()
    i = ids[k]
    x[i] = x0[k] + vel[k] * clock[0]
    v[i] = vel[k]


@wp.kernel
def _tick(dt: float, clock: wp.array[float]):
    clock[0] = clock[0] + dt


def grid(nx, ny, h):
    """A flat nx x ny cell grid in its own x-y plane, diagonals alternating (no preferred shear)."""
    v = [(i * h, j * h, 0.0) for j in range(ny + 1) for i in range(nx + 1)]
    t = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i, (j + 1) * (nx + 1) + i + 1
            t += [a, b, d, a, d, c] if (i + j) % 2 == 0 else [a, b, c, b, d, c]
    return v, t


def new_builder(solver):
    b = newton.ModelBuilder()
    if solver == "style3d":
        newton.solvers.SolverStyle3D.register_custom_attributes(b)
    return b


def add_sheet(b, solver, cloth, nx, ny, pos, rot=None):
    """A sheet of the given physical cloth; returns its particle ids."""
    v, t = grid(nx, ny, cloth["h"])
    first = b.particle_count
    common = dict(pos=wp.vec3(*pos), rot=rot if rot is not None else wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                  vertices=[wp.vec3(*p) for p in v], indices=t, density=cloth["w"], particle_radius=cloth["r"])
    Y, nu, B = cloth["Y"], cloth["nu"], cloth["B"]
    mu, lam = Y / (2.0 * (1.0 + nu)), Y * nu / (1.0 - nu * nu)          # 2D Lame from Y, nu (plane stress)
    if solver == "style3d":                            # Baraff-Witkin stretch (N/m) and Bergou bending (N m)
        style3d.add_cloth_mesh(b, **common, tri_aniso_ke=wp.vec3(Y, Y, mu), edge_aniso_ke=wp.vec3(B, B, B))
    elif solver == "vbd":                              # Neo-Hookean membrane; bending set per edge after finalize
        b.add_cloth_mesh(**common, scale=1.0, tri_ke=mu, tri_ka=lam, tri_kd=0.0, edge_ke=B, edge_kd=0.0)
    elif solver == "semi":                             # the same stable Neo-Hookean, area term in Smith's lambda + mu
        b.add_cloth_mesh(**common, scale=1.0, tri_ke=mu, tri_ka=lam + mu, tri_kd=SEMI["tri_kd"], edge_ke=B,
                         edge_kd=SEMI["edge_kd"], tri_drag=SEMI["rho_cd"] / 6.0)   # its drag acts 3 times per triangle
    else:                                              # xpbd: a spring on every mesh edge (axis + diagonal: Y = 1.5 ke)
        b.add_cloth_mesh(**common, scale=1.0, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0, edge_ke=B, edge_kd=0.0,
                         add_springs=True, spring_ke=Y / 1.5, spring_kd=0.0)
    return np.arange(first, b.particle_count)


def static_box(b, center, half, mu, rot=None):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu, cfg.ke, cfg.kd, cfg.kf = mu, CONTACT["ke"], CONTACT["kd"], SEMI["kf"]
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*center), rot if rot is not None else wp.quat_identity()),
                    hx=half[0], hy=half[1], hz=half[2], cfg=cfg)


class Sim:
    """One scene on one solver: substeps per 30 fps frame, air drag c (1/s) applied to every velocity."""

    def __init__(self, solver, scene, cloth, worlds=1, drag=0.0, **over):
        cfg = dict(SOLVERS[solver], **{k: v for k, v in over.items() if k in SOLVERS[solver]})
        self.solver, self.cfg, self.drag = solver, cfg, drag
        b0 = new_builder(solver)
        self.meta = scene(b0, solver, cloth)
        if worlds > 1:
            b = new_builder(solver)
            b.replicate(b0, worlds, spacing=(2.0, 0.0, 0.0))
        else:
            b = b0
        if solver == "vbd":
            b.color(include_bending=True)
        model = b.finalize()
        npw = model.particle_count // worlds
        self.npw, self.worlds = npw, worlds
        fixed = np.asarray(self.meta.get("fixed", []), dtype=int)
        if len(fixed):
            fixed = (fixed[None, :] + npw * np.arange(worlds)[:, None]).ravel()
            m = model.particle_mass.numpy()
            m[fixed] = 0.0
            model.particle_mass = wp.array(m, dtype=wp.float32)
            im = model.particle_inv_mass.numpy()
            im[fixed] = 0.0
            model.particle_inv_mass = wp.array(im, dtype=wp.float32)
            fl = model.particle_flags.numpy()
            fl[fixed] = fl[fixed] & ~int(newton.ParticleFlags.ACTIVE)
            model.particle_flags = wp.array(fl, dtype=model.particle_flags.dtype)
        if solver in ("vbd", "semi") and model.edge_count:   # discrete-shell bending from the plate stiffness B
            e = model.edge_indices.numpy()
            x = model.particle_q.numpy()
            ok = (e[:, 0] >= 0) & (e[:, 1] >= 0)
            L = np.linalg.norm(x[e[:, 3]] - x[e[:, 2]], axis=1)
            A = np.zeros(len(e))
            for o in (0, 1):
                A[ok] += 0.5 * np.linalg.norm(np.cross(x[e[ok, 2]] - x[e[ok, o]], x[e[ok, 3]] - x[e[ok, o]]), axis=1)
            props = model.edge_bending_properties.numpy()
            # E = k |e| theta^2 / 2 per edge, k = 3 B |e| / (4 (A1 + A2)) (discrete shells) times 2: on this grid
            # the plain discrete-shell k bends a cantilever along a grid axis like 0.52 B (explicit solver,
            # bending length 5 cells, against the heavy elastica)
            k = 2.0 * 3.0 * cloth["B"] * L / (4.0 * np.maximum(A, 1e-12))
            if solver == "semi":                       # its kernel pushes the wing vertices with ke theta |e|^2 (no
                k = k / np.maximum(A, 1e-12)           # 1 / height): ke = k / (A1 + A2) gives the same moment
            props[:, 0] = np.where(ok, k, 0.0)
            props[:, 1] = SEMI["edge_kd"] if solver == "semi" else 0.0
            model.edge_bending_properties = wp.array(props, dtype=wp.float32)
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = CONTACT["ke"], CONTACT["kd"], cloth["mu"]
        model.soft_contact_kf = SEMI["kf"]
        self.model = model
        if solver == "vbd":
            sc = dict(particle_enable_self_contact=True, particle_self_contact_margin=2.0 * cloth["r"],
                      particle_self_contact_gap=cloth["r"]) if over.get("self_contact") else {}
            self.s = newton.solvers.SolverVBD(model, iterations=cfg["iters"], friction_epsilon=cfg["feps"], **sc)
        elif solver == "style3d":
            _s3.update_velocity = _velocity_undamped if over.get("undamped", True) else _S3_VELOCITY
            _s3.init_step_kernel = _init_step_inertial if cfg["predictor"] else _S3_INIT
            self.s = newton.solvers.SolverStyle3D(model, iterations=cfg["iters"], linear_iterations=cfg["lin"])
            self.s.collision.friction_epsilon = cfg["feps"]
        elif solver == "semi":
            model.particle_grid = None                 # no particle-particle or all-pairs triangle contact
            self.s = newton.solvers.SolverSemiImplicit(model, enable_tri_contact=False)
        else:
            self.s = newton.solvers.SolverXPBD(model, iterations=cfg["iters"])
        self.s0, self.s1, self.ctrl = model.state(), model.state(), model.control()
        self.graph = None
        self.self_contact = bool(over.get("self_contact", 0)) and solver == "semi"
        if self.self_contact:                          # the cloth's own BVH over a copy of its positions
            self.sc_x = wp.clone(self.s0.particle_q)
            self.sc_mesh = wp.Mesh(points=self.sc_x, indices=model.tri_indices.flatten())
            self.sc_thick = SELF["thick"]
            self.sc_query = SELF["thick"] + 0.004      # 4 mm margin: what a vertex moves in one refresh
            self.sc_cand = wp.zeros((model.particle_count, 32), dtype=int)
            self.sc_count = wp.zeros(model.particle_count, dtype=int)
        self.kin = None
        self.clock = wp.zeros(1, dtype=float)
        self.pipe = newton.CollisionPipeline(model, soft_contact_gap=0.005)
        self.con = self.pipe.contacts()
        self.dt = 1.0 / (FPS * cfg["substeps"])
        self.mass = model.particle_mass.numpy()
        self.t, self.wall = 0.0, 0.0

    def _refit(self):
        wp.copy(self.sc_x, self.s0.particle_q)
        self.sc_mesh.refit()
        wp.launch(_self_candidates, dim=self.model.particle_count,
                  inputs=[self.sc_mesh.id, self.s0.particle_q, self.model.tri_indices, self.sc_query],
                  outputs=[self.sc_cand, self.sc_count])

    def grasp(self, ids, vel):
        """Pin these vertices (world 0) to a gripper moving at vel (m/s) from where they are now."""
        ids = np.asarray(ids, dtype=int)
        x0 = self.s0.particle_q.numpy()[ids]
        self.kin = (wp.array(ids, dtype=int), wp.array(x0, dtype=wp.vec3),
                    wp.array(np.tile(np.asarray(vel, dtype=np.float32), (len(ids), 1)), dtype=wp.vec3))
        self.clock.zero_()
        self.graph = None                              # re-capture with the gripper in the loop

    def grasp_force(self):
        """Force the gripper applies to the cloth: minus the cloth's internal force on the grasped vertices."""
        f = self.s1.particle_f.numpy()[self.kin[0].numpy()]
        return -f.sum(0)

    def _substeps(self, k, collide_every, damp):
        for i in range(k):
            if i % collide_every == 0:
                self.pipe.collide(self.s0, self.con)
            self.s0.clear_forces()
            if self.kin is not None:
                wp.launch(_kinematic, dim=len(self.kin[0]), inputs=[*self.kin, self.clock, self.s0.particle_q, self.s0.particle_qd])
            if self.solver == "semi" and SEMI["obj_kd"] > 0.0:
                wp.launch(_membrane_damping, dim=self.model.tri_count,
                          inputs=[self.s0.particle_q, self.s0.particle_qd, self.model.tri_indices, self.model.tri_poses,
                                  SEMI["obj_kd"], self.s0.particle_f])
            if self.self_contact:
                wp.launch(_self_contact, dim=self.model.particle_count,
                          inputs=[self.s0.particle_q, self.s0.particle_qd, self.model.tri_indices, self.sc_cand,
                                  self.sc_count, self.sc_thick, SELF["ke"], SEMI["kf"], SELF["mu"], self.s0.particle_f])
            self.s.step(self.s0, self.s1, self.ctrl, self.con, self.dt)
            if self.kin is not None:
                wp.launch(_tick, dim=1, inputs=[self.dt, self.clock])
            self.s0, self.s1 = self.s1, self.s0
            if self.drag > 0.0:
                wp.launch(_drag, dim=self.model.particle_count, inputs=[damp, self.s0.particle_qd])

    def frame(self, n=1):
        damp = float(np.exp(-self.drag * self.dt))
        t0 = time.time()
        if self.solver == "semi":                      # one captured graph per contact refresh (an even count,
            ce = self.cfg["collide_every"]             # so the state buffers end where they started)
            done = 0
            if self.graph is None:
                if self.self_contact:
                    self._refit()
                self._substeps(ce, ce, damp)           # the first chunk runs uncaptured (compiles the kernels)
                done = 1
                with wp.ScopedCapture() as cap:        # recorded, not run
                    self._substeps(ce, ce, damp)
                self.graph = cap.graph
            for _ in range(n):
                for _ in range(self.cfg["substeps"] // ce - done):
                    if self.self_contact:              # the BVH is refit outside the graph, once per refresh
                        self._refit()
                    wp.capture_launch(self.graph)
                done = 0
                self.t += 1.0 / FPS
            wp.synchronize()
            self.wall += time.time() - t0
            return
        for _ in range(n):
            for _ in range(self.cfg["substeps"]):
                self.s0.clear_forces()
                self.pipe.collide(self.s0, self.con)
                self.s.step(self.s0, self.s1, self.ctrl, self.con, self.dt)
                self.s0, self.s1 = self.s1, self.s0
                if self.drag > 0.0:
                    wp.launch(_drag, dim=self.model.particle_count, inputs=[damp, self.s0.particle_qd])
            self.t += 1.0 / FPS
        wp.synchronize()
        self.wall += time.time() - t0

    def rate(self):
        """Simulated seconds per wall second (all worlds)."""
        return self.worlds * self.t / max(self.wall, 1e-9)

    def x(self):
        return self.s0.particle_q.numpy()[:self.npw].astype(np.float64)

    def v(self):
        return self.s0.particle_qd.numpy()[:self.npw].astype(np.float64)


def _hanging_strip(width, length):
    def scene(b, solver, cloth):
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)    # grid y -> world z: hangs in x-z
        h = cloth["h"]
        ids = add_sheet(b, solver, cloth, int(round(width / h)), int(round(length / h)), (-width / 2, 0.0, 1.0), rot)
        z = np.array([b.particle_q[i][2] for i in ids])
        return {"fixed": ids[z > z.max() - 1e-6], "ids": ids}
    return scene


def check_stretch(solver, **over):
    out = {}
    for Y in (20.0, 1.0e3, 1.0e4):
        cloth = dict(CLOTH, Y=Y, nu=0.0)
        sim = Sim(solver, _hanging_strip(0.05, 0.30), cloth, drag=4.0, **over)
        x0 = sim.x()
        sim.frame(int(3.0 * FPS))
        xs = []
        for _ in range(int(1.0 * FPS)):              # the last second's mean: a stiff strip's 9 um extension sits
            sim.frame()                              # at fp32's resolution 1 m up and jitters (3 cm/s at Y 1e4)
            xs.append(sim.x())
        x = np.mean(xs, axis=0)
        act = sim.mass[:sim.npw] > 0
        bottom = x0[:, 2] < x0[:, 2].min() + 1e-6
        ext = float((x0[bottom, 2] - x[bottom, 2]).mean())
        L = float(x0[:, 2].max() - x0[:, 2].min())
        want = cloth["w"] * G * L ** 2 / (2.0 * Y)
        speed = float(np.linalg.norm(sim.v()[act], axis=1).max())
        out[f"Y{Y:g}"] = {"extension_mm": 1e3 * ext, "closed_form_mm": 1e3 * want, "ratio": ext / want,
                          "max_speed_end_m_s": speed, "sim_s_per_wall_s": sim.rate(),
                          "pass": bool(abs(ext / want - 1.0) < 0.05)}
        print(f"  stretch {solver:8s} Y {Y:7.0f} N/m: extension {1e3 * ext:8.4f} mm, closed form {1e3 * want:8.4f} mm "
              f"(ratio {ext / want:6.3f}), speed at end {speed:.4f} m/s, {sim.rate():.2f} sim-s/wall-s", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_swing(solver, **over):
    out = {}
    def scene(b, solver, cloth):
        h = cloth["h"]
        ids = add_sheet(b, solver, cloth, 30, 30, (0.0, -0.15, 1.0))
        x = np.array([b.particle_q[i] for i in ids])
        return {"fixed": ids[x[:, 0] < 1e-6], "ids": ids}
    for drag in (0.0, 2.0):
        sim = Sim(solver, scene, CLOTH, drag=drag, **over)
        m = sim.mass[:sim.npw]
        act = m > 0
        e0 = float((m * G * sim.x()[:, 2]).sum())
        worst, released, series = -np.inf, 0.0, []
        for f in range(int(6.0 * FPS)):
            sim.frame()
            x, v = sim.x(), sim.v()
            pe = float((m * G * x[:, 2]).sum())
            ke = 0.5 * float((m * (v ** 2).sum(1)).sum())
            worst = max(worst, ke + pe - e0)
            released = max(released, e0 - pe)
            series.append([round(sim.t, 4), ke, pe - e0, float(np.linalg.norm(v[act], axis=1).max())])   # every frame: it swings at ~1 Hz
            if f == int(5.0 * FPS) - 1:
                x5 = x.copy()
        sp = float(np.linalg.norm(sim.v()[act], axis=1).max())
        move = float(np.linalg.norm(sim.x() - x5, axis=1).max())      # over the last second: the rest test
        row = {"energy_gain_J": worst, "energy_released_J": released, "gain_fraction": worst / max(released, 1e-12),
               "max_speed_end_m_s": sp, "max_move_last_second_mm": 1e3 * move, "series_t_ke_dpe_vmax": series,
               "sim_s_per_wall_s": sim.rate(), "in_bound": bool(np.isfinite(sim.x()).all() and np.abs(sim.x()).max() < 3.0)}
        if drag == 0.0:
            row["pass"] = bool(row["in_bound"] and row["gain_fraction"] < 0.01)
        else:
            row["pass"] = bool(row["in_bound"] and move < 0.005)
        out[f"drag{drag:g}"] = row
        print(f"  swing   {solver:8s} drag {drag:3.1f}/s: max (KE+PE) gain {worst:+.2e} J of {released:.2e} J released "
              f"({100 * row['gain_fraction']:+.2f}%), moved {1e3 * move:.2f} mm in the last second (speed {sp:.4f} m/s), "
              f"{sim.rate():.2f} sim-s/wall-s", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_freefall(solver, **over):
    """Centroid acceleration from a quadratic fit over 0.5 s (implicit Euler adds a constant velocity
    offset of g dt / 2, which a fit of position against time absorbs)."""
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    def scene(b, solver, cloth):
        return {"ids": add_sheet(b, solver, cloth, 20, 20, (-0.1, -0.1, 2.0),
                                 wp.quat_from_axis_angle(wp.vec3(*axis), 0.4))}
    sim = Sim(solver, scene, CLOTH, **over)
    m = sim.mass[:sim.npw]
    x0 = sim.x()
    d0 = np.linalg.norm(x0 - x0[0], axis=1)
    ts, zs, shape = [], [], 0.0
    for f in range(15):
        sim.frame()
        x = sim.x()
        ts.append(sim.t)
        zs.append(float((m * x[:, 2]).sum() / m.sum()))
        shape = max(shape, float(np.abs(np.linalg.norm(x - x[0], axis=1) - d0).max()))
    a = -2.0 * np.polyfit(np.array(ts), np.array(zs), 2)[0]
    out = {"accel_m_s2": float(a), "rel_error": float(a / G - 1.0), "shape_change_mm": 1e3 * shape,
           "pass": bool(abs(a / G - 1.0) < 1e-3 and shape < 1e-4)}
    print(f"  freefall {solver:7s}: centroid accel {a:.4f} m/s2 ({100 * (a / G - 1):+.3f}% of g), "
          f"shape change {1e3 * shape:.3f} mm", flush=True)
    return out


def check_airfall(solver, **over):
    """Air drag on the normal velocity: a flat sheet falling face-down reaches the terminal speed
    sqrt(2 w g / (rho Cd)) (5%), while a sheet sliding in its own plane feels none (the slide check)."""
    if solver != "semi":
        return {"skipped": "no aerodynamic model"}
    SEMI["rho_cd"], rho_cd = 1.44, SEMI["rho_cd"]
    try:
        def scene(b, solver, cloth):
            return {"ids": add_sheet(b, solver, cloth, 20, 20, (-0.1, -0.1, 10.0))}
        sim = Sim(solver, scene, CLOTH, **over)
        sim.frame(int(3.0 * FPS))
        v = float(-sim.v()[:, 2].mean())
    finally:
        SEMI["rho_cd"] = rho_cd
    want = float(np.sqrt(2.0 * CLOTH["w"] * G / 1.44))
    out = {"terminal_m_s": v, "closed_form_m_s": want, "ratio": v / want, "pass": bool(abs(v / want - 1.0) < 0.05)}
    print(f"  airfall {solver:8s}: falls at {v:.3f} m/s after 3 s, terminal speed sqrt(2 w g / rho Cd) {want:.3f} "
          f"(ratio {v / want:.3f})", flush=True)
    return out


def check_slide(solver, theta_deg=20.0, **over):
    th = np.radians(theta_deg)
    out = {}
    for mu in (0.2, 0.5):
        cloth = dict(CLOTH, mu=mu)
        def scene(b, solver, cloth):
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-th))     # the plane rises toward +x
            R = np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)
            static_box(b, (0.0, 0.0, 0.5), (3.0, 0.5, 0.05), mu, q)                # 6 m: the sheet stays on it
            top = np.array([0.0, 0.0, 0.5]) + R @ np.array([0.0, 0.0, 0.05 + cloth["r"]])
            ids = add_sheet(b, solver, cloth, 10, 10, tuple(top + R @ np.array([-0.05, -0.05, 0.0])), q)
            return {"ids": ids, "down": -R[:, 0]}
        sim = Sim(solver, scene, cloth, **over)
        down = sim.meta["down"]
        m = sim.mass[:sim.npw]
        ts, ss = [], []
        for f in range(int(2.0 * FPS)):
            sim.frame()
            c = (m[:, None] * sim.x()).sum(0) / m.sum()
            ts.append(sim.t)
            ss.append(float(c @ down))
        ts, ss = np.array(ts), np.array(ss) - ss[0]
        want = G * (np.sin(th) - mu * np.cos(th))
        if want > 0:
            w = (ts > 0.3) & (ts < 1.3)
            a = 2.0 * np.polyfit(ts[w], ss[w], 2)[0]
            row = {"accel_m_s2": float(a), "slide_law_m_s2": float(want), "rel_error": float(a / want - 1.0),
                   "pass": bool(abs(a / want - 1.0) < 0.02)}
            print(f"  slide   {solver:8s} mu {mu}: accel {a:.4f} m/s2, law {want:.4f} ({100 * (a / want - 1):+.2f}%)", flush=True)
        else:
            w = ts > 1.0
            drift = float(np.polyfit(ts[w], ss[w], 1)[0])
            row = {"drift_mm_s": 1e3 * drift, "travel_mm": 1e3 * float(ss[-1]), "pass": bool(abs(drift) < 1e-3)}
            print(f"  slide   {solver:8s} mu {mu}: drift {1e3 * drift:+.3f} mm/s over 1-2 s, travel {1e3 * ss[-1]:+.3f} mm", flush=True)
        out[f"mu{mu}"] = row
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def elastica_tip(l, c):
    """Heavy cantilever clamped horizontal, uniform weight W, bending length c = (B / W)^(1/3):
    B th'' = W (l - s) cos th, th(0) = 0, th'(l) = 0. Returns tip drop / l and the tip chord angle
    (Peirce's 41.5 deg at l = 2 c)."""
    from scipy.integrate import solve_bvp
    k = 1.0 / c ** 3
    def f(s, y):                                       # y = theta, theta', x, z
        return np.vstack([y[1], k * (l - s) * np.cos(y[0]), np.cos(y[0]), np.sin(y[0])])
    s = np.linspace(0.0, l, 400)
    y0 = np.zeros((4, s.size))
    y0[2] = s
    sol = solve_bvp(f, lambda ya, yb: np.array([ya[0], yb[1], ya[2], ya[3]]), s, y0, tol=1e-8, max_nodes=100000)
    x, z = sol.y[2, -1], sol.y[3, -1]
    return float(-z / l), float(np.degrees(np.arctan2(-z, x)))


def check_bend(solver, **over):
    out = {}
    for tag, B, h in (("B", CLOTH["B"], 0.01), ("B_half_mesh", CLOTH["B"], 0.005), ("8B_double_overhang", 8 * CLOTH["B"], 0.01)):
        cloth = dict(CLOTH, B=B, h=h)
        c = (B / (cloth["w"] * G)) ** (1.0 / 3.0)
        l = round(2.0 * c / h) * h                     # Peirce: the tip chord falls ~41.5 deg at l = 2 c
        def scene(b, solver, cloth, l=l):
            nx = int(round((l + 0.03) / cloth["h"]))
            ids = add_sheet(b, solver, cloth, nx, int(round(0.03 / cloth["h"])), (-0.03, -0.015, 1.0))
            x = np.array([b.particle_q[i] for i in ids])
            return {"fixed": ids[x[:, 0] < 1e-6], "ids": ids, "tip": ids[x[:, 0] > x[:, 0].max() - 1e-6]}
        sim = Sim(solver, scene, cloth, drag=4.0, **over)
        sim.frame(int(5.0 * FPS))
        x = sim.x()
        tip = x[sim.meta["tip"]].mean(0)
        drop = 1.0 - tip[2]
        want_drop, want_angle = elastica_tip(l, c)
        angle = float(np.degrees(np.arctan2(drop, tip[0])))
        speed = float(np.linalg.norm(sim.v(), axis=1).max())
        row = {"overhang_m": l, "bending_length_m": c, "tip_drop_over_l": drop / l, "elastica_drop_over_l": want_drop,
               "tip_angle_deg": angle, "elastica_angle_deg": want_angle, "ratio": (drop / l) / want_drop,
               "max_speed_end_m_s": speed, "pass": bool(abs((drop / l) / want_drop - 1.0) < 0.10)}
        out[tag] = row
        print(f"  bend    {solver:8s} {tag:19s} l {100 * l:4.1f} cm c {100 * c:4.2f} cm: tip drop/l {drop / l:.3f}, "
              f"elastica {want_drop:.3f} (ratio {row['ratio']:.3f}); angle {angle:5.1f} vs {want_angle:5.1f} deg", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


TOP = 0.75


def _towel_on_table(over=0.3, L=0.30, drop=0.002):
    def scene(b, solver, cloth):
        b.add_ground_plane()
        static_box(b, (-0.5, 0.0, TOP / 2), (0.5, 0.5, TOP / 2), cloth["mu"])
        n = int(round(L / cloth["h"]))
        ids = add_sheet(b, solver, cloth, n, n, (-(1.0 - over) * L, -L / 2, TOP + cloth["r"] + drop))
        return {"ids": ids}
    return scene


def check_drape(solver, **over):
    sim = Sim(solver, _towel_on_table(), CLOTH, drag=1.0, **over)
    sim.frame(int(4.0 * FPS))
    xa = sim.x()
    sim.frame(int(1.0 * FPS))
    xb = sim.x()
    on = xb[:, 0] < -0.01
    drift = float(np.linalg.norm(xb[on] - xa[on], axis=1).max())
    move = float(np.linalg.norm(xb - xa, axis=1).max())
    sp = float(np.linalg.norm(sim.v(), axis=1).max())
    pen = float((xb[on, 2] - TOP - CLOTH["r"]).min())
    hang = float(TOP - xb[:, 2].min())
    out = {"max_move_last_second_mm": 1e3 * move, "max_speed_end_m_s": sp, "creep_on_table_mm_s": 1e3 * drift,
           "penetration_min_mm": 1e3 * pen, "on_table_fraction": float(on.mean()), "hanging_depth_m": hang,
           "sim_s_per_wall_s": sim.rate(),
           # a towel that bends over 2.5 cm hangs most of a 9 cm overhang (the heavy elastica: ~0.9 of it)
           "pass": bool(move < 0.005 and drift < 1e-3 and pen > -1e-3 and hang > 0.5 * 0.09)}
    print(f"  drape   {solver:8s}: moved {1e3 * move:.2f} mm in the last second (speed {sp:.4f} m/s), creep on table "
          f"{1e3 * drift:.3f} mm/s, lowest on-table vertex {1e3 * pen:+.3f} mm, on table {100 * on.mean():.1f}%, "
          f"hangs {100 * hang:.1f} cm, {sim.rate():.2f} sim-s/wall-s", flush=True)
    return out


def crossings(x, tris):
    """Mesh edges that pass through a triangle they do not touch (segment-triangle tests, all pairs)."""
    tris = np.asarray(tris).reshape(-1, 3)
    e = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    p, q = x[e[:, 0]], x[e[:, 1]]
    a, b, c = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
    n_hit = 0
    for s0 in range(0, len(e), 256):
        P, Q, E = p[s0:s0 + 256, None], q[s0:s0 + 256, None], e[s0:s0 + 256]
        share = (E[:, None, :, None] == tris[None, :, None, :]).any(axis=(2, 3))
        d = Q - P
        e1, e2 = b - a, c - a
        h = np.cross(d, e2[None])
        det = (h * e1[None]).sum(-1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = P - a[None]
        u = (sv * h).sum(-1) * inv
        qv = np.cross(sv, e1[None])
        v = (d * qv).sum(-1) * inv
        t = (e2[None] * qv).sum(-1) * inv
        hit = ok & ~share & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
        n_hit += int(hit.sum())
    return n_hit


def check_fold(solver, **over):
    """The fold a robot makes: a towel flat on the table, one corner grasped, carried over (lifted 12 cm)
    onto the opposite corner and let go. Count edges through triangles at the end, with self-contact and
    (the control that shows the count sees interpenetration) without; with it, the folded half rests on
    the other (two layers)."""
    out = {}
    def scene(b, solver, cloth):
        b.add_ground_plane()
        static_box(b, (0.0, 0.0, TOP / 2), (0.5, 0.5, TOP / 2), cloth["mu"])
        return {"ids": add_sheet(b, solver, cloth, 30, 30, (-0.40, -0.15, TOP + cloth["r"]))}
    v, t = grid(30, 30, CLOTH["h"])
    for sc in ((0, 1) if solver == "semi" else (1,)):
        sim = Sim(solver, scene, CLOTH, drag=1.0, **dict(over, self_contact=sc))
        sim.frame(5)
        x = sim.x()
        corner = int(np.argmin(np.abs(x[:, 0] - x[:, 0].max()) + np.abs(x[:, 1] - x[:, 1].max())))
        far = int(np.argmin(np.abs(x[:, 0] - x[:, 0].min()) + np.abs(x[:, 1] - x[:, 1].min())))
        sim.grasp([corner], (-0.25, -0.25, 0.20))     # up and across for 0.6 s,
        sim.frame(18)
        sim.grasp([corner], (-0.25, -0.25, -0.19))    # across and down for 0.6 s,
        sim.frame(18)
        sim.kin, sim.graph = None, None                # let go
        sim.frame(int(1.5 * FPS))
        x = sim.x()
        n = crossings(x, t)
        top = x[:, 2] > TOP + CLOTH["r"] + 0.5 * SELF["thick"]
        out[f"self_contact{sc}"] = {"crossings": n, "on_second_layer": float(top.mean()),
                                    "corner_to_far_corner_mm": 1e3 * float(np.linalg.norm(x[corner, :2] - x[far, :2])),
                                    "sim_s_per_wall_s": sim.rate()}
        print(f"  fold    {solver:8s} self-contact {sc}: {n} edges through triangles, {100 * top.mean():.0f}% of the "
              f"towel on a second layer, corner {1e3 * np.linalg.norm(x[corner, :2] - x[far, :2]):.0f} mm from the far "
              f"corner, {sim.rate():.2f} sim-s/wall-s", flush=True)
    out["pass"] = bool(out["self_contact1"]["crossings"] == 0 and out["self_contact1"]["on_second_layer"] > 0.3)
    return out


def check_grasp(solver, **over):
    """Lift a towel lying on the table by one corner (0.2 m/s for 3 s, then held 2 s): the gripper's
    force on the cloth equals the cloth's weight once it hangs free (2%)."""
    def scene(b, solver, cloth):
        b.add_ground_plane()
        static_box(b, (0.0, 0.0, TOP / 2), (0.5, 0.5, TOP / 2), cloth["mu"])
        return {"ids": add_sheet(b, solver, cloth, 30, 30, (-0.15, -0.15, TOP + cloth["r"]))}
    sim = Sim(solver, scene, CLOTH, drag=1.0, self_contact=1, **over)
    sim.frame(10)
    x = sim.x()
    corner = int(np.argmin(np.abs(x[:, 0] - x[:, 0].min()) + np.abs(x[:, 1] - x[:, 1].min())))
    sim.grasp([corner], (0.0, 0.0, 0.2))
    sim.frame(int(3.0 * FPS))
    sim.grasp([corner], (0.0, 0.0, 0.0))               # hold still where it is
    fs = []
    for f in range(int(2.0 * FPS)):
        sim.frame()
        if f >= int(1.5 * FPS):
            fs.append(sim.grasp_force())
    F = np.mean(fs, axis=0)
    W = float(sim.mass[:sim.npw].sum() * G)
    low = float(sim.x()[:, 2].min() - TOP)
    out = {"grasp_force_N": F.tolist(), "weight_N": W, "ratio": float(F[2] / W), "lowest_above_table_mm": 1e3 * low,
           "pass": bool(abs(F[2] / W - 1.0) < 0.02 and low > 0.0)}
    print(f"  grasp   {solver:8s}: gripper force {F[2]:.5f} N up ({F[0]:+.5f}, {F[1]:+.5f} sideways) against the weight "
          f"{W:.5f} N (ratio {F[2] / W:.4f}); lowest vertex {1e3 * low:.1f} mm above the table", flush=True)
    return out


def check_throughput(solver, worlds=64, seconds=2.0, **over):
    over = dict({"self_contact": 1}, **over)           # as the teacher runs: with self-contact
    sim = Sim(solver, _towel_on_table(), CLOTH, worlds=worlds, drag=1.0, **over)
    sim.frame(1)
    wp.synchronize()
    t0 = time.time()
    sim.frame(int(seconds * FPS))
    wp.synchronize()
    wall = time.time() - t0
    rate = worlds * seconds / wall
    out = {"worlds": worlds, "sim_seconds_per_wall_second": rate, "four_second_episodes_per_hour": 3600.0 * rate / 4.0}
    print(f"  throughput {solver:5s} x{worlds}: {rate:.2f} sim-s/wall-s = {out['four_second_episodes_per_hour']:.0f} "
          f"four-second towel episodes per hour", flush=True)
    return out


CHECKS = {"stretch": check_stretch, "swing": check_swing, "freefall": check_freefall, "airfall": check_airfall,
          "slide": check_slide,
          "bend": check_bend, "drape": check_drape, "fold": check_fold, "grasp": check_grasp,
          "throughput": check_throughput}


def main():
    args = sys.argv[1:]
    solvers = list(SOLVERS)
    over, names = {}, []
    i = 0
    while i < len(args):
        if args[i] == "--solver":
            solvers = args[i + 1].split(",")
            i += 2
        elif args[i].startswith("--"):
            k, v = args[i][2:], args[i + 1]
            over[k] = float(v) if any(ch in v for ch in ".e") else int(v)
            i += 2
        else:
            names.append(args[i])
            i += 1
    names = names or list(CHECKS)
    wp.init()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    tag = "" if not over else "_" + "_".join(f"{k}{v:g}" for k, v in over.items())
    for s in solvers:
        key = s + tag
        res.setdefault(key, {})
        for n in names:
            try:
                res[key][n] = CHECKS[n](s, **over)
            except Exception as exc:                    # record the failure, keep going
                res[key][n] = {"error": repr(exc)}
                print(f"  {n} {s}: ERROR {exc!r}", flush=True)
            OUT.write_text(json.dumps(res, indent=1))
    res["_versions"] = {"newton": newton.__version__, "warp": wp.config.version}
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
