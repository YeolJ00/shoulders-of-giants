"""The cloth teacher: Newton's explicit integrator (SolverSemiImplicit) at small steps, as the cloth teacher
gate passed it (DIRECTION.md "Cloth teacher gate", scripts/l0_cloth_gate.py).

Newton's implicit cloth solvers run stably and settle, yet simulate a different cloth from the one they are
given: VBD is not converged on a light, stiff cloth (a cotton towel reads 207x too stretchy, bends like ~1/500
of its stiffness), Style3D's fixed projective-dynamics matrix resists rotation (a cantilever stays level at any
bending stiffness). The explicit integrator has nothing to converge; it is right while the step is stable,
dt <= ~sqrt(m_vertex / Y) / 2. What the gate measured as needed on top of Newton's explicit solver:

  bending    its kernel pushes the wing vertices with ke theta |e|^2 (no 1/height): per edge
             ke = 2 x 3 B |e| / (4 (A1 + A2)^2) gives the plate stiffness B (discrete shells x 2, the regular
             grid's factor from the Peirce cantilever against the heavy elastica)
  damping    its membrane damping damps dF/dt, rotation included: replaced by ours on the strain rate
             C' = F'^T F + F^T F' (blind to rigid motion)
  contact    ours (`_shape_contact`): critically damped penalty and viscous friction below the Coulomb cap mu N,
             both per unit mass (2e7 / s^2, 2e4 / s: 1e3 N/m and 1 N s/m on a 50 mg vertex) so every cloth's
             contacts ring alike and stay stable stacked; mu the mean of the cloth's and the surface's (the rigid
             teacher's rule), per cloth; the force on the cloth by each kind of surface is recorded
  self       ours: vertex-triangle and edge-edge repulsion inside each cloth (the same per-mass law), 2 mm thick,
             candidates from a BVH refit
             every contact refresh
  air        Newton's quadratic drag on the normal velocity, tri_drag = rho Cd / 6 (it applies it at all
             three corners), rho Cd = 1.44 kg/m^3

The gripper is kinematic here: its path (a pose per frame, interpolated within the frame) is the action,
its finger boxes push the cloth, and a grasp pins the cloth vertices under the fingertips to it. The force it
applies through a grasp is recorded: minus the cloth's force on the pinned vertices, plus their own m (a - g).
Many worlds share one model; each world's cloth, material, obstacles and gripper are its own.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp
import newton
from newton._src.geometry.kernels import triangle_closest_point_barycentric

from lsim.teacher.cloth_mesh import grid, mesh_edges  # noqa: F401 -- re-exported

# Not bit-reproducible: atomic float sums land in a varying order, and a towel crumpling or falling amplifies
# that into its folds (a replay ended 21 mm, median, from its recording). Warp's deterministic atomics
# (RUN_TO_RUN, plus the pipeline's contact sort) made reruns exact but a batch 30x slower (867 s vs 29 s for
# 16 episodes), so the gates treat replay noise as each episode's sensitivity to tiny changes.
DETERMINISTIC = None

G = 9.81
FPS = 30
CONTACT_RATE = 2.0e7          # 1/s^2: contact stiffness per unit mass, ke = rate x m (1e3 N/m on a 50 mg vertex: a
                              # vertex sinks g / rate = 0.5 um, a contact rings at 4,500 rad/s on every cloth). A fixed
                              # 1e3 N/m rang at 14,000 rad/s on a 5 mg vertex: explicit steps of 1/30,000 s stayed
                              # stable for ~19 contacts at once, and a light towel pressed into a pile of itself
                              # exploded (piles of light sheets reached 290 m/s); now ~180
CONTACT_KE = CONTACT_RATE * 5.0e-5   # N/m on a 50 mg vertex, for reference
CONTACT_ZETA = 1.0            # normal damping, a fraction of critical (2 sqrt(ke m)): fabric does not bounce.
                              # Undamped, a towel dropped flat from 10 cm rebounded 28-49 mm and kept bouncing;
                              # a plain dashpot chattered (explicit kd dt / m > 2 on light vertices), so the
                              # damping force is implicit Euler's, -kd vn / (1 + kd dt / m): stable at any kd
FRICTION_RATE = 2.0e4         # 1/s: viscous friction below the Coulomb cap per unit mass (1 N s/m on 50 mg): stable
                              # while rate x dt < 2 (0.7 at 1,000 substeps); a resting vertex creeps g sin / rate
FRICTION_KF = FRICTION_RATE * 5.0e-5
SELF_THICK, SELF_MU = 0.002, 0.5
SELF_RATE = CONTACT_RATE      # self-contact stiffness per unit (effective) mass
SELF_KE = SELF_RATE * 5.0e-5
MEMBRANE_KD = 1.0e-3          # N s/m on the strain rate
RHO_CD = 1.44                 # kg/m^3: air density x a flat plate's drag coefficient
BEND_FACTOR = 2.0             # discrete shells x 2 on the alternating-diagonal grid
COLLIDE_EVERY = 50            # substeps per contact refresh (and per captured graph)
SELF_MARGIN = 0.004           # m: what a vertex may move in one refresh
SELF_EDGES = True             # edge-edge repulsion besides vertex-triangle
ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))
ROLE_TABLE, ROLE_GRIPPER, ROLE_OBSTACLE = 0, 1, 2


@dataclass
class Cloth:
    """A rectangular cloth in physical units: nx x ny cells of h, areal density w (kg/m^2), stretch
    modulus Y (N/m), Poisson ratio nu, bending stiffness B (N m), friction mu, vertex radius r."""
    nx: int
    ny: int
    h: float = 0.01
    w: float = 0.2
    Y: float = 1.0e4
    nu: float = 0.3
    B: float = 3.0e-5
    mu: float = 0.6
    r: float = 0.003

    @property
    def bending_length(self):
        return (self.B / (self.w * G)) ** (1.0 / 3.0)

    @property
    def vertex_mass(self):
        return self.w * self.h * self.h


def stable_substeps(cloths, fps=FPS, steps_per_period=2.0):
    """Substeps per frame for a batch: dt <= sqrt(m_vertex / Y) / steps_per_period for its hardest cloth
    (measured: stable at >= 1.55 steps per sqrt(m / Y), unstable at 1.34), at least 1,000 and a multiple
    of the contact refresh."""
    t = min(np.sqrt(c.vertex_mass / c.Y) for c in cloths)
    n = int(np.ceil(1.0 / (fps * t / steps_per_period)))
    return int(max(1000, int(np.ceil(n / COLLIDE_EVERY)) * COLLIDE_EVERY))


def add_cloth(b, cloth, pos, rot):
    """Add the cloth to builder b with the explicit solver's parameters. Returns (first particle, count,
    first edge, edge count)."""
    v, t = grid(cloth.nx, cloth.ny, cloth.h)
    p0, e0 = b.particle_count, b.edge_count
    mu_l = cloth.Y / (2.0 * (1.0 + cloth.nu))                    # 2D Lame from Y and nu (plane stress)
    lam = cloth.Y * cloth.nu / (1.0 - cloth.nu * cloth.nu)
    b.add_cloth_mesh(pos=wp.vec3(*pos), rot=rot, scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                     vertices=[wp.vec3(*p) for p in v], indices=t.flatten().tolist(), density=cloth.w,
                     tri_ke=mu_l, tri_ka=lam + mu_l,                # Newton's explicit triangle: stable Neo-Hookean
                     tri_kd=0.0, tri_drag=RHO_CD / 6.0, tri_lift=0.0,   # with its area term in Smith's lambda + mu
                     edge_ke=1.0, edge_kd=0.0, particle_radius=cloth.r)
    return p0, b.particle_count - p0, e0, b.edge_count - e0


def bending_ke(edge_indices, x, B):
    """Per-edge stiffness for the explicit kernel from the plate stiffness B (per edge)."""
    e = edge_indices
    ok = (e[:, 0] >= 0) & (e[:, 1] >= 0)
    L = np.linalg.norm(x[e[:, 3]] - x[e[:, 2]], axis=1)
    A = np.zeros(len(e))
    for o in (0, 1):
        A[ok] += 0.5 * np.linalg.norm(np.cross(x[e[ok, 2]] - x[e[ok, o]], x[e[ok, 3]] - x[e[ok, o]]), axis=1)
    A = np.maximum(A, 1e-12)
    return np.where(ok, BEND_FACTOR * 3.0 * B * L / (4.0 * A * A), 0.0)


# ---------------------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------------------
@wp.kernel
def _membrane_damping(x: wp.array[wp.vec3], v: wp.array[wp.vec3], tri: wp.array2d[int], pose: wp.array[wp.mat22],
                      kd: float, f: wp.array[wp.vec3]):
    """Rayleigh dissipation kd A |C'|^2 / 2 on the strain rate C' = F'^T F + F^T F': f_i = -2 kd A F C' g_i."""
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
    Fa = x10 * D[0, 0] + x20 * D[1, 0]
    Fb = x10 * D[0, 1] + x20 * D[1, 1]
    Va = v10 * D[0, 0] + v20 * D[1, 0]
    Vb = v10 * D[0, 1] + v20 * D[1, 1]
    caa = 2.0 * wp.dot(Va, Fa)
    cab = wp.dot(Va, Fb) + wp.dot(Fa, Vb)
    cbb = 2.0 * wp.dot(Vb, Fb)
    s = -2.0 * kd * area
    wp.atomic_add(f, i0, s * (Fa * (caa * g0[0] + cab * g0[1]) + Fb * (cab * g0[0] + cbb * g0[1])))
    wp.atomic_add(f, i1, s * (Fa * (caa * g1[0] + cab * g1[1]) + Fb * (cab * g1[0] + cbb * g1[1])))
    wp.atomic_add(f, i2, s * (Fa * (caa * g2[0] + cab * g2[1]) + Fb * (cab * g2[0] + cbb * g2[1])))


@wp.func
def _normal_force(ke: float, depth: float, vn: float, inv_m: float, zeta: float, dt: float, share: float):
    """Penalty plus critically damped normal force for a contact of effective inverse mass inv_m (vn < 0:
    closing), never pulling. The damping is implicit Euler's for the busiest vertex involved, as if its `share`
    contacts damped it alike: -kd vn / (1 + share kd dt / m) -- explicit damping summed over a pile's contacts
    overshot and popped compressed layers apart at 10-30 m/s; this never overshoots."""
    fd = float(0.0)
    if inv_m > 0.0:
        kd = 2.0 * zeta * wp.sqrt(ke / inv_m)
        fd = -kd * vn / (1.0 + share * kd * dt * inv_m)
    return wp.max(ke * depth + fd, 0.0)


@wp.kernel
def _offset_points(x: wp.array[wp.vec3], world: wp.array[int], spacing: float, out: wp.array[wp.vec3]):
    """Worlds share space; their copies in the self-contact BVH are set apart along y."""
    i = wp.tid()
    out[i] = x[i] + wp.vec3(0.0, spacing * float(world[i]), 0.0)


@wp.kernel
def _self_candidates(mesh: wp.uint64, xs: wp.array[wp.vec3], tri: wp.array2d[int], query: float,
                     cand: wp.array2d[int], count: wp.array[int]):
    """Once per contact refresh: the triangles of the same cloth, not its own, within the query distance."""
    i = wp.tid()
    xi = xs[i]
    q = wp.mesh_query_aabb(mesh, xi - wp.vec3(query), xi + wp.vec3(query))
    face = int(0)
    k = int(0)
    while wp.mesh_query_aabb_next(q, face):
        ia = tri[face, 0]
        ib = tri[face, 1]
        ic = tri[face, 2]
        if ia != i and ib != i and ic != i and k < cand.shape[1]:
            w = triangle_closest_point_barycentric(xs[ia], xs[ib], xs[ic], xi)
            if wp.length(xi - (w[0] * xs[ia] + w[1] * xs[ib] + w[2] * xs[ic])) < query:
                cand[i, k] = face
                k += 1
    count[i] = k


@wp.kernel
def _count_vt(x: wp.array[wp.vec3], tri: wp.array2d[int], cand: wp.array2d[int], count: wp.array[int], thick: float,
              nc: wp.array[float]):
    """Active self-contacts per vertex this substep (as the vertex and as a triangle's corner), for the damping's
    implicit share (_normal_force)."""
    i = wp.tid()
    xi = x[i]
    for k in range(count[i]):
        face = cand[i, k]
        ia = tri[face, 0]
        ib = tri[face, 1]
        ic = tri[face, 2]
        w = triangle_closest_point_barycentric(x[ia], x[ib], x[ic], xi)
        if wp.length(xi - (w[0] * x[ia] + w[1] * x[ib] + w[2] * x[ic])) < thick:
            wp.atomic_add(nc, i, 1.0)
            wp.atomic_add(nc, ia, 1.0)
            wp.atomic_add(nc, ib, 1.0)
            wp.atomic_add(nc, ic, 1.0)


@wp.kernel
def _count_ee(x: wp.array[wp.vec3], edges: wp.array2d[int], cand: wp.array2d[int], count: wp.array[int], thick: float,
              nc: wp.array[float]):
    i = wp.tid()
    a = edges[i, 0]
    b = edges[i, 1]
    for k in range(count[i]):
        j = cand[i, k]
        c = edges[j, 0]
        d = edges[j, 1]
        st = wp.closest_point_edge_edge(x[a], x[b], x[c], x[d], 1.0e-12)
        if st[2] < thick and st[0] > 1.0e-3 and st[0] < 0.999 and st[1] > 1.0e-3 and st[1] < 0.999:
            wp.atomic_add(nc, a, 1.0)
            wp.atomic_add(nc, b, 1.0)
            wp.atomic_add(nc, c, 1.0)
            wp.atomic_add(nc, d, 1.0)


@wp.kernel
def _self_contact(x: wp.array[wp.vec3], v: wp.array[wp.vec3], tri: wp.array2d[int], cand: wp.array2d[int],
                  count: wp.array[int], thick: float, rate: float, frate: float, mu: float, inv_mass: wp.array[float],
                  zeta: float, dt: float, nc: wp.array[float], f: wp.array[wp.vec3]):
    """A vertex closer than the thickness to a triangle of its cloth is pushed off along the separating
    direction (critically damped); the triangle's corners take the opposite force by barycentric weight."""
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
            vr = v[i] - (w[0] * v[ia] + w[1] * v[ib] + w[2] * v[ic])
            im = inv_mass[i] + w[0] * w[0] * inv_mass[ia] + w[1] * w[1] * inv_mass[ib] + w[2] * w[2] * inv_mass[ic]
            ke = rate / im
            kf = frate / im
            share = wp.max(1.0, wp.max(wp.max(nc[i], nc[ia]), wp.max(nc[ib], nc[ic])))
            fn = _normal_force(ke, thick - d, wp.dot(n, vr), im, zeta, dt, share)
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
def _edge_candidates(mesh: wp.uint64, xs: wp.array[wp.vec3], tri_edge: wp.array2d[int], edges: wp.array2d[int],
                     query: float, cand: wp.array2d[int], count: wp.array[int]):
    """Once per contact refresh: the edges of the same cloth within the query distance of this one that
    share no vertex with it, each pair listed once (under its lower index)."""
    i = wp.tid()
    a = edges[i, 0]
    b = edges[i, 1]
    xa = xs[a]
    xb = xs[b]
    q = wp.mesh_query_aabb(mesh, wp.min(xa, xb) - wp.vec3(query), wp.max(xa, xb) + wp.vec3(query))
    face = int(0)
    k = int(0)
    while wp.mesh_query_aabb_next(q, face):
        for m in range(3):
            j = tri_edge[face, m]
            if j > i:
                c = edges[j, 0]
                d = edges[j, 1]
                if c != a and c != b and d != a and d != b:
                    st = wp.closest_point_edge_edge(xa, xb, xs[c], xs[d], 1.0e-12)
                    if st[2] < query:
                        new = int(1)
                        for r in range(k):
                            if cand[i, r] == j:
                                new = 0
                        if new == 1 and k < cand.shape[1]:
                            cand[i, k] = j
                            k += 1
    count[i] = k


@wp.kernel
def _edge_contact(x: wp.array[wp.vec3], v: wp.array[wp.vec3], edges: wp.array2d[int], cand: wp.array2d[int],
                  count: wp.array[int], thick: float, rate: float, frate: float, mu: float, inv_mass: wp.array[float],
                  zeta: float, dt: float, nc: wp.array[float], f: wp.array[wp.vec3]):
    """Two edges of a cloth closer than the thickness at points inside both are pushed apart along their
    separation, the force shared by each edge's ends by weight. Vertex-triangle repulsion alone let edges
    slip through each other (12% of c1's episodes ended with the cloth through itself, a third of folds);
    a closest point at an edge's end is a vertex near a triangle, _self_contact's."""
    i = wp.tid()
    a = edges[i, 0]
    b = edges[i, 1]
    for k in range(count[i]):
        j = cand[i, k]
        c = edges[j, 0]
        d = edges[j, 1]
        st = wp.closest_point_edge_edge(x[a], x[b], x[c], x[d], 1.0e-12)
        s = st[0]
        t = st[1]
        dist = st[2]
        if dist < thick and dist > 1.0e-9 and s > 1.0e-3 and s < 0.999 and t > 1.0e-3 and t < 0.999:
            n = (x[a] + s * (x[b] - x[a]) - x[c] - t * (x[d] - x[c])) / dist
            vr = (1.0 - s) * v[a] + s * v[b] - (1.0 - t) * v[c] - t * v[d]
            im = (1.0 - s) * (1.0 - s) * inv_mass[a] + s * s * inv_mass[b] + (1.0 - t) * (1.0 - t) * inv_mass[c] + t * t * inv_mass[d]
            ke = rate / im
            kf = frate / im
            share = wp.max(1.0, wp.max(wp.max(nc[a], nc[b]), wp.max(nc[c], nc[d])))
            fn = _normal_force(ke, thick - dist, wp.dot(n, vr), im, zeta, dt, share)
            vt = vr - n * wp.dot(n, vr)
            vs = wp.length(vt)
            F = n * fn
            if vs > 1.0e-12:
                F = F - vt / vs * wp.min(kf * vs, mu * fn)
            wp.atomic_add(f, a, (1.0 - s) * F)
            wp.atomic_add(f, b, s * F)
            wp.atomic_add(f, c, -(1.0 - t) * F)
            wp.atomic_add(f, d, -t * F)


@wp.kernel
def _shape_contact(count: wp.array[int], cmax: int, c_particle: wp.array[int], c_shape: wp.array[int],
                   c_bpos: wp.array[wp.vec3], c_normal: wp.array[wp.vec3], shape_body: wp.array[int],
                   shape_mu: wp.array[float], shape_role: wp.array[int], body_q: wp.array[wp.transform],
                   body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], x: wp.array[wp.vec3],
                   v: wp.array[wp.vec3], radius: wp.array[float], flags: wp.array[wp.int32], p_mu: wp.array[float],
                   p_world: wp.array[int], rate: float, frate: float, inv_mass: wp.array[float], zeta: float, dt: float,
                   nc: wp.array[float], f: wp.array[wp.vec3], rec: wp.array2d[wp.vec3]):
    """Cloth on surfaces: critically damped penalty, friction viscous below the Coulomb cap with the mean of
    the cloth's and the surface's friction; the force on the cloth is summed per world and surface kind."""
    tid = wp.tid()
    if tid >= wp.min(cmax, count[0]):
        return
    i = c_particle[tid]
    if i < 0:
        return
    if (flags[i] & ACTIVE) == 0:
        return
    s = c_shape[tid]
    b = shape_body[s]
    X = wp.transform_identity()
    com = wp.vec3(0.0)
    vs = wp.spatial_vector()
    if b >= 0:
        X = body_q[b]
        com = body_com[b]
        vs = body_qd[b]
    bx = wp.transform_point(X, c_bpos[tid])
    n = c_normal[tid]
    c = wp.dot(n, x[i] - bx) - radius[i]
    if c > 0.0:
        return
    bv = wp.spatial_top(vs) + wp.cross(wp.spatial_bottom(vs), bx - wp.transform_point(X, com))
    vr = v[i] - bv
    ke = rate / inv_mass[i]
    kf = frate / inv_mass[i]
    fn = _normal_force(ke, -c, wp.dot(n, vr), inv_mass[i], zeta, dt, 1.0 + nc[i])
    vt = vr - n * wp.dot(n, vr)
    vsn = wp.length(vt)
    F = n * fn
    if vsn > 1.0e-12:
        F = F - vt / vsn * wp.min(kf * vsn, 0.5 * (p_mu[i] + shape_mu[s]) * fn)
    wp.atomic_add(f, i, F)
    wp.atomic_add(rec, p_world[i], shape_role[s], F)


@wp.kernel
def _gripper_pose(path: wp.array2d[wp.transform], clock: wp.array[int], substeps: int, fps: float,
                  gb: wp.array[int], body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]):
    """The gripper at the current substep: poses per frame, interpolated within it (nlerp)."""
    w = wp.tid()
    s = clock[0]
    f = s // substeps
    a = float(s - f * substeps) / float(substeps)
    nf = path.shape[0] - 1
    f0 = wp.min(f, nf)
    f1 = wp.min(f + 1, nf)
    p0 = wp.transform_get_translation(path[f0, w])
    p1 = wp.transform_get_translation(path[f1, w])
    q0 = wp.transform_get_rotation(path[f0, w])
    q1 = wp.transform_get_rotation(path[f1, w])
    if wp.dot(q0, q1) < 0.0:
        q1 = -q1
    q = wp.normalize(q0 * (1.0 - a) + q1 * a)
    dq = q1 * wp.quat_inverse(q0)
    om = 2.0 * wp.vec3(dq[0], dq[1], dq[2]) * fps
    body_q[gb[w]] = wp.transform(p0 * (1.0 - a) + p1 * a, q)
    body_qd[gb[w]] = wp.spatial_vector((p1 - p0) * fps, om)


@wp.kernel
def _attach(att_p: wp.array[int], att_w: wp.array[int], att_local: wp.array[wp.vec3], att_on: wp.array[int],
            gb: wp.array[int], body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
            x: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    """Grasped vertices follow the gripper rigidly."""
    k = wp.tid()
    if att_on[k] == 0:
        return
    i = att_p[k]
    b = gb[att_w[k]]
    X = body_q[b]
    x[i] = wp.transform_point(X, att_local[k])
    vs = body_qd[b]
    v[i] = wp.spatial_top(vs) + wp.cross(wp.spatial_bottom(vs), x[i] - wp.transform_get_translation(X))


@wp.kernel
def _follow(rp_p: wp.array[int], rp_on: wp.array[int], rp_path: wp.array2d[wp.vec3], clock: wp.array[int],
            substeps: int, fps: float, x: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    """Replayed vertices (a real recording's held patch) follow their recorded positions, interpolated within the
    frame (lsim.real.replay)."""
    k = wp.tid()
    if rp_on[k] == 0:
        return
    s = clock[0]
    f = s // substeps
    a = float(s - f * substeps) / float(substeps)
    nf = rp_path.shape[0] - 1
    f0 = wp.min(f, nf)
    f1 = wp.min(f + 1, nf)
    i = rp_p[k]
    x[i] = rp_path[f0, k] * (1.0 - a) + rp_path[f1, k] * a
    v[i] = (rp_path[f1, k] - rp_path[f0, k]) * fps


@wp.kernel
def _grasp_force(att_p: wp.array[int], att_w: wp.array[int], att_on: wp.array[int], f: wp.array[wp.vec3],
                 rec: wp.array[wp.vec3]):
    """Minus the cloth's force on the grasped vertices, per world (their own m (a - g) is added per frame)."""
    k = wp.tid()
    if att_on[k] == 0:
        return
    wp.atomic_add(rec, att_w[k], -f[att_p[k]])


@wp.kernel
def _damp(v: wp.array[wp.vec3], factor: float):
    i = wp.tid()
    v[i] = v[i] * factor


@wp.kernel
def _tick(clock: wp.array[int]):
    clock[0] = clock[0] + 1


# ---------------------------------------------------------------------------------------------------------
# the teacher
# ---------------------------------------------------------------------------------------------------------
class ClothTeacher:
    """Step a batch of cloth worlds. Built from a finalized-able builder and per-world bookkeeping:

      worlds: list of dicts with "particles" (first, count), "edges" (first, count), "cloth" (Cloth),
              "gripper" (its body index)
      shape_role: per shape, ROLE_TABLE / ROLE_GRIPPER / ROLE_OBSTACLE
      path: (n_frames + 1, W, 7) gripper poses per frame (x, y, z, qx, qy, qz, qw) for the whole timeline

    Each frame: grasp/release events are applied between frames (`grasp`, `release`), then the frame is
    simulated in captured graphs of COLLIDE_EVERY substeps, and `frame()` returns what was recorded."""

    MAX_ATTACH = 24                                    # pinned vertices per world

    def __init__(self, builder, worlds, shape_role, path, substeps, self_contact=True, settle_damping=0.0):
        """settle_damping (1/s): a linear velocity damping the closed-form checks use to reach equilibrium
        sooner (scripts/l0_cloth_teacher_check.py); 0 in every dataset."""
        model = builder.finalize()
        self.settle_damping = float(settle_damping)
        self.model, self.worlds, self.W = model, worlds, len(worlds)
        self.substeps, self.dt = int(substeps), 1.0 / (FPS * substeps)
        assert self.substeps % COLLIDE_EVERY == 0
        x0 = model.particle_q.numpy()
        if model.edge_count:                           # bending per edge from each cloth's B
            Bs = np.zeros(model.edge_count)
            for wd in worlds:
                e0, ne = wd["edges"]
                Bs[e0:e0 + ne] = wd["cloth"].B
            props = model.edge_bending_properties.numpy()
            props[:, 0] = bending_ke(model.edge_indices.numpy(), x0, Bs)
            props[:, 1] = 0.0
            model.edge_bending_properties = wp.array(props, dtype=wp.float32)
        pmu = np.zeros(model.particle_count, np.float32)
        for wd in worlds:
            p0, n = wd["particles"]
            pmu[p0:p0 + n] = wd["cloth"].mu
        self.p_mu = wp.array(pmu, dtype=wp.float32)
        self.shape_role = wp.array(np.asarray(shape_role, np.int32), dtype=int)
        model.particle_grid = None                     # no particle-particle or all-pairs triangle contact
        self.solver = newton.solvers.SolverSemiImplicit(model, enable_tri_contact=False, deterministic=DETERMINISTIC)
        self.s0, self.s1, self.ctrl = model.state(), model.state(), model.control()
        n_soft = int(model.particle_count * 4)
        self.pipe = newton.CollisionPipeline(model, soft_contact_gap=0.005, soft_contact_max=n_soft)
        self.con = self.pipe.contacts()
        self.gb = wp.array(np.array([wd["gripper"] for wd in worlds], np.int32), dtype=int)
        self.path = wp.array(np.asarray(path, np.float32), dtype=wp.transform)
        self.n_path = path.shape[0]
        self.clock = wp.zeros(1, dtype=int)
        # grasp attachments: fixed-size arrays, switched on and off between frames
        A = self.MAX_ATTACH * self.W
        self.att_p = wp.zeros(A, dtype=int)
        self.att_w = wp.array(np.repeat(np.arange(self.W), self.MAX_ATTACH).astype(np.int32), dtype=int)
        self.att_local = wp.zeros(A, dtype=wp.vec3)
        self.att_on = wp.zeros(A, dtype=int)
        self._att = {w: [] for w in range(self.W)}
        # records, summed over a frame's substeps
        self.rec_contact = wp.zeros((self.W, 3), dtype=wp.vec3)
        self.rec_grasp = wp.zeros(self.W, dtype=wp.vec3)
        self.flags = model.particle_flags.numpy()
        self.nc = wp.zeros(model.particle_count, dtype=float)    # active self-contacts per vertex, this substep
        self.mass = model.particle_mass.numpy()
        self.pworld = model.particle_world.numpy()
        self.self_contact = self_contact
        if self_contact:                               # one BVH over every cloth, worlds set apart along y
            self.sc_x = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._sc_offsets()
            self.sc_mesh = wp.Mesh(points=self.sc_x, indices=model.tri_indices.flatten())
            self.sc_cand = wp.zeros((model.particle_count, 64), dtype=int)
            self.sc_count = wp.zeros(model.particle_count, dtype=int)
            edges, tri_edge = mesh_edges(model.tri_indices.numpy())
            self.ee = wp.array(edges, dtype=int)
            self.ee_tri = wp.array(tri_edge, dtype=int)
            self.ee_cand = wp.zeros((len(edges), 96), dtype=int)
            self.ee_count = wp.zeros(len(edges), dtype=int)
        self.graph = None
        self.frame_index = 0
        self.rp_p = None                               # replayed vertices (follow): none in generated data

    # -- between frames ---------------------------------------------------------------------------------
    def grasp(self, w, tip, radius=0.012, below=0.01, reach=0.02):
        """Pin world w's cloth vertices under the fingertips -- within `radius` horizontally of `tip` (world)
        and at most `below` under it -- to the gripper's current pose; with fewer than 3 there, the nearest
        cloth within `reach` of the fingertips (a grasp that just missed by a few millimetres in one run
        and not in another split replays of the same episode by 600 mm). Returns their indices in the cloth
        (empty: the grasp missed)."""
        p0, n = self.worlds[w]["particles"]
        x = self.s0.particle_q.numpy()[p0:p0 + n]
        tip = np.asarray(tip, float)
        d = np.linalg.norm(x[:, :2] - tip[:2], axis=1)
        ids = np.flatnonzero((d < radius) & (x[:, 2] > tip[2] - below) & (x[:, 2] < tip[2] + 0.01))
        if len(ids) < 3:
            d3 = np.linalg.norm(x - tip, axis=1)
            near = np.argsort(d3)[:3]
            ids = np.union1d(ids, near[(d3[near] < reach) & (x[near, 2] < tip[2] + 0.01)]).astype(int)
        if len(ids) == 0:
            return ids
        return self.grasp_ids(w, ids[np.argsort(d[ids])][:self.MAX_ATTACH])

    def grasp_ids(self, w, ids):
        """Pin these vertices of world w's cloth (indices in the cloth) to the gripper where they are now
        (a replay pins the recorded ones: which cloth the gripper holds is part of the given action)."""
        p0, n = self.worlds[w]["particles"]
        ids = np.asarray(ids, int)[:self.MAX_ATTACH] + p0
        X = self.path.numpy()[min(self.frame_index, self.n_path - 1), w]
        local = (_quat_matrix(X[3:]).T @ (self.s0.particle_q.numpy()[ids] - X[:3]).T).T
        self._att[w] = list(ids)
        k0 = w * self.MAX_ATTACH
        on, ps, ls = self.att_on.numpy(), self.att_p.numpy(), self.att_local.numpy()
        on[k0:k0 + self.MAX_ATTACH] = 0
        on[k0:k0 + len(ids)] = 1
        ps[k0:k0 + len(ids)] = ids
        ls[k0:k0 + len(ids)] = local
        self.att_on.assign(on)
        self.att_p.assign(ps)
        self.att_local.assign(ls)
        self.flags[ids] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(self.flags)
        return ids - p0

    def release(self, w):
        ids = self._att.get(w, [])
        if not len(ids):
            return
        k0 = w * self.MAX_ATTACH
        on = self.att_on.numpy()
        on[k0:k0 + self.MAX_ATTACH] = 0
        self.att_on.assign(on)
        self.flags[ids] |= int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(self.flags)
        self._att[w] = []

    def grasped(self, w):
        return list(self._att.get(w, []))

    def follow(self, ids, path):
        """Replay a real recording: particles `ids` (global indices) follow `path` (n_frames + 1, len(ids), 3), their
        recorded positions per frame from the current one on, interpolated within each frame, while switched on
        (follow_on). Set up before the first frame: the captured graph includes it."""
        assert self.graph is None, "follow() must come before the first frame"
        self.rp_ids = np.asarray(ids, np.int64)
        self.rp_p = wp.array(self.rp_ids.astype(np.int32), dtype=int)
        self.rp_on = wp.zeros(len(self.rp_ids), dtype=int)
        self.rp_path = wp.array(np.asarray(path, np.float32), dtype=wp.vec3)

    def follow_on(self, on):
        """Which of the replayed particles follow their path for the coming frame (the rest move freely)."""
        on = np.asarray(on, bool)
        self.rp_on.assign(on.astype(np.int32))
        self.flags[self.rp_ids[on]] &= ~int(newton.ParticleFlags.ACTIVE)
        self.flags[self.rp_ids[~on]] |= int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(self.flags)

    def _follow_launch(self):
        if self.rp_p is not None:
            wp.launch(_follow, dim=len(self.rp_ids), inputs=[self.rp_p, self.rp_on, self.rp_path, self.clock, self.substeps,
                                                            float(FPS)], outputs=[self.s0.particle_q, self.s0.particle_qd])

    # -- stepping -----------------------------------------------------------------------------------------
    def _sc_offsets(self):
        wp.launch(_offset_points, dim=self.model.particle_count,
                  inputs=[self.s0.particle_q, self.model.particle_world, 10.0], outputs=[self.sc_x])

    def _refit(self):
        self._sc_offsets()
        self.sc_mesh.refit()
        wp.launch(_self_candidates, dim=self.model.particle_count,
                  inputs=[self.sc_mesh.id, self.sc_x, self.model.tri_indices, SELF_THICK + SELF_MARGIN],
                  outputs=[self.sc_cand, self.sc_count])
        if SELF_EDGES:
            wp.launch(_edge_candidates, dim=self.ee.shape[0],
                      inputs=[self.sc_mesh.id, self.sc_x, self.ee_tri, self.ee, SELF_THICK + SELF_MARGIN],
                      outputs=[self.ee_cand, self.ee_count])

    def candidate_load(self):
        """The fullest self-contact candidate lists (vertex-triangle of 64 slots, edge-edge of 96)."""
        return int(self.sc_count.numpy().max()), int(self.ee_count.numpy().max()) if SELF_EDGES else 0

    def _substeps(self, k):
        m = self.model
        for i in range(k):
            if i == 0:
                self.pipe.collide(self.s0, self.con)
            self.s0.clear_forces()
            wp.launch(_gripper_pose, dim=self.W, inputs=[self.path, self.clock, self.substeps, float(FPS), self.gb],
                      outputs=[self.s0.body_q, self.s0.body_qd])
            wp.launch(_attach, dim=len(self.att_on), inputs=[self.att_p, self.att_w, self.att_local, self.att_on, self.gb,
                                                           self.s0.body_q, self.s0.body_qd],
                      outputs=[self.s0.particle_q, self.s0.particle_qd])
            self._follow_launch()
            wp.launch(_membrane_damping, dim=m.tri_count, inputs=[self.s0.particle_q, self.s0.particle_qd, m.tri_indices,
                                                                  m.tri_poses, MEMBRANE_KD], outputs=[self.s0.particle_f])
            self.nc.zero_()
            if self.self_contact:
                wp.launch(_count_vt, dim=m.particle_count, inputs=[self.s0.particle_q, m.tri_indices, self.sc_cand,
                                                                   self.sc_count, SELF_THICK], outputs=[self.nc])
                if SELF_EDGES:
                    wp.launch(_count_ee, dim=self.ee.shape[0], inputs=[self.s0.particle_q, self.ee, self.ee_cand,
                                                                       self.ee_count, SELF_THICK], outputs=[self.nc])
                wp.launch(_self_contact, dim=m.particle_count,
                          inputs=[self.s0.particle_q, self.s0.particle_qd, m.tri_indices, self.sc_cand, self.sc_count,
                                  SELF_THICK, SELF_RATE, FRICTION_RATE, SELF_MU, m.particle_inv_mass, CONTACT_ZETA, self.dt,
                                  self.nc], outputs=[self.s0.particle_f])
                if SELF_EDGES:
                    wp.launch(_edge_contact, dim=self.ee.shape[0],
                              inputs=[self.s0.particle_q, self.s0.particle_qd, self.ee, self.ee_cand, self.ee_count,
                                      SELF_THICK, SELF_RATE, FRICTION_RATE, SELF_MU, m.particle_inv_mass, CONTACT_ZETA,
                                      self.dt, self.nc], outputs=[self.s0.particle_f])
            c = self.con
            wp.launch(_shape_contact, dim=c.soft_contact_max,
                      inputs=[c.soft_contact_count, c.soft_contact_max, c.soft_contact_particle, c.soft_contact_shape,
                              c.soft_contact_body_pos, c.soft_contact_normal, m.shape_body, m.shape_material_mu,
                              self.shape_role, self.s0.body_q, self.s0.body_qd, m.body_com, self.s0.particle_q,
                              self.s0.particle_qd, m.particle_radius, m.particle_flags, self.p_mu, m.particle_world,
                              CONTACT_RATE, FRICTION_RATE, m.particle_inv_mass, CONTACT_ZETA, self.dt, self.nc],
                      outputs=[self.s0.particle_f, self.rec_contact])
            self.solver.step(self.s0, self.s1, self.ctrl, None, self.dt)
            if self.settle_damping > 0.0:
                wp.launch(_damp, dim=m.particle_count, inputs=[self.s1.particle_qd, float(np.exp(-self.settle_damping * self.dt))])
            wp.launch(_grasp_force, dim=len(self.att_on), inputs=[self.att_p, self.att_w, self.att_on, self.s0.particle_f],
                      outputs=[self.rec_grasp])
            self.s0, self.s1 = self.s1, self.s0
            wp.launch(_tick, dim=1, inputs=[self.clock])

    def frame(self):
        """Simulate one frame; returns the frame's record (cloth vertices at its end, the gripper pose,
        mean contact forces on the cloth by surface kind, mean grasp force before its m (a - g))."""
        self.rec_contact.zero_()
        self.rec_grasp.zero_()
        n_chunks = self.substeps // COLLIDE_EVERY
        done = 0
        sub = getattr(self, "sub", 0)                  # sub > 1: also the record at sub - 1 instants inside the frame
        assert sub <= 1 or n_chunks % sub == 0, (self.substeps, sub)   # at equal times (frame-rate study)
        marks = {n_chunks * j // sub for j in range(1, sub)} if sub > 1 else set()
        x_sub, cum = [], []

        def mark():                                    # the held vertices put on the gripper as at a frame's end
            wp.launch(_gripper_pose, dim=self.W, inputs=[self.path, self.clock, self.substeps, float(FPS), self.gb],
                      outputs=[self.s0.body_q, self.s0.body_qd])
            wp.launch(_attach, dim=len(self.att_on), inputs=[self.att_p, self.att_w, self.att_local, self.att_on, self.gb,
                                                           self.s0.body_q, self.s0.body_qd],
                      outputs=[self.s0.particle_q, self.s0.particle_qd])
            self._follow_launch()
            x_sub.append(self.s0.particle_q.numpy().copy())
            cum.append((self.rec_contact.numpy().copy(), self.rec_grasp.numpy().copy()))
        if self.graph is None:
            if self.self_contact:
                self._refit()
            self._substeps(COLLIDE_EVERY)              # uncaptured: compiles every kernel
            done = 1
            if 1 in marks:
                mark()
            with wp.ScopedCapture() as cap:            # recorded, not run
                self._substeps(COLLIDE_EVERY)
            self.graph = cap.graph
        for c in range(done, n_chunks):
            if self.self_contact:
                self._refit()
            wp.capture_launch(self.graph)
            if c + 1 in marks:
                mark()
        self.x_sub = x_sub
        if sub > 1:                                    # mean contact / grasp forces over each of the sub intervals
            cum = [(0.0, 0.0)] + cum + [(self.rec_contact.numpy(), self.rec_grasp.numpy())]
            per = self.substeps / sub
            self.contact_sub = np.stack([(cum[k + 1][0] - cum[k][0]) / per for k in range(sub)])
            self.grasp_sub = np.stack([(cum[k + 1][1] - cum[k][1]) / per for k in range(sub)])
        self.frame_index += 1
        wp.launch(_gripper_pose, dim=self.W, inputs=[self.path, self.clock, self.substeps, float(FPS), self.gb],
                  outputs=[self.s0.body_q, self.s0.body_qd])
        wp.launch(_attach, dim=len(self.att_on), inputs=[self.att_p, self.att_w, self.att_local, self.att_on, self.gb,
                                                       self.s0.body_q, self.s0.body_qd],
                  outputs=[self.s0.particle_q, self.s0.particle_qd])
        self._follow_launch()
        return {"x": self.s0.particle_q.numpy(), "contact": self.rec_contact.numpy() / self.substeps,
                "grasp": self.rec_grasp.numpy() / self.substeps}


def _quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
