"""The rope teacher: the rope teacher gate's passing rope ("semi", scripts/l0_rope_gate.py, DIRECTION.md "Rope
teacher gate") batched over parallel worlds, with the cloth teacher's kinematic gripper.

A rope is a chain of particles, one per node (mass lam x half of each adjacent segment), integrated by Newton's
explicit SolverSemiImplicit. On top of Newton (all of it measured by the gate, 12 of 12 checks):

  stretch    a Newton spring per segment, ke = EA / l, axial damping AXIAL_RATE lam l (N s / m: per unit mass,
             = the gate's AXIAL_ETA / l at its 0.02 kg/m rope; stable for any lam)
  bending    ours (`_bend`): E = (EI / l_dual) (1 - t0 . t1) at every interior node (= EI theta^2 / 2 l_dual), the
             rest shape straight
  self       ours (`_seg_contact`): segment-segment capsules of radius r with the cloth teacher's per-mass law
             (critically damped 2e7 / s^2 penalty, viscous friction 2e4 / s below mu N, mu SELF_MU), candidates by
             brute force within each world once per contact refresh (`_seg_candidates`)
  surface    the cloth teacher's contact and friction (`lsim.teacher.cloth_teacher._shape_contact`): table, plank,
             floor, scanned objects and the gripper's fingers; mu the mean of the rope's and the surface's; the force
             on the rope recorded by surface kind
  air        ours (`_air_drag`), new since the gate: quadratic drag on each segment's normal velocity, the cross-flow
             drag of a cylinder, F = -1/2 rho Cd (2 r) l |v_n| v_n, rho Cd = RHO_CD (air 1.2 kg/m^3 x Cd 1.2); the
             axial skin friction (~1/50 of it) is left out. The gate's settling checks used uniform damping instead
             (1-4 / s); here that damping is `settle_damping`, used by the closed-form checks only, 0 in every dataset

The step: dt = sqrt(m / k) / STEPS_PER_PERIOD, k / m the stiffest of stretch EA / l, bending 16 EI / l^3 and the
contact rate per unit mass, with m = lam l (the gate's convention; stable from 1.2 steps per sqrt(m / k), run at
2.0), at least MIN_SUBSTEPS (the viscous friction's own limit, 2e4 dt < 2) and a multiple of the contact refresh.
A batch runs at the step of its stiffest rope: batch ropes by k / m (`k_over_m`).

The gripper is the cloth teacher's (`_gripper_pose`, `_attach`, `_grasp_force`; grasp / release reused from
`ClothTeacher`): a kinematic body following a pose per frame, its finger boxes pushing the rope, a grasp pinning the
rope nodes under the fingertips to it; the force it applies through a grasp is recorded the cloth teacher's way.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp
import newton

from lsim.teacher.cloth_teacher import (COLLIDE_EVERY, CONTACT_RATE, CONTACT_ZETA, FPS, FRICTION_RATE, G,  # noqa: F401
                                        ROLE_GRIPPER, ROLE_OBSTACLE, ROLE_TABLE, ClothTeacher, _attach, _damp,
                                        _grasp_force, _gripper_pose, _normal_force, _shape_contact, _tick)

AXIAL_ETA = 1.0e-3        # N s: the gate's axial (strain-rate) damping, kd = eta / l per spring (its 0.02 kg/m rope)
AXIAL_RATE = 500.0        # 1/s: the generator's, per unit mass, kd = AXIAL_RATE x lam x l -- the gate's value at its
                          # rope. A fixed 1e-3 N s is 1.3e5 / s on the end node of a 0.16 g/m thread: explicit damping
                          # needs rate x dt < 2, and the pilot's first such thread exploded when grasped (500 m/s)
SELF_MU = 0.5             # rope-on-rope friction (the gate's)
RHO_CD = 1.44             # kg/m^3: air density 1.2 x a cylinder's cross-flow drag coefficient 1.2 (Re 1e2-1e5)
STEPS_PER_PERIOD = 2.0    # explicit steps per sqrt(m / k) (stable from 1.2: the gate's stability sweep)
MIN_SUBSTEPS = 1000       # per frame: viscous friction 2e4 / s x dt = 0.67 (stable below 2), the cloth teacher's floor
SELF_QUERY = 0.01         # m beyond contact (2 r) for self-contact candidates: what a node may move in one refresh
SELF_CAND = 48            # candidate slots per segment (the gate's 24 filled up in tight piles of thin string)
EA_MAX = 1.0e4            # N: the stiffest rope given to the teacher (lsim.data.rope also caps the wave speed). PENDING (DIRECTION.md "Rope
                          # teacher gate": above ~1e4 N a rope's own-weight strain is < 2e-5 and the step shrinks as
                          # 1 / sqrt(EA)); every rope prior reads it from here


@dataclass
class Rope:
    """A rope in physical units: n segments of rest length l (m), linear density lam (kg/m), axial stiffness EA (N),
    bending stiffness EI (N m^2), friction mu, radius r (m)."""
    n: int
    l: float = 0.01
    lam: float = 0.02
    EA: float = 1.0e3
    EI: float = 1.0e-4
    mu: float = 0.5
    r: float = 0.004

    @property
    def length(self):
        return self.n * self.l

    @property
    def bending_length(self):
        return (self.EI / (self.lam * G)) ** (1.0 / 3.0)

    @property
    def node_mass(self):
        return self.lam * self.l

    @property
    def k_over_m(self):
        """The stiffest stiffness per unit mass of the explicit step (1/s^2): stretch, bending's zigzag mode, contact."""
        m = self.node_mass
        return max(self.EA / self.l, 16.0 * self.EI / self.l ** 3, CONTACT_RATE * m) / m


def stable_substeps(ropes, fps=FPS, steps_per_period=STEPS_PER_PERIOD, ce=COLLIDE_EVERY, floor=MIN_SUBSTEPS):
    """Substeps per frame for a batch: dt = sqrt(m / k) / steps_per_period for its stiffest rope, at least `floor`,
    a multiple of the contact refresh."""
    w = max(np.sqrt(r.k_over_m) for r in ropes)
    n = int(np.ceil(steps_per_period * w / fps / ce)) * ce
    return int(max(n, int(np.ceil(floor / ce)) * ce, ce))


def straight(p0, d, L, l):
    """Nodes of a straight rope from p0 along d, length L, segments of l."""
    n = int(round(L / l))
    d = np.asarray(d, float) / np.linalg.norm(d)
    return np.asarray(p0, float)[None] + np.linspace(0.0, L, n + 1)[:, None] * d[None]


def chord_resample(P, l, n):
    """n + 1 nodes along the dense polyline P, consecutive nodes exactly l apart (the springs' rest length is the
    distance they are built at); None if the polyline is too short."""
    P = np.asarray(P, float)
    out = [P[0]]
    j = 0
    cur = P[0]
    while len(out) < n + 1:
        while j + 1 < len(P) and np.linalg.norm(P[j + 1] - cur) < l:
            j += 1
        if j + 1 >= len(P):
            return None
        a, b = P[j], P[j + 1]                          # |a + s (b - a) - cur| = l, the root with s in [0, 1]
        d, w = b - a, a - cur
        A, B, C = d @ d, 2.0 * d @ w, w @ w - l * l
        s = (-B + np.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A)
        cur = a + np.clip(s, 0.0, 1.0) * d
        out.append(cur)
        P = np.concatenate([[cur], P[j + 1:]])
        j = 0
    return np.array(out)


def add_rope(b, rope, P):
    """Add the rope's nodes P ((n + 1, 3), world) to builder b: particles with the chain's lumped masses and a spring
    per segment. Returns (first particle, count)."""
    P = np.asarray(P, float)
    seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
    m = np.zeros(len(P))
    m[:-1] += 0.5 * rope.lam * seglen
    m[1:] += 0.5 * rope.lam * seglen
    p0 = b.particle_count
    for k, p in enumerate(P):
        b.add_particle(wp.vec3(*p), wp.vec3(0.0), float(m[k]), radius=rope.r)
    for k in range(len(P) - 1):
        b.add_spring(p0 + k, p0 + k + 1, rope.EA / seglen[k], AXIAL_RATE * rope.lam * seglen[k], 0)
    return p0, len(P)


# ---------------------------------------------------------------------------------------------------------
# kernels (the gate's _bend, _seg_candidates, _seg_contact moved here unchanged; scripts/l0_rope_gate.py imports them)
# ---------------------------------------------------------------------------------------------------------
@wp.kernel
def _bend(x: wp.array[wp.vec3], tri: wp.array2d[int], k: wp.array[float], f: wp.array[wp.vec3]):
    """E = k (1 - t0 . t1) at a node b between a and c (k = EI / l_dual): EI theta^2 / (2 l_dual) for small theta."""
    i = wp.tid()
    a = tri[i, 0]
    b = tri[i, 1]
    c = tri[i, 2]
    e0 = x[b] - x[a]
    e1 = x[c] - x[b]
    l0 = wp.length(e0)
    l1 = wp.length(e1)
    if l0 < 1.0e-9 or l1 < 1.0e-9:
        return
    t0 = e0 / l0
    t1 = e1 / l1
    cs = wp.dot(t0, t1)
    g0 = (t1 - cs * t0) / l0                           # d(t0 . t1) / d e0
    g1 = (t0 - cs * t1) / l1                           # d(t0 . t1) / d e1
    fa = -k[i] * g0
    fc = k[i] * g1
    wp.atomic_add(f, a, fa)
    wp.atomic_add(f, c, fc)
    wp.atomic_add(f, b, -(fa + fc))


@wp.kernel
def _seg_candidates(x: wp.array[wp.vec3], seg: wp.array2d[int], srope: wp.array[int], sidx: wp.array[int],
                    sworld: wp.array[int], wstart: wp.array[int], wcount: wp.array[int], query: float,
                    on: wp.array[int], cand: wp.array2d[int], count: wp.array[int]):
    """Once per contact refresh: segments j > i of the same world within the query distance (brute force), not
    the segment's neighbours on its own rope, both ropes with contact on."""
    i = wp.tid()
    k = int(0)
    a = seg[i, 0]
    b = seg[i, 1]
    w = sworld[i]
    if on[srope[i]] != 0:
        for jj in range(wcount[w]):
            j = wstart[w] + jj
            if j > i and on[srope[j]] != 0:
                if srope[j] != srope[i] or wp.abs(sidx[j] - sidx[i]) > 1:
                    st = wp.closest_point_edge_edge(x[a], x[b], x[seg[j, 0]], x[seg[j, 1]], 1.0e-12)
                    if st[2] < query and k < cand.shape[1]:
                        cand[i, k] = j
                        k += 1
    count[i] = k


@wp.kernel
def _seg_contact(x: wp.array[wp.vec3], v: wp.array[wp.vec3], seg: wp.array2d[int], cand: wp.array2d[int],
                 count: wp.array[int], thick: float, rate: float, frate: float, mu: float, inv_mass: wp.array[float],
                 zeta: float, dt: float, f: wp.array[wp.vec3]):
    """Two capsules of radius thick / 2 closer than thick are pushed apart along their closest points' separation,
    with the cloth teacher's per-mass law (critically damped penalty, viscous friction below mu N)."""
    i = wp.tid()
    a = seg[i, 0]
    b = seg[i, 1]
    for k in range(count[i]):
        j = cand[i, k]
        c = seg[j, 0]
        d = seg[j, 1]
        st = wp.closest_point_edge_edge(x[a], x[b], x[c], x[d], 1.0e-12)
        s = st[0]
        t = st[1]
        dist = st[2]
        if dist < thick and dist > 1.0e-9:
            n = (x[a] + s * (x[b] - x[a]) - x[c] - t * (x[d] - x[c])) / dist
            vr = (1.0 - s) * v[a] + s * v[b] - (1.0 - t) * v[c] - t * v[d]
            im = (1.0 - s) * (1.0 - s) * inv_mass[a] + s * s * inv_mass[b] + (1.0 - t) * (1.0 - t) * inv_mass[c] + t * t * inv_mass[d]
            if im > 0.0:
                ke = rate / im
                kf = frate / im
                fn = _normal_force(ke, thick - dist, wp.dot(n, vr), im, zeta, dt, 1.0)
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
def _seg_contact_rope(x: wp.array[wp.vec3], v: wp.array[wp.vec3], seg: wp.array2d[int], cand: wp.array2d[int],
                      count: wp.array[int], srad: wp.array[float], rate: float, frate: float, mu: float,
                      inv_mass: wp.array[float], zeta: float, dt: float, f: wp.array[wp.vec3]):
    """`_seg_contact` with each world's own rope radius (thickness 2 r of the segment's rope)."""
    i = wp.tid()
    a = seg[i, 0]
    b = seg[i, 1]
    thick = 2.0 * srad[i]
    for k in range(count[i]):
        j = cand[i, k]
        c = seg[j, 0]
        d = seg[j, 1]
        st = wp.closest_point_edge_edge(x[a], x[b], x[c], x[d], 1.0e-12)
        s = st[0]
        t = st[1]
        dist = st[2]
        if dist < thick and dist > 1.0e-9:
            n = (x[a] + s * (x[b] - x[a]) - x[c] - t * (x[d] - x[c])) / dist
            vr = (1.0 - s) * v[a] + s * v[b] - (1.0 - t) * v[c] - t * v[d]
            im = (1.0 - s) * (1.0 - s) * inv_mass[a] + s * s * inv_mass[b] + (1.0 - t) * (1.0 - t) * inv_mass[c] + t * t * inv_mass[d]
            if im > 0.0:
                ke = rate / im
                kf = frate / im
                fn = _normal_force(ke, thick - dist, wp.dot(n, vr), im, zeta, dt, 1.0)
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
def _seg_candidates_rope(x: wp.array[wp.vec3], seg: wp.array2d[int], sidx: wp.array[int], sworld: wp.array[int],
                         wstart: wp.array[int], wcount: wp.array[int], srad: wp.array[float], margin: float,
                         cand: wp.array2d[int], count: wp.array[int]):
    """`_seg_candidates` for one rope per world, the query 2 r + margin of the world's rope."""
    i = wp.tid()
    k = int(0)
    a = seg[i, 0]
    b = seg[i, 1]
    w = sworld[i]
    query = 2.0 * srad[i] + margin
    for jj in range(wcount[w]):
        j = wstart[w] + jj
        if j > i and wp.abs(sidx[j] - sidx[i]) > 1:
            st = wp.closest_point_edge_edge(x[a], x[b], x[seg[j, 0]], x[seg[j, 1]], 1.0e-12)
            if st[2] < query and k < cand.shape[1]:
                cand[i, k] = j
                k += 1
    count[i] = k


@wp.kernel
def _air_drag(x: wp.array[wp.vec3], v: wp.array[wp.vec3], seg: wp.array2d[int], srad: wp.array[float], rho_cd: float,
              f: wp.array[wp.vec3]):
    """Cross-flow drag of a cylinder on each segment's normal velocity, -1/2 rho Cd (2 r) l |v_n| v_n, half to each end."""
    i = wp.tid()
    a = seg[i, 0]
    b = seg[i, 1]
    e = x[b] - x[a]
    L = wp.length(e)
    if L < 1.0e-9:
        return
    t = e / L
    vm = 0.5 * (v[a] + v[b])
    vn = vm - t * wp.dot(t, vm)
    F = -0.5 * rho_cd * 2.0 * srad[i] * L * wp.length(vn) * vn
    wp.atomic_add(f, a, 0.5 * F)
    wp.atomic_add(f, b, 0.5 * F)


# ---------------------------------------------------------------------------------------------------------
# crossings (the gate's counter, moved here: the generator's quality record uses it)
# ---------------------------------------------------------------------------------------------------------
def seg_pairs(nodes_per_rope):
    """Global segment list (node index pairs into the concatenated nodes) and the non-adjacent pairs."""
    segs, rid, idx, off = [], [], [], 0
    for ri, n in enumerate(nodes_per_rope):
        for k in range(n - 1):
            segs.append((off + k, off + k + 1))
            rid.append(ri)
            idx.append(k)
        off += n
    segs, rid, idx = np.array(segs), np.array(rid), np.array(idx)
    I, J = np.triu_indices(len(segs), 1)
    keep = (rid[I] != rid[J]) | (np.abs(idx[I] - idx[J]) > 1)
    return segs, I[keep], J[keep]


def seg_geometry(X, segs, I, J):
    a, b = X[segs[I, 0]], X[segs[I, 1]]
    c, d = X[segs[J, 0]], X[segs[J, 1]]
    u, v, w = b - a, d - c, a - c
    A, B, C = (u * u).sum(1), (u * v).sum(1), (v * v).sum(1)
    D, E = (u * w).sum(1), (v * w).sum(1)
    den = A * C - B * B
    ok = den > 1e-18
    s = np.where(ok, (B * E - C * D) / np.where(ok, den, 1.0), -1.0)     # line-line closest parameters
    t = np.where(ok, (A * E - B * D) / np.where(ok, den, 1.0), -1.0)
    cr = np.cross(u, v)
    sig = (cr * (c - a)).sum(1)                                          # det[u, v, c - a]
    sin = np.linalg.norm(cr, axis=1) / np.sqrt(A * C + 1e-30)
    sc_, tc_ = np.clip(s, 0, 1), np.clip(t, 0, 1)                        # a cheap clamped distance (upper bound
    dist = np.linalg.norm(a + sc_[:, None] * u - c - tc_[:, None] * v, axis=1)   # of the true one; exact inside)
    e = 1e-3                                                             # a crossing at a shared node counts
    inside = (s > -e) & (s < 1 + e) & (t > -e) & (t < 1 + e)
    return dict(sig=sig, sin=sin, dist=dist, inside=inside)


class CrossingCounter:
    """Pairs of non-adjacent segments whose lines' triple product det[b - a, d - c, c - a] changes sign between
    frames while both line-closest points lie inside both segments, within `near` and at more than asin(min_sin)
    to each other at both frames: their centerlines passed through each other."""

    def __init__(self, nodes_per_rope, near=0.05, min_sin=0.2):
        self.segs, self.I, self.J = seg_pairs(nodes_per_rope)
        self.near, self.min_sin = near, min_sin
        self.prev, self.n, self.dmin = None, 0, np.inf

    def add(self, X):
        g = seg_geometry(X, self.segs, self.I, self.J)
        self.dmin = min(self.dmin, float(g["dist"].min())) if len(g["dist"]) else self.dmin
        if self.prev is not None:
            p = self.prev
            flip = np.sign(p["sig"]) * np.sign(g["sig"]) < 0
            ok = p["inside"] & g["inside"] & (p["sin"] > self.min_sin) & (g["sin"] > self.min_sin) & \
                 (p["dist"] < self.near) & (g["dist"] < self.near)
            self.n += int((flip & ok).sum())
        self.prev = g


# ---------------------------------------------------------------------------------------------------------
# the teacher
# ---------------------------------------------------------------------------------------------------------
class RopeTeacher:
    """Step a batch of rope worlds, one rope per world. Built from a builder and per-world bookkeeping:

      worlds: list of dicts with "particles" (first, count: the rope's nodes in chain order), "rope" (Rope),
              "gripper" (its body index)
      shape_role: per shape, ROLE_TABLE / ROLE_GRIPPER / ROLE_OBSTACLE
      path: (n_frames + 1, W, 7) gripper poses per frame (x, y, z, qx, qy, qz, qw) for the whole timeline

    The interface is ClothTeacher's: grasp / grasp_ids / release / grasped between frames, `frame()` returns the
    frame's record. fixed: global particle ids held in place (closed-form checks), drag: air drag on."""

    MAX_ATTACH = 24
    grasp = ClothTeacher.grasp
    grasp_ids = ClothTeacher.grasp_ids
    release = ClothTeacher.release
    grasped = ClothTeacher.grasped

    def __init__(self, builder, worlds, shape_role, path, substeps, self_contact=True, settle_damping=0.0, drag=True,
                 fixed=(), ce=COLLIDE_EVERY):
        """A world may hold several ropes ("chains": [(first particle, count, Rope), ...], the closed-form checks'
        crossing scenes); by default its one rope is (particles, rope). ce: substeps per contact refresh."""
        model = builder.finalize()
        self.model, self.worlds, self.W = model, worlds, len(worlds)
        self.substeps, self.dt, self.ce = int(substeps), 1.0 / (FPS * substeps), int(ce)
        assert self.substeps % self.ce == 0
        self.settle_damping, self.drag, self.self_contact = float(settle_damping), bool(drag), bool(self_contact)
        x0 = model.particle_q.numpy().astype(np.float64)
        seg, sidx, sworld, srad, tri, kb, pmu = [], [], [], [], [], [], np.zeros(model.particle_count, np.float32)
        wstart, wcount = [], []
        for w, wd in enumerate(worlds):
            chains = wd.get("chains") or [(*wd["particles"], wd["rope"])]
            wstart.append(len(seg))
            for ci, (p0, n, r) in enumerate(chains):
                pmu[p0:p0 + n] = r.mu
                for k in range(n - 1):
                    seg.append((p0 + k, p0 + k + 1))
                    sidx.append(k + 100000 * ci)              # another rope's segments are never adjacent
                    sworld.append(w)
                    srad.append(r.r)
                L = np.linalg.norm(np.diff(x0[p0:p0 + n], axis=0), axis=1)
                for k in range(1, n - 1):
                    tri.append((p0 + k - 1, p0 + k, p0 + k + 1))
                    kb.append(r.EI / (0.5 * (L[k - 1] + L[k])))
            wcount.append(len(seg) - wstart[-1])
        self.seg = wp.array(np.asarray(seg, np.int32).reshape(-1, 2), dtype=int)
        self.sidx = wp.array(np.asarray(sidx, np.int32), dtype=int)
        self.sworld = wp.array(np.asarray(sworld, np.int32), dtype=int)
        self.srad = wp.array(np.asarray(srad, np.float32), dtype=float)
        self.wstart = wp.array(np.asarray(wstart, np.int32), dtype=int)
        self.wcount = wp.array(np.asarray(wcount, np.int32), dtype=int)
        self.nseg = len(seg)
        self.cand = wp.zeros((max(self.nseg, 1), SELF_CAND), dtype=int)
        self.count = wp.zeros(max(self.nseg, 1), dtype=int)
        self.bend_tri = wp.array(np.asarray(tri, np.int32).reshape(-1, 3), dtype=int) if tri else None
        self.bend_k = wp.array(np.asarray(kb, np.float32), dtype=float) if tri else None
        self.p_mu = wp.array(pmu, dtype=wp.float32)
        self.shape_role = wp.array(np.asarray(shape_role, np.int32) if len(shape_role) else np.zeros(1, np.int32), dtype=int)
        model.particle_grid = None                     # no particle-particle contact: ours is segment-segment
        self.solver = newton.solvers.SolverSemiImplicit(model)
        self.s0, self.s1, self.ctrl = model.state(), model.state(), model.control()
        self.pipe = newton.CollisionPipeline(model, soft_contact_gap=0.005, soft_contact_max=int(model.particle_count * 4))
        self.con = self.pipe.contacts()
        self.gb = wp.array(np.array([wd["gripper"] for wd in worlds], np.int32), dtype=int)
        self.path = wp.array(np.asarray(path, np.float32), dtype=wp.transform)
        self.n_path = path.shape[0]
        self.clock = wp.zeros(1, dtype=int)
        A = self.MAX_ATTACH * self.W
        self.att_p = wp.zeros(A, dtype=int)
        self.att_w = wp.array(np.repeat(np.arange(self.W), self.MAX_ATTACH).astype(np.int32), dtype=int)
        self.att_local = wp.zeros(A, dtype=wp.vec3)
        self.att_on = wp.zeros(A, dtype=int)
        self._att = {w: [] for w in range(self.W)}
        self.rec_contact = wp.zeros((self.W, 3), dtype=wp.vec3)
        self.rec_grasp = wp.zeros(self.W, dtype=wp.vec3)
        self.flags = model.particle_flags.numpy()
        if len(fixed):
            self.flags[np.asarray(fixed, int)] &= ~int(newton.ParticleFlags.ACTIVE)
            model.particle_flags.assign(self.flags)
        self.nc = wp.zeros(model.particle_count, dtype=float)   # the cloth contact's implicit share: 0 (the gate's)
        self.mass = model.particle_mass.numpy()
        self.pworld = model.particle_world.numpy()
        self.graph = None
        self.frame_index = 0
        self.on_chunk = None                           # callable(teacher) after each captured chunk (diagnostics)

    def candidate_load(self):
        """The fullest self-contact candidate list (of SELF_CAND slots) at the last refresh."""
        return int(self.count.numpy().max()) if self.nseg else 0

    def _substeps(self, k):
        m = self.model
        for i in range(k):
            if i == 0:
                self.pipe.collide(self.s0, self.con)
                if self.self_contact and self.nseg:
                    wp.launch(_seg_candidates_rope, dim=self.nseg,
                              inputs=[self.s0.particle_q, self.seg, self.sidx, self.sworld, self.wstart, self.wcount,
                                      self.srad, SELF_QUERY], outputs=[self.cand, self.count])
            self.s0.clear_forces()
            wp.launch(_gripper_pose, dim=self.W, inputs=[self.path, self.clock, self.substeps, float(FPS), self.gb],
                      outputs=[self.s0.body_q, self.s0.body_qd])
            wp.launch(_attach, dim=len(self.att_on), inputs=[self.att_p, self.att_w, self.att_local, self.att_on, self.gb,
                                                           self.s0.body_q, self.s0.body_qd],
                      outputs=[self.s0.particle_q, self.s0.particle_qd])
            if self.bend_tri is not None:
                wp.launch(_bend, dim=self.bend_tri.shape[0], inputs=[self.s0.particle_q, self.bend_tri, self.bend_k],
                          outputs=[self.s0.particle_f])
            if self.self_contact and self.nseg:
                wp.launch(_seg_contact_rope, dim=self.nseg,
                          inputs=[self.s0.particle_q, self.s0.particle_qd, self.seg, self.cand, self.count, self.srad,
                                  CONTACT_RATE, FRICTION_RATE, SELF_MU, m.particle_inv_mass, CONTACT_ZETA, self.dt],
                          outputs=[self.s0.particle_f])
            if self.drag and RHO_CD > 0.0 and self.nseg:
                wp.launch(_air_drag, dim=self.nseg, inputs=[self.s0.particle_q, self.s0.particle_qd, self.seg, self.srad,
                                                            RHO_CD], outputs=[self.s0.particle_f])
            c = self.con
            if m.shape_count:
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
        """Simulate one frame; returns its record (rope nodes at its end, mean contact forces on the rope by surface
        kind, mean grasp force before the pinned nodes' own m (a - g))."""
        self.rec_contact.zero_()
        self.rec_grasp.zero_()
        n_chunks = self.substeps // self.ce
        done = 0
        if self.graph is None:
            self._substeps(self.ce)                    # uncaptured: compiles every kernel
            done = 1
            with wp.ScopedCapture() as cap:
                self._substeps(self.ce)
            self.graph = cap.graph
        for _ in range(n_chunks - done):
            wp.capture_launch(self.graph)
            if self.on_chunk is not None:              # diagnostics: the state after every contact refresh
                self.on_chunk(self)
        self.frame_index += 1
        wp.launch(_gripper_pose, dim=self.W, inputs=[self.path, self.clock, self.substeps, float(FPS), self.gb],
                  outputs=[self.s0.body_q, self.s0.body_qd])
        wp.launch(_attach, dim=len(self.att_on), inputs=[self.att_p, self.att_w, self.att_local, self.att_on, self.gb,
                                                       self.s0.body_q, self.s0.body_qd],
                  outputs=[self.s0.particle_q, self.s0.particle_qd])
        return {"x": self.s0.particle_q.numpy(), "contact": self.rec_contact.numpy() / self.substeps,
                "grasp": self.rec_grasp.numpy() / self.substeps}


class FineCrossings:
    """Self-crossings counted at contact-refresh resolution (RopeTeacher.on_chunk): the gate's sign-flip test on
    segment pairs, every chunk (substeps / COLLIDE_EVERY per frame, 20-94 a frame) instead of every frame. At 30 fps a
    strand moving a few cm per frame past a segment's end flips the triple product without passing through it (the
    pilot's frame counter flagged 13% of episodes whose strands were never closer than 2 r at any frame). Candidates:
    non-adjacent segment pairs of one world whose nodes come within 2 cm + twice what the world's fastest node moves in
    a frame, refreshed every frame (a KD-tree per world); the per-chunk geometry runs on the GPU (torch, zero-copy)."""

    def __init__(self, worlds, min_sin=0.2, near=0.05):
        import torch
        self.torch = torch
        self.worlds, self.W, self.min_sin, self.near = worlds, len(worlds), min_sin, near
        self.n = np.zeros(self.W, int)
        self.dmin = np.full(self.W, np.inf)
        self.frame, self.prev, self.lastX = -1, None, None
        self.I = None
        self._n_t = None

    def _refresh(self, X, V):
        from scipy.spatial import cKDTree
        torch = self.torch
        A, B, Wd = [], [], []
        for w, wd in enumerate(self.worlds):
            p0, n = wd["particles"]
            x = X[p0:p0 + n].astype(np.float64)
            r = 0.02 + 2.0 * float(np.linalg.norm(V[p0:p0 + n], axis=1).max()) / FPS
            pr = cKDTree(x).query_pairs(min(r, 2.0), output_type="ndarray")
            if not len(pr):
                continue
            segs = []                                  # node pairs -> the segments at each node (k - 1, k)
            for da in (-1, 0):
                for db in (-1, 0):
                    si, sj = pr[:, 0] + da, pr[:, 1] + db
                    ok = (si >= 0) & (si < n - 1) & (sj >= 0) & (sj < n - 1)
                    segs.append(np.stack([np.minimum(si[ok], sj[ok]), np.maximum(si[ok], sj[ok])], 1))
            c = np.unique(np.concatenate(segs), axis=0)
            c = c[np.abs(c[:, 0] - c[:, 1]) > 1]
            A.append(p0 + c[:, 0]); B.append(p0 + c[:, 1]); Wd.append(np.full(len(c), w))
        dev = "cuda"
        if A:
            self.I = torch.as_tensor(np.concatenate(A), device=dev)
            self.J = torch.as_tensor(np.concatenate(B), device=dev)
            self.Wd = torch.as_tensor(np.concatenate(Wd), device=dev)
        else:
            self.I = None

    def _geom(self, X):
        torch = self.torch
        a, b, c, d = X[self.I], X[self.I + 1], X[self.J], X[self.J + 1]
        u, v, w = b - a, d - c, a - c
        A_, B_, C_ = (u * u).sum(1), (u * v).sum(1), (v * v).sum(1)
        D_, E_ = (u * w).sum(1), (v * w).sum(1)
        den = A_ * C_ - B_ * B_
        ok = den > 1e-18
        dd = torch.where(ok, den, torch.ones_like(den))
        s = torch.where(ok, (B_ * E_ - C_ * D_) / dd, -torch.ones_like(den))
        t = torch.where(ok, (A_ * E_ - B_ * D_) / dd, -torch.ones_like(den))
        cr = torch.cross(u, v, dim=1)
        sig = (cr * (c - a)).sum(1)
        sin = cr.norm(dim=1) / torch.sqrt(A_ * C_ + 1e-30)
        sc, tc = s.clamp(0, 1), t.clamp(0, 1)
        dist = (a + sc[:, None] * u - c - tc[:, None] * v).norm(dim=1)
        e = 1e-3
        inside = (s > -e) & (s < 1 + e) & (t > -e) & (t < 1 + e)
        return sig, sin, dist, inside

    def __call__(self, t):
        with self.torch.cuda.stream(wp.stream_to_torch(wp.get_stream())):   # ordered with warp's own launches
            self._call(t)

    def _call(self, t):
        torch = self.torch
        X = wp.to_torch(t.s0.particle_q).double()
        if self._n_t is None:
            self._n_t = torch.zeros(self.W, dtype=torch.long, device=X.device)
            self._dmin_t = torch.full((self.W,), float("inf"), dtype=torch.float64, device=X.device)
        if t.frame_index != self.frame:                  # a new frame: new candidates, their geometry before it
            self.frame = t.frame_index
            base = self.lastX if self.lastX is not None else X
            self._refresh(base.cpu().numpy(), t.s0.particle_qd.numpy())
            self.prev = self._geom(base) if (self.I is not None and self.lastX is not None) else None
        if self.I is not None:
            g = self._geom(X)
            self._dmin_t.scatter_reduce_(0, self.Wd, g[2], reduce="amin")
            if self.prev is not None:
                p = self.prev
                flip = (torch.sign(p[0]) * torch.sign(g[0]) < 0) & p[3] & g[3] & (p[1] > self.min_sin) & (g[1] > self.min_sin) & \
                       (p[2] < self.near) & (g[2] < self.near)
                self._n_t.index_add_(0, self.Wd, flip.long())
            self.prev = g
        self.lastX = X.clone()

    def finish(self):
        if self._n_t is not None:
            self.n = self._n_t.cpu().numpy()
            self.dmin = self._dmin_t.cpu().numpy()
        return self
