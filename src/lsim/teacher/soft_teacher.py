"""The soft-body teacher: volumetric soft objects (foam, sponge, plush, rubber) as linear-tet FEM on Newton's explicit
integrator, with the contact, friction and damping the soft-body teacher gate passed (`semi_pm` in
scripts/l0_soft_gate.py; DIRECTION.md "Soft-body teacher gate"), batched over parallel worlds and driven by a
kinematic two-finger gripper.

What the gate measured as needed on top of Newton's SolverSemiImplicit (its stable Neo-Hookean tets):

  material   Newton's k_mu / k_lambda are Smith et al.'s, not Lame: k_mu = 4/3 mu, k_lambda = lambda + 5/6 mu (Lame
             passed directly is 31% soft); vertex masses lumped rho V / 4 per corner (`add_soft_mesh`; add_soft_grid
             gives every vertex a whole cell's mass)
  damping    the gate's Kelvin-Voigt on the Green strain rate, S = 2 eta E' (blind to rigid rotation; Newton's k_damp
             on dF/dt brakes a spinning body) is explicit, stable only for dt < 0.1 rho h^2 / eta: on these compact
             8-12-cell bodies that step is 10-100x below the elastic one for any foam-like damping (capped to cost <= 2x,
             76% of the prior's bodies were capped and every foam bounced like rubber). So the generator damps with
             implicit dashpots on every tet edge against its rate of change of length (`_edge_accum` / `_edge_update`:
             (M + dt C) v' = M v, two Jacobi sweeps a substep; rigid motion is in C's null space), c_e = 2.5 eta V_e / L_e^2,
             the dissipation of Kelvin-Voigt's 2 eta tr(D^2) for isotropic edges (plus a bulk part). Free-vibration decay
             against the gate's KV at the same eta: within -10% / +45% (stretch and shear modes, foam cube and rubber
             ball); the spin check holds (KE -0.03% in 1 s). DAMPING = "kv" restores the gate's form
  contact    the cloth teacher's penalty per unit mass (here 1e8 / s^2, critically damped, the damping implicit), mu the
             mean of the body's and the surface's; static friction as a stick spring (the same rate) to where the vertex
             first touched, slipping -- the anchor dragged -- wherever it would exceed mu N (a viscous friction cannot
             hold a column: the bottom vertex carries every layer's weight with only its own mass, 7.6 mm/s creep).
             Anchors are kept per vertex and per surface slot (table/floor, plank, each finger, the palm, objects), and
             ride with a moving surface; contact forces on the body are recorded per world and surface kind
  step       dt = 1.5 x min(0.4 h_alt / c_p, 0.5 / sqrt(contact rate)) (h_alt the mesh's smallest tet altitude, c_p
             the P-wave speed; + 0.1 rho h_alt^2 / eta with the explicit KV): the gate found 1.5x stable, 2x explodes;
             contacts refreshed every <= 0.8 ms of simulated time (a vertex at 3 m/s moves 2.4 mm, inside the 8 mm gap)

Damping prior: each body is given its first-mode damping ratio zeta on omega_1 = pi c_s / L (c_s the shear wave speed,
L its largest extent), eta = 2 zeta mu / omega_1 (lsim.data.soft: 0.02-0.06 for rubber up to 0.35-0.8 for memory foam).
With the implicit dashpots it costs nothing; the step is the elastic and contact limits' alone.

The gripper: a palm and two finger pads, kinematic. Its path is a pose per frame (the fingertip midpoint, yaw) and the
pads' inner opening per frame, interpolated within the frame; the pads push the body through the contact above, and a
grasp pins the vertices touching a pad's inner face to that pad (they follow it rigidly until released; the force it
applies through them is recorded, minus the body's force on them plus their own m (a - g)).
"""
from dataclasses import dataclass, field

import numpy as np
import warp as wp
import newton

from lsim.teacher import soft_mesh

G = 9.81
FPS = 30
CONTACT_RATE = 1.0e8          # 1/s^2: penalty (and stick-spring) stiffness per unit mass. The cloth teacher's 2e7 let a
                              # body's bottom layer sink 6.5 mm on impact (a light, stiff sponge dropped 13 cm: its
                              # column's momentum is stopped by vertices whose springs scale with their own mass) and
                              # pads sink 5.6 mm into light foam; 1e8 costs only the softest bodies' step (contact-bound)
CONTACT_ZETA = 1.0            # normal damping, a fraction of critical, implicit
CFL, DT_SCALE = 0.4, 1.5      # the gate's explicit limits and the factor it found stable
COLLIDE_DT = 8.0e-4           # s: longest simulated time between contact refreshes
GAP = 0.008                   # m: contact candidates within this of a surface
RADIUS = 0.001                # m: vertex contact radius (the gate's)
ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))
ROLE_TABLE, ROLE_GRIPPER, ROLE_OBSTACLE = 0, 1, 2      # recorded surface kinds (the cloth teacher's)
SLOT_TABLE, SLOT_PLANK, SLOT_FL, SLOT_FR, SLOT_PALM, SLOT_OBJ = range(6)   # friction anchors per vertex
N_SLOTS = 6
FINGER_HALF = (0.006, 0.015, 0.045)   # m: a pad 1.2 cm thick, 3 cm wide, 9 cm long; its tip at the gripper origin
PALM_HALF = (0.118, 0.025, 0.012)     # m: the palm over the fingers
W_MAX = 0.20                          # m: widest inner opening between the pads
R_ROUND = 0.004                       # m: the pads' and palm's edge radius (a box core inflated by a contact margin):
                                      # sharp box edges pressed into soft foam inverted its tets (pilot: 12 of 256)
DAMPING = "edge"                      # "edge": implicit edge dashpots (the default); "kv": the gate's explicit KV
EDGE_ITERS = 2                        # Jacobi sweeps of the implicit dashpot solve per substep
EDGE_ALPHA = 2.5                      # edge dashpot c_e = alpha eta V_e / L_e^2 (V_e: volume of the tets on the edge):
                                      # the dissipation of a uniform deviatoric strain rate equals KV's 2 eta tr(D^2)
                                      # for isotropic edge directions (sum_e V_e = 6 V, <(e.D.e)^2> = 2 tr D^2 / 15)


def lame(E, nu):
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def smith(E, nu):
    """Newton's (k_mu, k_lambda) for a body of Young's modulus E and Poisson ratio nu (the gate's mapping)."""
    mu, lam = lame(E, nu)
    return 4.0 / 3.0 * mu, lam + 5.0 / 6.0 * mu


def wave_speeds(E, nu, rho):
    mu, lam = lame(E, nu)
    return float(np.sqrt((lam + 2 * mu) / rho)), float(np.sqrt(mu / rho))


@dataclass
class Soft:
    """A soft body in physical units: its shape (kind and parameters, metres), density rho (kg/m^3), Young's modulus
    E (Pa), Poisson ratio nu, friction mu, first-mode damping ratio zeta (eta in Pa s derives from it)."""
    shape: str
    params: dict
    rho: float = 60.0
    E: float = 1.0e4
    nu: float = 0.3
    mu: float = 0.6
    zeta: float = 0.05
    r: float = RADIUS
    eta: float = None
    _mesh: tuple = field(default=None, repr=False)

    def mesh(self):
        """(vertices (V, 3) centred on the bounding box, tets (T, 4)) in the body frame."""
        if self._mesh is None:
            p = self.params
            if self.shape == "box":
                self._mesh = soft_mesh.box(p["dims"])
            elif self.shape == "cylinder":
                self._mesh = soft_mesh.cylinder(p["R"], p["H"])
            elif self.shape == "ellipsoid":
                self._mesh = soft_mesh.ellipsoid(p["axes"], fine=p.get("fine", False))
            elif self.shape == "blob":
                self._mesh = soft_mesh.blob(p["axes"], [(np.asarray(u), A, k) for u, A, k in p["lumps"]], fine=p.get("res", True))
            else:
                raise ValueError(self.shape)
        return self._mesh

    @property
    def extent(self):
        return np.ptp(self.mesh()[0], axis=0)

    @property
    def volume(self):
        v, t = self.mesh()
        return float(soft_mesh.signed_volume(v, t).sum())

    @property
    def mass(self):
        return self.rho * self.volume

    def h_alt(self):
        if "_h" not in self.params:
            v, t = self.mesh()
            self.params["_h"] = soft_mesh.quality(v, t)["h_min"]
        return self.params["_h"]

    def dt_limits(self):
        """The explicit limits (s): elastic, contact, and -- with the gate's explicit KV damping -- viscous."""
        cp, _ = wave_speeds(self.E, self.nu, self.rho)
        h = self.h_alt()
        out = {"elastic": CFL * h / cp, "contact": 0.5 / np.sqrt(CONTACT_RATE)}
        eta = self.damping()
        if eta > 0 and DAMPING == "kv":
            out["viscous"] = 0.1 * self.rho * h * h / eta
        return out

    def damping(self):
        """eta (Pa s), the Kelvin-Voigt shear viscosity, from zeta on the first mode omega_1 = pi c_s / L (L the largest
        extent): eta = 2 zeta mu / omega_1. With the explicit KV ("kv") it is capped so that the viscous step is >= half
        the others (the gate's form costs ~1 / zeta: 76% of the prior's bodies were capped, every foam bounced like
        rubber); the implicit edge dashpots need no cap."""
        if self.eta is None:
            mu, _ = lame(self.E, self.nu)
            cp, cs = wave_speeds(self.E, self.nu, self.rho)
            om = np.pi * cs / float(self.extent.max())
            eta = 2.0 * self.zeta * mu / om
            if DAMPING == "kv":
                h = self.h_alt()
                other = min(CFL * h / cp, 0.5 / np.sqrt(CONTACT_RATE))
                eta = min(eta, 0.1 * self.rho * h * h / (0.5 * other))
            self.params["zeta_effective"] = float(self.zeta * eta / (2.0 * self.zeta * mu / om))
            self.eta = float(eta)
        return self.eta

    def stable_dt(self):
        return DT_SCALE * min(self.dt_limits().values())

    def meta(self):
        p = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self.params.items() if not k.startswith("_")}
        v, t = self.mesh()
        return {"shape": self.shape, "params": p, "rho": self.rho, "E": self.E, "nu": self.nu, "mu": self.mu,
                "zeta": self.zeta, "zeta_effective": self.params.get("zeta_effective", self.zeta), "eta": self.damping(),
                "r": self.r, "mass": self.mass, "volume": self.volume, "extent": self.extent.tolist(),
                "vertices": int(len(v)), "tets": int(len(t)), "h_alt": self.h_alt()}


def substeps_for(softs, fps=FPS):
    """Substeps per frame for a batch (its hardest body sets the step) and substeps per contact refresh."""
    dt = min(s.stable_dt() for s in softs)
    k = int(np.clip(np.floor(COLLIDE_DT / dt), 4, 50))
    n = int(np.ceil(1.0 / (fps * dt)))
    return int(np.ceil(n / k) * k), k


def add_soft(b, soft, pos, rot, vel=(0.0, 0.0, 0.0)):
    """Add the body to builder b (pos: where its mesh's centre goes). Returns (first particle, count, first tet,
    count)."""
    v, t = soft.mesh()
    k_mu, k_lam = smith(soft.E, soft.nu)
    p0, t0 = b.particle_count, b.tet_count
    b.add_soft_mesh(pos=wp.vec3(*pos), rot=rot, scale=1.0, vel=wp.vec3(*vel), vertices=[wp.vec3(*p) for p in v],
                    indices=t.ravel().tolist(), density=soft.rho, k_mu=k_mu, k_lambda=k_lam, k_damp=0.0,
                    particle_radius=soft.r, add_surface_mesh_edges=False)
    return p0, b.particle_count - p0, t0, b.tet_count - t0


def add_gripper(b, mu):
    """The kinematic gripper: a hand body (the palm) and two finger bodies; returns their body indices and the
    shape slots of the shapes added, in order."""
    def cfg():
        s = newton.ModelBuilder.ShapeConfig()
        s.mu, s.ke, s.kd, s.kf = float(mu), 1.0e3, 0.0, 1.0
        s.density = 0.0
        s.margin = R_ROUND
        return s
    rc = lambda h: [x - R_ROUND for x in h]                                 # noqa: E731  the box core
    I = wp.quat_identity()
    park = wp.transform(wp.vec3(0.0, -0.75, 0.35), I)
    hand = b.add_body(xform=park, mass=0.0, label="gripper")
    b.add_shape_box(body=hand, xform=wp.transform(wp.vec3(0.0, 0.0, 2 * FINGER_HALF[2] + PALM_HALF[2]), I),
                    hx=rc(PALM_HALF)[0], hy=rc(PALM_HALF)[1], hz=rc(PALM_HALF)[2], cfg=cfg())
    fl = b.add_body(xform=park, mass=0.0, label="finger_l")
    b.add_shape_box(body=fl, xform=wp.transform(wp.vec3(0.0, 0.0, FINGER_HALF[2]), I), hx=rc(FINGER_HALF)[0],
                    hy=rc(FINGER_HALF)[1], hz=rc(FINGER_HALF)[2], cfg=cfg())
    fr = b.add_body(xform=park, mass=0.0, label="finger_r")
    b.add_shape_box(body=fr, xform=wp.transform(wp.vec3(0.0, 0.0, FINGER_HALF[2]), I), hx=rc(FINGER_HALF)[0],
                    hy=rc(FINGER_HALF)[1], hz=rc(FINGER_HALF)[2], cfg=cfg())
    return (hand, fl, fr), [SLOT_PALM, SLOT_FL, SLOT_FR]


def finger_boxes(q, w):
    """World poses (centre, rotation matrix) and half extents of the palm and both pads for a gripper pose q (7) and
    opening w (the meshes the renderer and the penetration check use)."""
    R = _quat_matrix(q[3:])
    p = np.asarray(q[:3], float)
    out = [(p + R @ np.array([0.0, 0.0, 2 * FINGER_HALF[2] + PALM_HALF[2]]), R, PALM_HALF)]
    for s in (-1.0, 1.0):
        out.append((p + R @ np.array([s * (w / 2 + FINGER_HALF[0]), 0.0, FINGER_HALF[2]]), R, FINGER_HALF))
    return out


# ---------------------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------------------
@wp.kernel
def _strain_rate_damping(x: wp.array[wp.vec3], v: wp.array[wp.vec3], tets: wp.array2d[int], pose: wp.array[wp.mat33],
                         eta: wp.array[float], f: wp.array[wp.vec3]):
    """The gate's Kelvin-Voigt damping on the Green strain rate, eta per tet: S = 2 eta E', E' = (F'^T F + F^T F') / 2,
    P = F S, f_i = -V P grad N_i. Rigid motion gives E' = 0."""
    t = wp.tid()
    i, j, k, l = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    Dm = pose[t]
    V = 1.0 / (6.0 * wp.determinant(Dm))
    F = wp.matrix_from_cols(x[j] - x[i], x[k] - x[i], x[l] - x[i]) * Dm
    Fd = wp.matrix_from_cols(v[j] - v[i], v[k] - v[i], v[l] - v[i]) * Dm
    Ed = 0.5 * (wp.transpose(Fd) * F + wp.transpose(F) * Fd)
    H = (F * Ed) * (2.0 * eta[t] * V) * wp.transpose(Dm)
    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])
    wp.atomic_add(f, i, f1 + f2 + f3)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)
    wp.atomic_sub(f, l, f3)


@wp.kernel
def _edge_accum(v: wp.array[wp.vec3], x: wp.array[wp.vec3], edges: wp.array2d[int], c: wp.array[float], dt: float,
                with_diag: int, acc: wp.array[wp.vec3], diag: wp.array[wp.mat33]):
    """One Jacobi sweep of the implicit edge dashpots (M + dt C) v' = M v: per edge, dt c e e^T applied to the other
    end's velocity (and, once per substep, to the diagonal)."""
    k = wp.tid()
    i = edges[k, 0]
    j = edges[k, 1]
    d = x[j] - x[i]
    L = wp.length(d)
    if L < 1.0e-9:
        return
    e = d / L
    a = dt * c[k]
    wp.atomic_add(acc, i, e * (a * wp.dot(e, v[j])))
    wp.atomic_add(acc, j, e * (a * wp.dot(e, v[i])))
    if with_diag == 1:
        ee = wp.outer(e, e) * a
        wp.atomic_add(diag, i, ee)
        wp.atomic_add(diag, j, ee)


@wp.kernel
def _edge_update(v0: wp.array[wp.vec3], acc: wp.array[wp.vec3], diag: wp.array[wp.mat33], mass: wp.array[float],
                 flags: wp.array[wp.int32], v: wp.array[wp.vec3]):
    i = wp.tid()
    m = mass[i]
    if (flags[i] & ACTIVE) == 0 or m <= 0.0:
        v[i] = v0[i]
    else:
        v[i] = wp.inverse(wp.identity(3, float) * m + diag[i]) * (v0[i] * m + acc[i])
    acc[i] = wp.vec3(0.0)


@wp.kernel
def _contact(count: wp.array[int], cmax: int, c_particle: wp.array[int], c_shape: wp.array[int],
             c_bpos: wp.array[wp.vec3], c_normal: wp.array[wp.vec3], shape_body: wp.array[int],
             shape_mu: wp.array[float], shape_role: wp.array[int], shape_slot: wp.array[int],
             shape_margin: wp.array[float], body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
             x: wp.array[wp.vec3], v: wp.array[wp.vec3], radius: wp.array[float], flags: wp.array[wp.int32],
             p_mu: wp.array[float], p_world: wp.array[int], inv_mass: wp.array[float], rate: float, zeta: float,
             dt: float, anchor: wp.array2d[wp.vec3], anchored: wp.array2d[int], touched: wp.array2d[int],
             f: wp.array[wp.vec3], rec: wp.array2d[wp.vec3], rec_slot: wp.array2d[wp.vec3]):
    """The gate's semi_pm contact on static or moving shapes: penalty rate m, critically damped (implicit, never
    pulling); a stick spring (rate m, critically damped) to the anchor where the vertex first touched this surface,
    the anchor riding with the surface and dragged along wherever the spring would exceed mu N (mu the mean of the
    body's and the surface's)."""
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
    c = wp.dot(n, x[i] - bx) - radius[i] - shape_margin[s]     # the margin rounds a box's edges
    if c > 0.0:
        return
    w = inv_mass[i]
    if w <= 0.0:
        return
    m = 1.0 / w
    cw = wp.transform_point(X, com)
    lin = wp.spatial_top(vs)
    ang = wp.spatial_bottom(vs)
    vr = v[i] - (lin + wp.cross(ang, bx - cw))
    vn = wp.dot(n, vr)
    ke = rate * m
    kd = 2.0 * zeta * wp.sqrt(ke * m)
    kdi = kd / (1.0 + kd * dt * w)
    fn = wp.max(-ke * c - kdi * vn, 0.0)
    vt = vr - n * vn
    cap = 0.5 * (p_mu[i] + shape_mu[s]) * fn
    k = shape_slot[s]
    touched[i, k] = 1
    if anchored[i, k] == 0:
        anchor[i, k] = x[i]
        anchored[i, k] = 1
    else:
        a = anchor[i, k]
        anchor[i, k] = a + (lin + wp.cross(ang, a - cw)) * dt
    d = x[i] - anchor[i, k]
    d = d - n * wp.dot(n, d)
    ft = -ke * d - kdi * vt
    lt = wp.length(ft)
    if lt > cap:
        ft = ft * (cap / lt)
        anchor[i, k] = x[i] + ft / ke - n * wp.dot(n, ft / ke)   # the spring alone now gives the cap
    F = n * fn + ft
    wp.atomic_add(f, i, F)
    wp.atomic_add(rec, p_world[i], shape_role[s], F)
    wp.atomic_add(rec_slot, p_world[i], k, F)


@wp.kernel
def _anchor_release(touched: wp.array2d[int], anchored: wp.array2d[int]):
    i, k = wp.tid()
    if touched[i, k] == 0:
        anchored[i, k] = 0
    touched[i, k] = 0


@wp.kernel
def _gripper_pose(path: wp.array2d[wp.transform], width: wp.array2d[float], clock: wp.array[int], substeps: int,
                  fps: float, bodies: wp.array2d[int], fhx: float, body_q: wp.array[wp.transform],
                  body_qd: wp.array[wp.spatial_vector]):
    """Palm and pads at the current substep: poses and opening per frame, interpolated within it."""
    wi = wp.tid()
    s = clock[0]
    fr = s // substeps
    a = float(s - fr * substeps) / float(substeps)
    nf = path.shape[0] - 1
    f0 = wp.min(fr, nf)
    f1 = wp.min(fr + 1, nf)
    p0 = wp.transform_get_translation(path[f0, wi])
    p1 = wp.transform_get_translation(path[f1, wi])
    q0 = wp.transform_get_rotation(path[f0, wi])
    q1 = wp.transform_get_rotation(path[f1, wi])
    if wp.dot(q0, q1) < 0.0:
        q1 = -q1
    q = wp.normalize(q0 * (1.0 - a) + q1 * a)
    dq = q1 * wp.quat_inverse(q0)
    om = 2.0 * wp.vec3(dq[0], dq[1], dq[2]) * fps
    p = p0 * (1.0 - a) + p1 * a
    lin = (p1 - p0) * fps
    w = width[f0, wi] * (1.0 - a) + width[f1, wi] * a
    wd = (width[f1, wi] - width[f0, wi]) * fps
    X = wp.transform(p, q)
    body_q[bodies[wi, 0]] = X
    body_qd[bodies[wi, 0]] = wp.spatial_vector(lin, om)
    ax = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    for side in range(2):
        sg = 2.0 * float(side) - 1.0
        off = ax * (sg * (0.5 * w + fhx))
        body_q[bodies[wi, 1 + side]] = wp.transform(p + off, q)
        body_qd[bodies[wi, 1 + side]] = wp.spatial_vector(lin + wp.cross(om, off) + ax * (sg * 0.5 * wd), om)


@wp.kernel
def _attach(att_p: wp.array[int], att_b: wp.array[int], att_local: wp.array[wp.vec3], att_on: wp.array[int],
            body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], x: wp.array[wp.vec3],
            v: wp.array[wp.vec3]):
    """Grasped vertices follow their pad rigidly."""
    k = wp.tid()
    if att_on[k] == 0:
        return
    i = att_p[k]
    X = body_q[att_b[k]]
    x[i] = wp.transform_point(X, att_local[k])
    vs = body_qd[att_b[k]]
    v[i] = wp.spatial_top(vs) + wp.cross(wp.spatial_bottom(vs), x[i] - wp.transform_get_translation(X))


@wp.kernel
def _grasp_force(att_p: wp.array[int], att_w: wp.array[int], att_on: wp.array[int], f: wp.array[wp.vec3],
                 rec: wp.array[wp.vec3]):
    """Minus the body's force on the grasped vertices, per world (their own m (a - g) is added per frame)."""
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
class SoftTeacher:
    """Step a batch of soft-body worlds. Built from a builder and per-world bookkeeping:

      worlds: list of dicts with "particles" (first, count), "tets" (first, count), "soft" (Soft), "gripper"
              (hand, left pad, right pad body indices; or None)
      shape_role, shape_slot: per shape, its recorded kind (ROLE_*) and its friction-anchor slot (SLOT_*)
      path: (n_frames + 1, W, 7) gripper poses per frame; width: (n_frames + 1, W) pad openings

    Each frame: grasp/release events are applied between frames (`grasp`, `release`), then the frame is simulated in
    captured graphs of `collide_every` substeps, and `frame()` returns what was recorded."""

    MAX_ATTACH = 160                                   # pinned vertices per world

    def __init__(self, builder, worlds, shape_role, shape_slot, path, width, substeps, collide_every, settle_damping=0.0,
                 velocities=None):
        model = builder.finalize()
        self.model, self.worlds, self.W = model, worlds, len(worlds)
        self.substeps, self.ce = int(substeps), int(collide_every)
        assert self.substeps % self.ce == 0
        self.dt = 1.0 / (FPS * self.substeps)
        self.settle_damping = float(settle_damping)
        P = model.particle_count
        pmu = np.zeros(P, np.float32)
        eta = np.zeros(model.tet_count, np.float32)
        for wd in worlds:
            p0, n = wd["particles"]
            t0, nt = wd["tets"]
            pmu[p0:p0 + n] = wd["soft"].mu
            eta[t0:t0 + nt] = wd["soft"].damping()
        self.p_mu = wp.array(pmu, dtype=wp.float32)
        self.eta = wp.array(eta, dtype=wp.float32)
        self.damping = DAMPING
        if DAMPING == "edge":                          # unique tet edges, their dashpots and the vertex valences
            T = model.tet_indices.numpy()
            Dm = model.tet_poses.numpy().astype(np.float64)
            vol = 1.0 / (6.0 * np.abs(np.linalg.det(Dm)))
            e = np.concatenate([T[:, [a, b_]] for a in range(4) for b_ in range(a + 1, 4)])
            e = np.sort(e, axis=1)
            ue, inv = np.unique(e, axis=0, return_inverse=True)
            inv = inv.ravel()
            Ve = np.zeros(len(ue))
            np.add.at(Ve, inv, np.tile(vol, 6))
            Ee = np.zeros(len(ue))
            np.add.at(Ee, inv, np.tile(eta.astype(np.float64) * vol, 6))
            eta_e = Ee / np.maximum(Ve, 1e-30)
            x0 = model.particle_q.numpy().astype(np.float64)
            L = np.linalg.norm(x0[ue[:, 1]] - x0[ue[:, 0]], axis=1)
            self.edges = wp.array(ue.astype(np.int32), dtype=int)
            self.edge_c = wp.array((EDGE_ALPHA * eta_e * Ve / (L * L)).astype(np.float32), dtype=float)
            self.v0 = wp.zeros(P, dtype=wp.vec3)
            self.acc = wp.zeros(P, dtype=wp.vec3)
            self.diag = wp.zeros(P, dtype=wp.mat33)
            self.mass_w = wp.array(model.particle_mass.numpy().astype(np.float32), dtype=float)
        self.shape_role = wp.array(np.asarray(shape_role, np.int32), dtype=int)
        self.shape_slot = wp.array(np.asarray(shape_slot, np.int32), dtype=int)
        sm = model.shape_margin.numpy() if model.shape_margin is not None else np.zeros(model.shape_count)
        self.shape_margin = wp.array(np.asarray(sm, np.float32), dtype=float)
        model.particle_grid = None                     # no particle-particle contact inside a body
        self.solver = newton.solvers.SolverSemiImplicit(model, enable_tri_contact=False)
        self.s0, self.s1, self.ctrl = model.state(), model.state(), model.control()
        if velocities is not None:                     # initial velocities (V, 3) for every particle
            self.s0.particle_qd.assign(np.asarray(velocities, np.float32))
        self.pipe = newton.CollisionPipeline(model, soft_contact_gap=GAP, soft_contact_max=int(P * 4))
        self.con = self.pipe.contacts()
        self.anchor = wp.zeros((P, N_SLOTS), dtype=wp.vec3)
        self.anchored = wp.zeros((P, N_SLOTS), dtype=int)
        self.touched = wp.zeros((P, N_SLOTS), dtype=int)
        self.has_grip = all(wd.get("gripper") is not None for wd in worlds)
        if self.has_grip:
            self.bodies = wp.array(np.array([wd["gripper"] for wd in worlds], np.int32), dtype=int)
        self.path = wp.array(np.asarray(path, np.float32), dtype=wp.transform)
        self.width = wp.array(np.asarray(width, np.float32), dtype=float)
        self.n_path = path.shape[0]
        self.clock = wp.zeros(1, dtype=int)
        A = self.MAX_ATTACH * self.W
        self.att_p = wp.zeros(A, dtype=int)
        self.att_b = wp.zeros(A, dtype=int)
        self.att_w = wp.array(np.repeat(np.arange(self.W), self.MAX_ATTACH).astype(np.int32), dtype=int)
        self.att_local = wp.zeros(A, dtype=wp.vec3)
        self.att_on = wp.zeros(A, dtype=int)
        self._att = {w: [] for w in range(self.W)}
        self.rec_contact = wp.zeros((self.W, 3), dtype=wp.vec3)
        self.rec_slot = wp.zeros((self.W, N_SLOTS), dtype=wp.vec3)     # per anchor slot: each pad apart
        self.rec_grasp = wp.zeros(self.W, dtype=wp.vec3)
        self.flags = model.particle_flags.numpy()
        self.mass = model.particle_mass.numpy()
        self.graph = None
        self.frame_index = 0

    # -- between frames -----------------------------------------------------------------------------------
    def body_q(self):
        return self.s0.body_q.numpy()

    def grasp(self, w, reach=0.003):
        """Pin world w's vertices within `reach` of either pad's inner face (and inside its footprint) to that pad
        where they are now. Returns their indices in the body (empty: nothing touched the pads)."""
        p0, n = self.worlds[w]["particles"]
        x = self.s0.particle_q.numpy()[p0:p0 + n].astype(np.float64)
        bq = self.body_q()
        ids, bodies, local = [], [], []
        for side, b in ((-1.0, self.worlds[w]["gripper"][1]), (1.0, self.worlds[w]["gripper"][2])):
            X = bq[b]
            R = _quat_matrix(X[3:])
            lp = (x - X[:3]) @ R                          # in the pad's frame (its centre line at local x = 0)
            inner = -side * FINGER_HALF[0]                # the face towards the other pad
            dist = (lp[:, 0] - inner) * (-side)           # > 0: between the pads
            ok = (dist > -0.002) & (dist < reach) & (np.abs(lp[:, 1]) < FINGER_HALF[1] + 0.001) & \
                 (lp[:, 2] > -0.001) & (lp[:, 2] < 2 * FINGER_HALF[2])
            k = np.flatnonzero(ok)
            k = k[np.argsort(dist[k])][: self.MAX_ATTACH // 2]
            ids += list(k)
            bodies += [b] * len(k)
            local += list(lp[k])
        return self._pin(w, np.asarray(ids, int), np.asarray(bodies, int), np.asarray(local).reshape(-1, 3))

    def _pin(self, w, ids, bodies, local):
        p0, _ = self.worlds[w]["particles"]
        ids = ids + p0
        self._att[w] = list(ids)
        k0 = w * self.MAX_ATTACH
        on, ps, bs, ls = self.att_on.numpy(), self.att_p.numpy(), self.att_b.numpy(), self.att_local.numpy()
        on[k0:k0 + self.MAX_ATTACH] = 0
        on[k0:k0 + len(ids)] = 1
        ps[k0:k0 + len(ids)] = ids
        bs[k0:k0 + len(ids)] = bodies
        ls[k0:k0 + len(ids)] = local
        self.att_on.assign(on)
        self.att_p.assign(ps)
        self.att_b.assign(bs)
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

    def fix(self, ids):
        """Hold these particles (global indices) where they are, for good (closed-form checks: a clamped face)."""
        ids = np.asarray(ids, int)
        self.flags[ids] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(self.flags)
        m = self.model.particle_inv_mass.numpy()
        m[ids] = 0.0
        self.model.particle_inv_mass.assign(m)

    # -- stepping -----------------------------------------------------------------------------------------
    def _substeps(self, k):
        m = self.model
        for i in range(k):
            self.s0.clear_forces()
            if self.has_grip:                          # the gripper first: contacts are found against where it is
                wp.launch(_gripper_pose, dim=self.W, inputs=[self.path, self.width, self.clock, self.substeps, float(FPS),
                                                             self.bodies, FINGER_HALF[0]],
                          outputs=[self.s0.body_q, self.s0.body_qd])
            wp.launch(_attach, dim=len(self.att_on), inputs=[self.att_p, self.att_b, self.att_local, self.att_on,
                                                           self.s0.body_q, self.s0.body_qd],
                      outputs=[self.s0.particle_q, self.s0.particle_qd])
            if i == 0:
                self.pipe.collide(self.s0, self.con)
            if self.damping == "kv":
                wp.launch(_strain_rate_damping, dim=m.tet_count, inputs=[self.s0.particle_q, self.s0.particle_qd,
                                                                         m.tet_indices, m.tet_poses, self.eta],
                          outputs=[self.s0.particle_f])
            else:                                      # the damping solve first, then the step from those velocities
                wp.copy(self.v0, self.s0.particle_qd)
                self.diag.zero_()
                for it in range(EDGE_ITERS):
                    src = self.v0 if it == 0 else self.s0.particle_qd
                    wp.launch(_edge_accum, dim=self.edges.shape[0],
                              inputs=[src, self.s0.particle_q, self.edges, self.edge_c, self.dt, int(it == 0)],
                              outputs=[self.acc, self.diag])
                    wp.launch(_edge_update, dim=m.particle_count,
                              inputs=[self.v0, self.acc, self.diag, self.mass_w, m.particle_flags],
                              outputs=[self.s0.particle_qd])
            c = self.con
            wp.launch(_contact, dim=c.soft_contact_max,
                      inputs=[c.soft_contact_count, c.soft_contact_max, c.soft_contact_particle, c.soft_contact_shape,
                              c.soft_contact_body_pos, c.soft_contact_normal, m.shape_body, m.shape_material_mu,
                              self.shape_role, self.shape_slot, self.shape_margin, self.s0.body_q, self.s0.body_qd, m.body_com,
                              self.s0.particle_q, self.s0.particle_qd, m.particle_radius, m.particle_flags, self.p_mu,
                              m.particle_world, m.particle_inv_mass, CONTACT_RATE, CONTACT_ZETA, self.dt,
                              self.anchor, self.anchored, self.touched],
                      outputs=[self.s0.particle_f, self.rec_contact, self.rec_slot])
            wp.launch(_anchor_release, dim=(m.particle_count, N_SLOTS), inputs=[self.touched, self.anchored])
            self.solver.step(self.s0, self.s1, self.ctrl, None, self.dt)
            if self.settle_damping > 0.0:
                wp.launch(_damp, dim=m.particle_count, inputs=[self.s1.particle_qd, float(np.exp(-self.settle_damping * self.dt))])
            wp.launch(_grasp_force, dim=len(self.att_on), inputs=[self.att_p, self.att_w, self.att_on, self.s0.particle_f],
                      outputs=[self.rec_grasp])
            self.s0, self.s1 = self.s1, self.s0
            wp.launch(_tick, dim=1, inputs=[self.clock])

    def frame(self):
        """Simulate one frame; returns its record (all vertices at its end, mean contact forces on each body by surface
        kind and per anchor slot (each pad apart: the two pads' squeeze cancels in the gripper's sum), mean grasp force
        before its m (a - g))."""
        self.rec_contact.zero_()
        self.rec_slot.zero_()
        self.rec_grasp.zero_()
        n_chunks = self.substeps // self.ce
        done = 0
        if self.graph is None:
            self._substeps(self.ce)                    # uncaptured: compiles every kernel
            done = 1
            with wp.ScopedCapture() as cap:            # recorded, not run
                self._substeps(self.ce)
            self.graph = cap.graph
        for _ in range(n_chunks - done):
            wp.capture_launch(self.graph)
        self.frame_index += 1
        if self.has_grip:
            wp.launch(_gripper_pose, dim=self.W, inputs=[self.path, self.width, self.clock, self.substeps, float(FPS),
                                                         self.bodies, FINGER_HALF[0]],
                      outputs=[self.s0.body_q, self.s0.body_qd])
        wp.launch(_attach, dim=len(self.att_on), inputs=[self.att_p, self.att_b, self.att_local, self.att_on,
                                                       self.s0.body_q, self.s0.body_qd],
                  outputs=[self.s0.particle_q, self.s0.particle_qd])
        return {"x": self.s0.particle_q.numpy(), "contact": self.rec_contact.numpy() / self.substeps,
                "slot": self.rec_slot.numpy() / self.substeps,
                "grasp": self.rec_grasp.numpy() / self.substeps}


def _quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
