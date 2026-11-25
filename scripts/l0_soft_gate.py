"""L0 -- soft-body teacher gate (T0-soft): which Newton solver, if any, can teach volumetric soft bodies?

A dataset inherits every defect of the solver that generated it, so the soft-body teacher (plush toys, sponges, foam,
rubber) is checked against closed forms and invariants before any episode is generated (CLAUDE.md rule 2), the way
`l0_cloth_gate.py` found that VBD and Style3D run stably yet simulate a different cloth. The body is given in physical
units -- density rho (kg/m^3; foam 30-100, rubber ~1,100), Young's modulus E (Pa; foam 1e3-1e5, rubber 1e6), Poisson
ratio nu, a Kelvin-Voigt shear viscosity eta (Pa s: stress 2 eta x strain rate), friction mu -- meshed as linear
tetrahedra (5 per cube cell, lumped mass rho V / 4 per corner: `add_soft_mesh`; `add_soft_grid` gives every vertex a
whole cell's mass, 2.5x too heavy on a 2 x 2 x 10 bar) and each solver gets the same body through its own parameters:

  semi     SolverSemiImplicit (explicit), Smith et al. 2018 stable Neo-Hookean: its k_mu, k_lambda are Smith's, not
           Lame (linearised at rest it is mu_lin = 3/4 k_mu, lambda_lin = k_lambda - 5/8 k_mu), so k_mu = 4/3 mu,
           k_lambda = lambda + 5/6 mu; k_damp = 2 eta on dF/dt (rotation included). Newton's contact (one ke, kf, mu).
  semi_pm  the same FEM, with the cloth teacher's contact (penalty per unit mass, critically damped, implicit), static
           friction as a stick spring to the touch point slipping at mu N, and Kelvin-Voigt damping on the Green strain
           rate (blind to rigid rotation). semi_pmv: the same with the cloth teacher's viscous friction (a 10 cm block
           creeps 7.6 mm/s at mu 0.5 on 20 deg: the bottom vertex holds a column with its own mass).
           Stable at dt = min(0.4 h_alt / c_p, 0.5 / sqrt(contact rate), viscous limit) x 1.5; explodes at x 2
           (squeeze, contact-bound) and x 3 (free vibration, 1.2 h_alt / c_p).
  vbd      SolverVBD, Neo-Hookean mu/2 (Ic-3) + (lambda+mu)/2 (J-1-mu/(lambda+mu))^2: takes Lame directly;
           damping d on dC/dt (C = F^T F; rotation-blind): d = eta / 2.
  xpbd     SolverXPBD, constraints C_dev = Ic - 3 and C_vol = J - 1 with compliance 1/(V k): energy k_mu/2 (Ic-3)^2 +
           k_lambda/2 (J-1)^2 -- its small-strain shear modulus is ZERO (Ic - 3 is first order only in tr eps), and it
           skips every tet with k_lambda = 0 (nu = 0); given k_mu = mu, k_lambda = lambda as its docstring says.
  mpm      SolverImplicitMPM, linear-elastic compliance per particle (E, nu given directly), voxel grid.

Checks (each compares with a closed form or an invariant; each was run on a deliberately wrong body -- `--control` --
to show it can fail):

  hang       bar 2 x 2 x 10 cm hung from its top face, nu = 0: bottom-face drop against rho g L^2 / (2 E) for foam
             E 1e3 (rho 30), 1e4 (rho 60) and rubber 1e6 (rho 1100); 5%
  poisson    bar 2 x 2 x 16 cm (0.5 cm cells), ends clamped and pulled 1% apart then pushed 1% together: lateral
             strain of the middle section / axial strain of the middle 6 cm against -nu (nu 0.3 and 0.45; 5%, |.| <
             0.01 at nu 0); volume change of the middle against (1 - 2 nu) eps at nu 0.45 (10%: linear tets lock)
  cantilever rubber beam 15 x 2 x 2 cm clamped at one end, sagging under its weight: tip deflection against
             Timoshenko q L^4/(8 E I) + q L^2/(2 k G A) (10%), at 1, 0.5 and 0.25 cm cells (E is physical: the answer
             must not depend on the mesh), and a foam beam (E 1e5, rho 60)
  vibration  the same beam released straight, undamped: first-mode frequency (zero crossings of the tip) against
             (1.875^2 / 2 pi) sqrt(E I / (rho A L^4)) (5%): mass and dynamics, not just statics
  energy     that free vibration: kinetic + gravity + the solver's own elastic energy never rises above its start
             (1% of the energy released); with eta it comes to rest (no vertex moves 0.5 mm in the 5th second)
  spin       a 5 cm cube spinning at 1 rev/s with no gravity and eta on: kinetic energy and angular momentum kept
             (1% after 1 s) -- damping must not brake rigid rotation
  freefall   tilted 5 cm cube falling 0.5 s: centroid acceleration g (0.1%, quadratic fit), shape kept (0.1 mm)
  slide      4 x 4 x 2 cm foam block on a 20 deg plane: a = g (sin - mu cos) within 2% at mu 0.2; drift < 1 mm/s
             at mu 0.5
  rest       10 cm foam cube (rho 60, E 1e3, nu 0) dropped 1 cm onto a table: at rest after 3 s (no vertex moves
             1 mm in the 4th second), no creep (< 1 mm/s), no penetration (> -1 mm), and it compresses under its own
             weight by rho g H^2 / (2 E) = 2.9 mm (top face relative to the bottom face; 10%)
  squeeze    5 cm foam cube (E 1e4, nu 0.3) on the table, two plates close to 50% of its width in 1 s, hold 0.5 s,
             open in 1 s: inside a spatial bound (never isfinite alone), no inverted tet at any frame, shape back
             within 1 mm of the rest shape (rigid alignment) 1 s after release
  throughput 64 worlds of the squeeze: simulated seconds per wall second

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l0_soft_gate.py [check ...] [--solver semi,semi_pm,vbd,xpbd] [--control 1]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l0_soft_gate.json"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    from lsim.gpu import wait_for_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(wait_for_gpu(min_free_gib=8.0, max_util=20))

import warp as wp                                     # noqa: E402
import newton                                         # noqa: E402

G = 9.81
FPS = 30
FOAM = dict(rho=60.0, E=1.0e4, nu=0.3, eta=0.0, mu=0.6, h=0.01, r=0.001)
SOLVERS = {
    # explicit: dt from the stability bound (fraction `cfl` of the P-wave crossing of the smallest tet altitude,
    # and of the contact and viscous limits); contacts refreshed every `collide_every` substeps
    "semi": dict(kind="semi", cfl=0.4, contact="newton", damping="newton", collide_every=10, dt_scale=1.0),
    "semi_pmv": dict(kind="semi", cfl=0.4, contact="pm", damping="strain", collide_every=10, dt_scale=1.0,
                     friction="viscous"),
    "semi_pm": dict(kind="semi", cfl=0.4, contact="pm", damping="strain", collide_every=10, dt_scale=1.0,
                    friction="anchor"),
    "vbd": dict(kind="vbd", substeps=20, iters=10, feps=1.0e-2),
    "xpbd": dict(kind="xpbd", substeps=20, iters=10),
    # implicit MPM: material points 2 per voxel per axis, implicit grid solve; voxel 5 mm
    "mpm": dict(kind="mpm", substeps=20, voxel=0.005, ppv=2, iters=250, tol=1.0e-6),
}
CONTACT = dict(ke=1.0e3, kd=0.0, kf=0.05)            # Newton's contact: N/m, N s/m (kf < 2 m / dt on a foam corner)
PM = dict(rate=2.0e7, zeta=1.0, frate=2.0e4)          # per unit mass: the cloth teacher's contact (1/s^2, -, 1/s)
CONTROL = {}                                          # a deliberately wrong body for --control (closed forms see none)
ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))


def lame(E, nu):
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# the structured box mesh and the volume ratio live in the soft-body teacher's mesher (moved there unchanged)
from lsim.teacher.soft_mesh import box_tets, tet_J  # noqa: E402,F401


def jmin(sim, x):
    """Smallest tet volume ratio (NaN without a mesh: MPM)."""
    return float(tet_J(x, sim.tets, sim.Dm_inv).min()) if len(sim.tets) else float("nan")


def material(solver, body):
    """The solver's own (k_mu, k_lambda, k_damp) for a body in physical units."""
    E, nu, eta = body["E"] * CONTROL.get("E", 1.0), body["nu"] + CONTROL.get("nu", 0.0), body["eta"]
    mu, lam = lame(E, nu)
    kind = SOLVERS[solver]["kind"]
    if kind == "semi":
        kd = 2.0 * eta if SOLVERS[solver]["damping"] == "newton" else 0.0
        if CONTROL.get("raw"):                        # Newton's docstring taken literally: Lame into Smith's slots
            return mu, max(lam, 1e-6 * mu), kd
        return 4.0 / 3.0 * mu, lam + 5.0 / 6.0 * mu, kd
    if kind == "vbd":
        return mu, lam, 0.5 * eta
    return mu, lam, 2.0 * eta                         # xpbd


def add_body(b, solver, body, dims, pos, rot=None, h=None):
    """A box of the given physical body, dims (m) at cell size h; returns (ids, tets local, rest positions)."""
    h = h or body["h"]
    b._gate_body = body
    if SOLVERS[solver]["kind"] == "mpm":              # material points filling the box, 2 per voxel per axis
        hp = SOLVERS[solver]["voxel"] / SOLVERS[solver]["ppv"]
        n = [int(round(d / hp)) for d in dims]
        first = b.particle_count
        R = np.array(wp.quat_to_matrix(rot if rot is not None else wp.quat_identity()), float).reshape(3, 3)
        for k in range(n[2]):
            for j in range(n[1]):
                for i in range(n[0]):
                    p = np.asarray(pos) + R @ (hp * (np.array([i, j, k]) + 0.5))
                    b.add_particle(wp.vec3(*p), wp.vec3(0.0, 0.0, 0.0), body["rho"] * CONTROL.get("rho", 1.0) * hp ** 3,
                                   0.5 * hp)
        return np.arange(first, b.particle_count), np.zeros((0, 4), int)
    n = [int(round(d / h)) for d in dims]
    v, t = box_tets(*n, h)
    k_mu, k_lam, k_damp = material(solver, body)
    first = b.particle_count
    b.add_soft_mesh(pos=wp.vec3(*pos), rot=rot if rot is not None else wp.quat_identity(), scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0), vertices=[wp.vec3(*p) for p in v], indices=t.ravel().tolist(),
                    density=body["rho"] * CONTROL.get("rho", 1.0), k_mu=k_mu, k_lambda=k_lam, k_damp=k_damp,
                    particle_radius=body["r"], add_surface_mesh_edges=False)
    return np.arange(first, b.particle_count), t


def static_box(b, center, half, mu, rot=None):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu, cfg.ke, cfg.kd, cfg.kf = mu, CONTACT["ke"], CONTACT["kd"], CONTACT["kf"]
    return b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*center), rot if rot is not None else wp.quat_identity()),
                           hx=half[0], hy=half[1], hz=half[2], cfg=cfg)


# ---------------------------------------------------------------------------------------------- our kernels (semi_pm)

@wp.kernel
def _strain_rate_damping(x: wp.array[wp.vec3], v: wp.array[wp.vec3], tets: wp.array2d[int], pose: wp.array[wp.mat33],
                         eta: float, f: wp.array[wp.vec3]):
    """Kelvin-Voigt on the Green strain rate: S = 2 eta E', E' = (F'^T F + F^T F') / 2, P = F S, f_i = -V P grad N_i.
    Rigid motion (F' = W F, W skew) gives E' = 0: a spinning body is not braked."""
    t = wp.tid()
    i, j, k, l = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    Dm = pose[t]
    V = 1.0 / (6.0 * wp.determinant(Dm))
    F = wp.matrix_from_cols(x[j] - x[i], x[k] - x[i], x[l] - x[i]) * Dm
    Fd = wp.matrix_from_cols(v[j] - v[i], v[k] - v[i], v[l] - v[i]) * Dm
    Ed = 0.5 * (wp.transpose(Fd) * F + wp.transpose(F) * Fd)
    H = (F * Ed) * (2.0 * eta * V) * wp.transpose(Dm)
    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])
    wp.atomic_add(f, i, f1 + f2 + f3)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)
    wp.atomic_sub(f, l, f3)


@wp.kernel
def _pm_contact(count: wp.array[int], cmax: int, c_particle: wp.array[int], c_shape: wp.array[int],
                c_bpos: wp.array[wp.vec3], c_normal: wp.array[wp.vec3], shape_mu: wp.array[float],
                shape_vel: wp.array[wp.vec3], x: wp.array[wp.vec3], v: wp.array[wp.vec3], radius: wp.array[float],
                flags: wp.array[wp.int32], inv_mass: wp.array[float], p_mu: float, rate: float, frate: float,
                zeta: float, dt: float, anchor_on: int, anchor: wp.array[wp.vec3], anchored: wp.array[int],
                touched: wp.array[int], f: wp.array[wp.vec3]):
    """The cloth teacher's surface contact on static (or prescribed-moving) shapes: penalty ke = rate m, damping
    critical and implicit (never pulls), mu the mean of the body's and the surface's. Friction either viscous at
    frate m below the Coulomb cap (the cloth teacher's), or -- anchor_on -- a stick spring to where the vertex first
    touched (rate m, critically damped), slipping (the anchor dragged along) wherever it would exceed mu N: a
    viscous friction cannot hold a column -- the bottom vertex carries n layers' weight with only its own mass to
    stop, so a resting body creeps n g sin dt / 2 at best."""
    tid = wp.tid()
    if tid >= wp.min(cmax, count[0]):
        return
    i = c_particle[tid]
    if i < 0:
        return
    if (flags[i] & ACTIVE) == 0:
        return
    s = c_shape[tid]
    bx = c_bpos[tid]                                  # static shapes: body frame = world frame
    n = c_normal[tid]
    c = wp.dot(n, x[i] - bx) - radius[i]
    if c > 0.0:
        return
    w = inv_mass[i]
    if w <= 0.0:
        return
    m = 1.0 / w
    vr = v[i] - shape_vel[s]
    vn = wp.dot(n, vr)
    ke = rate * m
    kd = 2.0 * zeta * wp.sqrt(ke * m)
    fn = wp.max(-ke * c - kd * vn / (1.0 + kd * dt * w), 0.0)
    vt = vr - n * vn
    vs = wp.length(vt)
    cap = 0.5 * (p_mu + shape_mu[s]) * fn
    F = n * fn
    if anchor_on == 0:
        if vs > 1.0e-12:
            F = F - vt / vs * wp.min(frate * m * vs, cap)
    else:
        touched[i] = 1
        if anchored[i] == 0:
            anchor[i] = x[i]
            anchored[i] = 1
        else:
            anchor[i] = anchor[i] + shape_vel[s] * dt
        d = x[i] - anchor[i]
        d = d - n * wp.dot(n, d)
        ft = -ke * d - (kd / (1.0 + kd * dt * w)) * vt
        lt = wp.length(ft)
        if lt > cap:
            ft = ft * (cap / lt)
            anchor[i] = x[i] + ft / ke - n * wp.dot(n, ft / ke)   # the spring alone now gives the cap
        F = F + ft
    wp.atomic_add(f, i, F)


@wp.kernel
def _anchor_release(touched: wp.array[int], anchored: wp.array[int]):
    i = wp.tid()
    if touched[i] == 0:
        anchored[i] = 0
    touched[i] = 0


@wp.kernel
def _drag(scale: float, vel: wp.array[wp.vec3]):
    i = wp.tid()
    vel[i] = vel[i] * scale


# -------------------------------------------------------------------------------------------------------------- sim

class Sim:
    """One scene on one solver. `scene(b, solver)` builds it and returns its metadata ({"ids", "tets", "fixed",
    "plates"}); frames are 1 / fps long; drag c (1/s) scales every velocity by exp(-c dt) (settling only)."""

    def __init__(self, solver, scene, worlds=1, drag=0.0, fps=FPS, gravity=-G, **over):
        cfg = dict(SOLVERS[solver])
        cfg.update({k: v for k, v in over.items() if k in cfg})
        self.solver, self.cfg, self.kind, self.drag, self.fps = solver, cfg, cfg["kind"], drag, fps
        b0 = newton.ModelBuilder(gravity=(0.0, 0.0, gravity * CONTROL.get("g", 1.0)))
        if self.kind == "mpm":
            newton.solvers.SolverImplicitMPM.register_custom_attributes(b0)
        self.meta = scene(b0, solver)
        if worlds > 1:
            b = newton.ModelBuilder(gravity=(0.0, 0.0, gravity * CONTROL.get("g", 1.0)))
            if self.kind == "mpm":
                newton.solvers.SolverImplicitMPM.register_custom_attributes(b)
            b.replicate(b0, worlds, spacing=(1.0, 0.0, 0.0))
        else:
            b = b0
        if self.kind == "vbd":
            b.color()
        model = b.finalize()
        npw = model.particle_count // worlds
        self.npw, self.worlds = npw, worlds
        fixed = np.asarray(self.meta.get("fixed", []), dtype=int)
        if len(fixed):
            fixed = (fixed[None, :] + npw * np.arange(worlds)[:, None]).ravel()
        if len(fixed) and self.kind != "mpm":         # MPM: a zero-mass particle is its kinematic boundary
            for name in ("particle_mass", "particle_inv_mass"):
                a = getattr(model, name).numpy()
                a[fixed] = 0.0
                setattr(model, name, wp.array(a, dtype=wp.float32))
            fl = model.particle_flags.numpy()
            fl[fixed] = fl[fixed] & ~int(newton.ParticleFlags.ACTIVE)
            model.particle_flags = wp.array(fl, dtype=model.particle_flags.dtype)
        self.mu = self.meta.get("mu", 0.5)
        model.soft_contact_ke, model.soft_contact_kd = CONTACT["ke"], CONTACT["kd"]
        model.soft_contact_kf, model.soft_contact_mu = CONTACT["kf"], self.mu + CONTROL.get("mu", 0.0)
        if cfg["kind"] not in ("xpbd", "mpm"):        # (XPBD asserts a particle grid)
            model.particle_grid = None                # no particle-particle contact inside a body
        self.model = model
        self.mass = model.particle_mass.numpy().astype(np.float64)
        if len(self.meta["tets"]):
            self.tets = model.tet_indices.numpy()[: len(self.meta["tets"])]
            self.Dm_inv = model.tet_poses.numpy()[: len(self.tets)].astype(np.float64)
            self.mat = model.tet_materials.numpy()[: len(self.tets)].astype(np.float64)
        else:                                         # MPM: no mesh
            self.tets, self.Dm_inv, self.mat = np.zeros((0, 4), int), np.zeros((0, 3, 3)), np.zeros((0, 3))
        self.x_rest = model.particle_q.numpy()[:npw].astype(np.float64)
        self.eta = self.meta.get("eta", 0.0)
        if self.kind == "semi":
            self.s = newton.solvers.SolverSemiImplicit(model, enable_tri_contact=False)
            self.dt = self._stable_dt()
            n = int(np.ceil(1.0 / (fps * self.dt)))
            ce = cfg["collide_every"]
            n = max(ce, int(np.ceil(n / (2 * ce))) * 2 * ce)  # an even number of contact refreshes per frame
        elif self.kind == "vbd":
            self.s = newton.solvers.SolverVBD(model, iterations=cfg["iters"], friction_epsilon=cfg["feps"],
                                              particle_enable_tile_solve=False)
            n = max(2, int(round(cfg["substeps"] * FPS / fps)))
            n += n % 2
        elif self.kind == "xpbd":
            self.s = newton.solvers.SolverXPBD(model, iterations=cfg["iters"])
            n = max(2, int(round(cfg["substeps"] * FPS / fps)))
            n += n % 2
        else:                                         # implicit MPM: E, nu given directly; damping = a relaxation
            body = b0._gate_body                      # time (s): eta / mu for a Kelvin-Voigt shear viscosity eta
            E, nu = body["E"] * CONTROL.get("E", 1.0), body["nu"] + CONTROL.get("nu", 0.0)
            mu_l, _ = lame(E, nu)
            model.mpm.young_modulus.fill_(E)
            model.mpm.poisson_ratio.fill_(nu)
            model.mpm.damping.fill_(self.meta.get("eta", 0.0) / mu_l)
            model.mpm.tensile_yield_ratio.fill_(1.0)  # elastic in tension too (as Newton's beam example)
            if len(fixed):
                model.particle_mass[wp.array(fixed, dtype=int)].fill_(0.0)
            mc = newton.solvers.SolverImplicitMPM.Config()
            mc.voxel_size, mc.max_iterations, mc.tolerance = cfg["voxel"], cfg["iters"], cfg["tol"]
            mc.warmstart_mode = "particles"
            mc.air_drag = 0.0
            self.s = newton.solvers.SolverImplicitMPM(model, config=mc)
            n = max(2, int(round(cfg["substeps"] * FPS / fps)))
            n += n % 2
        self.n = n
        self.dt = 1.0 / (fps * n)
        self.mass = model.particle_mass.numpy().astype(np.float64)
        self.s0, self.s1, self.ctrl = model.state(), model.state(), model.control()
        if self.kind != "mpm":
            self.pipe = newton.CollisionPipeline(model, soft_contact_gap=0.005)
            self.con = self.pipe.contacts()
        self.shape_tf = model.shape_transform.numpy().copy()
        self.shape_vel = wp.zeros(model.shape_count, dtype=wp.vec3)
        self.anchor = wp.zeros(model.particle_count, dtype=wp.vec3)
        self.anchored = wp.zeros(model.particle_count, dtype=int)
        self.touched = wp.zeros(model.particle_count, dtype=int)
        self.graph, self.t, self.wall = None, 0.0, 0.0
        self.t_plate = None

    def _stable_dt(self):
        """Explicit limits: P-wave across the smallest tet altitude, contact ring, friction gain, viscosity."""
        m = self.mass[self.mass > 0]
        mu_l = 3.0 / 4.0 * self.mat[:, 0]
        lam_l = self.mat[:, 1] - 5.0 / 8.0 * self.mat[:, 0]
        x = self.x_rest
        t = self.tets
        hmin = np.inf
        for o in range(4):                            # altitude of vertex o over its opposite face
            a, b_, c = [t[:, (o + k) % 4] for k in (1, 2, 3)]
            nrm = np.cross(x[b_] - x[a], x[c] - x[a])
            hmin = min(hmin, float(np.min(np.abs(((x[t[:, o]] - x[a]) * nrm).sum(1)) / np.linalg.norm(nrm, axis=1))))
        rho = float(m.sum() / (np.abs(np.linalg.det(np.linalg.inv(self.Dm_inv))).sum() / 6.0)) if len(m) else 1.0
        cp = float(np.sqrt(np.max(lam_l + 2.0 * mu_l) / rho))
        dts = [self.cfg["cfl"] * hmin / cp]
        if self.model.shape_count == 0:
            pass
        elif self.cfg["contact"] == "newton":
            dts += [0.5 * np.sqrt(m.min() / CONTACT["ke"]), m.min() / max(CONTACT["kf"], 1e-12)]
        else:
            dts += [0.5 / np.sqrt(PM["rate"])]
            if self.cfg.get("friction") != "anchor":
                dts.append(1.0 / PM["frate"])
        eta = self.eta if self.cfg["damping"] == "strain" else self.mat[:, 2].max() / 2.0
        if eta > 0:
            dts.append(0.1 * rho * hmin ** 2 / eta)
        self.hmin, self.cp = hmin, cp
        self.dt_limits = [float(d) for d in dts]
        return min(dts) * self.cfg["dt_scale"]

    def move_shapes(self, tf, vel):
        """Prescribed static-shape poses (the plates) for the next frame: {shape: (pos, quat)}, {shape: vel}."""
        for s, (p, q) in tf.items():
            self.shape_tf[s] = np.concatenate([p, q])
        self.model.shape_transform.assign(self.shape_tf.astype(np.float32))
        sv = np.zeros((self.model.shape_count, 3), np.float32)
        for s, u in vel.items():
            sv[s] = u
        self.shape_vel.assign(sv)

    def _substep_block(self, k):
        damp = float(np.exp(-self.drag * self.dt))
        for i in range(k):
            if self.kind == "mpm":
                self.s.step(self.s0, self.s1, None, None, self.dt)
                self.s0, self.s1 = self.s1, self.s0
                if self.drag > 0.0:
                    wp.launch(_drag, dim=self.model.particle_count, inputs=[damp, self.s0.particle_qd])
                continue
            if self.kind != "semi" or i % self.cfg["collide_every"] == 0:
                self.pipe.collide(self.s0, self.con)
            self.s0.clear_forces()
            if self.kind == "semi" and self.cfg["damping"] == "strain" and self.eta > 0:
                wp.launch(_strain_rate_damping, dim=self.model.tet_count,
                          inputs=[self.s0.particle_q, self.s0.particle_qd, self.model.tet_indices, self.model.tet_poses,
                                  self.eta, self.s0.particle_f])
            if self.kind == "semi" and self.cfg["contact"] == "pm":
                c = self.con
                anc = int(self.cfg.get("friction") == "anchor")
                wp.launch(_pm_contact, dim=c.soft_contact_max,
                          inputs=[c.soft_contact_count, c.soft_contact_max, c.soft_contact_particle, c.soft_contact_shape,
                                  c.soft_contact_body_pos, c.soft_contact_normal, self.model.shape_material_mu,
                                  self.shape_vel, self.s0.particle_q, self.s0.particle_qd,
                                  self.model.particle_radius, self.model.particle_flags, self.model.particle_inv_mass,
                                  float(self.model.soft_contact_mu), PM["rate"], PM["frate"], PM["zeta"], self.dt,
                                  anc, self.anchor, self.anchored, self.touched, self.s0.particle_f])
                if anc:
                    wp.launch(_anchor_release, dim=self.model.particle_count, inputs=[self.touched, self.anchored])
                self.s.step(self.s0, self.s1, self.ctrl, None, self.dt)
            else:
                self.s.step(self.s0, self.s1, self.ctrl, self.con, self.dt)
            self.s0, self.s1 = self.s1, self.s0
            if self.drag > 0.0:
                wp.launch(_drag, dim=self.model.particle_count, inputs=[damp, self.s0.particle_qd])

    def frame(self, n=1):
        t0 = time.time()
        for _ in range(n):
            if self.graph is None and self.kind == "mpm":
                self.graph = False                     # sparse grid: no capture
            if self.graph is None:
                self._substep_block(self.n)            # first frame uncaptured (compiles), then captured
                try:
                    with wp.ScopedCapture() as cap:
                        self._substep_block(self.n)
                    self.graph = cap.graph
                except Exception:
                    self.graph = False
                self.t += 1.0 / self.fps                # the captured block is recorded, not run
                continue
            if self.graph is False:
                self._substep_block(self.n)
            else:
                wp.capture_launch(self.graph)
            self.t += 1.0 / self.fps
        wp.synchronize()
        self.wall += time.time() - t0

    def rate(self):
        return self.worlds * self.t / max(self.wall, 1e-9)

    def x(self):
        return self.s0.particle_q.numpy()[:self.npw].astype(np.float64)

    def v(self):
        return self.s0.particle_qd.numpy()[:self.npw].astype(np.float64)

    def info(self):
        return {"substeps_per_s": self.n * self.fps, "dt_s": self.dt, "dt_limits_s": getattr(self, "dt_limits", None),
                "vertices": int(self.npw),
                "tets": int(len(self.tets))}

    def in_bound(self, x=None, bound=1.0):
        x = self.x() if x is None else x
        return bool(np.isfinite(x).all() and np.abs(x - self.x_rest).max() < bound)

    def elastic_energy(self, x):
        """The solver's own elastic energy (J), zero at rest."""
        t, Dm = self.tets, self.Dm_inv
        V = 1.0 / (6.0 * np.linalg.det(Dm))
        Ds = np.stack([x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]], x[t[:, 3]] - x[t[:, 0]]], -1)
        F = Ds @ Dm
        Ic = (F ** 2).sum((1, 2))
        J = np.linalg.det(F)
        km, kl = self.mat[:, 0], self.mat[:, 1]
        if self.kind == "semi":
            a = 1.0 + km / kl - km / (4.0 * kl)
            psi = km / 2 * (Ic - 3) - km / 2 * np.log((Ic + 1) / 4) + kl / 2 * ((J - a) ** 2 - (1 - a) ** 2)
        elif self.kind == "vbd":
            ln = kl + km
            a = 1.0 + km / ln
            psi = km / 2 * (Ic - 3) + ln / 2 * ((J - a) ** 2 - (1 - a) ** 2)
        else:
            psi = km / 2 * (Ic - 3) ** 2 + kl / 2 * (J - 1) ** 2
        return float((V * psi).sum())


def _report(name, solver, row):
    row["pass"] = bool(row.get("pass", False))
    return row


# --------------------------------------------------------------------------------------------------------- checks

def check_hang(solver, **over):
    out = {}
    for E, rho in ((1.0e3, 30.0), (1.0e4, 60.0), (1.0e6, 1100.0)):
        body = dict(FOAM, E=E, rho=rho, nu=0.0)
        L = 0.10

        def scene(b, solver):
            ids, t = add_body(b, solver, body, (0.02, 0.02, L), (-0.01, -0.01, 0.2))
            z = np.array([b.particle_q[i][2] for i in ids])
            return {"ids": ids, "tets": t, "fixed": ids[z > z.max() - 1e-6]}
        sim = Sim(solver, scene, drag=6.0, **over)
        x0 = sim.x()
        sim.frame(int(3.0 * FPS))
        xs = []
        for _ in range(FPS):
            sim.frame()
            xs.append(sim.x())
        x = np.mean(xs, axis=0)
        bottom = x0[:, 2] < x0[:, 2].min() + 1e-6
        ext = float((x0[bottom, 2] - x[bottom, 2]).mean())
        want = body["rho"] * G * L ** 2 / (2.0 * E)
        ok = sim.in_bound(x, 0.5)
        row = {"E": E, "rho": rho, "extension_mm": 1e3 * ext, "closed_form_mm": 1e3 * want, "ratio": ext / want,
               "in_bound": ok, **sim.info(), "sim_s_per_wall_s": sim.rate(),
               "pass": bool(ok and abs(ext / want - 1.0) < 0.05)}
        out[f"E{E:g}"] = row
        print(f"  hang     {solver:8s} E {E:7.0e} rho {rho:6.0f}: extension {1e3 * ext:9.5f} mm, closed form "
              f"{1e3 * want:9.5f} mm (ratio {ext / want:7.4f}) [{sim.n * FPS} substeps/s, {sim.rate():.2f} sim-s/wall-s]",
              flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def _bar_scene(body, L, h, strain):
    """Bar along z, ends clamped; the top face displaced by strain x L."""
    def scene(b, solver):
        ids, t = add_body(b, solver, body, (0.02, 0.02, L), (-0.01, -0.01, 0.1), h=h)
        z = np.array([b.particle_q[i][2] for i in ids])
        top = ids[z > z.max() - 1e-6]
        for i in top:
            p = b.particle_q[i]
            b.particle_q[i] = wp.vec3(p[0], p[1], p[2] + strain * L)
        bot = ids[z < z.min() + 1e-6]
        return {"ids": ids, "tets": t, "fixed": np.concatenate([bot, top])}
    return scene


def check_poisson(solver, **over):
    out = {}
    L, h = 0.16, 0.005
    for nu in (0.0, 0.3, 0.45):
        for eps in (0.01, -0.01):
            body = dict(FOAM, nu=nu)
            sim = Sim(solver, _bar_scene(body, L, h, eps), drag=6.0, gravity=0.0, **over)
            sim.frame(int(2.0 * FPS))
            xs = []
            for _ in range(FPS // 2):
                sim.frame()
                xs.append(sim.x())
            x = np.mean(xs, axis=0)
            x0 = sim.x_rest.copy()
            z0 = x0[:, 2] - x0[:, 2].min()
            x0z = x0.copy()                              # the rest bar (before the top face was moved)
            top = z0 > z0.max() - 1e-6
            x0z[top, 2] -= eps * L
            z0 = x0z[:, 2] - x0z[:, 2].min()
            mid = np.abs(z0 - L / 2) < 1e-6
            lat = []
            for ax in (0, 1):
                hi, lo = x0z[mid, ax] > x0z[mid, ax].max() - 1e-6, x0z[mid, ax] < x0z[mid, ax].min() + 1e-6
                w0 = x0z[mid][hi, ax].mean() - x0z[mid][lo, ax].mean()
                w1 = x[mid][hi, ax].mean() - x[mid][lo, ax].mean()
                lat.append(w1 / w0 - 1.0)
            lat = float(np.mean(lat))
            za, zb = np.abs(z0 - (L / 2 - 0.03)) < 1e-6, np.abs(z0 - (L / 2 + 0.03)) < 1e-6
            ax_strain = float((x[zb, 2].mean() - x[za, 2].mean()) / 0.06 - 1.0)
            ratio = -lat / ax_strain
            # volume of the tets between the two sections
            tc = x0z[sim.tets].mean(1)[:, 2] - x0z[:, 2].min()
            sel = np.abs(tc - L / 2) < 0.03
            vol0 = np.abs(tet_J(x0z, sim.tets[sel], np.eye(3)[None])).sum()
            vol1 = np.abs(tet_J(x, sim.tets[sel], np.eye(3)[None])).sum()
            dv = float(vol1 / vol0 - 1.0)
            want_dv = (1.0 - 2.0 * nu) * ax_strain
            Jmin = float(tet_J(sim.x(), sim.tets, sim.Dm_inv).min())
            ok = sim.in_bound(x, 0.5) and not Jmin <= 0
            row = {"nu": nu, "axial_strain": ax_strain, "lateral_strain": lat, "poisson_measured": ratio,
                   "volume_change": dv, "linear_volume_change": want_dv, "volume_ratio": dv / want_dv if want_dv else None,
                   "min_J": Jmin, "in_bound": ok, **sim.info()}
            p_ok = abs(ratio - nu) < 0.01 if nu == 0 else abs(ratio / nu - 1.0) < 0.05
            v_ok = abs(dv / want_dv - 1.0) < 0.10 if nu == 0.45 else True
            row["pass"] = bool(ok and p_ok and v_ok)
            out[f"nu{nu}_eps{eps:+g}"] = row
            print(f"  poisson  {solver:8s} nu {nu:4.2f} eps {eps:+.3f}: axial {ax_strain:+.5f}, lateral {lat:+.6f}, "
                  f"nu measured {ratio:6.4f}; dV/V {dv:+.6f} vs (1-2nu) eps {want_dv:+.6f}; min J {Jmin:.4f}",
                  flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


RUBBER = dict(FOAM, rho=1100.0, E=3.0e6, nu=0.3)


def _beam_scene(body, L, t, h):
    def scene(b, solver):
        ids, tt = add_body(b, solver, body, (L + h, t, t), (-h, -t / 2, 0.3 - t / 2), h=h)
        x = np.array([b.particle_q[i] for i in ids])
        return {"ids": ids, "tets": tt, "fixed": ids[x[:, 0] < x[:, 0].min() + 1e-6],
                "tip": ids[x[:, 0] > x[:, 0].max() - 1e-6]}
    return scene


def beam_closed_form(body, L, t):
    E, nu, rho = body["E"], body["nu"], body["rho"]
    A, I = t * t, t ** 4 / 12.0
    q = rho * G * A
    Gs = E / (2 * (1 + nu))
    k = 10 * (1 + nu) / (12 + 11 * nu)
    eb = q * L ** 4 / (8 * E * I)
    return eb, eb + q * L ** 2 / (2 * k * Gs * A)


def check_cantilever(solver, **over):
    """Clamped by one layer of cells behind x = 0 (both faces of that layer pinned would over-constrain: only the
    back face is pinned, so the clamp is one cell of rubber -- the free length L is measured from x = 0)."""
    out = {}
    L, t = 0.15, 0.02
    cases = [("rubber_h1cm", RUBBER, 0.01), ("rubber_h0.5cm", RUBBER, 0.005), ("rubber_h0.25cm", RUBBER, 0.0025),
             ("foam_E1e5_h0.5cm", dict(FOAM, E=1.0e5), 0.005), ("rubber_nu0.45_h0.5cm", dict(RUBBER, nu=0.45), 0.005)]
    for tag, body, h in cases:
        if SOLVERS[solver]["kind"] == "mpm" and h != 0.005:
            continue                                  # MPM's resolution is its voxel, not h
        sim = Sim(solver, _beam_scene(body, L, t, h), drag=8.0, **over)
        tip = sim.meta["tip"]
        z0 = sim.x()[tip, 2].mean()
        sim.frame(int(4.0 * FPS))
        zs = []
        for _ in range(FPS // 2):
            sim.frame()
            zs.append(sim.x()[tip, 2].mean())
        d = float(z0 - np.mean(zs))
        eb, timo = beam_closed_form(body, L + h, t)  # the clamp is the pinned back face: the free length is L + h
        ok = sim.in_bound(None, 0.5)
        row = {"h_m": h, "tip_deflection_mm": 1e3 * d, "euler_bernoulli_mm": 1e3 * eb, "timoshenko_mm": 1e3 * timo,
               "ratio": d / timo, "in_bound": ok, **sim.info(), "sim_s_per_wall_s": sim.rate(),
               "pass": bool(ok and abs(d / timo - 1.0) < 0.10)}
        out[tag] = row
        print(f"  cantil.  {solver:8s} {tag:17s}: tip {1e3 * d:8.4f} mm, Timoshenko {1e3 * timo:8.4f} (EB {1e3 * eb:8.4f}) "
              f"ratio {d / timo:6.3f} [{sim.n * FPS} substeps/s, {sim.rate():.2f} sim-s/wall-s]", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def _free_vibration(solver, body, L, t, h, eta, seconds, fps=300, **over):
    sc = _beam_scene(dict(body, eta=eta), L, t, h)

    def scene(b, solver):
        m = sc(b, solver)
        m["eta"] = eta
        return m
    sim = Sim(solver, scene, fps=fps, **over)
    m = sim.mass[:sim.npw]
    tip = sim.meta["tip"]
    x = sim.x()
    pe0 = float((m * G * x[:, 2]).sum())
    ts, zs, series = [], [], []
    worst, released = -np.inf, 0.0
    for f in range(int(seconds * fps)):
        sim.frame()
        x, v = sim.x(), sim.v()
        ts.append(sim.t)
        zs.append(x[tip, 2].mean())
        if f % 3 == 0:
            pe = float((m * G * x[:, 2]).sum()) - pe0
            ke = 0.5 * float((m * (v ** 2).sum(1)).sum())
            ee = sim.elastic_energy(x)
            worst = max(worst, ke + pe + ee)
            released = max(released, -pe)
            series.append([round(sim.t, 4), ke, pe, ee])
    return sim, np.array(ts), np.array(zs), worst, released, series


def zero_cross_freq(ts, zs):
    z = zs - zs.mean()
    s = np.sign(z)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    tc = ts[idx] - z[idx] * (ts[idx + 1] - ts[idx]) / (z[idx + 1] - z[idx])
    if len(tc) < 3:
        return float("nan")
    return float((len(tc) - 1) / 2.0 / (tc[-1] - tc[0]))


def check_vibration(solver, **over):
    out = {}
    L, t = 0.15, 0.02
    for tag, h in (("h1cm", 0.01), ("h0.5cm", 0.005), ("h0.25cm", 0.0025)):
        if SOLVERS[solver]["kind"] == "mpm" and h != 0.01:
            continue                                  # MPM's resolution is its voxel, not h
        body = RUBBER
        sim, ts, zs, worst, released, _ = _free_vibration(solver, body, L, t, h, 0.0, 1.5, **over)
        f = zero_cross_freq(ts, zs)
        E, rho = body["E"] * 1.0, body["rho"]
        Lf = L + h
        want = 1.875 ** 2 / (2 * np.pi) * np.sqrt(E * t ** 2 / 12.0 / rho) / Lf ** 2
        amp = zs - zs.mean()
        n = len(amp) // 3
        decay = float(np.abs(amp[-n:]).max() / max(np.abs(amp[:n]).max(), 1e-12))
        ok = sim.in_bound(None, 0.5)
        row = {"h_m": h, "freq_hz": f, "euler_bernoulli_hz": float(want), "ratio": f / want,
               "amplitude_last_third_over_first": decay, "in_bound": ok, **sim.info(),
               "pass": bool(ok and abs(f / want - 1.0) < 0.05)}
        out[tag] = row
        print(f"  vibr.    {solver:8s} {tag:7s}: first mode {f:7.3f} Hz, closed form {want:7.3f} Hz (ratio {f / want:6.3f}),"
              f" amplitude kept {decay:.3f} over 1.5 s", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_energy(solver, **over):
    out = {}
    L, t, h = 0.15, 0.02, 0.01
    sim, ts, zs, worst, released, series = _free_vibration(solver, RUBBER, L, t, h, 0.0, 2.0, fps=150, **over)
    ok = sim.in_bound(None, 0.5)
    out["undamped"] = {"energy_gain_J": worst, "energy_released_J": released,
                       "gain_fraction": worst / max(released, 1e-12), "series_t_ke_dpe_elastic": series[::5],
                       "in_bound": ok, "pass": bool(ok and worst / max(released, 1e-12) < 0.01)}
    print(f"  energy   {solver:8s} undamped: max (KE+PE+elastic) gain {worst:+.3e} J of {released:.3e} J released "
          f"({100 * worst / max(released, 1e-12):+.2f}%)", flush=True)
    # damped: Kelvin-Voigt eta = 2 zeta E / omega_1 for zeta = 5% on the first mode
    w1 = 2 * np.pi * 1.875 ** 2 / (2 * np.pi) * np.sqrt(RUBBER["E"] * t ** 2 / 12 / RUBBER["rho"]) / (L + h) ** 2
    eta = 2 * 0.05 * RUBBER["E"] / w1
    sc = _beam_scene(dict(RUBBER, eta=eta), L, t, h)

    def scene(b, solver):
        m = sc(b, solver)
        m["eta"] = eta
        return m
    sim = Sim(solver, scene, **over)
    sim.frame(4 * FPS)
    xa = sim.x()
    sim.frame(FPS)
    move = float(np.linalg.norm(sim.x() - xa, axis=1).max())
    ok = sim.in_bound(None, 0.5)
    out["damped"] = {"eta_Pa_s": eta, "max_move_5th_second_mm": 1e3 * move, "in_bound": ok, **sim.info(),
                     "pass": bool(ok and move < 5e-4)}
    print(f"  energy   {solver:8s} eta {eta:.0f} Pa s: moved {1e3 * move:.4f} mm in the 5th second", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_spin(solver, **over):
    eta = 1.0                                         # Pa s on foam E 1e4: zeta ~ 5% on its first mode
    body = dict(FOAM, eta=eta)
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    w = 2 * np.pi * axis

    def scene(b, solver):
        ids, t = add_body(b, solver, body, (0.05, 0.05, 0.05), (-0.025, -0.025, 0.3 - 0.025))
        return {"ids": ids, "tets": t, "eta": eta}
    sim = Sim(solver, scene, gravity=0.0, **over)
    x = sim.x()
    c = (sim.mass[:, None] * x).sum(0) / sim.mass.sum()
    v = np.cross(w[None], x - c)
    sim.s0.particle_qd.assign(v.astype(np.float32))
    sim.s1.particle_qd.assign(v.astype(np.float32))
    if sim.kind == "vbd":                             # VBD reads velocity from the previous position
        pass
    m = sim.mass

    def ke_L():
        x, v = sim.x(), sim.v()
        c = (m[:, None] * x).sum(0) / m.sum()
        return 0.5 * float((m * (v ** 2).sum(1)).sum()), (m[:, None] * np.cross(x - c, v)).sum(0)
    k0, L0 = ke_L()
    sim.frame(FPS)
    k1, L1 = ke_L()
    dk = k1 / k0 - 1.0
    dL = float(np.linalg.norm(L1 - L0) / np.linalg.norm(L0))
    ok = sim.in_bound(None, 1.0)
    out = {"ke_change": dk, "angular_momentum_change": dL, "in_bound": ok, **sim.info(),
           "pass": bool(ok and abs(dk) < 0.01 and dL < 0.01)}
    print(f"  spin     {solver:8s}: KE {100 * dk:+.3f}%, angular momentum {100 * dL:.3f}% after 1 s at 1 rev/s "
          f"(eta {eta} Pa s)", flush=True)
    return out


def check_freefall(solver, **over):
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

    def scene(b, solver):
        ids, t = add_body(b, solver, FOAM, (0.05, 0.05, 0.05), (0.0, 0.0, 2.0),
                          wp.quat_from_axis_angle(wp.vec3(*axis), 0.4))
        return {"ids": ids, "tets": t}
    sim = Sim(solver, scene, **over)
    m = sim.mass[:sim.npw]
    x0 = sim.x()
    d0 = np.linalg.norm(x0[:, None] - x0[None], axis=-1)
    ts, zs, shape = [], [], 0.0
    for f in range(15):
        sim.frame()
        x = sim.x()
        ts.append(sim.t)
        zs.append(float((m * x[:, 2]).sum() / m.sum()))
        shape = max(shape, float(np.abs(np.linalg.norm(x[:, None] - x[None], axis=-1) - d0).max()))
    a = -2.0 * np.polyfit(np.array(ts), np.array(zs), 2)[0]
    out = {"accel_m_s2": float(a), "rel_error": float(a / G - 1.0), "shape_change_mm": 1e3 * shape,
           "pass": bool(abs(a / G - 1.0) < 1e-3 and shape < 1e-4)}
    print(f"  freefall {solver:8s}: centroid accel {a:.4f} m/s2 ({100 * (a / G - 1):+.3f}% of g), "
          f"shape change {1e3 * shape:.4f} mm", flush=True)
    return out


def check_slide(solver, theta_deg=20.0, **over):
    """A 4 x 4 x 2 cm block at mu 0.2 (slides) and 0.5 (holds), and a 6 x 6 x 10 cm one -- a plush toy's height,
    eleven vertex layers carried by the bottom one -- at mu 0.5 (holds: w / h = 0.6 > tan 20, no tipping). Foam with
    eta 1 Pa s: undamped, the tall block keeps rocking from its release and walks down by lift-offs (17 mm in 4 s)."""
    th = np.radians(theta_deg)
    body = dict(FOAM, eta=1.0)
    out = {}
    for tag, mu, dims in (("mu0.2", 0.2, (0.04, 0.04, 0.02)), ("mu0.5", 0.5, (0.04, 0.04, 0.02)),
                          ("mu0.5_tall", 0.5, (0.06, 0.06, 0.10))):
        def scene(b, solver, mu=mu, dims=dims):
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-th))
            R = np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)
            static_box(b, (0.0, 0.0, 0.5), (3.0, 0.5, 0.05), mu, q)
            top = np.array([0.0, 0.0, 0.5]) + R @ np.array([0.0, 0.0, 0.05 + FOAM["r"]])
            ids, t = add_body(b, solver, body, dims, tuple(top + R @ np.array([-dims[0] / 2, -dims[1] / 2, 0.0])), q)
            return {"ids": ids, "tets": t, "down": -R[:, 0], "mu": mu, "eta": body["eta"]}
        sim = Sim(solver, scene, **over)
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
        ok = sim.in_bound(None, 3.0)
        if want > 0:
            w = (ts > 0.3) & (ts < 1.3)
            a = 2.0 * np.polyfit(ts[w], ss[w], 2)[0]
            row = {"accel_m_s2": float(a), "slide_law_m_s2": float(want), "rel_error": float(a / want - 1.0),
                   "in_bound": ok, "pass": bool(ok and abs(a / want - 1.0) < 0.02)}
            print(f"  slide    {solver:8s} {tag:10s}: accel {a:.4f} m/s2, law {want:.4f} ({100 * (a / want - 1):+.2f}%)",
                  flush=True)
        else:
            w = ts > 1.0
            drift = float(np.polyfit(ts[w], ss[w], 1)[0])
            row = {"drift_mm_s": 1e3 * drift, "travel_mm": 1e3 * float(ss[-1]), "in_bound": ok,
                   "pass": bool(ok and abs(drift) < 1e-3)}
            print(f"  slide    {solver:8s} {tag:10s}: drift {1e3 * drift:+.4f} mm/s over 1-2 s, travel "
                  f"{1e3 * ss[-1]:+.3f} mm", flush=True)
        row.update(sim.info())
        out[tag] = row
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


TOP = 0.10


def check_rest(solver, **over):
    H = 0.10
    eta = 1.0                                         # Pa s: ~5% of critical on the cube's first mode
    body = dict(FOAM, E=1.0e3, nu=0.0, eta=eta)

    def scene(b, solver):
        static_box(b, (0.0, 0.0, TOP / 2), (0.5, 0.5, TOP / 2), FOAM["mu"])
        ids, t = add_body(b, solver, body, (H, H, H), (-H / 2, -H / 2, TOP + body["r"] + 0.01))
        return {"ids": ids, "tets": t, "eta": eta, "mu": FOAM["mu"]}
    sim = Sim(solver, scene, **over)
    x0 = sim.x_rest
    top0 = x0[:, 2] > x0[:, 2].max() - 1e-6
    bot0 = x0[:, 2] < x0[:, 2].min() + 1e-6
    sim.frame(3 * FPS)
    xa = sim.x()
    sim.frame(FPS)
    xb = sim.x()
    move = float(np.linalg.norm(xb - xa, axis=1).max())
    m = sim.mass[:sim.npw]
    creep = float(np.linalg.norm(((m[:, None] * (xb - xa)).sum(0) / m.sum())[:2]))
    pen = float((xb[:, 2] - TOP - body["r"]).min())
    comp = float(H - (xb[top0, 2].mean() - xb[bot0, 2].mean()))
    want = body["rho"] * G * H ** 2 / (2 * body["E"])
    Jmin = jmin(sim, xb)
    ok = sim.in_bound(xb, 0.5)
    out = {"max_move_4th_second_mm": 1e3 * move, "creep_mm_s": 1e3 * creep, "penetration_min_mm": 1e3 * pen,
           "compression_mm": 1e3 * comp, "closed_form_mm": 1e3 * want, "ratio": comp / want, "min_J": Jmin,
           "in_bound": ok, **sim.info(), "sim_s_per_wall_s": sim.rate(),
           "pass": bool(ok and move < 1e-3 and creep < 1e-3 and pen > -1e-3 and abs(comp / want - 1) < 0.10)}
    print(f"  rest     {solver:8s}: moved {1e3 * move:.4f} mm in the 4th s, creep {1e3 * creep:.4f} mm/s, lowest "
          f"{1e3 * pen:+.4f} mm, compression {1e3 * comp:.4f} mm vs rho g H^2/2E {1e3 * want:.4f} (ratio "
          f"{comp / want:.3f}), min J {Jmin:.3f}", flush=True)
    return out


def _squeeze_scene(body, W):
    def scene(b, solver):
        static_box(b, (0.0, 0.0, TOP / 2), (0.5, 0.5, TOP / 2), FOAM["mu"])
        ids, t = add_body(b, solver, body, (W, W, W), (-W / 2, -W / 2, TOP + body["r"]))
        pl = [static_box(b, (s * (W / 2 + body["r"] + 0.01 + 0.002), 0.0, TOP + W / 2 + 0.01), (0.01, 0.05, W / 2 + 0.01),
                         0.5) for s in (-1.0, 1.0)]       # plates from the table top to 1 cm over the cube
        return {"ids": ids, "tets": t, "plates": pl, "eta": body["eta"], "mu": FOAM["mu"]}
    return scene


def squeeze_run(sim, W, record=True):
    """Plates close from 1 mm off the faces to 50% of the width in 1 s, hold 0.5 s, open in 1 s, wait 1 s."""
    r = FOAM["r"]
    gap0 = W / 2 + r + 0.001
    gap1 = W / 4 + r
    pl = sim.meta["plates"]
    z = TOP + W / 2 + 0.01
    q = np.array([0.0, 0.0, 0.0, 1.0])
    Jmin, bound, xs = np.inf, True, []
    nf = [int(1.0 * FPS), int(0.5 * FPS), int(1.0 * FPS), int(1.0 * FPS)]
    for phase, n in enumerate(nf):
        for k in range(n):
            if phase == 0:
                g, u = gap0 + (gap1 - gap0) * (k + 1) / n, (gap1 - gap0) / 1.0
            elif phase == 1:
                g, u = gap1, 0.0
            elif phase == 2:
                g, u = gap1 + (gap0 + 0.02 - gap1) * (k + 1) / n, (gap0 + 0.02 - gap1) / 1.0
            else:
                g, u = gap0 + 0.02, 0.0
            sim.move_shapes({pl[0]: (np.array([-(g + 0.01), 0.0, z]), q), pl[1]: (np.array([g + 0.01, 0.0, z]), q)},
                            {pl[0]: np.array([-u, 0, 0]), pl[1]: np.array([u, 0, 0])})
            sim.frame()
            if record:
                x = sim.x()
                Jmin = min(Jmin, jmin(sim, x))
                bound = bound and sim.in_bound(x, 0.3)
                xs.append(x)
    return Jmin, bound, xs


def procrustes_err(x, x0):
    a, b = x - x.mean(0), x0 - x0.mean(0)
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return float(np.linalg.norm(a - b @ R, axis=1).max())


def check_squeeze(solver, **over):
    W = 0.05
    body = dict(FOAM, eta=1.0)
    sim = Sim(solver, _squeeze_scene(body, W), **over)
    Jmin, bound, xs = squeeze_run(sim, W)
    x = xs[-1]
    nf = int(1.5 * FPS)
    x0 = sim.x_rest
    mid = np.abs(x0[:, 2] - x0[:, 2].mean()) < 0.006          # the two mid-height layers
    width_min = min(float(np.ptp(xx[mid, 0])) for xx in xs[: nf])
    err = procrustes_err(x, sim.x_rest)
    out = {"min_J": Jmin, "in_bound": bound, "narrowest_width_mm": 1e3 * width_min,
           "shape_error_after_release_mm": 1e3 * err, **sim.info(), "sim_s_per_wall_s": sim.rate(),
           "pass": bool(bound and not Jmin <= 0 and err < 1e-3 and width_min < 0.6 * W)}
    print(f"  squeeze  {solver:8s}: narrowest {1e3 * width_min:.1f} mm of {1e3 * W:.0f}, min J {Jmin:.3f}, in bound "
          f"{bound}, shape error 1 s after release {1e3 * err:.3f} mm [{sim.rate():.2f} sim-s/wall-s]", flush=True)
    return out


def check_throughput(solver, worlds=64, **over):
    W = 0.05
    body = dict(FOAM, eta=1.0)
    sim = Sim(solver, _squeeze_scene(body, W), worlds=worlds, **over)
    sim.frame(2)
    sim.wall, sim.t = 0.0, 0.0
    Jmin, bound, _ = squeeze_run(sim, W, record=False)
    x = sim.s0.particle_q.numpy()
    ok = bool(np.isfinite(x).all() and np.abs(x).max() < worlds * 1.0 + 2.0)
    rate = sim.rate()
    out = {"worlds": worlds, "vertices_per_world": int(sim.npw), "sim_seconds_per_wall_second": rate,
           "four_second_episodes_per_hour": 3600.0 * rate / 4.0, "in_bound": ok, **sim.info()}
    print(f"  throughput {solver:6s} x{worlds}: {rate:.2f} sim-s/wall-s = {out['four_second_episodes_per_hour']:.0f} "
          f"four-second episodes per hour ({sim.npw} vertices each)", flush=True)
    return out


CHECKS = {"hang": check_hang, "poisson": check_poisson, "cantilever": check_cantilever, "vibration": check_vibration,
          "energy": check_energy, "spin": check_spin, "freefall": check_freefall, "slide": check_slide,
          "rest": check_rest, "squeeze": check_squeeze, "throughput": check_throughput}

# the wrong body (or step) each check must catch, run on a passing solver with --control 1; the closed forms see none
CONTROLS = {"hang": [{"E": 1.1}], "poisson": [{"nu": 0.03}], "cantilever": [{"E": 1.15}, {"raw": 1}],
            "vibration": [{"rho": 1.15}], "freefall": [{"g": 1.002}], "slide": [{"mu": 0.01}], "rest": [{"E": 1.15}],
            "energy": [{"_over": {"dt_scale": 4.0}}], "squeeze": [{"_over": {"dt_scale": 4.0}}]}


def main():
    args = sys.argv[1:]
    solvers = list(SOLVERS)
    over, names, control = {}, [], 0
    i = 0
    while i < len(args):
        if args[i] == "--solver":
            solvers = args[i + 1].split(",")
            i += 2
        elif args[i] == "--control":
            control = int(args[i + 1])
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
    tag = "" if not over else "_" + "_".join(f"{k}{v:g}" for k, v in over.items())

    def save(key, n, row):                            # merge: several gate processes may share the file
        res = json.loads(OUT.read_text()) if OUT.exists() else {}
        res.setdefault(key, {})[n] = row
        res["_versions"] = {"newton": newton.__version__, "warp": wp.config.version}
        tmp = OUT.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(OUT)
    for s in solvers:
        key = s + tag + ("_control" if control else "")
        for n in names:
            runs = [(n, None, over)]
            if control:
                runs = [(f"{n}_{k}", c, dict(over, **c.get("_over", {}))) for k, c in enumerate(CONTROLS.get(n, []))]
            for name, c, ov in runs:
                CONTROL.clear()
                if c:
                    CONTROL.update({k: v for k, v in c.items() if k != "_over"})
                    print(f"  [control: {c}]", flush=True)
                t0 = time.time()
                try:
                    row = CHECKS[n](s, **ov)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    row = {"error": repr(exc)}
                    print(f"  {n} {s}: ERROR {exc!r}", flush=True)
                if c:
                    row["control"] = c
                row["wall_s"] = time.time() - t0
                save(key, name, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
