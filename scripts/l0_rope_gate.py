"""L0 -- rope teacher gate (T0-rope): which Newton rope/cable solver, if any, is fit to distil from?

A rope dataset inherits every defect of the solver that generated it, so the rope teacher is checked against
closed forms and invariants before any rope episode is generated (CLAUDE.md rule 2), as the cloth gate did
(`scripts/l0_cloth_gate.py`: VBD and Style3D ran stably and simulated the wrong cloth). The rope is given in
physical units -- linear density lam (kg/m), axial stiffness EA (N), bending stiffness EI (N m^2), friction mu,
radius r -- and each solver gets the same rope through its own parameters:

  vbd    Newton's cable: capsule bodies joined by ROD joints (`ModelBuilder.add_rod` with a `newton.Rod` carrying
         the section rigidities EA, kGA = EA, EI, GJ = EI / 1.3; Newton derives each joint's stiffness as
         R / L_dual), SolverVBD with compliant ALM, as `newton/examples/cable` run it (10 substeps x 5
         iterations at 60 fps = 20 x 5 at 30 fps); segment masses set to lam x length (Newton's capsules weigh
         their hemispherical caps too: +53% at r 4 mm, l 1 cm); capsule-capsule self-contact is Newton's
  semi   a chain of particles (one per node, mass lam x half of each adjacent segment) under SolverSemiImplicit:
         a spring per segment (ke = EA / l, axial damping 1e-3 N s), and ours: bending E = (EI / l_dual) (1 -
         t0 . t1) at every interior node (= EI theta^2 / 2 l_dual), segment-segment self-contact (capsules of
         radius r) and contact with surfaces with the cloth teacher's per-mass law (`lsim.teacher.cloth_teacher`:
         critically damped 2e7 / s^2 penalty, viscous friction 2e4 / s below the Coulomb cap); dt = sqrt(m / k) /
         2 for the stiffest of stretch, bending and contact
  xpbd   the same particle chain under SolverXPBD: springs as its distance constraints, its own particle-shape
         contact, our bending and self-contact as external forces (it has no rod bending), 100 substeps x 4
         iterations (explicit bending and contact need dt < ~1e-3 s)

Default rope: lam 0.02 kg/m (an 8 mm cord), EA 1e3 N, EI 1e-4 N m^2 (bending length (EI / lam g)^(1/3) = 8 cm),
r 4 mm, segments of 1 cm, mu 0.5. Checks (closed form, tolerance):

  stretch    1 m rope hanging from its top: extension of the free length L against lam g L^2 / (2 EA) at EA =
             20 / 1e3 / 1e4 N (5%)
  catenary   a string (EI 1e-6) of 1 m pinned at both ends 0.7 m apart, released from a V: sag against the
             catenary of that length (2%), node shape RMS against the catenary (3% of the sag)
  bend       heavy-elastica cantilever: clamped horizontal over 3 cm, overhang l = 2 (EI / lam g)^(1/3): tip drop
             against the elastica (10%), unchanged at half the segment length and at 8 EI with twice the overhang
  swing      0.5 m rope pinned at one end, released horizontal, no drag: KE + PE never above its start (1% of
             the energy released); with drag 2/s: no node moves 5 mm in the 6th second
  freefall   a tilted straight rope falling 0.5 s: centroid acceleration g within 0.1%, node distances kept (0.1 mm)
  slide      20 cm rope along a 20 deg plane: a = g (sin - mu cos) within 2% at mu 0.2; drift < 1 mm/s at mu 0.5
  drop       0.5 m wavy rope dropped 2 cm onto a table: no creep in the 4th second (< 1 mm/s), no penetration
             (> -1 mm)
  cross      the crossing counter's control: rope A dropped across rope B (pinned taut at both ends). Without
             contact between them the counter must register A passing through B (> 0); with it, 0 crossings and
             A hangs over B (its lowest node less than its 0.4 m length below B)
  pile       2 m rope dropped as a tilted coil onto a table: 0 crossings, centerlines never closer than r, in bound;
             (semi) the same without self-contact must register crossings
  gripper    0.5 m rope on the table lifted by one end (0.3 m/s, 2.5 s), dragged sideways (0.3 m/s, 1 s), held 4 s:
             inside the spatial bound at every frame (3 m, 20 m/s), hangs under the gripper (bottom within 2 cm
             horizontally, hanging length within 2% of L); (particles) gripper force / weight within 2%
  stability  (semi) the swing at EA 1e3 / 1e4 with dt = sqrt(m / k) / spp: smallest stable steps per sqrt(m/k)
  throughput 64 worlds of the drop, self-contact on: simulated seconds per wall second

A crossing is a pair of non-adjacent segments whose lines' triple product det[b - a, d - c, c - a] changes sign
between frames while both line-closest points lie inside both segments, the segments are within 5 cm and at
more than 12 deg to each other at both frames: their centerlines passed through each other.
Divergence is detected by a spatial bound (CLAUDE.md rule 3), never by isfinite alone.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l0_rope_gate.py [check ...] [--solver vbd,semi,xpbd] [--key value ...]
    $L scripts/l0_rope_gate.py --table          # print the table from docs/results/l0_rope_gate.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l0_rope_gate.json"

if "--table" not in sys.argv and "CUDA_VISIBLE_DEVICES" not in os.environ:   # pick an idle card before CUDA starts
    from lsim.gpu import wait_for_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(wait_for_gpu(min_free_gib=8.0, max_util=20))

G = 9.81
FPS = 30
ROPE = dict(lam=0.02, EA=1.0e3, EI=1.0e-4, mu=0.5, r=0.004, l=0.01)
SOLVERS = {"vbd": dict(substeps=20, iters=5),         # Newton's cable examples: 10 x 5 at 60 fps
           "semi": dict(spp=2.0, ce=50),              # dt = sqrt(m / k) / spp; contacts refreshed every ce substeps
           "xpbd": dict(substeps=100, iters=4, ce=10)}
VBD_KE = 1.0e5            # N/m: VBD contact stiffness (Newton's cable examples)
SEED = 0
TOP = 0.75                # table top (m)


def main_table():
    res = json.loads(OUT.read_text())
    for key, checks in res.items():
        if key.startswith("_"):
            continue
        print(f"== {key}")
        for name, r in checks.items():
            print(f"  {name:10s} pass={r.get('pass')}  {r.get('summary', r.get('error', ''))}")
    return 0


if "--table" in sys.argv:
    sys.exit(main_table())

import warp as wp                                     # noqa: E402
import newton                                         # noqa: E402
from lsim.teacher.cloth_teacher import (_normal_force, _shape_contact, CONTACT_RATE, CONTACT_ZETA,  # noqa: E402,F401
                                        FRICTION_RATE)
# the rope's own physics and the crossing counter live in the generator's teacher (moved there unchanged, 2026-10-30)
from lsim.teacher.rope_teacher import (AXIAL_ETA, SELF_MU, CrossingCounter, _bend, _seg_candidates,  # noqa: E402,F401
                                       _seg_contact, seg_geometry, seg_pairs, straight)

wp.config.quiet = True


# ----------------------------------------------------------------------------------------------- particle kernels
@wp.kernel
def _drive_particles(ids: wp.array[int], x0: wp.array[wp.vec3], vel: wp.array[wp.vec3], clock: wp.array[float],
                     x: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    """Gripped nodes (massless) follow the gripper: x = x0 + vel t."""
    k = wp.tid()
    i = ids[k]
    x[i] = x0[k] + vel[k] * clock[0]
    v[i] = vel[k]


@wp.kernel
def _drive_bodies(ids: wp.array[int], q0: wp.array[wp.transform], vel: wp.array[wp.vec3], clock: wp.array[float],
                  q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector]):
    """Gripped (kinematic) segments follow the gripper: translate at vel, no rotation."""
    k = wp.tid()
    i = ids[k]
    X = q0[k]
    q[i] = wp.transform(wp.transform_get_translation(X) + vel[k] * clock[0], wp.transform_get_rotation(X))
    qd[i] = wp.spatial_vector(vel[k], wp.vec3(0.0))


@wp.kernel
def _damp_vec(v: wp.array[wp.vec3], s: float):
    i = wp.tid()
    v[i] = v[i] * s


@wp.kernel
def _damp_body(v: wp.array[wp.spatial_vector], s: float):
    i = wp.tid()
    v[i] = v[i] * s


@wp.kernel
def _acc_force(ids: wp.array[int], f: wp.array[wp.vec3], acc: wp.array[wp.vec3]):
    k = wp.tid()
    wp.atomic_add(acc, 0, f[ids[k]])


@wp.kernel
def _tick(dt: float, clock: wp.array[float]):
    clock[0] = clock[0] + dt


# ------------------------------------------------------------------------------------------------------ geometry
def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Sim:
    """One scene on one solver. A scene is a dict: ropes (list of node arrays, m), fixed [(rope, node)], grip
    (rope, node) or None, boxes [(center, half, quat or None)], ground, contact (per rope: collides at all),
    self_contact (rope-rope contact on)."""

    def __init__(self, solver, scene, rope, worlds=1, drag=0.0, **over):
        cfg = dict(SOLVERS[solver], **{k: v for k, v in over.items() if k in SOLVERS[solver]})
        self.solver, self.cfg, self.rope, self.drag, self.worlds = solver, cfg, rope, drag, worlds
        self.sc = scene
        self.ropes = [np.asarray(p, float) for p in scene["ropes"]]
        self.contact_on = scene.get("contact", [True] * len(self.ropes))
        self.self_on = bool(scene.get("self_contact", True))
        self.t, self.wall, self.graph = 0.0, 0.0, None
        self.f_acc = wp.zeros(1, dtype=wp.vec3)
        self.clock = wp.zeros(1, dtype=float)
        if solver == "vbd":
            self._build_rod()
        else:
            self._build_particles()

    # ---- particle chains (semi, xpbd)
    def _build_particles(self):
        R, sc = self.rope, self.sc
        b0 = newton.ModelBuilder()
        fixed = set(sc.get("fixed", []))
        grip = sc.get("grip")
        if grip is not None:
            fixed.add(grip)
        node_ids, springs, bends, segs = [], [], [], []
        for ri, P in enumerate(self.ropes):
            seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
            m = np.zeros(len(P))
            m[:-1] += 0.5 * R["lam"] * seglen
            m[1:] += 0.5 * R["lam"] * seglen
            ids = []
            for k, p in enumerate(P):
                mk = 0.0 if (ri, k) in fixed else float(m[k])
                ids.append(b0.add_particle(wp.vec3(*p), wp.vec3(0.0), mk, radius=R["r"]))
            for k in range(len(P) - 1):
                springs.append((ids[k], ids[k + 1], R["EA"] / seglen[k], AXIAL_ETA / seglen[k]))
                segs.append((ids[k], ids[k + 1], ri, k))
            for k in range(1, len(P) - 1):
                bends.append((ids[k - 1], ids[k], ids[k + 1], R["EI"] / (0.5 * (seglen[k - 1] + seglen[k]))))
            node_ids.append(ids)
        for (i, j, ke, kd) in springs:
            b0.add_spring(i, j, ke, kd if self.solver == "semi" else 0.0, 0)
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.mu = R["mu"]
        if sc.get("ground"):
            b0.add_ground_plane(cfg=cfg)
        for (c, h, q) in sc.get("boxes", []):
            b0.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*c), q if q is not None else wp.quat_identity()),
                             hx=h[0], hy=h[1], hz=h[2], cfg=cfg)
        npw = b0.particle_count
        if self.worlds > 1:
            b = newton.ModelBuilder()
            b.replicate(b0, self.worlds, spacing=(2.0, 0.0, 0.0))
        else:
            b = b0
        model = b.finalize()
        self.model, self.npw = model, npw
        W = self.worlds
        sp = np.array([(i, j) for (i, j, _, _) in springs])
        segs = np.array(segs)
        bends = np.array(bends)
        offs = (np.arange(W) * npw)
        nseg = len(segs)
        seg = (segs[None, :, :2] + offs[:, None, None]).reshape(-1, 2)
        nrope = len(self.ropes)
        self.seg = wp.array(seg.astype(np.int32), dtype=int)
        self.srope = wp.array((segs[None, :, 2] + nrope * np.arange(W)[:, None]).ravel().astype(np.int32), dtype=int)
        self.sidx = wp.array(np.tile(segs[:, 3], W).astype(np.int32), dtype=int)
        self.sworld = wp.array(np.repeat(np.arange(W), nseg).astype(np.int32), dtype=int)
        self.wstart = wp.array((np.arange(W) * nseg).astype(np.int32), dtype=int)
        self.wcount = wp.array(np.full(W, nseg, np.int32), dtype=int)
        self.rope_on = wp.array(np.tile(np.array([1 if (c and self.self_on) else 0 for c in self.contact_on], np.int32), W),
                                dtype=int)
        self.cand = wp.zeros((len(seg), 24), dtype=int)
        self.count = wp.zeros(len(seg), dtype=int)
        self.nseg_total = len(seg)
        if len(bends):
            tri = (bends[None, :, :3].astype(np.int64) + offs[:, None, None]).reshape(-1, 3)
            self.bend_tri = wp.array(tri.astype(np.int32), dtype=int)
            self.bend_k = wp.array(np.tile(bends[:, 3], W).astype(np.float32), dtype=float)
        else:
            self.bend_tri = None
        self.node_ids = node_ids                         # world-0 particle ids per rope
        self.nodes_flat = np.concatenate(node_ids)
        mass = model.particle_mass.numpy()
        self.mass = mass
        # contact against surfaces: the cloth teacher's law (semi); XPBD's own (xpbd)
        model.particle_grid = None
        model.soft_contact_mu = R["mu"]
        sc_mask = np.zeros(model.particle_count, np.float32)   # particles of ropes with contact off: radius ~0
        rad = model.particle_radius.numpy()
        for ri, ids in enumerate(node_ids):
            if not self.contact_on[ri]:
                for w in range(W):
                    rad[np.asarray(ids) + w * npw] = 1.0e-6
        model.particle_radius = wp.array(rad, dtype=float)
        self.p_mu = wp.array(np.full(model.particle_count, R["mu"], np.float32), dtype=float)
        self.shape_role = wp.zeros(max(model.shape_count, 1), dtype=int)
        self.rec = wp.zeros((W, 3), dtype=wp.vec3)
        self.nc = wp.zeros(model.particle_count, dtype=float)
        # time step: the stiffest of stretch, bending (zigzag mode: 4 EI / l^3) and contact, per unit mass
        l = float(np.min(np.linalg.norm(np.diff(self.ropes[0], axis=0), axis=1)))
        mmin = R["lam"] * l
        k_eff = max(R["EA"] / l, 4.0 * R["EI"] / l ** 3 * 4.0, CONTACT_RATE * mmin)
        self.k_over_m = k_eff / mmin
        if self.solver == "semi":
            ce = int(self.cfg["ce"])
            n = int(np.ceil(self.cfg["spp"] * np.sqrt(k_eff / mmin) / FPS / ce)) * ce
            self.substeps = max(n, ce)
            self.s = newton.solvers.SolverSemiImplicit(model)
        else:
            self.substeps = int(self.cfg["substeps"])
            self.s = newton.solvers.SolverXPBD(model, iterations=int(self.cfg["iters"]))
        self.dt = 1.0 / (FPS * self.substeps)
        self.steps_per_sqrt_m_k = 1.0 / (self.dt * np.sqrt(k_eff / mmin))
        self.s0, self.s1, self.ctrl = model.state(), model.state(), model.control()
        self.pipe = newton.CollisionPipeline(model, soft_contact_gap=0.005, soft_contact_max=int(model.particle_count * 4))
        self.con = self.pipe.contacts()
        if self.solver == "xpbd":
            model.particle_max_radius = 0.0            # its particle-particle contact off: ours is segment-segment
        self.kin = None
        if grip is not None:
            self.grip_ids = [node_ids[grip[0]][grip[1]]]
            self.grip((0.0, 0.0, 0.0))

    def _particle_substeps(self, k, ce, damp):
        m = self.model
        for i in range(k):
            if i % ce == 0:
                self.pipe.collide(self.s0, self.con)
                if self.self_on:
                    wp.launch(_seg_candidates, dim=self.nseg_total,
                              inputs=[self.s0.particle_q, self.seg, self.srope, self.sidx, self.sworld, self.wstart,
                                      self.wcount, 2.0 * self.rope["r"] + 0.01, self.rope_on],
                              outputs=[self.cand, self.count])
            self.s0.clear_forces()
            if self.kin is not None:
                wp.launch(_drive_particles, dim=len(self.kin[0]), inputs=[*self.kin, self.clock],
                          outputs=[self.s0.particle_q, self.s0.particle_qd])
            if self.bend_tri is not None:
                wp.launch(_bend, dim=self.bend_tri.shape[0], inputs=[self.s0.particle_q, self.bend_tri, self.bend_k],
                          outputs=[self.s0.particle_f])
            if self.self_on:
                wp.launch(_seg_contact, dim=self.nseg_total,
                          inputs=[self.s0.particle_q, self.s0.particle_qd, self.seg, self.cand, self.count,
                                  2.0 * self.rope["r"], CONTACT_RATE, FRICTION_RATE, SELF_MU, m.particle_inv_mass,
                                  CONTACT_ZETA, self.dt], outputs=[self.s0.particle_f])
            if self.solver == "semi":
                c = self.con
                if m.shape_count:
                    wp.launch(_shape_contact, dim=c.soft_contact_max,
                              inputs=[c.soft_contact_count, c.soft_contact_max, c.soft_contact_particle,
                                      c.soft_contact_shape, c.soft_contact_body_pos, c.soft_contact_normal, m.shape_body,
                                      m.shape_material_mu, self.shape_role, self.s0.body_q, self.s0.body_qd, m.body_com,
                                      self.s0.particle_q, self.s0.particle_qd, m.particle_radius, m.particle_flags,
                                      self.p_mu, m.particle_world, CONTACT_RATE, FRICTION_RATE, m.particle_inv_mass,
                                      CONTACT_ZETA, self.dt, self.nc], outputs=[self.s0.particle_f, self.rec])
                self.s.step(self.s0, self.s1, self.ctrl, None, self.dt)
            else:
                self.s.step(self.s0, self.s1, self.ctrl, self.con, self.dt)
            if self.kin is not None:
                wp.launch(_acc_force, dim=len(self.kin[0]), inputs=[self.kin[0], self.s0.particle_f],
                          outputs=[self.f_acc])
                wp.launch(_tick, dim=1, inputs=[self.dt, self.clock])
            self.s0, self.s1 = self.s1, self.s0
            if self.drag > 0.0:
                wp.launch(_damp_vec, dim=m.particle_count, inputs=[self.s0.particle_qd, damp])

    # ---- VBD rods
    def _build_rod(self):
        R, sc = self.rope, self.sc
        b0 = newton.ModelBuilder()
        b0.rigid_gap = 0.0
        scfg = newton.ModelBuilder.ShapeConfig(mu=R["mu"], ke=VBD_KE, kd=0.0)
        if sc.get("ground"):
            b0.add_ground_plane(cfg=scfg)
        for (c, h, q) in sc.get("boxes", []):
            b0.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*c), q if q is not None else wp.quat_identity()),
                             hx=h[0], hy=h[1], hz=h[2], cfg=scfg)
        fixed = set(sc.get("fixed", []))
        grip = sc.get("grip")
        self.body_ids, kin, self.seg_len = [], [], []
        grip_body = None
        for ri, P in enumerate(self.ropes):
            rcfg = newton.ModelBuilder.ShapeConfig(mu=R["mu"], ke=VBD_KE, kd=0.0)
            if not self.contact_on[ri]:
                rcfg.has_shape_collision = False
            # built straight (Newton takes the rest shape from the build): the rest shape is straight as for the
            # particle chains, and the bodies are then placed on P (parallel-transport frames: bending, no twist)
            seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
            straight_pts = P[0][None] + np.r_[0.0, np.cumsum(seglen)][:, None] * np.array([0.0, 0.0, 1.0])[None]
            rod = newton.Rod(straight_pts.astype(np.float32), radius=R["r"], stretch_rigidity=R["EA"],
                             shear_rigidity=R["EA"], bend_rigidity=R["EI"], twist_rigidity=R["EI"] / 1.3)
            bodies, _ = b0.add_rod(rod=rod, cfg=rcfg, label=f"rope{ri}", body_frame_origin="com")
            frames = newton.Rod(P.astype(np.float32)).quaternions
            for k, bi in enumerate(bodies):
                c = 0.5 * (P[k] + P[k + 1])
                b0.body_q[bi] = wp.transform(wp.vec3(*c), wp.quat(*[float(v) for v in frames[k]]))
            nn = len(P) - 1
            # a fixed node holds the segment(s) at it: both nodes fixed -> that segment; a lone fixed node -> its
            # adjacent segment (VBD has no point pin on a ROD chain: the end is clamped)
            ks = set()
            for k in range(nn):
                if (ri, k) in fixed and (ri, k + 1) in fixed:
                    ks.add(k)
            for k in range(nn + 1):
                if (ri, k) in fixed and not any(s in ks for s in (k - 1, k)):
                    ks.add(min(k, nn - 1))
            if grip is not None and grip[0] == ri:
                grip_body = bodies[min(grip[1], nn - 1)]
                ks.add(min(grip[1], nn - 1))
            for k, bi in enumerate(bodies):
                ms = R["lam"] * seglen[k]
                Ip, Ia = ms * (seglen[k] ** 2 / 12.0 + R["r"] ** 2 / 4.0), ms * R["r"] ** 2 / 2.0
                if k in ks:
                    b0.body_mass[bi], b0.body_inv_mass[bi] = 0.0, 0.0
                    b0.body_inertia[bi], b0.body_inv_inertia[bi] = wp.mat33(0.0), wp.mat33(0.0)
                    kin.append(bi)
                else:
                    I = wp.mat33(Ip, 0.0, 0.0, 0.0, Ip, 0.0, 0.0, 0.0, Ia)
                    b0.body_mass[bi], b0.body_inv_mass[bi] = ms, 1.0 / ms
                    b0.body_inertia[bi], b0.body_inv_inertia[bi] = I, wp.inverse(I)
            self.body_ids.append(list(bodies))
            self.seg_len.append(seglen)
        if not self.self_on:                             # rope-rope contact off: filter every capsule pair of
            shapes = [s for s in range(b0.shape_count) if b0.shape_body[s] >= 0]   # different / same ropes
            for i in range(len(shapes)):
                for j in range(i + 1, len(shapes)):
                    b0.add_shape_collision_filter_pair(shapes[i], shapes[j])
        nbw = b0.body_count
        if self.worlds > 1:
            b = newton.ModelBuilder()
            b.replicate(b0, self.worlds, spacing=(2.0, 0.0, 0.0))
        else:
            b = b0
        b.color()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = b.finalize()
        self.model, self.nbw = model, nbw
        self.kin_bodies = kin
        self.mass_b = model.body_mass.numpy()[:nbw].astype(np.float64)
        self.inertia_b = model.body_inertia.numpy()[:nbw].astype(np.float64)
        self.substeps = int(self.cfg["substeps"])
        self.dt = 1.0 / (FPS * self.substeps)
        self.pipe = newton.CollisionPipeline(model, contact_matching="sticky")
        self.con = self.pipe.contacts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.s = newton.solvers.SolverVBD(model, iterations=int(self.cfg["iters"]), rigid_compliant_alm=True,
                                              rigid_contact_history=True, rigid_body_contact_buffer_size=256)
        self.s0, self.s1, self.ctrl = model.state(), model.state(), model.control()
        self.kin = None
        self.grip_body = grip_body
        if grip_body is not None:
            self.grip((0.0, 0.0, 0.0))

    def _rod_substeps(self, k, damp):
        for _ in range(k):
            self.s0.clear_forces()
            if self.kin is not None:
                wp.launch(_drive_bodies, dim=len(self.kin[0]), inputs=[*self.kin, self.clock],
                          outputs=[self.s0.body_q, self.s0.body_qd])
            self.pipe.collide(self.s0, self.con)
            self.s.step(self.s0, self.s1, self.ctrl, self.con, self.dt)
            if self.kin is not None:
                wp.launch(_tick, dim=1, inputs=[self.dt, self.clock])
                wp.launch(_drive_bodies, dim=len(self.kin[0]), inputs=[*self.kin, self.clock],
                          outputs=[self.s1.body_q, self.s1.body_qd])
            self.s0, self.s1 = self.s1, self.s0
            if self.drag > 0.0:
                wp.launch(_damp_body, dim=self.model.body_count, inputs=[self.s0.body_qd, damp])

    # ---- common
    def grip(self, vel):
        """The gripped node (particles) or segment (vbd) moves at vel (m/s) from where it is now."""
        if self.solver == "vbd":
            ids = np.array([self.grip_body])
            q0 = self.s0.body_q.numpy()[ids]
            self.kin = (wp.array(ids.astype(np.int32), dtype=int), wp.array(q0, dtype=wp.transform),
                        wp.array(np.asarray([vel], np.float32), dtype=wp.vec3))
        else:
            ids = np.array(self.grip_ids)
            x0 = self.s0.particle_q.numpy()[ids]
            self.kin = (wp.array(ids.astype(np.int32), dtype=int), wp.array(x0, dtype=wp.vec3),
                        wp.array(np.asarray([vel], np.float32), dtype=wp.vec3))
        self.clock.zero_()
        self.graph = None

    def grip_force(self):
        """Particles: the force the gripper applies = minus the rope's force on the massless gripped node, averaged
        over the last frame's substeps (one substep's force rings with the rope's axial modes)."""
        return -self.f_acc.numpy()[0].astype(np.float64) / self.substeps

    def frame(self, n=1):
        damp = float(np.exp(-self.drag * self.dt))
        t0 = time.time()
        if n == 1:
            self.f_acc.zero_()
        if self.solver == "vbd":
            k = self.substeps
            if k % 2:
                raise ValueError("vbd substeps must be even (captured graph)")
            if self.graph is None:
                self._rod_substeps(k, damp)            # uncaptured: compiles
                self.t += 1.0 / FPS
                n -= 1
                with wp.ScopedCapture() as cap:
                    self._rod_substeps(k, damp)
                self.graph = cap.graph
            for _ in range(n):
                wp.capture_launch(self.graph)
                self.t += 1.0 / FPS
        else:
            ce = int(self.cfg["ce"])
            chunks = self.substeps // ce
            done = 0
            if self.graph is None:
                self._particle_substeps(ce, ce, damp)
                done = 1
                with wp.ScopedCapture() as cap:
                    self._particle_substeps(ce, ce, damp)
                self.graph = cap.graph
            for _ in range(n):
                for _ in range(chunks - done):
                    wp.capture_launch(self.graph)
                done = 0
                self.t += 1.0 / FPS
        wp.synchronize()
        self.wall += time.time() - t0

    def rate(self):
        return self.worlds * self.t / max(self.wall, 1e-9)

    def nodes(self, rope=None):
        """World-0 rope nodes (m), all ropes concatenated or one rope."""
        if self.solver != "vbd":
            x = self.s0.particle_q.numpy().astype(np.float64)
            if rope is None:
                return x[self.nodes_flat]
            return x[self.node_ids[rope]]
        q = self.s0.body_q.numpy().astype(np.float64)
        out = []
        for ri, bodies in enumerate(self.body_ids):
            if rope is not None and ri != rope:
                continue
            pts = []
            for k, bi in enumerate(bodies):
                Rm = quat_matrix(q[bi, 3:7])
                pts.append(q[bi, :3] - Rm[:, 2] * 0.5 * self.seg_len[ri][k])
            Rm = quat_matrix(q[bodies[-1], 3:7])
            pts.append(q[bodies[-1], :3] + Rm[:, 2] * 0.5 * self.seg_len[ri][-1])
            out.append(np.array(pts))
        return np.concatenate(out)

    def node_speed_max(self):
        if self.solver != "vbd":
            return float(np.linalg.norm(self.s0.particle_qd.numpy()[self.nodes_flat], axis=1).max())
        qd = self.s0.body_qd.numpy()[:self.nbw].astype(np.float64)
        return float((np.linalg.norm(qd[:, :3], axis=1) + 0.5 * 0.01 * np.linalg.norm(qd[:, 3:], axis=1)).max())

    def energy(self):
        """World 0: kinetic (translational + rotational) and potential energy (J)."""
        if self.solver != "vbd":
            x = self.s0.particle_q.numpy()[:self.npw].astype(np.float64)
            v = self.s0.particle_qd.numpy()[:self.npw].astype(np.float64)
            m = self.mass[:self.npw]
            return 0.5 * float((m * (v ** 2).sum(1)).sum()), float((m * G * x[:, 2]).sum())
        q = self.s0.body_q.numpy()[:self.nbw].astype(np.float64)
        qd = self.s0.body_qd.numpy()[:self.nbw].astype(np.float64)
        m = self.mass_b
        ke = 0.5 * float((m * (qd[:, :3] ** 2).sum(1)).sum())
        for i in range(self.nbw):
            if m[i] > 0:
                Rm = quat_matrix(q[i, 3:7])
                Iw = Rm @ self.inertia_b[i] @ Rm.T
                w = qd[i, 3:]
                ke += 0.5 * float(w @ Iw @ w)
        return ke, float((m * G * q[:, 2]).sum())

    def total_mass(self):
        return float(self.mass[:self.npw].sum()) if self.solver != "vbd" else float(self.mass_b.sum())

    def free_length(self, rope=0):
        """Length of the rope not held (vbd holds whole segments)."""
        P = self.ropes[rope]
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        if self.solver == "vbd":
            held = [k for k, bi in enumerate(self.body_ids[rope]) if bi in self.kin_bodies]
            L -= float(sum(self.seg_len[rope][k] for k in held))
        return L


def in_bound(x, origin=None, bound=3.0):
    o = np.zeros(3) if origin is None else origin
    return bool(np.isfinite(x).all() and np.abs(x - o).max() < bound)


# ------------------------------------------------------------------------------------------------ crossings
# ---------------------------------------------------------------------------------------------------- checks
def check_stretch(solver, **over):
    out = {}
    for EA in (20.0, 1.0e3, 1.0e4):
        rope = dict(ROPE, EA=EA)
        scene = {"ropes": [straight((0, 0, 1.2), (0, 0, -1), 1.0, rope["l"])], "fixed": [(0, 0)]}
        sim = Sim(solver, scene, rope, drag=4.0, **over)
        x0 = sim.nodes()
        sim.frame(int(3.0 * FPS))
        xs = []
        for _ in range(FPS):
            sim.frame()
            xs.append(sim.nodes())
        x = np.mean(xs, axis=0)
        ext = float(x0[-1, 2] - x[-1, 2])
        L = sim.free_length()
        want = rope["lam"] * G * L ** 2 / (2.0 * EA)
        ok = in_bound(x)
        row = {"extension_mm": 1e3 * ext, "closed_form_mm": 1e3 * want, "ratio": ext / want, "free_length_m": L,
               "max_speed_end_m_s": sim.node_speed_max(), "substeps": sim.substeps, "sim_s_per_wall_s": sim.rate(),
               "in_bound": ok, "pass": bool(ok and abs(ext / want - 1.0) < 0.05)}
        out[f"EA{EA:g}"] = row
        print(f"  stretch {solver:5s} EA {EA:7.0f} N: extension {1e3 * ext:9.4f} mm, closed form {1e3 * want:8.4f} mm "
              f"(ratio {ext / want:7.3f}), {sim.substeps} substeps, {sim.rate():.2f} sim-s/wall-s", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["summary"] = "ratio " + " / ".join(f"{out[k]['ratio']:.3f}" for k in ("EA20", "EA1000", "EA10000"))
    return out


def catenary(S, L):
    from scipy.optimize import brentq
    a = brentq(lambda a: 2.0 * a * np.sinh(S / (2.0 * a)) - L, 1e-3 * S, 1e3 * S)
    return a, a * (np.cosh(S / (2.0 * a)) - 1.0)


def check_catenary(solver, **over):
    rope = dict(ROPE, EI=1.0e-6)                          # a string: the catenary is perfectly flexible
    S, L, z0 = 0.7, 1.0, 1.2
    n = int(round(L / rope["l"]))
    h = np.sqrt((L / 2) ** 2 - (S / 2) ** 2)
    P = np.array([(-S / 2 + S * k / n, 0.0, z0 - h * (1.0 - abs(2.0 * k / n - 1.0))) for k in range(n + 1)])
    scene = {"ropes": [P], "fixed": [(0, 0), (0, n)]}
    sim = Sim(solver, scene, rope, drag=4.0, **over)
    sim.frame(int(5.0 * FPS))
    xs = []
    for _ in range(FPS):
        sim.frame()
        xs.append(sim.nodes())
    x = np.mean(xs, axis=0)
    a, sag_want = catenary(S, L)
    sag = float(z0 - x[:, 2].min())
    z_cat = z0 + a * np.cosh(x[:, 0] / a) - a * np.cosh(S / (2 * a))
    rms = float(np.sqrt(np.mean((x[:, 2] - z_cat) ** 2)))
    ok = in_bound(x)
    out = {"sag_m": sag, "catenary_sag_m": float(sag_want), "sag_rel_error": sag / sag_want - 1.0,
           "shape_rms_mm": 1e3 * rms, "shape_rms_over_sag": rms / sag_want, "in_bound": ok,
           "max_speed_end_m_s": sim.node_speed_max(),
           "pass": bool(ok and abs(sag / sag_want - 1.0) < 0.02 and rms / sag_want < 0.03)}
    out["summary"] = f"sag {100 * out['sag_rel_error']:+.2f}%, shape rms {100 * rms / sag_want:.2f}% of sag"
    print(f"  catenary {solver:5s}: sag {1e3 * sag:.1f} mm vs catenary {1e3 * sag_want:.1f} mm "
          f"({100 * (sag / sag_want - 1):+.2f}%), shape rms {1e3 * rms:.2f} mm ({100 * rms / sag_want:.2f}% of sag)",
          flush=True)
    return out


def elastica_tip(l, c):
    """Heavy cantilever clamped horizontal (the cloth gate's): returns tip drop / l and the chord angle."""
    from scipy.integrate import solve_bvp
    k = 1.0 / c ** 3
    def f(s, y):
        return np.vstack([y[1], k * (l - s) * np.cos(y[0]), np.cos(y[0]), np.sin(y[0])])
    s = np.linspace(0.0, l, 400)
    y0 = np.zeros((4, s.size))
    y0[2] = s
    sol = solve_bvp(f, lambda ya, yb: np.array([ya[0], yb[1], ya[2], ya[3]]), s, y0, tol=1e-8, max_nodes=100000)
    x, z = sol.y[2, -1], sol.y[3, -1]
    return float(-z / l), float(np.degrees(np.arctan2(-z, x)))


def check_bend(solver, **over):
    out = {}
    for tag, EI, h in (("EI", ROPE["EI"], 0.01), ("EI_half_segment", ROPE["EI"], 0.005),
                       ("8EI_double_overhang", 8 * ROPE["EI"], 0.01)):
        rope = dict(ROPE, EI=EI, l=h)
        c = (EI / (rope["lam"] * G)) ** (1.0 / 3.0)
        l = round(2.0 * c / h) * h
        clamp = 0.03
        P = straight((-clamp, 0.0, 1.0), (1, 0, 0), l + clamp, h)
        nfix = int(round(clamp / h))
        scene = {"ropes": [P], "fixed": [(0, k) for k in range(nfix + 1)]}
        sim = Sim(solver, scene, rope, drag=4.0, **over)
        sim.frame(int(5.0 * FPS))
        x = sim.nodes()
        tip = x[-1]
        drop = 1.0 - tip[2]
        want, want_ang = elastica_tip(l, c)
        ok = in_bound(x)
        row = {"overhang_m": l, "bending_length_m": c, "tip_drop_over_l": drop / l, "elastica_drop_over_l": want,
               "tip_angle_deg": float(np.degrees(np.arctan2(drop, tip[0]))), "elastica_angle_deg": want_ang,
               "ratio": (drop / l) / want, "max_speed_end_m_s": sim.node_speed_max(), "in_bound": ok,
               "pass": bool(ok and abs((drop / l) / want - 1.0) < 0.10)}
        out[tag] = row
        print(f"  bend    {solver:5s} {tag:19s} l {100 * l:4.1f} cm c {100 * c:4.2f} cm: tip drop/l {drop / l:.3f}, "
              f"elastica {want:.3f} (ratio {row['ratio']:.3f})", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["summary"] = "ratio " + " / ".join(f"{out[k]['ratio']:.3f}" for k in ("EI", "EI_half_segment", "8EI_double_overhang"))
    return out


def _pendulum(rope, L=0.5):
    return {"ropes": [straight((0.0, 0.0, 1.2), (1, 0, 0), L, rope["l"])], "fixed": [(0, 0)]}


def check_swing(solver, **over):
    out = {}
    for drag in (0.0, 2.0):
        sim = Sim(solver, _pendulum(ROPE), ROPE, drag=drag, **over)
        ke0, pe0 = sim.energy()
        e0 = ke0 + pe0
        worst, released, bound = -np.inf, 0.0, True
        for f in range(int(6.0 * FPS)):
            sim.frame()
            ke, pe = sim.energy()
            worst = max(worst, ke + pe - e0)
            released = max(released, pe0 - pe)
            if f == int(5.0 * FPS) - 1:
                x5 = sim.nodes()
            if f % 10 == 0:
                bound = bound and in_bound(sim.nodes())
        x = sim.nodes()
        bound = bound and in_bound(x)
        move = float(np.linalg.norm(x - x5, axis=1).max())
        row = {"energy_gain_J": worst, "energy_released_J": released, "gain_fraction": worst / max(released, 1e-12),
               "max_move_last_second_mm": 1e3 * move, "in_bound": bound, "sim_s_per_wall_s": sim.rate()}
        row["pass"] = bool(bound and (row["gain_fraction"] < 0.01 if drag == 0.0 else move < 0.005))
        out[f"drag{drag:g}"] = row
        print(f"  swing   {solver:5s} drag {drag:3.1f}/s: max (KE+PE) gain {100 * row['gain_fraction']:+.3f}% of "
              f"{released:.2e} J released, moved {1e3 * move:.2f} mm in the 6th second, in bound {bound}, "
              f"{sim.rate():.2f} sim-s/wall-s", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["summary"] = (f"gain {100 * out['drag0']['gain_fraction']:+.3f}%, 6th-second move "
                      f"{out['drag2']['max_move_last_second_mm']:.2f} mm")
    return out


def check_freefall(solver, **over):
    P = straight((0.0, 0.0, 1.0), (1.0, 0.5, 0.6), 0.3, ROPE["l"])     # tilted straight (its rest shape), 1 m up
    sim = Sim(solver, {"ropes": [P]}, ROPE, **over)
    x0 = sim.nodes()
    d0 = np.linalg.norm(x0 - x0[0], axis=1)
    ts, zs, shape = [], [], 0.0
    if solver == "vbd":
        m = np.r_[sim.mass_b]
    for f in range(15):
        sim.frame()
        x = sim.nodes()
        ts.append(sim.t)
        if solver == "vbd":
            q = sim.s0.body_q.numpy()[:sim.nbw].astype(np.float64)
            zs.append(float((m * q[:, 2]).sum() / m.sum()))
        else:
            mm = sim.mass[sim.nodes_flat]
            zs.append(float((mm * x[:, 2]).sum() / mm.sum()))
        shape = max(shape, float(np.abs(np.linalg.norm(x - x[0], axis=1) - d0).max()))
    a = -2.0 * np.polyfit(np.array(ts), np.array(zs), 2)[0]
    out = {"accel_m_s2": float(a), "rel_error": float(a / G - 1.0), "shape_change_mm": 1e3 * shape,
           "pass": bool(abs(a / G - 1.0) < 1e-3 and shape < 1e-4)}
    out["summary"] = f"accel {100 * (a / G - 1):+.4f}% of g, shape {1e3 * shape:.4f} mm"
    print(f"  freefall {solver:5s}: centroid accel {a:.5f} m/s2 ({100 * (a / G - 1):+.4f}%), shape change "
          f"{1e3 * shape:.4f} mm", flush=True)
    return out


def check_slide(solver, theta_deg=20.0, **over):
    th = np.radians(theta_deg)
    out = {}
    for mu in (0.2, 0.5):
        rope = dict(ROPE, mu=mu)
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-th))           # rises toward +x
        Rm = np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)
        top = np.array([0.0, 0.0, 0.5]) + Rm @ np.array([0.0, 0.0, 0.05 + rope["r"]])
        P = top[None] + (np.linspace(-0.1, 0.1, 21)[:, None] * Rm[:, 0][None])
        scene = {"ropes": [P], "boxes": [((0.0, 0.0, 0.5), (3.0, 0.5, 0.05), q)]}
        sim = Sim(solver, scene, rope, **over)
        down = -Rm[:, 0]
        ts, ss = [], []
        for f in range(int(2.0 * FPS)):
            sim.frame()
            ts.append(sim.t)
            ss.append(float(sim.nodes().mean(0) @ down))
        ts, ss = np.array(ts), np.array(ss) - ss[0]
        want = G * (np.sin(th) - mu * np.cos(th))
        ok = in_bound(sim.nodes())
        if want > 0:
            w = (ts > 0.3) & (ts < 1.3)
            a = 2.0 * np.polyfit(ts[w], ss[w], 2)[0]
            row = {"accel_m_s2": float(a), "slide_law_m_s2": float(want), "rel_error": float(a / want - 1.0),
                   "in_bound": ok, "pass": bool(ok and abs(a / want - 1.0) < 0.02)}
            print(f"  slide   {solver:5s} mu {mu}: accel {a:.4f} m/s2, law {want:.4f} ({100 * (a / want - 1):+.2f}%)",
                  flush=True)
        else:
            w = ts > 1.0
            drift = float(np.polyfit(ts[w], ss[w], 1)[0])
            row = {"drift_mm_s": 1e3 * drift, "travel_mm": 1e3 * float(ss[-1]), "in_bound": ok,
                   "pass": bool(ok and abs(drift) < 1e-3)}
            print(f"  slide   {solver:5s} mu {mu}: drift {1e3 * drift:+.3f} mm/s over 1-2 s, travel {1e3 * ss[-1]:+.3f} mm",
                  flush=True)
        out[f"mu{mu}"] = row
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["summary"] = (f"mu0.2 {100 * out['mu0.2'].get('rel_error', np.nan):+.2f}%, mu0.5 drift "
                      f"{out['mu0.5'].get('drift_mm_s', np.nan):+.3f} mm/s")
    return out


def _table_scene(rope, P, **kw):
    return dict({"ropes": [P], "ground": True, "boxes": [((0.0, 0.0, TOP / 2), (0.6, 0.6, TOP / 2), None)]}, **kw)


def _wavy(rope, L=0.5, amp=0.03, cycles=1.5, z=TOP + 0.02):
    n = int(round(L / rope["l"]))
    xs = np.linspace(-L / 2, L / 2, 4 * n + 1)
    ys = amp * np.sin(2 * np.pi * cycles * (xs + L / 2) / L)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]     # resample to equal arc length
    sk = np.linspace(0.0, s[-1], n + 1)
    P = np.stack([np.interp(sk, s, xs), np.interp(sk, s, ys), np.full(n + 1, z + rope["r"])], 1)
    return P


def check_drop(solver, **over):
    sim = Sim(solver, _table_scene(ROPE, _wavy(ROPE)), ROPE, drag=1.0, **over)
    sim.frame(int(3.0 * FPS))
    xa = sim.nodes()
    pen = np.inf
    for _ in range(FPS):
        sim.frame()
        pen = min(pen, float((sim.nodes()[:, 2] - TOP - ROPE["r"]).min()))
    xb = sim.nodes()
    creep = float(np.linalg.norm(xb - xa, axis=1).max())
    ok = in_bound(xb)
    out = {"creep_mm_s": 1e3 * creep, "penetration_min_mm": 1e3 * pen, "in_bound": ok,
           "max_speed_end_m_s": sim.node_speed_max(), "sim_s_per_wall_s": sim.rate(),
           "pass": bool(ok and creep < 1e-3 and pen > -1e-3)}
    out["summary"] = f"creep {1e3 * creep:.3f} mm/s, penetration {1e3 * pen:+.3f} mm"
    print(f"  drop    {solver:5s}: creep {1e3 * creep:.3f} mm/s in the 4th second, lowest node {1e3 * pen:+.3f} mm, "
          f"{sim.rate():.2f} sim-s/wall-s", flush=True)
    return out


def check_cross(solver, **over):
    """Rope A dropped across rope B (pinned taut along y): the counter's control (no A-B contact) and the test."""
    out = {}
    l = ROPE["l"]
    B = straight((0.0, -0.2, 1.0), (0, 1, 0), 0.4, l)
    A = straight((-0.2 + 0.37 * l, 0.013, 1.05), (1, 0, 0), 0.4, l)   # crossing between nodes of both
    nb = len(B) - 1
    for sc in (0, 1):
        scene = {"ropes": [A, B], "fixed": [(1, 0), (1, nb)], "self_contact": bool(sc)}
        sim = Sim(solver, scene, ROPE, drag=1.0, **over)
        cc = CrossingCounter([len(A), len(B)])
        cc.add(sim.nodes())
        for _ in range(int(1.5 * FPS)):
            sim.frame()
            cc.add(sim.nodes())
        xa, xb = sim.nodes(0), sim.nodes(1)
        drop = float(xb[:, 2].min() - xa[:, 2].min())   # A's lowest node below B's: < its 0.4 m length if held
        out[f"self_contact{sc}"] = {"crossings": cc.n, "min_centerline_distance_mm": 1e3 * cc.dmin,
                                    "A_lowest_below_B_lowest_mm": 1e3 * drop, "A_held": bool(drop < 0.4),
                                    "in_bound": in_bound(sim.nodes()) if sc else True}
        print(f"  cross   {solver:5s} contact {sc}: {cc.n} crossings, closest centerlines {1e3 * cc.dmin:.2f} mm, "
              f"A's lowest node {1e3 * drop:.1f} mm below B's", flush=True)
    c0, c1 = out["self_contact0"], out["self_contact1"]
    out["counter_can_fail"] = bool(c0["crossings"] > 0)
    out["pass"] = bool(out["counter_can_fail"] and c1["crossings"] == 0 and c1["A_held"] and c1["in_bound"]
                       and not c0["A_held"])
    out["summary"] = (f"control {c0['crossings']} crossings (A fell {c0['A_lowest_below_B_lowest_mm'] / 1e3:.1f} m), "
                      f"with contact {c1['crossings']} (A held: {c1['A_held']})")
    return out


def _coil(rope, L=2.0, R=0.04, pitch=0.015, tilt=0.35):
    n = int(round(L / rope["l"]))
    turns = L / np.hypot(2 * np.pi * R, pitch)
    phi = np.linspace(0, 2 * np.pi * turns, 20 * n + 1)
    P = np.stack([R * np.cos(phi), R * np.sin(phi), pitch * phi / (2 * np.pi)], 1)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    sk = np.linspace(0, s[-1], n + 1)
    P = np.stack([np.interp(sk, s, P[:, i]) for i in range(3)], 1)
    c, si = np.cos(tilt), np.sin(tilt)
    P = P @ np.array([[1, 0, 0], [0, c, -si], [0, si, c]]).T
    P[:, 2] += TOP + rope["r"] + 0.01 - P[:, 2].min()
    return P


def check_pile(solver, **over):
    out = {}
    P = _coil(ROPE)
    for sc in ((1, 0) if solver != "vbd" else (1,)):
        sim = Sim(solver, _table_scene(ROPE, P, self_contact=bool(sc)), ROPE, drag=1.0, **over)
        cc = CrossingCounter([len(P)])
        cc.add(sim.nodes())
        bound, vmax = True, 0.0
        for _ in range(int(3.0 * FPS)):
            sim.frame()
            x = sim.nodes()
            cc.add(x)
            bound = bound and in_bound(x)
            vmax = max(vmax, sim.node_speed_max())
        pen = float((sim.nodes()[:, 2] - TOP - ROPE["r"]).min())
        out[f"self_contact{sc}"] = {"crossings": cc.n, "min_centerline_distance_mm": 1e3 * cc.dmin, "in_bound": bound,
                                    "max_speed_m_s": vmax, "penetration_end_mm": 1e3 * pen,
                                    "pile_height_mm": 1e3 * float(sim.nodes()[:, 2].max() - TOP),
                                    "sim_s_per_wall_s": sim.rate()}
        print(f"  pile    {solver:5s} self-contact {sc}: {cc.n} crossings, closest centerlines {1e3 * cc.dmin:.2f} mm, "
              f"in bound {bound}, max speed {vmax:.2f} m/s, lowest {1e3 * pen:+.2f} mm, {sim.rate():.2f} sim-s/wall-s",
              flush=True)
    r1 = out["self_contact1"]
    out["pass"] = bool(r1["crossings"] == 0 and r1["in_bound"] and r1["min_centerline_distance_mm"] > 1e3 * ROPE["r"]
                       and r1["max_speed_m_s"] < 20.0)
    if "self_contact0" in out:
        out["counter_can_fail"] = bool(out["self_contact0"]["crossings"] > 0)
    out["summary"] = (f"{r1['crossings']} crossings, closest {r1['min_centerline_distance_mm']:.2f} mm"
                      + (f" (control {out['self_contact0']['crossings']})" if "self_contact0" in out else ""))
    return out


def check_gripper(solver, **over):
    L = 0.5
    P = straight((-0.25, 0.0, TOP + ROPE["r"]), (1, 0, 0), L, ROPE["l"])
    sim = Sim(solver, _table_scene(ROPE, P, grip=(0, 0)), ROPE, drag=1.0, **over)
    sim.frame(5)
    origin = sim.nodes().mean(0)
    bound, vmax = True, 0.0
    fs = []
    for vel, sec in (((0.0, 0.0, 0.3), 2.5), ((0.0, 0.3, 0.0), 1.0), ((0.0, 0.0, 0.0), 4.0)):
        sim.grip(vel)
        for f in range(int(sec * FPS)):
            sim.frame()
            x = sim.nodes()
            bound = bound and in_bound(x, origin)
            vmax = max(vmax, sim.node_speed_max())
            if vel == (0.0, 0.0, 0.0) and f >= int((sec - 0.5) * FPS) and solver == "semi":   # xpbd: constraint forces never reach particle_f
                fs.append(sim.grip_force())
    x = sim.nodes()
    g = x[0]
    bottom = x[-1]
    horiz = float(np.hypot(*(bottom - g)[:2]))
    hang = float(g[2] - bottom[2])
    out = {"in_bound": bound, "max_speed_m_s": vmax, "bottom_offset_mm": 1e3 * horiz, "hanging_length_m": hang,
           "length_m": L, "lowest_above_table_mm": 1e3 * float(x[:, 2].min() - TOP)}
    ok = bound and vmax < 20.0 and horiz < 0.02 and abs(hang / L - 1.0) < 0.02
    if fs:
        F = np.mean(fs, axis=0)
        W = sim.total_mass() * G
        out.update({"grip_force_N": F.tolist(), "weight_N": W, "force_ratio": float(F[2] / W)})
        ok = ok and abs(F[2] / W - 1.0) < 0.02
    out["pass"] = bool(ok)
    out["summary"] = (f"bound {bound}, vmax {vmax:.2f} m/s, bottom off {1e3 * horiz:.1f} mm, hang {hang / L:.4f} L"
                      + (f", force/weight {out['force_ratio']:.4f}" if fs else ""))
    print(f"  gripper {solver:5s}: {out['summary']}", flush=True)
    return out


def check_stability(solver, **over):
    if solver != "semi":
        return {"skipped": "explicit-step sweep, semi only"}
    out = {}
    for EA in (1.0e3, 1.0e4):
        rope = dict(ROPE, EA=EA)
        rows = {}
        for spp in (0.8, 1.0, 1.2, 1.5, 2.0):
            sim = Sim(solver, _pendulum(rope), rope, **dict(over, spp=spp, ce=10))
            ke0, pe0 = sim.energy()
            worst, released, bound = -np.inf, 0.0, True
            for _ in range(FPS):
                sim.frame()
                ke, pe = sim.energy()
                worst, released = max(worst, ke + pe - ke0 - pe0), max(released, pe0 - pe)
                bound = bound and in_bound(sim.nodes())
                if not bound:
                    break
            stable = bool(bound and worst < 0.01 * max(released, 1e-12))
            rows[f"spp{spp:g}"] = {"steps_per_sqrt_m_k": sim.steps_per_sqrt_m_k, "substeps": sim.substeps,
                                   "stable": stable, "gain_fraction": worst / max(released, 1e-12)}
            print(f"  stability semi EA {EA:g}: {sim.substeps} substeps ({sim.steps_per_sqrt_m_k:.2f} per sqrt(m/k)): "
                  f"stable {stable}, gain {100 * worst / max(released, 1e-12):+.3f}%", flush=True)
        st = [v["steps_per_sqrt_m_k"] for v in rows.values() if v["stable"]]
        rows["min_stable_steps_per_sqrt_m_k"] = min(st) if st else None
        out[f"EA{EA:g}"] = rows
    out["pass"] = True
    out["summary"] = "min stable steps per sqrt(m/k): " + " / ".join(
        f"{out[k]['min_stable_steps_per_sqrt_m_k']}" for k in ("EA1000", "EA10000"))
    return out


def check_throughput(solver, worlds=64, seconds=2.0, **over):
    sim = Sim(solver, _table_scene(ROPE, _wavy(ROPE)), ROPE, worlds=worlds, drag=1.0, **over)
    sim.frame(1)
    t0 = time.time()
    sim.frame(int(seconds * FPS))
    wall = time.time() - t0
    rate = worlds * seconds / wall
    out = {"worlds": worlds, "sim_seconds_per_wall_second": rate, "four_second_episodes_per_hour": 3600.0 * rate / 4.0,
           "substeps": sim.substeps, "pass": True}
    out["summary"] = f"{rate:.1f} sim-s/wall-s ({out['four_second_episodes_per_hour']:.0f} 4-s episodes/h)"
    print(f"  throughput {solver:5s} x{worlds}: {rate:.2f} sim-s/wall-s, {sim.substeps} substeps", flush=True)
    return out


CHECKS = {"stretch": check_stretch, "catenary": check_catenary, "bend": check_bend, "swing": check_swing,
          "freefall": check_freefall, "slide": check_slide, "drop": check_drop, "cross": check_cross,
          "pile": check_pile, "gripper": check_gripper, "stability": check_stability, "throughput": check_throughput}


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
    np.random.seed(SEED)
    wp.init()
    wp.rand_init(SEED)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    tag = "" if not over else "_" + "_".join(f"{k}{v:g}" for k, v in over.items())
    for s in solvers:
        key = s + tag
        res.setdefault(key, {})
        for n in names:
            t0 = time.time()
            try:
                res[key][n] = CHECKS[n](s, **over)
            except Exception as exc:                    # record the failure, keep going
                import traceback
                traceback.print_exc()
                res[key][n] = {"error": repr(exc), "pass": False}
                print(f"  {n} {s}: ERROR {exc!r}", flush=True)
            res[key][n]["wall_s"] = time.time() - t0
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(res, indent=1, default=float))
    res["_versions"] = {"newton": newton.__version__, "warp": wp.config.version, "rope": ROPE, "solvers": SOLVERS}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
