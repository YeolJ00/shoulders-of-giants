"""One interface over Newton's solvers, so checks and the data engine can swap the teacher.

Pinned: newton 1.7.0, mujoco-warp 3.15.0, warp 1.18.0+cu12 (requirements-lsim.txt). Solvers and
what each needs:

  mujoco      MuJoCo-Warp, soft contacts by convex optimisation (Isaac Lab 3.0's default Newton
              backend), its own collision detection (a mesh becomes its convex hull).
              impratio 10 (the default 1 creeps 0.9 mm/s on a 20 deg ramp), elliptic cone,
              explicit contact buffers (overflow silently drops contacts).
  mujoco_ncd  the same solver on contacts from Newton's collision pipeline (concave meshes stay
              concave).
  kamino      hard contacts and joint limits as a complementarity problem (proximal ADMM),
              restitutive impacts; its own collision detector. BETA in Newton 1.7.
  xpbd        position-based; honours a restitution coefficient (enable_restitution).
  vbd         VBD/AVBD block descent; rigid bodies and cloth in one solver; stiff contacts.

Contact forces come from the solver's CONTACT_F observable (force on shape0's body by shape1's
body, world frame); the amendment needs them for the gripper-force slot.
"""
import numpy as np
import warp as wp
import newton

SOLVERS = ("mujoco", "mujoco_ncd", "kamino", "xpbd", "vbd")
# MuJoCo-Warp contact stiffness: the default 2.5e3 compresses a 5-cube stack 2.05 mm (static, no
# drift); 2.5e4 -> 0.4 mm; 1e5 is unstable at 240 Hz. Restitution is then set by kd through a
# measured map (docs/results/l0_mujoco_restitution_map.json). Joint limits: the default soft
# limit lets a 2 N pull overshoot 7.2 mm; limit_ke 1e5 / limit_kd 1e3 -> 0.04 mm.
MUJOCO_KE = 2.5e4
MUJOCO_LIMIT = {"limit_ke": 1.0e5, "limit_kd": 1.0e3}
SUBSTEPS = {"mujoco": 8, "mujoco_ncd": 8, "kamino": 8, "xpbd": 16, "vbd": 20}   # per 1/30 s frame
I = wp.quat_identity()


def joint_limit_cfg(solver):
    """Extra joint-limit settings per solver (Kamino's limits are hard constraints already)."""
    return dict(MUJOCO_LIMIT) if solver.startswith("mujoco") else {}


def new_builder(solver, **kw):
    b = newton.ModelBuilder(**kw)
    if solver == "kamino":
        newton.solvers.SolverKamino.register_custom_attributes(b)
        b.default_shape_cfg.margin = 0.0
        b.default_shape_cfg.gap = 0.0
    return b


def shape_cfg(solver, mu=0.5, restitution=0.0, density=500.0, ke=None, kd=None, collide=True):
    c = newton.ModelBuilder.ShapeConfig()
    c.mu = float(mu)
    c.density = float(density)
    c.restitution = float(restitution)
    if solver == "kamino":
        c.margin = 0.0
        c.gap = 0.0
    if solver == "vbd" and ke is None:                     # Newton's example: stiff, undamped
        ke, kd = 1.0e8, 0.0
    if solver.startswith("mujoco") and ke is None:
        ke = MUJOCO_KE
    if ke is not None:
        c.ke = float(ke)
    if kd is not None:
        c.kd = float(kd)
    if not collide:
        c.has_shape_collision = False
        c.has_particle_collision = False
    return c


class Teacher:
    """Finalize a builder for one solver and step it at `fps` with `substeps` per frame.

    pre_substep(teacher, t): called before every substep (prescribed motion, applied forces).
    force_pairs: [(a, b), ...] body pairs whose contact force on a by b is averaged per frame.
    """

    def __init__(self, builder, solver, fps=30.0, substeps=None, init_qd=None, prepare=None,
                 pre_substep=None, post_substep=None, force_pairs=(), contact_forces=False, **opt):
        self.name = solver
        if solver == "vbd":
            builder.color()
        if solver == "kamino":
            # Kamino registers one material per distinct (mu, e) and names it after the shape's
            # label; replicated worlds repeat labels, so per-world parameters would collide
            builder.shape_label = [f"{l or 'shape'}#{i}" for i, l in enumerate(builder.shape_label)]
        self.model = builder.finalize(**({"skip_validation_joints": True} if solver == "kamino" else {}))
        if prepare is not None:
            prepare(self.model)
        if init_qd is not None:                            # generalized velocities; free joint = (v, w)
            qd = self.model.joint_qd.numpy()
            for i, v in init_qd.items():
                qd[i:i + len(v)] = v
            self.model.joint_qd.assign(qd)
        self.fps, self.substeps = fps, substeps or SUBSTEPS[solver]
        self.dt = 1.0 / (fps * self.substeps)
        self.t = 0.0
        self.pre_substep = pre_substep
        self.post_substep = post_substep
        self.force_pairs = list(force_pairs)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        S = newton.solvers
        self.pipe = None
        if solver in ("mujoco", "mujoco_ncd"):
            self.solver = S.SolverMuJoCo(self.model, cone="elliptic", impratio=opt.get("impratio", 10.0),
                                         iterations=opt.get("iterations", 100), ls_iterations=50,
                                         njmax=opt.get("njmax", 600), nconmax=opt.get("nconmax", 150),
                                         use_mujoco_contacts=(solver == "mujoco"))
            if solver == "mujoco_ncd":
                self.pipe = newton.CollisionPipeline(self.model)
        elif solver == "kamino":
            cfg = S.SolverKamino.Config.from_model(self.model)
            cfg.use_collision_detector = True
            cfg.dynamics.preconditioning = True
            cfg.padmm.primal_tolerance = cfg.padmm.dual_tolerance = cfg.padmm.compl_tolerance = 1e-6
            cfg.padmm.max_iterations = opt.get("iterations", 200)
            cfg.padmm.rho_0 = 0.1
            cfg.padmm.use_acceleration = True
            cfg.padmm.warmstart_mode = "containers"
            cfg.padmm.contact_warmstart_method = "geom_pair_net_force"
            # the default contact capacity is a product over all shape pairs and overflows int32
            # for a few hundred mesh worlds; bound it per world
            if cfg.collision_detector is None:
                from newton._src.solvers.kamino.config import CollisionDetectorConfig
                cfg.collision_detector = CollisionDetectorConfig()
            cfg.collision_detector.max_contacts_per_world = opt.get("max_contacts_per_world", 512)
            self.solver = S.SolverKamino(self.model, config=cfg)
            self.solver.reset(state=self.s0)
            if init_qd is not None:                        # reset() restores the model's state
                self.s0.joint_qd.assign(self.model.joint_qd)
                newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        elif solver == "xpbd":
            self.solver = S.SolverXPBD(self.model, iterations=opt.get("iterations", 20), enable_restitution=True)
            self.pipe = newton.CollisionPipeline(self.model)
        elif solver == "vbd":
            self.solver = S.SolverVBD(self.model, iterations=opt.get("iterations", 10), rigid_compliant_alm=True)
            self.pipe = newton.CollisionPipeline(self.model)
        else:
            raise ValueError(solver)
        # contact storage: our pipeline's, or a container the solver exports its own contacts into
        self.contacts = (self.pipe or newton.CollisionPipeline(self.model)).contacts()
        self.obs = None
        if self.force_pairs or contact_forces:
            self.obs = self.solver.observables(contacts=self.contacts,
                                               kinds={S.SolverBase.ObservableKind.CONTACT_F})
        self.shape_body = self.model.shape_body.numpy()
        self.frame_forces = {p: np.zeros(3) for p in self.force_pairs}

    def _pair_forces(self):
        n = int(self.contacts.rigid_contact_count.numpy()[0])
        out = {p: np.zeros(3) for p in self.force_pairs}
        if n == 0:
            return out
        s0 = self.contacts.rigid_contact_shape0.numpy()[:n]
        s1 = self.contacts.rigid_contact_shape1.numpy()[:n]
        ok = (s0 >= 0) & (s1 >= 0)
        b0, b1 = np.full(n, -2), np.full(n, -2)
        b0[ok], b1[ok] = self.shape_body[s0[ok]], self.shape_body[s1[ok]]
        f = self.obs.contact_f.numpy()[:n, :3]
        for a, b in self.force_pairs:                      # force on a exerted by b
            out[(a, b)] = f[(b0 == a) & (b1 == b)].sum(0) - f[(b0 == b) & (b1 == a)].sum(0)
        return out

    def frame(self):
        acc = {p: np.zeros(3) for p in self.force_pairs}
        for _ in range(self.substeps):
            self.s0.clear_forces()
            if self.pre_substep is not None:
                self.pre_substep(self, self.t)
            if self.pipe is not None:
                self.pipe.collide(self.s0, self.contacts)
            if self.obs is not None:
                self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt, observables=self.obs)
            else:
                self.solver.step(self.s0, self.s1, self.control, None if self.pipe is None and self.name != "kamino"
                                 else self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0
            self.t += self.dt
            if self.post_substep is not None:
                self.post_substep(self)
            if self.force_pairs:
                for p, v in self._pair_forces().items():
                    acc[p] += v
        self.frame_forces = {p: v / self.substeps for p, v in acc.items()}

    def rollout(self, seconds, forces=False):
        """Body poses (n+1, B, 7) and velocities (n+1, B, 6: linear, angular) at every frame;
        with forces=True also {pair: (n, 3)} frame-averaged contact forces."""
        n = int(round(seconds * self.fps))
        q, qd = [self.s0.body_q.numpy().copy()], [self.s0.body_qd.numpy().copy()]
        F = {p: [] for p in self.force_pairs}
        for _ in range(n):
            self.frame()
            q.append(self.s0.body_q.numpy().copy())
            qd.append(self.s0.body_qd.numpy().copy())
            for p in self.force_pairs:
                F[p].append(self.frame_forces[p])
        if forces:
            return np.array(q), np.array(qd), {p: np.array(v) for p, v in F.items()}
        return np.array(q), np.array(qd)


def set_kinematic_free_joint(teacher, joint, pos, vel, quat=(0.0, 0.0, 0.0, 1.0), ang=(0.0, 0.0, 0.0)):
    """Prescribe a kinematic free-base body (Newton's test recipe): write its joint q/qd, then
    forward kinematics for kinematic bodies only."""
    m, st = teacher.model, teacher.s0
    qs, ds = int(m.joint_q_start.numpy()[joint]), int(m.joint_qd_start.numpy()[joint])
    q, qd = st.joint_q.numpy(), st.joint_qd.numpy()
    q[qs:qs + 7] = [*pos, *quat]
    qd[ds:ds + 6] = [*vel, *ang]
    st.joint_q.assign(q)
    st.joint_qd.assign(qd)
    newton.eval_fk(m, st.joint_q, st.joint_qd, st, body_flag_filter=newton.BodyFlags.KINEMATIC)
