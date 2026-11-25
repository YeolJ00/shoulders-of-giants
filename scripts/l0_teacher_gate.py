"""L0 -- teacher gate T0: is the teacher simulator fit to distil from?

A dataset inherits every defect of the simulator that generated it, so the teacher is checked
against invariants before any episode is generated (CLAUDE.md rule 2). Candidate rigid
solvers: MuJoCo-Warp (Isaac Lab 3.0's default Newton backend) and XPBD. Checks:

  slide     cube on a tilted static plane, both shapes at friction mu: along-slope
            acceleration vs g (sin a - mu cos a), within 2%
  rest      cube on a 20 deg plane with mu above tan 20: drift < 1 mm/s, penetration < 2 mm
  bounce    sphere dropped 0.20 m: restitution e = v_out / v_in against the solver's contact
            damping (MuJoCo: ke/kd -> solref; XPBD: its restitution field). This is a
            calibration map, stored, not a pass/fail.
  momentum  two cubes, no gravity, head-on at 1 m/s: total momentum conserved to 1e-3
  throughput  N parallel worlds of a 6-cube scene: simulated seconds per wall second

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l0_teacher_gate.py [all|slide|rest|bounce|momentum|throughput] [--solver mujoco|xpbd]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l0_teacher_gate.json"

if "CUDA_VISIBLE_DEVICES" not in os.environ:          # pick an idle card before CUDA starts
    from lsim.gpu import wait_for_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(wait_for_gpu(min_free_gib=8.0, max_util=20))

import warp as wp                                     # noqa: E402
import newton                                         # noqa: E402

G = 9.81
FPS = 30.0
EDGE = 0.07                                           # 7 cm cube, the Physics 101 cardboard size


def quat_x(deg):
    a = np.radians(deg) / 2
    return wp.quat(float(np.sin(a)), 0.0, 0.0, float(np.cos(a)))


def shape_cfg(mu, ke=None, kd=None, restitution=0.0, density=500.0):
    c = newton.ModelBuilder.ShapeConfig()
    c.mu = float(mu)
    c.density = float(density)
    c.restitution = float(restitution)
    if ke is not None:
        c.ke = float(ke)
    if kd is not None:
        c.kd = float(kd)
    return c


def make_solver(model, name, **kw):
    if name == "mujoco":
        return newton.solvers.SolverMuJoCo(model, cone=kw.get("cone", "elliptic"),
                                           iterations=kw.get("iterations", 100),
                                           ls_iterations=kw.get("ls_iterations", 50),
                                           impratio=kw.get("impratio", 10.0),          # 1 creeps 0.9 mm/s on a 20 deg ramp; 10: 0.09
                                           njmax=kw.get("njmax", 600), nconmax=kw.get("nconmax", 150))
    if name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=kw.get("iterations", 20),
                                         enable_restitution=kw.get("enable_restitution", False))
    raise ValueError(name)


class Run:
    """Step a finalized model with one solver, recording body poses and velocities per frame.

    MuJoCo-Warp computes its own contacts (use_mujoco_contacts, the Isaac Lab default); the
    other solvers get Newton's CollisionPipeline in deterministic mode."""

    def __init__(self, builder, solver_name, substeps, init_qd=None, prepare=None, **kw):
        self.model = builder.finalize()
        if prepare is not None:                            # e.g. per-world materials
            prepare(self.model)
        if init_qd is not None:                            # generalized velocities, free joints: (v, w)
            qd = self.model.joint_qd.numpy()
            for i, v in init_qd.items():
                qd[i:i + len(v)] = v
            self.model.joint_qd.assign(qd)
        self.solver = make_solver(self.model, solver_name, **kw)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipe = None if solver_name == "mujoco" else newton.CollisionPipeline(self.model, deterministic=True)
        self.contacts = self.pipe.contacts() if self.pipe else None
        self.dt = 1.0 / (FPS * substeps)
        self.substeps = substeps
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)

    def frame(self):
        for _ in range(self.substeps):
            self.s0.clear_forces()
            if self.pipe:
                self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0

    def rollout(self, seconds):
        n = int(round(seconds * FPS))
        q = [self.s0.body_q.numpy().copy()]
        qd = [self.s0.body_qd.numpy().copy()]
        for _ in range(n):
            self.frame()
            q.append(self.s0.body_q.numpy().copy())
            qd.append(self.s0.body_qd.numpy().copy())
        return np.array(q), np.array(qd)          # (n+1, bodies, 7), (n+1, bodies, 6): qd = (v, w)


def ramp_scene(tilt, mu_ramp, mu_cube, cube_along=-0.3, edge=EDGE, density=500.0, **cfg):
    """Static 2 m x 1 m ramp tilted about x by `tilt`; a cube resting on it, seated exactly."""
    b = newton.ModelBuilder()
    q = quat_x(tilt)
    half_t = 0.05
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), q), hx=0.5, hy=1.0, hz=half_t,
                    cfg=shape_cfg(mu_ramp, **cfg))
    a = np.radians(tilt)
    n = np.array([0.0, -np.sin(a), np.cos(a)])            # ramp normal (rotation about +x)
    t = np.array([0.0, np.cos(a), np.sin(a)])             # up-slope tangent
    c = n * (half_t + edge / 2) + t * (-cube_along)       # cube centre, uphill of the middle
    body = b.add_body(xform=wp.transform(wp.vec3(*c), q), label="cube")
    b.add_shape_box(body, hx=edge / 2, hy=edge / 2, hz=edge / 2, cfg=shape_cfg(mu_cube, density=density, **cfg))
    return b, n, -t                                       # downhill direction


def check_slide(solver, substeps):
    rows = []
    for tilt in (20.0, 30.0):
        for mu in (0.10, 0.20, 0.30):
            if mu >= np.tan(np.radians(tilt)) - 0.05:
                continue
            b, n, down = ramp_scene(tilt, mu, mu)
            r = Run(b, solver, substeps)
            q, _ = r.rollout(0.6)
            s = (q[:, 0, :3] - q[0, 0, :3]) @ down
            tt = np.arange(len(s)) / FPS
            m = tt >= 0.1                                  # skip the first contact transient
            a_fit = 2 * np.polyfit(tt[m], s[m], 2)[0]
            a_th = G * (np.sin(np.radians(tilt)) - mu * np.cos(np.radians(tilt)))
            rows.append({"tilt": tilt, "mu": mu, "a_sim": float(a_fit), "a_law": float(a_th),
                         "rel_err": float(a_fit / a_th - 1), "pass": bool(abs(a_fit / a_th - 1) < 0.02)})
            print(f"  slide {solver:6s} tilt {tilt:4.1f} mu {mu:.2f}: a {a_fit:6.3f} vs law {a_th:6.3f} "
                  f"({100*(a_fit/a_th-1):+5.1f}%)", flush=True)
    return rows


def cube_corners(qrow):
    p, qq = qrow[:3], qrow[3:]
    rot = wp.quat(*[float(x) for x in qq])
    h = EDGE / 2
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                v = wp.quat_rotate(rot, wp.vec3(sx * h, sy * h, sz * h))
                out.append(p + np.array([v[0], v[1], v[2]]))
    return np.array(out)


def check_rest(solver, substeps):
    tilt, mu = 20.0, 0.6
    b, n, down = ramp_scene(tilt, mu, mu)
    r = Run(b, solver, substeps)
    q, _ = r.rollout(2.0)
    s = (q[:, 0, :3] - q[0, 0, :3]) @ down
    drift = float((s[-1] - s[int(FPS)]) / 1.0)           # m/s over the second second
    pen = []
    for k in range(0, len(q), 3):
        d = (cube_corners(q[k, 0]) @ n) - 0.05             # signed distance to the ramp top face
        pen.append(float(max(0.0, -d.min())))
    row = {"tilt": tilt, "mu": mu, "drift_mm_s": 1000 * drift, "max_penetration_mm": 1000 * max(pen),
           "pass": bool(abs(drift) < 1e-3 and max(pen) < 2e-3)}
    print(f"  rest  {solver:6s} drift {1000*drift:+.3f} mm/s, max penetration {1000*max(pen):.2f} mm", flush=True)
    return row


def _drop_e(solver, substeps, h, R=0.035, density=500.0, **kw):
    """Restitution from heights, e = sqrt(apex rise / drop), so the 30 fps frame grid does not
    bias it (finite-differenced speeds at 30 fps were 10-20% low: 0.44 vs 0.54 at kd 20)."""
    b = newton.ModelBuilder()
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
                    hx=1.0, hy=1.0, hz=0.05, cfg=shape_cfg(0.5, **kw))
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, h + R), wp.quat_identity()))
    b.add_shape_sphere(body, radius=R, cfg=shape_cfg(0.5, density=density, **kw))
    r = Run(b, solver, substeps * 4, enable_restitution=("restitution" in kw))
    q, qd = r.rollout(np.sqrt(2 * h / G) + 0.5)
    z = q[:, 0, 2] - R                                    # height of the lowest point
    k = int(np.argmax(qd[:, 0, 2] > 0)) if (qd[:, 0, 2] > 0).any() else len(z) - 1   # first upward frame
    rise = float(z[k:].max()) if (qd[:, 0, 2] > 0).any() else 0.0
    return float(np.sqrt(max(rise, 0.0) / h)), rise


def check_bounce(solver, substeps):
    rows = []
    settings = ([("kd", kd) for kd in (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)] if solver == "mujoco"
                else [("restitution", e) for e in (0.0, 0.25, 0.5, 0.75, 0.9)])
    for key, val in settings:
        kw = {"kd": val, "ke": 2.5e3} if key == "kd" else {"restitution": val}
        e, rise = _drop_e(solver, substeps, 0.20, **kw)
        rows.append({key: val, "drop_m": 0.20, "rise_m": rise, "e": e})
        print(f"  bounce {solver:6s} {key}={val:<7g} e = {e:.3f}  (rise {100*rise:.1f} cm of 20 cm)", flush=True)
    return rows


def check_momentum(solver, substeps):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x in (0.0, -0.12):
        body = b.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.5), wp.quat_identity()))
        b.add_shape_box(body, hx=EDGE / 2, hy=EDGE / 2, hz=EDGE / 2, cfg=shape_cfg(0.5))
    r = Run(b, solver, substeps, init_qd={6: [1.0, 0.0, 0.0]})   # second body moves +x at 1 m/s
    m = r.model.body_mass.numpy()
    q, qd = r.rollout(0.3)
    lin = qd[:, :, :3]
    P = (lin * m[None, :, None]).sum(1)
    rel = float(np.abs(P[-1] - P[0]).max() / (np.abs(P[0]).max() + 1e-12))
    row = {"P0": P[0].tolist(), "P1": P[-1].tolist(), "rel_change": rel, "pass": bool(rel < 1e-3),
           "vA_end": lin[-1, 0].tolist(), "vB_end": lin[-1, 1].tolist()}
    print(f"  momentum {solver:6s} P0 {P[0][0]:+.4f}  P_end {P[-1][0]:+.4f}  rel change {rel:.2e}; "
          f"A {lin[-1,0,0]:+.3f} B {lin[-1,1,0]:+.3f} m/s", flush=True)
    return row


def _scene(kind, rng):
    import trimesh
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    for k in range(6):
        body = scene.add_body(xform=wp.transform(wp.vec3(*(rng.uniform(-0.15, 0.15, 2)), 0.05 + 0.09 * k),
                                                 wp.quat_identity()))
        if kind == "box":
            scene.add_shape_box(body, hx=0.03, hy=0.04, hz=0.03, cfg=shape_cfg(0.5))
        else:                                              # a convex mesh of ~300 vertices, mug-sized
            m = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
            m.apply_scale([1.0, 0.8, 1.2])
            mesh = newton.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32).flatten())
            scene.add_shape_mesh(body, mesh=mesh, cfg=shape_cfg(0.5))
    return scene


def check_throughput(solver, substeps, worlds=(64, 256, 1024)):
    rows = []
    for kind in ("box", "mesh"):
        for nw in worlds:
            b = newton.ModelBuilder()
            b.replicate(_scene(kind, np.random.default_rng(0)), nw)
            r = Run(b, solver, substeps)
            r.frame(); wp.synchronize()                    # compile + warm up
            t0 = time.time()
            nf = 60                                        # 2 s: objects fall, land and pile up
            for _ in range(nf):
                r.frame()
            wp.synchronize()
            wall = time.time() - t0
            sim_s = nw * nf / FPS
            z = r.s0.body_q.numpy()[:, 2]
            rows.append({"kind": kind, "worlds": nw, "substeps": substeps,
                         "sim_seconds_per_wall_second": sim_s / wall,
                         "episodes_6s_per_hour": sim_s / wall * 3600 / 6,
                         "min_body_z": float(z.min())})
            print(f"  throughput {solver:6s} {kind:4s} {nw:5d} worlds x 6: {sim_s/wall:8.1f} sim-s/wall-s = "
                  f"{sim_s/wall*3600/6:9.0f} six-second episodes/hour  (lowest body z {z.min():+.3f} m)", flush=True)
    return rows


def check_restitution_dependence(solver, substeps):
    """MuJoCo maps (ke, kd) to a mass-normalised solref, so e should not depend on mass; our
    penalty contact gave e as a function of cd/m (p7). Check mass and impact speed."""
    rows = []
    for dens in (100.0, 500.0, 2500.0):
        for h in (0.05, 0.20, 0.50):
            e, rise = _drop_e(solver, substeps, h, density=dens, kd=20.0, ke=2.5e3)
            rows.append({"density": dens, "drop_m": h, "v_in": float(np.sqrt(2 * G * h)), "e": e})
            print(f"  e-dependence {solver:6s} density {dens:6.0f}  drop {h:.2f} m (v {np.sqrt(2*G*h):.2f}): e = {e:.3f}", flush=True)
    return rows


def check_rest_impratio(solver, substeps):
    rows = []
    for imp in (1.0, 10.0, 100.0):
        b, n, down = ramp_scene(20.0, 0.6, 0.6)
        r = Run(b, solver, substeps, impratio=imp)
        q, _ = r.rollout(3.0)
        s = (q[:, 0, :3] - q[0, 0, :3]) @ down
        drift = float(s[-1] - s[int(FPS)]) / 2.0
        rows.append({"impratio": imp, "drift_mm_s": 1000 * drift})
        print(f"  rest-impratio {solver:6s} impratio {imp:5.0f}: drift {1000*drift:+.3f} mm/s", flush=True)
    return rows


def check_phys101(solver, substeps):
    """Teacher vs reality: the 93 real Physics 101 cardboard slides (20 deg board) re-simulated
    with each cube's measured size and mass; mu swept, one world per mu; each real trial aligned
    in release time and offset exactly as p4 did for the repo's own Newton simulator."""
    from lsim.real.phys101 import align_err, objects, ramp_segment, ramp_tracks
    objs = objects()
    R = ramp_tracks("cardboard")
    ref = REPO / "archive/theta_track/docs/results/p4_sim_loop.json"   # the repo's own simulator, for comparison
    p4 = json.loads(ref.read_text()) if ref.exists() else {}
    MUS = np.round(np.arange(0.15, 0.351, 0.01), 3)
    out = {}
    for cube in sorted({r["clip"].split("/")[0] for r in R}):
        mass_g, vol_cm3 = objs[cube]
        edge = vol_cm3 ** (1 / 3) / 100.0
        dens = mass_g / 1000.0 / edge ** 3
        scene, n, down = ramp_scene(20.0, 0.3, 0.3, cube_along=-0.9, edge=edge, density=dens)
        b = newton.ModelBuilder()
        b.replicate(scene, len(MUS))

        def prepare(model, MUS=MUS):
            mu = model.shape_material_mu.numpy()
            w = model.shape_world.numpy()
            for k, u in enumerate(MUS):
                mu[w == k] = u                             # ramp and cube of world k at mu_k
            model.shape_material_mu.assign(mu)

        r = Run(b, solver, substeps, prepare=prepare)
        q, _ = r.rollout(1.2)
        curves = {float(u): (q[:, k, :3] - q[0, k, :3]) @ down for k, u in enumerate(MUS)}
        trials = [t for t in R if t["clip"].startswith(cube + "/20") and t.get("fit")]
        per = []
        for t in trials:
            rt, rs = ramp_segment(t)
            errs = np.array([align_err(rt, rs, curves[float(u)], FPS)[0] for u in MUS])
            i = int(np.argmin(errs))
            mu_hat = float(MUS[i])
            if 0 < i < len(MUS) - 1:
                a2, b2, _ = np.polyfit(MUS[i - 1:i + 2], errs[i - 1:i + 2], 2)
                mu_hat = float(-b2 / (2 * a2)) if a2 > 0 else mu_hat
            per.append({"clip": t["clip"], "mu": mu_hat, "rms_mm": 1000 * float(errs[i])})
        o = {"edge_cm": 100 * edge, "mass_g": mass_g, "n": len(per),
             "mu_teacher": float(np.median([p_["mu"] for p_ in per])),
             "rms_teacher_mm": float(np.median([p_["rms_mm"] for p_ in per])),
             "mu_repo_newton": p4.get(cube, {}).get("mu_sim_median", float("nan")),
             "rms_repo_newton_mm": p4.get(cube, {}).get("rms_sim_mm", float("nan")),
             "rms_parabola_mm": p4.get(cube, {}).get("rms_parabola_mm", float("nan")), "trials": per}
        out[cube] = o
        print(f"  phys101 {solver:6s} {cube:13s} mu teacher {o['mu_teacher']:.3f} (repo sim {o['mu_repo_newton']:.3f})  "
              f"rms teacher {o['rms_teacher_mm']:.2f} mm, repo sim {o['rms_repo_newton_mm']:.2f}, parabola {o['rms_parabola_mm']:.2f}",
              flush=True)
    allr = [p_["rms_mm"] for o in out.values() for p_ in o["trials"]]
    out["_all"] = {"n_trials": len(allr), "rms_median_mm": float(np.median(allr)),
                   "rms_p90_mm": float(np.percentile(allr, 90)), "pass": bool(np.median(allr) < 2.0)}
    print(f"  phys101 {solver:6s} ALL {len(allr)} trials: median RMS {np.median(allr):.2f} mm, p90 {np.percentile(allr, 90):.2f} mm", flush=True)
    return out


def _towel_world(tri_ke, edge_ke, areal_density=0.2, cells=30, cell=0.01):
    """A 30 x 30 cm towel (1 cm cells) dropped 2 cm onto a 75 cm table, half over the edge."""
    b = newton.ModelBuilder()
    b.add_ground_plane()
    # VBD takes particle-shape contact stiffness from the SHAPE's ke/kd: the default 2.5e3/1e2 lets
    # the towel sink through the table to the floor; 5e6/1e4 (Newton's poker-cards example) holds it
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.375), wp.quat_identity()),
                    hx=0.5, hy=0.5, hz=0.375, cfg=shape_cfg(0.6, ke=5.0e6, kd=1.0e4))
    m_particle = areal_density * (cells * cell) ** 2 / (cells + 1) ** 2
    b.add_cloth_grid(pos=wp.vec3(0.35, -0.15, 0.77), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                     dim_x=cells, dim_y=cells, cell_x=cell, cell_y=cell, mass=m_particle,
                     tri_ke=tri_ke, tri_ka=tri_ke, tri_kd=1.0e-2 * tri_ke, edge_ke=edge_ke, edge_kd=0.0,
                     particle_radius=0.003)
    b.color(include_bending=True)
    return b


def check_cloth(solver="vbd", substeps=20, seconds=3.0):
    """Cloth teacher (VBD): stays inside a spatial bound (rule 3), settles, drapes over the edge;
    plus simulated seconds per wall second for 1 and 64 worlds."""
    rows = []
    for tri_ke, edge_ke in ((1.0e3, 1.0e-2), (1.0e4, 1.0e-2), (1.0e4, 1.0)):
        for nw in (1, 64):
            scene = _towel_world(tri_ke, edge_ke)
            b = newton.ModelBuilder()
            b.replicate(scene, nw, spacing=(2.0, 0.0, 0.0))
            b.color(include_bending=True)
            model = b.finalize()
            model.soft_contact_ke = 1.0e5
            model.soft_contact_kd = 1.0e2
            model.soft_contact_mu = 0.6
            vbd = newton.solvers.SolverVBD(model, iterations=10)
            s0, s1, ctrl = model.state(), model.state(), model.control()
            pipe = newton.CollisionPipeline(model, soft_contact_gap=0.005)
            con = pipe.contacts()
            dt = 1.0 / (FPS * substeps)
            x0 = s0.particle_q.numpy().copy()
            t0 = time.time(); wp.synchronize()
            for f in range(int(seconds * FPS)):
                for _ in range(substeps):
                    s0.clear_forces()
                    pipe.collide(s0, con)
                    vbd.step(s0, s1, ctrl, con, dt)
                    s0, s1 = s1, s0
            wp.synchronize(); wall = time.time() - t0
            x = s0.particle_q.numpy(); v = s0.particle_qd.numpy()
            npw = len(x) // nw
            xw = x[:npw]                                   # world 0 (replicate offsets worlds, so
            x0w = x0[:npw]                                 # measure against its own start)
            disp = float(np.abs(x - x0).max())
            hang = float((xw[:, 2] < 0.74).mean())         # fraction below the table top
            v_w = np.linalg.norm(v[:npw], axis=1)
            on_top = xw[xw[:, 0] - x0w[:, 0].min() < 0.13]   # towel x in [0, 0.30], table edge at 0.15
            row = {"tri_ke": tri_ke, "edge_ke": edge_ke, "worlds": nw, "finite": bool(np.isfinite(x).all()),
                   "max_displacement_m": disp, "in_bound": bool(np.isfinite(x).all() and disp < 0.6 and x[:, 2].min() > -0.01),
                   "on_table_mean_z": float(on_top[:, 2].mean()) if len(on_top) else float("nan"),
                   "max_speed_end_m_s": float(v_w.max()), "p95_speed_end_m_s": float(np.percentile(v_w, 95)),
                   "median_speed_end_m_s": float(np.median(v_w)), "settled": bool(v_w.max() < 0.02), "hanging_fraction": hang,
                   "lowest_z_m": float(xw[:, 2].min()), "sim_seconds_per_wall_second": nw * seconds / wall}
            rows.append(row)
            print(f"  cloth vbd tri_ke {tri_ke:7.0e} edge_ke {edge_ke:6.0e} x{nw:3d}: in bound {row['in_bound']}, "
                  f"speed at end {row['max_speed_end_m_s']:.3f} m/s, hanging {100*hang:4.1f}%, on-table z {row['on_table_mean_z']:.4f}, lowest z {row['lowest_z_m']:+.3f}, "
                  f"{row['sim_seconds_per_wall_second']:.2f} sim-s/wall-s", flush=True)
    return rows


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    solvers = [sys.argv[sys.argv.index("--solver") + 1]] if "--solver" in sys.argv else ["mujoco", "xpbd"]
    wp.init()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    sub = {"mujoco": 8, "xpbd": 16}
    if what == "cloth":                                    # cloth has its own solver (VBD)
        res = json.loads(OUT.read_text()) if OUT.exists() else {}
        res.setdefault("vbd", {})["cloth"] = check_cloth()
        OUT.write_text(json.dumps(res, indent=1))
        return 0
    for s in solvers:
        res.setdefault(s, {})
        for name, fn in (("slide", check_slide), ("rest", check_rest), ("bounce", check_bounce),
                         ("momentum", check_momentum), ("throughput", check_throughput),
                         ("e_dependence", check_restitution_dependence), ("rest_impratio", check_rest_impratio),
                         ("phys101", check_phys101), ("cloth", check_cloth)):
            if what in ("all", name):
                try:
                    res[s][name] = fn(s, sub[s])
                except Exception as exc:                    # record the failure, keep going
                    res[s][name] = {"error": repr(exc)}
                    print(f"  {name} {s}: ERROR {exc!r}", flush=True)
                OUT.write_text(json.dumps(res, indent=1))
    res["_versions"] = {"newton": newton.__version__, "warp": wp.config.version}
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
