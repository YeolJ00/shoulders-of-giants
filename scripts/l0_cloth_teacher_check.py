"""L0 -- the cloth teacher's own checks: the cloth gate's closed forms run on the module that generates the
data (lsim.teacher.cloth_teacher), plus what the first generated episodes showed the gate missed.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l0_cloth_teacher_check.py [check ...] [--objects gso|simready|both]

The gate (scripts/l0_cloth_gate.py) chose the explicit integrator but ran its own copy of the physics, with
Newton's soft contact; the teacher has its own contact (per-cloth friction, forces recorded by surface kind),
self-contact and a kinematic gripper. Here the gate's checks run through ClothTeacher itself (`TeacherSim` has
the gate's Sim interface), and three more:

  bounce     towels dropped flat from 10 cm come to rest without rebounding. Undamped contact let a towel
             rebound 28-49 mm and keep bouncing; fabric does not bounce
  crossing   two thin quads crossing as an X, only their edges meeting, the upper falling onto the lower: it
             must rest a thickness above it (vertex-triangle repulsion alone let edges slip through each other)
  pile       cloth on cloth: a sheet dropped onto a sheet does not bounce; sheets started crushed to half the
             self-contact thickness spring back but stay bounded (with a fixed contact stiffness, light sheets went
             through the table at 290 m/s: stacked contacts outran the explicit step)
  episodes   generator batches (the natural mix; folds; plank starts; draped starts): no cloth exploded (a vertex
             moving 0.4 m in a frame), the largest contact force over the cloth's weight, and how many episodes
             end with the cloth through itself. c1's first 16,384 episodes: 26 exploded (a gripper driven
             into the plank or an object pinched cloth between two penalty surfaces), ~12% ended crossed
Results: docs/results/l0_cloth_teacher_check.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
import newton

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import l0_cloth_gate as gate                                    # noqa: E402
import lsim.teacher.cloth_teacher as ct                         # noqa: E402

OUT = REPO / "docs/results/l0_cloth_teacher_check.json"
OBJECTS = "gso"                                      # obstacle library of the generator batches (--objects)
I = wp.quat_identity()
PATH_SECONDS = 40


def _cloth(d, nx=1, ny=1):
    return ct.Cloth(nx=nx, ny=ny, h=d["h"], w=d["w"], Y=d["Y"], nu=d["nu"], B=d["B"], mu=d["mu"], r=d["r"])


def add_sheet(b, solver, cloth, nx, ny, pos, rot=None):
    """The gate's sheet, built the teacher's way; returns its particle ids."""
    p0, n, _, _ = ct.add_cloth(b, _cloth(cloth, nx, ny), pos, rot if rot is not None else I)
    return np.arange(p0, p0 + n)


def static_box(b, center, half, mu, rot=None):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu, cfg.ke, cfg.kd, cfg.kf = mu, ct.CONTACT_KE, 0.0, ct.FRICTION_KF
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*center), rot if rot is not None else I),
                    hx=half[0], hy=half[1], hz=half[2], cfg=cfg)


class TeacherSim:
    """The gate's Sim interface over ClothTeacher: the scene's fixed vertices are made inactive (they stay
    put), grasp(ids, vel) pins vertices to a kinematic gripper moving at vel, `kin = None` lets go; the
    gate's linear settling drag becomes the teacher's settle_damping."""

    def __init__(self, solver, scene, cloth, worlds=1, drag=0.0, **over):
        wb = newton.ModelBuilder()
        wb.add_ground_plane = lambda *a, **k: None               # the table is all the gate's scenes stand on
        self.meta = scene(wb, solver, cloth)
        npw = wb.particle_count
        g = wb.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 5.0), I), mass=0.0, label="gripper")
        c = _cloth(cloth)
        b = newton.ModelBuilder()
        ws = []
        for _ in range(worlds):
            po, eo, bo = b.particle_count, b.edge_count, b.body_count
            b.add_world(wb)
            ws.append({"particles": (po, npw), "edges": (eo, wb.edge_count), "cloth": c, "gripper": bo + g})
        self.path = np.zeros((PATH_SECONDS * ct.FPS + 1, worlds, 7), np.float32)
        self.path[:, :, 2] = 5.0
        self.path[:, :, 6] = 1.0
        ct.ClothTeacher.MAX_ATTACH = 64
        self.teacher = ct.ClothTeacher(b, ws, [ct.ROLE_TABLE] * b.shape_count, self.path, ct.stable_substeps([c]),
                                       self_contact=bool(over.get("self_contact", 1)), settle_damping=drag)
        self.npw, self.worlds = npw, worlds
        m = self.teacher.model.particle_mass.numpy()
        fixed = np.asarray(self.meta.get("fixed", []), int)
        if len(fixed):
            allf = (fixed[None, :] + npw * np.arange(worlds)[:, None]).ravel()
            self.teacher.flags[allf] &= ~int(newton.ParticleFlags.ACTIVE)
            self.teacher.model.particle_flags.assign(self.teacher.flags)
            m[allf] = 0.0                                         # the gate's convention: fixed = massless
        self.mass = m
        self._kin, self.graph, self.t, self.wall, self.last = None, None, 0.0, 0.0, None

    @property
    def kin(self):
        return self._kin

    @kin.setter
    def kin(self, value):
        if value is None:
            for w in range(self.worlds):
                self.teacher.release(w)
        self._kin = value

    def grasp(self, ids, vel):
        f = self.teacher.frame_index
        k = np.arange(self.path.shape[0] - f, dtype=np.float32)[:, None, None] / ct.FPS
        self.path[f:, :, :3] = self.path[f, :, :3][None] + k * np.asarray(vel, np.float32)[None, None, :]
        self.teacher.path.assign(self.path)
        for w in range(self.worlds):
            self.teacher.grasp_ids(w, ids)
        self._kin = ids

    def grasp_force(self):
        return np.asarray(self.last["grasp"][0], np.float64)

    def frame(self, n=1):
        t0 = time.time()
        for _ in range(n):
            self.last = self.teacher.frame()
            self.t += 1.0 / ct.FPS
        wp.synchronize()
        self.wall += time.time() - t0

    def rate(self):
        return self.worlds * self.t / max(self.wall, 1e-9)

    def x(self):
        return self.teacher.s0.particle_q.numpy()[:self.npw].astype(np.float64)

    def v(self):
        return self.teacher.s0.particle_qd.numpy()[:self.npw].astype(np.float64)


gate.Sim, gate.add_sheet, gate.static_box = TeacherSim, add_sheet, static_box


def check_freefall(solver, **over):
    rho = ct.RHO_CD
    ct.RHO_CD = 0.0                                               # the closed form is drag-free
    try:
        return gate.check_freefall(solver, **over)
    finally:
        ct.RHO_CD = rho


def check_airfall(solver, **over):
    """A flat sheet falling face-down reaches the terminal speed sqrt(2 w g / (rho Cd)) (5%)."""
    def scene(b, solver, cloth):
        return {"ids": add_sheet(b, solver, cloth, 20, 20, (-0.1, -0.1, 10.0))}
    sim = TeacherSim(solver, scene, gate.CLOTH)
    sim.frame(int(3.0 * ct.FPS))
    v = float(-sim.v()[:, 2].mean())
    want = float(np.sqrt(2.0 * gate.CLOTH["w"] * ct.G / ct.RHO_CD))
    out = {"terminal_m_s": v, "closed_form_m_s": want, "ratio": v / want, "pass": bool(abs(v / want - 1.0) < 0.05)}
    print(f"  airfall {solver}: falls at {v:.3f} m/s after 3 s, terminal speed {want:.3f} (ratio {v / want:.3f})", flush=True)
    return out


def check_bounce(solver, **over):
    """Towels (light, medium, heavy) dropped flat from 10 cm: after first touch the centroid rises at most
    2 mm above where it comes to rest, and no vertex moves up faster than 0.1 m/s."""
    out = {}
    for tag, w, Y, lb in (("light", 0.05, 1.0e3, 0.02), ("medium", 0.2, 3.0e3, 0.03), ("heavy", 0.5, 1.0e4, 0.04)):
        cloth = dict(gate.CLOTH, w=w, Y=Y, B=w * ct.G * lb ** 3)
        def scene(b, solver, cloth):
            static_box(b, (0.0, 0.0, -0.05), (0.6, 0.5, 0.05), cloth["mu"])
            return {"ids": add_sheet(b, solver, cloth, 30, 30, (-0.15, -0.15, 0.10))}
        sim = TeacherSim(solver, scene, cloth)
        cz, vup, touch = [], [], None
        prev = sim.x()
        for f in range(int(1.5 * ct.FPS)):
            sim.frame()
            x = sim.x()
            cz.append(float(x[:, 2].mean()))
            if touch is None and x[:, 2].min() < cloth["r"] + 0.0015:
                touch = f
            if touch is not None:
                vup.append(float(((x[:, 2] - prev[:, 2]) * ct.FPS).max()))
            prev = x
        rebound = float(max(cz[touch:]) - cz[-1])
        row = {"rebound_mm": 1e3 * rebound, "max_upward_speed_m_s": max(vup), "rest_height_mm": 1e3 * cz[-1],
               "pass": bool(rebound < 0.002 and max(vup) < 0.1)}
        out[tag] = row
        print(f"  bounce  {tag:6s} w {w}: rebound {1e3 * rebound:.2f} mm above rest ({1e3 * cz[-1]:.2f} mm), fastest upward "
              f"vertex after touch {max(vup):.3f} m/s", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_crossing(solver, **over):
    """Two long thin quads crossing as an X, no vertex over the other's triangles, the upper given 1 m/s
    downward: with edge-edge repulsion it rests a thickness above the lower; without, it passes through."""
    out = {}
    for edges in (False, True):
        ct.SELF_EDGES, keep = edges, ct.SELF_EDGES
        try:
            b = newton.ModelBuilder()
            static_box(b, (0.0, 0.0, -0.05), (0.5, 0.5, 0.05), 0.5)
            c = ct.Cloth(nx=1, ny=1, h=0.01, w=0.2, Y=3e3, B=1e-6, mu=0.5)
            quads = [((0.0, -0.02, 0.004), (0.0, 0.03, 0.004), (0.005, -0.02, 0.004), (0.005, 0.03, 0.004), 0.0),
                     ((-0.02, 0.0, 0.012), (0.03, 0.0, 0.012), (-0.02, 0.005, 0.012), (0.03, 0.005, 0.012), -1.0)]
            for *q, vz in quads:
                b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=I, scale=1.0, vel=wp.vec3(0.0, 0.0, vz),
                                 vertices=[wp.vec3(*p) for p in q], indices=[0, 1, 2, 2, 1, 3], density=c.w,
                                 tri_ke=1e3, tri_ka=1e3, tri_kd=0.0, tri_drag=0.0, tri_lift=0.0, edge_ke=0.0, edge_kd=0.0,
                                 particle_radius=c.r)
            g = b.add_body(xform=wp.transform(wp.vec3(0.0, -0.75, 0.35), I), mass=0.0)
            path = np.zeros((32, 1, 7), np.float32)
            path[:, :, :3] = (0.0, -0.75, 0.35)
            path[:, :, 6] = 1.0
            t = ct.ClothTeacher(b, [{"particles": (0, 8), "edges": (0, b.edge_count), "cloth": c, "gripper": g}],
                                [ct.ROLE_TABLE], path, 1000)
            for _ in range(30):
                x = t.frame()["x"]
            gap = float(x[4:, 2].mean() - x[:4, 2].mean())
        finally:
            ct.SELF_EDGES = keep
        out[f"edges{int(edges)}"] = {"upper_over_lower_mm": 1e3 * gap}
        print(f"  crossing edge-edge {'on ' if edges else 'off'}: the upper quad rests {1e3 * gap:+.2f} mm over the lower", flush=True)
    out["pass"] = bool(out["edges1"]["upper_over_lower_mm"] > 0.5e3 * ct.SELF_THICK and out["edges0"]["upper_over_lower_mm"] < 0.5e3 * ct.SELF_THICK)
    return out


def _static_world(b, cloth, frames):
    g = b.add_body(xform=wp.transform(wp.vec3(0.0, -0.75, 0.35), I), mass=0.0)
    path = np.zeros((frames + 2, 1, 7), np.float32)
    path[:, :, :3] = (0.0, -0.75, 0.35)
    path[:, :, 6] = 1.0
    n = b.particle_count
    return ct.ClothTeacher(b, [{"particles": (0, n), "edges": (0, b.edge_count), "cloth": cloth, "gripper": g}],
                           [ct.ROLE_TABLE] * b.shape_count, path, 1000)


def check_pile(solver, **over):
    """Cloth on cloth. A sheet dropped flat from 3 cm onto one lying on the table does not bounce: it rises at
    most 2 mm after touching, no vertex faster than 0.1 m/s upward. Sheets started crushed to 1-1.5 mm apart (half
    the self-contact thickness: a press no guarded gripper makes) spring back -- every vertex meets a dozen
    overlapping vertex-triangle and edge-edge contacts, each a spring -- but stay bounded: no vertex past 20 m/s,
    none through the table (with a fixed contact stiffness, light sheets left through it at 290 m/s)."""
    out = {}
    for w in (0.05, 0.5):
        b = newton.ModelBuilder()
        static_box(b, (0.0, 0.0, -0.05), (0.6, 0.5, 0.05), 0.5)
        c = ct.Cloth(nx=20, ny=20, w=w, Y=2000.0, B=w * ct.G * 0.03 ** 3, mu=0.5)
        ct.add_cloth(b, c, (-0.1, -0.1, c.r + 0.0002), I)
        ct.add_cloth(b, c, (-0.1, -0.1, c.r + 0.0302), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.2))
        t = _static_world(b, c, 30)
        n0 = (c.nx + 1) * (c.ny + 1)
        zs, vup, x0 = [], [], t.s0.particle_q.numpy().copy()
        for _ in range(30):
            x = t.frame()["x"]
            zs.append(float(x[n0:, 2].mean()))
            vup.append(float(((x[n0:, 2] - x0[n0:, 2]) * ct.FPS).max()))
            x0 = x.copy()
        zs = np.array(zs)
        touch = int(np.argmax(zs < zs[-1] + 0.002))
        row = {"rebound_mm": 1e3 * float(zs[touch:].max() - zs[-1]), "max_upward_m_s": max(vup[touch:])}
        row["pass"] = bool(row["rebound_mm"] < 2.0 and row["max_upward_m_s"] < 0.1)
        out[f"drop_w{w}"] = row
        print(f"  pile    w {w} sheet dropped 3 cm onto a sheet: rebound {row['rebound_mm']:.2f} mm, fastest upward "
              f"{row['max_upward_m_s']:.3f} m/s", flush=True)
    for w in (0.05, 0.5):
        for layers, gap in ((4, 0.0015), (6, 0.001)):
            b = newton.ModelBuilder()
            static_box(b, (0.0, 0.0, -0.05), (0.6, 0.5, 0.05), 0.5)
            c = ct.Cloth(nx=12, ny=12, w=w, Y=400.0, B=w * ct.G * 0.03 ** 3, mu=0.5)
            rng = np.random.default_rng(0)
            for k in range(layers):
                rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.3, 0.3)))
                ct.add_cloth(b, c, (-0.06 + rng.uniform(-0.005, 0.005), -0.06 + rng.uniform(-0.005, 0.005),
                                    c.r + 0.0005 + k * gap), rot)
            t = _static_world(b, c, 30)
            x0 = t.s0.particle_q.numpy().copy()
            vmax, zmin = 0.0, 1.0
            for _ in range(30):
                x = t.frame()["x"]
                vmax = max(vmax, float(np.linalg.norm(x - x0, axis=1).max() * ct.FPS))
                zmin = min(zmin, float(x[:, 2].min()))
                x0 = x.copy()
            row = {"fastest_m_s": vmax, "lowest_mm": 1e3 * zmin, "candidates_vt_ee": list(t.candidate_load()),
                   "pass": bool(vmax < 20.0 and zmin > -c.r)}
            out[f"crushed_w{w}_{layers}x{1e3 * gap:g}mm"] = row
            print(f"  pile    w {w} {layers} sheets crushed {1e3 * gap:.1f} mm apart: fastest vertex {vmax:.3f} m/s, lowest "
                  f"{1e3 * zmin:.2f} mm, candidates {row['candidates_vt_ee']}", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_episodes(solver, **over):
    """Generator batches of 64: the natural mix, folds, plank starts, draped starts."""
    import torch
    import lsim.data.cloth as cl
    from lsim.data import library
    train, _ = library.object_split(OBJECTS)          # --objects gso (default) | simready | both
    out = {}
    for tag, seed, init, family in (("mix", 101, None, None), ("fold", 102, None, "fold"), ("ramp", 103, "ramp", None),
                                    ("draped", 104, "draped", None)):
        keep = cl.INIT_P.copy(), cl.FAMILY_P.copy()
        if init:
            cl.INIT_P = np.array([1.0 if i == init else 0.0 for i in cl.INITS])
        if family:
            cl.FAMILY_P = np.array([1.0 if f == family else 0.0 for f in cl.FAMILIES])
        try:
            rng = np.random.default_rng(seed)
            specs = [cl.sample_spec(rng, train, seed=seed * 10000 + k) for k in range(64)]
            specs.sort(key=lambda sp: sp.cloth.vertex_mass / sp.cloth.Y)
            t0 = time.time()
            res = cl.run_batch(specs, seed=seed, log=lambda m: None)
            wall = time.time() - t0
        finally:
            cl.INIT_P, cl.FAMILY_P = keep
        qc = cl.quality(res)
        crossed = []
        for wd in res["worlds"]:
            p0, n = wd["particles"]
            c = wd["cloth"]
            _, tris = ct.grid(c.nx, c.ny, c.h)
            crossed.append(_crossings(res["x"][-1, p0:p0 + n], tris, torch))
        cw = np.array([max(q["contact_over_weight"]) for q in qc])
        row = {"episodes": len(qc), "exploded": int(sum(q["exploded"] for q in qc)),
               "crossed_at_end": int(sum(k > 0 for k in crossed)), "crossings": [int(k) for k in crossed if k],
               "contact_over_weight_p99": float(np.percentile(cw, 99)), "contact_over_weight_max": float(cw.max()),
               "fastest_vertex_m_s": float(max(q["vmax"] for q in qc)), "substeps": res["substeps"], "wall_s": wall,
               "candidates_vt_ee": list(res["teacher"].candidate_load())}
        row["pass"] = bool(row["exploded"] == 0 and row["crossed_at_end"] <= 0.05 * len(qc))
        out[tag] = row
        print(f"  episodes {tag:6s}: {row['exploded']} exploded, {row['crossed_at_end']}/64 end crossed {row['crossings']}, "
              f"contact/weight p99 {row['contact_over_weight_p99']:.0f} max {row['contact_over_weight_max']:.0f}, fastest vertex "
              f"{row['fastest_vertex_m_s']:.2f} m/s, {wall:.0f} s, candidates {row['candidates_vt_ee']}", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def _crossings(x, tris, torch):
    """Mesh edges through triangles they do not touch (the gate's count, on the GPU)."""
    tris = np.asarray(tris).reshape(-1, 3)
    e = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(np.asarray(x, np.float64), device=dev)
    T, E = torch.as_tensor(tris, device=dev), torch.as_tensor(e, device=dev)
    a, b_, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    e1, e2 = b_ - a, c - a
    n = 0
    for s in range(0, len(E), 1024):
        Es = E[s:s + 1024]
        P, Q = X[Es[:, 0]][:, None], X[Es[:, 1]][:, None]
        share = (Es[:, None, :, None] == T[None, :, None, :]).any(-1).any(-1)
        d = Q - P
        h = torch.cross(d.expand(-1, len(T), -1), e2[None].expand(len(Es), -1, -1), dim=-1)
        det = (h * e1[None]).sum(-1)
        ok = det.abs() > 1e-14
        inv = torch.where(ok, 1.0 / torch.where(ok, det, torch.ones_like(det)), torch.zeros_like(det))
        sv = P - a[None]
        u = (sv * h).sum(-1) * inv
        qv = torch.cross(sv, e1[None].expand(len(Es), -1, -1), dim=-1)
        v = (d * qv).sum(-1) * inv
        t = (e2[None] * qv).sum(-1) * inv
        n += int((ok & ~share & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)).sum())
    return n


CHECKS = {"stretch": gate.check_stretch, "swing": gate.check_swing, "freefall": check_freefall, "airfall": check_airfall,
          "slide": gate.check_slide, "bend": gate.check_bend, "drape": gate.check_drape, "fold": gate.check_fold,
          "grasp": gate.check_grasp, "bounce": check_bounce, "crossing": check_crossing, "pile": check_pile,
          "episodes": check_episodes,
          "throughput": gate.check_throughput}


def main():
    global OUT, OBJECTS
    args = sys.argv[1:]
    if "--objects" in args:                                # results apart from the GSO ones
        k = args.index("--objects")
        OBJECTS = args[k + 1]
        args = args[:k] + args[k + 2:]
        if OBJECTS != "gso":
            OUT = OUT.with_name(OUT.stem + f"_{OBJECTS}.json")
    names = args or list(CHECKS)
    wp.init()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for n in names:
        try:
            res[n] = CHECKS[n]("teacher")
        except Exception as exc:                              # record the failure, keep going
            res[n] = {"error": repr(exc)}
            print(f"  {n}: ERROR {exc!r}", flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    res["_versions"] = {"newton": newton.__version__, "warp": wp.config.version,
                        "teacher": {"contact_rate": ct.CONTACT_RATE, "contact_zeta": ct.CONTACT_ZETA, "friction_rate": ct.FRICTION_RATE,
                                    "self": [ct.SELF_THICK, ct.SELF_RATE, ct.SELF_MU, ct.SELF_MARGIN], "self_edges": ct.SELF_EDGES,
                                    "membrane_kd": ct.MEMBRANE_KD, "rho_cd": ct.RHO_CD, "bend_factor": ct.BEND_FACTOR}}
    OUT.write_text(json.dumps(res, indent=1))
    print("pass: " + ", ".join(f"{k} {v.get('pass')}" for k, v in res.items() if isinstance(v, dict) and "pass" in v or "error" in v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
