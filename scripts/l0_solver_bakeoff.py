"""L0 -- solver bake-off: which Newton solver should teach?

Every candidate (see lsim.teacher.newton_teacher) on the same checks. Thresholds are set before
the run; a solver that lacks a feature records the exception, which counts as a fail.

  slide       cube on a tilted plane: acceleration vs g(sin a - mu cos a)          < 2%
  rest        cube on 20 deg at mu 0.6: drift, penetration                         < 0.2 mm/s, < 1 mm
  bounce      sphere dropped 0.2 m, restitution set to 0.2/0.5/0.8 (MuJoCo: through
              its damping map): e from rebound height                              |e - target| < 0.05
  momentum    head-on collision without gravity                                    < 1e-3
  stack       5 cubes, 5 s                                                         top cube moves < 2 mm
  concave     ball dropped into a bowl mesh: rests inside, not on the convex hull  ball z < 5 cm
  topple_com  20 cm box on a 14 deg ramp, mu 1: centre of mass at the centre stays,
              1 cm downhill topples (threshold tan = (3 - 1)/10 cm -> 11.3 deg)     both right
  drawer      prismatic joint, damping 5 N s/m, 2 N pull: v(0.6 s) vs 0.380 m/s;
              stops at the 0.30 m limit                                            < 3%, < 5 mm
  pusher      kinematic pusher drives a 1 kg box (mu 0.3) at a = 1 m/s^2 then constant
              0.2 m/s: reported contact force vs m a + mu m g = 3.94 N, then 2.94 N   < 10%
  determinism the stack run twice                                                  max |dx|
  throughput  256 worlds x 6 convex meshes, simulated seconds per wall second

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l0_solver_bakeoff.py [solver ...]
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l0_solver_bakeoff.json"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    from lsim.gpu import wait_for_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(wait_for_gpu(min_free_gib=8.0, max_util=20))

import warp as wp                                                        # noqa: E402
import newton                                                            # noqa: E402
from lsim.teacher.newton_teacher import (I, MUJOCO_KE, SOLVERS, SUBSTEPS, Teacher, joint_limit_cfg,  # noqa: E402
                                         new_builder, set_kinematic_free_joint, shape_cfg)

G = 9.81
MAP_FILE = REPO / "docs/results/l0_mujoco_restitution_map.json"
MUJOCO_E_MAP = None                                                        # kd -> e at MUJOCO_KE, measured below


def quat_x(deg):
    a = np.radians(deg) / 2
    return wp.quat(float(np.sin(a)), 0.0, 0.0, float(np.cos(a)))


def ground(b, s, mu=0.5, **kw):
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), I), hx=2.0, hy=2.0, hz=0.05,
                    cfg=shape_cfg(s, mu, **kw))


def ramp(b, s, tilt, mu):
    q = quat_x(tilt)
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), q), hx=0.5, hy=1.0, hz=0.05,
                    cfg=shape_cfg(s, mu))
    a = np.radians(tilt)
    return q, np.array([0.0, -np.sin(a), np.cos(a)]), np.array([0.0, -np.cos(a), -np.sin(a)])   # rot, normal, downhill


def cube_on_ramp(s, tilt, mu, edge=0.07, along=-0.3):
    b = new_builder(s)
    q, n, down = ramp(b, s, tilt, mu)
    c = n * (0.05 + edge / 2) - down * (-along)
    body = b.add_body(xform=wp.transform(wp.vec3(*c), q))
    b.add_shape_box(body, hx=edge / 2, hy=edge / 2, hz=edge / 2, cfg=shape_cfg(s, mu))
    return b, n, down


def t_slide(s):
    rows = []
    for tilt in (20.0, 30.0):
        for mu in (0.1, 0.2, 0.3):
            b, n, down = cube_on_ramp(s, tilt, mu)
            q, _ = Teacher(b, s).rollout(0.6)
            x = (q[:, 0, :3] - q[0, 0, :3]) @ down
            t = np.arange(len(x)) / 30.0
            m = t >= 0.1
            a = 2 * np.polyfit(t[m], x[m], 2)[0]
            law = G * (np.sin(np.radians(tilt)) - mu * np.cos(np.radians(tilt)))
            rows.append({"tilt": tilt, "mu": mu, "a": float(a), "law": float(law), "rel": float(a / law - 1)})
    worst = max(abs(r["rel"]) for r in rows)
    return {"rows": rows, "worst_rel": worst, "pass": worst < 0.02}


def corners(qrow, h):
    rot = wp.quat(*[float(v) for v in qrow[3:]])
    return np.array([qrow[:3] + np.array(wp.quat_rotate(rot, wp.vec3(sx * h, sy * h, sz * h)))
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def t_rest(s):
    b, n, down = cube_on_ramp(s, 20.0, 0.6)
    q, _ = Teacher(b, s).rollout(2.0)
    x = (q[:, 0, :3] - q[0, 0, :3]) @ down
    drift = float(x[-1] - x[30])
    pen = max(max(0.0, -float(((corners(q[k, 0], 0.035) @ n) - 0.05).min())) for k in range(0, len(q), 3))
    return {"drift_mm_s": 1000 * drift, "penetration_mm": 1000 * pen,
            "pass": abs(drift) < 2e-4 and pen < 1e-3}


def drop_e(s, h=0.20, R=0.035, **kw):
    b = new_builder(s)
    ground(b, s, 0.5, **kw)
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, h + R), I))
    b.add_shape_sphere(body, radius=R, cfg=shape_cfg(s, 0.5, **kw))
    q, qd = Teacher(b, s, substeps=4 * SUBSTEPS[s]).rollout(np.sqrt(2 * h / G) + 0.5)
    z, vz = q[:, 0, 2] - R, qd[:, 0, 2]
    up = np.where(vz > 0)[0]
    rise = float(z[up[0]:].max()) if len(up) else 0.0
    return float(np.sqrt(max(rise, 0.0) / h))


def mujoco_e_map():
    """kd -> e at the tuned ke for MuJoCo-Warp (mass-independent, T0); measured once, stored."""
    global MUJOCO_E_MAP
    if MUJOCO_E_MAP is None:
        if MAP_FILE.exists() and json.loads(MAP_FILE.read_text()).get("ke") == MUJOCO_KE:
            MUJOCO_E_MAP = [tuple(r) for r in json.loads(MAP_FILE.read_text())["map"]]
        else:
            MUJOCO_E_MAP = [(kd, drop_e("mujoco", ke=MUJOCO_KE, kd=kd))
                            for kd in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)]
            MAP_FILE.write_text(json.dumps({"ke": MUJOCO_KE, "drop_m": 0.20, "map": MUJOCO_E_MAP}, indent=1))
    return MUJOCO_E_MAP


def kd_for_e(e):
    m = sorted(mujoco_e_map(), key=lambda r: -r[1])
    kd = np.array([k for k, _ in m]); ee = np.array([v for _, v in m])
    return float(np.exp(np.interp(-e, -ee, np.log(kd))))


def t_bounce(s):
    rows = []
    R, h = 0.035, 0.20
    for e_t in (0.2, 0.5, 0.8):
        kw = {"ke": MUJOCO_KE, "kd": kd_for_e(e_t)} if s.startswith("mujoco") else {"restitution": e_t}
        rows.append({"target": e_t, "e": drop_e(s, h, R, **kw), **({"kd": kw["kd"]} if "kd" in kw else {})})
    worst = max(abs(r["e"] - r["target"]) for r in rows)
    return {"rows": rows, "worst_abs": worst, "pass": worst < 0.05}


def t_momentum(s):
    b = new_builder(s, gravity=(0.0, 0.0, 0.0))
    for x in (0.0, -0.12):
        body = b.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.5), I))
        b.add_shape_box(body, hx=0.035, hy=0.035, hz=0.035, cfg=shape_cfg(s, 0.5))
    tch = Teacher(b, s, init_qd={6: [1.0, 0.0, 0.0]})
    m = tch.model.body_mass.numpy()
    q, qd = tch.rollout(0.3)
    P = (qd[:, :, :3] * m[None, :, None]).sum(1)
    rel = float(np.abs(P[-1] - P[0]).max() / np.abs(P[0]).max())
    return {"rel_change": rel, "vA": float(qd[-1, 0, 0]), "vB": float(qd[-1, 1, 0]), "pass": rel < 1e-3}


def stack_run(s, seconds=5.0):
    b = new_builder(s)
    ground(b, s, 0.5)
    for k in range(5):
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.03 + 0.06 * k), I))
        b.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03, cfg=shape_cfg(s, 0.5))
    q, _ = Teacher(b, s).rollout(seconds)
    return q


def t_stack(s):
    q = stack_run(s)
    move = float(np.linalg.norm(q[-1, 4, :3] - q[0, 4, :3]))
    return {"top_moved_mm": 1000 * move, "pass": move < 2e-3}


def t_determinism(s):
    a, b = stack_run(s, 2.0), stack_run(s, 2.0)
    d = float(np.abs(a - b).max())
    return {"max_abs_diff_m": d, "bit_identical": d == 0.0}


def bowl_mesh(r_out=0.10, t=0.01, n_lat=12, n_lon=32):
    """Hemispherical shell open at the top: outer radius r_out, wall t; bottom at z = 0."""
    verts, faces = [], []
    c = np.array([0.0, 0.0, r_out])
    for r in (r_out, r_out - t):
        for i in range(n_lat + 1):
            ph = (np.pi / 2) * i / n_lat                    # 0 at the bottom, pi/2 at the rim
            for j in range(n_lon):
                lam = 2 * np.pi * j / n_lon
                verts.append(c + r * np.array([np.sin(ph) * np.cos(lam), np.sin(ph) * np.sin(lam), -np.cos(ph)]))
    V = lambda shell, i, j: shell * (n_lat + 1) * n_lon + i * n_lon + (j % n_lon)
    for i in range(n_lat):
        for j in range(n_lon):
            a, b_, c_, d = V(0, i, j), V(0, i, j + 1), V(0, i + 1, j + 1), V(0, i + 1, j)
            faces += [(a, c_, b_), (a, d, c_)]              # outer, normals outward
            a, b_, c_, d = V(1, i, j), V(1, i, j + 1), V(1, i + 1, j + 1), V(1, i + 1, j)
            faces += [(a, b_, c_), (a, c_, d)]              # inner, normals into the cavity
    for j in range(n_lon):                                   # rim ring
        a, b_, c_, d = V(0, n_lat, j), V(0, n_lat, j + 1), V(1, n_lat, j + 1), V(1, n_lat, j)
        faces += [(a, c_, b_), (a, d, c_)]
    return np.array(verts, np.float32), np.array(faces, np.int32)


def t_concave(s):
    v, f = bowl_mesh()
    b = new_builder(s)
    ground(b, s, 0.5)
    b.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.flatten()), xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), I),
                     cfg=shape_cfg(s, 0.5))
    body = b.add_body(xform=wp.transform(wp.vec3(0.01, 0.0, 0.15), I))
    b.add_shape_sphere(body, radius=0.02, cfg=shape_cfg(s, 0.5))
    q, _ = Teacher(b, s).rollout(2.0)
    z = float(q[-1, 0, 2])
    return {"ball_z_m": z, "inside": z < 0.05, "pass": z < 0.05}


def topple_run(s, com_down, tilt=14.0):
    b = new_builder(s)
    q, n, down = ramp(b, s, tilt, 1.0)
    hx, hy, hz = 0.03, 0.03, 0.10
    m = 0.5
    Ixx = m * (hy ** 2 + hz ** 2) * 4 / 12; Iyy = m * (hx ** 2 + hz ** 2) * 4 / 12; Izz = m * (hx ** 2 + hy ** 2) * 4 / 12
    c = n * (0.05 + hz)
    # body frame = ramp frame: +y is up-slope, so "com_down" metres downhill is -y in the body frame
    body = b.add_body(xform=wp.transform(wp.vec3(*c), q), com=wp.vec3(0.0, -com_down, 0.0), mass=m,
                      inertia=wp.mat33(Ixx, 0, 0, 0, Iyy, 0, 0, 0, Izz), lock_inertia=True)
    b.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=shape_cfg(s, 1.0, density=0.0))
    qq, _ = Teacher(b, s).rollout(2.0)
    rot = wp.quat(*[float(v) for v in qq[-1, 0, 3:]])
    z_axis = np.array(wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0)))
    return float(np.degrees(np.arccos(np.clip(z_axis @ n, -1, 1))))


def t_topple_com(s):
    a0, a1 = topple_run(s, 0.0), topple_run(s, 0.01)
    return {"tilt_deg_centred": a0, "tilt_deg_offset": a1, "pass": a0 < 3.0 and a1 > 45.0}


def t_drawer(s):
    b = new_builder(s)
    m, c, F = 1.0, 5.0, 2.0
    link = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), I))
    b.add_shape_box(link, hx=0.15, hy=0.1, hz=0.05, cfg=shape_cfg(s, 0.5, density=m / (0.3 * 0.2 * 0.1)))
    j = b.add_joint_prismatic(parent=-1, child=link, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), I),
                              axis=newton.Axis.X, damping=c, limit_lower=0.0, limit_upper=0.3,
                              **joint_limit_cfg(s))
    b.add_articulation([j])

    def pull(tch, t):
        f = tch.s0.body_f.numpy()
        f[link] = [F, 0.0, 0.0, 0.0, 0.0, 0.0]
        tch.s0.body_f.assign(f)

    q, qd = Teacher(b, s, pre_substep=pull).rollout(1.5)
    x = q[:, link, 0] - q[0, link, 0]
    v06 = float(qd[18, link, 0])
    law = F / c * (1 - np.exp(-c / m * 0.6))
    return {"v_at_0.6s": v06, "law": float(law), "rel": float(v06 / law - 1), "x_end_m": float(x[-1]),
            "pass": abs(v06 / law - 1) < 0.03 and abs(x[-1] - 0.3) < 5e-3}


def t_pusher(s):
    b = new_builder(s)
    ground(b, s, 0.3)
    m, mu = 1.0, 0.3
    box = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), I))
    b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=shape_cfg(s, mu, density=m / 0.001))
    x0 = -0.05 - 0.02 - 0.001
    pusher = b.add_body(xform=wp.transform(wp.vec3(x0, 0.0, 0.05), I), is_kinematic=True)
    b.add_shape_box(pusher, hx=0.02, hy=0.05, hz=0.04, cfg=shape_cfg(s, 0.0, density=0.0))

    def drive(tch, t):
        if t < 0.2:
            x, v = x0 + 0.5 * 1.0 * t * t, 1.0 * t
        else:
            x, v = x0 + 0.02 + 0.2 * (t - 0.2), 0.2
        set_kinematic_free_joint(tch, tch.kin_joint, (x, 0.0, 0.05), (v, 0.0, 0.0))

    tch = Teacher(b, s, pre_substep=None, force_pairs=[(box, pusher)])
    tch.kin_joint = int(np.where(tch.model.joint_child.numpy() == pusher)[0][0])
    tch.pre_substep = drive
    q, qd, Fd = tch.rollout(0.8, forces=True)
    fx = Fd[(box, pusher)][:, 0]
    t = np.arange(1, len(fx) + 1) / 30.0
    acc = fx[(t > 0.08) & (t < 0.18)].mean()
    const = fx[(t > 0.4) & (t < 0.75)].mean()
    law_a, law_c = m * 1.0 + mu * m * G, mu * m * G
    ok = abs(acc / law_a - 1) < 0.10 and abs(const / law_c - 1) < 0.10
    return {"F_accel_N": float(acc), "law_accel_N": law_a, "F_const_N": float(const), "law_const_N": law_c,
            "box_speed_end": float(qd[-1, box, 0]), "pass": bool(ok)}


def t_throughput(s, nw=256):
    import trimesh
    scene = new_builder(s)
    ground(scene, s, 0.5)
    rng = np.random.default_rng(0)
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
    m.apply_scale([1.0, 0.8, 1.2])
    for k in range(6):
        body = scene.add_body(xform=wp.transform(wp.vec3(*(rng.uniform(-0.15, 0.15, 2)), 0.06 + 0.1 * k), I))
        scene.add_shape_mesh(body, mesh=newton.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32).flatten()),
                             cfg=shape_cfg(s, 0.5))
    b = new_builder(s)
    b.replicate(scene, nw, spacing=(1.0, 1.0, 0.0))
    tch = Teacher(b, s)
    tch.frame(); wp.synchronize()
    t0 = time.time()
    for _ in range(60):
        tch.frame()
    wp.synchronize()
    rate = nw * 2.0 / (time.time() - t0)
    z = tch.s0.body_q.numpy()[:, 2]
    return {"worlds": nw, "sim_s_per_wall_s": rate, "episodes_6s_per_hour": rate * 600, "lowest_z": float(z.min())}


def _jsonable(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


TESTS = [("slide", t_slide), ("rest", t_rest), ("bounce", t_bounce), ("momentum", t_momentum),
         ("stack", t_stack), ("concave", t_concave), ("topple_com", t_topple_com), ("drawer", t_drawer),
         ("pusher", t_pusher), ("determinism", t_determinism), ("throughput", t_throughput)]


def main():
    solvers = [a for a in sys.argv[1:] if a in SOLVERS] or list(SOLVERS)
    only = [a for a in sys.argv[1:] if a in dict(TESTS)]
    wp.init()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for s in solvers:
        res.setdefault(s, {})
        for name, fn in TESTS:
            if only and name not in only:
                continue
            t0 = time.time()
            try:
                r = fn(s)
            except Exception as exc:
                r = {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "pass": False}
                traceback.print_exc(limit=2)
            r["seconds"] = round(time.time() - t0, 1)
            res[s][name] = r
            short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("rows",)}
            print(f"[{s:10s}] {name:11s} {short}", flush=True)
            OUT.write_text(json.dumps(res, indent=1, default=_jsonable))
    res["_versions"] = {"newton": newton.__version__, "warp": wp.config.version}
    OUT.write_text(json.dumps(res, indent=1, default=_jsonable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
