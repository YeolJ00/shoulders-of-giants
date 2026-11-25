"""L0 -- the soft-body teacher's own checks: the soft gate's closed forms run on the module that generates the data
(lsim.teacher.soft_teacher), plus checks on what generated batches do (the cloth lesson: the cloth gate ran its own copy
of the physics and never dropped, folded or piled cloth; the first 16k cloth episodes bounced, crossed and exploded).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l0_soft_teacher_check.py [check ...] [--objects gso|simready|both]

The gate's checks (scripts/l0_soft_gate.py) run through SoftTeacher itself (`TeacherSim` has the gate's Sim interface):

  hang       bar hung from its top face: drop against rho g L^2 / 2E (E 1e3, 1e4, 1e6; 5%)
  slide      foam blocks on 20 deg: the slide law at mu 0.2 (2%), no creep at mu 0.5 (< 1 mm/s), a 10 cm column too
  rest       10 cm foam cube dropped 1 cm: at rest, no creep, no penetration, self-weight compression rho g H^2 / 2E (10%)
  freefall   centroid acceleration g (0.1%), shape kept
  spin       a spinning cube keeps KE and angular momentum (1%): the damping is blind to rotation

and the generator's:

  squeeze    the gripper's pads close on a 6 cm foam cube on the table to 50% and open again: no inverted tet, back to
             its rest shape (1 mm, rigid alignment) 1 s after release, the pads' force over the body's weight
  bounce     a rubber ball and a foam ball dropped 20 cm: restitution (the rebound apex over the drop, from the centroid)
             and rest (no vertex moves 1 mm in the last 0.5 s)
  batch      a pilot of generator batches (the natural mix, 4 x 64 = 256 episodes by default): explosions (a vertex
             moving 0.4 m in a frame, rule 3), inverted tets at any frame, contact force over weight, penetration of
             the table and of the pads, bodies left moving at the end; throughput
Results: docs/results/l0_soft_teacher_check.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
import warp as wp                                               # noqa: E402
import newton                                                   # noqa: E402

import l0_soft_gate as gate                                     # noqa: E402
import lsim.teacher.soft_teacher as st                          # noqa: E402
from lsim.teacher import soft_mesh                              # noqa: E402

OUT = REPO / "docs/results/l0_soft_teacher_check.json"
OBJECTS = "gso"                                      # obstacle library of the generator batches (--objects)
I = wp.quat_identity()
FPS = st.FPS
G = st.G


def add_body(b, solver, body, dims, pos, rot=None, h=None):
    """The gate's box (corner at pos), built the teacher's way: a Soft with the gate's mesh and eta given directly."""
    h = h or body["h"]
    n = [int(round(d / h)) for d in dims]
    v, t = soft_mesh.box_tets(*n, h)
    s = st.Soft("box", {"dims": list(dims)}, rho=body["rho"] * gate.CONTROL.get("rho", 1.0),
                E=body["E"] * gate.CONTROL.get("E", 1.0), nu=body["nu"] + gate.CONTROL.get("nu", 0.0), mu=body["mu"],
                r=body["r"], eta=float(body["eta"]), _mesh=(v, t))
    p0, cnt, t0, nt = st.add_soft(b, s, pos, rot if rot is not None else I)
    b._softs = getattr(b, "_softs", []) + [(s, (p0, cnt), (t0, nt))]
    return np.arange(p0, p0 + cnt), t


def static_box(b, center, half, mu, rot=None):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu, cfg.ke, cfg.kd, cfg.kf = mu, 1.0e3, 0.0, 1.0
    return b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*center), rot if rot is not None else I),
                           hx=half[0], hy=half[1], hz=half[2], cfg=cfg)


class TeacherSim:
    """The gate's Sim interface over SoftTeacher (no gripper): fixed vertices held, the gate's settling drag as the
    teacher's settle_damping, the scene's mu on the body (the gate gives contact one mu)."""

    def __init__(self, solver, scene, worlds=1, drag=0.0, fps=FPS, gravity=-gate.G, **over):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, gravity * gate.CONTROL.get("g", 1.0)))
        self.meta = scene(b, solver)
        (s, parts, tets), = b._softs
        s.mu = self.meta.get("mu", s.mu) + gate.CONTROL.get("mu", 0.0)
        n, ce = st.substeps_for([s])
        if "dt_scale" in over:                                  # a control: a step the gate showed unstable
            dt = s.stable_dt() * over["dt_scale"] / st.DT_SCALE
            n = int(np.ceil(1.0 / (FPS * dt) / ce)) * ce
        roles = [st.ROLE_TABLE] * b.shape_count
        path = np.zeros((2, 1, 7), np.float32)
        self.teacher = st.SoftTeacher(b, [{"particles": parts, "tets": tets, "soft": s, "gripper": None}], roles,
                                      [st.SLOT_TABLE] * b.shape_count, path, np.zeros((2, 1)), n, ce, settle_damping=drag)
        fixed = np.asarray(self.meta.get("fixed", []), int)
        if len(fixed):
            self.teacher.fix(fixed)
        self.kind, self.n, self.npw = "semi", n, parts[1]
        m = self.teacher.model.particle_mass.numpy().astype(np.float64)
        if len(fixed):
            m[fixed] = 0.0
        self.mass = m
        self.tets = self.teacher.model.tet_indices.numpy()
        self.Dm_inv = self.teacher.model.tet_poses.numpy().astype(np.float64)
        self.x_rest = self.x()
        self.soft, self.t, self.wall = s, 0.0, 0.0

    @property
    def s0(self):
        return self.teacher.s0

    @property
    def s1(self):
        return self.teacher.s1

    def frame(self, n=1):
        t0 = time.time()
        for _ in range(n):
            self.teacher.frame()
            self.t += 1.0 / FPS
        wp.synchronize()
        self.wall += time.time() - t0

    def rate(self):
        return self.t / max(self.wall, 1e-9)

    def x(self):
        return self.teacher.s0.particle_q.numpy()[:self.npw].astype(np.float64)

    def v(self):
        return self.teacher.s0.particle_qd.numpy()[:self.npw].astype(np.float64)

    def info(self):
        return {"substeps_per_s": self.n * FPS, "dt_s": 1.0 / (FPS * self.n), "dt_limits_s": self.soft.dt_limits(),
                "vertices": int(self.npw), "tets": int(len(self.tets))}

    def in_bound(self, x=None, bound=1.0):
        x = self.x() if x is None else x
        return bool(np.isfinite(x).all() and np.abs(x - self.x_rest).max() < bound)


gate.Sim, gate.add_body, gate.static_box = TeacherSim, add_body, static_box


# ------------------------------------------------------------------------------------------- generator scenes
def _single(soft, pos, rot=I, vel=None, frames=2, gripper=True, table_mu=0.5):
    """One body over the generator's table with the gripper parked; returns a SoftTeacher and its world record."""
    from lsim.data import soft as sd
    sp = sd.SoftSpec(soft=soft, init="rest", pose=(0.0, 0.0, 0.0), table_mu=table_mu)
    b, info = sd.build_world(sp, np.random.default_rng(0), place=(pos, rot))
    W = {"particles": info["particles"], "tets": info["tets"], "soft": soft, "gripper": info["gripper"]}
    path = np.zeros((frames + 1, 1, 7), np.float32)
    path[:, :, :3] = sd.PARK
    path[:, :, 6] = 1.0
    width = np.full((frames + 1, 1), 0.1, np.float32)
    n, ce = st.substeps_for([soft])
    v0 = None
    if vel is not None:
        v0 = np.zeros((info["particles"][1], 3), np.float32) + np.asarray(vel, np.float32)
    t = st.SoftTeacher(b, [W], info["roles"], info["slots"], path, width, n, ce, velocities=v0)
    return t, W


def check_squeeze(solver="teacher", **over):
    """6 cm foam cube (E 1e4, nu 0.3, zeta 0.1) on the table; the pads close from 1 mm off its faces to 50% of its width
    in 1 s at mid-height, hold 0.5 s, open in 1 s, wait 1 s."""
    W = 0.06
    s = st.Soft("box", {"dims": [W, W, W]}, rho=60.0, E=1.0e4, nu=0.3, mu=0.6, zeta=0.1)
    nf = [30, 15, 30, 30]
    F = sum(nf) + 2
    t, wd = _single(s, (0.0, 0.0, W / 2 + s.r + 0.0002), frames=F)
    path = t.path.numpy().reshape(F + 1, 1, 7)
    width = t.width.numpy()
    g0, g1 = W + 0.002, 0.5 * W
    path[:, 0, :3] = (0.0, 0.0, 0.005)
    f = 0
    for ph, n in enumerate(nf):
        for k in range(n):
            u = (k + 1) / n
            width[f + 1, 0] = [g0 + (g1 - g0) * u, g1, g1 + (g0 + 0.02 - g1) * u, g0 + 0.02][ph]
            f += 1
    width[0, 0] = g0
    width[f + 1:, 0] = width[f, 0]
    t.path.assign(path)
    t.width.assign(width)
    p0, n = wd["particles"]
    tets = t.model.tet_indices.numpy()
    Dm = t.model.tet_poses.numpy().astype(np.float64)
    x0 = t.s0.particle_q.numpy()[p0:p0 + n].astype(np.float64)
    Jmin, fmax, wmin, xs = np.inf, 0.0, np.inf, []
    mid = (np.abs(x0[:, 2] - x0[:, 2].mean()) < 0.004) & (np.abs(x0[:, 1]) < 0.008)   # under the 3 cm pads
    for k in range(F):
        r = t.frame()
        x = r["x"][p0:p0 + n].astype(np.float64)
        Jmin = min(Jmin, float(soft_mesh.tet_J(x, tets, Dm).min()))
        fmax = max(fmax, float(np.linalg.norm(r["slot"][0, st.SLOT_FL])))
        if k < 45:
            wmin = min(wmin, float(np.ptp(x[mid, 0])))
        xs.append(x)
    err = gate.procrustes_err(xs[-1], x0)
    wt = s.mass * st.G
    out = {"min_J": Jmin, "narrowest_mm": 1e3 * wmin, "shape_error_after_release_mm": 1e3 * err,
           "left_pad_force_max_N": fmax, "pad_force_over_weight": fmax / wt, "substeps": t.substeps,
           "pass": bool(Jmin > 0 and err < 1e-3 and wmin < 0.6 * W)}
    print(f"  squeeze : narrowest {1e3 * wmin:.1f} of {1e3 * W:.0f} mm, min J {Jmin:.3f}, shape error 1 s after release "
          f"{1e3 * err:.3f} mm, pad force max {fmax:.2f} N ({fmax / wt:.0f}x weight)", flush=True)
    return out


def check_bounce(solver="teacher", **over):
    """Balls (r 4 cm) dropped from 20 cm (their bottom) onto the table: the first rebound's apex over the drop height
    (restitution e ~ sqrt(apex / drop), from the centroid), and the total energy (kinetic + gravity) never above its
    start (the teacher must not make energy; a ball that rolls off the table makes any 'rest' measure meaningless)."""
    out = {}
    for tag, rho, E, nu, zeta in (("rubber", 1100.0, 1.0e6, 0.45, 0.03), ("rubber_damped", 1100.0, 1.0e6, 0.45, 0.06),
                                  ("closed_foam", 80.0, 3.0e5, 0.2, 0.07), ("foam", 40.0, 2.0e4, 0.1, 0.15),
                                  ("plush", 70.0, 3.0e3, 0.2, 0.25), ("memory_foam", 60.0, 5.0e3, 0.2, 0.5)):
        s = st.Soft("ellipsoid", {"axes": [0.04, 0.04, 0.04]}, rho=rho, E=E, nu=nu, mu=0.8, zeta=zeta)
        H = 0.20
        t, wd = _single(s, (0.0, 0.0, 0.04 + s.r + H), frames=2 * FPS + 1)
        p0, n = wd["particles"]
        m = t.mass[p0:p0 + n].astype(np.float64)
        cz, En = [], []
        for k in range(2 * FPS):
            x = t.frame()["x"][p0:p0 + n].astype(np.float64)
            v = t.s0.particle_qd.numpy()[p0:p0 + n].astype(np.float64)
            cz.append(float((m * x[:, 2]).sum() / m.sum()))
            En.append(float(0.5 * (m * (v ** 2).sum(1)).sum() + G * (m * x[:, 2]).sum()))
        cz, En = np.array(cz), np.array(En)
        first = int(np.argmax(np.diff(cz) > 0))           # the first frame it moves up
        down = first + int(np.argmax(np.diff(cz[first:]) < 0))
        apex = float(cz[down] - (0.04 + s.r))
        gain = float((En[first:].max() - En[0]) / (s.mass * G * H))
        e = float(np.sqrt(max(apex, 0.0) / H))
        out[tag] = {"rebound_apex_mm": 1e3 * apex, "restitution_est": e, "energy_gain_over_drop": gain,
                    "substeps": t.substeps, "eta": s.damping(), "zeta": zeta, "pass": bool(gain < 0.01)}
        print(f"  bounce  {tag:14s}: first rebound {1e3 * apex:6.1f} mm of {1e3 * H:.0f} (e ~ {e:.2f}), energy above its "
              f"start {100 * gain:+.2f}% of the drop [{t.substeps} substeps, eta {s.damping():.3g} Pa s]", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_batch(solver="teacher", n_batches=4, worlds=64, seed=7, **over):
    from lsim.data import library
    from lsim.data import soft as sd
    train, _ = library.object_split(OBJECTS)          # --objects gso (default) | simready | both
    out, rows = {}, []
    rng = np.random.default_rng(seed)
    pool = [sd.sample_spec(rng, train, seed=seed * 10000 + k) for k in range(n_batches * worlds)]
    pool.sort(key=lambda sp: sp.soft.stable_dt())
    for i in range(n_batches):
        specs = pool[i * worlds:(i + 1) * worlds]
        t0 = time.time()
        res = sd.run_batch(specs, seed=seed * 100 + i, log=print)
        wall = time.time() - t0
        qc = sd.quality(res)
        pilot = REPO / "outputs/lsim/soft_pilot"
        pilot.mkdir(parents=True, exist_ok=True)
        sd.save_shard(pilot, i, res, seed * 100 + i)
        rows += qc
        print(f"  batch {i}: {len(qc)} episodes in {wall:.0f} s ({res['substeps']} substeps/frame), exploded "
              f"{sum(q['exploded'] for q in qc)}, inverted {sum(q['inverted'] for q in qc)}, contact/weight max "
              f"{max(max(q['contact_over_weight']) for q in qc):.1f}, table pen max {max(q['table_pen_mm'] for q in qc):.2f} mm, "
              f"pad pen max {max(q['pad_pen_mm'] for q in qc):.2f} mm", flush=True)
        out[f"batch{i}"] = {"wall_s": wall, "substeps": res["substeps"], "episodes": len(qc),
                            "episodes_per_gpu_hour": 3600.0 * len(qc) / wall}
    def pct(k, p):
        return float(np.percentile([q[k] for q in rows], p))
    cw = np.array([max(q["contact_over_weight"]) for q in rows])
    out["summary"] = {
        "episodes": len(rows), "exploded": int(sum(q["exploded"] for q in rows)),
        "inverted": int(sum(q["inverted"] for q in rows)), "min_J_p1": pct("min_J", 1), "min_J_min": pct("min_J", 0),
        "contact_over_weight_p50": float(np.median(cw)), "contact_over_weight_p99": float(np.percentile(cw, 99)),
        "contact_over_weight_max": float(cw.max()), "table_pen_mm_p99": pct("table_pen_mm", 99),
        "table_pen_mm_max": pct("table_pen_mm", 100), "pad_pen_mm_p99": pct("pad_pen_mm", 99),
        "pad_pen_mm_max": pct("pad_pen_mm", 100), "end_speed_p50": pct("end_speed", 50),
        "end_speed_p90": pct("end_speed", 90), "vmax_max": pct("vmax", 100),
        "vertices_mean": float(np.mean([q["vertices"] for q in rows])), "vertices_max": int(max(q["vertices"] for q in rows)),
        "episodes_per_gpu_hour": float(3600.0 * len(rows) / sum(out[f"batch{i}"]["wall_s"] for i in range(n_batches))),
        "families": {}, "inits": {}, "shapes": {}}
    for q in rows:
        for k, key in (("families", "families"), ("inits", "init"), ("shapes", "shape")):
            vals = q[key] if isinstance(q[key], list) else [q[key]]
            for v in vals:
                out["summary"][k][v] = out["summary"][k].get(v, 0) + 1
    s = out["summary"]
    out["pass"] = bool(s["exploded"] == 0 and s["inverted"] == 0 and s["table_pen_mm_max"] < 3.0 and s["pad_pen_mm_max"] < 5.0)
    print(json.dumps(s, indent=1), flush=True)
    return out


CHECKS = {"hang": gate.check_hang, "slide": gate.check_slide, "rest": gate.check_rest, "freefall": gate.check_freefall,
          "spin": gate.check_spin, "squeeze": check_squeeze, "bounce": check_bounce, "batch": check_batch}


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
        t0 = time.time()
        try:
            res[n] = CHECKS[n]("teacher")
        except Exception as exc:                              # record the failure, keep going
            import traceback
            traceback.print_exc()
            res[n] = {"error": repr(exc)}
            print(f"  {n}: ERROR {exc!r}", flush=True)
        res[n]["wall_s"] = time.time() - t0
        OUT.write_text(json.dumps(res, indent=1, default=float))
    res["_versions"] = {"newton": newton.__version__, "warp": wp.config.version,
                        "teacher": {"contact_rate": st.CONTACT_RATE, "contact_zeta": st.CONTACT_ZETA, "cfl": st.CFL,
                                    "dt_scale": st.DT_SCALE, "collide_dt": st.COLLIDE_DT, "gap": st.GAP}}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("pass: " + ", ".join(f"{k} {v.get('pass')}" for k, v in res.items() if isinstance(v, dict) and ("pass" in v or "error" in v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
