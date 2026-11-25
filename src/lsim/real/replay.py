"""The cloth teacher on a real recording (lsim.real.cloth): the recording's cloth from one of its frames, its held
patch following the recorded positions (ClothTeacher.follow), everything else simulated -- one world per material,
so a batch of worlds is a batch of candidate materials (fit by lsim.real.replay.fit_material).

The teacher simulates the cloth it is given with its own physics (membrane, discrete-shell bending, our contact
and damping, Newton's normal-velocity air drag); what it cannot match with any material in its range is where it is
wrong about real cloth.
"""
import numpy as np
import warp as wp
import newton

from lsim.teacher.cloth_teacher import FPS, G, ROLE_TABLE, Cloth, ClothTeacher, add_cloth, stable_substeps

I = wp.quat_identity()
PARK = (0.0, 0.0, 5.0)                     # the teacher's gripper body, unused: the hands are replayed as vertices
# the dataset's material prior (lsim.data.cloth.sample_cloth): the teacher's range
RANGE = {"w": (0.05, 0.5), "Y": (3e2, 3e4), "c": (0.015, 0.05), "mu": (0.1, 1.2)}
Y_OVER_W_MAX = 1.0e5


def _cfg(mu):
    s = newton.ModelBuilder.ShapeConfig()
    s.mu, s.ke, s.kd, s.kf = float(mu), 1.0e3, 0.0, 1.0
    return s


def simulate(ep, materials, t_start, n_frames, self_contact=True, table_mu=0.5, log=print):
    """One world per material dict (w kg/m^2, Y N/m, c bending length m, mu): the cloth of real episode `ep` from
    its fitted frame t_start (and velocity from the frame before), its held patch following the recording from then
    on. Returns positions (W, n_frames + 1, V, 3), frame t_start first."""
    g = ep.meta["grid"]
    nx, ny, h = int(g["nx"]), int(g["ny"]), float(g["h"])
    b = newton.ModelBuilder()
    worlds, roles = [], []
    for m in materials:
        wb = newton.ModelBuilder()
        wb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), I), hx=3.0, hy=3.0, hz=0.05,
                         cfg=_cfg(table_mu))
        gb = wb.add_body(xform=wp.transform(wp.vec3(*PARK), I), mass=0.0, label="gripper")
        cloth = Cloth(nx=nx, ny=ny, h=h, w=float(m["w"]), Y=float(m["Y"]), nu=0.3, B=float(m["w"]) * G * float(m["c"]) ** 3,
                      mu=float(m["mu"]), r=0.003)
        p0, n, e0, ne = add_cloth(wb, cloth, (0.0, 0.0, 0.003), I)
        po, eo, bo = b.particle_count, b.edge_count, b.body_count
        b.add_world(wb)
        worlds.append({"particles": (po + p0, n), "edges": (eo + e0, ne), "cloth": cloth, "gripper": bo + gb})
        roles += [ROLE_TABLE]
    W = len(materials)
    path = np.zeros((n_frames + 2, W, 7), np.float32)
    path[:, :, :3] = PARK
    path[:, :, 6] = 1.0
    substeps = stable_substeps([w["cloth"] for w in worlds])
    t = ClothTeacher(b, worlds, roles, path, substeps, self_contact=self_contact)
    V = ep._x.shape[1]
    X = ep.x(np.arange(t_start, t_start + n_frames + 1)).astype(np.float64)       # (n + 1, V, 3)
    v0 = (X[0] - ep.x([max(t_start - 1, 0)])[0]) * FPS
    q, qd = t.s0.particle_q.numpy(), t.s0.particle_qd.numpy()
    for wd in worlds:
        p0, n = wd["particles"]
        assert n == V, (n, V)
        q[p0:p0 + n] = np.maximum(X[0], [-1e9, -1e9, 0.003])                        # on the table, not in it
        qd[p0:p0 + n] = v0
    t.s0.particle_q.assign(q)
    t.s0.particle_qd.assign(qd)
    held = ep.held[t_start:t_start + n_frames + 1]                                # (n + 1, V)
    hv = np.flatnonzero(held.any(0))
    if len(hv):
        ids = np.concatenate([wd["particles"][0] + hv for wd in worlds])
        rp = np.concatenate([X[:, hv] for _ in worlds], 1)                         # (n + 1, W * len(hv), 3)
        t.follow(ids, np.concatenate([rp, rp[-1:]], 0))
    out = np.zeros((W, n_frames + 1, V, 3), np.float32)
    out[:, 0] = X[0]
    for k in range(n_frames):
        if len(hv):
            t.follow_on(np.tile(held[k, hv], W))
        x = t.frame()["x"]
        for w, wd in enumerate(worlds):
            p0, n = wd["particles"]
            out[w, k + 1] = x[p0:p0 + n]
    return out


def track_errors(ep, Xw, t_start, moved=True):
    """Per world, the median distance of the visible tracked points to their embedded positions, per frame:
    (W, n_frames + 1). moved: only the points that have moved over 1 cm since t_start (the lifted cloth; NaN where
    none has yet)."""
    frames = np.arange(t_start, t_start + Xw.shape[1])
    return np.stack([ep.track_error(Xw[w], frames, moved_from=t_start if moved else None) for w in range(len(Xw))])


def encode(m):
    return np.array([np.log(m["w"]), np.log(m["Y"]), np.log(m["c"]), np.log(m["mu"])])


def decode(z):
    lo = np.log([RANGE[k][0] for k in ("w", "Y", "c", "mu")])
    hi = np.log([RANGE[k][1] for k in ("w", "Y", "c", "mu")])
    z = np.clip(z, lo, hi)
    w, Y, c, mu = np.exp(z)
    Y = min(Y, Y_OVER_W_MAX * w)                                                  # the prior's wave-speed bound
    return {"w": float(w), "Y": float(Y), "c": float(c), "mu": float(mu)}


def fit_material(ep, t_start, n_frames, pop=32, elite=8, gens=10, seed=0, self_contact=True, log=print):
    """The teacher's best material for frames t_start .. t_start + n_frames of `ep`: cross-entropy search in the
    log of (w, Y, bending length, mu) within the dataset's range, a generation a batch of `pop` worlds; the score is
    the mean over the frames of the median error of the tracked points that have moved over 1 cm (the lifted cloth:
    over every point, the cloth lying still on the table dominated). Returns (best material, its score, history)."""
    rng = np.random.default_rng(seed)
    lo = np.log([RANGE[k][0] for k in ("w", "Y", "c", "mu")])
    hi = np.log([RANGE[k][1] for k in ("w", "Y", "c", "mu")])
    mean, sd = 0.5 * (lo + hi), 0.35 * (hi - lo)
    best, best_s, hist = None, np.inf, []
    for gen in range(gens):
        Z = mean + sd * rng.standard_normal((pop, 4))
        mats = [decode(z) for z in Z]
        Xw = simulate(ep, mats, t_start, n_frames, self_contact=self_contact, log=log)
        err = track_errors(ep, Xw, t_start)[:, 1:]
        score = np.nanmean(err, axis=1)
        score[~np.isfinite(score)] = np.inf
        order = np.argsort(score)
        el = np.array([encode(mats[i]) for i in order[:elite]])
        mean, sd = el.mean(0), np.maximum(el.std(0), 0.02 * (hi - lo))
        if score[order[0]] < best_s:
            best, best_s = mats[order[0]], float(score[order[0]])
        hist.append({"gen": gen, "best": float(score[order[0]]), "elite_mean": float(score[order[:elite]].mean()),
                     "mean": decode(mean)})
        log(f"  gen {gen}: best {1e3 * score[order[0]]:.2f} mm (elite {1e3 * score[order[:elite]].mean():.2f}) at "
            f"{ {k: round(v, 4) for k, v in mats[order[0]].items()} }", flush=True)
    return best, best_s, hist
