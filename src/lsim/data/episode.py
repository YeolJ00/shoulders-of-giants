"""Run a batch of scenes through the teacher and record episodes (PHASE1_AMENDMENT.md section 6).

Per batch: build one Newton world per scene, settle 0.5 s (not recorded), open hinged parts that
start open, plan each world's first action family from the settled scene, lift the objects that
will be dropped, then record `seconds` at 30 fps. Each later family is planned when the previous
one ends, from the objects' CURRENT poses; when the sampled families are used up and an object
has still not been acted on, a fill-in family targets it. The gripper's PD force is capped
(force-limited pushes, amendment section 3). Recorded: every body's pose and velocity, the
gripper path, applied pokes, frame-averaged contact forces between all slot pairs.
"""
import time

import numpy as np
import warp as wp
import newton

from lsim.data import actions, library
from lsim.data.scene import GRIPPER_MASS, PARK, RAMP, build_world, ramp_frame
from lsim.teacher.contacts import ContactRecorder
from lsim.teacher.newton_teacher import Teacher, new_builder

FPS = 30.0
CAP_SAFE = 60.0                                   # N: a household gripper never pushes harder
CAP_LIMITED = (5.0, 15.0)                         # N: force-limited pushes (20% of worlds)
KP, KD = 3000.0, 2.0 * np.sqrt(3000.0 * GRIPPER_MASS)     # ~12 Hz tracking, critically damped
KR, KW, TMAX = 2.0, 0.06, 2.0
KP_TRACK = 1.0e4                                  # replaying a recorded gripper path: 3.3x stiffer
KD_TRACK = 2.0 * np.sqrt(KP_TRACK * GRIPPER_MASS)  # (KD dt / m = 1.2 at 240 Hz; 3e4 is unstable)
LAST_START = 1.0                                  # s before the end: no family starts after this
FILL_LAST_START = 1.5                             # ... and no fill-in family
FILL_TRIES = 4


@wp.kernel
def _gripper_pd(s: int, gb: wp.array(dtype=wp.int32), pos: wp.array2d(dtype=wp.vec3), quat: wp.array2d(dtype=wp.quat),
                vel: wp.array2d(dtype=wp.vec3), wz: wp.array2d(dtype=wp.float32), body_q: wp.array(dtype=wp.transform),
                body_qd: wp.array(dtype=wp.spatial_vector), cap: wp.array(dtype=wp.float32), mass: float,
                kp: float, kd: float, kr: float, kw: float, tmax: float, g: float,
                body_f: wp.array(dtype=wp.spatial_vector), sat: wp.array(dtype=wp.int32)):
    """Track the planned pose with a spring-damper whose force is capped (the gripper's strength),
    plus gravity compensation (not counted against the cap)."""
    w = wp.tid()
    b = gb[w]
    X = body_q[b]
    p = wp.transform_get_translation(X)
    r = wp.transform_get_rotation(X)
    v = wp.spatial_top(body_qd[b])
    om = wp.spatial_bottom(body_qd[b])
    F = kp * (pos[s, w] - p) + kd * (vel[s, w] - v)
    n = wp.length(F)
    if n > cap[w]:
        F = F * (cap[w] / n)
        sat[w] = 1
    F = F + wp.vec3(0.0, 0.0, mass * g)
    qe = quat[s, w] * wp.quat_inverse(r)
    if qe[3] < 0.0:
        qe = wp.quat(-qe[0], -qe[1], -qe[2], -qe[3])
    aa = 2.0 * wp.vec3(qe[0], qe[1], qe[2])
    tau = kr * aa + kw * (wp.vec3(0.0, 0.0, wz[s, w]) - om)
    tn = wp.length(tau)
    if tn > tmax:
        tau = tau * (tmax / tn)
    body_f[b] = body_f[b] + wp.spatial_vector(F, tau)


@wp.kernel
def _add_wrench(f: int, wrench: wp.array2d(dtype=wp.spatial_vector), body_f: wp.array(dtype=wp.spatial_vector)):
    b = wp.tid()
    body_f[b] = body_f[b] + wrench[f, b]


def _quat_z(yaw):
    return np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)], -1)


def rotate(qxyzw, V):
    """Rotate row vectors V (n, 3) by the unit quaternion (x, y, z, w)."""
    u, w = np.asarray(qxyzw[:3], float), float(qxyzw[3])
    return V + 2.0 * np.cross(u, np.cross(u, V) + w * V)


def objects_now(q, spec, info):
    objs = []
    for k, o in enumerate(spec.objects):
        bi = info["objects"][k]
        a = library.load(o.name)
        va = rotate(q[bi, 3:], a["collide_v"].astype(float)) + q[bi, :3]
        surf = rotate(q[bi, 3:], a["points_v"].astype(float)) + q[bi, :3]     # 1.5 cm surface samples
        objs.append({"body": bi, "center": q[bi, :3].astype(float).copy(), "top": float(va[:, 2].max()),
                     "bottom": float(va[:, 2].min()), "verts": va[::4], "verts_all": va, "surf": surf,
                     "mass": float(info["labels"][k]["mass"])})
    return objs


def _count(c, entry):
    for key in ("target", "other"):
        if entry.get(key, -1) >= 0:
            c[entry[key]] += 1


def _track_targets(track, ts, fps=FPS):
    """PD targets at substep times from a recorded gripper trajectory (nf+1, 13): pose (7) and
    velocity (linear, angular) sampled at frames 0, 1/fps, ..."""
    tf = np.arange(len(track)) / fps
    pos = np.stack([np.interp(ts, tf, track[:, i]) for i in range(3)], -1)
    q = track[:, 3:7].astype(float).copy()
    for i in range(1, len(q)):                                     # one hemisphere, then nlerp
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    qi = np.stack([np.interp(ts, tf, q[:, i]) for i in range(4)], -1)
    qi /= np.linalg.norm(qi, axis=1, keepdims=True)
    vel = np.stack([np.interp(ts, tf, track[:, 7 + i]) for i in range(3)], -1)
    return pos, qi, vel, np.interp(ts, tf, track[:, 12])


def obstacles_now(spec):
    """Static things the gripper must not run into besides the objects: the ramp plank."""
    if spec.ramp is None:
        return []
    o, ax, lat, n = ramp_frame(spec.ramp)
    u, v = np.meshgrid(np.linspace(0, RAMP["L"], 11), np.linspace(-0.5, 0.5, 7) * RAMP["W"], indexing="ij")
    pts = o + u.reshape(-1, 1) * ax + v.reshape(-1, 1) * lat + RAMP["T"] * n
    return [{"static": True, "center": pts.mean(0), "top": float(pts[:, 2].max()), "bottom": 0.0,
             "verts": pts, "verts_all": pts, "surf": pts, "mass": 0.0}]


def ramp_pose(spec, q_body, verts, s, gap=0.002):
    """Pose putting an object at rest on the ramp, `s` up the slope: its settled orientation
    tilted onto the plank (base along the surface), lowest point `gap` above the top face."""
    o, ax, lat, n = ramp_frame(spec.ramp)
    R_tilt = np.stack([ax, lat, n], 1)
    yaw = 2.0 * np.arctan2(q_body[2], q_body[3])                       # keep the object's own yaw
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = R_tilt @ np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    lift = -(verts @ R.T @ n).min() + gap
    pos = o + s * ax + RAMP["T"] * n + lift * n
    from scipy.spatial.transform import Rotation
    return pos, Rotation.from_matrix(R).as_quat()


def run_batch(specs, seconds=6.0, settle=0.5, solver="kamino", seed=0, log=print, replay=None,
              kp=KP, kd=KD, poke_scale=1.0):
    """replay: per world, from a recorded episode (identifiability gate) -- no planning:
      {"keyframes", "pokes", "hinge": (mode, q0) or None, "cap", "hinge_used"} replays the gripper's
      planned path; adding "track" (nf+1, 13) makes the gripper follow the RECORDED gripper pose
      instead (pass kp=KP_TRACK, kd=KD_TRACK); "wrench" {object index or "hinge": (nf, 6)} adds
      world-frame wrenches about each body's centre of mass (the recorded gripper contact forces,
      replayed open loop with the gripper parked). Pokes are defined by the velocity change they
      cause, so their force scales with the object's mass_scale."""
    rng = np.random.default_rng(seed)
    t_build = time.time()
    b = new_builder(solver)
    infos = []
    for sp in specs:
        w, info = build_world(sp, solver)
        bo, jo = b.body_count, b.joint_count                      # world-local -> batch indices
        b.add_world(w)
        info["objects"] = [bo + i for i in info["objects"]]
        info["gripper"] += bo
        info["gripper_joint"] += jo
        if "hinge" in info:
            info["hinge"]["link"] += bo
            info["hinge"]["joint"] += jo
        infos.append(info)
    W = len(specs)
    teacher = Teacher(b, solver, contact_forces=True)
    m = teacher.model
    nb = m.body_count
    K = 2 + max(len(i["objects"]) for i in infos) + 1          # 0 static, 1 gripper, objects, hinge
    body_slot = np.full(nb, -1, np.int32)
    for inf in infos:
        body_slot[inf["gripper"]] = 1
        for k, ob in enumerate(inf["objects"]):
            body_slot[ob] = 2 + k
        if "hinge" in inf:
            body_slot[inf["hinge"]["link"]] = K - 1
    rec = ContactRecorder(teacher, body_slot, W, K)
    log(f"[episode] built {W} worlds, {nb} bodies in {time.time() - t_build:.1f} s")

    # gripper controller state (targets filled after planning; PARK during settling)
    nf = int(round(seconds * FPS))
    S = nf * teacher.substeps
    ts = (np.arange(S) + 1) * teacher.dt
    P = np.tile(np.array(PARK, np.float32), (S, W, 1)); Qq = np.tile(np.array([0, 0, 0, 1], np.float32), (S, W, 1))
    V = np.zeros((S, W, 3), np.float32); Wz = np.zeros((S, W), np.float32)
    gP, gQ = wp.array(P, dtype=wp.vec3), wp.array(Qq, dtype=wp.quat)
    gV, gW = wp.array(V, dtype=wp.vec3), wp.array(Wz, dtype=wp.float32)
    gb = wp.array(np.array([i["gripper"] for i in infos], np.int32), dtype=wp.int32)
    caps = np.where(rng.random(W) < 0.2, rng.uniform(*CAP_LIMITED, W), CAP_SAFE).astype(np.float32)
    if replay is not None:
        caps = np.array([r["cap"] for r in replay], np.float32)
    gcap = wp.array(caps, dtype=wp.float32)
    sat = wp.zeros(W, dtype=wp.int32)
    state = {"s": 0, "f": 0}
    gWr = wp.zeros((1, nb), dtype=wp.spatial_vector)

    def pre(t, _time):
        wp.launch(_gripper_pd, dim=W, inputs=[state["s"], gb, gP, gQ, gV, gW, t.s0.body_q, t.s0.body_qd, gcap,
                                              GRIPPER_MASS, kp, kd, KR, KW, TMAX, 9.81], outputs=[t.s0.body_f, sat])
        wp.launch(_add_wrench, dim=nb, inputs=[state["f"], gWr], outputs=[t.s0.body_f])

    # 1. settle, gripper held at PARK, nothing recorded
    teacher.pre_substep = pre
    for _ in range(int(round(settle * FPS))):
        teacher.frame()

    # 2. per world: families, pokes, hinge start state, and the FIRST family planned now
    q = teacher.s0.body_q.numpy()
    paths, queues, logs, pokes_all, sampled, counts, fills = [], [], [], [], [], [], np.zeros(W, int)
    for w, (sp, inf) in enumerate(zip(specs, infos)):
        objs = objects_now(q, sp, inf)
        counts.append(np.zeros(len(objs), int))                   # times each object was targeted
        hinge = inf.get("hinge")
        if replay is not None:
            r = replay[w]
            if hinge is not None and r["hinge"] is not None:
                hinge["mode"], hinge["q0"] = r["hinge"]
                if r["hinge_used"] and hinge["q0"] > 0:
                    _open_hinge(teacher, hinge, hinge["q0"])
            path = actions.Path(PARK)
            path.k = [(float(k[0]), np.array(k[1:4], float), float(k[4])) for k in r["keyframes"]]
            paths.append(path); queues.append([]); logs.append(r.get("families", [])); pokes_all.append(r["pokes"])
            sampled.append(r.get("families_sampled", []))
            continue
        obst = obstacles_now(sp)
        fams = actions.sample_families(rng, len(objs), hinge is not None)
        sampled.append(list(fams))
        if hinge is not None:
            hinge["mode"], hinge["q0"] = actions.sample_hinge_mode(rng, hinge)
            if "hinge" in fams and hinge["q0"] > 0:
                _open_hinge(teacher, hinge, hinge["q0"])
        path = actions.Path(PARK)
        log_w = []
        while fams:
            entry = actions.plan_family(rng, fams.pop(0), path, objs, hinge, counts[w], obstacles=obst)
            if entry is not None:
                log_w.append(entry); _count(counts[w], entry)
                break
        paths.append(path); queues.append(fams); logs.append(log_w)
        pokes_all.append(actions.sample_pokes(rng, objs, seconds, scale=poke_scale))

    # 3. lift the objects to be dropped; put the ramp's objects on it, at rest
    q = teacher.s0.body_q.numpy(); qd = teacher.s0.body_qd.numpy()
    for sp, inf in zip(specs, infos):
        for k, o in enumerate(sp.objects):
            if o.drop_height > 0:
                q[inf["objects"][k], 2] += o.drop_height
                qd[inf["objects"][k]] = 0.0
            if o.on_ramp >= 0 and sp.ramp is not None:
                bi = inf["objects"][k]
                q[bi, :3], q[bi, 3:] = ramp_pose(sp, q[bi, 3:], library.load(o.name)["collide_v"].astype(float), o.on_ramp)
                qd[bi] = 0.0
    teacher.s0.body_q.assign(q); teacher.s0.body_qd.assign(qd)

    # 4. gripper targets at every substep, poke wrenches at every frame
    def fill(w, s_from=0):
        if replay is not None and "track" in replay[w]:
            P[:, w], Qq[:, w], V[:, w], Wz[:, w] = _track_targets(np.asarray(replay[w]["track"]), ts)
            return
        pos, yaw, vel, wz = paths[w].sample(ts[s_from:])
        P[s_from:, w], Qq[s_from:, w], V[s_from:, w], Wz[s_from:, w] = pos, _quat_z(yaw), vel, wz

    for w in range(W):
        fill(w)
    gP.assign(P); gQ.assign(Qq); gV.assign(V); gW.assign(Wz)
    wrench = np.zeros((nf, nb, 6), np.float32)
    com_local = m.body_com.numpy()
    for w, pks in enumerate(pokes_all):
        for pk in pks:
            bi = infos[w]["objects"][pk["object"]]
            f0 = min(int(pk["t"] * FPS), nf - 1)
            r = np.array(pk["point_local"]) - rotate(q[bi, 3:], com_local[bi][None])[0]
            F = np.array(pk["force"]) * specs[w].objects[pk["object"]].mass_scale
            wrench[f0:f0 + pk["frames"], bi, :3] += F
            wrench[f0:f0 + pk["frames"], bi, 3:] += np.cross(r, F)
    for w, r in enumerate(replay or []):
        for key, arr in r.get("wrench", {}).items():
            bi = infos[w]["hinge"]["link"] if key == "hinge" else infos[w]["objects"][int(key)]
            wrench[:, bi] += np.asarray(arr, np.float32)[:nf]
    gWr = wp.array(wrench, dtype=wp.spatial_vector)

    def post(t):
        rec.accumulate()
        state["s"] += 1

    teacher.post_substep = post
    rec.reset()
    rec.start(nf)
    dev = m.device
    bq_rec = wp.zeros((nf + 1) * nb, dtype=wp.transform, device=dev)      # the episode stays on the device
    bqd_rec = wp.zeros((nf + 1) * nb, dtype=wp.spatial_vector, device=dev)
    sat_rec = wp.zeros(nf * W, dtype=wp.int32, device=dev)
    wp.copy(bq_rec, teacher.s0.body_q, count=nb); wp.copy(bqd_rec, teacher.s0.body_qd, count=nb)
    t0 = time.time()
    for f in range(nf):
        state["f"] = f
        teacher.frame()
        wp.copy(bq_rec, teacher.s0.body_q, dest_offset=(f + 1) * nb, count=nb)
        wp.copy(bqd_rec, teacher.s0.body_qd, dest_offset=(f + 1) * nb, count=nb)
        rec.store(f)
        wp.copy(sat_rec, sat, dest_offset=f * W, count=W)
        sat.zero_()
        now, s0 = (f + 1) / FPS, state["s"]
        need = []                                                  # worlds whose next family starts now
        for w in range(W):
            if replay is not None or now < paths[w].t or now > seconds - LAST_START:
                continue
            if queues[w] or ((counts[w] == 0).any() and fills[w] < FILL_TRIES and now <= seconds - FILL_LAST_START):
                need.append(w)
        if not need:
            continue
        q_now = teacher.s0.body_q.numpy()                          # the one sync, only when planning
        for w in need:
            is_fill = False
            if not queues[w]:
                queues[w].append(actions.fill_family(rng)); fills[w] += 1; is_fill = True
            objs = objects_now(q_now, specs[w], infos[w])
            path = paths[w]
            path.k.append((now, path.p.copy(), path.k[-1][2]))
            while queues[w]:
                entry = actions.plan_family(rng, queues[w].pop(0), path, objs, infos[w].get("hinge"), counts[w],
                                            obstacles=obstacles_now(specs[w]))
                if entry is not None:
                    entry["fill"] = is_fill
                    logs[w].append(entry); _count(counts[w], entry)
                    break
            fill(w, s0)
        gP.assign(P); gQ.assign(Qq); gV.assign(V); gW.assign(Wz)
    body_q = bq_rec.numpy().reshape(nf + 1, nb, 7).astype(np.float32)
    body_qd = bqd_rec.numpy().reshape(nf + 1, nb, 6).astype(np.float32)
    saturated = sat_rec.numpy().reshape(nf, W) > 0
    Fc, Pc, Tc = rec.collect()
    log(f"[episode] {W} worlds x {seconds:.0f} s in {time.time() - t0:.1f} s")
    return {"specs": specs, "infos": infos, "plans": list(zip(paths, logs, pokes_all)), "families_sampled": sampled,
            "body_q": body_q,
            "body_qd": body_qd, "contact_F": Fc, "contact_P": Pc, "contact_T": Tc, "K": K, "body_world": m.body_world.numpy(),
            "body_slot": body_slot, "force_cap": caps, "saturated": saturated,
            "poke_wrench": wrench, "substeps": teacher.substeps,
            "body_mass": m.body_mass.numpy(), "body_com": m.body_com.numpy(), "body_inertia": m.body_inertia.numpy()}


def _open_hinge(teacher, hinge, q0):
    """Open a drawer (translate along -y) or lid (rotate about its hinge line) by q0, in place."""
    q = teacher.s0.body_q.numpy()
    b = hinge["link"]
    if hinge["kind"] == "drawer":
        q[b, 1] -= q0
    else:
        a = q0 / 2.0
        rx, rw = -np.sin(a), np.cos(a)                                     # about -x
        x, y, z, w = q[b, 3:]
        q1 = np.array([rw * x + rx * w, rw * y - rx * z, rw * z + rx * y, rw * w - rx * x])
        q[b, 3:] = q1 / np.linalg.norm(q1)
    teacher.s0.body_q.assign(q)
    jq = teacher.s0.joint_q.numpy()
    jq[int(teacher.model.joint_q_start.numpy()[hinge["joint"]])] = q0
    teacher.s0.joint_q.assign(jq)
