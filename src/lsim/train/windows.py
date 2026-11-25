"""Teacher episodes -> training windows over mesh vertices (DIRECTION.md "Encodings").

An episode shard stores body poses; the model sees vertices. Every body gets a learner mesh in its
own frame: scanned objects their isotropic remesh (lsim.data.assets), the gripper and hinged parts
remeshed boxes, the table a flat grid kept only near moving things. A window is H past + K chunk
frames of vertex positions, plus per-object slots (mass, CoM offset, inertia scale, friction,
restitution, mass-model flag, joint damping; each given or hidden) and per-vertex external force
slots in three states (PHASE1_AMENDMENT.md sections 2 and 5): GIVEN (the gripper's contact force
on the vertices it touches), ZERO, or EMPTY (unseen pokes; the gripper's force when hidden, on
every dynamic vertex within 2 cm of it). Forces at window frame j move the physics layer from
frame j to j + 1. The physics layer runs on frame-average velocities, for which the matching
force is the mean of the teacher's frame forces on either side of frame j (exact for forces
constant within a frame; replaying the teacher's own wrenches this way halved the error).
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from lsim.data import library
from lsim.data.scene import CAB, FLOOR_MAT, HANDLE_OFF, RAMP, TABLE_MAT, ramp_frame

EDGE = 0.015
ROLE = {"static": 0, "kinematic": 1, "dynamic": 2}
TYPE = {"rigid": 0, "hinged": 1, "cloth": 2, "soft": 3}
F_ZERO, F_GIVEN, F_EMPTY = 0, 1, 2
FPS = 30.0
P_HIDE_GRIP = 0.3                       # gripper force slot hidden (amendment section 2)
NEAR_HAND = 0.02                        # empty force slots allowed within 2 cm of a kinematic body


def _boxes_points(boxes, spacing, normals=False):
    """Surface points of a union of axis-aligned boxes [(center, half)]: a grid on every face at
    <= spacing. (Subdividing box meshes until the longest edge was short split every face alike,
    so thin walls came out at ~12,000 vertices for one drawer -- 75% of a window's tokens.)
    normals: also each point's outward unit normal (the mean of its faces' at edges and corners)."""
    pts, nrm = [], []
    for c, h in boxes:
        c, h = np.asarray(c, float), np.asarray(h, float)
        ax = [np.linspace(-h[i], h[i], max(2, int(np.ceil(2 * h[i] / spacing)) + 1)) for i in range(3)]
        for a in range(3):
            u, v = [i for i in range(3) if i != a]
            U, V = np.meshgrid(ax[u], ax[v], indexing="ij")
            for sgn in (-1.0, 1.0):
                p = np.zeros((U.size, 3))
                p[:, a], p[:, u], p[:, v] = sgn * h[a], U.ravel(), V.ravel()
                pts.append(p + c)
                nrm.append(np.broadcast_to(np.eye(3)[a] * sgn, p.shape))
    P, inv = np.unique(np.round(np.concatenate(pts), 6), axis=0, return_inverse=True)
    if not normals:
        return P.astype(np.float32)
    Nm = np.zeros_like(P)
    np.add.at(Nm, inv.ravel(), np.concatenate(nrm))
    return P.astype(np.float32), (Nm / np.linalg.norm(Nm, axis=1, keepdims=True).clip(1e-9)).astype(np.float32)


@lru_cache(maxsize=1)
def gripper_mesh():
    """Coarser than objects (2 cm): it is a 10 cm block whose path is given."""
    return _boxes_points([((0.0, 0.0, 0.03), (0.05, 0.02, 0.015)), ((-0.0078, 0.0, -0.02), (0.0075, 0.015, 0.035)),
                          ((0.0078, 0.0, -0.02), (0.0075, 0.015, 0.035))], 0.02)


@lru_cache(maxsize=64)
def cabinet_points(kind, x0, y0):
    """The static cabinet around a hinged part, as surface points and outward normals (world
    frame): the walls the teacher collides with (scene._hinge), so objects pushed against it meet
    something visible."""
    W, D, H, t = CAB["W"], CAB["D"], CAB["H"], CAB["t"]
    if kind == "drawer":
        boxes = [((x0, y0, H + t / 2), (W / 2 + t, D / 2, t / 2)), ((x0 - W / 2 - t / 2, y0, H / 2), (t / 2, D / 2, H / 2)),
                 ((x0 + W / 2 + t / 2, y0, H / 2), (t / 2, D / 2, H / 2)), ((x0, y0 + D / 2 + t / 2, H / 2), (W / 2 + t, t / 2, H / 2))]
    else:
        boxes = [((x0, y0, t / 2), (W / 2, D / 2, t / 2)), ((x0 - W / 2 + t / 2, y0, H / 2), (t / 2, D / 2, H / 2)),
                 ((x0 + W / 2 - t / 2, y0, H / 2), (t / 2, D / 2, H / 2)), ((x0, y0 + D / 2 - t / 2, H / 2), (W / 2, t / 2, H / 2)),
                 ((x0, y0 - D / 2 + t / 2, H / 2), (W / 2, t / 2, H / 2))]
    return _boxes_points(boxes, 0.02, normals=True)


def ramp_points(r, spacing=0.02):
    """The ramp plank's surface as points and outward normals (world frame)."""
    o, ax, lat, n = ramp_frame(r)
    local, nl = _boxes_points([((0.5 * RAMP["L"], 0.0, 0.5 * RAMP["T"]), (0.5 * RAMP["L"], 0.5 * RAMP["W"], 0.5 * RAMP["T"]))],
                              spacing, normals=True)
    Rm = np.stack([ax, lat, n], 0)
    return (o + local @ Rm).astype(np.float32), (nl @ Rm).astype(np.float32)


@lru_cache(maxsize=2)
def hinge_mesh(kind):
    W, D, H, t = CAB["W"], CAB["D"], CAB["H"], CAB["t"]
    if kind == "drawer":
        w, d, hg = W - 0.004, D - 0.004, H - 0.01
        boxes = [((0, 0, t / 2), (w / 2, d / 2, t / 2)), ((-w / 2 + t / 2, 0, hg / 2), (t / 2, d / 2, hg / 2)),
                 ((w / 2 - t / 2, 0, hg / 2), (t / 2, d / 2, hg / 2)), ((0, d / 2 - t / 2, hg / 2), (w / 2, t / 2, hg / 2)),
                 ((0, -d / 2 + t / 2, hg / 2), (w / 2, t / 2, hg / 2)), ((0, -d / 2 - HANDLE_OFF, hg / 2), (0.05, 0.005, 0.008)),
                 ((-0.045, -d / 2 - HANDLE_OFF / 2, hg / 2), (0.005, HANDLE_OFF / 2, 0.008)),
                 ((0.045, -d / 2 - HANDLE_OFF / 2, hg / 2), (0.005, HANDLE_OFF / 2, 0.008))]
    else:
        boxes = [((0.0, -D / 2, 0.0), (W / 2 + 0.005, D / 2 + 0.005, 0.005)), ((0.0, -D - 0.015, 0.0), (0.04, 0.01, 0.005))]
    return _boxes_points(boxes, 0.03)


def _face_edges(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0).astype(np.int32)


def rotate(q, v):
    """q (..., 4) xyzw, v (n, 3) -> (..., n, 3)."""
    u, w = q[..., None, :3], q[..., None, 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


@dataclass
class Body:
    name: str
    role: str
    type: str
    body: int                          # index into the shard's body arrays
    verts: np.ndarray                  # rest vertices in the body frame
    edges: np.ndarray                  # (E, 2) within-body neighbour pairs
    slot: int                          # contact-record slot (0 static, 1 gripper, 2.. objects, K-1 hinge)
    params: dict = field(default_factory=dict)
    inertia_ref: tuple = None          # (eigenvalues, eigenvectors) of the unit-mass solid inertia


class Episode:
    def __init__(self, npz, meta, w):
        self.w = w
        wd = meta["worlds"][w]
        self.qd = npz["body_qd"]
        self.mass, self.com, self.inertia = npz["body_mass"], npz["body_com"], npz["body_inertia"]
        self.meta, self.wd = meta, wd
        self.q = npz["body_q"]                                          # the shard's arrays, shared, not copied
        self.K = meta["K"]
        self.bodies = []
        for k, lab in enumerate(wd["objects"]):
            a = library.load(lab["name"])
            com = np.asarray(lab["com"], float)
            size = float(np.max(a["extents"]))
            lam, R = np.linalg.eigh(np.asarray(a["inertia_solid_unit"], float))   # about the solid CoM = origin
            J = np.asarray(lab["inertia"], float) / lab["mass"]                     # true, per unit mass, about the CoM
            scale = np.diag(R.T @ J @ R) / lam                                       # amendment: mesh inertia x scale
            self.bodies.append(Body(lab["name"], "dynamic", "rigid", wd["object_bodies"][k], a["points_v"], a["points_edges"],
                                    2 + k, {"mass": lab["mass"], "mu": lab["mu"], "restitution": lab["restitution"],
                                            "com_offset": (com / size).tolist(), "inertia_scale": scale.tolist(),
                                            "solid": 1.0 if lab["build"] == "solid" else 0.0, "size": size},
                                    (lam, R)))
        none = np.zeros((0, 2), np.int32)                              # rigid bodies need no mesh edges
        self.bodies.append(Body("gripper", "kinematic", "rigid", wd["gripper_body"], gripper_mesh(), none, 1))
        h = wd.get("hinge")
        if h:
            self.bodies.append(Body(h["kind"], "dynamic", "hinged", h["link"], hinge_mesh(h["kind"]), none, self.K - 1,
                                    {"joint_damping": h["damping"], "mu": h["mu"], "restitution": h["restitution"]}))
        # this world's gripper contact forces, and its unseen pokes as (body, vertices, frames, force)
        # (a per-episode int64 copy of the shard's whole contact index took 650 MB per shard: 305 GB
        # per training process for v0 + v0r)
        idx = npz["contact_idx"]
        sel = (idx[:, 1] == w) & (idx[:, 3] == 1)
        self.g_idx = idx[sel][:, [0, 2]].astype(np.int64)
        self.g_F, self.g_P = npz["contact_F"][sel], npz["contact_P"][sel]
        self.pokes = []
        for pk in wd["pokes"]:
            b = self.bodies[pk["object"]]
            Rt = rotate(self.q[0, b.body, 3:], np.eye(3))       # rows: body axes in world, so Rt @ x is world -> body
            p_body = Rt @ np.asarray(pk["point_local"], float)    # the poke point was stored as a world offset
            dist = np.linalg.norm(b.verts - p_body, axis=1)
            verts = np.flatnonzero(dist < NEAR_HAND)
            if len(verts) == 0:
                verts = np.array([int(dist.argmin())])
            f0 = int(pk["t"] * FPS)
            self.pokes.append((pk["object"], verts, f0, f0 + pk["frames"], np.asarray(pk["force"], float)))

    def vertices(self, frames):
        """World positions of every body's vertices at the given frames: list of (len(frames), n, 3)."""
        out = []
        for b in self.bodies:
            q = self.q[frames, b.body]
            out.append(rotate(q[:, 3:], b.verts.astype(float)) + q[:, None, :3])
        return out

    def gripper_force(self, frame, body):
        """Frame-averaged force on `body` by the gripper and where it acts (world), or None."""
        sel = (self.g_idx[:, 0] == frame) & (self.g_idx[:, 1] == body.slot)
        if not sel.any():
            return None
        return self.g_F[sel][0], self.g_P[sel][0]


def load_shard(path):
    path = Path(path)
    npz = dict(np.load(path.with_suffix(".npz")))
    meta = json.loads(path.with_suffix(".json").read_text())
    return [Episode(npz, meta, w) for w in range(len(meta["worlds"]))]


# ---------------------------------------------------------------------------------------------
# window tensors
# ---------------------------------------------------------------------------------------------
PARAMS = ("mass", "mu", "restitution", "com_x", "com_y", "com_z", "inertia_1", "inertia_2", "inertia_3", "solid",
          "joint_damping")
# slot value = (raw - centre) / scale; raw: log kg, log mu, logit e, CoM offset / size, log inertia
# scale (true per-unit-mass inertia in the mesh's solid principal frame / solid), flag, log damping
CENTRE = np.array([np.log(0.2), np.log(0.35), -0.6, 0, 0, 0, 0, 0, 0, 0, -1.0], np.float32)
SCALE = np.array([1.5, 0.7, 1.3, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 1.0, 2.5], np.float32)


def _param_vector(b):
    """Normalised slot values and which apply to this body."""
    p = b.params
    v = np.zeros(len(PARAMS), np.float32)
    has = np.zeros(len(PARAMS), bool)
    if b.role == "dynamic" and b.type == "rigid":
        e = float(np.clip(p["restitution"], 1e-3, 1 - 1e-3))
        v[:9] = [np.log(p["mass"]), np.log(p["mu"]), np.log(e / (1 - e)), *p["com_offset"],
                 *np.log(np.maximum(np.asarray(p["inertia_scale"]), 1e-6))]
        v[9] = p["solid"]
        has[:10] = True
    elif b.type == "hinged":
        e = float(np.clip(p["restitution"], 1e-3, 1 - 1e-3))
        v[1], v[2], v[10] = np.log(p["mu"]), np.log(e / (1 - e)), np.log(p["joint_damping"])
        has[[1, 2, 10]] = True
    return np.where(has, (v - CENTRE) / SCALE, 0.0).astype(np.float32), has


PLANES = ((0.0, (0.6, 0.5), 0.02, TABLE_MAT), (-0.75, (10.0, 10.0), 0.03, FLOOR_MAT))   # z, half extent, spacing, material


def surface_feature(mu, e):
    """A static surface's material as two vertex features: log friction, logit restitution."""
    e = float(np.clip(e, 1e-3, 1 - 1e-3))
    return np.array([np.log(mu), np.log(e / (1 - e))], np.float32)


def _plane_patch(moving, z, half, spacing, near):
    """Lattice points of a static plane within `near` of any moving point, built only over the
    moving points' footprint (objects roll metres across the floor)."""
    m = moving[np.abs(moving[:, 2] - z) < near]
    if len(m) == 0:
        return np.zeros((0, 3))
    lo = np.maximum(np.floor((m[:, :2].min(0) - near) / spacing), np.ceil(-np.asarray(half) / spacing))
    hi = np.minimum(np.ceil((m[:, :2].max(0) + near) / spacing), np.floor(np.asarray(half) / spacing))
    if (hi < lo).any():
        return np.zeros((0, 3))
    X, Y = np.meshgrid(np.arange(lo[0], hi[0] + 1) * spacing, np.arange(lo[1], hi[1] + 1) * spacing, indexing="ij")
    g = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], -1)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(m).query(g, k=1, distance_upper_bound=near)
    return g[np.isfinite(d)]


def joint_info(ep):
    """Per body: joint kind (0 none, 1 prismatic, 2 revolute), axis and pivot (world), limits in
    the joint coordinate -- the drawer's travel from closed (m) or the lid's angle (rad), both
    read off the link's pose (scene._hinge: drawer along -y from (x0, y0, 0.0005); lid about -x
    through its back edge, closed at the identity orientation)."""
    nb = len(ep.bodies)
    kind, axis, pivot, lim = np.zeros(nb, np.int64), np.zeros((nb, 3)), np.zeros((nb, 3)), np.zeros((nb, 2))
    h = ep.wd.get("hinge")
    for i, b in enumerate(ep.bodies):
        if b.type != "hinged":
            continue
        x0, y0 = map(float, h["xy"])
        if h["kind"] == "drawer":
            kind[i], axis[i], pivot[i], lim[i] = 1, (0.0, -1.0, 0.0), (x0, y0, 0.0005), (0.0, 0.8 * CAB["D"])
        else:
            kind[i], axis[i], pivot[i], lim[i] = 2, (-1.0, 0.0, 0.0), (x0, y0 + CAB["D"] / 2, CAB["H"] + 0.006), (0.0, np.radians(110.0))
    return {"body_joint": kind, "joint_axis": axis.astype(np.float32), "joint_pivot": pivot.astype(np.float32),
            "joint_lim": lim.astype(np.float32)}


def make_window(ep, t_cur, n_past, K, rng=None, near=0.03, p_hide_grip=P_HIDE_GRIP):
    """Frames t_cur - n_past + 1 .. t_cur are given (the last is "now"), t_cur + 1 .. t_cur + K are
    the chunk. Returns numpy arrays; the collate step batches them."""
    rng = np.random.default_rng() if rng is None else rng
    from scipy.spatial import cKDTree
    frames = np.arange(t_cur - n_past + 1, t_cur + K + 1)
    Fw = len(frames)
    fr = np.clip(frames, 0, ep.q.shape[0] - 1)
    pos_b = ep.vertices(fr)
    prev = ep.vertices([max(frames[0] - 1, 0)])
    pos, body_of, role, typ, rest, edges, n0 = [], [], [], [], [], [], 0
    for bi, (b, P) in enumerate(zip(ep.bodies, pos_b)):
        pos.append(P)
        body_of.append(np.full(len(b.verts), bi))
        role.append(np.full(len(b.verts), ROLE[b.role]))
        typ.append(np.full(len(b.verts), TYPE[b.type]))
        size = b.params.get("size", float(np.ptp(b.verts, 0).max()))
        rest.append(b.verts / size)
        if b.type not in ("rigid", "hinged"):                    # cloth/soft keep their mesh edges
            edges.append(b.edges + n0)
        n0 += len(b.verts)
    pos = np.concatenate(pos, 1)                                    # (Fw, Nb, 3)
    role_b = np.concatenate(role)
    # static surfaces: only points within reach of something that moves in this window -- the
    # table top, the floor, and a hinged part's cabinet
    moving = pos[:, role_b > 0].reshape(-1, 3)
    stat, mats, nrms = [], [], []
    for z, half, sp, mat in PLANES:
        p = _plane_patch(moving, z, half, sp, near)
        stat.append(p); mats.append(mat); nrms.append(np.broadcast_to(np.array([0.0, 0.0, 1.0]), p.shape))
    h = ep.wd.get("hinge")
    extra = [(*cabinet_points(h["kind"], *map(float, h["xy"])), (h["mu"], h["restitution"]))] if h else []
    if ep.wd.get("ramp"):
        extra.append((*ramp_points(ep.wd["ramp"]), (ep.wd["ramp"]["mu"], ep.wd["ramp"]["restitution"])))
    for pts, nrm, mat in extra:
        d, _ = cKDTree(moving).query(pts, k=1, distance_upper_bound=near)
        keep = np.isfinite(d)
        stat.append(pts[keep]); nrms.append(nrm[keep]); mats.append(mat)
    surf = np.concatenate([np.broadcast_to(surface_feature(*mt), (len(st), 2)) for st, mt in zip(stat, mats)])
    nrms = np.concatenate(nrms)
    stat = np.concatenate(stat)
    n_static = len(stat)
    nb_verts = pos.shape[1]
    pos = np.concatenate([pos, np.broadcast_to(stat, (Fw, n_static, 3))], 1).astype(np.float32)
    nb = len(ep.bodies)
    body_of = np.concatenate(body_of + [np.full(n_static, nb)])
    role = np.concatenate([role_b, np.zeros(n_static, int)])
    typ = np.concatenate(typ + [np.zeros(n_static, int)])
    rest = np.concatenate(rest + [np.zeros((n_static, 3))]).astype(np.float32)
    prev_pos = np.concatenate([np.concatenate(prev, 1)[0], stat], 0)
    vel = np.diff(np.concatenate([prev_pos[None], pos], 0), axis=0) * FPS

    # external force slots: true value (gripper contact force + unseen pokes) and state per frame
    N = len(body_of)
    offs = np.cumsum([0] + [len(b.verts) for b in ep.bodies])
    T_end = ep.q.shape[0] - 1

    def teacher_frame(f, V):
        """Per-vertex external force during teacher frame f (vertices posed at V): the gripper's
        contact force spread over the vertices within 2 cm of where it acts, and pokes."""
        out, grip, poke = np.zeros((N, 3), np.float32), np.zeros(N, bool), np.zeros(N, bool)
        if f < 0 or f >= T_end:
            return out, grip, poke
        for bi, b in enumerate(ep.bodies):
            if b.role != "dynamic":
                continue
            r = ep.gripper_force(int(f), b)
            if r is None:
                continue
            F, Pc = r
            d = np.linalg.norm(V[offs[bi]:offs[bi + 1]] - Pc, axis=1)
            t = np.flatnonzero(d < 0.02)
            t = offs[bi] + (t if len(t) else np.array([int(d.argmin())]))
            out[t] += F / len(t); grip[t] = True
        for k, verts, a, b_, F in ep.pokes:
            if a <= f < b_:
                out[offs[k] + verts] += (F / len(verts)).astype(np.float32); poke[offs[k] + verts] = True
        return out, grip, poke

    per = [teacher_frame(int(frames[0]) - 1, prev_pos)] + [teacher_frame(int(f), pos[j]) for j, f in enumerate(frames)]
    f_ext = np.stack([0.5 * (per[j][0] + per[j + 1][0]) for j in range(Fw)])
    f_state = np.zeros((Fw, N), np.int8)
    grip_hidden = bool(rng.random() < p_hide_grip)
    gi = next(i for i, b in enumerate(ep.bodies) if b.role == "kinematic")
    dyn = np.flatnonzero(role == ROLE["dynamic"])
    for j in range(Fw):
        touched = per[j][1] | per[j + 1][1]
        if grip_hidden:                                             # what a hand gives: nothing but proximity
            gd, _ = cKDTree(pos[j, offs[gi]:offs[gi + 1]]).query(pos[j, dyn], k=1, distance_upper_bound=NEAR_HAND)
            f_state[j, dyn[np.isfinite(gd)]] = F_EMPTY
            f_state[j, touched] = F_EMPTY
        else:
            f_state[j, touched] = F_GIVEN
        f_state[j, per[j][2] | per[j + 1][2]] = F_EMPTY            # unseen pokes: always empty

    # per-body slots, and the state at "now" as two poses (now - 1, now): the physics layer works
    # with frame-average velocities (finite differences), with which one step per frame can
    # follow any trajectory; with the teacher's instantaneous velocities an impact inside a frame
    # cannot be matched and replaying the teacher's own wrenches missed fast objects by 35 mm
    params, has = zip(*[_param_vector(b) for b in ep.bodies])
    ids = [b.body for b in ep.bodies]
    q_now, q_prev = ep.q[t_cur, ids], ep.q[max(t_cur - 1, 0), ids]
    lam = np.ones((nb, 3)); Rs = np.tile(np.eye(3), (nb, 1, 1)); size = np.ones(nb)
    for i, b in enumerate(ep.bodies):
        if b.inertia_ref is not None:
            lam[i], Rs[i] = b.inertia_ref
            size[i] = b.params["size"]
    return {"frames": frames, "n_past": n_past, "pos": pos, "vel": vel.astype(np.float32), "body_of": body_of,
            "role": role, "type": typ, "rest": rest, "edges": np.concatenate(edges) if edges else np.zeros((0, 2), int),
            "params": np.array(params, np.float32), "param_has": np.array(has),
            "body_q_now": q_now.astype(np.float32), "body_q_prev": q_prev.astype(np.float32),
            "body_q_chunk": ep.q[fr[n_past:]][:, ids].astype(np.float32),
            "body_q_win": ep.q[np.clip(np.r_[frames[0] - 1, frames], 0, ep.q.shape[0] - 1)][:, ids].astype(np.float32),
            "body_mass": ep.mass[ids].astype(np.float32), "body_com": ep.com[ids].astype(np.float32),
            "body_inertia": ep.inertia[ids].astype(np.float32), "body_size": size.astype(np.float32),
            "body_lam": lam.astype(np.float32), "body_R": Rs.astype(np.float32),
            "body_role": np.array([ROLE[b.role] for b in ep.bodies]), "body_type": np.array([TYPE[b.type] for b in ep.bodies]),
            **joint_info(ep),
            "body_verts": [b.verts.astype(np.float32) for b in ep.bodies],
            "f_ext": f_ext, "f_state": f_state, "grip_hidden": grip_hidden, "n_static": n_static,
            "surface": np.concatenate([np.zeros((nb_verts, 2), np.float32), surf]).astype(np.float32),
            "normal": np.concatenate([np.zeros((nb_verts, 3), np.float32), nrms]).astype(np.float32)}
