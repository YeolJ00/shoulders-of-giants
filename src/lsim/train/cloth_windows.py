"""Cloth teacher episodes (lsim.data.cloth shards) -> training windows (phase 3).

A cloth window has the rigid windows' kinds of tokens (lsim.train.windows): the cloth's vertices (dynamic, type
cloth: every vertex moves on its own, held together by its mesh edges), the gripper's points (kinematic), and the
static surface points within reach of the cloth -- the table top, the floor, the plank, the scanned objects on the
table (their learner points, with outward normals). Per (vertex, frame):

  held     the vertex is pinned to the gripper for the step from this frame to the next: its next position is
           given, as the gripper's is (which cloth the gripper holds is part of the action)
  force    the external force slot: on held vertices the grasp force (the gripper's pull, shared equally),
           GIVEN or EMPTY (a hand's force is not measured: p = 0.3); ZERO elsewhere

Per cloth, four slots, each hidden with p = 0.5 (10% of windows all hidden, 10% none): areal density w (kg/m^2),
stretch modulus Y (N/m), bending length c = (B / w g)^(1/3) (m), friction mu; normalised logs. A vertex's mass is
w times its area share (the teacher's lumping), so with the density hidden the physics layer uses the predicted w.

Beyond cloth (2026-10-30; DIRECTION.md "Beyond cloth"): rope (lsim.data.rope) and soft-body (lsim.data.soft) shards
make the same windows (load_shard reads any of the three; RopeEpisode / SoftWindowEpisode give ClothEpisode's
interface). One deformable object per window; its vertices are the "cloth" tokens (KIND["cloth"], batch["is_cloth"]:
read "deformable"), and the window says which material it is (MATERIAL, per body in batch["material"]):

  cloth  rest (u, v, 0) / size; mesh edges, triangles; "area" = each vertex's area share (m^2)
  rope   rest (s, 0, 0) / size (size = its length); chain edges, no triangles; "area" = each node's length share
         (l, l / 2 at the ends: the teacher's lumping, lam l per segment half to each end), m
  soft   rest = the rest mesh (body frame) / size (size = its largest extent); tet edges; "area" = each vertex's
         volume share, the stored lumped mass over rho (m^3); its outward surface triangles in "surf_tris"

so that every vertex's mass is slot 0 (w, lam or rho, given or predicted) x "area" (cloth_step, decode). Four slots
per material, normalised per material (SLOT_NAMES, KIND_CENTRE / KIND_SCALE, logs except soft nu; centres and scales
are the r1 and soft-pilot priors' log means and standard deviations, rounded):

  cloth  w (kg/m^2), Y (N/m), bending length (m), mu       log; centres 0.16, 3e3, 0.027, 0.35; scales 0.6, 1.2, 0.3, 0.7
  rope   lam (kg/m), EA (N, the capped value), bending     log; centres 0.008, 1e3, 0.04, 0.5;  scales 1.8, 2.2, 1.1, 0.55
         length (EI / lam g)^(1/3) (m), mu
  soft   rho (kg/m^3), E (Pa, capped), nu (capped), mu     log rho, E, mu; centres 150, 4e4, 0.25 (nu linear), 0.6;
                                                            scales 1.4, 2.1, 0.14 (nu), 0.65

A soft body's damping (zeta) has no slot: it is part of the material the network must read from motion.
"""
import os
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from lsim.data import library
from lsim.data.scene import FLOOR_MAT, RAMP, TABLE_HALF, ramp_frame
from lsim.teacher.cloth_teacher import grid, mesh_edges
from lsim.train.windows import F_EMPTY, F_GIVEN, F_ZERO, ROLE, _plane_patch, gripper_mesh, ramp_points, surface_feature

FPS = float(os.environ.get("LSIM_FPS", 30.0))        # the windows' frame rate (frame-rate study: 60, 120)
CLOTH_PARAMS = ("density", "stretch", "bend", "mu")
C_CENTRE = np.log(np.array([0.16, 3.0e3, 0.027, 0.35])).astype(np.float32)   # log w, log Y, log c, log mu
C_SCALE = np.array([0.6, 1.2, 0.3, 0.7], np.float32)
GROUPS = [[0], [1], [2], [3]]
P_HIDE_GRASP = 0.3
KIND = {"cloth": 0, "gripper": 1, "static": 2}
R_STATIC = 0.03                                   # static points kept within this of the cloth or gripper
EXPLODED_ABOVE = 200.0                            # contact over weight: a pinched episode (c1 before the fix)
MATERIAL = {"cloth": 0, "rope": 1, "soft": 2}
SLOT_NAMES = {"cloth": CLOTH_PARAMS, "rope": ("lam", "EA", "bend", "mu"), "soft": ("rho", "E", "nu", "mu")}
KIND_CENTRE = np.stack([C_CENTRE, np.log([0.008, 1.0e3, 0.04, 0.5]).astype(np.float32),
                        np.array([np.log(150.0), np.log(4.0e4), 0.25, np.log(0.6)], np.float32)]).astype(np.float32)
KIND_SCALE = np.stack([C_SCALE, np.array([1.8, 2.2, 1.1, 0.55], np.float32),
                       np.array([1.4, 2.1, 0.14, 0.65], np.float32)]).astype(np.float32)
KIND_LOG = np.array([[True] * 4, [True] * 4, [True, True, False, True]])     # soft nu (0-0.45) is linear


def cloth_slots(c):
    """Normalised slot values of a cloth (its shard record)."""
    raw = np.log(np.array([c["w"], c["Y"], c["bending_length"], c["mu"]], np.float64))
    return ((raw - C_CENTRE) / C_SCALE).astype(np.float32)


def decode_slots(v, material=None):
    """Normalised slots (..., 4) -> physical w, Y, c, mu (cloth; material None or 0), or the material's own four
    (material: an int or an array over the leading axis, MATERIAL)."""
    if material is None:
        return np.exp(np.asarray(v) * C_SCALE + C_CENTRE)
    m = np.asarray(material)
    z = np.asarray(v) * KIND_SCALE[m] + KIND_CENTRE[m]
    return np.where(KIND_LOG[m], np.exp(z), z)


def encode_slots(raw, material):
    """Physical slot values (4,) of a material -> normalised."""
    m = MATERIAL[material] if isinstance(material, str) else int(material)
    raw = np.asarray(raw, np.float64)
    z = np.where(KIND_LOG[m], np.log(np.maximum(raw, 1e-30)), raw)
    return ((z - KIND_CENTRE[m]) / KIND_SCALE[m]).astype(np.float32)


def rope_slots(r):
    """Normalised slot values of a rope (its shard record): lam, EA (capped), bending length, mu."""
    return encode_slots([r["lam"], r["EA"], r["bending_length"], r["mu"]], "rope")


def soft_slots(c):
    """Normalised slot values of a soft body (its shard record): rho, E (capped), nu (capped), mu."""
    return encode_slots([c["rho"], c["E"], c["nu"], c["mu"]], "soft")


@lru_cache(maxsize=256)
def _grid(nx, ny, h):
    v, t = grid(nx, ny, h)
    e, _ = mesh_edges(t)
    area = np.zeros(len(v))
    a = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    for k in range(3):
        np.add.at(area, t[:, k], a / 3.0)
    return v.astype(np.float32), t.astype(np.int64), e.astype(np.int64), area.astype(np.float32)


def greedy_colors(elems):
    """Colours of mesh elements (n, k vertex ids: edges, tets) such that no two elements of a colour share a vertex,
    greedily in order (the exact layer's shape restoration sweeps them colour by colour, Gauss-Seidel)."""
    elems = np.asarray(elems, np.int64)
    if len(elems) == 0:
        return np.zeros(0, np.int64)
    used = [0] * (int(elems.max()) + 1)                                 # per vertex: a bit per colour taken
    col = np.zeros(len(elems), np.int64)
    for e, vs in enumerate(elems.tolist()):
        m = 0
        for v in vs:
            m |= used[v]
        c = (~m & (m + 1)).bit_length() - 1                              # the lowest free colour
        col[e] = c
        for v in vs:
            used[v] |= 1 << c
    return col


def mesh_colors(ep):
    """(edge colours, tets (T, 4), tet colours) of an episode's body, computed once per episode."""
    c = getattr(ep, "_mesh_colors", None)
    if c is None:
        tets = np.asarray(getattr(ep, "tets", np.zeros((0, 4), np.int64)), np.int64).reshape(-1, 4)
        c = (greedy_colors(ep.edges), tets, greedy_colors(tets))
        try:
            ep._mesh_colors = c
        except AttributeError:
            pass
    return c


@lru_cache(maxsize=512)
def _object_points(name):
    a = library.load(name)
    return a["points_v"].astype(np.float64), a["points_n"].astype(np.float64)


class ClothEpisode:
    """One world of a cloth shard: the cloth's vertices over the recorded frames, which are held when, the
    gripper's poses, the scene's static surfaces, the recorded forces."""
    material = "cloth"                                                  # its record in the world's JSON, MATERIAL

    def _mesh(self, c):
        return _grid(int(c["nx"]), int(c["ny"]), float(c["h"]))

    def __init__(self, npz, meta, w):
        self.w, self.meta = w, meta
        wd = meta["worlds"][w]
        self.wd, self.c = wd, wd[self.material]
        o0, o1 = int(npz["cloth_off"][w]), int(npz["cloth_off"][w + 1])
        self.cc = npz["cloth_c"][:, w]                                  # (F + 1, 3) centroids
        self.cd = npz["cloth_d"][:, o0:o1]                              # (F + 1, V, 3) offsets (fp16 in c1_pinch)
        self.cv = npz["cloth_v"][:, o0:o1] if "cloth_v" in npz else None  # (F + 1, V, 3) the teacher's instantaneous
        self.grip = npz["grip_q"][:, w]                                 # (F + 1, 7)
        self.grasp_F = npz["grasp_F"][:, w]                             # (F + 1, 3)
        self.contact_F = npz["contact_F"][:, w]                         # (F + 1, 3 surface kinds, 3)
        self.n_frames = self.cc.shape[0]
        self.rest, self.tris, self.edges, self.area = self._mesh(self.c)
        V = len(self.rest)
        # held[j]: pinned during the step from recorded frame j to j + 1 (events at the start of a frame, on
        # the whole timeline; recorded frame j is timeline frame settle + j)
        settle = int(meta["settle_frames"])
        held = np.zeros((self.n_frames, V), bool)
        cur = np.zeros(V, bool)
        ev = sorted(wd["events"], key=lambda e: e[0])
        k = 0
        for j in range(self.n_frames):
            while k < len(ev) and ev[k][0] <= settle + j:
                e = ev[k]
                cur = np.zeros(V, bool)
                if e[1] == "grasp" and len(e) > 3 and len(e[3]):
                    cur[np.asarray(e[3], int)] = True
                k += 1
            held[j] = cur
        self.held = held
        self.qc = wd.get("qc", {})

    def x(self, frames):
        """World positions of the cloth's vertices at the given frames (len, V, 3), float32."""
        frames = np.clip(np.asarray(frames), 0, self.n_frames - 1)
        return self.cd[frames].astype(np.float32) + self.cc[frames][:, None, :].astype(np.float32)

    def gripper_points(self, frames):
        from lsim.train.windows import rotate
        q = self.grip[np.clip(np.asarray(frames), 0, self.n_frames - 1)]
        return (rotate(q[:, 3:], gripper_mesh().astype(np.float64)) + q[:, None, :3]).astype(np.float32)

    def statics(self):
        """Static surface points (world), outward normals, materials (log mu, logit e) and point spacing for the
        scene's plank and objects; the table top and floor are lattices built per window near what moves."""
        pts, nrm, mat, spc, onp = [], [], [], [], []
        wd = self.wd
        if wd.get("ramp"):
            p, n = ramp_points(wd["ramp"], spacing=0.02)
            pts.append(p); nrm.append(n); mat.append(np.broadcast_to(surface_feature(wd["ramp"]["mu"], 0.01), (len(p), 2)))
            spc.append(np.full(len(p), 0.02))
            onp.append(n @ ramp_frame(wd["ramp"])[3] > 0.99)              # its top face: a plane for the default
        for o in wd["obstacles"]:
            v, n = _object_points(o["name"])
            qz, qw = o["quat"][2], o["quat"][3]
            yaw = 2.0 * np.arctan2(qz, qw)
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
            pts.append(v @ R.T + np.asarray(o["pos"])); nrm.append(n @ R.T)
            mat.append(np.broadcast_to(surface_feature(o["mu"], 0.01), (len(v), 2)))
            spc.append(np.full(len(v), float(library.load(o["name"])["learner_edge"])))
            onp.append(np.zeros(len(v), bool))
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), np.float32), np.zeros(0), np.zeros(0, bool)
        return (np.concatenate(pts), np.concatenate(nrm), np.concatenate(mat).astype(np.float32), np.concatenate(spc),
                np.concatenate(onp))

    def planes(self):
        """The flat surfaces as exact rectangles for the contact default: (origin, normal, u, v, half extents along u
        and v, log mu) for the table top, the floor and the plank's top face."""
        out = [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), *TABLE_HALF, np.log(self.wd.get("table_mu", 0.5))),
               ((0.0, 0.0, -0.75), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 10.0, 10.0, np.log(FLOOR_MAT[0]))]
        r = self.wd.get("ramp")
        if r:
            o, ax, lat, n = ramp_frame(r)
            out.append((o + 0.5 * RAMP["L"] * ax + RAMP["T"] * n, n, ax, lat, 0.5 * RAMP["L"], 0.5 * RAMP["W"], np.log(r["mu"])))
        return np.array([np.r_[np.ravel(o), np.ravel(n), np.ravel(u), np.ravel(v), a, b, m] for o, n, u, v, a, b, m in out],
                        np.float32)

    def ok(self):
        """Usable for training: nothing exploded or pinched (c1's episodes before the teacher fix; shards without a
        quality record are checked here)."""
        q = self.qc
        if not q:
            mv = np.linalg.norm(np.diff(self.x(np.arange(self.n_frames)), axis=0), axis=-1).max()
            c = self.c
            wt = c["w"] * c["nx"] * c["ny"] * c["h"] ** 2 * 9.81
            q = {"exploded": bool(mv > 0.4), "contact_over_weight": (np.linalg.norm(self.contact_F, axis=-1).max(0) / wt).tolist()}
            self.qc = q
        return (not q.get("exploded", False)) and max(q.get("contact_over_weight", [0.0])) < EXPLODED_ABOVE


class RopeEpisode(ClothEpisode):
    """One world of a rope shard (lsim.data.rope.save_shard: the cloth shards' fields, kind "rope", a "rope" record
    per world) with ClothEpisode's interface: rest (s, 0, 0) in m along the chain (n_nodes nodes, l apart), chain
    edges, no triangles, each node's length share as its "area"; grasps list the pinned nodes as cloth's do."""
    material = "rope"

    def _mesh(self, r):
        n, l = int(r["n_nodes"]), float(r["l"])
        rest = np.zeros((n, 3), np.float32)
        rest[:, 0] = np.arange(n) * l
        area = np.full(n, l, np.float32)
        area[[0, -1]] = 0.5 * l
        self.size, self.radius, self.h = float(l * (n - 1)), float(r["r"]), l
        return rest, np.zeros((0, 3), np.int64), np.stack([np.arange(n - 1), np.arange(1, n)], 1).astype(np.int64), area

    def slots(self):
        return rope_slots(self.c)


class SoftWindowEpisode(ClothEpisode):
    """A soft body's episode (lsim.data.soft.SoftEpisode) with ClothEpisode's interface for the windows: the rest
    mesh, tet edges, no drag triangles (the soft teacher has no air drag), its outward surface triangles as surf_tris,
    each vertex's volume share (stored lumped mass / rho) as its "area", and the gripper's points with the pads at the
    recorded opening (the cloth gripper's block is a different hand)."""
    material = "soft"

    def __init__(self, se):
        self.se = se
        self.w, self.meta, self.wd, self.c = se.w, se.meta, se.wd, se.c
        self.cc, self.cd, self.grip, self.grasp_F, self.contact_F = se.cc, se.cd, se.grip, se.grasp_F, se.contact_F
        self.grip_w, self.n_frames, self.held, self.qc = se.grip_w, se.n_frames, se.held, se.qc
        self.rest = se.rest.astype(np.float32)
        self.surf_tris = se.tris.astype(np.int64)
        self.tris = np.zeros((0, 3), np.int64)
        self.edges = se.edges.astype(np.int64)
        self.tets = se.tets.astype(np.int64)                             # for the exact layer's volume restoration
        self.area = (se.mass / float(se.c["rho"])).astype(np.float32)
        sv = np.zeros(len(self.rest), bool)
        sv[self.surf_tris.ravel()] = True
        self.surf_v = sv
        self.size = float(max(se.c["extent"]))
        self.radius = float(se.c["r"])
        self.h = float(se.c.get("h_alt", 0.005))
        self.gripper_rest = soft_gripper_points(0.05)

    def slots(self):
        return soft_slots(self.c)

    def gripper_points(self, frames):
        from lsim.train.windows import rotate
        fr = np.clip(np.asarray(frames), 0, self.n_frames - 1)
        q = self.grip[fr]
        local = np.stack([soft_gripper_points(float(w)) for w in self.grip_w[fr]]).astype(np.float64)
        return (rotate(q[:, 3:], local) + q[:, None, :3]).astype(np.float32)

    def ok(self):
        return self.se.ok()


SOFT_FINGER_HALF = (0.006, 0.015, 0.045)            # lsim.teacher.soft_teacher.FINGER_HALF / PALM_HALF (copied: no
SOFT_PALM_HALF = (0.118, 0.025, 0.012)              # teacher import for a constant)


@lru_cache(maxsize=1)
def _soft_gripper_parts():
    from lsim.train.windows import _boxes_points
    fh, ph = SOFT_FINGER_HALF, SOFT_PALM_HALF
    palm = _boxes_points([((0.0, 0.0, 2 * fh[2] + ph[2]), ph)], 0.02)
    pad = _boxes_points([((0.0, 0.0, fh[2]), fh)], 0.02)
    return palm, pad


def soft_gripper_points(w):
    """The soft teacher's gripper (lsim.teacher.soft_teacher.finger_boxes) as points in its frame at inner opening w
    (m): the palm and both pads at +-(w / 2 + pad half thickness) along x; a fixed number of points at any w."""
    palm, pad = _soft_gripper_parts()
    s = w / 2 + SOFT_FINGER_HALF[0]
    return np.concatenate([palm, pad - [s, 0.0, 0.0], pad + [s, 0.0, 0.0]]).astype(np.float32)


def load_shard(path, only_ok=True):
    """Any deformable shard -- cloth (lsim.data.cloth), rope (lsim.data.rope) or soft (lsim.data.soft), by its JSON's
    kind -- as episodes with ClothEpisode's interface."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    kind = meta.get("kind", "cloth")
    if kind == "soft":
        from lsim.data.soft import load_soft_shard
        eps = [SoftWindowEpisode(e) for e in load_soft_shard(path, only_ok=False)]
    elif kind == "rope":
        npz = dict(np.load(path.with_suffix(".npz")))
        eps = [RopeEpisode(npz, meta, w) for w in range(len(meta["worlds"]))]
    else:
        return load_cloth_shard(path, only_ok)
    return [e for e in eps if e.ok()] if only_ok else eps


def load_cloth_shard(path, only_ok=True):
    path = Path(path)
    npz = dict(np.load(path.with_suffix(".npz")))
    meta = json.loads(path.with_suffix(".json").read_text())
    assert float(meta.get("fps", 30.0)) == FPS, f"{path}: recorded at {meta.get('fps')} fps, the windows at {FPS:g} (LSIM_FPS)"
    eps = [ClothEpisode(npz, meta, w) for w in range(len(meta["worlds"]))]
    return [e for e in eps if e.ok()] if only_ok else eps


def make_cloth_window(ep, t_cur, n_past, K, rng=None, near=R_STATIC, p_hide_grasp=P_HIDE_GRASP):
    """Frames t_cur - n_past + 1 .. t_cur are given (the last is "now"), t_cur + 1 .. t_cur + K the chunk. Vertex
    order: cloth, gripper, static. Returns numpy arrays; collate_cloth batches them."""
    rng = np.random.default_rng() if rng is None else rng
    from scipy.spatial import cKDTree
    frames = np.arange(t_cur - n_past + 1, t_cur + K + 1)
    Fw = len(frames)
    fr = np.clip(frames, 0, ep.n_frames - 1)
    xc = ep.x(fr)                                                        # (Fw, Vc, 3)
    xprev = ep.x([max(frames[0] - 1, 0)])[0]
    xg = ep.gripper_points(fr)                                           # (Fw, Vg, 3)
    gprev = ep.gripper_points([max(frames[0] - 1, 0)])[0]
    Vc, Vg = xc.shape[1], xg.shape[1]
    moving = np.concatenate([xc.reshape(-1, 3), xg.reshape(-1, 3)])
    # static surfaces near anything that moves in the window
    # static surfaces near anything that moves, as points for the network (the table top and floor at 2 cm); the
    # contact default takes the flat ones as exact rectangles (planes()) and only the objects' points as points (a
    # 1 cm table lattice for the default made static points 48% of a window's tokens)
    stat, nrm, mat, spc, onp = [], [], [], [], []
    for z, half, m in ((0.0, TABLE_HALF, (ep.wd.get("table_mu", 0.5), 0.01)), (-0.75, (10.0, 10.0), FLOOR_MAT)):
        p = _plane_patch(moving, z, half, 0.02, near)
        stat.append(p); nrm.append(np.broadcast_to([0.0, 0.0, 1.0], p.shape)); mat.append(np.broadcast_to(surface_feature(*m), (len(p), 2)))
        spc.append(np.full(len(p), 0.02)); onp.append(np.ones(len(p), bool))
    sp_, sn_, sm_, ss_, so_ = ep.statics()
    if len(sp_):
        d, _ = cKDTree(moving).query(sp_, k=1, distance_upper_bound=near)
        keep = np.isfinite(d)
        stat.append(sp_[keep]); nrm.append(sn_[keep]); mat.append(sm_[keep]); spc.append(ss_[keep]); onp.append(so_[keep])
    stat, nrm, mat, spc, onp = (np.concatenate(stat), np.concatenate(nrm), np.concatenate(mat), np.concatenate(spc),
                                np.concatenate(onp))
    Vs = len(stat)
    pos = np.concatenate([xc, xg, np.broadcast_to(stat, (Fw, Vs, 3))], 1).astype(np.float32)
    prev = np.concatenate([xprev, gprev, stat], 0)
    vel = np.diff(np.concatenate([prev[None], pos], 0), axis=0) * FPS
    N = Vc + Vg + Vs
    kind = np.concatenate([np.full(Vc, KIND["cloth"]), np.full(Vg, KIND["gripper"]), np.full(Vs, KIND["static"])])
    role = np.concatenate([np.full(Vc, ROLE["dynamic"]), np.full(Vg, ROLE["kinematic"]), np.full(Vs, ROLE["static"])])
    kind_m = getattr(ep, "material", "cloth")
    size = float(max(ep.c["nx"], ep.c["ny"]) * ep.c["h"]) if kind_m == "cloth" else ep.size
    rest = np.zeros((N, 3), np.float32)
    rest[:Vc] = (ep.rest - ep.rest.mean(0)) / size
    grest = getattr(ep, "gripper_rest", None)                          # a real hand's points, or the gripper's
    rest[Vc:Vc + Vg] = (gripper_mesh() if grest is None else grest) / 0.1
    held = np.zeros((Fw, N), bool)
    held[:, :Vc] = ep.held[fr]
    held[frames < 0] = False
    # the grasp force on held vertices: GIVEN (shared equally) or EMPTY; frame j's force moves j -> j + 1, the mean
    # of the recorded frame forces on either side (as the rigid windows do)
    f_ext = np.zeros((Fw, N, 3), np.float32)
    f_state = np.full((Fw, N), F_ZERO, np.int8)
    hidden = bool(rng.random() < p_hide_grasp) or not getattr(ep, "grasp_measured", True)   # real: never measured
    for j, f in enumerate(fr):
        h = np.flatnonzero(held[j, :Vc])
        if len(h) and frames[j] >= 0:
            F = 0.5 * (ep.grasp_F[f] + ep.grasp_F[min(f + 1, ep.n_frames - 1)])
            f_ext[j, h] = F / len(h)
            f_state[j, h] = F_EMPTY if hidden else F_GIVEN
    surface = np.zeros((N, 2), np.float32)
    surface[Vc + Vg:] = mat
    spacing = np.zeros(N, np.float32)
    spacing[Vc + Vg:] = spc
    point_contact = np.zeros(N, bool)                                   # static points the default treats as points
    point_contact[Vc + Vg:] = ~onp
    normal = np.zeros((N, 3), np.float32)
    normal[Vc + Vg:] = nrm
    extra = {}
    if getattr(ep, "cv", None) is not None:                             # the teacher's instantaneous cloth velocities
        vt = np.zeros((Fw, N, 3), np.float32)                           # (record_v shards; --oracle_v reads the past's)
        vt[:, :Vc] = ep.cv[fr]
        vt[frames < 0] = 0.0
        extra["vtrue"] = vt
    return {**extra, "frames": frames, "n_past": n_past, "pos": pos, "vel": vel.astype(np.float32), "kind": kind, "role": role,
            "rest": rest, "held": held, "edges": ep.edges, "tris": ep.tris, "area": ep.area, "n_cloth": Vc,
            "params": cloth_slots(ep.c) if kind_m == "cloth" else ep.slots(), "f_ext": f_ext, "f_state": f_state,
            "grasp_hidden": hidden, "surface": surface, "normal": normal, "spacing": spacing, "point_contact": point_contact,
            "planes": ep.planes(), "radius": float(ep.c["r"]) if kind_m == "cloth" else ep.radius,
            "h": float(ep.c["h"]) if kind_m == "cloth" else ep.h, "size": size, "material": MATERIAL[kind_m],
            "surf_tris": getattr(ep, "surf_tris", ep.tris),             # area-type quantities (soft: its surface)
            "surf_v": getattr(ep, "surf_v", None),                       # vertices that can touch anything (None: all)
            **dict(zip(("edge_color", "tets", "tet_color"), mesh_colors(ep)))}   # for the shape restoration


def collate_cloth(wins, device, rng, p_hidden=0.5, augment=True, all_past=False, hidden=None, aug_yaw=True,
                  aug_shift=True):
    """Concatenate cloth windows: vertices (with per-window sample ids), the cloths' mesh edges and triangles
    offset into the batch, one body per window (its four slots) and a random yaw + shift per window."""
    import torch
    from lsim.train.train import _yaw
    keys = ("pos", "vel", "rest", "kind", "role", "held", "f_ext", "f_state", "surface", "normal", "sample_of", "absent",
            "is_cloth", "area", "spacing", "point_contact")
    has_vt = all("vtrue" in w for w in wins)                           # every window from a record_v shard
    if has_vt:
        keys = keys + ("vtrue",)
    B = {k: [] for k in keys}
    edges, tris, params, given, cloth_of, radius, size, planes, plane_s = [], [], [], [], [], [], [], [], []
    material, surf_tris, surf_v, segs, seg_r = [], [], [], [], []
    e_col, tets, t_col = [], [], []
    nv = 0
    P = wins[0]["n_past"]
    for s, w in enumerate(wins):
        Fw, n = w["pos"].shape[:2]
        R = _yaw(rng.uniform(-np.pi, np.pi)) if augment else np.eye(3, dtype=np.float32)
        t = np.r_[rng.uniform(-0.2, 0.2, 2), 0.0].astype(np.float32) if augment else np.zeros(3, np.float32)
        if not aug_yaw:                                                  # diagnostics (l2_overfit_batch --aug shift / yaw):
            R = np.eye(3, dtype=np.float32)                              # one half of the augmentation alone
        if not aug_shift:
            t = np.zeros(3, np.float32)
        B["pos"].append(torch.from_numpy(w["pos"] @ R.T + t).transpose(0, 1))
        B["vel"].append(torch.from_numpy(w["vel"] @ R.T).transpose(0, 1))
        if has_vt:
            B["vtrue"].append(torch.from_numpy(w["vtrue"] @ R.T).transpose(0, 1))
        B["f_ext"].append(torch.from_numpy(w["f_ext"] @ R.T).transpose(0, 1))
        B["f_state"].append(torch.from_numpy(w["f_state"]).transpose(0, 1))
        B["held"].append(torch.from_numpy(w["held"]).transpose(0, 1))
        B["rest"].append(torch.from_numpy(w["rest"]))
        B["kind"].append(torch.from_numpy(w["kind"]))
        B["role"].append(torch.from_numpy(w["role"]))
        B["surface"].append(torch.from_numpy(w["surface"]))
        B["normal"].append(torch.from_numpy(w["normal"] @ R.T))
        B["spacing"].append(torch.from_numpy(w["spacing"]))
        B["point_contact"].append(torch.from_numpy(w["point_contact"]))
        pl = w["planes"].copy()
        pl[:, 0:3] = pl[:, 0:3] @ R.T + t
        for k in (3, 6, 9):
            pl[:, k:k + 3] = pl[:, k:k + 3] @ R.T
        planes.append(torch.from_numpy(pl)); plane_s.append(torch.full((len(pl),), s))
        B["sample_of"].append(torch.full((n,), s))
        B["is_cloth"].append(torch.from_numpy(w["kind"] == KIND["cloth"]))
        area = np.zeros(n, np.float32)
        area[:w["n_cloth"]] = w["area"]
        B["area"].append(torch.from_numpy(area))
        n_prev = P - 1 if all_past else int(rng.integers(0, P))
        absent = np.zeros(Fw, bool)
        absent[:P - 1 - n_prev] = True
        absent |= w["frames"] < 0
        B["absent"].append(torch.from_numpy(np.broadcast_to(absent, (n, Fw)).copy()))
        edges.append(torch.from_numpy(w["edges"]) + nv)
        tris.append(torch.from_numpy(w["tris"]) + nv)
        params.append(torch.from_numpy(w["params"]))
        hide = rng.random(len(GROUPS)) < p_hidden
        u = rng.random()
        hide[:] = True if (u < 0.1 and p_hidden > 0) else (False if u < 0.2 or p_hidden == 0 else hide)
        if hidden is not None:
            hide[:] = np.isin(CLOTH_PARAMS, list(hidden))
        given.append(torch.from_numpy(~hide))
        cloth_of.append(torch.full((n,), s))
        radius.append(w["radius"])
        size.append(w["size"])
        m = int(w.get("material", 0))
        material.append(m)
        surf_tris.append(torch.from_numpy(np.asarray(w.get("surf_tris", w["tris"]))).long() + nv)
        sv = np.zeros(n, bool)
        sv[:w["n_cloth"]] = True if w.get("surf_v") is None else w["surf_v"]
        sv[w["n_cloth"]:] = True
        surf_v.append(torch.from_numpy(sv))
        if m == MATERIAL["rope"]:                                        # the rope teacher's air drag acts per segment
            segs.append(torch.from_numpy(w["edges"]).long() + nv)
            seg_r.append(torch.full((len(w["edges"]),), float(w["radius"])))
        ec_, tt_, tc_ = ((w["edge_color"], w["tets"], w["tet_color"]) if "edge_color" in w else
                         (greedy_colors(w["edges"]), np.zeros((0, 4), np.int64), np.zeros(0, np.int64)))
        e_col.append(torch.from_numpy(ec_)); tets.append(torch.from_numpy(tt_) + nv); t_col.append(torch.from_numpy(tc_))
        nv += n
    out = {k: torch.cat(v).to(device) for k, v in B.items()}
    for k in ("pos", "vel", "f_ext", "rest", "normal", "area", "spacing") + (("vtrue",) if has_vt else ()):
        out[k] = out[k].float()
    out["edges"] = torch.cat(edges).to(device).long()
    out["tris"] = torch.cat(tris).to(device).long()
    out["params"] = torch.stack(params).to(device).float()
    out["param_given"] = torch.stack(given).to(device)
    out["body_of"] = torch.cat(cloth_of).to(device)                   # one body (the cloth) per window
    out["radius"] = torch.tensor(radius, device=device).float()
    out["size"] = torch.tensor(size, device=device).float()
    out["planes"] = torch.cat(planes).to(device).float()               # (n_planes, 15)
    out["plane_sample"] = torch.cat(plane_s).to(device)
    out["n_bodies"] = len(wins)
    out["n_past"] = P
    # beyond cloth (read by the 3D model, the losses and the exact layer; for cloth alone they change nothing): each
    # body's material, the surface triangles (cloth: its triangles), which vertices can touch anything (a soft body's
    # surface; the rest all), the rope segments for its air drag
    out["material"] = torch.tensor(material, device=device, dtype=torch.long)
    out["surf_tris"] = torch.cat(surf_tris).to(device).long()
    out["surf_v"] = torch.cat(surf_v).to(device)
    out["segs"] = torch.cat(segs).to(device).long() if segs else torch.zeros(0, 2, dtype=torch.long, device=device)
    out["seg_r"] = torch.cat(seg_r).to(device).float() if seg_r else torch.zeros(0, device=device)
    # for the exact layer's shape restoration (off by default): edge colours (no two edges of a colour share a
    # vertex), a soft body's tets and their colours
    out["edge_color"] = torch.cat(e_col).to(device).long()
    out["tets"] = torch.cat(tets).to(device).long()
    out["tet_color"] = torch.cat(t_col).to(device).long()
    return out
