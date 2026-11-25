"""Cloth episodes (phase 3): towels and napkins on the table, handled by the gripper, simulated by the cloth
teacher (lsim.teacher.cloth_teacher) and recorded as shards next to the rigid ones.

A scene: one cloth (15-50 cm a side at 1 cm cells, material sampled in physical units), 0-2 static household
objects on the table (scanned GSO objects, collided as their meshes; they stay put), the gripper parked. The
cloth starts flat, partly over the table edge, crumpled (dropped from a height before recording), draped over
an object, or held up by a corner. Recorded: 4 s at 30 fps after an unrecorded settling phase, during which the
gripper does one or two action families, each planned from the cloth's state when it starts:

  lift_drop    grasp, lift 15-40 cm, hold, let go               mass (grasp force), drag, bending (how it hangs)
  lift_place   grasp, carry 10-30 cm at 8-25 cm up, lay it down  bending, friction (where it lands, how it slides)
  drag         grasp, pull 10-35 cm along the table             friction, stretch (bunching vs sliding)
  fold         grasp a corner, carry it onto another corner      bending (fold radius), self-contact, friction
  shake        grasp, lift, shake 2-6 cm at 1.5-3 Hz             stretch and mass (dynamic tension), damping
  push         closed fingers drive into the cloth's edge          friction, bending (wrinkles)
  slide        grasp, drag fast, let go while moving: it skids     friction (stopping distance v^2 / 2 mu g)

The table's friction varies per episode (0.2-1.0) and 12% of cloths start on a plank at 10-40 deg, where they
slide or stick: with the table fixed at 0.5 and no slope, friction was identified in 37% of actions (gate).

Which parameter each family exposes is the PHASE1_AMENDMENT principle applied to cloth; whether it really does
is measured by the cloth identifiability gate, not assumed.

Long episodes (sample_spec(actions=LONG_ACTIONS), run_batch(record=20, idle=LONG_IDLE, fit=True, keep=True,
own_settle=True)): 15-30 s recorded,
4-8 families chained, each planned from the state the previous one left (folded, then dragged, then dropped on a
pile), a family redrawn until it completes before the end, and after each one, sometimes, an idle gap in which the
cloth settles. With the defaults (record RECORD, actions, idle and fit off) the 4 s episodes come out exactly as
before: the long options draw nothing from the random streams unless they are on.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp
import newton

from lsim.data import library
from lsim.data.scene import FLOOR_MAT, GRIPPER_MU, PARK, RAMP, TABLE_HALF, ramp_frame, resting_z, sample_ramp
from lsim.teacher.cloth_teacher import (FPS, G, ROLE_GRIPPER, ROLE_OBSTACLE, ROLE_TABLE, Cloth, ClothTeacher,
                                        add_cloth, grid, stable_substeps)

I = wp.quat_identity()
RECORD = 4.0                                      # s recorded
TIP = 0.055                                       # m: gripper origin above its fingertips
PLACE = 0.02                                      # m: fingertips above the table when letting go (a grasp holds
                                                  # the layers within 1 cm under them; lower and they hit the table)
WORKSPACE = (0.42, 0.34)                          # cloth centres and obstacles stay inside +-x, +-y
FAMILIES = ("lift_drop", "lift_place", "drag", "fold", "shake", "push", "slide")
FAMILY_P = np.array([0.18, 0.15, 0.17, 0.15, 0.09, 0.11, 0.15])
INITS = ("flat", "overhang", "crumpled", "draped", "held", "ramp")
INIT_P = np.array([0.28, 0.13, 0.2, 0.14, 0.13, 0.12])
SETTLE = {"flat": 0.3, "overhang": 0.8, "crumpled": 1.5, "draped": 1.2, "held": 1.2, "ramp": 0.1}   # s, unrecorded
LONG_ACTIONS = (4, 8)                             # long episodes: families per episode, uniform
LONG_IDLE = (0.35, 1.0, 3.0)                      # ... and after each, with probability p, a further lo-hi s idle
PLAN_TRIES = 6                                    # fit: family draws per planning slot before giving the slot up
KEEP_TRIES = 12                                   # keep: direction draws before falling back to the table centre
KEEP_MARGIN = 0.02                                # m: keep's footprint stays this far inside the table edge


def sample_cloth(rng, h=0.01):
    """The cloth prior (DIRECTION.md "Parameter priors"): sides 15-50 cm; areal density 0.05-0.5 kg/m^2;
    stretch modulus 3e2-3e4 N/m with the membrane wave speed sqrt(Y / w) <= 316 m/s (real fabrics: light
    ones are not the stiffest; it also bounds the explicit step); bending length 1.5-5 cm; friction 0.1-1.2 (the
    objects' range: silk to rubberised fabric; 0.3-1.0 left it identified in half the actions). Log-uniform,
    independent (decorrelated for identifiability)."""
    while True:
        w = float(np.exp(rng.uniform(np.log(0.05), np.log(0.5))))
        Y = float(np.exp(rng.uniform(np.log(3e2), np.log(3e4))))
        if Y / w <= 1.0e5:
            break
    c = float(np.exp(rng.uniform(np.log(0.015), np.log(0.05))))
    nx, ny = (int(rng.integers(15, 51)) for _ in range(2))
    if max(nx, ny) > 2 * min(nx, ny):                  # at most 2:1
        ny = int(np.clip(ny, (nx + 1) // 2, 2 * nx))
    return Cloth(nx=nx, ny=ny, h=h, w=w, Y=Y, nu=0.3, B=w * G * c ** 3,
                 mu=float(np.exp(rng.uniform(np.log(0.1), np.log(1.2)))))


@dataclass
class Obstacle:
    name: str
    xy: tuple
    yaw: float
    mu: float


@dataclass
class ClothSpec:
    cloth: Cloth
    init: str
    pose: tuple                                    # x, y of the cloth centre, yaw (flat inits)
    obstacles: list = field(default_factory=list)
    n_families: int = 1
    seed: int = 0
    table_mu: float = 0.5
    ramp: dict = None


def sample_spec(rng, names, seed=0, actions=None):
    """A cloth scene from the priors. actions (lo, hi): a long episode's number of families, drawn uniformly after
    every other draw (None: one or two, none or one on a plank, as c1)."""
    cloth = sample_cloth(rng)
    init = str(rng.choice(INITS, p=INIT_P))
    n_obs = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
    if init == "draped":
        n_obs = max(n_obs, 1)
    L = cloth.h * max(cloth.nx, cloth.ny)
    if init == "overhang":                         # centre so that 10-45% of its length is past the edge
        side = float(rng.choice([-1, 1]))
        pose = (side * (TABLE_HALF[0] - cloth.nx * cloth.h / 2 + rng.uniform(0.1, 0.45) * cloth.nx * cloth.h),
                float(rng.uniform(-0.2, 0.2)), 0.0)
    else:
        pose = (float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-np.pi, np.pi)))
    obstacles = []
    for k in range(n_obs):
        for _try in range(100):
            if init == "draped" and k == 0:         # under the cloth's centre
                xy = (pose[0], pose[1])
            else:                                   # clear of the cloth (unless it is dropped from above)
                xy = (float(rng.uniform(-WORKSPACE[0], WORKSPACE[0])), float(rng.uniform(-WORKSPACE[1], WORKSPACE[1])))
                if init in ("flat", "overhang", "held") and np.hypot(xy[0] - pose[0], xy[1] - pose[1]) < 0.71 * L + 0.09:
                    continue
            if all(np.hypot(xy[0] - o.xy[0], xy[1] - o.xy[1]) > 0.18 for o in obstacles):
                break
        else:
            continue
        obstacles.append(Obstacle(str(rng.choice(names)), xy, float(rng.uniform(-np.pi, np.pi)),
                                  float(np.exp(rng.uniform(np.log(0.2), np.log(1.0))))))
    if init == "draped" and not obstacles:
        init = "crumpled"
    table_mu = float(np.exp(rng.uniform(np.log(0.2), np.log(1.0))))
    ramp, n_fam = None, int(rng.choice([1, 2], p=[0.6, 0.4]))
    if init == "ramp":
        ramp = sample_ramp(rng)
        if ramp is not None:                           # steeper than the objects' 5-30 deg: at those angles most
            ramp["angle"] = float(np.radians(rng.uniform(10.0, 40.0)))   # cloths stick, whatever their friction
            o, ax, lat, n = ramp_frame(ramp)
            corners = [o + sx * RAMP["L"] * ax + sy * 0.5 * RAMP["W"] * lat for sx in (0, 1) for sy in (-1, 1)]
            if not all(abs(c[0]) < TABLE_HALF[0] - 0.03 and abs(c[1]) < TABLE_HALF[1] - 0.03 for c in corners):
                ramp = None
        if ramp is None:
            init = "flat"
        else:
            obstacles = [o for o in obstacles if np.hypot(o.xy[0] - ramp["mid"][0], o.xy[1] - ramp["mid"][1]) > 0.35]
            n_fam = int(rng.choice([0, 1]))                # often just watch it slide or stick
    if actions is not None:
        n_fam = int(rng.integers(actions[0], actions[1] + 1))
    return ClothSpec(cloth, init, pose, obstacles, n_fam, seed, table_mu, ramp)


def _cloth_placement(spec, rng):
    """Initial pose of the cloth grid (its local origin and rotation)."""
    c = spec.cloth
    Lx, Ly = c.nx * c.h, c.ny * c.h
    x, y, yaw = spec.pose
    if spec.init in ("flat", "overhang"):
        z, rot = c.r + 0.001, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    elif spec.init == "draped":                   # flat, above the obstacle
        a = library.load(spec.obstacles[0].name)
        z = float(np.ptp(a["collide_v"][:, 2])) + 0.02
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    elif spec.init == "crumpled":                 # tilted in the air, clear of everything: it lands folded
        ax = rng.normal(size=3)
        rot = wp.quat_from_axis_angle(wp.vec3(*(ax / np.linalg.norm(ax))), float(rng.uniform(0.6, 1.4)))
        v, _ = grid(c.nx, c.ny, c.h)
        lowest = float(((v - np.array([Lx / 2, Ly / 2, 0.0])) @ _rot(rot).T)[:, 2].min())
        top = max([0.0] + [float(np.ptp(library.load(o.name)["collide_v"][:, 2])) for o in spec.obstacles])
        z = top + 0.03 - lowest + float(rng.uniform(0.0, 0.15))
    elif spec.init == "ramp":                     # flat on the plank, across it, its middle on the plank's middle
        o, ax, lat, n = ramp_frame(spec.ramp)
        R = np.stack([ax, lat, n], 1)
        rot = wp.quat_from_matrix(wp.mat33(*R.flatten()))
        top = o + 0.5 * RAMP["L"] * ax + (RAMP["T"] + c.r + 0.001) * n
        return top - R @ np.array([Lx / 2, Ly / 2, 0.0]), rot
    else:                                         # held: flat, the settling phase lifts it by a corner
        z, rot = c.r + 0.001, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    R = _rot(rot)
    origin = np.array([x, y, z]) - R @ np.array([Lx / 2, Ly / 2, 0.0])
    return origin, rot


def _rot(q):
    return np.array(wp.quat_to_matrix(q), np.float64).reshape(3, 3)


def build_world(spec, rng):
    """Newton builder for one cloth scene; the shape roles in order of creation."""
    b = newton.ModelBuilder()
    roles = []

    def cfg(mu):
        s = newton.ModelBuilder.ShapeConfig()
        s.mu, s.ke, s.kd, s.kf = float(mu), 1.0e3, 0.0, 1.0
        return s
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), I), hx=TABLE_HALF[0], hy=TABLE_HALF[1], hz=0.05,
                    cfg=cfg(spec.table_mu)); roles.append(ROLE_TABLE)
    if spec.ramp is not None:                      # a plank leaning on the table (lsim.data.scene's ramp)
        o, ax, lat, n = ramp_frame(spec.ramp)
        c = o + 0.5 * RAMP["L"] * ax + 0.5 * RAMP["T"] * n
        q = wp.quat_from_matrix(wp.mat33(*np.stack([ax, lat, n], 1).flatten()))
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*c), q), hx=0.5 * RAMP["L"], hy=0.5 * RAMP["W"],
                        hz=0.5 * RAMP["T"], cfg=cfg(spec.ramp["mu"])); roles.append(ROLE_TABLE)
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.80), I), hx=10.0, hy=10.0, hz=0.05,
                    cfg=cfg(FLOOR_MAT[0])); roles.append(ROLE_TABLE)
    obs = []
    for o in spec.obstacles:
        a = library.load(o.name)
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), o.yaw)
        pos = (o.xy[0], o.xy[1], resting_z(a))
        b.add_shape_mesh(body=-1, xform=wp.transform(wp.vec3(*pos), q),
                         mesh=newton.Mesh(a["collide_v"].astype(np.float32), a["collide_f"].flatten().astype(np.int32)),
                         cfg=cfg(o.mu)); roles.append(ROLE_OBSTACLE)
        obs.append({"name": o.name, "library": library.source_of(o.name), "pos": list(map(float, pos)), "quat": [0.0, 0.0, float(np.sin(o.yaw / 2)), float(np.cos(o.yaw / 2))],
                    "mu": o.mu})
    g = b.add_body(xform=wp.transform(wp.vec3(*PARK), I), mass=0.0, label="gripper")   # kinematic: its path is given
    for c, hh in (((0.0, 0.0, 0.03), (0.05, 0.02, 0.015)), ((-0.0078, 0.0, -0.02), (0.0075, 0.015, 0.035)),
                  ((0.0078, 0.0, -0.02), (0.0075, 0.015, 0.035))):
        s = cfg(GRIPPER_MU)
        s.density = 0.0
        b.add_shape_box(body=g, xform=wp.transform(wp.vec3(*c), I), hx=hh[0], hy=hh[1], hz=hh[2], cfg=s)
        roles.append(ROLE_GRIPPER)
    origin, rot = _cloth_placement(spec, rng)
    p0, n, e0, ne = add_cloth(b, spec.cloth, origin, rot)
    return b, {"gripper": g, "particles": (p0, n), "edges": (e0, ne), "obstacles": obs, "roles": roles,
               "origin": origin.tolist(), "rot": [float(v) for v in rot]}


# ---------------------------------------------------------------------------------------------------------
# what the gripper must clear
# ---------------------------------------------------------------------------------------------------------
CELL = 0.004                                      # m: height map resolution
FINGERS = (0.0153, 0.015)                         # half extents (x, y) of the closed fingers (build_world's boxes)
HAND = (0.05, 0.02)                               # the hand box, its underside 1.5 cm above the gripper origin
CLEAR = 0.005                                     # m: fingertips above a static surface; a cloth layer between
                                                  # is squeezed by at most 0.5 N a vertex
HOLD_CLEAR = 0.015                                # m while holding: a grasp pins the layers up to 1 cm under the
                                                  # fingertips, and the held layer must not be driven through the
                                                  # one under it (fold onto an object: 6 of 64 folds crossed)
LEAD = 6                                          # frames: the gripper starts rising this early before an edge
PRESS = 0.01                                      # N: held cloth pushed up this hard by what is under it ...
GUARD_RISE = 0.001                                # m per frame: ... stops the gripper's descent and lifts it
_SAMPLES = {}


def _surface_samples(name, spacing=0.002):
    """Points on an object's collision surface (its frame), about one per spacing^2, and its vertices."""
    if name not in _SAMPLES:
        a = library.load(name)
        v, f = a["collide_v"].astype(np.float64), a["collide_f"].reshape(-1, 3)
        tri = v[f]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        rng = np.random.default_rng(0)
        n = int(min(4.0e5, area.sum() / spacing ** 2))
        k = rng.choice(len(f), n, p=area / area.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        p = (1 - s)[:, None] * tri[k, 0] + (s * (1 - r2))[:, None] * tri[k, 1] + (s * r2)[:, None] * tri[k, 2]
        _SAMPLES[name] = np.concatenate([p, v])
    return _SAMPLES[name]


class Clearance:
    """The lowest the gripper origin may go over each point of the scene: its fingertips CLEAR above the
    static geometry under them (the table, the plank, the objects, the floor off the table), and the hand
    above whatever is under it. Everything under a top surface counts as solid.

    A kinematic gripper driven into the plank or an object pins the cloth between two penalty surfaces: the
    planner flew at a fixed 30 cm (the plank's top reaches 33 cm) and laid cloth down 2 cm above the table
    wherever it was, contact forces grew to 1,000x the cloth's weight, and 26 of c1's first 16,384 episodes
    exploded (20 of them on the plank; 1% of plank starts)."""

    def __init__(self, spec, extent=0.8):
        self.lo, n = -extent, int(round(2 * extent / CELL)) + 1
        self.n = n
        hm = np.full((n, n), -0.75)                                  # the floor
        c = self.lo + CELL * np.arange(n)
        on = (np.abs(c)[:, None] <= TABLE_HALF[0] + 0.5 * CELL) & (np.abs(c)[None, :] <= TABLE_HALF[1] + 0.5 * CELL)
        hm[on] = 0.0                                                  # the table top: hm[ix, iy]
        pts = []
        if spec.ramp is not None:                                     # the plank's top face
            o, ax, lat, nrm = ramp_frame(spec.ramp)
            s, t = np.meshgrid(np.arange(0.0, RAMP["L"] + 1e-9, CELL / 2), np.arange(-RAMP["W"] / 2, RAMP["W"] / 2 + 1e-9, CELL / 2))
            pts.append(o + s.reshape(-1, 1) * ax + t.reshape(-1, 1) * lat + RAMP["T"] * nrm)
        for ob in spec.obstacles:
            a = library.load(ob.name)
            cy, sy = np.cos(ob.yaw), np.sin(ob.yaw)
            R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            pts.append(_surface_samples(ob.name) @ R.T + np.array([ob.xy[0], ob.xy[1], resting_z(a)]))
        if pts:
            p = np.concatenate(pts)
            ix, iy = self._index(p[:, :2])
            np.maximum.at(hm, (ix, iy), p[:, 2])
        from scipy.ndimage import maximum_filter
        size = lambda h: (2 * int(np.ceil(h[0] / CELL)) + 3, 2 * int(np.ceil(h[1] / CELL)) + 3)
        self.hm = hm
        self.need = np.maximum(maximum_filter(hm, size=size(FINGERS)) + TIP + CLEAR,
                               maximum_filter(hm, size=size(HAND)) - 0.015 + CLEAR)

    def _index(self, xy):
        xy = np.atleast_2d(xy)
        n = self.n
        return (np.clip(np.round((xy[:, 0] - self.lo) / CELL).astype(int), 0, n - 1),
                np.clip(np.round((xy[:, 1] - self.lo) / CELL).astype(int), 0, n - 1))

    def lowest(self, xy):
        """The lowest allowed gripper origin height over each point (N, 2)."""
        return self.need[self._index(xy)]

    def reachable(self, x, tol=0.004):
        """Cloth vertices the fingertips can come down onto from above."""
        return x[:, 2] + TIP + 0.002 + tol >= self.lowest(x[:, :2])

    def clamp(self, path, f0, holds=()):
        """Raise the path from frame f0 on wherever it would cut into the scene, smoothly: the gripper
        starts rising LEAD frames before an edge (a running maximum, then a running mean over 2 LEAD + 1
        frames, never below what each frame needs); holds: (first, last) frames the gripper holds cloth, where
        it keeps HOLD_CLEAR. Returns the number of frames raised."""
        need = self.lowest(path[:, :2])
        extra, ramp = np.zeros(len(need)), np.zeros(len(need))
        for a, b in holds:                            # the hold's extra comes in over LEAD frames after the
            k = np.arange(a + 1, min(b, len(need) - 1) + 1)          # grasp (raising the gripper before it
            extra[k] = HOLD_CLEAR - CLEAR                             # would lift the fingers off the cloth)
            ramp[k] = np.minimum(1.0, (k - a) / LEAD)
        free, held = _smooth(need), _smooth(need + extra)
        target = free + ramp * (held - free)
        low = np.flatnonzero(path[f0:, 2] < target[f0:]) + f0
        path[low, 2] = target[low]
        return len(low)


def _smooth(need):
    """A running maximum over 2 LEAD + 1 frames, then a running mean over as many: never below `need`, and
    rising LEAD frames ahead of an edge."""
    k = 2 * LEAD + 1
    pad = np.concatenate([np.full(LEAD, need[0]), need, np.full(LEAD, need[-1])])
    run = np.lib.stride_tricks.sliding_window_view(pad, k).max(1)
    pad = np.concatenate([np.full(LEAD, run[0]), run, np.full(LEAD, run[-1])])
    return np.maximum(np.lib.stride_tricks.sliding_window_view(pad, k).mean(1), need)


# ---------------------------------------------------------------------------------------------------------
# gripper paths: poses per frame for the whole timeline; families write segments of it
# ---------------------------------------------------------------------------------------------------------
def _seg_raw(path, f0, p0, p1, seconds, ease=True):
    """Move the gripper from p0 to p1 over `seconds` starting at frame f0 (smoothstep); returns the end frame.
    A segment cut off by the end of the episode stays cut off (setting the last pose to p1 made the gripper
    jump there in the last frame and yank the cloth)."""
    n = max(1, int(round(seconds * FPS)))
    for k in range(1, n + 1):
        if f0 + k >= len(path):
            return len(path) - 1
        s = k / n
        s = s * s * (3 - 2 * s) if ease else s
        path[f0 + k, :3] = (1 - s) * np.asarray(p0) + s * np.asarray(p1)
    path[f0 + n:, :3] = p1
    return f0 + n


_seg = _seg_raw


def _corners(c):
    return [0, c.nx, c.ny * (c.nx + 1), (c.ny + 1) * (c.nx + 1) - 1]


def _boundary(c):
    return [j * (c.nx + 1) + i for j in range(c.ny + 1) for i in range(c.nx + 1) if i in (0, c.nx) or j in (0, c.ny)]


def _approach(path, f, grip_pos, tip, rng, clear=None):
    """Over at a safe height (30 cm, and 2 cm over everything on the way), down to 8 cm above, then onto the
    fingertip point; returns the end frame."""
    H = max(float(grip_pos[2]), 0.3)
    if clear is not None:
        line = np.linspace(np.asarray(grip_pos, float)[:2], np.asarray(tip, float)[:2], 64)
        H = max(H, float(clear.lowest(line).max()) + 0.02)
    hi = np.array([tip[0], tip[1], H])
    above = tip + np.array([0.0, 0.0, 0.08])
    dist = float(np.linalg.norm(hi - np.asarray(grip_pos)))
    f = _seg(path, f, grip_pos, hi, max(float(rng.uniform(0.3, 0.45)), dist / 1.2))   # <= 1.2 m/s on average
    f = _seg(path, f, hi, above, max(0.15, float(np.linalg.norm(hi - above)) / 1.2))
    return _seg(path, f, above, tip, 0.15)


def _on_table(x, margin=0.04):
    """Vertices the gripper may target: over the table top (not hanging off it or on the floor)."""
    return (np.abs(x[:, 0]) < TABLE_HALF[0] - margin) & (np.abs(x[:, 1]) < TABLE_HALF[1] - margin) & (x[:, 2] > -0.005)


def _grasp_vertex(rng, c, x, kind, ok):
    """A vertex to grasp: a corner, an edge midpoint, or anywhere -- among the `ok` ones (over the table: a
    cloth that slid off sent the gripper diving to the floor at 7 m/s; reachable from above); None if none is."""
    if kind == "corner":
        cand = [v for v in _corners(c) if ok[v]]
    elif kind == "edge":
        cand = [j * (c.nx + 1) + i for i, j in [(c.nx // 2, 0), (c.nx // 2, c.ny), (0, c.ny // 2), (c.nx, c.ny // 2)]
                if ok[j * (c.nx + 1) + i]]
    else:
        cand = list(np.flatnonzero(ok))
    if not cand:
        cand = list(np.flatnonzero(ok))
    return int(rng.choice(cand)) if cand else None


class _Cut(Exception):
    """A family's path reached the end of the episode: nothing after it is written."""


def plan_family(rng, family, spec, x, grip_pos, f0, n_frames, path, held=False, clear=None, keep=False):
    """Write the gripper's path for one family starting at frame f0 from the cloth's vertices x (world) and
    the gripper's current position; held: the cloth is already in the gripper (continue from there); clear:
    the scene's Clearance (targets the fingers can reach; the caller raises the path over the scene); keep (long
    episodes): redraw a direction that would carry the cloth further off the table (KEEP_TRIES, then towards the
    table's centre) -- over a chain of actions towels otherwise walk off the table (a quarter of a 20 s pilot ended
    on the floor). Returns (events [(frame, "grasp", fingertip point) | (frame, "release")], end frame, description)."""
    ev, desc = [], {"family": family, "start": int(f0)}
    try:
        return _plan(rng, family, spec, x, grip_pos, f0, path, held, ev, desc, clear, keep)
    except _Cut:
        desc["cut_by_episode_end"] = True
        return ev, len(path) - 1, desc


def _keeper(x, size):
    """keep's tests from the vertices x (world) now: stays(shift) -- the footprint moved by shift (xy) reaches no
    further past the table (KEEP_MARGIN inside its edge) than it does now; lands(p) -- a body hanging from p (xy) and
    let down there, its footprint within size / 2 of p, does not either; inward(p): the unit vector from p to the
    table's centre."""
    box = np.array(TABLE_HALF) - KEEP_MARGIN
    lo0, hi0 = x[:, :2].min(0), x[:, :2].max(0)
    over = lambda lo_, hi_: float(np.maximum(-lo_ - box, 0.0).sum() + np.maximum(hi_ - box, 0.0).sum())
    now = over(lo0, hi0) + 1e-9
    stays = lambda shift: over(lo0 + shift, hi0 + shift) <= now
    lands = lambda p: over(np.asarray(p) - 0.5 * size, np.asarray(p) + 0.5 * size) <= max(now, 1e-9)
    inward = lambda p: -np.asarray(p, float) / max(float(np.linalg.norm(p)), 1e-6)
    return stays, lands, inward


def _plan(rng, family, spec, x, grip_pos, f0, path, held, ev, desc, clear=None, keep=False):
    c = spec.cloth
    ok = _on_table(x) & (clear.reachable(x) if clear is not None else True)
    lo, hi = np.array([-TABLE_HALF[0] + 0.05, -TABLE_HALF[1] + 0.05]), np.array([TABLE_HALF[0] - 0.05, TABLE_HALF[1] - 0.05])
    last = len(path) - 1
    tries = KEEP_TRIES if keep else 1                 # keep off: one draw each, as c1
    stays, lands, inward = _keeper(x, c.h * max(c.nx, c.ny))

    def _seg(path, f, p0, p1, seconds, ease=True):           # a segment, stopping the family at the end
        f = _seg_raw(path, f, p0, p1, seconds, ease)
        if f >= last:
            raise _Cut()
        return f

    if family == "push":                              # closed fingers into the cloth's edge, no grasp
        ctr = x.mean(0)
        bnd = [v for v in _boundary(c) if ok[v]]
        if not bnd:
            desc["skipped"] = "nothing over the table"
            return ev, f0, desc
        for _k in range(tries):                        # keep: a push that does not drive it off the table
            k = int(rng.choice(bnd))
            d = x[k, :2] - ctr[:2]
            d = d / max(np.linalg.norm(d), 1e-6)
            start = x[k, :2] + 0.04 * d
            dist = float(rng.uniform(0.05, 0.2))
            if not keep or stays(-dist * d):
                break
        else:                                          # from the edge furthest out: it pushes the cloth inwards
            k = int(bnd[int(np.argmax(np.linalg.norm(x[bnd, :2], axis=1)))])
            d = x[k, :2] - ctr[:2]
            d = d / max(np.linalg.norm(d), 1e-6)
            start = x[k, :2] + 0.04 * d
            desc["kept_on_table"] = True
        z = float(rng.uniform(0.003, 0.008)) + TIP
        f = _approach(path, f0, grip_pos, np.array([*start, z]), rng, clear)
        if f >= last:
            raise _Cut()
        end = start - dist * d
        f = _seg(path, f, np.array([*start, z]), np.array([*end, z]), dist / float(rng.uniform(0.1, 0.3)), ease=False)
        f = _seg(path, f, np.array([*end, z]), np.array([*end, z + 0.12]), 0.3)
        desc.update(point=start.tolist(), direction=(-d).tolist(), distance=dist)
        return ev, f, desc
    if held:                                          # the fingertips are where the gripper is
        tip = np.asarray(grip_pos, float)
        f = f0
        desc.update(grasp_kind="held")
    else:
        kind = "corner" if family == "fold" else str(rng.choice(["corner", "edge", "any"], p=[0.5, 0.25, 0.25]))
        v = _grasp_vertex(rng, c, x, kind, ok)
        if v is None:
            desc["skipped"] = "nothing over the table"
            return ev, f0, desc
        p = x[v].copy()
        near = np.linalg.norm(x[:, :2] - p[:2], axis=1) < 0.012
        p[2] = float(x[near, 2].max())                # the top layer under the fingers
        tip = p + np.array([0.0, 0.0, TIP + 0.002])
        f = _approach(path, f0, grip_pos, tip, rng, clear)
        if f >= last:
            raise _Cut()
        ev.append((f, "grasp", (tip - np.array([0.0, 0.0, TIP])).tolist()))
        desc.update(vertex=v, grasp_kind=kind)
    if family in ("lift_drop", "shake"):
        H = float(rng.uniform(0.15, 0.4)) if family == "lift_drop" else float(rng.uniform(0.2, 0.35))
        for _k in range(tries):                        # keep: let go where it lands on the table
            top = tip + np.array([*rng.uniform(-0.12, 0.12, 2), H])
            if not keep or lands(top[:2]):
                break
        else:
            top = tip + np.array([*(0.12 * inward(tip[:2])), H])
            desc["kept_on_table"] = True
        f = _seg(path, f, tip, top, H / float(rng.uniform(0.2, 0.5)))
        if family == "shake":
            amp, hz, dur = float(rng.uniform(0.02, 0.06)), float(rng.uniform(1.5, 3.0)), float(rng.uniform(0.6, 1.2))
            d = rng.normal(size=2)
            d = d / np.linalg.norm(d)
            n = int(dur * FPS)
            for k in range(1, n + 1):
                if f + k < len(path):
                    path[f + k, :3] = top + np.array([*(amp * np.sin(2 * np.pi * hz * k / FPS) * d), 0.0])
            if f + n >= last:
                raise _Cut()
            f = f + n
            path[f:, :3] = path[f, :3]
            desc.update(amplitude=amp, hz=hz, seconds=dur)
            top = path[f, :3].copy()
        hold = int(rng.uniform(0.2, 0.8) * FPS)
        path[f:, :3] = top
        if f + hold >= last:
            raise _Cut()
        f = f + hold
        ev.append((f, "release"))
        f = _seg(path, f, top, top + np.array([0.0, 0.0, 0.1]), 0.3)
        desc.update(height=H)
        return ev, f, desc
    if family == "lift_place":
        H = float(rng.uniform(0.08, 0.25))
        dist = float(rng.uniform(0.1, 0.3))
        for _k in range(tries):                        # keep: lay it down where it lands on the table
            d = rng.normal(size=2)
            d = d / np.linalg.norm(d)
            goal = np.clip(tip[:2] + dist * d, lo, hi)
            if not keep or lands(goal):
                break
        else:
            goal = np.clip(tip[:2] + dist * inward(tip[:2]), lo, hi)
            desc["kept_on_table"] = True
        mid = np.array([*(0.5 * (tip[:2] + goal)), tip[2] + H])
        f = _seg(path, f, tip, mid, float(rng.uniform(0.4, 0.8)))
        down = np.array([*goal, TIP + PLACE])
        f = _seg(path, f, mid, down, float(rng.uniform(0.4, 0.8)))
        ev.append((f, "release"))
        f = _seg(path, f, down, down + np.array([0.0, 0.0, 0.12]), 0.3)
        desc.update(height=H, goal=goal.tolist())
        return ev, f, desc
    if family == "slide":                             # drag fast, let go while moving: it skids to a stop
        lift = float(rng.uniform(0.004, 0.01))
        speed = float(rng.uniform(0.3, 0.6))
        dist = float(rng.uniform(0.1, 0.2))
        skid = speed ** 2 / (2.0 * G * 0.5 * (c.mu + spec.table_mu))   # it skids v^2 / 2 mu g on
        for _k in range(tries):                        # keep: a slide that stops on the table
            ang = float(rng.uniform(-np.pi, np.pi))
            u = np.array([np.cos(ang), np.sin(ang)])
            goal = np.clip(tip[:2] + dist * u, lo, hi)
            if not keep or stays(goal - tip[:2] + skid * u):
                break
        else:
            u = inward(x.mean(0)[:2])
            goal = np.clip(tip[:2] + dist * u, lo, hi)
            desc["kept_on_table"] = True
        a = tip + np.array([0.0, 0.0, lift])
        f = _seg(path, f, tip, a, 0.1)
        b = np.array([*goal, a[2]])
        f = _seg(path, f, a, b, max(np.linalg.norm(goal - tip[:2]), 0.02) / speed, ease=False)
        ev.append((f, "release"))
        f = _seg(path, f, b, b + np.array([0.0, 0.0, 0.12]), 0.25)
        desc.update(lift=lift, speed=speed, goal=goal.tolist())
        return ev, f, desc
    if family == "drag":
        lift = float(rng.uniform(0.01, 0.03))
        dist = float(rng.uniform(0.1, 0.35))
        away = tip[:2] - x.mean(0)[:2]
        for _k in range(tries):                        # keep: a drag that leaves it on the table
            ang = np.arctan2(away[1], away[0]) + rng.uniform(-1.2, 1.2)
            goal = np.clip(tip[:2] + dist * np.array([np.cos(ang), np.sin(ang)]), lo, hi)
            if not keep or stays(goal - tip[:2]):
                break
        else:
            goal = np.clip(tip[:2] + dist * inward(x.mean(0)[:2]), lo, hi)
            desc["kept_on_table"] = True
        a = tip + np.array([0.0, 0.0, lift])
        f = _seg(path, f, tip, a, 0.15)
        b = np.array([*goal, a[2]])
        f = _seg(path, f, a, b, np.linalg.norm(goal - tip[:2]) / float(rng.uniform(0.1, 0.4)), ease=False)
        ev.append((f, "release"))
        f = _seg(path, f, b, b + np.array([0.0, 0.0, 0.12]), 0.3)
        desc.update(lift=lift, goal=goal.tolist())
        return ev, f, desc
    if family == "fold":                              # onto the opposite corner, an adjacent one, or the middle
        cs = _corners(c)
        opp = {cs[0]: cs[3], cs[3]: cs[0], cs[1]: cs[2], cs[2]: cs[1]}
        adj = {cs[0]: [cs[1], cs[2]], cs[3]: [cs[1], cs[2]], cs[1]: [cs[0], cs[3]], cs[2]: [cs[0], cs[3]]}
        which = str(rng.choice(["opposite", "adjacent", "middle"], p=[0.5, 0.3, 0.2]))
        if v not in opp:                              # no corner was reachable: fold what was grasped onto the middle
            which = "middle"
        target = x[opp[v]] if which == "opposite" else (x[int(rng.choice(adj[v]))] if which == "adjacent" else x.mean(0))
        H = float(rng.uniform(0.05, 0.15))
        mid = np.array([*(0.5 * (tip[:2] + target[:2])), tip[2] + H])
        f = _seg(path, f, tip, mid, float(rng.uniform(0.4, 0.7)))
        down = np.array([*target[:2], TIP + PLACE])
        f = _seg(path, f, mid, down, float(rng.uniform(0.4, 0.7)))
        ev.append((f, "release"))
        f = _seg(path, f, down, down + np.array([0.0, 0.0, 0.12]), 0.3)
        desc.update(onto=which, height=H)
        return ev, f, desc
    raise ValueError(family)


# ---------------------------------------------------------------------------------------------------------
# a batch
# ---------------------------------------------------------------------------------------------------------
def run_batch(specs, seed=0, record=RECORD, log=print, self_contact=True, replay=None, settle=None, start=None,
              idle=None, fit=False, keep=False, own_settle=False, sub=0, record_v=False, substeps=None):
    """Simulate one batch of cloth scenes. replay: per world, a recorded episode's gripper path over the
    whole timeline (frames, 7) and its events (identifiability gate): no planning; pass the recorded
    `settle` (frames) with it. start: per world, the cloth's vertex positions and velocities to begin from
    (a window of a recorded episode: settle 0, the window's path and events in replay).
    Long episodes: record (s) any length; idle (p, lo, hi): after each family the gripper waits a further lo-hi s
    with probability p (the cloth settles); fit: a family is redrawn (PLAN_TRIES) until it completes before the
    episode ends, else the slot is skipped, each plan records its end frame, and a slot skipped for want of reachable
    cloth waits 1 s (not 0.3) before the next; keep: plan_family's keep (goals that keep the cloth on the table);
    substeps: per frame, in place of the batch's stable count (which must not be above it).
    own_settle: each world settles for its own start's time (SETTLE), held still (inactive) before that -- the batch
    settles for its slowest start, and a cloth put on a 10-40 deg plank (0.1 s) otherwise slid unrecorded for up to
    1.5 s: most of c1's plank starts begin after their slide, on the table or the floor. Off (the defaults), nothing
    extra is drawn and the episodes are c1's."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    b = newton.ModelBuilder()
    worlds, roles, infos = [], [], []
    for k, sp in enumerate(specs):
        wb, info = build_world(sp, np.random.default_rng(sp.seed))
        po, eo = b.particle_count, b.edge_count
        b.add_world(wb)
        worlds.append({"particles": (po + info["particles"][0], info["particles"][1]),
                       "edges": (eo + info["edges"][0], info["edges"][1]), "cloth": sp.cloth,
                       "gripper": b.body_count - 1})
        roles += info["roles"]
        infos.append(info)
    W = len(specs)
    if settle is None:
        settle = max(int(round(SETTLE[sp.init] * FPS)) for sp in specs)
    n_rec = int(round(record * FPS))
    n_tot = settle + n_rec
    path = np.zeros((n_tot + 1, W, 7), np.float32)
    path[:, :, :3] = PARK
    path[:, :, 6] = 1.0
    need = stable_substeps([sp.cloth for sp in specs])
    assert substeps is None or substeps >= need, (substeps, need)       # substeps=: hold the count fixed (or raise
    substeps = need if substeps is None else int(substeps)              # it) across batches; never below stable
    if sub > 1:                                           # the sub instants at equal times: whole contact refreshes
        from lsim.teacher.cloth_teacher import COLLIDE_EVERY
        substeps = int(np.ceil(substeps / (COLLIDE_EVERY * sub)) * COLLIDE_EVERY * sub)
    t = ClothTeacher(b, worlds, roles, path, substeps, self_contact=self_contact)
    log(f"[cloth] built {W} worlds, {t.model.particle_count} cloth vertices, {substeps} substeps a frame, "
        f"settle {settle} frames, in {time.time() - t0:.1f} s")
    pw = [path[:, w].copy() for w in range(W)]
    clears = [Clearance(sp) for sp in specs] if replay is None else None

    def holds(ev):                                        # (grasp frame, release frame) on the timeline
        out, g = [], None
        for e in sorted(ev, key=lambda e: e[0]):
            if e[1] == "grasp":
                g = e[0]
            elif g is not None:
                out.append((g, e[0]))
                g = None
        return out + ([(g, n_tot)] if g is not None else [])
    events = {w: [] for w in range(W)}                    # (frame, kind, ...) on the whole timeline
    plans = {w: [] for w in range(W)}
    busy_until = np.full(W, settle, int)
    remaining = [sp.n_families for sp in specs]
    if replay is not None:
        for w, r in enumerate(replay):
            pw[w][:] = r["path"]
            events[w] = [tuple(e) for e in r["events"]]
            remaining[w] = 0
    # "held": during settling, lift the cloth by a corner and keep it up
    for w, sp in enumerate(specs):
        if sp.init == "held" and replay is None:
            x = _cloth_x(t, w)
            v = int(rng.choice(_corners(sp.cloth)))
            tip = x[v] + np.array([0.0, 0.0, TIP + 0.002])
            f = _approach(pw[w], 0, np.array(PARK, float), tip, rng, clears[w])
            events[w].append((f, "grasp", (tip - np.array([0.0, 0.0, TIP])).tolist()))
            top = tip + np.array([*rng.uniform(-0.05, 0.05, 2), float(rng.uniform(0.25, 0.45))])
            _seg(pw[w], f, tip, top, float(rng.uniform(0.3, 0.45)))
            plans[w].append({"family": "held", "vertex": v, "start": 0, "raised_frames": clears[w].clamp(pw[w], 1, holds(events[w]))})
    t.path.assign(np.stack(pw, 1))
    frozen = {}                                           # own_settle: world -> the frame it is let go
    if own_settle and replay is None and start is None:
        for w, sp in enumerate(specs):
            k = settle - int(round(SETTLE[sp.init] * FPS))
            if sp.init != "held" and k > 0:               # (a held start's gripper works through the settling)
                p0, n = worlds[w]["particles"]
                t.flags[p0:p0 + n] &= ~int(newton.ParticleFlags.ACTIVE)
                frozen[w] = k
        if frozen:
            t.model.particle_flags.assign(t.flags)
    if start is not None:                                 # begin from a recorded state
        q, qd = t.s0.particle_q.numpy(), t.s0.particle_qd.numpy()
        for w, st in enumerate(start):
            p0, n = worlds[w]["particles"]
            q[p0:p0 + n], qd[p0:p0 + n] = st["x"], st["v"]
        t.s0.particle_q.assign(q)
        t.s0.particle_qd.assign(qd)
    fam_seen = {w: [] for w in range(W)}
    guarded = np.zeros(W, int)                            # frames each gripper's descent was stopped
    rec_x, rec_c, rec_g, pinned_m = [], [], [], []
    rec_sub, rec_cs, rec_gs = [], [], []
    rec_v = []
    t.sub = sub
    for f in range(n_tot + 1):
        let_go = [w for w, k in frozen.items() if k == f]
        if let_go:                                        # own_settle: this world's settling starts now
            qd = t.s0.particle_qd.numpy()
            for w in let_go:
                p0, n = worlds[w]["particles"]
                t.flags[p0:p0 + n] |= int(newton.ParticleFlags.ACTIVE)
                qd[p0:p0 + n] = 0.0
            t.model.particle_flags.assign(t.flags)
            t.s0.particle_qd.assign(qd)
            t.s1.particle_qd.assign(qd)
        # plan families when free (from the cloth's state now), recorded phase only
        if replay is None and f >= settle:
            planned = False
            for w, sp in enumerate(specs):
                if remaining[w] > 0 and f >= busy_until[w] and f < n_tot - int(1.2 * FPS):
                    held = len(t.grasped(w)) > 0
                    xw = _cloth_x(t, w)
                    for _try in range(PLAN_TRIES if fit else 1):    # fit: a family that completes before the end
                        fam = str(rng.choice(["lift_drop", "lift_place", "shake"])) if held else str(rng.choice(FAMILIES, p=FAMILY_P))
                        trial = pw[w].copy() if fit else pw[w]
                        ev, end, desc = plan_family(rng, fam, sp, xw, trial[f, :3].copy(), f, n_tot, trial, held, clears[w],
                                                    keep=keep)
                        if not fit or not desc.get("cut_by_episode_end"):
                            break
                    else:
                        ev, end, desc = [], f, {"family": fam, "start": int(f), "skipped": "no family fits the time left"}
                        trial = pw[w]
                    if fit:
                        pw[w][:] = trial
                        desc["end"] = int(end)
                    events[w] += ev
                    desc["raised_frames"] = clears[w].clamp(pw[w], f + 1, holds(events[w]))
                    plans[w].append(desc)
                    fam_seen[w].append(fam)
                    busy_until[w] = end + int((1.0 if fit and desc.get("skipped") == "nothing over the table" else 0.3) * FPS)
                    if idle is not None and rng.random() < idle[0]:   # long episodes: sometimes let it settle
                        gap = int(round(float(rng.uniform(idle[1], idle[2])) * FPS))
                        busy_until[w] += gap
                        desc["idle_after_s"] = gap / FPS
                    remaining[w] -= 1
                    planned = True
            if planned:
                t.path.assign(np.stack(pw, 1))
        for w in range(W):                                # this frame's grasps and releases
            for k, e in enumerate(events[w]):
                if e[0] == f:
                    if e[1] == "grasp":
                        if len(e) > 3 and len(e[3]):         # a replay: the recorded vertices
                            ids = t.grasp_ids(w, e[3])
                        else:                                 # where the fingertips are now (the path was
                            e = (f, "grasp", (pw[w][f, :3] - np.array([0.0, 0.0, TIP])).tolist())   # raised
                            ids = t.grasp(w, e[2])           # over the scene after planning)
                        events[w][k] = (f, "grasp", e[2], [int(i) for i in ids])
                    else:
                        t.release(w)
        if f >= settle:                                   # record the state at the start of frame f
            rec_x.append(t.s0.particle_q.numpy().copy())
            if record_v:                                  # the instantaneous velocities too (res["v"])
                rec_v.append(t.s0.particle_qd.numpy().copy())
            pinned_m.append([float(t.mass[t.grasped(w)].sum()) if t.grasped(w) else 0.0 for w in range(W)])
        if f == n_tot:
            break
        r = t.frame()
        if sub > 1 and f >= settle:                       # the record inside the frame (frame-rate study)
            rec_sub.append(np.stack(t.x_sub))
            rec_cs.append(t.contact_sub)
            rec_gs.append(t.grasp_sub)
        if replay is None:                                # a guarded move, as a robot places: when the held cloth is
            pressed = False                               # pushed up by what is under it (a pile of itself, which no
            for w in range(W):                            # clearance map knows), the gripper stops going down
                if t.grasped(w) and r["grasp"][w, 2] < -PRESS:
                    rel = [e[0] for e in events[w] if e[1] == "release" and e[0] > f]
                    end = min(min(rel) + 9 if rel else n_tot, n_tot)          # and through the lift after letting go
                    pw[w][f + 2:end + 1, 2] = np.maximum(pw[w][f + 2:end + 1, 2], pw[w][f + 1, 2] + GUARD_RISE)
                    guarded[w] += 1
                    pressed = True
            if pressed:
                t.path.assign(np.stack(pw, 1))
        if f >= settle:
            rec_c.append(r["contact"])
            rec_g.append(r["grasp"])
    log(f"[cloth] simulated {W} x {(n_tot) / FPS:.1f} s in {time.time() - t0:.1f} s")
    X = np.stack(rec_x)                                       # (F + 1, V, 3)
    grip = np.stack(pw, 1)[settle:]                           # (F + 1, W, 7)
    # the grasp force: the cloth's pull on the pinned vertices plus their own m (a - g)
    Fg = np.zeros((len(rec_x), W, 3), np.float32)
    acc = np.zeros_like(grip[:, :, :3])
    acc[1:-1] = (grip[2:, :, :3] - 2 * grip[1:-1, :, :3] + grip[:-2, :, :3]) * FPS * FPS
    for j in range(len(rec_g)):
        m = np.asarray(pinned_m[j])[:, None]
        Fg[j] = np.where(m > 0, rec_g[j] + m * (acc[j] - np.array([0.0, 0.0, -G])), 0.0)
    mass = t.mass
    res = {"specs": specs, "infos": infos, "worlds": worlds, "x": X, "grip": grip, "path": np.stack(pw, 1), "grasp_F": Fg,
           "contact_F": np.stack(rec_c + [rec_c[-1]]).astype(np.float32), "events": events, "plans": plans,
           "settle": settle, "substeps": substeps, "mass": mass, "teacher": t, "guarded": guarded}
    if record_v:
        res["v"] = np.stack(rec_v)
    if sub > 1:                                           # (F, sub - 1, V, 3); (F, sub, W, 3, 3); (F, sub, W, 3); (F + 1, W)
        res.update(sub=sub, x_sub=np.stack(rec_sub), contact_sub=np.stack(rec_cs), grasp_sub=np.stack(rec_gs),
                   pinned_m=np.asarray(pinned_m, np.float64))
    return res


def path_at(path, k):
    """A gripper path (frames + 1, W, 7) at k instants per frame, as the teacher's _gripper_pose interpolates it
    inside a frame: positions linear, rotations nlerp along the shorter arc -> (frames * k + 1, W, 7)."""
    p0, p1 = path[:-1].astype(np.float64), path[1:].astype(np.float64)
    q1 = np.where((p0[..., 3:] * p1[..., 3:]).sum(-1, keepdims=True) < 0.0, -p1[..., 3:], p1[..., 3:])
    out = []
    for j in range(k):
        a = j / k
        q = p0[..., 3:] * (1.0 - a) + q1 * a
        out.append(np.concatenate([p0[..., :3] * (1.0 - a) + p1[..., :3] * a, q / np.linalg.norm(q, axis=-1, keepdims=True)], -1))
    return np.concatenate([np.stack(out, 1).reshape(-1, *path.shape[1:]), path[-1:].astype(np.float64)], 0)


def rate_view(res, k):
    """A run_batch(sub=...) record read at k frames per teacher frame (k divides sub; 30 k fps): what save_shard
    stores, with frames, events and the settling counted in the finer frames. k = 1 is the record itself (grasp_F
    to float precision). The grasp force is each interval's mean pull of the cloth on the pinned vertices plus their
    m (a - g), a the gripper path's second difference at that rate (the path is linear inside a teacher frame, so at
    k > 1 the whole change of velocity falls on the frames that start a teacher frame)."""
    sub = res["sub"]
    assert sub % k == 0, (sub, k)
    s = sub // k
    X, S = res["x"], res["x_sub"]
    F, W = S.shape[0], len(res["specs"])
    fine = np.concatenate([np.concatenate([X[:F, None], S], 1).reshape(F * sub, *X.shape[1:]), X[F:F + 1]], 0)
    path = path_at(res["path"], k)
    grip = path[res["settle"] * k:]
    mean = lambda a: a.reshape(F * k, s, *a.shape[2:]).mean(1)
    rec_g, rec_c = mean(res["grasp_sub"]), mean(res["contact_sub"])
    fps = FPS * k
    acc = np.zeros_like(grip[:, :, :3])
    acc[1:-1] = (grip[2:, :, :3] - 2 * grip[1:-1, :, :3] + grip[:-2, :, :3]) * fps * fps
    m = np.repeat(res["pinned_m"][:F], k, 0)[..., None]                # the grasp set changes at teacher frames only
    Fg = np.zeros((F * k + 1, W, 3), np.float32)
    Fg[:-1] = np.where(m > 0, rec_g + m * (acc[:-1] - np.array([0.0, 0.0, -G])), 0.0)
    out = dict(res, x=fine[::s], grip=grip, path=path, grasp_F=Fg, settle=res["settle"] * k, fps=fps,
               contact_F=np.concatenate([rec_c, rec_c[-1:]]).astype(np.float32),
               events={w: [(e[0] * k, *e[1:]) for e in ev] for w, ev in res["events"].items()})
    for key in ("x_sub", "contact_sub", "grasp_sub", "sub"):
        out.pop(key)
    return out


def spec_from_meta(wd):
    """The scene a shard's world was generated from (the same initial state: its seed draws it)."""
    c = wd["cloth"]
    cloth = Cloth(nx=c["nx"], ny=c["ny"], h=c["h"], w=c["w"], Y=c["Y"], nu=c["nu"], B=c["B"], mu=c["mu"], r=c["r"])
    obs = [Obstacle(o["name"], tuple(o["pos"][:2]), 2.0 * float(np.arctan2(o["quat"][2], o["quat"][3])), o["mu"])
           for o in wd["obstacles"]]
    return ClothSpec(cloth, wd["init"], tuple(wd["pose"]), obs, len(wd["families"]), wd["seed"],
                     wd.get("table_mu", 0.5), wd.get("ramp"))


def _cloth_x(t, w):
    p0, n = t.worlds[w]["particles"]
    return t.s0.particle_q.numpy()[p0:p0 + n].astype(np.float64)


EXPLODED = 0.4                                     # m in one frame (12 m/s): no cloth on c1's paths reaches it


def quality(res):
    """Per world: the fastest vertex move in a frame (m/s), whether the cloth exploded (rule 3: a spatial
    bound, not isfinite), and the largest contact force by surface kind and grasp force over its weight."""
    X, out = res["x"], []
    for w, wd in enumerate(res["worlds"]):
        p0, n = wd["particles"]
        c = wd["cloth"]
        mv = np.linalg.norm(np.diff(X[:, p0:p0 + n], axis=0), axis=-1).max(1)
        wt = c.w * c.nx * c.ny * c.h * c.h * G
        cf = np.linalg.norm(res["contact_F"][:, w], axis=-1).max(0) / wt
        out.append({"exploded": bool(mv.max() > EXPLODED), "vmax": float(mv.max() * FPS),
                    "contact_over_weight": [float(v) for v in cf],
                    "grasp_over_weight": float(np.linalg.norm(res["grasp_F"][:, w], axis=-1).max() / wt)})
    return out


def save_shard(out, i, res, seed, qc=None):
    """shard_<i>.npz: cloth vertices as fp32 offsets from fp32 centroids (fp16's 0.12 mm at 25 cm hid every strain
    below 1% on 1 cm edges and put 1.4% g of noise in each one-step target), gripper poses,
    grasp and contact forces; shard_<i>.json: per world the cloth, its material, scene, actions, events."""
    import json
    import os
    X = res["x"]
    W = len(res["specs"])
    offs = np.cumsum([0] + [wd["particles"][1] for wd in res["worlds"]])
    p0s = [wd["particles"][0] for wd in res["worlds"]]
    C = np.stack([X[:, p0:p0 + n].mean(1) for p0, n in zip(p0s, offs[1:] - offs[:-1])], 1).astype(np.float32)
    D = np.concatenate([X[:, p0:p0 + n] - C[:, w:w + 1] for w, (p0, n) in enumerate(zip(p0s, offs[1:] - offs[:-1]))], 1)
    tmp = out / f"shard_{i:05d}.tmp.npz"
    np.savez_compressed(tmp, cloth_c=C, cloth_d=D.astype(np.float32), cloth_off=offs.astype(np.int64),
                        grip_q=res["grip"].astype(np.float32), grip_path=res["path"].astype(np.float32),
                        grasp_F=res["grasp_F"], contact_F=res["contact_F"],
                        **({"cloth_v": np.concatenate([res["v"][:, p0:p0 + n] for p0, n in zip(p0s, offs[1:] - offs[:-1])],
                                                      1).astype(np.float32)} if "v" in res else {}))   # record_v=True
    meta = {"seed": seed, "kind": "cloth", "fps": res.get("fps", FPS), "substeps": res["substeps"], "settle_frames": res["settle"],
            "events_on": "the whole timeline (recorded frame = frame - settle_frames); grip_path too",
            "contact_kinds": ["table", "gripper", "obstacle"], "worlds": []}
    qc = quality(res) if qc is None else qc              # (a rate_view: the 30 fps record's, its thresholds per frame)
    for w in range(len(qc)):
        qc[w]["guarded_frames"] = int(res.get("guarded", np.zeros(len(qc), int))[w])
        if "crossings_end" in res:                    # long episodes: edges through triangles in the last frame
            qc[w]["crossings_end"] = int(res["crossings_end"][w])
    for w, (sp, inf) in enumerate(zip(res["specs"], res["infos"])):
        c = sp.cloth
        meta["worlds"].append({"qc": qc[w], "cloth": {"nx": c.nx, "ny": c.ny, "h": c.h, "w": c.w, "Y": c.Y, "nu": c.nu, "B": c.B,
                                         "mu": c.mu, "r": c.r, "bending_length": c.bending_length},
                               "init": sp.init, "pose": list(sp.pose), "origin": inf["origin"], "rot": inf["rot"],
                               "table_mu": sp.table_mu, "ramp": sp.ramp,
                               "obstacles": inf["obstacles"], "families": res["plans"][w],
                               "events": [list(e) for e in res["events"][w]], "seed": sp.seed})
    (out / f"shard_{i:05d}.json").write_text(json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, out / f"shard_{i:05d}.npz")
