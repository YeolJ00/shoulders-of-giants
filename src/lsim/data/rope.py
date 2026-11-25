"""Rope episodes (beyond cloth): strings, cords and cables on the table, handled by the gripper, simulated by the rope
teacher (lsim.teacher.rope_teacher) and recorded as shards in the cloth shards' format.

A scene: one rope (20-150 cm of 1 cm segments, material sampled in physical units), 0-2 static household objects on
the table (scanned GSO objects, the cloth scenes' library and placement), the gripper parked; the table's friction
varies per episode (0.2-1.0) and some ropes start on a plank at 10-40 deg, as for cloth. The rope starts

  straight     flat on the table, straight
  curved       flat on the table along a random smooth curve (it relaxes toward straight as its stiffness and the
               table's friction allow)
  coiled       dropped as a tilted helix, or let fall end-first from upright: it piles up on the table
  overhang     straight across a table edge, 5-45% of it hanging (less than friction can hold)
  held         hanging from the gripper by an end (or anywhere), the rest lying on the table
  ramp         on a plank at 10-40 deg (ropes up to 45 cm): it slides or sticks

and is recorded for 4 s at 30 fps after an unrecorded settling phase, during which the gripper does one or two
action families, each planned from the rope's state when it starts (the cloth families where they make sense, the
same planner: lsim.data.cloth's approach, clearance map and guarded placing):

  lift_drop    grasp, lift 15-40 cm, hold, let go               lam (grasp force), bending (how it hangs), drag
  lift_place   grasp, carry 10-30 cm at 8-25 cm up, lay it down  bending, friction (where it lands, how it slides)
  drag         grasp, pull 10-35 cm along the table             friction, stretch, bending (how the tail follows)
  shake        grasp, lift, shake 2-6 cm at 1.5-3 Hz             stretch and lam (dynamic tension), bending (waves)
  push         closed fingers drive into the rope from its side  friction, bending (how far the bend spreads)
  slide        grasp, drag fast, let go while moving: it skids   friction (stopping distance v^2 / 2 mu g)
  pick_end     grasp an END, lift 20-50 cm (it peels off the table and hangs), carry 10-30 cm, lay it down
               lam (the force grows with the length lifted), bending (the peel's curve), friction

Whether each family exposes its parameters is for the rope identifiability gate to measure, not assumed.

Long episodes (sample_spec(actions=LONG_ACTIONS), run_batch(record=20, idle=LONG_IDLE, keep=True, own_settle=True)): 15-30 s
recorded, 4-8 families
chained from the state each leaves, idle gaps after some (lsim.data.cloth's long episodes). The defaults reproduce the
4 s episodes exactly: the long options draw nothing unless they are on.

Shards use the cloth shards' names (lsim.data.cloth.save_shard): `cloth_c` / `cloth_d` / `cloth_off` hold the rope
nodes (chain order), `grip_q`, `grip_path`, `grasp_F`, `contact_F` as for cloth; the JSON's `kind` is "rope" and each
world has a "rope" record (n, l, lam, EA, EI, mu, r, length, bending_length) in place of "cloth".
"""
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp
import newton

from lsim.data import library
from lsim.data.cloth import (CLEAR, KEEP_TRIES, LONG_ACTIONS, LONG_IDLE, PLACE, TIP, Clearance, _approach, _Cut,  # noqa: F401
                             _keeper, _seg_raw, Obstacle)
from lsim.data.scene import FLOOR_MAT, GRIPPER_MU, PARK, RAMP, TABLE_HALF, ramp_frame, resting_z, sample_ramp
from lsim.teacher.rope_teacher import (EA_MAX, FPS, G, ROLE_GRIPPER, ROLE_OBSTACLE, ROLE_TABLE, CrossingCounter, FineCrossings, Rope,
                                       RopeTeacher, add_rope, chord_resample, stable_substeps)

I = wp.quat_identity()
RECORD = 4.0                                      # s recorded
WORKSPACE = (0.42, 0.34)
FAMILIES = ("lift_drop", "lift_place", "drag", "shake", "push", "slide", "pick_end")
FAMILY_P = np.array([0.17, 0.15, 0.15, 0.11, 0.12, 0.13, 0.17])
INITS = ("straight", "curved", "coiled", "overhang", "held", "ramp")
INIT_P = np.array([0.18, 0.25, 0.17, 0.14, 0.14, 0.12])
SETTLE = {"straight": 0.3, "curved": 0.4, "coiled": 1.5, "overhang": 1.0, "held": 1.3, "ramp": 0.1}   # s, unrecorded
INSIDE = (TABLE_HALF[0] - 0.05, TABLE_HALF[1] - 0.05)   # rope starts on the table stay inside +-x, +-y
EXPLODED = 0.4                                   # m in one frame (12 m/s)
LENGTH = (0.2, 1.5)                              # m, uniform

# The rope prior (the user, 2026-10-30: a realistic distribution over household ropes). 70% of draws from a category,
# 30% independent log-uniform over the union range (coverage). Per category, centre values (lam kg/m, EA N, EI N m^2,
# mu, r m) and log-normal spreads; within a category one size factor z correlates them as a section scales (lam and
# EA ~ d^2, EI ~ d^4, r ~ d). Centres are order-of-magnitude estimates, not measurements -- assumptions and sources:
#   EA = E_eff x area: fibre ropes are far softer axially than their fibres because of twist and braid; we take EA
#        from catalogue break load / elongation at break (e.g. paracord 550: ~2.4 kN at ~30% -> ~8e3 N; 8 mm polyester
#        braid ~10 kN at ~20% -> ~5e4 N); cables from their copper core (E 117 GPa x 0.5-1 mm^2 -> 6e4-1e5 N);
#        rubber from E ~ 1.5 MPa x area (a 1.5 mm rubber band ~3 N; a sheathed bungee ~50-200 N).
#   EI = E_eff x I: for fibre ropes the strands slide over each other and E_eff in bending is orders of magnitude below
#        the solid value, so EI is set from the bending length c = (EI / lam g)^(1/3) one sees on a table edge (yarn
#        and string ~1-2 cm, paracord ~3-4 cm, braided rope 6-10 cm); jacketed power / charging cables are stiff in
#        bending (c ~10-20 cm: a USB cable holds a loop); a chain has no bending stiffness (EI at the floor).
#   lam from mass per length (yarn ~0.5 g/m (200 m per 100 g), twine 1.5-3 g/m, paracord ~6 g/m, 8 mm braid ~40 g/m,
#        USB cable ~20 g/m, lamp cord 30-60 g/m, a 4 mm steel chain ~0.3 kg/m).
#   mu: textiles 0.4-0.7, PVC / rubber jackets 0.6-0.9, rubber ~1, metal 0.2-0.4 (on the table, combined with its own).
#   r: the true radius, clipped to 0.5-4.5 mm: 2 r < l, or two non-adjacent 1 cm segments of a straight rope touch (a
#      12 mm rope keeps its mass and stiffness but collides as 9 mm).
# The simulated EA is capped (EA_MAX, and a wave speed sqrt(EA / lam) <= WAVE_MAX) and the CAPPED value is the slot
# label: above the caps a rope's strain under its own weight and the gripper's pull is < ~1e-4, an unobservable
# difference, which must not be a label difference. Rubber is linear here (Hookean springs), not hyperelastic.
CATEGORIES = {   # name: (weight, lam, EA, EI, mu, r)
    "yarn_thread":   (0.12, 5.0e-4, 1.5e2, 2.0e-8, 0.50, 0.0008),
    "cotton_string": (0.15, 2.0e-3, 1.0e3, 2.0e-7, 0.50, 0.0012),
    "paracord_lace": (0.17, 4.0e-3, 4.0e3, 2.0e-6, 0.60, 0.0020),
    "braided_rope":  (0.18, 4.0e-2, 4.0e4, 2.0e-4, 0.50, 0.0040),
    "power_cable":   (0.18, 2.5e-2, 8.0e4, 6.0e-4, 0.65, 0.0025),
    "rubber_bungee": (0.10, 1.0e-2, 5.0e1, 5.0e-6, 1.00, 0.0025),
    "chain_heavy":   (0.10, 1.0e-1, 1.0e5, 1.0e-7, 0.30, 0.0030),
}
SPREAD = {"z": 0.25, "lam": 0.3, "EA": 0.5, "EI": 0.6, "mu": 0.2, "r": 0.1}   # log sd: the size factor, the rest
WIDE = {"lam": (2.0e-4, 0.2), "EA": (3.0, 1.0e5), "EI": (1.0e-8, 3.0e-3), "mu": (0.1, 1.2), "rho": (300.0, 1500.0)}
P_CATEGORY = 0.7
R_CLIP = (0.0005, 0.0045)
MU_CLIP = (0.1, 1.2)
WAVE_MAX = 700.0           # m/s: EA <= lam WAVE_MAX^2 too. Strain at 3x the weight of 1.5 m hanging is then < 1e-4 for
                           # any lam (EA_MAX alone allows 4e-4 at lam 0.1), and the step stays >= 1/4,700 frame (without
                           # it a 0.2 g/m thread at 1e4 N needs ~40,000 substeps). PENDING, with EA_MAX: the user's call


def sample_rope(rng, l=0.01):
    """A rope from the prior above. Returns the Rope the teacher simulates (EA capped) and what was drawn."""
    n = int(round(rng.uniform(*LENGTH) / l))
    if rng.random() < P_CATEGORY:
        names = list(CATEGORIES)
        w = np.array([CATEGORIES[k][0] for k in names])
        cat = names[int(rng.choice(len(names), p=w / w.sum()))]
        _, lam, EA, EI, mu, r = CATEGORIES[cat]
        z = rng.normal() * SPREAD["z"]                                  # log of the diameter's ratio to the centre
        e = rng.normal(size=5)
        lam = lam * np.exp(2 * z + SPREAD["lam"] * e[0])
        EA = EA * np.exp(2 * z + SPREAD["EA"] * e[1])
        EI = EI * np.exp(4 * z + SPREAD["EI"] * e[2])
        mu = mu * np.exp(SPREAD["mu"] * e[3])
        r = r * np.exp(z + SPREAD["r"] * e[4])
    else:
        cat = "wide"
        lu = lambda k: float(np.exp(rng.uniform(np.log(WIDE[k][0]), np.log(WIDE[k][1]))))
        lam, EA, EI, mu, rho = lu("lam"), lu("EA"), lu("EI"), lu("mu"), lu("rho")
        r = np.sqrt(lam / (np.pi * rho))
    cap = min(EA_MAX, lam * WAVE_MAX ** 2)
    drawn = {"category": cat, "EA_drawn": float(EA), "capped": bool(EA > cap),
             "cap": "EA_MAX" if EA > cap and cap == EA_MAX else ("wave" if EA > cap else None), "r_drawn": float(r)}
    rope = Rope(n=n, l=l, lam=float(lam), EA=float(min(EA, cap)), EI=float(EI), mu=float(np.clip(mu, *MU_CLIP)),
                r=float(np.clip(r, *R_CLIP)))
    return rope, drawn


@dataclass
class RopeSpec:
    rope: Rope
    init: str
    nodes: np.ndarray = None                      # initial nodes (world), set by sample_spec
    obstacles: list = field(default_factory=list)
    n_families: int = 1
    seed: int = 0
    table_mu: float = 0.5
    ramp: dict = None
    start: dict = field(default_factory=dict)     # how the start was drawn
    material: dict = field(default_factory=dict)  # how the material was drawn (category, EA before the cap)


# ---------------------------------------------------------------------------------------------------------
# starts
# ---------------------------------------------------------------------------------------------------------
def _rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _fits(P, box=INSIDE):
    return bool((np.abs(P[:, 0]) < box[0]).all() and (np.abs(P[:, 1]) < box[1]).all())


def _place_flat(rng, xy, box=INSIDE):
    """A planar curve (N, 2) rotated at random and moved to a random spot where it fits; None if it cannot."""
    for _ in range(50):
        Q = (xy - xy.mean(0)) @ _rot2(rng.uniform(-np.pi, np.pi)).T
        span = box - 0.5 * np.ptp(Q, axis=0)
        if (span <= 0).any():
            continue
        Q = Q - 0.5 * (Q.max(0) + Q.min(0)) + rng.uniform(-span, span)
        if _fits(Q, box):
            return Q
    return None


def _curve(rng, rope, turning):
    """A smooth planar curve of the rope's length: heading = sum of three random sinusoids of arc length, total
    absolute turning about `turning` rad; dense, then resampled to chords of l."""
    L = rope.length
    s = np.linspace(0.0, L * 1.05, int(L * 1.05 / 0.001) + 1)
    k = rng.uniform(0.5, 3.0, 3)
    ph = rng.uniform(0, 2 * np.pi, 3)
    a = rng.normal(size=3)
    th = sum(a[i] * np.sin(2 * np.pi * k[i] * s / L + ph[i]) for i in range(3))
    tot = np.abs(np.diff(th)).sum()
    th = th * turning / max(tot, 1e-9)
    xy = np.stack([np.cumsum(np.cos(th)) * 0.001, np.cumsum(np.sin(th)) * 0.001], 1)
    P = chord_resample(np.c_[xy, np.zeros(len(xy))], rope.l, rope.n)
    return None if P is None else P[:, :2]


def _helix(rope, R, pitch, tilt, yaw):
    L = rope.length * 1.05
    turns = L / np.hypot(2 * np.pi * R, pitch)
    phi = np.linspace(0, 2 * np.pi * turns, int(L / 0.0005) + 1)
    H = np.stack([R * np.cos(phi), R * np.sin(phi), pitch * phi / (2 * np.pi)], 1)
    c, s = np.cos(tilt), np.sin(tilt)
    H = H @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]).T
    cy, sy = np.cos(yaw), np.sin(yaw)
    H = H @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]).T
    return chord_resample(H, rope.l, rope.n)


def _start(rng, rope, init, table_mu, ramp):
    """Initial nodes (n + 1, 3) for the start kind, and how it was drawn; (None, ...) if this rope cannot take it."""
    r, L, z = rope.r, rope.length, rope.r + 0.001
    if init == "straight":
        P = _place_flat(rng, np.stack([np.linspace(0, L, rope.n + 1), np.zeros(rope.n + 1)], 1))
        return (None if P is None else np.c_[P, np.full(len(P), z)]), {}
    if init == "curved":
        for turning in (rng.uniform(1.5, 6.0), 8.0, 12.0, 18.0):        # curlier until it fits on the table
            xy = _curve(rng, rope, turning)
            if xy is None:
                continue
            P = _place_flat(rng, xy)
            if P is not None and _min_gap(np.c_[P, np.zeros(len(P))]) > 2.2 * r:
                return np.c_[P, np.full(len(P), z)], {"turning": float(turning)}
        return None, {}
    if init == "coiled":
        if rng.random() < 0.5:                                           # a tilted helix dropped from 1-6 cm
            R = float(rng.uniform(0.03, 0.08))
            pitch = 2.0 * r + float(rng.uniform(0.004, 0.012))
            P = _helix(rope, R, pitch, float(rng.uniform(0.0, 0.6)), float(rng.uniform(-np.pi, np.pi)))
            how = {"kind": "helix", "R": R, "pitch": pitch}
        else:                                                            # upright, let fall end-first: it piles
            tilt = float(rng.uniform(0.02, 0.15))
            d = np.array([np.sin(tilt) * np.cos(a := rng.uniform(-np.pi, np.pi)), np.sin(tilt) * np.sin(a), np.cos(tilt)])
            P = np.linspace(0.0, L, rope.n + 1)[:, None] * d[None]
            how = {"kind": "upright", "tilt": tilt}
        if P is None:
            return None, {}
        xy = rng.uniform(-np.array(WORKSPACE) * 0.6, np.array(WORKSPACE) * 0.6)
        P[:, :2] += xy - 0.5 * (P[:, :2].max(0) + P[:, :2].min(0))
        P[:, 2] += r + 0.002 + float(rng.uniform(0.0, 0.05)) - P[:, 2].min()
        return P, how
    if init == "overhang":                                               # straight across an edge
        mu = 0.5 * (rope.mu + table_mu)
        f = float(rng.uniform(0.05, min(0.45, 0.8 * mu / (1.0 + mu))))   # hanging fraction friction holds
        ax = int(rng.integers(2))
        side = float(rng.choice([-1, 1]))
        edge = TABLE_HALF[ax]
        on = (1.0 - f) * L
        if on > 2 * TABLE_HALF[ax] - 0.1:
            return None, {}
        s = np.linspace(0.0, L, rope.n + 1)
        P = np.zeros((rope.n + 1, 3))
        P[:, ax] = side * (edge + f * L - s)
        P[:, 1 - ax] = float(rng.uniform(-TABLE_HALF[1 - ax] + 0.1, TABLE_HALF[1 - ax] - 0.1))
        P[:, 2] = z
        return P, {"hanging_fraction": f, "edge": ["x", "y"][ax], "side": side}
    if init == "held":                                                   # flat, the settling phase lifts it
        if rng.random() < 0.5:
            P = _place_flat(rng, np.stack([np.linspace(0, L, rope.n + 1), np.zeros(rope.n + 1)], 1))
            P = None if P is None else np.c_[P, np.full(len(P), z)]
        else:
            P, _ = _start(rng, rope, "curved", table_mu, ramp)
        return P, {}
    if init == "ramp":                                                   # along (or slanted across) the plank
        o, ax, lat, n = ramp_frame(ramp)
        a = float(rng.uniform(-0.5, 0.5)) if rng.random() < 0.6 else float(rng.choice([-1, 1]) * rng.uniform(0.9, 1.4))
        d = np.cos(a) * ax + np.sin(a) * lat
        c = o + 0.5 * RAMP["L"] * ax + (RAMP["T"] + r + 0.001) * n
        P = c[None] + (np.linspace(0.0, L, rope.n + 1) - 0.5 * L)[:, None] * d[None]
        u, v = (P - o) @ ax, (P - o) @ lat                              # on the plank's top face
        if (u < 0.01).any() or (u > RAMP["L"] - 0.01).any() or (np.abs(v) > 0.5 * RAMP["W"] - 0.01).any():
            return None, {}
        return P, {"angle_on_plank": a}
    raise ValueError(init)


def _min_gap(P):
    """The closest two non-adjacent nodes (a start must not overlap itself)."""
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    n = len(P)
    i, j = np.triu_indices(n, 3)
    return float(d[i, j].min()) if len(i) else np.inf


def sample_spec(rng, names, seed=0, actions=None):
    """A rope scene from the priors. actions (lo, hi): a long episode's number of families, drawn uniformly after
    every other draw (None: one or two, none or one on a plank, as r1)."""
    rope, material = sample_rope(rng)
    init = str(rng.choice(INITS, p=INIT_P))
    table_mu = float(np.exp(rng.uniform(np.log(0.2), np.log(1.0))))
    ramp, n_fam = None, int(rng.choice([1, 2], p=[0.6, 0.4]))
    if init == "ramp":
        ramp = sample_ramp(rng)
        if ramp is not None:
            ramp["angle"] = float(np.radians(rng.uniform(10.0, 40.0)))
            o, ax, lat, nn = ramp_frame(ramp)
            corners = [o + sx * RAMP["L"] * ax + sy * 0.5 * RAMP["W"] * lat for sx in (0, 1) for sy in (-1, 1)]
            if not all(abs(c[0]) < TABLE_HALF[0] - 0.03 and abs(c[1]) < TABLE_HALF[1] - 0.03 for c in corners):
                ramp = None
        if ramp is None:
            init = "straight"
        else:
            rope.n = min(rope.n, int(rng.integers(20, 46)))             # it fits on the 50 cm plank
            n_fam = int(rng.choice([0, 1]))
    P, how = _start(rng, rope, init, table_mu, ramp)
    if P is None and init == "straight":
        init = "curved"
        P, how = _start(rng, rope, init, table_mu, ramp)
    if P is None and init in ("ramp", "overhang"):
        if init == "ramp":
            n_fam = int(rng.choice([1, 2], p=[0.6, 0.4]))
        init, ramp = "curved", None
        P, how = _start(rng, rope, init, table_mu, ramp)
    if P is None:                                                        # anything fits as a pile
        init = "coiled"
        while P is None:
            P, how = _start(rng, rope, init, table_mu, ramp)
    n_obs = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
    obstacles = []
    for k in range(n_obs):                                               # clear of the rope and of each other
        name = str(rng.choice(names))
        a = library.load(name)
        rad = 0.5 * float(np.hypot(*a["extents"][:2])) + 0.03
        for _try in range(100):
            xy = np.array([rng.uniform(-WORKSPACE[0], WORKSPACE[0]), rng.uniform(-WORKSPACE[1], WORKSPACE[1])])
            if np.linalg.norm(P[:, :2] - xy, axis=1).min() < rad:
                continue
            if init == "overhang" and np.linalg.norm(P[:, :2] - xy, axis=1).min() < rad + 0.05:
                continue
            if ramp is not None and np.hypot(*(xy - np.asarray(ramp["mid"]))) < 0.35:
                continue
            if all(np.hypot(xy[0] - o.xy[0], xy[1] - o.xy[1]) > 0.18 for o in obstacles):
                obstacles.append(Obstacle(name, (float(xy[0]), float(xy[1])), float(rng.uniform(-np.pi, np.pi)),
                                          float(np.exp(rng.uniform(np.log(0.2), np.log(1.0))))))
                break
    if actions is not None:
        n_fam = int(rng.integers(actions[0], actions[1] + 1))
    return RopeSpec(rope, init, P, obstacles, n_fam, seed, table_mu, ramp, how, material)


# ---------------------------------------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------------------------------------
def build_world(spec):
    """Newton builder for one rope scene (lsim.data.cloth.build_world's table, plank, floor, objects and gripper);
    the shape roles in order of creation."""
    b = newton.ModelBuilder()
    roles = []

    def cfg(mu):
        s = newton.ModelBuilder.ShapeConfig()
        s.mu, s.ke, s.kd, s.kf = float(mu), 1.0e3, 0.0, 1.0
        return s
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), I), hx=TABLE_HALF[0], hy=TABLE_HALF[1], hz=0.05,
                    cfg=cfg(spec.table_mu)); roles.append(ROLE_TABLE)
    if spec.ramp is not None:
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
    p0, n = add_rope(b, spec.rope, spec.nodes)
    return b, {"gripper": g, "particles": (p0, n), "obstacles": obs, "roles": roles}


# ---------------------------------------------------------------------------------------------------------
# gripper paths
# ---------------------------------------------------------------------------------------------------------
def _on_table(x, margin=0.04):
    return (np.abs(x[:, 0]) < TABLE_HALF[0] - margin) & (np.abs(x[:, 1]) < TABLE_HALF[1] - margin) & (x[:, 2] > -0.005)


def _grasp_node(rng, n, ok, kind):
    """A node to grasp: an end, the middle, or anywhere -- among the `ok` ones; None if none is."""
    if kind == "end":
        cand = [v for v in (0, n) if ok[v]]
        if not cand:                                    # the end nearest an end that is
            idx = np.flatnonzero(ok)
            cand = [int(idx.min()), int(idx.max())] if len(idx) else []
    elif kind == "middle":
        idx = np.flatnonzero(ok)
        cand = [int(idx[np.argmin(np.abs(idx - n / 2))])] if len(idx) else []
    else:
        cand = list(np.flatnonzero(ok))
    return int(rng.choice(cand)) if cand else None


def plan_family(rng, family, spec, x, grip_pos, f0, path, held=False, clear=None, keep=False):
    """Write the gripper's path for one family starting at frame f0 from the rope's nodes x and the gripper's
    position; held: the rope is already in the gripper. Returns (events, end frame, description); a push also
    returns its low frames in the description (`low`: first, last, fingertip height), kept under the clearance. keep (long
    episodes): redraw a direction that would carry the rope's footprint further off the table (lsim.data.cloth's keep)."""
    ev, desc = [], {"family": family, "start": int(f0)}
    try:
        return _plan(rng, family, spec, x, grip_pos, f0, path, held, ev, desc, clear, keep)
    except _Cut:
        desc["cut_by_episode_end"] = True
        return ev, len(path) - 1, desc


def _plan(rng, family, spec, x, grip_pos, f0, path, held, ev, desc, clear, keep=False):
    rope = spec.rope
    tries = KEEP_TRIES if keep else 1                 # keep off: one draw each, as r1
    stays, _, inward = _keeper(x, 0.0)
    n = rope.n
    ok = _on_table(x) & (clear.reachable(x) if clear is not None else True)
    lo, hi = np.array([-TABLE_HALF[0] + 0.05, -TABLE_HALF[1] + 0.05]), np.array([TABLE_HALF[0] - 0.05, TABLE_HALF[1] - 0.05])
    last = len(path) - 1

    def _seg(path, f, p0, p1, seconds, ease=True):
        f = _seg_raw(path, f, p0, p1, seconds, ease)
        if f >= last:
            raise _Cut()
        return f

    if family == "push":                              # closed fingers into the rope from its side, below its axis
        lying = np.flatnonzero(ok & (x[:, 2] < 2.5 * rope.r + 0.002))   # nodes lying on the table
        for _try in range(12):
            if not len(lying):
                break
            k = int(rng.choice(lying))
            t = x[min(k + 1, n), :2] - x[max(k - 1, 0), :2]
            t = t / max(np.linalg.norm(t), 1e-9)
            d = np.array([-t[1], t[0]]) * float(rng.choice([-1, 1]))     # the push direction, across the rope
            a = rng.uniform(-0.4, 0.4)
            d = np.cos(a) * d + np.sin(a) * t
            start = x[k, :2] - (0.05 + rope.r) * d          # 5 cm off: a coil springing open moves while it comes
            dist = 0.05 + rope.r + float(rng.uniform(0.02, 0.15))   # through the rope and 2-15 cm on
            end = start + dist * d
            line = start[None] + np.linspace(-0.01, dist + 0.04, 40)[:, None] * d[None]   # and 4 cm beyond
            if clear is not None and (clear.lowest(line) > TIP + CLEAR + 1e-6).any():
                continue                              # never drive the rope into the plank or an object
            if keep and not stays((dist - 0.05 - rope.r) * d):
                continue                              # keep: nor off the table
            if np.linalg.norm(x[:, :2] - start, axis=1).min() < 0.03 + rope.r:
                continue                              # nor come down onto another strand (the fingers span 2.2 cm)
            break
        else:
            k = None
        if not len(lying) or k is None:
            desc["skipped"] = "no clear push"
            return ev, f0, desc
        z = float(rng.uniform(0.3, 0.7)) * rope.r + TIP                  # fingertips under the rope's axis
        f = _approach(path, f0, grip_pos, np.array([*start, z]), rng, clear)
        if f >= last:
            raise _Cut()
        a0 = f
        desc["clear_check"] = [int(f - 5), start.tolist(), 0.03 + rope.r]   # still clear before it comes down?
        f = _seg(path, f, np.array([*start, z]), np.array([*end, z]), dist / float(rng.uniform(0.1, 0.3)), ease=False)
        desc["low"] = [int(a0 - 4), int(f), float(z)]
        f = _seg(path, f, np.array([*end, z]), np.array([*end, z + 0.12]), 0.3)
        desc.update(node=k, point=start.tolist(), direction=d.tolist(), distance=dist)
        return ev, f, desc
    if held:
        tip = np.asarray(grip_pos, float)
        f = f0
        desc.update(grasp_kind="held")
    else:
        kind = "end" if family == "pick_end" else str(rng.choice(["end", "middle", "any"], p=[0.35, 0.2, 0.45]))
        v = _grasp_node(rng, n, ok, kind)
        if v is None:
            desc["skipped"] = "nothing over the table"
            return ev, f0, desc
        p = x[v].copy()
        near = np.linalg.norm(x[:, :2] - p[:2], axis=1) < 0.012
        p[2] = float(x[near, 2].max()) + rope.r + 0.001                  # the fingertips on the top strand
        tip = p + np.array([0.0, 0.0, TIP])
        f = _approach(path, f0, grip_pos, tip, rng, clear)
        if f >= last:
            raise _Cut()
        ev.append((f, "grasp", (tip - np.array([0.0, 0.0, TIP])).tolist()))
        desc.update(node=v, grasp_kind=kind, aim=[int(f - 5), int(v), x[v].tolist()])   # re-aimed then (run_batch)
    if family in ("lift_drop", "shake"):
        H = float(rng.uniform(0.15, 0.4)) if family == "lift_drop" else float(rng.uniform(0.2, 0.35))
        desc.update(height=H)
        for _k in range(tries):                        # keep: let go where it stays on the table
            top = tip + np.array([*rng.uniform(-0.12, 0.12, 2), H])
            if not keep or stays(top[:2] - tip[:2]):
                break
        else:
            top = tip + np.array([*(0.12 * inward(x.mean(0)[:2])), H])
            desc["kept_on_table"] = True
        f = _seg(path, f, tip, top, H / float(rng.uniform(0.2, 0.5)))
        if family == "shake":
            amp, hz, dur = float(rng.uniform(0.02, 0.06)), float(rng.uniform(1.5, 3.0)), float(rng.uniform(0.6, 1.2))
            d = rng.normal(size=3)
            d[2] *= 0.5
            d = d / np.linalg.norm(d)
            m = int(dur * FPS)
            for k in range(1, m + 1):
                if f + k < len(path):
                    path[f + k, :3] = top + amp * np.sin(2 * np.pi * hz * k / FPS) * d
            if f + m >= last:
                raise _Cut()
            f = f + m
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
    if family in ("lift_place", "pick_end"):
        if family == "lift_place":
            H, dist = float(rng.uniform(0.08, 0.25)), float(rng.uniform(0.1, 0.3))
        else:
            H, dist = float(rng.uniform(0.2, 0.5)), float(rng.uniform(0.1, 0.3))
        desc.update(height=H)
        for _k in range(tries):                        # keep: lay it down where it stays on the table
            d = rng.normal(size=2)
            d = d / np.linalg.norm(d)
            goal = np.clip(tip[:2] + dist * d, lo, hi)
            if not keep or stays(goal - tip[:2]):
                break
        else:
            goal = np.clip(tip[:2] + dist * inward(x.mean(0)[:2]), lo, hi)
            desc["kept_on_table"] = True
        if family == "pick_end":                       # straight up (it peels off), then across, then down
            up = tip + np.array([0.0, 0.0, H])
            f = _seg(path, f, tip, up, H / float(rng.uniform(0.3, 0.6)))   # 0.3-0.6 m/s: done inside 4 s
            hold = int(rng.uniform(0.1, 0.5) * FPS)
            path[f:, :3] = up
            if f + hold >= last:
                raise _Cut()
            f += hold
            mid = np.array([*goal, up[2]])
            f = _seg(path, f, up, mid, float(rng.uniform(0.4, 0.9)))
        else:
            mid = np.array([*(0.5 * (tip[:2] + goal)), tip[2] + H])
            f = _seg(path, f, tip, mid, float(rng.uniform(0.4, 0.8)))
        down = np.array([*goal, TIP + PLACE])
        f = _seg(path, f, mid, down, float(rng.uniform(0.4, 0.8)))
        ev.append((f, "release"))
        f = _seg(path, f, down, down + np.array([0.0, 0.0, 0.12]), 0.3)
        desc.update(height=H, goal=goal.tolist())
        return ev, f, desc
    if family == "slide":
        lift = float(rng.uniform(0.004, 0.01))
        desc.update(lift=lift)
        speed = float(rng.uniform(0.3, 0.6))
        dist = float(rng.uniform(0.1, 0.2))
        skid = speed ** 2 / (2.0 * G * 0.5 * (rope.mu + spec.table_mu))   # it skids v^2 / 2 mu g on
        for _k in range(tries):                        # keep: a slide that stops on the table
            ang = float(rng.uniform(-np.pi, np.pi))
            u = np.array([np.cos(ang), np.sin(ang)])
            goal = np.clip(tip[:2] + dist * u, lo, hi)
            if not keep or stays(goal - tip[:2] + skid * u):
                break
        else:
            goal = np.clip(tip[:2] + dist * inward(x.mean(0)[:2]), lo, hi)
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
        desc.update(lift=lift)
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
        f = _seg(path, f, a, b, max(np.linalg.norm(goal - tip[:2]), 0.02) / float(rng.uniform(0.1, 0.4)), ease=False)
        ev.append((f, "release"))
        f = _seg(path, f, b, b + np.array([0.0, 0.0, 0.12]), 0.3)
        desc.update(lift=lift, goal=goal.tolist())
        return ev, f, desc
    raise ValueError(family)


def _holds(ev, n_tot):
    out, g = [], None
    for e in sorted(ev, key=lambda e: e[0]):
        if e[1] == "grasp":
            g = e[0]
        elif g is not None:
            out.append((g, e[0]))
            g = None
    return out + ([(g, n_tot)] if g is not None else [])


def _keep_low(path, clear, low):
    """A push's travel under the clearance map's 5 mm (fingertips under the rope's axis), wherever only the table
    top is under the fingers: the map keeps them clear of the plank and the objects."""
    a, b, z = low
    a, b = max(int(a), 0), min(int(b), len(path) - 1)
    if (clear.lowest(path[a:b + 1, :2]) > TIP + CLEAR + 1e-6).any():   # all or nothing: no jump at an edge
        return 0
    path[a:b + 1, 2] = np.minimum(path[a:b + 1, 2], z)
    return int(b + 1 - a)


# ---------------------------------------------------------------------------------------------------------
# a batch
# ---------------------------------------------------------------------------------------------------------
PLAN_TRIES = 6                                    # family draws per planning slot before giving the slot up
AIM_SHIFT = 0.003                                 # m: re-aim the gripper when its node moved more than this
AIM_MAX = 0.08                                    # m: ... and less than this (a 35 cm re-aim would be a lunge)
PRESS = 0.01                                      # N: held rope pushed up this hard by what is under it ...
GUARD_RISE = 0.001                                # m per frame: ... stops the gripper's descent and lifts it
FREE_RISE = 0.006                                 # m per frame: free fingers pressing the rope (7 frames at 1 mm)


def run_batch(specs, seed=0, record=RECORD, log=print, self_contact=True, drag=True, on_chunk=None, fine_crossings=True,
              idle=None, keep=False, own_settle=False):
    """Simulate one batch of rope scenes (one rope per world). on_chunk(teacher): diagnostics after every contact
    refresh; fine_crossings: count self-crossings at that resolution over the recorded frames (FineCrossings).
    Long episodes: record (s) any length; idle (p, lo, hi): after each family the gripper waits a further lo-hi s with
    probability p (None: nothing drawn, r1's episodes); keep: plan_family's keep, and a slot skipped (nothing reachable)
    waits 1 s before the next, not 0.3; own_settle: each world settles for its own start's time (SETTLE), held still
    (inactive) before that -- the batch settles for its slowest start, and a rope on a plank (0.1 s) otherwise slid
    unrecorded for up to 1.5 s (lsim.data.cloth). Off (the defaults): r1's episodes."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    b = newton.ModelBuilder()
    worlds, roles, infos = [], [], []
    for sp in specs:
        wb, info = build_world(sp)
        po = b.particle_count
        b.add_world(wb)
        worlds.append({"particles": (po + info["particles"][0], info["particles"][1]), "rope": sp.rope,
                       "gripper": b.body_count - 1})
        roles += info["roles"]
        infos.append(info)
    W = len(specs)
    settle = max(int(round(SETTLE[sp.init] * FPS)) for sp in specs)
    n_rec = int(round(record * FPS))
    n_tot = settle + n_rec
    path = np.zeros((n_tot + 1, W, 7), np.float32)
    path[:, :, :3] = PARK
    path[:, :, 6] = 1.0
    substeps = stable_substeps([sp.rope for sp in specs])
    t = RopeTeacher(b, worlds, roles, path, substeps, self_contact=self_contact, drag=drag)
    fine = FineCrossings(worlds) if fine_crossings else None
    if fine is not None and on_chunk is not None:
        t.on_chunk = lambda tt: (on_chunk(tt), fine(tt) if tt.frame_index >= settle else None)
    elif fine is not None:
        t.on_chunk = lambda tt: fine(tt) if tt.frame_index >= settle else None
    else:
        t.on_chunk = on_chunk
    log(f"[rope] built {W} worlds, {t.model.particle_count} rope nodes, {substeps} substeps a frame, settle {settle} "
        f"frames, in {time.time() - t0:.1f} s")
    pw = [path[:, w].copy() for w in range(W)]
    clears = [Clearance(sp) for sp in specs]
    events = {w: [] for w in range(W)}
    plans = {w: [] for w in range(W)}
    lows = {w: [] for w in range(W)}
    busy_until = np.full(W, settle, int)
    remaining = [sp.n_families for sp in specs]
    for w, sp in enumerate(specs):                    # "held": during settling, lift it by an end (or anywhere)
        if sp.init == "held":
            x = _rope_x(t, w)
            kind = "end" if rng.random() < 0.7 else "any"
            v = _grasp_node(rng, sp.rope.n, _on_table(x) & clears[w].reachable(x), kind)
            if v is None:
                continue
            tip = x[v] + np.array([0.0, 0.0, TIP + sp.rope.r + 0.001])
            f = _approach(pw[w], 0, np.array(PARK, float), tip, rng, clears[w])
            events[w].append((f, "grasp", (tip - np.array([0.0, 0.0, TIP])).tolist()))
            top = tip + np.array([*rng.uniform(-0.05, 0.05, 2), float(rng.uniform(0.25, 0.6))])
            _seg_raw(pw[w], f, tip, top, float(rng.uniform(0.3, 0.45)))
            plans[w].append({"family": "held", "node": v, "start": 0,
                             "raised_frames": clears[w].clamp(pw[w], 1, _holds(events[w], n_tot))})
    t.path.assign(np.stack(pw, 1))
    frozen = {}                                       # own_settle: world -> the frame it is let go
    if own_settle:
        for w, sp in enumerate(specs):
            k = settle - int(round(SETTLE[sp.init] * FPS))
            if sp.init != "held" and k > 0:
                p0, n = worlds[w]["particles"]
                t.flags[p0:p0 + n] &= ~int(newton.ParticleFlags.ACTIVE)
                frozen[w] = k
        if frozen:
            t.model.particle_flags.assign(t.flags)
    rec_x, rec_c, rec_g, pinned_m = [], [], [], []
    guarded = np.zeros(W, int)
    weights = np.array([sp.rope.lam * sp.rope.length * G for sp in specs])
    for f in range(n_tot + 1):
        let_go = [w for w, k in frozen.items() if k == f]
        if let_go:                                    # own_settle: this world's settling starts now
            qd = t.s0.particle_qd.numpy()
            for w in let_go:
                p0, n = worlds[w]["particles"]
                t.flags[p0:p0 + n] |= int(newton.ParticleFlags.ACTIVE)
                qd[p0:p0 + n] = 0.0
            t.model.particle_flags.assign(t.flags)
            t.s0.particle_qd.assign(qd)
            t.s1.particle_qd.assign(qd)
        if f >= settle:
            planned = False
            for w, sp in enumerate(specs):
                if remaining[w] > 0 and f >= busy_until[w] and f < n_tot - int(1.2 * FPS):
                    held = len(t.grasped(w)) > 0
                    xw = _rope_x(t, w)
                    for _try in range(PLAN_TRIES):         # a family that completes before the episode ends: a
                        fam = str(rng.choice(["lift_drop", "lift_place", "shake"])) if held else str(rng.choice(FAMILIES, p=FAMILY_P))
                        trial = pw[w].copy()                # grasp cut off by the end never releases (pick-by-end
                        ev, end, desc = plan_family(rng, fam, sp, xw, trial[f, :3].copy(), f, trial, held, clears[w], keep=keep)
                        if not desc.get("cut_by_episode_end"):    # completed 49% of the time)
                            break
                    else:
                        ev, end, desc = [], f, {"family": fam, "start": int(f), "skipped": "no family fits the time left"}
                        trial = pw[w]
                    pw[w][:] = trial
                    desc["end"] = int(end)
                    events[w] += ev
                    desc["raised_frames"] = clears[w].clamp(pw[w], f + 1, _holds(events[w], n_tot))
                    if "low" in desc:
                        desc["low_frames"] = _keep_low(pw[w], clears[w], desc["low"])
                    plans[w].append(desc)
                    busy_until[w] = end + int((1.0 if keep and desc.get("skipped") in ("nothing over the table", "no clear push")
                                               else 0.3) * FPS)
                    if idle is not None and rng.random() < idle[0]:   # long episodes: sometimes let it settle
                        gap = int(round(float(rng.uniform(idle[1], idle[2])) * FPS))
                        busy_until[w] += gap
                        desc["idle_after_s"] = gap / FPS
                    remaining[w] -= 1
                    planned = True
            if planned:
                t.path.assign(np.stack(pw, 1))
        aimed = False                                     # re-aim: 5 frames before a grasp the gripper (8 cm above
        for w in range(W):                                # the rope) moves its remaining path by what its node moved
            for pl in plans[w]:                           # since planning (a coil springing open left 3% of grasps
                a = pl.get("aim")                         # closing on nothing)
                if a and a[0] == f and "end" in pl:
                    now = _rope_x(t, w)[a[1]]
                    dxy = now[:2] - np.asarray(a[2][:2])
                    if AIM_SHIFT < np.linalg.norm(dxy) < AIM_MAX:   # moved more: the rope left; the grasp may miss
                        k = np.arange(f + 1, min(pl["end"], n_tot) + 1)
                        pw[w][k, :2] += np.minimum(1.0, (k - f) / 3.0)[:, None] * dxy   # over 3 frames, not a jump
                        clears[w].clamp(pw[w], f + 1, _holds(events[w], n_tot))
                        pl["aim_shift_mm"] = float(1e3 * np.linalg.norm(dxy))
                        aimed = True
            for pl in plans[w]:                           # a push comes down only where no rope lies now (a rope still
                c = pl.get("clear_check")                 # settling moved under a planned start: 342 x its weight)
                if c and c[0] == f and "end" in pl:
                    if np.linalg.norm(_rope_x(t, w)[:, :2] - np.asarray(c[1]), axis=1).min() < c[2]:
                        k = np.arange(f + 1, min(pl["end"], n_tot) + 1)
                        pw[w][k, 2] = np.maximum(pw[w][k, 2], pw[w][f, 2])   # it stays at its hover height
                        pl["aborted"] = "rope under the start"
                        aimed = True
        if aimed:
            t.path.assign(np.stack(pw, 1))
        for w in range(W):
            for k, e in enumerate(events[w]):
                if e[0] == f:
                    if e[1] == "grasp":
                        e = (f, "grasp", (pw[w][f, :3] - np.array([0.0, 0.0, TIP])).tolist())   # where the fingertips are
                        ids = t.grasp(w, e[2], below=0.012)
                        events[w][k] = (f, "grasp", e[2], [int(i) for i in ids])
                    else:
                        t.release(w)
        if f >= settle:
            rec_x.append(t.s0.particle_q.numpy().copy())
            pinned_m.append([float(t.mass[t.grasped(w)].sum()) if t.grasped(w) else 0.0 for w in range(W)])
        if f == n_tot:
            break
        r = t.frame()
        pressed = False                               # guarded moves: held rope pushed up stops the descent, and
        for w in range(W):                            # free fingers pressing the rope down harder than its weight
            if t.grasped(w) and r["grasp"][w, 2] < -PRESS:   # stop where they are and rise (a coil springing open
                rel = [e[0] for e in events[w] if e[1] == "release" and e[0] > f]   # under a push pinched a cable
                end = min(min(rel) + 9 if rel else n_tot, n_tot)                   # against the table: 580 x its
                pw[w][f + 2:end + 1, 2] = np.maximum(pw[w][f + 2:end + 1, 2], pw[w][f + 1, 2] + GUARD_RISE)   # weight)
                guarded[w] += 1
                pressed = True
            elif not t.grasped(w) and r["contact"][w, ROLE_GRIPPER, 2] < -weights[w]:
                end = min(max(int(busy_until[w]), f + 2), n_tot)
                pw[w][f + 2:end + 1, 2] = np.maximum(pw[w][f + 2:end + 1, 2], pw[w][f + 1, 2] + FREE_RISE)
                guarded[w] += 1
                pressed = True
        if pressed:
            t.path.assign(np.stack(pw, 1))
        if f >= settle:
            rec_c.append(r["contact"])
            rec_g.append(r["grasp"])
    wall = time.time() - t0
    log(f"[rope] simulated {W} x {n_tot / FPS:.1f} s in {wall:.1f} s")
    X = np.stack(rec_x)
    grip = np.stack(pw, 1)[settle:]
    Fg = np.zeros((len(rec_x), W, 3), np.float32)
    acc = np.zeros_like(grip[:, :, :3])
    acc[1:-1] = (grip[2:, :, :3] - 2 * grip[1:-1, :, :3] + grip[:-2, :, :3]) * FPS * FPS
    for j in range(len(rec_g)):
        m = np.asarray(pinned_m[j])[:, None]
        Fg[j] = np.where(m > 0, rec_g[j] + m * (acc[j] - np.array([0.0, 0.0, -G])), 0.0)
    return {"specs": specs, "infos": infos, "worlds": worlds, "x": X, "grip": grip, "path": np.stack(pw, 1), "grasp_F": Fg,
            "contact_F": np.stack(rec_c + [rec_c[-1]]).astype(np.float32), "events": events, "plans": plans,
            "settle": settle, "substeps": substeps, "mass": t.mass, "teacher": t, "guarded": guarded, "wall_s": wall,
            "candidate_load": t.candidate_load(), "fine": fine.finish() if fine is not None else None}


def _rope_x(t, w):
    p0, n = t.worlds[w]["particles"]
    return t.s0.particle_q.numpy()[p0:p0 + n].astype(np.float64)


def quality(res, crossings=True):
    """Per world: the fastest node move in a frame (m/s) and whether the rope exploded (rule 3: a spatial bound), the
    largest contact force by surface kind and grasp force over the rope's weight, self-crossings over the recorded
    frames (the gate's counter), how fast the rope moves at the end and whether it is then held or the gripper busy."""
    X, out = res["x"], []
    for w, wd in enumerate(res["worlds"]):
        p0, n = wd["particles"]
        rope = wd["rope"]
        Xw = X[:, p0:p0 + n].astype(np.float64)
        mv = np.linalg.norm(np.diff(Xw, axis=0), axis=-1).max(1)
        wt = rope.lam * rope.length * G
        cf = np.linalg.norm(res["contact_F"][:, w], axis=-1).max(0) / wt
        q = {"exploded": bool(mv.max() > EXPLODED or not np.isfinite(Xw).all()), "vmax": float(mv.max() * FPS),
             "contact_over_weight": [float(v) for v in cf],
             "grasp_over_weight": float(np.linalg.norm(res["grasp_F"][:, w], axis=-1).max() / wt),
             "end_speed": float(np.linalg.norm(Xw[-1] - Xw[-2], axis=-1).max() * FPS),
             "lowest_z": float(Xw[:, :, 2].min()),
             "on_table_at_end": bool((Xw[-1, :, 2] > -0.003).all()),
             "speed_last_half_s": [float(v) for v in np.linalg.norm(np.diff(Xw[-16:], axis=0), axis=-1).max(1) * FPS],
             "strain_max": float(np.abs(np.linalg.norm(np.diff(Xw, axis=1), axis=-1) / rope.l - 1.0).max())}
        ev = res["events"][w]
        last_ev = max([e[0] for e in ev], default=-1) - res["settle"]
        grasps = sorted(e[0] for e in ev if e[1] == "grasp")
        rels = sorted(e[0] for e in ev if e[1] == "release")
        q["held_at_end"] = bool(grasps and (not rels or rels[-1] < grasps[-1]))
        busy = [p.get("start", 0) for p in res["plans"][w]]
        ends = [p["end"] - res["settle"] for p in res["plans"][w] if "end" in p and "skipped" not in p]
        q["last_event_frame"] = int(max([last_ev] + ends))           # the last grasp, release or family end
        q["families_started"] = len(busy)
        if res.get("fine") is not None:
            q["crossings_fine"] = int(res["fine"].n[w])
            q["min_centerline_gap_fine_mm"] = 1e3 * float(res["fine"].dmin[w])
        if crossings:
            cc = CrossingCounter([n])
            for j in range(len(Xw)):
                cc.add(Xw[j])
            q["crossings"] = int(cc.n)
            q["min_centerline_gap_mm"] = 1e3 * float(cc.dmin)
        out.append(q)
    return out


def save_shard(out, i, res, seed, qc=None):
    """shard_<i>.npz in the cloth shards' layout (lsim.data.cloth.save_shard): rope nodes as fp32 offsets from fp32
    centroids under cloth_c / cloth_d / cloth_off, gripper poses, grasp and contact forces; shard_<i>.json: per world
    the rope, its material, scene, start, actions, events and quality record."""
    import json
    import os
    X = res["x"]
    offs = np.cumsum([0] + [wd["particles"][1] for wd in res["worlds"]])
    p0s = [wd["particles"][0] for wd in res["worlds"]]
    ns = offs[1:] - offs[:-1]
    C = np.stack([X[:, p0:p0 + n].mean(1) for p0, n in zip(p0s, ns)], 1).astype(np.float32)
    D = np.concatenate([X[:, p0:p0 + n] - C[:, w:w + 1] for w, (p0, n) in enumerate(zip(p0s, ns))], 1)
    tmp = out / f"shard_{i:05d}.tmp.npz"
    np.savez_compressed(tmp, cloth_c=C, cloth_d=D.astype(np.float32), cloth_off=offs.astype(np.int64),
                        grip_q=res["grip"].astype(np.float32), grip_path=res["path"].astype(np.float32),
                        grasp_F=res["grasp_F"], contact_F=res["contact_F"])
    meta = {"seed": seed, "kind": "rope", "fps": FPS, "substeps": res["substeps"], "settle_frames": res["settle"],
            "events_on": "the whole timeline (recorded frame = frame - settle_frames); grip_path too",
            "contact_kinds": ["table", "gripper", "obstacle"], "nodes": "cloth_c / cloth_d / cloth_off hold the rope nodes",
            "wall_s": res.get("wall_s"), "worlds": []}
    qc = quality(res) if qc is None else qc
    for w in range(len(qc)):
        qc[w]["guarded_frames"] = int(res.get("guarded", np.zeros(len(qc), int))[w])
    for w, (sp, inf) in enumerate(zip(res["specs"], res["infos"])):
        r = sp.rope
        meta["worlds"].append({"qc": qc[w], "rope": {"n_nodes": r.n + 1, "n": r.n, "l": r.l, "lam": r.lam, "EA": r.EA, "EI": r.EI,
                                                     "mu": r.mu, "r": r.r, "length": r.length,
                                                     "bending_length": r.bending_length},
                               "material": sp.material, "init": sp.init, "start": sp.start, "table_mu": sp.table_mu, "ramp": sp.ramp,
                               "obstacles": inf["obstacles"], "families": res["plans"][w],
                               "events": [list(e) for e in res["events"][w]], "seed": sp.seed})
    (out / f"shard_{i:05d}.json").write_text(json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, out / f"shard_{i:05d}.npz")
