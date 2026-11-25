"""Soft-body episodes (phase 3+, "Beyond cloth"): squishy household objects -- sponges, foams, plush toys, rubber and
silicone toys, gel stress toys, bread -- on the table, handled by the gripper, simulated by the soft-body teacher
(lsim.teacher.soft_teacher) and recorded as shards in the cloth shards' format (`save_shard`, `load_soft_shard`).

A scene: one soft body (a box, cylinder, ellipsoid or a rounded plush-like blob, 3-15 cm, meshed as linear tets with
>= 8 cells through its thinnest section, <= 1,500 vertices), 0-2 static scanned objects (GSO, collided as their meshes),
the table's friction 0.2-1.0, sometimes a plank at 10-35 deg. The body starts

  rest       on the table (a random face down; cylinders upright or lying), settled 0.5 s unrecorded
  dropped    tossed: released 3-20 cm up near the middle in a random orientation, <= 0.15 m/s and 1.5 rad/s (recorded)
  ramp       at rest on a plank that runs down towards the table's middle, released at the first recorded frame: it
             slides, sticks, rolls or tips
  on_object  released 0.5-2 cm over a scanned object: it lands on it and settles there or slides off
  against    at rest on the table touching a scanned object's side

and the gripper does 0-2 families, each planned from the body's state when it starts:

  lift_drop  pinch-grasp, lift 8-30 cm, hold, let go              mass (grasp force), bounce (E, damping), drop shape
  lift_place grasp, carry 8-25 cm, set down (guarded), let go     sag while held (E, rho), friction on landing
  drag       grasp, pull 10-30 cm along the table, let go          friction (shear of the dragged body), stiffness
  slide      grasp, throw sideways at 0.3-0.8 m/s, let go moving   friction (stopping distance), rolling
  push       closed fingers drive into its side                     friction, stiffness (indentation), tipping
  squeeze    pads close to 15-60% of its width (force-capped), hold, open:   E, nu (bulging), damping (recovery)
  press      one closed finger pokes / presses its top 10-50% deep (force-capped), holds, lifts:   E, damping

A grasp closes the pads on the body (force-guarded) and pins the vertices touching them; squeezing, pushing and
pressing go through contact only. Every motion into the body is force-guarded (a kinematic finger driven into a stiff
body over a fixed path pinches it: the cloth lesson): when the pads' force passes the move's cap the motion stops there
and the rest of the family is shifted by what was not done.

Material prior (the user, 2026-10-30: a realistic distribution over real household squishy objects, not independent
ranges). 70% of bodies are drawn from a CATEGORY, each a log-normal (density, E, damping) and a range (nu, mu) around
literature values, with its own shapes and sizes; 30% are independent wide draws over the union range for coverage:

  category        rho kg/m^3   E (Pa)            nu          mu        zeta       shapes             sources / assumptions
  sponge          30 (x1.4)    8e3 (x2.2)        0.0-0.2     0.6-1.0   0.15-0.40  box (60% slabs), cyl   open-cell PU / cellulose sponge
  memory_foam     60 (x1.25)   2.5e3 (x1.8)      0.1-0.3     0.6-1.0   0.35-0.80  box, cyl, blob     viscoelastic PU (pillows, toys)
  closed_foam     80 (x1.6)    2e5 (x2.2)        0.1-0.3     0.4-0.8   0.04-0.10  box, cyl, ellipsoid   EVA / PE foam (mats, blocks, balls): 0.1-10 MPa
  plush           70 (x1.5)    2e3 (x2)          0.1-0.3     0.6-1.0   0.25-0.60  blob (lobed), ellipsoid   fibre-filled toy, ~0.1-0.3 kg in 1-3 L
  rubber          1100 (x1.08) 1e6 (x2)          0.45 (cap)  0.8-1.2   0.02-0.06  ellipsoid, cyl, blob, box   natural rubber 1-3 MPa, silicone 0.5-5 MPa (Shore 20-60A), nu 0.47-0.499
  gel             1100 (x1.08) 2e4 (x2)          0.45 (cap)  0.8-1.2   0.15-0.40  ellipsoid, blob    TPR / silicone gel stress toys, nearly incompressible, tacky
  bread           220 (x1.25)  2e4 (x2)          0.2-0.3     0.5-0.9   0.15-0.35  box, ellipsoid, cyl   bread crumb 10-60 kPa (Liu & Scanlon 2003), 0.2-0.3 g/cm^3; elastic only
  wide            20-1200      1e3-3e6           0.0-0.45    0.1-1.2   0.02-0.60  any                log-uniform (nu, mu, zeta uniform / log-uniform)

(x s: a log-normal of median m and geometric standard deviation s.) The foams' E is the modulus that matters at the
deformations a hand makes (20-50%), not the small-strain one: a Neo-Hookean tet has no foam plateau, so it is given the
secant modulus. Flexible PU foam is linear only to ~5% strain, then buckles at a plateau stress sigma_pl ~ 0.05 E_s
(rho / rho_s)^2 (Gibson & Ashby, Cellular Solids, 2nd ed. 1997, ch. 5), and its rated compression -- the force
at 40% (ASTM D3574 / ISO 3386 CLD40) -- is 1-6 kPa for comfort and sponge foams, 0.5-2 kPa for memory foam, so
the secant modulus at 40% is ~2.5-15 kPa and ~1-5 kPa (median 8 and 2.5 kPa here; 1e3-5e4 Pa over the sponge
prior's 95%). Plush stuffing (polyester fibrefill) has no standard number: its 2 kPa is an assumption chosen to
give a toy that squashes visibly in the hand. Gel stress balls: 10-50 kPa (an assumption from Shore 00-10 gels). What the teacher cannot simulate affordably is
CAPPED in the simulated body and the capped value is the label (a difference the motion cannot show must not be a
label difference): nu at 0.45 (linear tets lock volumetrically towards 0.5 and the gate checked Poisson only to 0.45),
E so that the P-wave speed stays <= 100 m/s and the body's explicit step needs <= 2,400 substeps a frame (0.6 h_alt / c_p:
small stiff balls -- the O-grid's smallest altitude is 6% of the radius -- needed up to 10,900, and one such batch took
536 s for 64 episodes). zeta is the first-mode damping ratio the body
is given (eta = 2 zeta mu / omega_1, omega_1 = pi c_s / L; eta in Pa s is recorded beside it); the teacher's implicit
edge dashpots make any damping affordable (the gate's explicit Kelvin-Voigt needed a cap that hit 76% of draws and
left every foam bouncing like rubber, e ~ 0.85-0.9). Restitution measured on 8 cm balls dropped 20 cm: rubber 0.88-0.90,
closed-cell foam 0.85, sponge foam 0.81, plush and memory foam 0.63 (scripts/l0_soft_teacher_check.py bounce). Bodies that would sag more than 15% under their own weight
(rho g L / E) are redrawn (a category tail, not a material a household holds), and so are bodies whose shear wave
speed is below 3 m/s (a 2-4 m/s impact -- a drop, a roll off the table -- inverted their tets in the pilot: ultra-soft
real foam densifies, which a Neo-Hookean tet does not). `material_report` counts the caps.

Which parameter each family exposes is the PHASE1_AMENDMENT principle applied to soft bodies; whether it really does is
for an identifiability gate to measure, not assumed.
"""
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import warp as wp
import newton

from lsim.data import library
from lsim.data.scene import FLOOR_MAT, GRIPPER_MU, PARK, RAMP, TABLE_HALF, ramp_frame, resting_z, sample_ramp
from lsim.teacher import soft_mesh
from lsim.teacher import soft_teacher as st
from lsim.teacher.soft_teacher import (FINGER_HALF, FPS, G, ROLE_GRIPPER, ROLE_OBSTACLE, ROLE_TABLE, SLOT_OBJ, SLOT_PLANK,
                                       SLOT_TABLE, W_MAX, Soft, SoftTeacher, add_gripper, add_soft, substeps_for)

I = wp.quat_identity()
RECORD = 4.0                                      # s recorded
WORKSPACE = (0.40, 0.32)
FAMILIES = ("lift_drop", "lift_place", "drag", "slide", "push", "squeeze", "press")
FAMILY_P = np.array([0.15, 0.13, 0.12, 0.12, 0.14, 0.18, 0.16])
INITS = ("rest", "dropped", "ramp", "on_object", "against")
INIT_P = np.array([0.42, 0.22, 0.12, 0.10, 0.14])
SETTLE = {"rest": 0.5, "against": 0.5, "dropped": 0.0, "ramp": 0.0, "on_object": 0.0}   # s, unrecorded
CLEAR = 0.004                                     # m: fingertips above any static surface
W_CLOSED = -2 * FINGER_HALF[0]                    # "closed" (push, press): both pads in one place, one 1.2 cm finger.
                                                  # Two pads touching at a seam trapped pressed foam between their
                                                  # inner faces and lifted a 30 g sponge into the air (pilot)
MAX_VERTICES = 3000                               # (1,500 allowed no slab: 8 cells through 3 cm)
NU_CAP, CP_CAP, SAG_MAX, CS_MIN = 0.45, 100.0, 0.08, 2.0
MAX_SUBSTEPS = 2400                               # per frame: E is capped so that no body needs more
START = (0.0, -0.42, 0.30)                        # the gripper starts here, over the table's near edge

# category: weight, (rho median, gsd), (E median, gsd), nu range, mu range, zeta range, shapes {kind: p}, size range (m)
CATEGORIES = {
    "sponge": (0.17, (30, 1.4), (8e3, 2.2), (0.0, 0.2), (0.6, 1.0), (0.15, 0.40), {"box": 0.8, "cylinder": 0.2}, (0.06, 0.15)),
    "memory_foam": (0.12, (60, 1.25), (2.5e3, 1.8), (0.1, 0.3), (0.6, 1.0), (0.35, 0.80), {"box": 0.5, "cylinder": 0.3, "blob": 0.2}, (0.05, 0.15)),
    "closed_foam": (0.12, (80, 1.6), (2e5, 2.2), (0.1, 0.3), (0.4, 0.8), (0.04, 0.10), {"box": 0.5, "cylinder": 0.25, "ellipsoid": 0.25}, (0.04, 0.15)),
    "plush": (0.18, (70, 1.5), (2e3, 2.0), (0.1, 0.3), (0.6, 1.0), (0.25, 0.60), {"blob": 0.75, "ellipsoid": 0.25}, (0.06, 0.15)),
    "rubber": (0.17, (1100, 1.08), (1e6, 2.0), (0.47, 0.499), (0.8, 1.2), (0.02, 0.06), {"ellipsoid": 0.5, "cylinder": 0.2, "blob": 0.15, "box": 0.15}, (0.03, 0.10)),
    "gel": (0.12, (1100, 1.08), (2e4, 2.0), (0.47, 0.499), (0.8, 1.2), (0.15, 0.40), {"ellipsoid": 0.6, "blob": 0.4}, (0.04, 0.10)),
    "bread": (0.12, (220, 1.25), (2e4, 2.0), (0.2, 0.3), (0.5, 0.9), (0.15, 0.35), {"box": 0.4, "ellipsoid": 0.4, "cylinder": 0.2}, (0.06, 0.15)),
}
P_WIDE = 0.30
SHAPES_ALL = {"box": 0.3, "cylinder": 0.2, "ellipsoid": 0.3, "blob": 0.2}


# ---------------------------------------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------------------------------------
def _plush_lumps(rng):
    """A sitting toy: a big head, two ears, four stubby limbs and a slightly lopsided body (lobes, not an egg)."""
    hd = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.25), 1.0])
    lumps = [(hd, float(rng.uniform(0.5, 0.7)), float(rng.uniform(3.0, 4.0)))]
    if rng.random() < 0.85:
        for sx in (-1, 1):
            lumps.append((hd + np.array([0.6 * sx, 0.0, 0.25]), float(rng.uniform(0.3, 0.45)), float(rng.uniform(8.0, 10.0))))
    for k in range(4):
        a = 2 * np.pi * (k + 0.5) / 4 + float(rng.uniform(-0.2, 0.2))
        lumps.append((np.array([np.cos(a), np.sin(a), -0.7]), float(rng.uniform(0.25, 0.4)), float(rng.uniform(5.5, 7.0))))
    lumps.append((rng.normal(size=3), float(rng.uniform(0.0, 0.1)), 2.0))
    return lumps


def sample_shape(rng, kind, L, category=None):
    """A shape of largest extent ~L (m) whose mesh has <= MAX_VERTICES vertices; >= 8 cells through its thinnest section
    (so the aspect ratios are bounded). Sponges are often slabs (thinnest side 30-45% of the longest); plush blobs get
    a head, ears and limbs."""
    for _ in range(200):
        if kind == "box":
            if category == "sponge" and rng.random() < 0.6:          # a slab: L x (thin..0.6 L) x thin
                t = float(rng.uniform(0.3, 0.45))
                u = [float(rng.uniform(t, 0.62)), t]
            else:
                u = np.sort(rng.uniform(0.62, 1.0, 2))[::-1]
            dims = [L, L * u[0], L * u[1]]
            if category != "sponge" or u[1] > 0.5:
                rng.shuffle(dims)
            s = Soft("box", {"dims": [float(d) for d in dims]})
        elif kind == "cylinder":
            ar = float(np.exp(rng.uniform(np.log(0.55), np.log(2.5))))      # H / 2R
            R, H = (L / 2, L * ar / 2) if ar < 1 else (L / (2 * ar), L)
            s = Soft("cylinder", {"R": float(R), "H": float(H)})
        elif kind == "ellipsoid":
            u = np.sort(rng.uniform(0.55, 1.0, 2))
            ax = [L / 2, L / 2 * u[1], L / 2 * u[0]]
            if rng.random() < 0.35:                                          # a ball
                ax = [L / 2] * 3
            rng.shuffle(ax)
            s = Soft("ellipsoid", {"axes": [float(a) for a in ax], "fine": bool(L > 0.08)})
        else:                                                                # a plush-like blob, sitting upright
            ax = np.array([rng.uniform(0.85, 1.0), rng.uniform(0.75, 0.9), 0.85])
            lumps = _plush_lumps(rng)
            v, _ = soft_mesh.blob(ax, lumps, fine="xfine")
            ax = ax * L / float(np.ptp(v, axis=0).max())
            s = Soft("blob", {"axes": ax.tolist(), "lumps": [(u.tolist(), A, k) for u, A, k in lumps], "res": "xfine"})
        if len(s.mesh()[0]) <= MAX_VERTICES:
            return s
    raise RuntimeError("no shape within the vertex budget")


def _lognormal(rng, med, gsd):
    return float(med * np.exp(rng.normal() * np.log(gsd)))


def sample_material(rng):
    """(category, rho, E, nu, mu, zeta, shape kind, size), uncapped: a category draw (70%) or a wide one (30%)."""
    if rng.random() < P_WIDE:
        cat = "wide"
        rho = float(np.exp(rng.uniform(np.log(20), np.log(1200))))
        E = float(np.exp(rng.uniform(np.log(1e3), np.log(3e6))))
        nu = float(rng.uniform(0.0, 0.45))
        mu = float(np.exp(rng.uniform(np.log(0.1), np.log(1.2))))
        zeta = float(np.exp(rng.uniform(np.log(0.02), np.log(0.6))))
        kinds = SHAPES_ALL
        size = (0.03, 0.15)
    else:
        names = list(CATEGORIES)
        w = np.array([CATEGORIES[n][0] for n in names])
        cat = str(rng.choice(names, p=w / w.sum()))
        _, r, e, nu_r, mu_r, z_r, kinds, size = CATEGORIES[cat]
        rho, E = _lognormal(rng, *r), _lognormal(rng, *e)
        nu, mu = float(rng.uniform(*nu_r)), float(rng.uniform(*mu_r))
        zeta = float(np.exp(rng.uniform(np.log(z_r[0]), np.log(z_r[1]))))
    k = list(kinds)
    kind = str(rng.choice(k, p=np.array([kinds[x] for x in k]) / sum(kinds.values())))
    L = float(np.exp(rng.uniform(np.log(size[0]), np.log(size[1]))))
    return cat, rho, E, nu, mu, zeta, kind, L


def sample_soft(rng):
    """A soft body: the material prior above, capped where the teacher cannot follow (the caps are the labels)."""
    for _ in range(200):
        cat, rho, E, nu, mu, zeta, kind, L = sample_material(rng)
        if rho * G * L / E > SAG_MAX:                     # sags > 8% under its own weight: redraw
            continue
        if st.wave_speeds(E, min(nu, NU_CAP), rho)[1] < CS_MIN:   # an impact at 2-4 m/s outruns its shear waves and
            continue                                              # inverts linear tets (real foam densifies instead)
        true = {"rho": rho, "E": E, "nu": nu, "mu": mu, "zeta": zeta}
        capped = []
        if nu > NU_CAP:
            nu = NU_CAP
            capped.append("nu")
        s = sample_shape(rng, kind, L, cat)
        cp, _ = st.wave_speeds(E, nu, rho)
        cp_max = min(CP_CAP, st.CFL * st.DT_SCALE * s.h_alt() * FPS * MAX_SUBSTEPS)   # the step's own limit
        if cp > cp_max:
            E = E * (cp_max / cp) ** 2
            capped.append("E")
        s.rho, s.E, s.nu, s.mu, s.zeta = rho, E, nu, mu, zeta
        s.params["category"] = cat
        s.damping()
        if s.params.get("zeta_effective", zeta) < zeta * 0.999:
            capped.append("zeta")
        s.params["capped"] = capped
        s.params["true"] = true
        return s
    raise RuntimeError("material prior: no draw within the sag limit")


def material_report(softs):
    """Percentiles of the simulated (label) values, per category counts, and the fraction capped per field."""
    out = {}
    for k, f in (("rho", lambda s: s.rho), ("E", lambda s: s.E), ("nu", lambda s: s.nu), ("mu", lambda s: s.mu),
                 ("zeta", lambda s: s.params.get("zeta_effective", s.zeta)), ("mass_kg", lambda s: s.mass),
                 ("extent_m", lambda s: float(s.extent.max())), ("vertices", lambda s: len(s.mesh()[0])),
                 ("substeps_per_frame", lambda s: 1.0 / (FPS * s.stable_dt()))):
        v = np.array([f(s) for s in softs], float)
        out[k] = {p: float(np.percentile(v, p)) for p in (1, 10, 50, 90, 99)}
    cats = [s.params["category"] for s in softs]
    out["categories"] = {c: cats.count(c) for c in sorted(set(cats))}
    shapes = [s.shape for s in softs]
    out["shapes"] = {c: shapes.count(c) for c in sorted(set(shapes))}
    for k in ("nu", "E", "zeta"):
        out[f"capped_{k}"] = float(np.mean([k in s.params["capped"] for s in softs]))
    return out


# ---------------------------------------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Obstacle:
    name: str
    xy: tuple
    yaw: float
    mu: float


@dataclass
class SoftSpec:
    soft: Soft
    init: str
    pose: tuple                                    # x, y, yaw
    obstacles: list = field(default_factory=list)
    n_families: int = 1
    seed: int = 0
    table_mu: float = 0.5
    ramp: dict = None
    orient: list = None                            # body orientation before the yaw (quaternion x, y, z, w)
    height: float = 0.0                            # dropped: lowest point above what is under it
    vel: tuple = (0.0, 0.0, 0.0)                   # initial linear (m/s) ...
    spin: tuple = (0.0, 0.0, 0.0)                  # ... and angular velocity (rad/s)


def _random_quat(rng):
    q = rng.normal(size=4)
    return (q / np.linalg.norm(q)).tolist()


def _face_down(rng, soft):
    """A resting orientation: boxes on a random face, cylinders upright (70%) or lying, ellipsoids on a principal
    axis, blobs sitting upright."""
    if soft.shape == "box":
        R = np.eye(3)[:, rng.permutation(3)] * rng.choice([-1, 1], 3)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
    elif soft.shape == "cylinder":
        R = np.eye(3) if rng.random() < 0.7 else np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    elif soft.shape == "ellipsoid":                 # on its shortest axis (stable; on a long one a ball-like body
        ax = np.asarray(soft.params["axes"])        # rocks for seconds and the planner grasps where it was)
        order = np.argsort(-ax) if rng.random() < 0.85 else rng.permutation(3)
        R = np.eye(3)[:, order]
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
    else:
        R = np.eye(3)
    return [float(v) for v in wp.quat_from_matrix(wp.mat33(*R.flatten()))]


def sample_spec(rng, names, seed=0):
    soft = sample_soft(rng)
    init = str(rng.choice(INITS, p=INIT_P))
    n_obs = int(rng.choice([0, 1, 2], p=[0.45, 0.4, 0.15]))
    if init in ("on_object", "against"):
        n_obs = max(n_obs, 1)
    L = float(soft.extent.max())
    pose = (float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-np.pi, np.pi)))
    if init == "dropped":                          # near the middle, chosen before the objects are placed clear of it
        pose = (float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.12, 0.12)), pose[2])   # (chosen after them,
        # one landed inside a blender jar and left at 14 m/s)
    obstacles = []
    for k in range(n_obs):
        for _try in range(100):
            xy = (float(rng.uniform(-WORKSPACE[0], WORKSPACE[0])), float(rng.uniform(-WORKSPACE[1], WORKSPACE[1])))
            if init == "on_object" and k == 0:
                xy = (pose[0], pose[1])
            elif init != "against" or k > 0:
                if np.hypot(xy[0] - pose[0], xy[1] - pose[1]) < 0.75 * L + 0.12:
                    continue
            if all(np.hypot(xy[0] - o.xy[0], xy[1] - o.xy[1]) > 0.2 for o in obstacles):
                break
        else:
            continue
        obstacles.append(Obstacle(str(rng.choice(names)), xy, float(rng.uniform(-np.pi, np.pi)),
                                  float(np.exp(rng.uniform(np.log(0.2), np.log(1.0))))))
    if init in ("on_object", "against") and not obstacles:
        init = "rest"
    if init == "against":                          # the body beside the first object, just clear of it
        o = obstacles[0]
        a = library.load(o.name)
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        cy, sy = np.cos(o.yaw), np.sin(o.yaw)
        v = a["collide_v"][:, :2] @ np.array([[cy, sy], [-sy, cy]])
        reach = float((v @ d).max())
        pose = (o.xy[0] + d[0] * (reach + 0.5 * L + 0.004), o.xy[1] + d[1] * (reach + 0.5 * L + 0.004), pose[2])
        if max(abs(pose[0]), abs(pose[1]) * TABLE_HALF[0] / TABLE_HALF[1]) > TABLE_HALF[0] - 0.08:
            init = "rest"
            pose = (float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.2, 0.2)), pose[2])
    table_mu = float(np.exp(rng.uniform(np.log(0.2), np.log(1.0))))
    ramp, n_fam = None, int(rng.choice([1, 2], p=[0.55, 0.45]))
    orient = _face_down(rng, soft)
    height, vel, spin = 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    if init == "ramp":
        for _ in range(50):                        # the plank runs down towards the table's middle: a body that rolls
            ramp = sample_ramp(rng)                # or slides off its foot stays on the table (all 7 pilot ramps
            if ramp is None:                       # pointing anywhere ended on the floor)
                break
            d = np.array([np.cos(ramp["yaw"]), np.sin(ramp["yaw"])])
            mid = np.asarray(ramp["mid"])
            if d @ mid > 0.5 * np.linalg.norm(mid) and np.linalg.norm(mid) > 0.12:
                break
        else:
            ramp = None
        if ramp is not None:
            ramp["angle"] = float(np.radians(rng.uniform(10.0, 35.0) if soft.shape != "ellipsoid" else rng.uniform(5.0, 15.0)))
            o, ax, lat, n = ramp_frame(ramp)
            corners = [o + sx * RAMP["L"] * ax + sy * 0.5 * RAMP["W"] * lat for sx in (0, 1) for sy in (-1, 1)]
            if not all(abs(c[0]) < TABLE_HALF[0] - 0.03 and abs(c[1]) < TABLE_HALF[1] - 0.03 for c in corners):
                ramp = None
        if ramp is None:
            init = "rest"
        else:
            obstacles = [o for o in obstacles if np.hypot(o.xy[0] - ramp["mid"][0], o.xy[1] - ramp["mid"][1]) > 0.4]
            n_fam = int(rng.choice([0, 1]))
    if init in ("dropped", "on_object"):
        orient = _random_quat(rng)
        height = float(rng.uniform(0.03, 0.20)) if init == "dropped" else float(rng.uniform(0.005, 0.02))
        if init == "dropped":                      # near the middle, gently tossed: 12 of 21 pilot drops (5-30 cm,
            pass                                                                                   # up to 0.3 m/s,
            a = rng.uniform(-np.pi, np.pi)                                                         # 3 rad/s) bounced
            sp = float(rng.uniform(0.0, 0.15))                                                     # off the table
            vel = (sp * np.cos(a), sp * np.sin(a), 0.0)
            spin = tuple(float(x) for x in rng.normal(size=3) * rng.uniform(0.0, 1.5) / np.sqrt(3))
        n_fam = int(rng.choice([0, 1], p=[0.3, 0.7]))
    return SoftSpec(soft, init, pose, obstacles, n_fam, seed, table_mu, ramp, orient, height, vel, spin)


def _rot(q):
    return np.array(wp.quat_to_matrix(wp.quat(*q)), np.float64).reshape(3, 3)


def _placement(spec):
    """Centre position and orientation (quaternion) of the body's mesh."""
    s = spec.soft
    v, _ = s.mesh()
    x, y, yaw = spec.pose
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ _rot(spec.orient)
    lift = s.r + 0.0003
    if spec.init == "ramp":
        o, ax, lat, n = ramp_frame(spec.ramp)
        Rp = np.stack([ax, lat, n], 1)
        R = Rp @ np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]]) @ _rot(spec.orient)
        vr = v @ R.T
        top = o + 0.55 * RAMP["L"] * ax + RAMP["T"] * n
        c = top - n * float((vr @ n).min()) + n * lift
        return c, R
    vr = v @ R.T
    base = 0.0
    if spec.init == "on_object":
        a = library.load(spec.obstacles[0].name)
        base = float(np.ptp(a["collide_v"][:, 2]))
    c = np.array([x, y, base - float(vr[:, 2].min()) + lift + spec.height])
    return c, R


def build_world(spec, rng, place=None):
    """Newton builder for one soft-body scene; shape roles and anchor slots in order of creation. place: (centre,
    orientation) overriding the spec's (checks)."""
    b = newton.ModelBuilder()
    roles, slots = [], []

    def cfg(mu):
        s = newton.ModelBuilder.ShapeConfig()
        s.mu, s.ke, s.kd, s.kf = float(mu), 1.0e3, 0.0, 1.0
        return s
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), I), hx=TABLE_HALF[0], hy=TABLE_HALF[1], hz=0.05,
                    cfg=cfg(spec.table_mu)); roles.append(ROLE_TABLE); slots.append(SLOT_TABLE)
    if spec.ramp is not None:
        o, ax, lat, n = ramp_frame(spec.ramp)
        c = o + 0.5 * RAMP["L"] * ax + 0.5 * RAMP["T"] * n
        q = wp.quat_from_matrix(wp.mat33(*np.stack([ax, lat, n], 1).flatten()))
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*c), q), hx=0.5 * RAMP["L"], hy=0.5 * RAMP["W"],
                        hz=0.5 * RAMP["T"], cfg=cfg(spec.ramp["mu"])); roles.append(ROLE_TABLE); slots.append(SLOT_PLANK)
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.80), I), hx=10.0, hy=10.0, hz=0.05,
                    cfg=cfg(FLOOR_MAT[0])); roles.append(ROLE_TABLE); slots.append(SLOT_TABLE)
    obs = []
    for o in spec.obstacles:
        a = library.load(o.name)
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), o.yaw)
        pos = (o.xy[0], o.xy[1], resting_z(a))
        b.add_shape_mesh(body=-1, xform=wp.transform(wp.vec3(*pos), q),
                         mesh=newton.Mesh(a["collide_v"].astype(np.float32), a["collide_f"].flatten().astype(np.int32)),
                         cfg=cfg(o.mu)); roles.append(ROLE_OBSTACLE); slots.append(SLOT_OBJ)
        obs.append({"name": o.name, "library": library.source_of(o.name), "pos": list(map(float, pos)), "quat": [0.0, 0.0, float(np.sin(o.yaw / 2)), float(np.cos(o.yaw / 2))],
                    "mu": o.mu})
    grip, gslots = add_gripper(b, GRIPPER_MU)
    roles += [ROLE_GRIPPER] * 3
    slots += gslots
    c, R = place if place is not None else _placement(spec)
    rot = wp.quat_from_matrix(wp.mat33(*np.asarray(R, float).flatten())) if np.ndim(R) == 2 else R
    p0, n, t0, nt = add_soft(b, spec.soft, c, rot)
    return b, {"gripper": grip, "particles": (p0, n), "tets": (t0, nt), "obstacles": obs, "roles": roles, "slots": slots,
               "centre": [float(v) for v in c], "rot": [float(v) for v in rot]}


# ---------------------------------------------------------------------------------------------------------
# what is under the gripper
# ---------------------------------------------------------------------------------------------------------
CELL = 0.004


class Floor:
    """Height of the static scene (table, plank, objects, the floor off the table) on a 4 mm grid."""

    def __init__(self, spec, extent=0.8):
        from lsim.data.cloth import _surface_samples         # (read-only reuse of the cloth planner's sampler)
        self.lo, self.n = -extent, int(round(2 * extent / CELL)) + 1
        hm = np.full((self.n, self.n), -0.75)
        c = self.lo + CELL * np.arange(self.n)
        on = (np.abs(c)[:, None] <= TABLE_HALF[0] + 0.5 * CELL) & (np.abs(c)[None, :] <= TABLE_HALF[1] + 0.5 * CELL)
        hm[on] = 0.0
        pts = []
        if spec.ramp is not None:
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
        self.hm = hm

    def _index(self, xy):
        xy = np.atleast_2d(xy)
        return (np.clip(np.round((xy[:, 0] - self.lo) / CELL).astype(int), 0, self.n - 1),
                np.clip(np.round((xy[:, 1] - self.lo) / CELL).astype(int), 0, self.n - 1))

    def at(self, xy):
        return self.hm[self._index(xy)]

    def under_box(self, centre, yaw, half):
        """Highest static surface under a box footprint (centre xy, yaw, half extents x, y)."""
        u, v = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
        p = np.stack([u.ravel() * half[0], v.ravel() * half[1]], 1)
        c, s = np.cos(yaw), np.sin(yaw)
        p = p @ np.array([[c, s], [-s, c]]) + np.asarray(centre)[:2]
        return float(self.at(p).max())

    def top(self):
        return float(self.hm[self.hm > -0.5].max())


def _finger_floor(floor, xy, yaw, w):
    """Highest static surface under either pad at opening w."""
    ax = np.array([np.cos(yaw), np.sin(yaw)])
    return max(floor.under_box(np.asarray(xy) + s * (w / 2 + FINGER_HALF[0]) * ax, yaw, FINGER_HALF[:2]) for s in (-1, 1))


# ---------------------------------------------------------------------------------------------------------
# gripper paths: position (3), yaw, opening per frame for the whole timeline; families write segments of them
# ---------------------------------------------------------------------------------------------------------
class _Cut(Exception):
    """A family's path reached the end of the episode."""


class Path:
    def __init__(self, n):
        self.p = np.tile(np.array(START, float), (n + 1, 1))
        self.yaw = np.zeros(n + 1)
        self.w = np.full(n + 1, 0.1)
        self.last = n

    def seg(self, f, p1=None, yaw1=None, w1=None, seconds=0.3, ease=True):
        """From frame f's state to (p1, yaw1, w1) over `seconds`; the end state holds after. Returns the end frame."""
        p0, y0, w0 = self.p[f].copy(), self.yaw[f], self.w[f]
        p1 = p0 if p1 is None else np.asarray(p1, float)
        yaw1 = y0 if yaw1 is None else y0 + (np.angle(np.exp(1j * (yaw1 - y0))))
        w1 = w0 if w1 is None else float(np.clip(w1, W_CLOSED, W_MAX))
        n = max(1, int(round(seconds * FPS)))
        for k in range(1, n + 1):
            if f + k > self.last:
                raise _Cut()
            s = k / n
            s = s * s * (3 - 2 * s) if ease else s
            self.p[f + k] = (1 - s) * p0 + s * p1
            self.yaw[f + k] = (1 - s) * y0 + s * yaw1
            self.w[f + k] = (1 - s) * w0 + s * w1
        self.p[f + n:], self.yaw[f + n:], self.w[f + n:] = p1, yaw1, w1
        if f + n >= self.last:
            raise _Cut()
        return f + n

    def hold(self, f, seconds):
        n = max(1, int(round(seconds * FPS)))
        if f + n >= self.last:
            raise _Cut()
        return f + n

    def quats(self):
        h = 0.5 * self.yaw
        q = np.zeros((len(self.yaw), 7), np.float32)
        q[:, :3] = self.p
        q[:, 5], q[:, 6] = np.sin(h), np.cos(h)
        return q

    def abort(self, f):
        """A grasp that pinned nothing: open, rise and wait (the pads would plough through the body otherwise)."""
        try:
            g = self.seg(f, w1=min(W_MAX, self.w[f] + 0.03), seconds=0.15)
            self.seg(g, self.p[g] + np.array([0.0, 0.0, 0.12]), seconds=0.3)
        except _Cut:
            pass

    def stop(self, f, a, b):
        """A guard tripped at the end of frame f inside the move over frames (a, b]: hold frame f + 1's state through b,
        shift everything after b by what was not done."""
        g = f + 1
        if g >= b:
            return
        dp, dw = self.p[b] - self.p[g], self.w[b] - self.w[g]
        self.p[g + 1:b + 1], self.w[g + 1:b + 1] = self.p[g], self.w[g]
        self.p[b + 1:] -= dp
        self.w[b + 1:] = np.clip(self.w[b + 1:] - dw, W_CLOSED, W_MAX)


def _body_axes(x):
    """The body's horizontal principal axes: the yaw of its long axis, extents along long and short axes."""
    xy = x[:, :2] - x[:, :2].mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    e1 = vt[0]
    yaw = float(np.arctan2(e1[1], e1[0]))
    return yaw


def _extent_along(x, yaw, z0, z1):
    """Width of the body along direction yaw between heights z0 and z1 (the pads' band), and its centre there."""
    d = np.array([np.cos(yaw), np.sin(yaw)])
    band = (x[:, 2] >= z0) & (x[:, 2] <= z1)
    if band.sum() < 4:
        band = np.ones(len(x), bool)
    s = x[band, :2] @ d
    return float(s.max() - s.min()), float(0.5 * (s.max() + s.min()))


def _travel(path, f, xy, yaw, w, z_target, floor, x, rng):
    """Up to a safe height, across (turning and opening), down to 4 cm over z_target. Returns the end frame."""
    H = max(0.25, floor.top() + 0.12, float(x[:, 2].max()) + 0.1, path.p[f, 2])
    f = path.seg(f, [path.p[f, 0], path.p[f, 1], H], seconds=max(0.15, (H - path.p[f, 2]) / 0.8))
    dist = float(np.hypot(*(np.asarray(xy) - path.p[f, :2])))
    f = path.seg(f, [xy[0], xy[1], H], yaw, w, seconds=max(float(rng.uniform(0.3, 0.45)), dist / 1.2))
    return path.seg(f, [xy[0], xy[1], z_target + 0.04], seconds=max(0.15, (H - z_target - 0.04) / 1.2))


def _grasp_setup(x, floor, soft, rng):
    """Where and how to grasp: the yaw that closes the pads across the body's narrow side, the fingertip height, the
    width at the pads' band and the band's centre; None if it does not fit or a pad would land on something."""
    yaw_long = _body_axes(x)
    zlo, zhi = float(x[:, 2].min()), float(x[:, 2].max())
    H = zhi - zlo
    c = x.mean(0)
    for yaw in (yaw_long + np.pi / 2, yaw_long):                         # pads close along local x
        z = max(zlo + float(rng.uniform(0.25, 0.45)) * H, zhi - (2 * FINGER_HALF[2] - 0.012))
        wc, mid = _extent_along(x, yaw, z, z + 2 * FINGER_HALF[2])
        if wc + 0.02 > W_MAX:
            continue
        d = np.array([np.cos(yaw), np.sin(yaw)])
        n = np.array([-d[1], d[0]])
        xy = d * mid + n * float(c[:2] @ n)
        w_open = min(W_MAX, wc + float(rng.uniform(0.025, 0.04)))
        z = max(z, _finger_floor(floor, xy, yaw, w_open) + CLEAR)
        if z > zlo + 0.8 * H:
            continue
        return {"xy": xy, "yaw": float(yaw), "z": float(z), "wc": wc, "w_open": w_open, "zlo": zlo, "zhi": zhi}
    return None


F_HAND = (8.0, 40.0)                              # N: squeeze / press / grip force limits, a hand's or a parallel
                                                  # gripper's working range (human pinch ~10-60 N; Robotiq 2F-85
                                                  # 20-235 N, Franka hand 70 N continuous). The squeeze and the press
                                                  # go to a target width / depth (30-80% of the width, 10-50% of the
                                                  # height) and stop early only at this force: a cap of 0.2 E x pad
                                                  # area left soft bodies dented by 2% of their size (pilot v1)


def plan_family(rng, family, spec, x, path, f0, floor, weight):
    """Write the gripper's path for one family from frame f0 and the body's vertices x (world). Returns (events
    [(frame, "grasp") | (frame, "release")], guards [(first, last, kind, cap)], end frame, description)."""
    ev, guards, desc = [], [], {"family": family, "start": int(f0)}
    try:
        f = _plan(rng, family, spec, x, path, f0, floor, weight, ev, guards, desc)
        return ev, guards, f, desc
    except _Cut:
        desc["cut_by_episode_end"] = True
        return ev, guards, path.last, desc


def _clear_goal(rng, floor, start, dist, r, along=True, tries=12):
    """A goal dist away on the table whose way there (along=True: the whole sweep, else the landing spot) is clear of
    objects and the plank by r: a held body dragged into an object is pinched between a kinematic pad and a static
    surface (a pilot drag reached 18 kN and blew up)."""
    for _ in range(tries):
        ang = float(rng.uniform(-np.pi, np.pi))
        goal = np.asarray(start) + dist * np.array([np.cos(ang), np.sin(ang)])
        if np.any(np.abs(goal) > np.array(TABLE_HALF) - 0.08):
            continue
        pts = [goal] if not along else [np.asarray(start) + u * (goal - np.asarray(start)) for u in np.linspace(0.15, 1.0, 8)]
        if all(floor.under_box(q, 0.0, (r, r)) < 0.003 for q in pts):
            return goal
    return None


def _clip_xy(xy, m=0.08):
    return np.clip(xy, [-TABLE_HALF[0] + m, -TABLE_HALF[1] + m], [TABLE_HALF[0] - m, TABLE_HALF[1] - m])


def _plan(rng, family, spec, x, path, f0, floor, weight, ev, guards, desc):
    s = spec.soft
    c = x.mean(0)
    if np.any(np.abs(c[:2]) > np.array(TABLE_HALF) - 0.04) or c[2] < -0.05:
        desc["skipped"] = "not on the table"
        return f0
    if family in ("push", "press"):
        if family == "push":
            dist = 0.03 + float(rng.uniform(0.04, 0.16))
            r = 0.5 * float(np.ptp(x[:, :2], axis=0).max())
            for _ in range(12):                           # a way ahead clear of objects and the plank: a body pushed
                ang = float(rng.uniform(-np.pi, np.pi))   # into an object is pinched (pilot: 9 N on a 56 g sponge, then
                d = np.array([np.cos(ang), np.sin(ang)])  # dragged up the object by the rising finger)
                pts = [c[:2] + d * u for u in np.linspace(0.3 * r, r + dist, 8)]
                if all(floor.under_box(q, ang, (0.5 * r, 0.5 * r)) < 0.003 for q in pts):
                    break
            else:
                desc["skipped"] = "no clear way"
                return f0
            ext = x[:, :2] @ d
            zlo, zhi = float(x[:, 2].min()), float(x[:, 2].max())
            z = max(zlo + float(rng.uniform(0.15, 0.5)) * (zhi - zlo), zlo + CLEAR,
                    zhi - (2 * FINGER_HALF[2] - 0.01))            # the palm clears the top (pilot: it dented it)
            side = np.array([-d[1], d[0]])
            lat = float(rng.uniform(-0.3, 0.3)) * float(np.ptp(x[:, :2] @ side))
            start = c[:2] + d * (float(ext.min() - c[:2] @ d) - FINGER_HALF[0] * 2 - 0.03) + side * lat
            z = max(z, _finger_floor(floor, start, ang, W_CLOSED) + CLEAR)
            f = _travel(path, f0, start, ang, W_CLOSED, z, floor, x, rng)
            a = f
            f = path.seg(f, [start[0], start[1], z], seconds=0.2)
            guards.append((a, f, "pad", 2.0 + 0.5 * weight))
            end = start + d * dist
            speed = float(rng.uniform(0.05, 0.3))
            a = f
            f = path.seg(f, [end[0], end[1], z], seconds=dist / speed, ease=False)
            cap = float(np.clip(5.0 * weight, 3.0, 40.0))
            guards.append((a, f, "pad", cap))
            f = path.seg(f, [end[0] - 0.01 * d[0], end[1] - 0.01 * d[1], z + 0.12], seconds=0.35)
            desc.update(direction=d.tolist(), distance=dist, speed=speed, height=z, cap_N=cap)
            return f
        # press: a closed finger onto a point of the top
        top = x[:, 2] > x[:, 2].max() - 0.3 * np.ptp(x[:, 2])
        k = int(rng.choice(np.flatnonzero(top)))
        xy = x[k, :2]
        near = np.linalg.norm(x[:, :2] - xy, axis=1) < 0.015
        zt = float(x[near, 2].max()) + s.r + 0.001
        H = zt - float(x[:, 2].min())
        yaw = float(rng.uniform(-np.pi, np.pi))
        f = _travel(path, f0, xy, yaw, W_CLOSED, zt, floor, x, rng)
        a = f
        f = path.seg(f, [xy[0], xy[1], zt + 0.003], seconds=0.2)
        guards.append((a, f, "pad", 1.0 + 0.5 * weight))
        depth = float(rng.uniform(0.1, 0.5)) * H
        zb = max(zt - depth, _finger_floor(floor, xy, yaw, W_CLOSED) + CLEAR)
        speed = float(np.exp(rng.uniform(np.log(0.02), np.log(0.15))))
        cap = float(np.exp(rng.uniform(np.log(F_HAND[0]), np.log(F_HAND[1]))))
        a = f
        f = path.seg(f, [xy[0], xy[1], zb], seconds=max(0.1, (zt - zb) / speed), ease=False)
        guards.append((a, f, "pad", cap))
        f = path.hold(f, float(rng.uniform(0.2, 0.8)))
        f = path.seg(f, [xy[0], xy[1], zt + 0.1], seconds=max(0.15, 0.1 / float(rng.uniform(0.05, 0.2))), ease=False)
        desc.update(point=xy.tolist(), depth=depth, speed=speed, cap_N=cap, vertex=k)
        return f
    g = _grasp_setup(x, floor, s, rng)
    if g is None:
        desc["skipped"] = "does not fit between the pads"
        return f0
    xy, yaw, z = g["xy"], g["yaw"], g["z"]
    f = _travel(path, f0, xy, yaw, g["w_open"], z, floor, x, rng)
    a = f
    f = path.seg(f, [xy[0], xy[1], z], seconds=0.25)
    guards.append((a, f, "pad", 1.0 + 0.5 * weight))
    if family == "squeeze":
        sq = float(rng.uniform(0.2, 0.7))                       # to 30-80% of its width ...
        speed = float(rng.uniform(0.04, 0.12))
        cap = float(np.exp(rng.uniform(np.log(F_HAND[0]), np.log(F_HAND[1]))))   # ... unless the hand's force runs out
        w1 = g["wc"] * (1.0 - sq)
        a = f
        f = path.seg(f, w1=w1, seconds=max(0.1, (g["w_open"] - w1) / speed), ease=False)
        guards.append((a, f, "pad", cap))
        f = path.hold(f, float(rng.uniform(0.3, 1.0)))
        f = path.seg(f, w1=g["w_open"], seconds=max(0.1, (g["w_open"] - w1) / speed), ease=False)
        f = path.hold(f, 0.1)
        f = path.seg(f, [xy[0], xy[1], z + 0.15], seconds=0.4)
        desc.update(squeeze=sq, speed=speed, cap_N=cap, width=g["wc"], yaw=yaw, height=z)
        return f
    # a pinch grasp: close until the pads hold it (force-guarded), pin what touches them
    sq = float(rng.uniform(0.1, 0.3))
    w1 = g["wc"] * (1.0 - sq)
    cap = float(np.clip(3.0 * weight, F_HAND[0], F_HAND[1]))
    a = f
    f = path.seg(f, w1=w1, seconds=max(0.15, (g["w_open"] - w1) / 0.06), ease=False)
    guards.append((a, f, "pad", cap))
    ev.append((f, "grasp"))
    desc.update(grasp_squeeze=sq, grasp_cap_N=cap, width=g["wc"], yaw=yaw, height=z)
    p = path.p[f].copy()

    def let_go(f, rise=0.1, secs=0.3):
        ev.append((f, "release"))
        f = path.seg(f, w1=min(W_MAX, path.w[f] + 0.03), seconds=0.1)
        return path.seg(f, path.p[f] + np.array([0.0, 0.0, rise]), seconds=secs)
    if family == "lift_drop":
        Hl = float(rng.uniform(0.08, 0.3))
        top = p + np.array([*rng.uniform(-0.08, 0.08, 2), Hl])
        top[:2] = _clip_xy(top[:2])
        f = path.seg(f, top, seconds=Hl / float(rng.uniform(0.2, 0.5)))
        f = path.hold(f, float(rng.uniform(0.2, 0.6)))
        f = let_go(f)
        desc.update(lift=Hl)
        return f
    if family == "lift_place":
        Hl = float(rng.uniform(0.05, 0.15))
        dist = float(rng.uniform(0.08, 0.25))
        goal = _clear_goal(rng, floor, p[:2], dist, 0.5 * g["wc"] + 0.03, along=False)
        if goal is None:
            goal = p[:2].copy()                                   # nowhere clear: set it back down where it was
        below = z - g["zlo"]                                    # fingertips above the body's bottom at the grasp
        land = floor.under_box(goal, yaw, (0.5 * g["wc"] + 0.02, 0.5 * g["wc"] + 0.02))
        zg = land + below + 0.004
        f = path.seg(f, p + np.array([0.0, 0.0, Hl]), seconds=float(rng.uniform(0.3, 0.6)))
        f = path.seg(f, [goal[0], goal[1], max(zg, p[2]) + Hl], seconds=max(0.3, dist / 0.4))
        a = f
        f = path.seg(f, [goal[0], goal[1], zg], seconds=float(rng.uniform(0.4, 0.8)))
        guards.append((a, f, "place", 0.3))
        f = path.hold(f, 0.1)
        f = let_go(f, 0.12)
        desc.update(lift=Hl, goal=goal.tolist())
        return f
    if family in ("drag", "slide"):
        lift = float(rng.uniform(0.0, 0.01)) if family == "drag" else float(rng.uniform(0.003, 0.012))
        dist = float(rng.uniform(0.1, 0.3)) if family == "drag" else float(rng.uniform(0.08, 0.2))
        speed = float(rng.uniform(0.1, 0.4)) if family == "drag" else float(rng.uniform(0.3, 0.8))
        goal = _clear_goal(rng, floor, p[:2], dist + (0.15 if family == "slide" else 0.0), 0.5 * g["wc"] + 0.03)
        if goal is None:
            desc["skipped"] = "no clear way"
            f = path.seg(f, w1=min(W_MAX, path.w[f] + 0.03), seconds=0.15)
            ev.append((f, "release"))
            return path.seg(f, path.p[f] + np.array([0.0, 0.0, 0.12]), seconds=0.3)
        if family == "slide":                                     # it skids on past the release point
            goal = p[:2] + (goal - p[:2]) * dist / (dist + 0.15)
        f = path.seg(f, p + np.array([0.0, 0.0, lift]), seconds=0.15)
        f = path.seg(f, [goal[0], goal[1], p[2] + lift], seconds=max(0.1, float(np.linalg.norm(goal - p[:2])) / speed),
                     ease=family == "drag")
        if family == "slide":                                    # open and rise at once: it skids on
            ev.append((f, "release"))
            f = path.seg(f, path.p[f] + np.array([0.0, 0.0, 0.12]), w1=min(W_MAX, path.w[f] + 0.04), seconds=0.25)
        else:
            f = let_go(f)
        desc.update(lift=lift, speed=speed, goal=goal.tolist())
        return f
    raise ValueError(family)


# ---------------------------------------------------------------------------------------------------------
# a batch
# ---------------------------------------------------------------------------------------------------------
def run_batch(specs, seed=0, record=RECORD, log=print):
    """Simulate one batch of soft-body scenes (one teacher, every world its own body, material, scene and gripper)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    b = newton.ModelBuilder()
    worlds, roles, slots, infos = [], [], [], []
    for sp in specs:
        wb, info = build_world(sp, np.random.default_rng(sp.seed))
        po, to, bo = b.particle_count, b.tet_count, b.body_count
        b.add_world(wb)
        worlds.append({"particles": (po + info["particles"][0], info["particles"][1]),
                       "tets": (to + info["tets"][0], info["tets"][1]), "soft": sp.soft,
                       "gripper": tuple(bo + g for g in info["gripper"])})
        roles += info["roles"]
        slots += info["slots"]
        infos.append(info)
    W = len(specs)
    settle = max(int(round(SETTLE[sp.init] * FPS)) for sp in specs)
    n_rec = int(round(record * FPS))
    n_tot = settle + n_rec
    paths = [Path(n_tot) for _ in range(W)]
    substeps, ce = substeps_for([sp.soft for sp in specs])
    t = SoftTeacher(b, worlds, roles, slots, np.stack([p.quats() for p in paths], 1), np.stack([p.w for p in paths], 1),
                    substeps, ce)
    log(f"[soft] built {W} worlds, {t.model.particle_count} vertices, {t.model.tet_count} tets, {substeps} substeps a "
        f"frame (contacts every {ce}), settle {settle} frames, in {time.time() - t0:.1f} s")
    floors = [Floor(sp) for sp in specs]
    weights = [sp.soft.mass * G for sp in specs]
    # bodies that start moving at the first recorded frame (dropped, ramp, on_object) wait, held, until then
    held_until = [settle - int(round(SETTLE[sp.init] * FPS)) for sp in specs]
    for w in range(W):
        if held_until[w] > 0:
            p0, n = worlds[w]["particles"]
            t.flags[p0:p0 + n] &= ~int(newton.ParticleFlags.ACTIVE)
    t.model.particle_flags.assign(t.flags)

    def release_hold(w):
        p0, n = worlds[w]["particles"]
        t.flags[p0:p0 + n] |= int(newton.ParticleFlags.ACTIVE)
        t.model.particle_flags.assign(t.flags)
        sp = specs[w]
        if any(sp.vel) or any(sp.spin):
            x = t.s0.particle_q.numpy()
            qd = t.s0.particle_qd.numpy()
            m = t.mass[p0:p0 + n, None]
            c = (m * x[p0:p0 + n]).sum(0) / m.sum()
            qd[p0:p0 + n] = np.asarray(sp.vel) + np.cross(np.asarray(sp.spin), x[p0:p0 + n] - c)
            t.s0.particle_qd.assign(qd)

    for w in range(W):
        if held_until[w] == 0:
            release_hold(w)
    events = {w: [] for w in range(W)}
    guards = {w: [] for w in range(W)}
    plans = {w: [] for w in range(W)}
    busy_until = np.array([settle + (int(1.0 * FPS) if sp.init in ("dropped", "on_object") else 0) for sp in specs])
    remaining = [sp.n_families for sp in specs]
    tripped = np.zeros(W, int)
    safe_done = np.zeros(W, bool)
    rec_x, rec_c, rec_s, rec_g, pinned_m = [], [], [], [], []

    def assign():
        t.path.assign(np.stack([p.quats() for p in paths], 1))
        t.width.assign(np.stack([p.w for p in paths], 1).astype(np.float32))
    for f in range(n_tot + 1):
        for w in range(W):
            if f == held_until[w] and f > 0:
                release_hold(w)
        if f >= settle:
            planned = False
            for w, sp in enumerate(specs):
                if remaining[w] > 0 and f >= busy_until[w] and f < n_tot - int(1.2 * FPS):
                    p0, n = worlds[w]["particles"]
                    x = t.s0.particle_q.numpy()[p0:p0 + n].astype(np.float64)
                    vc = np.linalg.norm(t.s0.particle_qd.numpy()[p0:p0 + n].mean(0))
                    if vc > 0.03 and f < busy_until[w] + FPS:       # still rolling or rocking: wait (up to 1 s)
                        continue
                    fam = str(rng.choice(FAMILIES, p=FAMILY_P))
                    ev, gd, end, desc = plan_family(rng, fam, sp, x, paths[w], f, floors[w], weights[w])
                    events[w] += ev
                    guards[w] += [list(g_) + [False] for g_ in gd]
                    plans[w].append(desc)
                    busy_until[w] = end + int(0.3 * FPS)
                    remaining[w] -= 1
                    planned = True
            if planned:
                assign()
        for w in range(W):
            for e in events[w]:
                if e[0] == f:
                    if e[1] == "grasp":
                        ids = t.grasp(w)
                        e_ids = [int(i) for i in ids]
                        events[w][events[w].index(e)] = (f, "grasp", e_ids)
                        if not e_ids:                     # missed: open and rise; the rest of the family is void
                            paths[w].abort(f)
                            for gd in guards[w]:
                                if gd[0] >= f:
                                    gd[4] = True
                            plans[w][-1]["grasp_missed"] = True
                            assign()
                    elif e[1] == "release":
                        t.release(w)
        if f >= settle:
            rec_x.append(t.s0.particle_q.numpy().copy())
            pinned_m.append([float(t.mass[t.grasped(w)].sum()) if t.grasped(w) else 0.0 for w in range(W)])
        if f == n_tot:
            break
        r = t.frame()
        changed = False
        for w in range(W):                                # force guards: stop a move into the body where it pushes back
            for gd in guards[w]:
                a, bnd, kind, cap, done = gd
                if done or not (a <= f < bnd):
                    continue
                if kind == "pad":                         # the harder-pressed pad (or the palm)
                    hit = float(np.linalg.norm(r["slot"][w, st.SLOT_FL:st.SLOT_PALM + 1], axis=-1).max()) > cap
                else:                                     # placing: the held body is pushed up by what it lands on
                    hit = bool(t.grasped(w)) and float(r["grasp"][w, 2]) < cap * weights[w]   # holds < 30% of it
                if hit:
                    paths[w].stop(f, a, bnd)
                    gd[4] = True
                    tripped[w] += 1
                    changed = True
        for w in range(W):                                # a pinch the guards did not foresee: let go and back off
            if f < settle or safe_done[w]:
                continue
            Fs = np.linalg.norm(r["slot"][w], axis=-1)
            lim = max(F_SAFE_W * weights[w], F_SAFE)
            if Fs[st.SLOT_FL:st.SLOT_PALM + 1].max() > lim or (t.grasped(w) and max(Fs[st.SLOT_OBJ], Fs[st.SLOT_PLANK]) > lim):
                t.release(w)
                paths[w].abort(f + 1)
                for gd in guards[w]:
                    gd[4] = True
                events[w] = [e for e in events[w] if e[0] <= f]
                events[w].append((f + 1, "release"))
                remaining[w] = 0
                busy_until[w] = n_tot
                safe_done[w] = True
                if plans[w]:
                    plans[w][-1]["safety_release"] = int(f - settle)
                changed = True
        if changed:
            assign()
        if f >= settle:
            rec_c.append(r["contact"])
            rec_s.append(r["slot"])
            rec_g.append(r["grasp"])
    log(f"[soft] simulated {W} x {n_tot / FPS:.1f} s in {time.time() - t0:.1f} s")
    X = np.stack(rec_x)
    grip = np.stack([p.quats() for p in paths], 1)[settle:]
    gw = np.stack([p.w for p in paths], 1)[settle:].astype(np.float32)
    Fg = np.zeros((len(rec_x), W, 3), np.float32)
    acc = np.zeros_like(grip[:, :, :3])
    acc[1:-1] = (grip[2:, :, :3] - 2 * grip[1:-1, :, :3] + grip[:-2, :, :3]) * FPS * FPS
    for j in range(len(rec_g)):
        m = np.asarray(pinned_m[j])[:, None]
        Fg[j] = np.where(m > 0, rec_g[j] + m * (acc[j] - np.array([0.0, 0.0, -G])), 0.0)
    return {"specs": specs, "infos": infos, "worlds": worlds, "x": X, "grip": grip, "grip_w": gw,
            "path": np.stack([p.quats() for p in paths], 1), "path_w": np.stack([p.w for p in paths], 1).astype(np.float32),
            "grasp_F": Fg, "contact_F": np.stack(rec_c + [rec_c[-1]]).astype(np.float32),
            "slot_F": np.stack(rec_s + [rec_s[-1]]).astype(np.float32), "events": events, "plans": plans,
            "guards": guards, "tripped": tripped, "settle": settle, "substeps": substeps, "collide_every": ce,
            "mass": t.mass, "tets": t.model.tet_indices.numpy(), "Dm": t.model.tet_poses.numpy(), "teacher": t}


def deformation(x, tets, Dm, rest, step=2):
    """How much the body deforms, largest over frames (every `step`-th): strain_p99, the 99th percentile over tets of
    the largest principal strain max |sigma_i - 1| (sigma: singular values of F) -- local, pad edges and impacts
    count; strain_p50, the same at the median tet (the body as a whole); shape_dev, the 90th percentile of the
    vertices' distance from the best rigid fit of the rest shape over the body's largest extent (what one sees)."""
    s99 = s50 = dev = 0.0
    r0 = rest - rest.mean(0)
    L = float(np.ptp(rest, axis=0).max())
    for k in range(0, len(x), step):
        xk = np.asarray(x[k], np.float64)
        Ds = np.stack([xk[tets[:, 1]] - xk[tets[:, 0]], xk[tets[:, 2]] - xk[tets[:, 0]], xk[tets[:, 3]] - xk[tets[:, 0]]], -1)
        e = np.abs(np.linalg.svd(Ds @ Dm, compute_uv=False) - 1.0).max(1)
        s99, s50 = max(s99, float(np.percentile(e, 99))), max(s50, float(np.median(e)))
        a = xk - xk.mean(0)
        U, _, Vt = np.linalg.svd(r0.T @ a)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        dev = max(dev, float(np.percentile(np.linalg.norm(a - r0 @ R, axis=1), 90)) / L)
    return {"strain_p99": s99, "strain_p50": s50, "shape_dev": dev}


EXPLODED = 0.4                                     # m in one frame
F_SAFE, F_SAFE_W = 80.0, 200.0                     # N, x weight: past both, any pinch ends the gripper's family


def quality(res):
    """Per world: fastest vertex (m/s), exploded (a 0.4 m move in a frame or a vertex 3 rest radii from the body's
    centroid: rule 3's spatial bound on the body itself -- balls that roll off the table cross the floor), left the table, the smallest
    tet volume ratio over every recorded frame (inverted: <= 0), contact and grasp force over weight, the deepest vertex
    under the table top and inside a pad (not pinned), speed at the end."""
    X, out = res["x"], []
    tets, Dm = res["tets"], res["Dm"].astype(np.float64)
    for w, wd in enumerate(res["worlds"]):
        p0, n = wd["particles"]
        t0, nt = wd["tets"]
        s = wd["soft"]
        x = X[:, p0:p0 + n].astype(np.float64)
        mv = np.linalg.norm(np.diff(x, axis=0), axis=-1).max(1)
        T = tets[t0:t0 + nt] - p0
        D = Dm[t0:t0 + nt]
        Jmin = np.inf
        for k in range(0, len(x), 2):
            Jmin = min(Jmin, float(soft_mesh.tet_J(x[k], T, D).min()))
        wt = s.mass * G
        on = (np.abs(x[..., 0]) < TABLE_HALF[0] - 0.01) & (np.abs(x[..., 1]) < TABLE_HALF[1] - 0.01) & (x[..., 2] > -0.05)
        pen_t = float(np.max(np.where(on, s.r - x[..., 2], -1.0), initial=-1.0))
        pinned = set()
        for e in res["events"][w]:
            if e[1] == "grasp" and len(e) > 2:
                pinned |= set(e[2])
        free = np.ones(n, bool)
        free[list(pinned)] = False
        pen_p = 0.0
        for k in range(len(x)):
            for c, R, h in st.finger_boxes(res["grip"][k, w], float(res["grip_w"][k, w]))[1:]:
                lp = (x[k, free] - c) @ R
                inside = np.min(np.asarray(h) - np.abs(lp), axis=1)
                if inside.size:
                    pen_p = max(pen_p, float(inside.max()))
        sp = np.linalg.norm(x[-1] - x[-2], axis=1) * FPS
        sp = sp[np.isfinite(sp)]
        rest_r = float(np.linalg.norm(s.mesh()[0], axis=1).max())
        spread = float(np.linalg.norm(x - x.mean(1, keepdims=True), axis=-1).max())
        c_end = x[-1].mean(0)
        out.append({**deformation(x, T, D, s.mesh()[0]), "exploded": bool(mv.max() > EXPLODED or not np.isfinite(x).all() or spread > 3.0 * rest_r),
                    "spread_over_rest": spread / rest_r,
                    "left_table": bool(c_end[2] < -0.3 or np.any(np.abs(c_end[:2]) > np.array(TABLE_HALF))),
                    "vmax": float(mv.max() * FPS), "min_J": Jmin, "inverted": bool(Jmin <= 0.0),
                    "contact_over_weight": [float(v) for v in np.linalg.norm(res["contact_F"][:, w], axis=-1).max(0) / wt],
                    "pad_force_N": float(np.linalg.norm(res["slot_F"][:, w, st.SLOT_FL:st.SLOT_PALM + 1], axis=-1).max()),
                    "grasp_over_weight": float(np.linalg.norm(res["grasp_F"][:, w], axis=-1).max() / wt),
                    "table_pen_mm": 1e3 * max(pen_t, 0.0), "pad_pen_mm": 1e3 * pen_p,
                    "end_speed": float(sp.max()) if sp.size else float("nan"), "vertices": int(n),
                    "guards_tripped": int(res["tripped"][w]), "init": res["specs"][w].init, "shape": s.shape,
                    "category": s.params.get("category", "?"), "families": [p["family"] for p in res["plans"][w]],
                    "skipped": [p.get("skipped") for p in res["plans"][w] if p.get("skipped")]})
    return out


def save_shard(out, i, res, seed):
    """shard_<i>.npz in the cloth shards' layout -- cloth_c (F + 1, W, 3) centroids, cloth_d (F + 1, sum V, 3) fp32
    offsets of EVERY tet vertex (interior too), cloth_off (W + 1), grip_q / grip_path, grasp_F, contact_F -- plus what a
    soft body adds: grip_w / grip_w_path (the pads' inner opening, m), soft_rest (sum V, 3) the rest mesh in the body
    frame, soft_tets (sum T, 4) and soft_tris (sum S, 3) the tets and outward surface triangles (indices into the
    world's own vertices) with tet_off / tri_off, soft_mass (sum V,) lumped vertex masses, slot_F (F + 1, W, 6, 3) the
    contact force on the body per surface slot (table/floor, plank, left pad, right pad, palm, objects). shard_<i>.json: per world
    the body ("soft": shape, material, caps, mesh sizes), its scene, families, events (a grasp lists the pinned
    vertices), guards, quality."""
    X = res["x"]
    W = len(res["specs"])
    offs = np.cumsum([0] + [wd["particles"][1] for wd in res["worlds"]])
    C = np.stack([X[:, wd["particles"][0]:wd["particles"][0] + wd["particles"][1]].mean(1) for wd in res["worlds"]], 1).astype(np.float32)
    D = np.concatenate([X[:, wd["particles"][0]:wd["particles"][0] + wd["particles"][1]] - C[:, w:w + 1]
                        for w, wd in enumerate(res["worlds"])], 1)
    rest, tets, tris, toff, soff, mass = [], [], [], [0], [0], []
    for wd in res["worlds"]:
        v, t = wd["soft"].mesh()
        S = soft_mesh.surface(t)
        rest.append(v)
        tets.append(t)
        tris.append(S)
        toff.append(toff[-1] + len(t))
        soff.append(soff[-1] + len(S))
        p0, n = wd["particles"]
        mass.append(res["mass"][p0:p0 + n])
    tmp = out / f"shard_{i:05d}.tmp.npz"
    np.savez_compressed(tmp, cloth_c=C, cloth_d=D.astype(np.float32), cloth_off=offs.astype(np.int64),
                        grip_q=res["grip"].astype(np.float32), grip_path=res["path"].astype(np.float32),
                        grip_w=res["grip_w"], grip_w_path=res["path_w"], grasp_F=res["grasp_F"], contact_F=res["contact_F"],
                        slot_F=res["slot_F"],
                        soft_rest=np.concatenate(rest).astype(np.float32), soft_tets=np.concatenate(tets).astype(np.int32),
                        tet_off=np.array(toff, np.int64), soft_tris=np.concatenate(tris).astype(np.int32),
                        tri_off=np.array(soff, np.int64), soft_mass=np.concatenate(mass).astype(np.float32))
    meta = {"seed": seed, "kind": "soft", "fps": FPS, "substeps": res["substeps"], "collide_every": res["collide_every"],
            "settle_frames": res["settle"],
            "events_on": "the whole timeline (recorded frame = frame - settle_frames); grip_path too",
            "contact_kinds": ["table", "gripper", "obstacle"],
            "slot_kinds": ["table/floor", "plank", "left pad", "right pad", "palm", "objects"],
            "gripper": {"origin": "the pads' fingertip midpoint", "yaw_only": True, "finger_half": FINGER_HALF,
                        "palm_half": st.PALM_HALF,
                        "width": "inner opening between the pads (m); -0.012: both pads in one place (closed finger)"},
            "worlds": []}
    qc = quality(res)
    for w, (sp, inf) in enumerate(zip(res["specs"], res["infos"])):
        meta["worlds"].append({"qc": qc[w], "soft": sp.soft.meta(), "init": sp.init, "pose": list(sp.pose),
                               "centre": inf["centre"], "rot": inf["rot"], "height": sp.height, "vel": list(sp.vel),
                               "spin": list(sp.spin), "table_mu": sp.table_mu, "ramp": sp.ramp, "obstacles": inf["obstacles"],
                               "families": res["plans"][w], "events": [list(e) for e in res["events"][w]],
                               "guards": res["guards"][w], "seed": sp.seed})
    (out / f"shard_{i:05d}.json").write_text(json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, out / f"shard_{i:05d}.npz")
    return qc


# ---------------------------------------------------------------------------------------------------------
# loading (the cloth episode's interface)
# ---------------------------------------------------------------------------------------------------------
class SoftEpisode:
    """One world of a soft shard with lsim.train.cloth_windows.ClothEpisode's fields: cc, cd, grip, grasp_F, contact_F,
    n_frames, rest, tris, edges, held, x(frames), ok() -- and grip_w, tets, mass, slots. ClothEpisode itself cannot
    read these: it rebuilds the mesh from the cloth's grid (nx, ny, h) and weighs it by areal density (w); a soft body
    gives its mesh (soft_rest / soft_tets / soft_tris) and lumped masses instead."""

    def __init__(self, npz, meta, w):
        self.w, self.meta = w, meta
        wd = meta["worlds"][w]
        self.wd, self.c = wd, wd["soft"]
        o0, o1 = int(npz["cloth_off"][w]), int(npz["cloth_off"][w + 1])
        self.cc = npz["cloth_c"][:, w]
        self.cd = npz["cloth_d"][:, o0:o1]
        self.grip = npz["grip_q"][:, w]
        self.grip_w = npz["grip_w"][:, w]
        self.grasp_F = npz["grasp_F"][:, w]
        self.contact_F = npz["contact_F"][:, w]
        self.n_frames = self.cc.shape[0]
        self.rest = npz["soft_rest"][o0:o1]
        self.mass = npz["soft_mass"][o0:o1]
        self.tets = npz["soft_tets"][npz["tet_off"][w]:npz["tet_off"][w + 1]]
        self.tris = npz["soft_tris"][npz["tri_off"][w]:npz["tri_off"][w + 1]]
        e = np.concatenate([self.tets[:, [a, b]] for a in range(4) for b in range(a + 1, 4)])
        self.edges = np.unique(np.sort(e, axis=1), axis=0)
        settle = int(meta["settle_frames"])
        V = o1 - o0
        held = np.zeros((self.n_frames, V), bool)
        cur = np.zeros(V, bool)
        ev = sorted(wd["events"], key=lambda e: e[0])
        k = 0
        for j in range(self.n_frames):
            while k < len(ev) and ev[k][0] <= settle + j:
                e = ev[k]
                cur = np.zeros(V, bool)
                if e[1] == "grasp" and len(e) > 2 and len(e[2]):
                    cur[np.asarray(e[2], int)] = True
                k += 1
            held[j] = cur
        self.held = held
        self.qc = wd.get("qc", {})
        self.slots = {k: self.c[k] for k in ("rho", "E", "nu", "mu")}
        self.slots["zeta"] = self.c.get("zeta_effective", self.c["zeta"])

    def x(self, frames):
        frames = np.clip(np.asarray(frames), 0, self.n_frames - 1)
        return self.cd[frames].astype(np.float32) + self.cc[frames][:, None, :].astype(np.float32)

    def ok(self):
        q = self.qc
        return not q.get("exploded", False) and not q.get("inverted", False)


def load_soft_shard(path, only_ok=True):
    from pathlib import Path as P
    path = P(path)
    npz = dict(np.load(path.with_suffix(".npz")))
    meta = json.loads(path.with_suffix(".json").read_text())
    eps = [SoftEpisode(npz, meta, w) for w in range(len(meta["worlds"]))]
    return [e for e in eps if e.ok()] if only_ok else eps
