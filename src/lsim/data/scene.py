"""Scene sampling and construction: one Newton world per scene (Kamino accepts different
objects in every world of a batch).

World layout: a static table whose top is at z = 0 (1.2 x 0.8 m); 1-6 scanned objects; an optional
hinged part (drawer or lid); a kinematic gripper parked out of reach. Objects' mass, centre of mass
and inertia are computed here (base build + hidden dense spheres) and given to the body
explicitly, so the labels are exact (PHASE1_AMENDMENT.md section 2).
"""
from dataclasses import dataclass, field

import numpy as np
import trimesh
import warp as wp
import newton

from lsim.data import library
from lsim.data.params import HingeParams, ObjectParams, sample_hinge, sample_object
from lsim.teacher.newton_teacher import I, joint_limit_cfg, new_builder, shape_cfg

TABLE_HALF = (0.6, 0.5)
WORKSPACE = (0.38, 0.32)                         # object centres stay inside +-x, +-y
MASS_RANGE = (0.005, 5.0)                        # kg: household objects; densities rescaled into it
GRIPPER_MU = 0.6                                 # rubber pads; recorded per episode
GRIPPER_MASS = 0.5                               # kg
HANDLE_OFF = 0.05                                # drawer handle bar stand-off from the drawer front
CAB = {"W": 0.30, "D": 0.24, "H": 0.10, "t": 0.01}  # hinged furniture dimensions
PARK = (0.0, -0.75, 0.35)                        # gripper start, off the table and above it
RAMP = {"L": 0.5, "W": 0.3, "T": 0.012}          # a plank leaning on the table: slope 5-30 deg
TABLE_MAT, FLOOR_MAT = (0.5, 0.3), (0.6, 0.2)    # (friction, restitution) of the table top and the floor


def ramp_frame(r):
    """Origin (midpoint of the plank's low bottom edge, on the table) and the plank's axes in
    world: along the slope uphill, across it, and its top surface's normal."""
    th, yaw = r["angle"], r["yaw"]
    d, lat, z = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0]), np.array([0, 0, 1.0])
    return np.array([r["low"][0], r["low"][1], 0.0]), np.cos(th) * d + np.sin(th) * z, lat, -np.sin(th) * d + np.cos(th) * z


def sample_ramp(rng):
    """A plank at 5-30 deg whose whole footprint lies on the table."""
    for _ in range(100):
        th, yaw = float(np.radians(rng.uniform(5.0, 30.0))), float(rng.uniform(-np.pi, np.pi))
        mid = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.25, 0.2)])
        d = np.array([np.cos(yaw), np.sin(yaw)])
        low = mid - 0.5 * RAMP["L"] * np.cos(th) * d
        corners = [mid + sx * 0.5 * RAMP["L"] * np.cos(th) * d + sy * 0.5 * RAMP["W"] * np.array([-d[1], d[0]])
                   for sx in (-1, 1) for sy in (-1, 1)]
        if all(abs(c[0]) < TABLE_HALF[0] - 0.03 and abs(c[1]) < TABLE_HALF[1] - 0.03 for c in corners):
            return {"angle": th, "yaw": yaw, "low": low.tolist(), "mid": mid.tolist(),
                    "mu": float(np.exp(rng.uniform(np.log(0.2), np.log(0.8)))), "restitution": float(rng.uniform(0.05, 0.5))}
    return None



@dataclass
class ObjSpec:
    name: str
    params: ObjectParams
    xy: tuple
    yaw: float
    on_top_of: int = -1                          # index of the object it is stacked on, or -1
    drop_height: float = 0.0                     # > 0: lifted and released after settling
    on_ramp: float = -1.0                        # >= 0: moved onto the ramp at rest after settling, this far up it
    com_shift: tuple = (0.0, 0.0, 0.0)           # identifiability gate only: move the CoM, keep m and I
    mass_scale: float = 1.0                      # identifiability gate only: scale the mass (and I)


@dataclass
class SceneSpec:
    seed: int
    objects: list
    hinge: HingeParams | None = None
    hinge_xy: tuple = (0.0, 0.35)
    families: list = field(default_factory=list)
    ramp: dict | None = None


def mass_properties(asset, p: ObjectParams):
    """Mass, centre of mass (object frame) and inertia about that CoM, built from the scan
    (solid: rho V; shell: rho A t) plus hidden dense spheres."""
    if p.build == "solid":
        m0, c0, I0 = p.density * float(asset["volume"]), np.zeros(3), np.asarray(asset["inertia_solid_unit"], float)
    else:
        m0, c0, I0 = p.density * float(asset["area"]) * p.wall, np.asarray(asset["com_shell"], float), \
            np.asarray(asset["inertia_shell_unit"], float)
    parts = [(m0, c0, I0 * m0)]
    for xyz, r, rho in p.regions:
        mr = rho * 4.0 / 3.0 * np.pi * r ** 3
        parts.append((mr, np.asarray(xyz, float), np.eye(3) * 0.4 * mr * r * r))
    M = sum(m for m, _, _ in parts)
    scale = float(np.clip(M, *MASS_RANGE) / M)
    if scale != 1.0:                                   # rescale densities into the household range
        parts = [(m * scale, c, Ic * scale) for m, c, Ic in parts]
        p.density *= scale
        p.regions = [(xyz, r, rho * scale) for xyz, r, rho in p.regions]
        M = sum(m for m, _, _ in parts)
    C = sum(m * c for m, c, _ in parts) / M
    Itot = np.zeros((3, 3))
    for m, c, Ic in parts:                        # parallel-axis theorem to the composite CoM
        d = c - C
        Itot += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return float(M), C, Itot


def add_convex_parts(b, solver, body, asset, mu, restitution):
    """Collide an object as its convex decomposition (CoACD parts, lsim.data.assets). Kamino makes
    no contacts between two moving triangle meshes (a box-shaped mesh fell through another;
    the same shapes as convex hulls stack with a support force of exactly m g)."""
    vo = fo = 0
    for nv, nf in zip(asset["convex_vcount"], asset["convex_fcount"]):
        pv = asset["convex_v"][vo:vo + nv]
        pf = asset["convex_f"][fo:fo + nf] - vo                # stored with offsets into convex_v
        b.add_shape_convex_hull(body, mesh=newton.Mesh(pv, pf.flatten()),
                                cfg=shape_cfg(solver, mu, restitution=restitution, density=0.0))
        vo += nv
        fo += nf


def interior_points(asset, n=200, seed=0):
    mesh = trimesh.Trimesh(asset["collide_v"], asset["collide_f"], process=False)
    pts = trimesh.sample.volume_mesh(mesh, n * 4)
    rng = np.random.default_rng(seed)
    return pts[rng.permutation(len(pts))[:n]] if len(pts) else np.zeros((0, 3))


def resting_z(asset, yaw=0.0):
    """Height of the object-frame origin when its lowest collision vertex touches z = 0."""
    return float(-asset["collide_v"][:, 2].min())


def sample_scene(rng, names, seed, n_obj=(1, 6), p_hinge=0.4, p_stack=0.2, p_drop=0.7, p_ramp=0.0):
    """p_ramp > 0 (later datasets): a plank leaning on the table with 1-2 objects started at rest
    on it. With p_ramp = 0 no random numbers are drawn for it, so v0's scenes regenerate."""
    n = int(rng.integers(n_obj[0], n_obj[1] + 1))
    hinge = sample_hinge(rng, "drawer" if rng.random() < 0.6 else "lid") if rng.random() < p_hinge else None
    hinge_xy = (float(rng.uniform(-0.2, 0.2)), 0.35)
    # objects stay clear of a fully opened drawer and its handle (an open drawer crushed one)
    y_clear = hinge_xy[1] - CAB["D"] / 2 - 0.15 - HANDLE_OFF - 0.02 if hinge else WORKSPACE[1]
    objs, placed = [], []
    ramp = sample_ramp(rng) if (p_ramp > 0 and rng.random() < p_ramp and hinge is None) else None
    if ramp is not None:                                              # its footprint is taken
        placed.append((tuple(ramp["mid"]), 0.5 * float(np.hypot(RAMP["L"], RAMP["W"]))))
    for _ in range(n):
        name = names[int(rng.integers(len(names)))]
        a = library.load(name)
        rad = 0.5 * float(np.hypot(*a["extents"][:2])) + 0.01
        ymax = min(WORKSPACE[1], y_clear - rad)
        if ymax <= -WORKSPACE[1]:
            continue
        for _try in range(60):
            xy = (float(rng.uniform(-WORKSPACE[0], WORKSPACE[0])), float(rng.uniform(-WORKSPACE[1], ymax)))
            if all(np.hypot(xy[0] - q[0], xy[1] - q[1]) > rad + r for q, r in placed):
                break
        else:
            continue
        placed.append((xy, rad))
        objs.append(ObjSpec(name, sample_object(rng, interior_points(a, seed=seed), a["extents"]), xy,
                            float(rng.uniform(-np.pi, np.pi))))
    if len(objs) >= 2 and rng.random() < p_stack:                     # stack/load family
        base, top = sorted(rng.choice(len(objs), 2, replace=False))       # base is built first
        objs[top].on_top_of = int(base)
        objs[top].xy = objs[base].xy
    # pick-lift-drop analogue, decided PER OBJECT: restitution is only readable through impacts
    # (a drop identifies it 96% of the time, other interactions rarely), and 1-3 drops per scene
    # reached 38% of objects, p 0.5 per object 45%. Stacked objects are never lifted (lifting a
    # base drove it into the object resting on it).
    stacked = {o.on_top_of for o in objs if o.on_top_of >= 0} | {k for k, o in enumerate(objs) if o.on_top_of >= 0}
    if ramp is not None:                                              # the 1-2 smallest free objects that fit
        fit = sorted((float(max(library.load(o.name)["extents"][:2])), k) for k, o in enumerate(objs)
                     if k not in stacked and max(library.load(o.name)["extents"][:2]) < 0.6 * RAMP["W"])
        for _, k in fit[:int(rng.integers(1, 3))]:
            objs[k].on_ramp = float(rng.uniform(0.45, 0.8) * RAMP["L"])
    for k, o in enumerate(objs):
        if k not in stacked and o.on_ramp < 0 and rng.random() < p_drop:
            o.drop_height = float(rng.uniform(0.05, 0.25))
    return SceneSpec(seed, objs, hinge, hinge_xy, ramp=ramp)


def _box(b, solver, body, center, half, mu, e=0.3, density=0.0, **kw):
    b.add_shape_box(body, xform=wp.transform(wp.vec3(*center), I), hx=half[0], hy=half[1], hz=half[2],
                    cfg=shape_cfg(solver, mu, restitution=e, density=density, **kw))


def build_world(spec: SceneSpec, solver="kamino"):
    """Newton builder for one scene, plus the bookkeeping the episode runner needs."""
    b = new_builder(solver)
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), I), hx=TABLE_HALF[0],
                    hy=TABLE_HALF[1], hz=0.05, cfg=shape_cfg(solver, TABLE_MAT[0], restitution=TABLE_MAT[1]))
    b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.80), I), hx=10.0, hy=10.0, hz=0.05,
                    cfg=shape_cfg(solver, FLOOR_MAT[0], restitution=FLOOR_MAT[1]))   # floor, 75 cm below (a can rolled off a 4 m one)
    info = {"objects": [], "labels": []}
    if spec.ramp is not None:                                         # static plank leaning on the table
        o, ax, lat, n = ramp_frame(spec.ramp)
        c = o + 0.5 * RAMP["L"] * ax + 0.5 * RAMP["T"] * n
        q = wp.quat_from_matrix(wp.mat33(*np.stack([ax, lat, n], 1).flatten()))
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*c), q), hx=0.5 * RAMP["L"], hy=0.5 * RAMP["W"],
                        hz=0.5 * RAMP["T"], cfg=shape_cfg(solver, spec.ramp["mu"], restitution=spec.ramp["restitution"]))
    tops = {}
    for k, o in enumerate(spec.objects):
        a = library.load(o.name)
        M, C, Itot = mass_properties(a, o.params)
        M, Itot, C = M * o.mass_scale, Itot * o.mass_scale, C + np.asarray(o.com_shift, float)
        z = resting_z(a)
        if o.on_top_of >= 0 and o.on_top_of in tops:
            z += tops[o.on_top_of] + 0.002
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), o.yaw)
        body = b.add_body(xform=wp.transform(wp.vec3(o.xy[0], o.xy[1], z), q), mass=M, com=wp.vec3(*C),
                          inertia=wp.mat33(*Itot.flatten()), lock_inertia=True, label=f"obj{k}")
        add_convex_parts(b, solver, body, a, o.params.mu, o.params.restitution)
        tops[k] = z + float(a["collide_v"][:, 2].max())
        info["objects"].append(body)
        info["labels"].append({"name": o.name, "mass": M, "com": C.tolist(), "inertia": Itot.tolist(),
                               "mu": o.params.mu, "restitution": o.params.restitution, "build": o.params.build,
                               "density": o.params.density, "wall": o.params.wall, "regions": o.params.regions,
                               "on_top_of": o.on_top_of, "drop_height": o.drop_height, "on_ramp": o.on_ramp})
    if spec.hinge is not None:
        info["hinge"] = _hinge(b, solver, spec.hinge, spec.hinge_xy)
    info["gripper"], info["gripper_joint"] = _gripper(b, solver)
    return b, info


def _gripper(b, solver):
    """A parallel-jaw shape with its fingers closed (palm + two touching fingers: a 3 cm block).
    Open fingers let objects narrower than the gap pass between them (seen in testing).
    A dynamic body that tracks its planned path through a force-limited controller (episode.py):
    a kinematic gripper has infinite strength, and wedging an object against the table then gave
    frames of up to 303 kN. Its recorded path is still the action."""
    m = GRIPPER_MASS                                # explicit mass: shapes carry none
    Ig = np.diag([m / 12 * (0.04 ** 2 + 0.10 ** 2), m / 12 * (0.10 ** 2 + 0.10 ** 2), m / 12 * (0.10 ** 2 + 0.04 ** 2)])
    g = b.add_body(xform=wp.transform(wp.vec3(*PARK), I), mass=m, inertia=wp.mat33(*Ig.flatten()),
                   lock_inertia=True, label="gripper")
    _box(b, solver, g, (0.0, 0.0, 0.03), (0.05, 0.02, 0.015), GRIPPER_MU)
    for sx in (-1, 1):
        _box(b, solver, g, (sx * 0.0078, 0.0, -0.02), (0.0075, 0.015, 0.035), GRIPPER_MU)
    j = len(b.joint_type) - 1                     # add_body created its free joint last
    return g, j


def _hinge(b, solver, h: HingeParams, xy):
    """Drawer: a static cabinet with a drawer on a prismatic joint pulled toward -y.
    Lid: a static open box with a lid on a revolute joint along its back edge."""
    x0, y0 = xy
    W, D, H, t = CAB["W"], CAB["D"], CAB["H"], CAB["t"]
    if h.kind == "drawer":
        for c, hh in (((x0, y0, H + t / 2), (W / 2 + t, D / 2, t / 2)),                     # top
                      ((x0 - W / 2 - t / 2, y0, H / 2), (t / 2, D / 2, H / 2)),           # sides
                      ((x0 + W / 2 + t / 2, y0, H / 2), (t / 2, D / 2, H / 2)),
                      ((x0, y0 + D / 2 + t / 2, H / 2), (W / 2 + t, t / 2, H / 2))):        # back
            _box(b, solver, -1, c, hh, h.mu, h.restitution)
        link = b.add_link(xform=wp.transform(wp.vec3(x0, y0, 0.0), I), label="drawer")
        w, d, hgt = W - 0.004, D - 0.004, H - 0.01
        for c, hh in (((0, 0, t / 2), (w / 2, d / 2, t / 2)), ((-w / 2 + t / 2, 0, hgt / 2), (t / 2, d / 2, hgt / 2)),
                      ((w / 2 - t / 2, 0, hgt / 2), (t / 2, d / 2, hgt / 2)), ((0, d / 2 - t / 2, hgt / 2), (w / 2, t / 2, hgt / 2)),
                      ((0, -d / 2 + t / 2, hgt / 2), (w / 2, t / 2, hgt / 2)),
                      # handle: a bar standing 5 cm off the front on two posts, so a finger fits behind it
                      ((0, -d / 2 - HANDLE_OFF, hgt / 2), (0.05, 0.005, 0.008)),
                      ((-0.045, -d / 2 - HANDLE_OFF / 2, hgt / 2), (0.005, HANDLE_OFF / 2, 0.008)),
                      ((0.045, -d / 2 - HANDLE_OFF / 2, hgt / 2), (0.005, HANDLE_OFF / 2, 0.008))):
            _box(b, solver, link, c, hh, h.mu, h.restitution, density=600.0)
        j = b.add_joint_prismatic(parent=-1, child=link, parent_xform=wp.transform(wp.vec3(x0, y0, 0.0005), I),
                                  axis=wp.vec3(0.0, -1.0, 0.0), damping=h.damping, limit_lower=0.0,
                                  limit_upper=0.8 * D, **joint_limit_cfg(solver))
    else:
        for c, hh in (((x0, y0, t / 2), (W / 2, D / 2, t / 2)), ((x0 - W / 2 + t / 2, y0, H / 2), (t / 2, D / 2, H / 2)),
                      ((x0 + W / 2 - t / 2, y0, H / 2), (t / 2, D / 2, H / 2)), ((x0, y0 + D / 2 - t / 2, H / 2), (W / 2, t / 2, H / 2)),
                      ((x0, y0 - D / 2 + t / 2, H / 2), (W / 2, t / 2, H / 2))):
            _box(b, solver, -1, c, hh, h.mu, h.restitution)
        hinge_p = wp.vec3(x0, y0 + D / 2, H + 0.006)
        link = b.add_link(xform=wp.transform(hinge_p, I), label="lid")
        _box(b, solver, link, (0.0, -D / 2, 0.0), (W / 2 + 0.005, D / 2 + 0.005, 0.005), h.mu, h.restitution, density=600.0)
        _box(b, solver, link, (0.0, -D - 0.015, 0.0), (0.04, 0.01, 0.005), h.mu, h.restitution, density=600.0)  # handle tab
        j = b.add_joint_revolute(parent=-1, child=link, parent_xform=wp.transform(hinge_p, I),
                                 axis=wp.vec3(-1.0, 0.0, 0.0), damping=h.damping, limit_lower=0.0,
                                 limit_upper=float(np.radians(110.0)), **joint_limit_cfg(solver))
    b.add_articulation([j])
    return {"kind": h.kind, "link": link, "joint": j, "damping": h.damping, "mu": h.mu, "restitution": h.restitution,
            "xy": [x0, y0]}
