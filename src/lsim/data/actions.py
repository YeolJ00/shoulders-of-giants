"""Action families (PHASE1_AMENDMENT.md section 3) as gripper keyframe paths, planned per world
from the current scene. The gripper tracks its path with a force-capped PD controller; its
recorded path is the action, and the teacher records the contact force it exerts (episode.py).

Gripper frame: two touching fingers (a 3.1 x 3 x 7 cm block) under a 10 x 4 x 3 cm palm; finger
bottoms at local z = -0.055 m. Pushing uses the fingers' faces normal to local +y, so the gripper
yaw turns local +y onto the push direction.
"""
import numpy as np

SAFE_Z = 0.30
TRANSIT_V = 1.0                                   # m/s; at 0.6 with a fixed 0.30 m transit height a
DESCEND_V = 0.7                                   # family took 2.3-2.9 s and only two fit in 6 s
TRANSIT_MIN, TRANSIT_CLEAR = 0.15, 0.10           # transit height: tallest object top + 10 cm
YAW_RATE = 6.0                                    # rad/s the planner allows the gripper to turn
FINGER_BOTTOM = 0.055
FAMILY_P = {"push": 0.6, "off_centre": 0.4, "tilt_release": 0.3, "strike": 0.3, "collision": 0.3,
            "hinge": 0.3, "sweep": 0.15, "random": 0.2}          # fractions of episodes (amendment)
POKE_P = 0.3
TARGETED = ("push", "off_centre", "tilt_release", "strike", "collision", "sweep")
BAND = 0.035                                      # the finger block spans +-3.5 cm about the push height


def yaw_for(d):
    return float(np.arctan2(-d[0], d[1]))                         # local +y -> direction d


class Path:
    def __init__(self, start, yaw=0.0):
        self.k = [(0.0, np.asarray(start, float), float(yaw))]

    @property
    def t(self):
        return self.k[-1][0]

    @property
    def p(self):
        return self.k[-1][1]

    @property
    def yaw(self):
        return self.k[-1][2]

    def to(self, p, v, yaw=None):
        p = np.asarray(p, float)
        dt = max(float(np.linalg.norm(p - self.p)) / max(v, 1e-3), 1.0 / 30.0)
        if yaw is not None:                                       # turning sets a floor on the duration
            dt = max(dt, abs(float(yaw) - self.yaw) / YAW_RATE)
        self.k.append((self.t + dt, p, self.yaw if yaw is None else float(yaw)))

    def turn(self, yaw, dt=0.3):
        self.k.append((self.t + dt, self.p.copy(), float(yaw)))

    def wait(self, dt):
        self.k.append((self.t + dt, self.p.copy(), self.k[-1][2]))

    def sample(self, times):
        """Position, yaw, linear and yaw velocity at the given times (piecewise linear)."""
        T = np.array([k[0] for k in self.k]); P = np.array([k[1] for k in self.k]); Y = np.unwrap([k[2] for k in self.k])
        pos = np.stack([np.interp(times, T, P[:, i]) for i in range(3)], -1)
        yaw = np.interp(times, T, Y)
        seg = np.clip(np.searchsorted(T, times, side="right") - 1, 0, len(T) - 2)
        dT = np.maximum(T[seg + 1] - T[seg], 1e-9)
        vel = np.where((times < T[-1])[:, None], (P[seg + 1] - P[seg]) / dT[:, None], 0.0)
        wz = np.where(times < T[-1], (Y[seg + 1] - Y[seg]) / dT, 0.0)
        return pos, yaw, vel, wz


def _support(verts_world, center, d):
    """Distance from the object's centre to its surface along -d (conservative, all vertices)."""
    return float(np.max((verts_world[:, :2] - center[:2]) @ (-d)))


def _band(obj, hc):
    """Surface points the finger block can reach at push height hc (all of them if none): with the
    support taken over the whole object, a push near the top of a tapered object started behind
    its widest point and stopped short of the surface -- 36-44% of off-centre pushes and tilts
    never touched their target."""
    s = obj["surf"]
    m = np.abs(s[:, 2] - hc) <= BAND
    return s[m] if m.sum() >= 3 else s


def push_height(obj, h_frac):
    return obj["bottom"] + h_frac * (obj["top"] - obj["bottom"])


PALM = (0.05, 0.02, 0.015, 0.045)                 # half x, half y, bottom and top above the origin
FINGERS = (0.0155, 0.015, -0.055)                 # half x, half y, bottom


def clear_column(objs, xy, z, yaw, margin=0.01):
    """True if descending the gripper to height z at xy hits no object vertex (the palm and the
    finger block are checked as boxes in the gripper frame, expanded by `margin`)."""
    c, s_ = np.cos(yaw), np.sin(yaw)
    for o in objs:
        v = o["verts_all"]
        dx, dy = v[:, 0] - xy[0], v[:, 1] - xy[1]
        lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy           # into the gripper frame
        palm = (np.abs(lx) < PALM[0] + margin) & (np.abs(ly) < PALM[1] + margin) & (v[:, 2] > z + PALM[2] - margin)
        fing = (np.abs(lx) < FINGERS[0] + margin) & (np.abs(ly) < FINGERS[1] + margin) & (v[:, 2] > z + FINGERS[2] - margin)
        if palm.any() or fing.any():
            return False
    return True


def transit_height(objs, z_target=0.0):
    """Above every object's current top (finger bottoms clear it by 4.5 cm) and the target."""
    top = max([o["top"] for o in objs], default=0.0)
    return max(TRANSIT_MIN, top + TRANSIT_CLEAR, z_target + 0.02)


def _near_yaw(yaw, cur):
    """The gripper is symmetric under a half turn: of yaw and yaw + k pi, the one nearest cur."""
    return float(yaw + np.pi * np.round((cur - yaw) / np.pi))


def _approach(path, xy, z, yaw, tz):
    """Rise to the transit height, cross to above xy while turning, descend."""
    if path.p[2] < tz:
        path.to([path.p[0], path.p[1], tz], TRANSIT_V)
    path.to([xy[0], xy[1], tz], TRANSIT_V, yaw=_near_yaw(yaw, path.yaw))
    path.to([xy[0], xy[1], z], DESCEND_V)


def _retreat(path, d, tz, v=0.6):
    p = path.p - 0.03 * np.array([d[0], d[1], 0.0])
    path.to([p[0], p[1], p[2]], v)
    path.to([p[0], p[1], max(tz, p[2])], TRANSIT_V)


def _push(path, rng, obj, d, h_frac, speed, dist, lateral=0.0, lead=0.02, objs=(), tries=8):
    """Approach behind the object along d and push. If the descent column is blocked (another
    object, or the target's own overhang), try other directions; return None if all are."""
    c = obj["center"]
    hc = max(push_height(obj, h_frac), obj["bottom"] + 0.004)
    z = max(hc + 0.02, FINGER_BOTTOM + 0.003)
    band = _band(obj, hc)
    for k in range(tries):
        if k:
            a = rng.uniform(-np.pi, np.pi)
            d = np.array([np.cos(a), np.sin(a)])
        perp = np.array([-d[1], d[0]])
        start = c[:2] - d * (_support(band, c, d) + 0.0155 + 0.012 + lead) + lateral * perp
        if clear_column(objs, start, z, yaw_for(d)):
            break
    else:
        return None
    _approach(path, start, z, yaw_for(d), transit_height(objs))
    end = start + d * (lead + dist)
    path.to([end[0], end[1], z], speed)
    return hc, d


def sample_families(rng, n_objs, hinge):
    """Which action families this episode gets (amendment fractions), at most four, in order.
    Families that do not fit before the episode ends are dropped by the runner."""
    fams = [f for f, p in FAMILY_P.items() if rng.random() < p and (f != "hinge" or hinge) and
            (f != "collision" or n_objs >= 2) and (f in ("random", "hinge") or n_objs)]
    rng.shuffle(fams)
    fams = fams[:4] or (["random"] if not n_objs else ["push"])
    if n_objs and not any(f in TARGETED for f in fams):     # every episode touches an object on purpose
        fams = ["push"] + fams[:2]
    return fams


def fill_family(rng):
    """A targeted family for an object nothing has acted on yet, drawn in the amendment's ratios."""
    p = np.array([FAMILY_P[f] for f in TARGETED])
    return str(rng.choice(TARGETED, p=p / p.sum()))


def plan_family(rng, f, path, objs, hinge, counts=None, obstacles=()):
    """Append one family's keyframes to `path`, planned from the CURRENT object poses (planning
    every family up front let later approaches descend onto objects an earlier push had moved).
    Targets the object acted on least so far (`counts`; at random a third of objects were never
    touched). Returns the family's log entry, or None if it could not be placed."""
    t0 = path.t
    on_table = [k for k, ob in enumerate(objs) if ob["center"][2] > -0.02]   # not fallen to the floor
    clear = list(objs) + list(obstacles)                           # what transits and descents must clear
    if f in TARGETED and on_table:
        c = np.zeros(len(objs)) if counts is None else np.asarray(counts, float)
        target = int(rng.choice([k for k in on_table if c[k] == min(c[j] for j in on_table)]))
    elif f in TARGETED:
        return None
    else:
        target = -1
    o = objs[target] if target >= 0 else None
    ang = rng.uniform(-np.pi, np.pi)
    d = np.array([np.cos(ang), np.sin(ang)])
    if hinge is not None and o is not None:                        # never push toward the cabinet
        to_cab = np.asarray(hinge["xy"]) - o["center"][:2]
        if d @ to_cab > 0.3 * np.linalg.norm(to_cab):
            d = -d
    info = {"family": f, "t_start": t0, "target": target}
    if f in ("push", "off_centre", "tilt_release", "strike", "collision", "sweep"):
        if f == "push":
            args = (rng.uniform(0.2, 0.6), np.exp(rng.uniform(np.log(0.1), np.log(0.8))), rng.uniform(0.03, 0.20))
            kw, v_out = {}, 0.6
        elif f == "off_centre":
            h_frac = rng.uniform(0.7, 0.95)
            w = float(np.ptp(_band(o, push_height(o, h_frac))[:, :2] @ np.array([-d[1], d[0]])))
            args = (h_frac, rng.uniform(0.05, 0.3), rng.uniform(0.02, 0.08))
            kw, v_out = {"lateral": rng.uniform(-0.4, 0.4) * w}, 0.6
        elif f == "tilt_release":
            args = (0.9, rng.uniform(0.03, 0.1), (o["top"] - o["bottom"]) * np.sin(np.radians(rng.uniform(10.0, 40.0))))
            kw, v_out = {}, 1.0
        elif f == "strike":
            args = (rng.uniform(0.3, 0.6), rng.uniform(0.5, 1.5), rng.uniform(0.02, 0.04))
            kw, v_out = {"lead": 0.12}, 1.0
        elif f == "sweep":
            # push the target through to its nearest neighbour and on past it, driving both: the
            # sweep along a fixed line at the table's side touched 0-2 objects by chance
            dist_xy = [np.linalg.norm(objs[k]["center"][:2] - o["center"][:2]) for k in range(len(objs))]
            near = [k for k in np.argsort(dist_xy) if k != target and 0.05 < dist_xy[k] < 0.30
                    and objs[k]["center"][2] > -0.02]
            far = 0.0
            if near:
                other = int(near[0]); info["other"] = other
                v = objs[other]["center"][:2] - o["center"][:2]
                d = v / max(np.linalg.norm(v), 1e-6)
                far = float(np.linalg.norm(v)) + _support(objs[other]["verts_all"], objs[other]["center"], -d)
            args = (0.3, rng.uniform(0.2, 0.5), max(far, 0.0) + rng.uniform(0.03, 0.10))
            kw, v_out = {"tries": 1 if near else 8}, 0.6
        else:
            # push A toward its nearest neighbour B and let go 1-3 cm short of it, so A arrives
            # free and the two exchange momentum (driven into B, A acts as infinitely heavy and
            # the mass ratio never enters; at a fixed 4-10 cm push A rarely reached B at all)
            dist_xy = [np.linalg.norm(objs[k]["center"][:2] - o["center"][:2]) for k in range(len(objs))]
            near = [k for k in np.argsort(dist_xy) if k != target and dist_xy[k] > 0.05
                    and objs[k]["center"][2] > -0.02]
            if not near:
                return None
            other = int(near[0])
            v = objs[other]["center"][:2] - o["center"][:2]
            d = v / max(np.linalg.norm(v), 1e-6)
            info["other"] = other
            gap = (np.linalg.norm(v) - _support(o["verts_all"], o["center"], -d)
                   - _support(objs[other]["verts_all"], objs[other]["center"], d))
            args = (0.5, rng.uniform(0.4, 1.0), float(np.clip(gap - rng.uniform(0.01, 0.03), 0.02, 0.30)))
            kw, v_out = {"tries": 1}, 1.0
        r = _push(path, rng, o, d, *args, objs=clear, **kw)
        if r is None:
            return None
        info["h"], d = r
        _retreat(path, d, transit_height(clear), v=v_out)
    elif f == "random":                                            # 3-5 waypoints at 0.1-0.5 m/s took ~6 s
        for _ in range(int(rng.integers(2, 5))):
            path.to([rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.25), rng.uniform(0.12, 0.25)], rng.uniform(0.2, 0.8),
                    yaw=rng.uniform(-np.pi, np.pi))
    elif f == "hinge":
        info.update(_hinge_action(path, rng, hinge, transit_height(clear)))
    info["t_end"] = path.t
    path.wait(rng.uniform(0.05, 0.2))
    return info


def sample_pokes(rng, objs, t_end, scale=1.0):
    pokes = []
    if objs and rng.random() < POKE_P:
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(len(objs)))
            vtx = objs[k]["verts"][int(rng.integers(len(objs[k]["verts"])))]
            dirn = rng.normal(size=3); dirn[2] = abs(dirn[2]) * 0.5
            frames = int(rng.integers(1, 7))
            dv = rng.uniform(0.1, 1.0) * scale                      # velocity change it would cause, m/s
            fmag = objs[k]["mass"] * dv / (frames / 30.0)
            pokes.append({"t": float(rng.uniform(0.2, t_end - 0.3)), "frames": frames, "object": k, "dv": float(dv),
                          "point_local": (vtx - objs[k]["center"]).tolist(),
                          "force": (dirn / np.linalg.norm(dirn) * fmag).tolist()})
    return pokes


def sample_hinge_mode(rng, h):
    """Decided at episode start because some modes open the part before recording begins."""
    if h["kind"] == "drawer":
        return ("pull_open", 0.0) if rng.random() < 0.5 else ("push_shut", float(rng.uniform(0.06, 0.15)))
    return ("release" if rng.random() < 0.5 else "press"), float(np.radians(rng.uniform(20.0, 80.0)))


def _hinge_action(path, rng, h, tz):
    """Drawer: either hook the stand-off handle (gripper turned 90 deg so one finger sits behind
    the bar), pull it open and let go mid-motion; or start it open and push it shut, letting go
    mid-motion. Lid (a gripper cannot get under a 1 cm tab 10 cm above the table without
    pitching): start it open and either release it or press it down from above. Coasting and
    falling read joint damping and limits. Returns the info dict incl. the joint's start q."""
    from lsim.data.scene import CAB, HANDLE_OFF
    x0, y0 = h["xy"]
    hz = (CAB["H"] - 0.01) / 2
    mode, q0 = h["mode"], h["q0"]
    if h["kind"] == "drawer":
        if mode == "pull_open":
            hy = y0 - CAB["D"] / 2 - HANDLE_OFF                    # handle bar, drawer closed
            slot_y = hy + 0.005 + 0.007 + FINGERS[0]               # finger block in the slot behind the bar
            z = 0.078                                              # palm clears the drawer front, block spans the bar
            _approach(path, [x0, slot_y], z, np.pi / 2, max(tz, 0.16))
            pull = rng.uniform(0.05, 0.15)
            path.to([x0, slot_y - pull, z], rng.uniform(0.1, 0.4))  # the block hooks the bar
            path.to([x0, slot_y - pull, max(tz, 0.16)], 0.6)       # let go mid-motion
            return {"mode": "pull_open", "q0": 0.0, "pull": float(pull)}
        front = y0 - CAB["D"] / 2 - q0                             # drawer front face when open
        z = max(hz + 0.02, FINGER_BOTTOM + 0.003)
        d = np.array([0.0, 1.0])
        x_off = 0.10 if rng.random() < 0.5 else -0.10              # clear of the handle bar and posts
        start = [x0 + x_off, front - 0.012 - FINGERS[1] - 0.03]
        _approach(path, start, z, yaw_for(d), max(tz, 0.16))
        push = rng.uniform(0.3, 0.9) * q0
        path.to([start[0], start[1] + 0.03 + push, z], rng.uniform(0.1, 0.4))
        path.to([start[0], start[1] + push, max(tz, 0.16)], 0.8)   # let go mid-motion
        return {"mode": "push_shut", "q0": float(q0), "push": float(push)}
    if mode == "release":
        return {"mode": "release", "q0": q0}
    # press down on the open lid's front edge from above, then let go
    hinge_y, hinge_z = y0 + CAB["D"] / 2, CAB["H"] + 0.006
    r = CAB["D"] * 0.8
    ey, ez = hinge_y - r * np.cos(q0), hinge_z + r * np.sin(q0)
    zt = ez + FINGER_BOTTOM + 0.04
    _approach(path, [x0, ey], zt, 0.0, max(tz, zt + 0.02))
    path.to([x0, ey, ez + FINGER_BOTTOM - 0.01], rng.uniform(0.1, 0.3))
    path.to([x0, ey, max(tz, zt)], 0.8)
    return {"mode": "press", "q0": q0}
