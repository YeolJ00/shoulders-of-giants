"""L0 -- the rope teacher's own checks: the rope gate's closed forms run on the module that generates the data
(lsim.teacher.rope_teacher), plus generator-batch checks of what the gate never did (CLAUDE.md: "gate the module that
generates the data, on what the data does").

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l0_rope_teacher_check.py [check ...] [--objects gso|simready|both] [--pilot outputs/lsim/rope_pilot]

The gate (scripts/l0_rope_gate.py) ran its own `Sim` on one world; the generator runs RopeTeacher over a batch, with
air drag, per-world ropes and materials, the cloth teacher's kinematic gripper and pinned grasps. `TeacherSim` gives
RopeTeacher the gate's Sim interface, so the gate's checks run through it unchanged:

  stretch, catenary, bend, swing, slide, drop, cross, pile, gripper, stability, throughput   (the gate's)
  freefall   the gate's, with air drag off (the closed form is drag-free)
  airfall    a horizontal rope falling from 20 m reaches the cylinder's terminal speed sqrt(2 lam g / (rho Cd 2 r)) (5%)
  bounce     straight ropes (light / heavy, floppy / stiff) dropped flat from 10 cm onto the generator's table: no
             rebound (< 2 mm above where they come to rest), no node faster than 0.1 m/s upward after touching
  episodes   generator batches of 64 (the natural mix twice, coiled starts, pick-by-end and push): no rope exploded
             (a node moving 0.4 m in a frame), self-crossings (the gate's counter at contact-refresh resolution, FineCrossings; the frame-rate count beside it), a
             no-self-contact control batch that must cross, contact force over
             the rope's weight, ropes still moving at the end where nothing should move them, strain; the shards are
             kept (--pilot) for rendering
Results: docs/results/l0_rope_teacher_check.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l0_rope_teacher_check.json"
OBJECTS = "gso"                                      # obstacle library of the generator batches (--objects)
sys.path.insert(0, str(REPO / "scripts"))

import warp as wp                                               # noqa: E402
import newton                                                   # noqa: E402
import l0_rope_gate as gate                                     # noqa: E402
import lsim.teacher.rope_teacher as rt                          # noqa: E402

I = wp.quat_identity()
PATH_SECONDS = 40
PILOT = REPO / "outputs/lsim/rope_pilot"


def _rope(d, n):
    return rt.Rope(n=n, l=d["l"], lam=d["lam"], EA=d["EA"], EI=d["EI"], mu=d["mu"], r=d["r"])


class TeacherSim:
    """The gate's Sim interface over RopeTeacher: the scene's fixed nodes are made inactive (they stay put),
    grip(vel) pins the gripped node to a kinematic gripper moving at vel from where the node is; the gate's uniform
    settling drag becomes the teacher's settle_damping. Air drag is the teacher's (on) unless air=False."""

    air = True

    def __init__(self, solver, scene, rope, worlds=1, drag=0.0, **over):
        self.rope, self.sc, self.worlds = rope, scene, worlds
        self.ropes = [np.asarray(p, float) for p in scene["ropes"]]
        wb = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.mu, cfg.ke, cfg.kd, cfg.kf = rope["mu"], 1.0e3, 0.0, 1.0
        if scene.get("ground"):
            wb.add_ground_plane(cfg=cfg)
        for (c, h, q) in scene.get("boxes", []):
            wb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*c), q if q is not None else I), hx=h[0], hy=h[1], hz=h[2],
                             cfg=cfg)
        nshape = wb.shape_count
        chains, node_ids = [], []
        for P in self.ropes:
            R = _rope(rope, len(P) - 1)
            p0, n = rt.add_rope(wb, R, P)
            chains.append((p0, n, R))
            node_ids.append(list(range(p0, p0 + n)))
        npw = wb.particle_count
        g = wb.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 9.0), I), mass=0.0, label="gripper")
        b = newton.ModelBuilder()
        ws = []
        for _ in range(worlds):
            po, bo = b.particle_count, b.body_count
            b.add_world(wb)
            ws.append({"particles": (po, npw), "rope": chains[0][2], "gripper": bo + g,
                       "chains": [(po + p0, n, R) for (p0, n, R) in chains]})
        self.path = np.zeros((PATH_SECONDS * rt.FPS + 1, worlds, 7), np.float32)
        self.path[:, :, 2] = 9.0
        self.path[:, :, 6] = 1.0
        spp = float(over.get("spp", rt.STEPS_PER_PERIOD))
        ce = int(over.get("ce", rt.COLLIDE_EVERY))
        floor = 0 if "spp" in over else rt.MIN_SUBSTEPS
        Rs = [c[2] for c in chains]
        self.substeps = rt.stable_substeps(Rs, steps_per_period=spp, ce=ce, floor=floor)
        fixed = [node_ids[ri][k] for (ri, k) in scene.get("fixed", [])]
        allf = [p + w * npw for w in range(worlds) for p in fixed]
        self.teacher = rt.RopeTeacher(b, ws, [rt.ROLE_TABLE] * (nshape * worlds), self.path, self.substeps,
                                      self_contact=bool(scene.get("self_contact", True)), settle_damping=drag,
                                      drag=self.air, fixed=allf, ce=ce)
        self.dt = self.teacher.dt
        self.steps_per_sqrt_m_k = 1.0 / (self.dt * np.sqrt(max(R.k_over_m for R in Rs)))
        self.npw, self.node_ids = npw, node_ids
        self.nodes_flat = np.concatenate(node_ids)
        m = self.teacher.mass.astype(np.float64).copy()
        m[np.asarray(allf, int)] = 0.0                            # the gate's convention: fixed = massless
        self.mass = m
        self.t, self.wall, self.last, self.gripped = 0.0, 0.0, None, None
        if scene.get("grip") is not None:
            ri, k = scene["grip"]
            self.gripped = node_ids[ri][k]
            self.grip((0.0, 0.0, 0.0))

    def grip(self, vel):
        f = self.teacher.frame_index
        x = self.teacher.s0.particle_q.numpy()
        k = np.arange(self.path.shape[0] - f, dtype=np.float32)[:, None] / rt.FPS
        for w in range(self.worlds):
            self.path[f:, w, :3] = x[self.gripped + w * self.npw][None] + k * np.asarray(vel, np.float32)[None]
        self.teacher.path.assign(self.path)
        for w in range(self.worlds):
            self.teacher.grasp_ids(w, [self.gripped])
        if self.gripped is not None:                               # the gate's convention: gripped = massless
            self.mass[self.gripped] = 0.0

    def grip_force(self):
        return np.asarray(self.last["grasp"][0], np.float64)

    def frame(self, n=1):
        t0 = time.time()
        for _ in range(n):
            self.last = self.teacher.frame()
            self.t += 1.0 / rt.FPS
        wp.synchronize()
        self.wall += time.time() - t0

    def rate(self):
        return self.worlds * self.t / max(self.wall, 1e-9)

    def nodes(self, rope=None):
        x = self.teacher.s0.particle_q.numpy().astype(np.float64)
        return x[self.nodes_flat] if rope is None else x[self.node_ids[rope]]

    def node_speed_max(self):
        return float(np.linalg.norm(self.teacher.s0.particle_qd.numpy()[self.nodes_flat], axis=1).max())

    def energy(self):
        x = self.teacher.s0.particle_q.numpy()[:self.npw].astype(np.float64)
        v = self.teacher.s0.particle_qd.numpy()[:self.npw].astype(np.float64)
        m = self.mass[:self.npw]
        return 0.5 * float((m * (v ** 2).sum(1)).sum()), float((m * gate.G * x[:, 2]).sum())

    def total_mass(self):
        return float(self.mass[:self.npw].sum())

    def free_length(self, rope=0):
        return float(np.linalg.norm(np.diff(self.ropes[rope], axis=0), axis=1).sum())


gate.Sim = TeacherSim


def check_freefall(solver, **over):
    TeacherSim.air = False                                        # the closed form is drag-free
    try:
        return gate.check_freefall(solver, **over)
    finally:
        TeacherSim.air = True


def check_airfall(solver, **over):
    out = {}
    for tag, lam, r in (("cord", 0.02, 0.004), ("string", 0.005, 0.0015), ("heavy", 0.1, 0.0045)):
        rope = dict(gate.ROPE, lam=lam, r=r)
        P = gate.straight((-0.25, 0.0, 30.0), (1, 0, 0), 0.5, rope["l"])
        sim = TeacherSim(solver, {"ropes": [P]}, rope)
        sim.frame(int(3.5 * rt.FPS))
        v = float(-sim.teacher.s0.particle_qd.numpy()[:, 2].mean())
        want = float(np.sqrt(2.0 * lam * rt.G / (rt.RHO_CD * 2.0 * r)))
        out[tag] = {"speed_m_s": v, "terminal_m_s": want, "ratio": v / want, "pass": bool(abs(v / want - 1.0) < 0.05)}
        print(f"  airfall {tag:6s} lam {lam} r {1e3 * r:.1f} mm: falls at {v:.3f} m/s after 3.5 s, terminal {want:.3f} "
              f"(ratio {v / want:.3f})", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["summary"] = "ratio " + " / ".join(f"{out[k]['ratio']:.3f}" for k in ("cord", "string", "heavy"))
    return out


def check_bounce(solver, **over):
    """Straight ropes dropped flat from 10 cm onto the generator's table (top at z = 0)."""
    out = {}
    for tag, lam, EI, EA, r in (("string", 0.005, 1e-6, 1e2, 0.0015), ("cord", 0.02, 1e-4, 1e3, 0.004),
                                ("cable", 0.1, 1e-3, 1e4, 0.0045), ("light_stiff", 0.005, 1e-3, 1e4, 0.0015)):
        rope = dict(gate.ROPE, lam=lam, EI=EI, EA=EA, r=r)
        P = gate.straight((-0.25, 0.0, 0.10 + r), (1, 0, 0), 0.5, rope["l"])
        sim = TeacherSim(solver, {"ropes": [P], "boxes": [((0.0, 0.0, -0.05), (0.6, 0.5, 0.05), None)]}, rope)
        cz, vup, touch, prev = [], [], None, sim.nodes()
        for f in range(int(1.5 * rt.FPS)):
            sim.frame()
            x = sim.nodes()
            cz.append(float(x[:, 2].mean()))
            if touch is None and (x[:, 2].min() < r + 0.0015 or (len(cz) > 1 and cz[-1] > cz[-2])):
                touch = f                                 # near the table, or already on the way back up
            if touch is not None:
                vup.append(float(((x[:, 2] - prev[:, 2]) * rt.FPS).max()))
            prev = x
        rebound = float(max(cz[touch:]) - cz[-1])
        out[tag] = {"rebound_mm": 1e3 * rebound, "max_upward_speed_m_s": max(vup), "rest_height_minus_r_mm": 1e3 * (cz[-1] - r),
                    "substeps": sim.substeps, "pass": bool(rebound < 0.002 and max(vup) < 0.1)}
        print(f"  bounce  {tag:11s}: rebound {1e3 * rebound:.3f} mm, fastest upward after touch {max(vup):.4f} m/s, rests "
              f"{1e3 * (cz[-1] - r):+.3f} mm off r, {sim.substeps} substeps", flush=True)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    out["summary"] = "rebound " + " / ".join(f"{out[k]['rebound_mm']:.2f}" for k in out if isinstance(out[k], dict)) + " mm"
    return out


def check_episodes(solver, pilot=PILOT, **over):
    """Generator batches of 64 (sorted by stiffness as the generator does), kept as pilot shards."""
    import lsim.data.rope as rp
    from lsim.data import library
    train, _ = library.object_split(OBJECTS)          # --objects gso (default) | simready | both
    pilot = Path(pilot)
    pilot.mkdir(parents=True, exist_ok=True)
    out, rows = {}, []
    batches = (("mix_a", 101, None, None), ("mix_b", 102, None, None), ("coiled", 103, "coiled", None),
               ("pick_push", 104, None, ("pick_end", "push")))
    control = ("control_noself", 103, "coiled", None)          # the counter's control: self-contact off
    for i, (tag, seed, init, fams) in enumerate(batches + (control,)):
        keep = rp.INIT_P.copy(), rp.FAMILY_P.copy()
        if init:
            rp.INIT_P = np.array([1.0 if s == init else 0.0 for s in rp.INITS])
        if fams:
            rp.FAMILY_P = np.array([1.0 if f in fams else 0.0 for f in rp.FAMILIES])
            rp.FAMILY_P /= rp.FAMILY_P.sum()
        try:
            rng = np.random.default_rng(seed)
            specs = [rp.sample_spec(rng, train, seed=seed * 10000 + k) for k in range(64)]
            specs.sort(key=lambda sp: sp.rope.k_over_m)
            res = rp.run_batch(specs, seed=seed, log=lambda m: None, self_contact=tag != control[0])
        finally:
            rp.INIT_P, rp.FAMILY_P = keep
        qc = rp.quality(res)
        if tag == control[0]:
            out[tag] = {"crossed_fine": int(sum(q["crossings_fine"] > 0 for q in qc)),
                        "crossed_frame": int(sum(q["crossings"] > 0 for q in qc)),
                        "crossings_fine_total": int(sum(q["crossings_fine"] for q in qc))}
            print(f"  episodes {tag}: without self-contact {out[tag]['crossed_fine']}/64 crossed (fine counter), "
                  f"{out[tag]['crossed_frame']}/64 (frame counter)", flush=True)
            continue
        rp.save_shard(pilot, i, res, seed, qc=qc)
        for w, q in enumerate(qc):
            sp = specs[w]
            q.update(batch=tag, world=w, init=sp.init, families=[p["family"] for p in res["plans"][w] if "skipped" not in p],
                     length=sp.rope.length, EA=sp.rope.EA, EI=sp.rope.EI, lam=sp.rope.lam, mu=sp.rope.mu,
                     n_settle=res["settle"])
            rows.append(q)
        cw = np.array([max(q["contact_over_weight"]) for q in qc])
        row = {"episodes": len(qc), "exploded": int(sum(q["exploded"] for q in qc)),
               "crossed": int(sum(q["crossings_fine"] > 0 for q in qc)),
               "crossings": [q["crossings_fine"] for q in qc if q["crossings_fine"]],
               "crossed_frame_counter": int(sum(q["crossings"] > 0 for q in qc)),
               "min_gap_over_2r": float(min(q["min_centerline_gap_fine_mm"] / (2e3 * specs[w].rope.r) for w, q in enumerate(qc))),
               "contact_over_weight_p99": float(np.percentile(cw, 99)), "contact_over_weight_max": float(cw.max()),
               "fastest_node_m_s": float(max(q["vmax"] for q in qc)), "strain_max": float(max(q["strain_max"] for q in qc)),
               "substeps": res["substeps"], "wall_s": res["wall_s"], "candidate_load": res["candidate_load"],
               "episodes_per_gpu_hour": 3600.0 * len(qc) / res["wall_s"]}
        out[tag] = row
        print(f"  episodes {tag:9s}: {row['exploded']} exploded, {row['crossed']}/64 crossed {row['crossings']} (frame counter "
              f"{row['crossed_frame_counter']}), closest strands {row['min_gap_over_2r']:.2f} x 2r, contact/weight "
              f"p99 {row['contact_over_weight_p99']:.1f} max {row['contact_over_weight_max']:.1f}, fastest {row['fastest_node_m_s']:.2f} m/s, "
              f"strain max {row['strain_max']:.4f}, {res['substeps']} substeps, {res['wall_s']:.0f} s "
              f"({row['episodes_per_gpu_hour']:.0f}/h), candidates {res['candidate_load']}/{rt.SELF_CAND}", flush=True)
    # at rest where nothing moves it: not held, the last grasp / release 1.5 s before the end, the gripper gone
    F = int(round(rp.RECORD * rp.FPS))
    idle = [q for q in rows if not q["held_at_end"] and q["last_event_frame"] < F - 45 and not q["exploded"]]
    moving = [q for q in idle if q["end_speed"] > 0.01]
    lying = [q for q in idle if q["on_table_at_end"]]                # nothing hangs over an edge or falls
    lying_moving = [q for q in lying if q["end_speed"] > 0.01]
    out["rest"] = {"idle_episodes": len(idle), "moving_at_end_over_1cm_s": len(moving),
                   "end_speed_p50_mm_s": 1e3 * float(np.median([q["end_speed"] for q in idle])) if idle else None,
                   "end_speed_p99_mm_s": 1e3 * float(np.percentile([q["end_speed"] for q in idle], 99)) if idle else None,
                   "lying_on_table": len(lying), "lying_moving_over_1cm_s": len(lying_moving),
                   "moving": [{k: q[k] for k in ("batch", "world", "init", "families", "end_speed", "lowest_z", "on_table_at_end",
                                                 "EI", "lam", "length")} for q in moving]}
    print(f"  rest: {len(idle)} episodes idle for the last 1.5 s, {len(moving)} still moving > 1 cm/s ({len(lying_moving)} of the "
          f"{len(lying)} lying wholly on the table); median end speed "
          f"{out['rest']['end_speed_p50_mm_s']:.3f} mm/s", flush=True)
    for q in moving:
        print(f"     {q['batch']} w{q['world']} {q['init']} {q['families']}: {1e3 * q['end_speed']:.1f} mm/s, lowest z {q['lowest_z']:.3f}")
    tot = sum(out[k]["episodes"] for k, _, _, _ in batches)
    out["total"] = {"episodes": tot, "exploded": sum(out[k]["exploded"] for k, _, _, _ in batches),
                    "crossed": sum(out[k]["crossed"] for k, _, _, _ in batches),
                    "contact_over_weight_max": max(out[k]["contact_over_weight_max"] for k, _, _, _ in batches)}
    (pilot / "pilot_quality.json").write_text(json.dumps(rows, default=float))
    out["counter_can_fail"] = bool(out[control[0]]["crossed_fine"] > 0)
    out["pass"] = bool(out["total"]["exploded"] == 0 and out["total"]["crossed"] <= 0.01 * tot and out["counter_can_fail"])
    out["summary"] = (f"{tot} episodes: {out['total']['exploded']} exploded, {out['total']['crossed']} crossed, "
                      f"contact/weight max {out['total']['contact_over_weight_max']:.0f}, {len(moving)}/{len(idle)} idle still moving "
                      f"({len(lying_moving)}/{len(lying)} lying on the table)")
    return out


CHECKS = {"stretch": gate.check_stretch, "catenary": gate.check_catenary, "bend": gate.check_bend,
          "swing": gate.check_swing, "freefall": check_freefall, "airfall": check_airfall, "slide": gate.check_slide,
          "drop": gate.check_drop, "cross": gate.check_cross, "pile": gate.check_pile, "gripper": gate.check_gripper,
          "bounce": check_bounce, "stability": gate.check_stability, "throughput": gate.check_throughput,
          "episodes": check_episodes}


def main():
    global OUT, OBJECTS
    args = sys.argv[1:]
    pilot = PILOT
    if "--objects" in args:                                # results and pilot shards apart from the GSO ones
        k = args.index("--objects")
        OBJECTS = args[k + 1]
        args = args[:k] + args[k + 2:]
        if OBJECTS != "gso":
            OUT = OUT.with_name(OUT.stem + f"_{OBJECTS}.json")
            pilot = PILOT.with_name(PILOT.name + f"_{OBJECTS}")
    if "--pilot" in args:
        k = args.index("--pilot")
        pilot = REPO / args[k + 1]
        args = args[:k] + args[k + 2:]
    names = args or list(CHECKS)
    np.random.seed(gate.SEED)
    wp.init()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for n in names:
        t0 = time.time()
        try:
            res[n] = CHECKS[n]("semi", **({"pilot": pilot} if n == "episodes" else {}))
        except Exception as exc:                              # record the failure, keep going
            import traceback
            traceback.print_exc()
            res[n] = {"error": repr(exc), "pass": False}
            print(f"  {n}: ERROR {exc!r}", flush=True)
        res[n]["wall_s_check"] = time.time() - t0
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))
    res["_versions"] = {"newton": newton.__version__, "warp": wp.config.version, "rope": gate.ROPE,
                        "teacher": {"axial_eta": rt.AXIAL_ETA, "self_mu": rt.SELF_MU, "rho_cd": rt.RHO_CD,
                                    "steps_per_period": rt.STEPS_PER_PERIOD, "min_substeps": rt.MIN_SUBSTEPS,
                                    "self_query": rt.SELF_QUERY, "self_cand": rt.SELF_CAND, "ea_max": rt.EA_MAX}}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    for k, v in res.items():
        if isinstance(v, dict) and ("pass" in v or "error" in v) and not k.startswith("_"):
            print(f"  {k:10s} pass={v.get('pass')}  {v.get('summary', v.get('error', ''))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
