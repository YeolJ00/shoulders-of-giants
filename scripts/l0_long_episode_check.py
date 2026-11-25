"""L0 -- long episodes checked on the modules that generate them (lsim.data.cloth / lsim.data.rope with record 15-30 s
and 4-8 chained action families, CLAUDE.md: "gate the module that generates the data, on what the data does").

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l0_long_episode_check.py [cloth rope rest_cloth rest_rope] [--seconds 20]
        [--cloth_n 64 --rope_n 128 --out outputs/lsim/long_pilot --tag <name>]

A 4 s episode never shows what a chain of actions leaves behind: a fold dragged, then dropped on a pile, then lifted
again, nor a towel left alone for seconds. The 4 s checks (l0_cloth_teacher_check.py, l0_rope_teacher_check.py) run
unchanged beside these:

  cloth       a generator batch of long cloth episodes (the natural mix of starts, actions and idle gaps): explosions
              (rule 3: a vertex moving 0.4 m in a frame, or any vertex more than 3 m from the table), self-crossings (the
              teacher check's edge-through-triangle count) every second and at the end, contact over the weight and
              pinches (> 200x, what the loaders drop), guarded frames, how many of the requested actions start and
              complete, held at the end, whether the cloth left the table, and how still it is at the end when nothing
              has touched it for the last 1.5 s
  rope        the same for ropes (crossings by the rope teacher's counter at contact-refresh resolution)
  rest_cloth  flat and draped towels left alone for the whole episode: drift from 1 s on; flat ones must not creep (< 1
              mm/s over the last 5 s, the teacher gate's resting bound); draped ones are reported (they settle, some
              slide off their object, and still ones creep up to g / FRICTION_RATE = 0.49 mm/s, the regularised friction)
  rest_rope   straight and curved ropes left alone, the same (curved ones relax towards straight)
Pilot shards are kept under --out (cloth/, rope/, rest_cloth/, rest_rope/) for rendering. Results:
docs/results/l0_long_episode_check[_<tag>].json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FAR = 3.0                                                 # m from the table centre: exploded whatever the frame step
PINCH = 200.0                                             # contact over weight: what the loaders drop
IDLE_TAIL = 45                                            # frames: nothing touched the body for the last 1.5 s
STILL = 0.01                                              # m/s: a body faster than this at the end still moves
FIXES = True                                              # keep + own_settle (--no_fixes: the first pilot's planner)


def _last_move(grip):
    """The last recorded frame in which the gripper moved."""
    d = np.linalg.norm(np.diff(grip[:, :3], axis=0), axis=1)
    k = np.flatnonzero(d > 1e-6)
    return int(k[-1] + 1) if len(k) else 0


def _actions(res, w, settle):
    plans = [p for p in res["plans"][w] if p.get("family") != "held"]   # the "held" start is not an action
    started = [p for p in plans if "skipped" not in p]
    done = [p for p in started if not p.get("cut_by_episode_end")]
    why = {}
    for p in plans:
        if "skipped" in p:
            why[p["skipped"]] = why.get(p["skipped"], 0) + 1
    return {"requested": int(res["specs"][w].n_families), "slots": len(plans), "started": len(started),
            "completed": len(done), "cut": len(started) - len(done), "skipped": why,
            "families": [p["family"] for p in started], "idle_s": float(sum(p.get("idle_after_s", 0.0) for p in plans)),
            "kept": int(sum(p.get("kept_on_table", False) for p in plans))}


def _held_at_end(ev):
    g = sorted(e[0] for e in ev if e[1] == "grasp")
    r = sorted(e[0] for e in ev if e[1] == "release")
    return bool(g and (not r or r[-1] < g[-1]))


def _common(res, w, Xw, wt, q):
    F = Xw.shape[0]
    grip = res["grip"][:, w]
    lm = _last_move(grip)
    held = _held_at_end(res["events"][w])
    sp_end = float(np.linalg.norm(Xw[-1] - Xw[-2], axis=-1).max() * 30.0)
    sp_half = float(np.linalg.norm(Xw[-1] - Xw[-16], axis=-1).max() * 30.0 / 15.0)
    far = bool(np.linalg.norm(Xw[..., :2], axis=-1).max() > FAR or np.abs(Xw[..., 2]).max() > FAR)
    out = {"exploded": bool(q["exploded"] or far or not np.isfinite(Xw).all()), "vmax": q["vmax"],
           "contact_over_weight": float(max(q["contact_over_weight"])), "grasp_over_weight": q["grasp_over_weight"],
           "pinched": bool(max(q["contact_over_weight"]) > PINCH), "guarded_frames": int(res["guarded"][w]),
           "held_at_end": held, "gripper_last_move": lm, "idle_at_end": bool(not held and lm < F - 1 - IDLE_TAIL),
           "end_speed": sp_end, "end_speed_half_s": sp_half,
           "off_table_end": float((Xw[-1, :, 2] < -0.01).mean()), "on_floor_end": bool((Xw[-1, :, 2] < -0.6).any()),
           "start_on_floor": bool((Xw[0, :, 2] < -0.6).any()), "start_off_table": bool((Xw[0, :, 2] < -0.01).any())}
    out.update(_actions(res, w, res["settle"]))
    return out


def run_cloth(n, seconds, seed, out, rest=False):
    import torch
    import lsim.data.cloth as cl
    import lsim.teacher.cloth_teacher as ct
    from lsim.data import library
    from l0_cloth_teacher_check import _crossings
    train, _ = library.split(library.tabletop())
    rng = np.random.default_rng(seed)
    keep = cl.INIT_P.copy()
    if rest:                                              # left alone: half flat, half draped over an object
        specs = []
        for init in ("flat", "draped"):
            cl.INIT_P = np.array([1.0 if i == init else 0.0 for i in cl.INITS])
            specs += [cl.sample_spec(rng, train, seed=seed * 10000 + len(specs) + k) for k in range(n // 2)]
        cl.INIT_P = keep
        for sp in specs:
            sp.n_families = 0
    else:
        specs = [cl.sample_spec(rng, train, seed=seed * 10000 + k, actions=cl.LONG_ACTIONS) for k in range(n)]
    specs.sort(key=lambda sp: sp.cloth.vertex_mass / sp.cloth.Y)
    t0 = time.time()
    res = cl.run_batch(specs, seed=seed, record=seconds, log=print, idle=None if rest else cl.LONG_IDLE, fit=not rest,
                       keep=not rest and FIXES, own_settle=FIXES)
    wall = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    cl.save_shard(out, 0, res, seed)
    qc = cl.quality(res)
    rows = []
    X = res["x"]
    every = np.unique(np.r_[np.arange(0, X.shape[0], 30), X.shape[0] - 1])
    for w, wd in enumerate(res["worlds"]):
        p0, nv = wd["particles"]
        c = wd["cloth"]
        Xw = X[:, p0:p0 + nv].astype(np.float64)
        wt = c.w * c.nx * c.ny * c.h * c.h * ct.G
        row = _common(res, w, Xw, wt, qc[w])
        _, tris = ct.grid(c.nx, c.ny, c.h)
        cr = [_crossings(Xw[j], tris, torch) for j in every] if not rest else [_crossings(Xw[-1], tris, torch)]
        row.update(init=specs[w].init, crossed_at_end=bool(cr[-1] > 0), crossings_end=int(cr[-1]),
                   crossed_ever=bool(max(cr) > 0), first_crossed_s=float(every[int(np.argmax(np.array(cr) > 0))] / 30.0) if max(cr) else None)
        if rest:
            row.update(_drift(Xw))
        rows.append(row)
    return rows, {"wall_s": wall, "substeps": res["substeps"], "worlds": len(specs), "seconds": seconds,
                  "sim_s_per_wall_s": len(specs) * seconds / wall}


def run_rope(n, seconds, seed, out, rest=False):
    import lsim.data.rope as rp
    from lsim.data import library
    train, _ = library.split(library.tabletop())
    rng = np.random.default_rng(seed)
    keep = rp.INIT_P.copy()
    if rest:                                              # left alone: half straight, half curved
        specs = []
        for init in ("straight", "curved"):
            rp.INIT_P = np.array([1.0 if i == init else 0.0 for i in rp.INITS])
            specs += [rp.sample_spec(rng, train, seed=seed * 10000 + len(specs) + k) for k in range(n // 2)]
        rp.INIT_P = keep
        for sp in specs:
            sp.n_families = 0
    else:
        specs = [rp.sample_spec(rng, train, seed=seed * 10000 + k, actions=rp.LONG_ACTIONS) for k in range(n)]
    specs.sort(key=lambda sp: sp.rope.k_over_m)
    t0 = time.time()
    res = rp.run_batch(specs, seed=seed, record=seconds, log=print, idle=None if rest else rp.LONG_IDLE, fine_crossings=True,
                       keep=not rest and FIXES, own_settle=FIXES)
    wall = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    qc = rp.quality(res, crossings=False)
    rp.save_shard(out, 0, res, seed, qc=qc)
    rows = []
    for w, wd in enumerate(res["worlds"]):
        p0, nv = wd["particles"]
        r = wd["rope"]
        Xw = res["x"][:, p0:p0 + nv].astype(np.float64)
        row = _common(res, w, Xw, r.lam * r.length * rp.G, qc[w])
        row.update(init=specs[w].init, crossings=int(qc[w]["crossings_fine"]), crossed=bool(qc[w]["crossings_fine"] > 0),
                   strain_max=qc[w]["strain_max"], category=specs[w].material.get("category"))
        if rest:
            row.update(_drift(Xw))
        rows.append(row)
    return rows, {"wall_s": wall, "substeps": res["substeps"], "worlds": len(specs), "seconds": seconds,
                  "sim_s_per_wall_s": len(specs) * seconds / wall}


def _drift(Xw):
    """A body left alone, from 1 s on: how far its vertices go (mean, largest) and their speed over the last 5 s."""
    a = Xw[30]
    d = np.linalg.norm(Xw[-1] - a, axis=-1)
    last5 = np.linalg.norm(Xw[-1] - Xw[-151], axis=-1) / 5.0
    return {"drift_mean_mm": 1e3 * float(d.mean()), "drift_max_mm": 1e3 * float(d.max()),
            "creep_last5s_max_mm_s": 1e3 * float(last5.max()), "creep_last5s_mean_mm_s": 1e3 * float(last5.mean())}


def summarise(kind, rows, info, rest=False):
    n = len(rows)
    s = {"episodes": n, **info, "exploded": sum(r["exploded"] for r in rows), "pinched": sum(r["pinched"] for r in rows),
         "contact_over_weight_p99": float(np.percentile([r["contact_over_weight"] for r in rows], 99)),
         "contact_over_weight_max": float(max(r["contact_over_weight"] for r in rows)),
         "fastest_m_s": float(max(r["vmax"] for r in rows)),
         "guarded_episodes": sum(r["guarded_frames"] > 0 for r in rows)}
    if kind == "cloth":
        s.update(crossed_at_end=sum(r["crossed_at_end"] for r in rows), crossed_ever=sum(r["crossed_ever"] for r in rows))
    else:
        s.update(crossed=sum(r["crossed"] for r in rows), strain_max=float(max(r["strain_max"] for r in rows)))
    if rest:
        dm = np.array([r["drift_max_mm"] for r in rows])
        cm = np.array([r["creep_last5s_max_mm_s"] for r in rows])
        s.update(drift_max_mm_p50=float(np.median(dm)), drift_max_mm_p90=float(np.percentile(dm, 90)), drift_max_mm_max=float(dm.max()),
                 drift_mean_mm_p50=float(np.median([r["drift_mean_mm"] for r in rows])),
                 creep_last5s_max_mm_s_max=float(cm.max()), creep_last5s_max_mm_s_p50=float(np.median(cm)),
                 creeping_over_1mm_s=int((cm > 1.0).sum()),
                 by_init={i: {"n": sum(r["init"] == i for r in rows),
                              "drift_max_mm_p50": float(np.median([r["drift_max_mm"] for r in rows if r["init"] == i])),
                              "drift_max_mm_max": float(max(r["drift_max_mm"] for r in rows if r["init"] == i)),
                              "creep_max_mm_s": float(max(r["creep_last5s_max_mm_s"] for r in rows if r["init"] == i))}
                          for i in sorted({r["init"] for r in rows})})
        flat = [r for r in rows if r["init"] in ("flat", "straight")]  # creep: what lies flat must stay put (draped
        s["flat_creeping_over_1mm_s"] = int(sum(r["creep_last5s_max_mm_s"] > 1.0 for r in flat))   # and curved ones settle)
        s["pass"] = bool(s["exploded"] == 0 and s["flat_creeping_over_1mm_s"] == 0)
        return s
    req = sum(r["requested"] for r in rows)
    st = sum(r["started"] for r in rows)
    dn = sum(r["completed"] for r in rows)
    why = {}
    for r in rows:
        for k, v in r["skipped"].items():
            why[k] = why.get(k, 0) + v
    idle = [r for r in rows if r["idle_at_end"]]
    sp = np.array([r["end_speed"] for r in idle]) if idle else np.zeros(0)
    on_table = [r for r in idle if r["off_table_end"] == 0.0]
    s.update(actions_requested=req, actions_started=st, actions_completed=dn, completed_of_requested=dn / max(req, 1),
             completed_of_started=dn / max(st, 1), skipped=why,
             actions_per_episode_p50=float(np.median([r["completed"] for r in rows])),
             actions_per_episode_min=int(min(r["completed"] for r in rows)),
             actions_per_episode_max=int(max(r["completed"] for r in rows)),
             idle_gap_s_mean=float(np.mean([r["idle_s"] for r in rows])), kept_on_table=int(sum(r["kept"] for r in rows)),
             start_on_floor=sum(r["start_on_floor"] for r in rows), start_off_table=sum(r["start_off_table"] for r in rows),
             held_at_end=sum(r["held_at_end"] for r in rows), left_table=sum(r["off_table_end"] > 0 for r in rows),
             on_floor=sum(r["on_floor_end"] for r in rows),
             idle_at_end=len(idle), idle_end_speed_p50_mm_s=1e3 * float(np.median(sp)) if len(sp) else None,
             idle_end_speed_p90_mm_s=1e3 * float(np.percentile(sp, 90)) if len(sp) else None,
             idle_moving_over_1cm_s=int((sp > STILL).sum()),
             idle_on_table=len(on_table), idle_on_table_moving_over_1cm_s=int(sum(r["end_speed"] > STILL for r in on_table)))
    crossed = s["crossed_at_end"] if kind == "cloth" else s["crossed"]
    s["pass"] = bool(s["exploded"] == 0 and s["pinched"] == 0 and crossed <= 0.05 * n and s["completed_of_started"] >= 0.9)
    return s


def main():
    args = sys.argv[1:]
    opt = {"--seconds": 20.0, "--cloth_n": 64, "--rope_n": 128, "--out": "outputs/lsim/long_pilot", "--tag": "", "--seed": 301}
    for k in list(opt):
        if k in args:
            i = args.index(k)
            opt[k] = type(opt[k])(args[i + 1])
            args = args[:i] + args[i + 2:]
    global FIXES
    if "--no_fixes" in args:
        FIXES = False
        args.remove("--no_fixes")
    names = args or ["cloth", "rope", "rest_cloth", "rest_rope"]
    import warp as wp
    wp.config.quiet = True
    wp.init()
    out = REPO / opt["--out"]
    path = REPO / f"docs/results/l0_long_episode_check{'_' + opt['--tag'] if opt['--tag'] else ''}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for n in names:
        rest = n.startswith("rest_")
        kind = n.replace("rest_", "")
        N = opt["--cloth_n"] if kind == "cloth" else opt["--rope_n"]
        fn = run_cloth if kind == "cloth" else run_rope
        rows, info = fn(N, opt["--seconds"], opt["--seed"] + (1 if rest else 0), out / n, rest=rest)
        s = summarise(kind, rows, info, rest)
        res[n] = {"summary": s, "episodes": rows}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1, default=float))
        print(f"[{n}] " + json.dumps(s, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
