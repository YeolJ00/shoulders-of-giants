"""How far does information travel through the cloth within one 30 fps frame (2026-11-19)?

Hypothesis: stretch waves move at sqrt(Y / w) (~140 m/s for Y 3,000 N/m, w 0.16 kg/m^2: ~4.5 m per frame), so the
whole cloth is coupled every frame and a network has to solve a global problem in one pass. Measured with the teacher
(model-free): c1-style held-out episodes recorded with instantaneous velocities; at frame instants t the teacher is
restarted from its full state (l1_teacher_from_inputs' restart) for one frame, once as is and once with ONE source
vertex's velocity changed by dv (per world: a moving vertex off the flat surfaces, not held). The response of every
other unheld vertex of that cloth, |dx_j| / (|dv| dt), against its rest distance from the source:
  1.0 = the source moved as if free (dx = dv dt);  << 1 = untouched.
Copies per restart: base, a second base (the noise floor), dv = 0.5 / 1 / 2 cm/s along one random direction
(linearity); and the same base and dv = 1 cm/s with Y / 100 (a cloth whose waves travel 10x slower).

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_frame_influence.py
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
FPS = 30.0
DV = (0.005, 0.01, 0.02)                                             # m/s
BANDS = ((0.0, 0.001), (0.005, 0.015), (0.015, 0.03), (0.03, 0.06), (0.06, 0.12), (0.12, 0.24), (0.24, 0.48))
NAMES = ("0", "1", "2", "4", "8", "16", "32")                        # cm: the bands' nominal distances


def restart(specs, X, V, t, settle, path, events, worlds, sources, dirs, dvs, log=lambda m: None):
    """One run_batch: copy k = 0.. len(dvs) - 1 of every world from recorded frame t, the source vertex's velocity
    changed by dvs[k] * dirs[w]. -> x_{t+1} (copies, total cloth vertices, 3)."""
    from l1_teacher_from_inputs import held_at
    from lsim.data.cloth import run_batch
    T = settle + t
    start, replay = [], []
    for w in range(len(worlds)):
        p0, n = worlds[w]["particles"]
        replay.append({"path": path[T:T + 2, w], "events": held_at(events[w], T, n)[1]})
    for dv in dvs:
        for w in range(len(worlds)):
            p0, n = worlds[w]["particles"]
            v = V[t, p0:p0 + n].copy()
            if sources[w] >= 0:
                v[sources[w]] += dv * dirs[w]
            start.append({"x": X[t, p0:p0 + n], "v": v})
    r = run_batch(specs * len(dvs), seed=0, record=1.0 / FPS, log=log, replay=replay * len(dvs), settle=0, start=start)
    nv = sum(wd["particles"][1] for wd in worlds)
    return r["x"][1].astype(np.float64).reshape(len(dvs), nv, 3), r["substeps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_t", type=int, default=10)
    a = ap.parse_args()
    import warp as wp
    from l1_teacher_from_inputs import held_at, on_surface
    from lsim.data import library
    from lsim.data.cloth import run_batch, sample_spec
    from lsim.teacher.cloth_mesh import grid
    wp.init()
    _, test = library.object_split("gso")
    rng = np.random.default_rng(a.seed)
    specs = [sample_spec(rng, test, seed=a.seed * 10000 + k) for k in range(a.worlds)]
    soft = copy.deepcopy(specs)
    for sp in soft:
        sp.cloth.Y /= 100.0
    t0 = time.time()
    res = run_batch(specs, seed=a.seed, record=4.0, log=lambda m: None, record_v=True)
    X, V, settle, path, events, worlds = res["x"].astype(np.float64), res["v"].astype(np.float64), res["settle"], \
        res["path"], res["events"], res["worlds"]
    sub0 = res["substeps"]
    del res
    W = a.worlds
    rest = [grid(sp.cloth.nx, sp.cloth.ny, sp.cloth.h)[0][:, :2] for sp in specs]
    stiff = np.array([sp.cloth.Y / sp.cloth.w for sp in specs])
    speed = np.sqrt(stiff)
    print(f"recorded {W} episodes x {X.shape[0] - 1} frames, {sub0} substeps, in {time.time() - t0:.0f} s; "
          f"wave speed sqrt(Y / w) {np.median(speed):.0f} m/s median ({speed.min():.0f}-{speed.max():.0f}): "
          f"{np.median(speed) / FPS:.1f} m per frame", flush=True)
    F = X.shape[0] - 1
    ts = np.unique(np.linspace(8, F - 2, a.n_t).round().astype(int))
    rows = []                                                       # one per (t, world) with a source
    for t in ts:
        T = settle + t
        sources, dirs = np.full(W, -1), np.zeros((W, 3))
        for w in range(W):
            p0, n = worlds[w]["particles"]
            h = held_at(events[w], T, n)[0]
            x = X[t, p0:p0 + n]
            mv = np.linalg.norm(x - X[t - 1, p0:p0 + n], axis=-1) * FPS > 0.02
            ok = np.flatnonzero(mv & ~h & ~on_surface(x, specs[w]))
            if len(ok):
                sources[w] = rng.choice(ok)
                d = rng.normal(size=3)
                dirs[w] = d / np.linalg.norm(d)
        stiff_x, s1 = restart(specs, X, V, t, settle, path, events, worlds, sources, dirs, (0.0, 0.0) + DV)
        soft_x, s2 = restart(soft, X, V, t, settle, path, events, worlds, sources, dirs, (0.0, 0.0, 0.01))
        assert s1 == sub0, (s1, sub0)                               # (the softer cloth runs at its own, fewer substeps)
        sub_soft = s2
        for w in range(W):
            if sources[w] < 0:
                continue
            p0, n = worlds[w]["particles"]
            h = held_at(events[w], T, n)[0]
            d = np.linalg.norm(rest[w] - rest[w][sources[w]], axis=-1)
            keep = ~h
            sl = slice(p0, p0 + n)
            resp = lambda A, k, dv: np.linalg.norm(A[k, sl] - A[0, sl], axis=-1) / (dv / FPS)
            row = {"t": int(t), "world": w, "Y_over_w": float(stiff[w]), "noise": {}, "stiff": {}, "soft": {}}
            for name, (lo, hi) in zip(NAMES, BANDS):
                m = keep & (d >= lo) & (d < hi)
                if not m.any():
                    continue
                row["noise"][name] = float(np.mean(resp(stiff_x, 1, 0.01)[m]))
                for k, dv in enumerate(DV):
                    row["stiff"].setdefault(name, {})[str(dv)] = float(np.mean(resp(stiff_x, 2 + k, dv)[m]))
                row["soft"][name] = float(np.mean(resp(soft_x, 2, 0.01)[m]))
            dpv = lambda A, k, dv: ((A[k, sl] - A[0, sl]) @ dirs[w])[keep].sum() / (dv / FPS)   # momentum along dv
            row["total"] = {"stiff": float(dpv(stiff_x, 3, 0.01)), "soft": float(dpv(soft_x, 2, 0.01)),
                            "noise": float(dpv(stiff_x, 1, 0.01))}
            far = keep & (d >= 0.08)
            if far.any():
                row["stiff_far"] = float(np.mean(resp(stiff_x, 3, 0.01)[far]))
                row["soft_far"] = float(np.mean(resp(soft_x, 2, 0.01)[far]))
                row["noise_far"] = float(np.mean(resp(stiff_x, 1, 0.01)[far]))
            rows.append(row)
        print(f"  t {t:3d}: {sum(s >= 0 for s in sources)} sources", flush=True)
    out = {"worlds": W, "frames": [int(t) for t in ts], "substeps": int(sub0), "substeps_soft": int(sub_soft), "n_sources": len(rows),
           "wave_speed_ms": {"median": float(np.median(speed)), "min": float(speed.min()), "max": float(speed.max())},
           "bands_cm": NAMES, "summary": {}, "rows": rows}
    q = lambda xs: (float(np.median(xs)), float(np.quantile(xs, 0.9))) if len(xs) else (float("nan"),) * 2
    print(f"{len(rows)} sources; response |dx| / (|dv| dt) by rest distance from the source (median / p90 over sources)")
    print("  cm    noise         dv 0.5 cm/s     dv 1 cm/s       dv 2 cm/s       Y / 100 (1 cm/s)")
    for name in NAMES:
        r = [x for x in rows if name in x["stiff"]]
        S = {"noise": q([x["noise"][name] for x in r]), "soft": q([x["soft"][name] for x in r])}
        for dv in DV:
            S[str(dv)] = q([x["stiff"][name][str(dv)] for x in r])
        out["summary"][name] = S
        print(f"  {name:>3s}  " + "  ".join(f"{S[k][0]:.4f}/{S[k][1]:.4f}" for k in ("noise", "0.005", "0.01", "0.02", "soft")))
    out["total"] = {k: q([x["total"][k] for x in rows]) for k in ("stiff", "soft", "noise")}
    print("  the kick's momentum found in the cloth after the frame (sum over its vertices of dx . dv / (|dv| dt)): " +
          "  ".join(f"{k} {v[0]:.3f} (p90 {v[1]:.3f})" for k, v in out["total"].items()))
    cut = np.quantile([x["Y_over_w"] for x in rows], [1 / 3, 2 / 3])
    out["tertiles"] = []
    print("  by stiffness tertile (dv 1 cm/s): response at 1 and 2 cm against 8+ cm, median")
    for k in range(3):
        r = [x for x in rows if np.digitize(x["Y_over_w"], cut) == k and "stiff_far" in x]
        near = [np.mean([x["stiff"][nm]["0.01"] for nm in ("1", "2") if nm in x["stiff"]]) for x in r]
        tr = {"Y_over_w": (float(min(x["Y_over_w"] for x in r)), float(max(x["Y_over_w"] for x in r))), "n": len(r),
              "near_1_2cm": q(near), "far_8cm": q([x["stiff_far"] for x in r]), "soft_far_8cm": q([x["soft_far"] for x in r]),
              "noise_far_8cm": q([x["noise_far"] for x in r]),
              "wave_m_per_frame": float(np.sqrt(np.median([x["Y_over_w"] for x in r])) / FPS)}
        out["tertiles"].append(tr)
        print(f"  Y / w {tr['Y_over_w'][0]:7.0f}-{tr['Y_over_w'][1]:7.0f} ({tr['wave_m_per_frame']:.1f} m / frame, n {tr['n']}): "
              f"1-2 cm {tr['near_1_2cm'][0]:.4f}  8+ cm {tr['far_8cm'][0]:.4f}  (Y / 100: {tr['soft_far_8cm'][0]:.4f}; "
              f"noise {tr['noise_far_8cm'][0]:.4f})", flush=True)
    (REPO / "docs/results").mkdir(exist_ok=True)
    json.dump(out, open(REPO / "docs/results/l1_frame_influence.json", "w"), indent=1)


if __name__ == "__main__":
    main()
