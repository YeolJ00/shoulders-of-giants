"""Do the model's inputs carry the next frame (2026-11-19)? An information bound from the teacher itself, model-free.

87% of the trained model's squared one-step error is on free cloth (l2_error_where), where it sits at 0.67 of constant
velocity's error. The model sees cloth positions at the 30 fps frame instants only. Here the teacher is restarted from
recorded frame t for one frame (run_batch start= / replay=, same scenes, same substeps) with the positions x_t and four
velocities, and its x_{t+1} compared with the recorded one on moving free (unheld) cloth vertices:
  true     the teacher's instantaneous velocity at t (validates the restart: should reproduce to float noise)
  back1    (x_t - x_{t-1}) * 30
  back2    (3 x_t - 4 x_{t-1} + x_{t-2}) * 15
  back3    (11 x_t - 18 x_{t-1} + 9 x_{t-2} - 2 x_{t-3}) * 5
  cubic    per vertex a cubic least-squares fit to x_{t-7..t}, differentiated at t (smooth over all 8 past frames)
each against constant velocity (2 x_t - x_{t-1}) on the same vertices, split into on a flat surface (within 3 mm of
the table / floor / plank top) and not.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_teacher_from_inputs.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0
VARIANTS = ("true", "back1", "back2", "back3", "cubic")


def velocities(X, V, t):
    """(V_true, back1, back2, cubic) at recorded frame t: arrays (n_vertices, 3)."""
    k = np.arange(-7, 1, dtype=np.float64) / FPS                     # times of x_{t-7..t}, t at 0
    A = np.stack([np.ones_like(k), k, k ** 2, k ** 3], 1)           # cubic in time
    coef = np.linalg.lstsq(A, X[t - 7:t + 1].reshape(8, -1), rcond=None)[0]   # (4, V * 3)
    return {"true": V[t], "back1": (X[t] - X[t - 1]) * FPS, "back2": (3 * X[t] - 4 * X[t - 1] + X[t - 2]) * FPS / 2,
            "back3": (11 * X[t] - 18 * X[t - 1] + 9 * X[t - 2] - 2 * X[t - 3]) * FPS / 6, "cubic": coef[1].reshape(-1, 3)}


def on_surface(x, spec, r=0.003, tol=0.003):
    """Within tol of a flat surface's top (table, floor, plank; the planes ClothEpisode.planes gives the model)."""
    from lsim.data.scene import RAMP, TABLE_HALF, ramp_frame
    planes = [(np.zeros(3), np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), *TABLE_HALF),
              (np.array([0.0, 0, -0.75]), np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), 10.0, 10.0)]
    if spec.ramp is not None:
        o, ax, lat, n = ramp_frame(spec.ramp)
        planes.append((o + 0.5 * RAMP["L"] * ax + RAMP["T"] * n, n, ax, lat, 0.5 * RAMP["L"], 0.5 * RAMP["W"]))
    on = np.zeros(len(x), bool)
    for o, n, u, v, a, b in planes:
        d = x - o
        on |= (np.abs(d @ u) <= a) & (np.abs(d @ v) <= b) & (d @ n < r + tol)
    return on


def held_at(events, T, n):
    """Cloth vertex indices pinned during the step from timeline frame T to T + 1, and the replay's events from T."""
    cur, out = np.zeros(n, bool), []
    for e in sorted(events, key=lambda e: e[0]):
        if e[0] <= T:
            cur = np.zeros(n, bool)
            if e[1] == "grasp" and len(e) > 3 and len(e[3]):
                cur[np.asarray(e[3], int)] = True
        else:
            out.append((e[0] - T,) + tuple(e[1:]))
    if cur.any():
        out = [(0, "grasp", [0.0, 0.0, 0.0], [int(i) for i in np.flatnonzero(cur)])] + out
    return cur, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_t", type=int, default=20)
    a = ap.parse_args()
    import warp as wp
    from lsim.data import library
    from lsim.data.cloth import run_batch, sample_spec
    wp.init()
    _, test = library.object_split("gso")
    rng = np.random.default_rng(a.seed)
    specs = [sample_spec(rng, test, seed=a.seed * 10000 + k) for k in range(a.worlds)]
    t0 = time.time()
    res = run_batch(specs, seed=a.seed, record=4.0, log=lambda m: None, record_v=True)
    X, V, settle, path, events, worlds = res["x"].astype(np.float64), res["v"].astype(np.float64), res["settle"], \
        res["path"], res["events"], res["worlds"]
    sub0 = res["substeps"]
    del res
    print(f"recorded {a.worlds} episodes x {X.shape[0] - 1} frames, {sub0} substeps per frame, settle {settle} frames, "
          f"in {time.time() - t0:.0f} s", flush=True)
    F = X.shape[0] - 1
    ts = np.unique(np.linspace(8, F - 2, a.n_t).round().astype(int))
    W, nv = a.worlds, len(VARIANTS)
    err = {v: {"surface": [], "off": []} for v in VARIANTS + ("cv",)}
    per_t = []
    for t in ts:
        T = settle + t
        vel = velocities(X, V, t)
        start, replay, sel = [], [], []
        for w in range(W):
            p0, n = worlds[w]["particles"]
            h, ev = held_at(events[w], T, n)
            replay.append({"path": path[T:T + 2, w], "events": ev})
            moving = np.linalg.norm(X[t, p0:p0 + n] - X[t - 1, p0:p0 + n], axis=-1) * FPS > 0.02
            sel.append(moving & ~h)
        for v in VARIANTS:                                           # the four restarts in one batch: worlds v * W + w
            for w in range(W):
                p0, n = worlds[w]["particles"]
                start.append({"x": X[t, p0:p0 + n], "v": vel[v][p0:p0 + n]})
        r = run_batch(specs * nv, seed=a.seed, record=1.0 / FPS, log=lambda m: None, replay=replay * nv, settle=0,
                      start=start)
        assert r["substeps"] == sub0, (r["substeps"], sub0)
        Y = r["x"][1].astype(np.float64)
        del r
        row = {"t": int(t)}
        for w in range(W):
            p0, n = worlds[w]["particles"]
            s = sel[w]
            if not s.any():
                continue
            tgt = X[t + 1, p0:p0 + n][s]
            surf = on_surface(X[t, p0:p0 + n][s], specs[w])
            cv = np.linalg.norm(2 * X[t, p0:p0 + n][s] - X[t - 1, p0:p0 + n][s] - tgt, axis=-1)
            err["cv"]["surface"].append(cv[surf]); err["cv"]["off"].append(cv[~surf])
            for k, v in enumerate(VARIANTS):
                q0 = worlds[w]["particles"][0] + k * sum(wd["particles"][1] for wd in worlds)
                e = np.linalg.norm(Y[q0:q0 + n][s] - tgt, axis=-1)
                err[v]["surface"].append(e[surf]); err[v]["off"].append(e[~surf])
        cvm = np.concatenate([np.concatenate(err["cv"][c][-W:]) for c in ("surface", "off") if err["cv"][c]]).mean()
        print(f"  t {t:3d}: " + "  ".join(f"{v} {np.concatenate([np.concatenate(err[v][c][-W:]) for c in ('surface', 'off')]).mean() / cvm:.3f}"
                                         for v in VARIANTS) + f"   (constant velocity {1e3 * cvm:.3f} mm)", flush=True)
    out = {"worlds": W, "frames": [int(t) for t in ts], "substeps": int(sub0), "classes": {}}
    for c in ("all", "off", "surface"):
        pick = (lambda d: np.concatenate(d["surface"] + d["off"])) if c == "all" else (lambda d, c=c: np.concatenate(d[c]))
        cv = pick(err["cv"])
        row = {"n": int(len(cv)), "cv_mm": 1e3 * float(cv.mean()), "cv_rms_mm": 1e3 * float(np.sqrt((cv ** 2).mean()))}
        for v in VARIANTS:
            e = pick(err[v])
            row[v] = {"mm": 1e3 * float(e.mean()), "ratio": float(e.mean() / cv.mean()),
                      "ratio_rms": float(np.sqrt((e ** 2).mean() / (cv ** 2).mean())), "p90_mm": 1e3 * float(np.quantile(e, 0.9))}
        out["classes"][c] = row
        print(f"{c:8s} n {row['n']:7d}  constant velocity {row['cv_mm']:.3f} mm  " +
              "  ".join(f"{v} {row[v]['mm']:.4f} mm ({row[v]['ratio']:.3f})" for v in VARIANTS), flush=True)
    (REPO / "docs/results").mkdir(exist_ok=True)
    json.dump(out, open(REPO / "docs/results/l1_teacher_from_inputs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
