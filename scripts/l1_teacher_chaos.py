"""The replay floor: how far is the TEACHER from its own episode after one imperfect frame? (2026-11-22)

A network that fits one frame to 0.10 of constant velocity replays its memorised episode to 5 / 16 / 50-90 mm at
1 s / 2 s / 3.8 s (--own_up), and the late numbers differ run to run. Is that the network, or the cloth? Here the
teacher is restarted at frame t0 of its own recorded episode (run_batch start= / replay=, the same scene, gripper path,
events and substeps, as l1_teacher_from_inputs) from a state that is one frame's error off, and run exactly to the end:
  none         the recorded state (the restart's own float-order floor)
  iid S        every unheld vertex moved by Gaussian noise, S per axis (10 um, 100 um)
  shrink s     x - s a, v - s a FPS with a = x_t - 2 x_{t-1} + x_{t-2}: one frame of a predictor that gives (1 - s) of
               the true acceleration -- the networks' error is this shrinkage (regression 0.645 fully trained: s = 0.35;
               a memorised episode at 0.10: s = 0.1)
Per episode and frame: the median distance of unheld vertices from the recorded episode (the replay metric of
l2_small_fit / l2_coverage_score), beside a cloth left still at frame t0. Whatever a variant reaches is what a perfect
simulator does after ONE such frame; a network making that error every frame cannot be asked for less.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_teacher_chaos.py

--shard data/datasets/c1/shard_00000 --episodes N: the same for the first N episodes of a dataset shard (the small-fit
test's own episodes; l2_small_fit replays them from frame 7). c1 records no velocities, so the restart's velocity is
the backward difference (x_7 - x_6) fps -- itself an imperfect first frame on cloth that already moves (1.2-2.2 mm on
the three held-and-shaken episodes of the 16, 0.00-0.11 on the rest) -- and the shard's own substep count is kept.
--copies exact restarts side by side: what they differ by is float-order noise alone. Result (2026-11-22, 16 episodes):
the teacher ends 131 / 131 / 242 mm from its own recording of episode 0 (2.5 mm at 1 s, 2-6 at 2 s), and over 40 mm on 6
of 16; docs/results/l1_teacher_chaos_<shard>.json holds every curve.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from l1_teacher_from_inputs import FPS, held_at

REPO = Path(__file__).resolve().parents[1]


def recorded(a):
    """The first a.episodes of a dataset shard, restarted at frame a.t0 from recorded positions."""
    import copy
    from lsim.data.cloth import run_batch, spec_from_meta
    from lsim.train.cloth_windows import load_shard
    base = str((REPO / a.shard).resolve()) if not Path(a.shard).is_absolute() else a.shard
    eps = load_shard(base)[:a.episodes]
    npz, meta = np.load(base + ".npz"), json.load(open(base + ".json"))
    settle, fps, t = meta["settle_frames"], meta["fps"], a.t0
    var = [("exact", 0.0)] * a.copies + [(f"iid {u} um", 1e-6 * float(u)) for u in a.iid_um.split(",")]
    rng = np.random.default_rng(a.seed)
    specs, replay, start, info = [], [], [], []
    for e in eps:
        w, wd = int(e.w), meta["worlds"][int(e.w)]
        X = np.asarray(e.x(list(range(e.n_frames)))).astype(np.float64)
        n = e.n_frames - 1 - t
        ev, grasp = [], None
        for g in wd["events"]:                                        # events on the whole timeline -> frames from t
            k = g[0] - settle - t
            if g[1] == "grasp" and k <= 0:
                grasp = (0, "grasp", g[2], g[3] if len(g) > 3 else [])
            elif g[1] == "release" and k <= 0:
                grasp = None
            elif 0 < k <= n:
                ev.append((k, *g[1:]))
        if grasp is not None and len(grasp[3]):
            ev = [grasp] + ev
        v0 = (X[t] - X[t - 1]) * fps
        for name, sg in var:
            specs.append(spec_from_meta(copy.deepcopy(wd)))
            x, fr = X[t].copy(), ~e.held[t]
            x[fr] += sg * rng.standard_normal((int(fr.sum()), 3))
            start.append({"x": x, "v": v0})
            replay.append({"path": npz["grip_path"][settle + t:settle + t + n + 1, w], "events": ev})
        info.append((X, e, n, wd))
    res = run_batch(specs, seed=0, log=lambda m: None, replay=replay, settle=0, record=max(i[2] for i in info) / fps,
                    start=start, substeps=meta["substeps"])
    out = {"shard": a.shard, "t0": t, "substeps": int(res["substeps"]), "variants": [v[0] for v in var], "episodes": []}
    k = 0
    for X, e, n, wd in info:
        free = ~e.held[t + 1:t + 1 + n]
        med = lambda P: [1e3 * float(np.median(np.linalg.norm(P[j] - X[t + 1 + j], axis=-1)[free[j]])) if free[j].any()
                         else float("nan") for j in range(n)]
        row = {"w": int(e.w), "init": wd["init"], "families": [f["family"] for f in wd["families"]],
               "speed_mm_s": 1e3 * float(np.median(np.linalg.norm(X[t] - X[t - 1], axis=-1)) * fps),
               "held": [int(h.sum()) for h in e.held[t + 1:t + 1 + n]], "still": med(np.repeat(X[t:t + 1], n, 0)), "curves": []}
        print(f"episode {row['w']:2d} {row['init']:9s} {'+'.join(row['families']):22s} still 1 s {row['still'][29]:6.1f}  "
              f"2 s {row['still'][59]:6.1f}  end {row['still'][-1]:6.1f} mm")
        for name, sg in var:
            p0, nn = res["worlds"][k]["particles"]
            c = med(res["x"][1:n + 1, p0:p0 + nn].astype(np.float64))
            row["curves"].append(c)
            print(f"    {name:11s} 1 frame {c[0]:6.3f}  10 frames {c[9]:6.3f}  1 s {c[29]:7.2f}  2 s {c[59]:7.2f}  end {c[-1]:7.2f} mm")
            k += 1
        out["episodes"].append(row)
    ex = np.array([[r["curves"][j][i] for i in (29, 59, -1)] for r in out["episodes"] for j in range(a.copies)])
    print(f"exact restarts over {len(eps)} episodes x {a.copies}: 1 s / 2 s / end  median {np.median(ex, 0).round(2).tolist()}  "
          f"mean {ex.mean(0).round(1).tolist()}  over 40 mm at the end in {int((ex[:, 2] > 40).sum())} of {len(ex)}")
    json.dump(out, open(REPO / f"docs/results/l1_teacher_chaos_{Path(a.shard).parent.name}_{Path(a.shard).name}.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="", help="a dataset shard (no extension): restart its first --episodes instead")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--t0", type=int, default=7)
    ap.add_argument("--iid_um", default="10,100")
    ap.add_argument("--shrink", default="0.1,0.35")
    a = ap.parse_args()
    if a.shard:
        return recorded(a)
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
    F, W, t = X.shape[0] - 1, a.worlds, a.t0
    n_fr = F - t
    print(f"recorded {W} episodes x {F} frames, {sub0} substeps per frame, in {time.time() - t0:.0f} s", flush=True)
    variants = [("none", None)] + [(f"iid {u} um", ("iid", 1e-6 * float(u))) for u in a.iid_um.split(",")] + \
        [(f"shrink {s}", ("shrink", float(s))) for s in a.shrink.split(",")]
    T = settle + t
    jit = np.random.default_rng(a.seed + 1)
    start, replay, held = [], [], []
    for w in range(W):
        p0, n = worlds[w]["particles"]
        h, ev = held_at(events[w], T, n)
        replay.append({"path": path[T:T + n_fr + 1, w], "events": ev})
        held.append(np.stack([held_at(events[w], settle + k, n)[0] for k in range(t, F)]))    # (n_fr, n): step k -> k + 1
    acc = X[t] - 2 * X[t - 1] + X[t - 2]
    first = []
    for name, kind in variants:
        for w in range(W):
            p0, n = worlds[w]["particles"]
            x, v, free = X[t, p0:p0 + n].copy(), V[t, p0:p0 + n].copy(), ~held[w][0]
            if kind and kind[0] == "iid":
                x[free] += kind[1] * jit.standard_normal((int(free.sum()), 3))
            elif kind:
                x[free] -= kind[1] * acc[p0:p0 + n][free]
                v[free] -= kind[1] * acc[p0:p0 + n][free] * FPS
            start.append({"x": x, "v": v})
            first.append(1e3 * float(np.median(np.linalg.norm(x - X[t, p0:p0 + n], axis=-1)[free])) if free.any() else 0.0)
    nv = len(variants)
    r = run_batch(specs * nv, seed=a.seed, record=n_fr / FPS, log=lambda m: None, replay=replay * nv, settle=0, start=start)
    assert r["substeps"] == sub0, (r["substeps"], sub0)
    Y = r["x"].astype(np.float64)                                     # (n_fr + 1, all vertices of nv copies, 3)
    del r
    tot = sum(wd["particles"][1] for wd in worlds)
    out = {"worlds": W, "t0": t, "frames": n_fr, "substeps": int(sub0), "variants": {}}

    def curve(P, w):                                                  # P (n_fr, n, 3): frames t + 1 .. F of world w
        p0, n = worlds[w]["particles"]
        d = np.linalg.norm(P - X[t + 1:F + 1, p0:p0 + n], axis=-1)
        fr = ~held[w]
        return np.array([1e3 * np.median(d[k][fr[k]]) if fr[k].any() else np.nan for k in range(n_fr)])

    rows = {"still": np.stack([curve(np.repeat(X[t:t + 1, worlds[w]["particles"][0]:sum(worlds[w]["particles"])], n_fr, 0), w)
                               for w in range(W)])}
    for k, (name, _) in enumerate(variants):
        rows[name] = np.stack([curve(Y[1:, k * tot + worlds[w]["particles"][0]:k * tot + sum(worlds[w]["particles"])], w)
                               for w in range(W)])
    print(f"median distance of unheld vertices from the recorded episode, mm, over {W} episodes: mean (median, p90)")
    print(f"{'':14s}{'start':>9s}" + "".join(f"{s:>26s}" for s in ("1 frame", "10 frames", "1 s", "2 s", f"end ({n_fr / FPS:.1f} s)")))
    for k, (name, c) in enumerate(rows.items()):
        st = np.mean(first[(k - 1) * W:k * W]) if k else float("nan")
        cells = {}
        line = f"{name:14s}{st:9.4f}"
        for lab, i in (("f1", 0), ("f10", 9), ("e30", 29), ("e60", 59), ("end", n_fr - 1)):
            col = c[:, i]
            cells[lab] = {"mean": float(np.nanmean(col)), "median": float(np.nanmedian(col)), "p90": float(np.nanquantile(col, 0.9))}
            line += f"   {cells[lab]['mean']:7.2f} ({cells[lab]['median']:6.2f}, {cells[lab]['p90']:6.1f})"
        out["variants"][name] = {"start_mm": float(st), **cells, "curve_mean": np.nanmean(c, 0).tolist(),
                                 "curve_median": np.nanmedian(c, 0).tolist(), "per_episode_end": c[:, -1].tolist()}
        print(line, flush=True)
    (REPO / "docs/results").mkdir(exist_ok=True)
    json.dump(out, open(REPO / "docs/results/l1_teacher_chaos.json", "w"), indent=1)


if __name__ == "__main__":
    main()
