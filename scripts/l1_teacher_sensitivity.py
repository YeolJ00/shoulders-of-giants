"""How much of the teacher's next frame is set by the material, and is it converged in the substep (2026-11-19)?

Given the teacher's true velocities, the 64M network still predicts free cloth's next frame at 0.80 of constant
velocity's error (0.84 without them; the teacher from its full state 0.01): it fails to learn the one-frame map itself,
not to read the state. Here the teacher is restarted from its full state at recorded frame t for one frame (as
l1_teacher_from_inputs) with one thing changed, and its x_{t+1} compared with the recorded one on moving unheld cloth
vertices, against constant velocity's error on the same vertices. The error is then the teacher's response to the
change:
  as is            the restart itself (reproduces to float noise, 0.01)
  substeps x2      the same physics at half the time step: is the recorded map converged in dt
  Y x0.9 / x0.5    stretch modulus
  w x1.1           areal density
  B x2             bending stiffness
  swap             the whole material (w, Y, nu, B, mu) of another episode in the batch: a draw from the prior
  back2            the velocity from 3 frame positions (l1_teacher_from_inputs' 0.25), true material
  back2 + swap     the model's information without the material
The substep count is held at the recorded one (run_batch substeps=) except for "substeps x2". Set against
l2_error_where --material (how far the network's prediction moves when its given material is hidden or swapped).

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_teacher_sensitivity.py
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from l1_teacher_from_inputs import FPS, held_at, on_surface, velocities

REPO = Path(__file__).resolve().parents[1]
# (name, velocity, material change (cloth, other cloth) -> cloth, substep multiplier); grouped by batch
BATCHES = [
    [("as is", "true", None, 1), ("Y x0.9", "true", lambda c, o: replace(c, Y=0.9 * c.Y), 1),
     ("Y x0.5", "true", lambda c, o: replace(c, Y=0.5 * c.Y), 1), ("w x1.1", "true", lambda c, o: replace(c, w=1.1 * c.w), 1)],
    [("B x2", "true", lambda c, o: replace(c, B=2.0 * c.B), 1),
     ("swap", "true", lambda c, o: replace(c, w=o.w, Y=o.Y, nu=o.nu, B=o.B, mu=o.mu), 1),
     ("back2", "back2", None, 1), ("back2 + swap", "back2", lambda c, o: replace(c, w=o.w, Y=o.Y, nu=o.nu, B=o.B, mu=o.mu), 1)],
    [("substeps x2", "true", None, 2)],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_t", type=int, default=16)
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
    print(f"recorded {a.worlds} episodes x {X.shape[0] - 1} frames, {sub0} substeps per frame, in {time.time() - t0:.0f} s",
          flush=True)
    W = a.worlds
    other = [(w + W // 2) % W for w in range(W)]                       # the swap's donor: another episode's material
    F = X.shape[0] - 1
    ts = np.unique(np.linspace(8, F - 2, a.n_t).round().astype(int))
    names = [v[0] for B in BATCHES for v in B]
    err = {v: {"surface": [], "off": []} for v in names + ["cv"]}
    for t in ts:
        T = settle + t
        vel = velocities(X, V, t)
        replay, sel = [], []
        for w in range(W):
            p0, n = worlds[w]["particles"]
            h, ev = held_at(events[w], T, n)
            replay.append({"path": path[T:T + 2, w], "events": ev})
            moving = np.linalg.norm(X[t, p0:p0 + n] - X[t - 1, p0:p0 + n], axis=-1) * FPS > 0.02
            sel.append(moving & ~h)
        tgt = [X[t + 1, worlds[w]["particles"][0]:sum(worlds[w]["particles"])] for w in range(W)]
        for w in range(W):
            p0, n = worlds[w]["particles"]
            s = sel[w]
            if s.any():
                surf = on_surface(X[t, p0:p0 + n][s], specs[w])
                cv = np.linalg.norm(2 * X[t, p0:p0 + n][s] - X[t - 1, p0:p0 + n][s] - tgt[w][s], axis=-1)
                err["cv"]["surface"].append(cv[surf]); err["cv"]["off"].append(cv[~surf])
        for B in BATCHES:
            bspecs, start = [], []
            for name, vk, change, mult in B:
                for w in range(W):
                    p0, n = worlds[w]["particles"]
                    c = specs[w].cloth if change is None else change(specs[w].cloth, specs[other[w]].cloth)
                    bspecs.append(replace(specs[w], cloth=c))
                    start.append({"x": X[t, p0:p0 + n], "v": vel[vk][p0:p0 + n]})
            r = run_batch(bspecs, seed=a.seed, record=1.0 / FPS, log=lambda m: None, replay=replay * len(B), settle=0,
                          start=start, substeps=sub0 * B[0][3])
            Y = r["x"][1].astype(np.float64)
            tot = sum(wd["particles"][1] for wd in worlds)
            del r
            for k, (name, *_ ) in enumerate(B):
                for w in range(W):
                    p0, n = worlds[w]["particles"]
                    s = sel[w]
                    if not s.any():
                        continue
                    surf = on_surface(X[t, p0:p0 + n][s], specs[w])
                    q0 = p0 + k * tot
                    e = np.linalg.norm(Y[q0:q0 + n][s] - tgt[w][s], axis=-1)
                    err[name]["surface"].append(e[surf]); err[name]["off"].append(e[~surf])
        cvm = np.concatenate([np.concatenate(err["cv"][c][-W:]) for c in ("surface", "off")]).mean()
        print(f"  t {t:3d}: " + "  ".join(f"{v} {np.concatenate([np.concatenate(err[v][c][-W:]) for c in ('surface', 'off')]).mean() / cvm:.3f}"
                                         for v in names) + f"   (constant velocity {1e3 * cvm:.3f} mm)", flush=True)
    out = {"worlds": W, "frames": [int(t) for t in ts], "substeps": int(sub0), "classes": {}}
    for c in ("all", "off", "surface"):
        pick = (lambda d: np.concatenate(d["surface"] + d["off"])) if c == "all" else (lambda d, c=c: np.concatenate(d[c]))
        cv = pick(err["cv"])
        row = {"n": int(len(cv)), "cv_mm": 1e3 * float(cv.mean())}
        for v in names:
            e = pick(err[v])
            row[v] = {"mm": 1e3 * float(e.mean()), "ratio": float(e.mean() / cv.mean()), "p90_mm": 1e3 * float(np.quantile(e, 0.9))}
        out["classes"][c] = row
        print(f"{c:8s} n {row['n']:7d}  constant velocity {row['cv_mm']:.3f} mm  " +
              "  ".join(f"{v} {row[v]['ratio']:.3f}" for v in names), flush=True)
    json.dump(out, open(REPO / "docs/results/l1_teacher_sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
