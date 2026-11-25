"""L1 -- the cloth identifiability gate (PHASE1_AMENDMENT.md applied to cloth): does each cloth parameter slot
change what the model sees?

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_cloth_gate.py --data data/datasets/c1 [--shards 0,1,2,3 --per_shard 16]

What is read in context is the cloth's response to an action, so the test is per action: from the recorded
state at the moment the gripper first touches the cloth (positions, velocities from the last frame, the
held vertices) -- and, for a cloth starting on the plank, from the first recorded frame -- one second is
simulated again -- twice unchanged (the noise: the teacher is not
bit-reproducible) and once per parameter changed by its prior's half-width in log space (up or down at
random): areal density x/ 3.16, stretch modulus x/ 10, bending stiffness x/ 6.1 (bending length 1.5-5 cm),
friction x/ 3.46 (0.1-1.2). The gripper's path and which cloth it holds are given, so they are the recorded ones.
  G   force slot given (70% of grasps): identified if, at some frame of the second, the motion (mean vertex
      distance) differs by more than 5 x max(1 mm, the noise at that frame) OR the grasp-force impulse so far
      by more than 5 x max(0.005 N s, its noise) -- frame by frame, because the noise is chaos: two runs from
      the same state differ by 0.03 mm after two frames and 5 mm after one second (median)
  H   force slot hidden (real hands): the motion only
Whole-episode replays are not the test: a towel that falls or crumples amplifies any difference, the same
episode replayed twice ended 21-33 mm apart (median), and re-selecting the cloth under the fingertips split
replays by 11-49 mm within a second of the grasp. Episodes come from one stiffness group (the generator
sorts each 4 shards by stiffness).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l1_cloth_gate.json"
FACTORS = {"w": np.sqrt(10.0), "Y": 10.0, "B": np.sqrt((5.0 / 1.5) ** 3), "mu": np.sqrt(1.2 / 0.1)}


def episode_x(npz, off, w):
    return npz["cloth_d"][:, off[w]:off[w + 1]].astype(np.float64) + npz["cloth_c"][:, w:w + 1]


def windows(npz, meta, eps, length=30, per_episode=2):
    """Per episode, up to `per_episode` windows: (episode, start frame, start state, path, events), the start
    being the first recorded frame at which the gripper touches the cloth after an action begins."""
    off, settle = npz["cloth_off"], meta["settle_frames"]
    out = []
    for e in eps:
        wd = meta["worlds"][e]
        x = episode_x(npz, off, e)
        touch = np.linalg.norm(npz["grasp_F"][:, e], axis=-1) + np.linalg.norm(npz["contact_F"][:, e, 1], axis=-1)
        n = 0
        starts = []
        if wd["init"] == "ramp":                         # on the plank it slides or sticks by itself from the start
            starts.append((1, "ramp"))
        for fam in wd["families"]:
            s0 = max(1, fam["start"] - settle)
            hit = np.flatnonzero(touch[s0:] > 1e-4)
            if len(hit):
                starts.append((s0 + int(hit[0]), fam["family"]))
        for f, family in starts:
            if f + length >= len(x) or n >= per_episode:
                continue
            ev, held = [], None
            for g in wd["events"]:                       # events on the whole timeline -> window frames
                k = g[0] - settle - f
                if g[1] == "grasp" and k <= 0:
                    held = (0, "grasp", g[2], g[3] if len(g) > 3 else [])
                elif g[1] == "release" and k <= 0:
                    held = None
                elif 0 < k <= length:
                    ev.append((k, *g[1:]))
            if held is not None and len(held[3]):
                ev = [held] + ev
            out.append({"episode": e, "frame": f, "x": x[f], "v": (x[f] - x[f - 1]) * meta["fps"],
                        "path": npz["grip_path"][settle + f:settle + f + length + 1, e], "events": ev,
                        "family": family})
            n += 1
    return out


def run_windows(meta, wins, change=None, rng=None, length=30):
    from lsim.data.cloth import run_batch, spec_from_meta
    specs, factors = [], []
    for wn in wins:
        sp = spec_from_meta(meta["worlds"][wn["episode"]])
        f = 1.0
        if change is not None:
            f = float(FACTORS[change] ** rng.choice([-1, 1]))
            setattr(sp.cloth, change, getattr(sp.cloth, change) * f)
            if change == "Y" and sp.cloth.Y / sp.cloth.w > 3.0e5:      # keep the explicit step affordable
                sp.cloth.Y = 3.0e5 * sp.cloth.w
        specs.append(sp)
        factors.append(f)
    res = run_batch(specs, seed=0, log=lambda m: None, replay=[{"path": wn["path"], "events": wn["events"]} for wn in wins],
                    settle=0, record=length / meta["fps"], start=[{"x": wn["x"], "v": wn["v"]} for wn in wins])
    out = []
    for k in range(len(wins)):
        p0, n = res["worlds"][k]["particles"]
        out.append({"x": res["x"][:, p0:p0 + n], "F": res["grasp_F"][:, k], "factor": factors[k]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/c1")
    ap.add_argument("--shards", default="0,1,2,3", help="one stiffness group: the generator sorts each 4 shards")
    ap.add_argument("--per_shard", type=int, default=16)
    a = ap.parse_args()
    import warp as wp
    wp.init()
    rng = np.random.default_rng(0)
    rows, noises, n_win = defaultdict(list), [], 0
    for sh in map(int, a.shards.split(",")):
        path = REPO / a.data / f"shard_{sh:05d}"
        npz = dict(np.load(path.with_suffix(".npz")))
        meta = json.loads(path.with_suffix(".json").read_text())
        wins = windows(npz, meta, list(range(min(a.per_shard, len(meta["worlds"])))))
        if not wins:
            continue
        n_win += len(wins)
        dt = 1.0 / meta["fps"]

        def dist(r0, r1):
            """Mean vertex distance per frame, and the grasp-force impulse difference up to each frame."""
            m = np.linalg.norm(r0["x"] - r1["x"], axis=-1).mean(-1)
            f = np.cumsum(np.linalg.norm(r0["F"] - r1["F"], axis=-1)) * dt
            return m, f

        base, again = run_windows(meta, wins), run_windows(meta, wins)
        noise = [dist(b_, a_) for b_, a_ in zip(base, again)]
        noises += noise
        print(f"shard {sh}: {len(wins)} windows, noise motion median at 0.33 / 1 s {1e3 * np.median([n[0][10] for n in noise]):.2f} / "
              f"{1e3 * np.median([n[0][-1] for n in noise]):.2f} mm", flush=True)
        for p in FACTORS:
            pert = run_windows(meta, wins, change=p, rng=rng)
            for wn, b_, q_, (nm, nf) in zip(wins, base, pert, noise):
                m, f = dist(b_, q_)
                # chaos grows (0.03 mm at two frames, 5 mm at one second): beat the noise at the same moment
                h = bool((m > 5.0 * np.maximum(1.0e-3, nm)).any())
                gf = bool((f > 5.0 * np.maximum(0.005, nf)).any())
                k = int(np.argmax(m / (5.0 * np.maximum(1.0e-3, nm))))
                rows[p].append((h or gf, h, 1e3 * float(m[k]), float(f[-1]), wn["family"], meta["worlds"][wn["episode"]]["cloth"]))
    curve = np.array([n[0] for n in noises])
    result = {"windows": n_win, "shards": a.shards,
              "noise_motion_mm_median_per_frame": (1e3 * np.median(curve, 0)).tolist(),
              "noise_motion_mm_p90_per_frame": (1e3 * np.percentile(curve, 90, 0)).tolist(),
              "noise_force_Ns_median": float(np.median([n[1][-1] for n in noises])), "params": {}}
    print("noise (the same second twice), median mm at frames 1/2/5/10/20/30: "
          + " / ".join(f"{1e3 * np.median(curve[:, k]):.2f}" for k in (1, 2, 5, 10, 20, 30)), flush=True)
    for p, rs in rows.items():
        fam = defaultdict(list)
        for r in rs:
            fam[r[4]].append(r[0])
        G, H = float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))
        result["params"][p] = {"identified_G": G, "identified_H": H, "motion_mm_median": float(np.median([r[2] for r in rs])),
                               "force_Ns_median": float(np.median([r[3] for r in rs])),
                               "per_family_G": {k: [float(np.mean(v)), len(v)] for k, v in fam.items()}}
        if p == "Y":                                     # stretch shows only on soft cloth
            soft = [r[0] for r in rs if r[5]["Y"] < 2.0e3]
            result["params"][p]["identified_G_Y_below_2000"] = [float(np.mean(soft)) if soft else None, len(soft)]
        print(f"{p:3s} x/{FACTORS[p]:.2f}: identified G {G:.2f} H {H:.2f} | motion change at its clearest, median {np.median([r[2] for r in rs]):.1f} mm, "
              f"force impulse change median {np.median([r[3] for r in rs]):.4f} N s | per family "
              + ", ".join(f"{k} {np.mean(v):.2f} ({len(v)})" for k, v in sorted(fam.items())), flush=True)
    OUT.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
