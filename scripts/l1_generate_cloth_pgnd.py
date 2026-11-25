"""L1 -- teacher cloth episodes from PGND's real training windows (c1_pgnd): coverage of two-gripper, 50-63 cm cloth.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_generate_cloth_pgnd.py [--materials 8 --part 0/2]

c1's episodes have one gripper and towels of 15-50 cm; PGND's robot holds a 50-63 cm cloth by two points of an edge
for the whole window, and v3 drifts from its own teacher there by 37 / 62 / 139 mm at 1 / 2 / 4 s. Per box-free
training window (the test recordings never used): `materials` materials drawn from the dataset's prior (log-uniform
in lsim.real.replay.RANGE, the wave-speed bound applied), the teacher run from the window's fitted frame 7 with both
held patches on the robot's recorded path to frame 120 (lsim.real.replay.simulate). Frames 0-6 are the fitted real
frames (the history), frames 7-120 the teacher's: every vertex known every frame, as in c1. A world counts as
exploded when a vertex leaves 2 m of the start (rule 3) and is dropped.

Writes data/datasets/c1_pgnd/<case>.npz (x (M, 121, V, 3) float32, materials (M, 4) w Y c mu) and .json; resumable.
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/datasets/c1_pgnd"
T0 = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", type=int, default=8)
    ap.add_argument("--part", default="0/1", help="i/n: every n-th window from the i-th")
    a = ap.parse_args()
    from lsim.real import replay
    from lsim.real.pgnd import load_pgnd
    i, n = map(int, a.part.split("/"))
    eps = [e for e in load_pgnd("train") if e.n_frames == 121][i::n]
    OUT.mkdir(parents=True, exist_ok=True)
    lo = np.log([replay.RANGE[k][0] for k in ("w", "Y", "c", "mu")])
    hi = np.log([replay.RANGE[k][1] for k in ("w", "Y", "c", "mu")])
    import time
    t0 = time.time()
    for k, ep in enumerate(eps):
        case = ep.meta["case"]
        if (OUT / f"{case}.json").exists():
            continue
        rng = np.random.default_rng(zlib.crc32(case.encode()))
        mats = [replay.decode(lo + (hi - lo) * rng.random(4)) for _ in range(a.materials)]
        Xw = replay.simulate(ep, mats, T0, ep.n_frames - 1 - T0, log=lambda *x: None)       # (M, 114, V, 3)
        x = np.concatenate([np.broadcast_to(ep._x[:T0], (len(mats),) + ep._x[:T0].shape), Xw], 1).astype(np.float32)
        ok = np.isfinite(x).all(axis=(1, 2, 3)) & (np.linalg.norm(x - x[:, T0:T0 + 1], axis=-1).max(axis=(1, 2)) < 2.0)
        np.savez(OUT / f"{case}.npz", x=x[ok],
                 materials=np.array([[m["w"], m["Y"], m["c"], m["mu"]] for m in mats], np.float32)[ok])
        (OUT / f"{case}.json").write_text(json.dumps({"case": case, "materials": [m for m, o in zip(mats, ok) if o],
                                                      "exploded": int((~ok).sum()), "t0": T0}))
        print(f"[{k + 1}/{len(eps)}] {case}: {ok.sum()} worlds ({(~ok).sum()} exploded)  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
