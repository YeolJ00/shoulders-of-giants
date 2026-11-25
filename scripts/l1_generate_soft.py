"""L1 -- generate soft-body episodes (lsim.data.soft, teacher lsim.teacher.soft_teacher).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_generate_soft.py --out data/datasets/s1 --episodes 20000 --worlds 64 [--part k/n]
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_generate_soft.py --out data/datasets/s1_test --episodes 1024 --split test --seed 1

Batches of `--worlds` scenes (a different body, material and scene in every world), each a shard in --out:
shard_<i>.npz (every tet vertex, gripper poses and openings, grasp, contact and per-slot forces, the meshes) +
shard_<i>.json (per world: the body and its material, scene, families, events, guards, quality). Obstacles from the
training assets (the held-out 10% never appear), or with --split test the held-out ones only (use another --seed).
Resumable by shard index.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/datasets/s1")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="0/1", help="k/n: this process makes shards i with i % n == k")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--objects", default="gso", choices=("gso", "simready", "both"),
                    help="obstacle library: GSO scans (default, today's scenes), SimReady table objects, or both")
    a = ap.parse_args()
    part, parts = map(int, a.part.split("/"))
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    import warp as wp
    from lsim.data import library
    from lsim.data.soft import run_batch, sample_spec, save_shard
    wp.init()
    train, test = library.object_split(a.objects)
    (out / "assets_split.json").write_text(json.dumps({"objects": a.objects, "train": train, "test": test}))
    n_shards = int(np.ceil(a.episodes / a.worlds))
    GROUP = 4                                          # shards drawn together, then sorted by stable step: a batch
    for i in range(part, n_shards, parts):            # runs at the step of its hardest body
        if (out / f"shard_{i:05d}.npz").exists():
            continue
        g = i // GROUP
        seed = a.seed * 1_000_003 + g
        rng = np.random.default_rng(seed)
        pool = [sample_spec(rng, train if a.split == "train" else test, seed=seed * 10000 + k) for k in range(GROUP * a.worlds)]
        pool.sort(key=lambda sp: sp.soft.stable_dt())
        specs = pool[(i % GROUP) * a.worlds:(i % GROUP + 1) * a.worlds]
        seed = seed * GROUP + i % GROUP
        t0 = time.time()
        res = run_batch(specs, seed=seed, log=lambda m: None)
        qc = save_shard(out, i, res, seed)
        print(f"shard {i + 1}/{n_shards}: {a.worlds} episodes in {time.time() - t0:.0f} s, {res['substeps']} substeps, "
              f"fastest vertex {max(q['vmax'] for q in qc):.2f} m/s, exploded {sum(q['exploded'] for q in qc)}, "
              f"inverted {sum(q['inverted'] for q in qc)}, largest contact / weight "
              f"{max(max(q['contact_over_weight']) for q in qc):.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
