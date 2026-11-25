"""L1 -- generate rope episodes (lsim.data.rope; teacher lsim.teacher.rope_teacher).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_generate_rope.py --out data/datasets/r1 --episodes 20000 --worlds 128 [--part k/n]
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_generate_rope.py --out data/datasets/r1_test --episodes 1024 --split test --seed 1
    long episodes (20 s, 4-8 actions, idle gaps):
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_generate_rope.py --out data/datasets/r1_long --episodes 600 --worlds 150 \
        --seconds 20 --actions 4-8 --idle 0.35,1,3 --own_settle --seed 10

Batches of `--worlds` scenes (a different rope, material, start and obstacles in every world), each a shard in --out
in the cloth shards' layout: shard_<i>.npz (rope nodes under cloth_c / cloth_d / cloth_off, gripper poses, grasp and
contact forces) + shard_<i>.json (kind "rope"; per world the rope and its material -- the simulated, capped EA --,
how it was drawn, the start, action families, grasp events, a quality record). Obstacles from the training assets
(the held-out 10% never appear), or with --split test the held-out ones only (use a different --seed).
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
    ap.add_argument("--out", default="data/datasets/r1")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--worlds", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="0/1", help="k/n: this process makes shards i with i % n == k")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--objects", default="gso", choices=("gso", "simready", "both"),
                    help="obstacle library: GSO scans (default, today's scenes), SimReady table objects, or both")
    ap.add_argument("--seconds", type=float, default=None, help="recorded seconds per episode (default: the module's RECORD, 4)")
    ap.add_argument("--actions", default=None, help="lo-hi: long episodes, lo-hi chained action families each (default: 1-2)")
    ap.add_argument("--idle", default=None, help="p,lo,hi: after each family, with probability p, a further lo-hi s idle")
    ap.add_argument("--own_settle", action="store_true", help="each world settles for its own start's time (plank starts "
                    "otherwise slide unrecorded during the batch's longest settling; off: as r1)")
    ap.add_argument("--fine_crossings", action="store_true", help="count self-crossings per chunk (~+40%% time)")
    a = ap.parse_args()
    part, parts = map(int, a.part.split("/"))
    acts = tuple(int(v) for v in a.actions.split("-")) if a.actions else None
    idle = tuple(float(v) for v in a.idle.split(",")) if a.idle else None
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    import warp as wp
    wp.config.quiet = True
    from lsim.data import library
    from lsim.data.rope import RECORD, quality, run_batch, sample_spec, save_shard
    wp.init()
    train, test = library.object_split(a.objects)
    (out / "assets_split.json").write_text(json.dumps({"objects": a.objects, "train": train, "test": test}))
    n_shards = int(np.ceil(a.episodes / a.worlds))
    GROUP = 4                                          # shards drawn together, then sorted by stiffness: a batch
    for i in range(part, n_shards, parts):            # runs at the step of its stiffest rope (sqrt(k / m))
        if (out / f"shard_{i:05d}.npz").exists():
            continue
        g = i // GROUP
        seed = a.seed * 1_000_003 + g
        rng = np.random.default_rng(seed)
        pool = [sample_spec(rng, train if a.split == "train" else test, seed=seed * 10000 + k, actions=acts)
                for k in range(GROUP * a.worlds)]
        pool.sort(key=lambda sp: sp.rope.k_over_m)
        specs = pool[(i % GROUP) * a.worlds:(i % GROUP + 1) * a.worlds]
        seed = seed * GROUP + i % GROUP
        t0 = time.time()
        res = run_batch(specs, seed=seed, log=lambda m: None, fine_crossings=a.fine_crossings, record=a.seconds or RECORD,
                        idle=idle, keep=acts is not None, own_settle=a.own_settle)
        qc = quality(res, crossings=False)
        save_shard(out, i, res, seed, qc=qc)
        print(f"shard {i + 1}/{n_shards}: {a.worlds} episodes in {time.time() - t0:.0f} s, {res['substeps']} substeps, "
              f"fastest node {max(q['vmax'] for q in qc):.2f} m/s, exploded {sum(q['exploded'] for q in qc)}, "
              f"crossed {sum(q.get('crossings_fine', 0) > 0 for q in qc) if a.fine_crossings else 'n/a'}, largest contact / weight "
              f"{max(max(q['contact_over_weight']) for q in qc):.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
