"""L1 -- generate cloth episodes (phase 3; lsim.data.cloth, teacher lsim.teacher.cloth_teacher).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_generate_cloth.py --out data/datasets/c1 --episodes 20000 --worlds 64 [--part k/n]
    long episodes (20 s, 4-8 actions, idle gaps; families redrawn until they complete):
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_generate_cloth.py --out data/datasets/c1_long --episodes 600 --worlds 75 \
        --seconds 20 --actions 4-8 --idle 0.35,1,3 --own_settle --seed 10

Batches of `--worlds` scenes (a different cloth, material and obstacles in every world), each a shard in
--out: shard_<i>.npz (cloth vertices, gripper poses, grasp and contact forces) + shard_<i>.json (per world:
the cloth and its material, scene, action families, grasp events). Obstacles from the training assets (the
held-out 10% never appear), or with --split test the held-out ones only (use a different --seed).
Resumable by shard index.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/datasets/c1")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="0/1", help="k/n: this process makes shards i with i % n == k")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--objects", default="gso", choices=("gso", "simready", "both"),
                    help="obstacle library: GSO scans (default, today's scenes), SimReady table objects, or both")
    ap.add_argument("--seconds", type=float, default=None, help="recorded seconds per episode (default: the module's RECORD, 4)")
    ap.add_argument("--actions", default=None, help="lo-hi: long episodes, lo-hi chained action families each (default: 1-2)")
    ap.add_argument("--idle", default=None, help="p,lo,hi: after each family, with probability p, a further lo-hi s idle")
    ap.add_argument("--sub", type=int, default=0, help="frame-rate study: record sub instants per 30 fps frame and write the "
                    "SAME episodes at every rate 30 k fps, k a power of two up to sub, to <out>_fps<rate> (30 written last)")
    ap.add_argument("--record_v", action="store_true", help="also write the teacher's instantaneous cloth velocities at "
                    "each recorded frame (cloth_v; the oracle-velocity diagnostic, cloth_train --oracle_v)")
    ap.add_argument("--material", default=None, help="w,Y,c,mu: every cloth gets this material (areal density kg/m^2, "
                    "stretch modulus N/m, bending length m, friction), drawn as usual otherwise -- the fit test at one "
                    "material (2026-11-19)")
    ap.add_argument("--own_settle", action="store_true", help="each world settles for its own start's time (plank starts "
                    "otherwise slide unrecorded during the batch's longest settling; off: as c1)")
    a = ap.parse_args()
    part, parts = map(int, a.part.split("/"))
    acts = tuple(int(v) for v in a.actions.split("-")) if a.actions else None
    idle = tuple(float(v) for v in a.idle.split(",")) if a.idle else None
    out = REPO / a.out
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= a.sub and a.sub % k == 0][::-1] if a.sub > 1 else []
    outs = {k: out.parent / f"{out.name}_fps{30 * k}" for k in ks}
    if ks:
        out = outs[1]                                 # the shard that marks the group done; the split file's place
    for o in [out, *outs.values()]:
        o.mkdir(parents=True, exist_ok=True)
    import warp as wp
    from lsim.data import library
    from lsim.data.cloth import RECORD, quality, run_batch, sample_spec, save_shard
    wp.init()
    train, test = library.object_split(a.objects)
    (out / "assets_split.json").write_text(json.dumps({"objects": a.objects, "train": train, "test": test}))
    n_shards = int(np.ceil(a.episodes / a.worlds))
    GROUP = 4                                          # shards drawn together, then sorted by stiffness: a batch
    for i in range(part, n_shards, parts):            # runs at the step of its hardest cloth (sqrt(m / Y))
        if (out / f"shard_{i:05d}.npz").exists():
            continue
        g = i // GROUP
        seed = a.seed * 1_000_003 + g
        rng = np.random.default_rng(seed)
        pool = [sample_spec(rng, train if a.split == "train" else test, seed=seed * 10000 + k, actions=acts)
                for k in range(GROUP * a.worlds)]
        if a.material:                                # after the draws: the scenes and actions are the seed's own
            from dataclasses import replace
            from lsim.teacher.cloth_teacher import G
            mw, mY, mc, mmu = (float(v) for v in a.material.split(","))
            pool = [replace(sp, cloth=replace(sp.cloth, w=mw, Y=mY, B=mw * G * mc ** 3, mu=mmu)) for sp in pool]
        pool.sort(key=lambda sp: sp.cloth.vertex_mass / sp.cloth.Y)
        specs = pool[(i % GROUP) * a.worlds:(i % GROUP + 1) * a.worlds]
        seed = seed * GROUP + i % GROUP
        t0 = time.time()
        res = run_batch(specs, seed=seed, log=lambda m: None, record=a.seconds or RECORD, idle=idle, fit=acts is not None,
                        keep=acts is not None, own_settle=a.own_settle, sub=a.sub,
                        record_v=a.record_v)
        if acts is not None:                          # long episodes: whether each ends through itself (quality record)
            import torch
            from l0_cloth_teacher_check import _crossings
            from lsim.teacher.cloth_teacher import grid
            res["crossings_end"] = [_crossings(res["x"][-1, p0:p0 + n], grid(wd["cloth"].nx, wd["cloth"].ny, wd["cloth"].h)[1], torch)
                                    for wd in res["worlds"] for p0, n in [wd["particles"]]]
        qc = quality(res)
        if ks:
            from lsim.data.cloth import rate_view
            for k in ks:
                save_shard(outs[k], i, rate_view(res, k), seed, qc=qc)
        else:
            save_shard(out, i, res, seed)
        print(f"shard {i + 1}/{n_shards}: {a.worlds} episodes in {time.time() - t0:.0f} s, {res['substeps']} substeps, "
              f"fastest vertex {max(q['vmax'] for q in qc):.2f} m/s, exploded {sum(q['exploded'] for q in qc)}, "
              f"largest contact / weight {max(max(q['contact_over_weight']) for q in qc):.0f}"
              + (f", ending crossed {sum(k > 0 for k in res['crossings_end'])}" if "crossings_end" in res else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
