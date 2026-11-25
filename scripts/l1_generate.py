"""L1 -- generate teacher episodes (dataset v0: rigid + hinged; PHASE1_AMENDMENT.md).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l1_generate.py --out data/datasets/v0 --episodes 50000 --worlds 128 [--seconds 6] [--part k/n]

Batches of `--worlds` scenes (different objects in every world), each a shard in --out:
shard_<i>.npz (arrays) + shard_<i>.json (per-world specs, labels, actions, pokes, caps).
Training assets only (the held-out 10% of names never appear), or with --split test the
held-out ones only (evaluation sets; use a different --seed). Resumable by shard index.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def save_shard(out, i, res, seed, p_ramp=0.0):
    W, K = len(res["specs"]), res["K"]
    F, P = res["contact_F"], res["contact_P"]
    nz = np.argwhere(np.linalg.norm(F, axis=-1) > 1e-6)              # sparse: (frame, world, a, b)
    wr = res["poke_wrench"]
    wz = np.argwhere(np.abs(wr).sum(-1) > 0)
    tmp = out / f"shard_{i:05d}.tmp.npz"                               # the npz appears last, whole
    np.savez_compressed(tmp,
                        body_q=res["body_q"], body_qd=res["body_qd"], body_world=res["body_world"],
                        body_slot=res["body_slot"], body_mass=res["body_mass"], body_com=res["body_com"],
                        body_inertia=res["body_inertia"], contact_idx=nz.astype(np.int16),
                        contact_F=F[tuple(nz.T)], contact_P=P[tuple(nz.T)], contact_T=res["contact_T"][tuple(nz.T)],
                        poke_idx=wz.astype(np.int32), poke_wrench=wr[tuple(wz.T)],
                        force_cap=res["force_cap"], saturated=res["saturated"])
    meta = {"seed": seed, "K": K, "substeps": res["substeps"], "fps": 30, "p_ramp": p_ramp, "worlds": []}
    for sp, inf, (path, fams, pokes), fs in zip(res["specs"], res["infos"], res["plans"], res["families_sampled"]):
        meta["worlds"].append({"families_sampled": fs,
            "spec": [{"name": o.name, "xy": list(o.xy), "yaw": o.yaw, "on_top_of": o.on_top_of,
                      "drop_height": o.drop_height, "on_ramp": o.on_ramp} for o in sp.objects], "hinge_xy": list(sp.hinge_xy),
            "ramp": sp.ramp,
            "objects": inf["labels"], "object_bodies": inf["objects"], "gripper_body": inf["gripper"],
            "hinge": inf.get("hinge"), "families": fams, "pokes": pokes,
            "gripper_keyframes": [[k[0], *k[1].tolist(), k[2]] for k in path.k]})
    (out / f"shard_{i:05d}.json").write_text(json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, out / f"shard_{i:05d}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/datasets/v0")
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--worlds", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="0/1", help="k/n: this process makes shards i with i % n == k")
    ap.add_argument("--split", default="train", choices=("train", "test"), help="test: held-out objects only")
    ap.add_argument("--p_ramp", type=float, default=0.0, help="fraction of scenes with a plank to slide on")
    ap.add_argument("--reverse", action="store_true", help="work from the last shard down (a second process per GPU)")
    ap.add_argument("--mass_scale", type=float, default=1.0, help="G3 test sets: every object x this (3 or 1/3)")
    ap.add_argument("--poke_scale", type=float, default=1.0, help="G3 test sets: unseen pokes' velocity change x this")
    a = ap.parse_args()
    part, parts = map(int, a.part.split("/"))
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    import warp as wp
    from lsim.data import library
    from lsim.data.episode import run_batch
    from lsim.data.scene import sample_scene
    wp.init()
    train, test = library.split(library.tabletop())
    (out / "assets_split.json").write_text(json.dumps({"train": train, "test": test}))
    n_shards = int(np.ceil(a.episodes / a.worlds))
    order = list(range(part, n_shards, parts))
    for i in (order[::-1] if a.reverse else order):
        if (out / f"shard_{i:05d}.npz").exists():
            continue
        seed = a.seed * 1_000_003 + i
        rng = np.random.default_rng(seed)
        specs = [sample_scene(rng, train if a.split == "train" else test, seed=seed * 1000 + k, p_ramp=a.p_ramp)
                 for k in range(a.worlds)]
        for sp in specs:                                               # outside the training range (G3)
            for o in sp.objects:
                o.mass_scale = a.mass_scale
        t0 = time.time()
        res = run_batch(specs, seconds=a.seconds, seed=seed, log=lambda m: None, poke_scale=a.poke_scale)
        save_shard(out, i, res, seed, a.p_ramp)
        print(f"shard {i + 1}/{n_shards}: {a.worlds} episodes in {time.time() - t0:.0f} s, "
              f"gripper at its force cap in {res['saturated'].mean():.1%} of frames", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
