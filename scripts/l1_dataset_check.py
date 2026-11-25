"""L1 -- checks on a generated dataset (PHASE1_AMENDMENT.md section 7, besides the gate):
invariants on a sample of shards, seed reproducibility of one shard, and the action-family and
per-object interaction fractions over the sample.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l1_dataset_check.py --data data/datasets/v0 --sample 4 --repro 0
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l1_dataset_check.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/v0")
    ap.add_argument("--sample", type=int, default=4, help="shards to check (spread over the dataset)")
    ap.add_argument("--repro", type=int, default=0, help="shard to regenerate and compare (-1: skip)")
    a = ap.parse_args()
    ds = REPO / a.data
    shards = sorted(ds.glob("shard_*.npz"))
    pick = [shards[int(i)] for i in np.linspace(0, len(shards) - 1, min(a.sample, len(shards)))]
    inv, fam, n_ep, inter = [], Counter(), 0, Counter()
    for sh in pick:
        d = np.load(sh)
        meta = json.loads(sh.with_suffix(".json").read_text())
        q, qd = d["body_q"], d["body_qd"]
        inv.append({"shard": sh.name, "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
                    "max_speed_m_s": float(np.linalg.norm(qd[:, :, :3], axis=-1).max()),
                    "lowest_z_m": float(q[:, :, 2].min()), "max_xy_m": float(np.abs(q[:, :, :2]).max())})
        for wd in meta["worlds"]:
            n_ep += 1
            for f in {x["family"] for x in wd["families"]}:
                fam[f] += 1
        idx = d["contact_idx"].astype(int)
        for w, wd in enumerate(meta["worlds"]):
            n = len(wd["objects"])
            g = idx[(idx[:, 1] == w) & (idx[:, 3] == 1) & (idx[:, 2] >= 2) & (idx[:, 2] < 2 + n)]
            touched = set(g[:, 2].tolist())
            for k, o in enumerate(wd["objects"]):
                inter["objects"] += 1
                inter["grip"] += (2 + k) in touched
                inter["drop"] += o["drop_height"] > 0
    ok = all(r["finite"] and r["max_speed_m_s"] < 20 and r["lowest_z_m"] > -0.80 and r["max_xy_m"] < 10 for r in inv)
    print(f"invariants on {len(inv)} shards: {'OK' if ok else 'FAILED'}  " +
          ", ".join(f"{r['shard']}: speed {r['max_speed_m_s']:.1f}, z {r['lowest_z_m']:.2f}, xy {r['max_xy_m']:.2f}" for r in inv), flush=True)
    fractions = {f: round(c / n_ep, 3) for f, c in sorted(fam.items())}
    objects = {k: round(inter[k] / inter["objects"], 3) for k in ("grip", "drop")}
    print(f"family fractions over {n_ep} episodes: {fractions}; per object: {objects}", flush=True)
    repro = None
    if a.repro >= 0:
        i = a.repro
        meta = json.loads((ds / f"shard_{i:05d}.json").read_text())
        n_worlds = len(meta["worlds"])
        seed0 = (meta["seed"] - i) // 1_000_003
        with tempfile.TemporaryDirectory(dir=REPO / "outputs") as tmp:
            # regenerate exactly shard i: same seed, same world count, only that shard index
            cmd = [sys.executable, str(REPO / "scripts/l1_generate.py"), "--out", tmp, "--episodes",
                   str(n_worlds * (i + 1)), "--worlds", str(n_worlds), "--seed", str(seed0), "--part", f"{i}/{i + 1}"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            a_q = np.load(ds / f"shard_{i:05d}.npz")["body_q"]
            b_q = np.load(Path(tmp) / f"shard_{i:05d}.npz")["body_q"]
            diff = np.linalg.norm(a_q[..., :3] - b_q[..., :3], axis=-1)
            repro = {"shard": i, "median_mm": 1e3 * float(np.median(diff)), "p99_mm": 1e3 * float(np.percentile(diff, 99)),
                     "max_mm": 1e3 * float(diff.max())}
        print(f"seed reproducibility, shard {i}: position difference median {repro['median_mm']:.4f} mm, "
              f"p99 {repro['p99_mm']:.3f} mm, max {repro['max_mm']:.1f} mm", flush=True)
    OUT.write_text(json.dumps({"data": a.data, "shards": len(shards), "invariants": inv, "invariants_ok": ok,
                               "family_fractions": fractions, "per_object": objects, "reproducibility": repro}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
