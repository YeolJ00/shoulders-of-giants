"""The coverage study's score (claim 1, 2026-11-22): one network on each gripper family of held-out episodes.

c1_test's single-family episodes, one family at a time (fold and shake are held out of every coverage arm but the
last; scripts/l1_coverage_subsets.py):
  one frame   --n windows per family (seeded: the same windows for every checkpoint), clean history, every slot
              given: error / constant velocity's on moving (> 2 cm/s) cloth vertices that are not held, and on those
              in the air (l2_error_where's "free" class); per-window sums for paired comparisons
  rollouts    --n_roll episodes per family (seeded), from frame 7 alone to the end (up to 113 frames, 3.8 s), the
              material hidden, the gripper on its recorded path (l2_cloth_long's protocol): median distance of the
              free vertices from the teacher at 1 s, 2 s and the end, mean |edge strain| and mesh area over the
              teacher's at the end

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_drag/step11999.pt --tag cov_drag

Writes docs/results/l2_coverage_<tag>.json.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
FAMILIES = ("fold", "shake", "drag", "lift_place", "lift_drop", "push", "slide")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="c1_test")
    ap.add_argument("--n", type=int, default=200, help="one-frame windows per family")
    ap.add_argument("--n_roll", type=int, default=48, help="rollout episodes per family (0: one frame only)")
    ap.add_argument("--roll_given", action="store_true", help="rollouts with the material GIVEN (for networks trained with every slot given, cloth_train --given: hidden is an input they never saw; 2026-11-24)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--subset", default="", help="score only this subset's episodes of --data (a coverage_subsets json: "
                    "an arm's own training episodes, with --data c1, for its train-vs-held-out gap)")
    a = ap.parse_args()
    from l1_coverage_subsets import families
    from l2_cloth_long import shape_stats
    from l2_error_where import classify
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import _windows, rollout, step_loss
    from lsim.train.cloth_windows import collate_cloth, load_cloth_shard
    dev = "cuda"
    by = {f: [] for f in a.families.split(",")}
    keep = None
    if a.subset:
        import json as _json
        keep = {(str(sh), int(w)) for sh, w in _json.loads((REPO / a.subset).read_text())["episodes"]}
    for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz"))):
        if keep is not None and Path(p).stem not in {sh for sh, _ in keep}:
            continue
        for e in load_cloth_shard(p[:-4]):
            e.shard = Path(p).stem
            if keep is not None and (e.shard, int(e.w)) not in keep:
                continue
            fs = families(e.wd)
            if len(fs) == 1 and next(iter(fs)) in by:
                by[next(iter(fs))].append(e)
    m = load_model(REPO / a.ckpt, dev)
    P = 8
    out = {"ckpt": a.ckpt, "data": a.data, "families": {}}
    for fi, (fam, eps) in enumerate(by.items()):
        wins = _windows(eps, a.n, P, 8, np.random.default_rng(100 + fi))
        acc = {"moving": [0.0, 0.0, 0], "free": [0.0, 0.0, 0]}
        per_window = []
        with torch.no_grad():
            for i, w in enumerate(wins):
                b = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
                parts = {}
                step_loss(m, b, np.random.default_rng(i), 8, dev, sig=0.0, parts=parts)
                x, fr = parts["x"][:, 0], parts["free"][:, 0]
                pos = b["pos"]
                tgt, cv = pos[:, P], 2 * pos[:, P - 1] - pos[:, P - 2]
                moving = (pos[:, P - 1] - pos[:, P - 2]).norm(dim=-1) * 30 > 0.02
                pw = []
                for k, sel in (("moving", fr & moving), ("free", fr & moving & classify(b, P)["free"])):
                    em, ec, n = float((x - tgt)[sel].norm(dim=-1).sum()), float((cv - tgt)[sel].norm(dim=-1).sum()), int(sel.sum())
                    acc[k][0] += em; acc[k][1] += ec; acc[k][2] += n
                    pw.append([em, ec, n])
                per_window.append(pw)
        one = {k: {"ratio": v[0] / max(v[1], 1e-12), "model_mm": 1e3 * v[0] / max(v[2], 1), "cv_mm": 1e3 * v[1] / max(v[2], 1),
                   "n": v[2]} for k, v in acc.items()}
        rows = []
        pick = np.random.default_rng(200 + fi).choice(len(eps), min(a.n_roll, len(eps)), replace=False)
        for j in pick:
            e = eps[j]
            T0, n = 7, min(113, e.n_frames - 8)
            free = ~e.held[T0 + 1:T0 + 1 + n]
            with torch.no_grad():
                Pr, Tr, _ = rollout(m, e, T0, n, dev="cuda", hidden=() if a.roll_given else ("density", "stretch", "bend", "mu"))
            err = np.array([np.median(np.linalg.norm(Pr[k] - Tr[k], axis=-1)[free[k]]) for k in range(n)])
            am, _, sm = shape_stats(Pr[-1], e)
            at, _, st = shape_stats(Tr[-1], e)
            rows.append({"shard": e.shard, "w": int(e.w), "seed": int(e.wd["seed"]),
                         "e30": 1e3 * float(err[min(29, n - 1)]), "e60": 1e3 * float(err[min(59, n - 1)]),
                         "end": 1e3 * float(err[-1]), "strain_pct": 100 * sm, "teacher_strain_pct": 100 * st,
                         "area_ratio": am / at})
        roll = {k: float(np.mean([r[k] for r in rows])) if rows else float("nan") for k in ("e30", "e60", "end", "strain_pct")}
        roll["area_ratio_median"] = float(np.median([r["area_ratio"] for r in rows])) if rows else float("nan")
        out["families"][fam] = {"episodes": len(eps), "one_frame": one, "per_window": per_window, "rollout": roll,
                                "rollout_rows": rows}
        print(f"{fam:10s} {len(eps):3d} episodes  one frame {one['moving']['ratio']:.3f} (in the air {one['free']['ratio']:.3f})"
              f"  rollout 1 s {roll['e30']:5.1f}  2 s {roll['e60']:5.1f}  end {roll['end']:6.1f} mm  strain {roll['strain_pct']:.1f}%"
              f"  area {roll['area_ratio_median']:.2f}", flush=True)
    (REPO / f"docs/results/l2_coverage_{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
