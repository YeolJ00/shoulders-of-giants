"""WorldParticle as a baseline on our cloth data (2026-11-24; the user: "run WorldParticle as a baseline on our cloth
data"; arXiv 2605.15305, code at data/external/WorldParticle, run in its own env by scripts/l2_wp_train.py and
scripts/l2_wp_eval.py). This script is the lsim side: it writes c1 episodes in the form that model takes, and scores
its predictions exactly as scripts/l2_coverage_score.py scores ours (same windows, same masks, same rollouts, same
JSON), so scratch paired_cov.py / paired_roll.py compare the two directly.

Per episode, one npz (particles = the cloth's vertices, then the gripper's 110 surface points):
  x          (T, Vc + Vg, 3) float32   world positions every recorded frame (30 fps)
  n_cloth    Vc
  held       (T, Vc) bool              pinned during the step from frame j to j + 1 (ClothEpisode.held)
  slots      (4,) float32              the cloth's material slots (cloth_slots: density, stretch, bend, friction)
  box, box_feats  (M, 3), (M, 4)       static points within R_STATIC of anything that moves in the episode (table top
                                       and floor lattices at 2 cm, the plank and the objects' points), with their
                                       outward normals and log friction
  nbr_index, nbr_splits                CSR mesh neighbours (self + the mesh's edges) of the cloth's vertices; none for
                                       the gripper's points

    $L scripts/l2_wp_data.py export --data c1 --subset outputs/lsim/coverage_subsets/seven.json --out outputs/lsim/wp/data/train
    $L scripts/l2_wp_data.py export --data c1_test --out outputs/lsim/wp/data/test --plan outputs/lsim/wp/data/plan.json
    $L scripts/l2_wp_data.py score --plan outputs/lsim/wp/data/plan.json --pred outputs/lsim/wp/<run>/pred_<step>.npz --tag cov_wp_<run>

The plan holds what l2_coverage_score.py draws (its seeds): per family, 200 one-frame windows (episode, frame) with
their masks (free and moving cloth vertices; those also in the air), and 48 rollout episodes from frame 7.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
FAMILIES = ("fold", "shake", "drag", "lift_place", "lift_drop", "push", "slide")
P = 8


def episode_arrays(e):
    from scipy.spatial import cKDTree
    from lsim.data.scene import FLOOR_MAT, TABLE_HALF
    from lsim.train.cloth_windows import R_STATIC, _plane_patch, cloth_slots, surface_feature
    T = e.n_frames
    xc = e.x(np.arange(T))
    xg = e.gripper_points(np.arange(T))
    x = np.concatenate([xc, xg], 1).astype(np.float32)
    moving = x[::2].reshape(-1, 3)
    pts, nrm, mat = [], [], []
    for z, half, m in ((0.0, TABLE_HALF, (e.wd.get("table_mu", 0.5), 0.01)), (-0.75, (10.0, 10.0), FLOOR_MAT)):
        p = _plane_patch(moving, z, half, 0.02, R_STATIC)
        pts.append(p); nrm.append(np.broadcast_to([0.0, 0.0, 1.0], p.shape)); mat.append(np.broadcast_to(surface_feature(*m), (len(p), 2)))
    sp, sn, sm, _, _ = e.statics()
    if len(sp):
        d, _ = cKDTree(moving).query(sp, k=1, distance_upper_bound=R_STATIC)
        keep = np.isfinite(d)
        pts.append(sp[keep]); nrm.append(sn[keep]); mat.append(sm[keep])
    box = np.concatenate(pts).astype(np.float32)
    box_feats = np.concatenate([np.concatenate(nrm), np.concatenate(mat)[:, :1]], 1).astype(np.float32)
    Vc = xc.shape[1]
    nb = [[i] for i in range(Vc)] + [[] for _ in range(xg.shape[1])]
    for a, b in np.asarray(e.edges):
        nb[int(a)].append(int(b)); nb[int(b)].append(int(a))
    splits = np.cumsum([0] + [len(n) for n in nb]).astype(np.int64)
    index = np.array([j for n in nb for j in n], np.int64)
    return dict(x=x, n_cloth=np.int64(Vc), held=e.held.astype(bool), slots=np.asarray(cloth_slots(e.c), np.float32),
                box=box, box_feats=box_feats, nbr_index=index, nbr_splits=splits)


def load_eps(data, subset=""):
    from l1_coverage_subsets import families
    from lsim.train.cloth_windows import load_cloth_shard
    keep = None
    if subset:
        keep = {(str(sh), int(w)) for sh, w in json.loads((REPO / subset).read_text())["episodes"]}
    by = {f: [] for f in FAMILIES}
    alle = []
    for p in sorted(glob.glob(str(REPO / f"data/datasets/{data}/shard_*.npz"))):
        if keep is not None and Path(p).stem not in {sh for sh, _ in keep}:
            continue
        for e in load_cloth_shard(p[:-4]):
            e.shard = Path(p).stem
            if keep is not None and (e.shard, int(e.w)) not in keep:
                continue
            fs = families(e.wd)
            if len(fs) == 1 and next(iter(fs)) in by:
                by[next(iter(fs))].append(e)
            if keep is not None or (len(fs) == 1 and next(iter(fs)) in by):
                alle.append(e)
    return by, alle


def export(a):
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    by, alle = load_eps(a.data, a.subset)
    for k, e in enumerate(alle):
        f = out / f"{e.shard}_w{int(e.w):04d}.npz"
        if not f.exists():
            np.savez(f, **episode_arrays(e))
        if k % 200 == 0:
            print(f"[export] {k} / {len(alle)}", flush=True)
    print(f"[export] {len(alle)} episodes in {out}")
    if not a.plan:
        return
    import torch
    from l2_error_where import classify
    from lsim.train.cloth_train import _windows
    from lsim.train.cloth_windows import collate_cloth
    plan = {}
    for fi, fam in enumerate(FAMILIES):
        eps = by[fam]
        wins = _windows(eps, a.n, P, 8, np.random.default_rng(100 + fi))
        # _windows draws an episode and a frame per window; recover which episode by matching the window's frames
        ids, ts, m_mov, m_free = [], [], [], []
        for i, w in enumerate(wins):
            b = collate_cloth([w], "cpu", np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
            pos = b["pos"]
            Vc = int(w["n_cloth"])
            fr = (b["is_cloth"] & ~b["held"][:, P - 1])
            moving = (pos[:, P - 1] - pos[:, P - 2]).norm(dim=-1) * 30 > 0.02
            free_cls = classify(b, P)["free"]
            t = int(w["frames"][P - 1])
            xc = w["pos"][:, :Vc]
            match = [j for j, e in enumerate(eps) if len(e.rest) == Vc and np.allclose(e.x([t])[0], xc[P - 1], atol=1e-6)]
            assert len(match) == 1, (fam, i, len(match))
            e = eps[match[0]]
            ids.append(f"{e.shard}_w{int(e.w):04d}"); ts.append(t)
            m_mov.append((fr & moving)[:Vc].numpy()); m_free.append((fr & moving & free_cls)[:Vc].numpy())
        pick = np.random.default_rng(200 + fi).choice(len(eps), min(a.n_roll, len(eps)), replace=False)
        plan[fam] = {"win_ep": ids, "win_t": ts, "win_moving": m_mov, "win_free": m_free,
                     "roll_ep": [f"{eps[j].shard}_w{int(eps[j].w):04d}" for j in pick]}
        print(f"[plan] {fam}: {len(ids)} windows, {len(pick)} rollouts", flush=True)
    masks = {}
    for fam, pl in plan.items():
        for i, (mm, mf) in enumerate(zip(pl.pop("win_moving"), pl.pop("win_free"))):
            masks[f"{fam}_mov{i:03d}"] = mm; masks[f"{fam}_free{i:03d}"] = mf
    (REPO / a.plan).write_text(json.dumps(plan))
    np.savez(str(REPO / a.plan).replace(".json", "_masks.npz"), **masks)
    print(f"[plan] {REPO / a.plan}")


def score(a):
    from l2_cloth_long import shape_stats
    by, _ = load_eps("c1_test")
    plan = json.loads((REPO / a.plan).read_text())
    masks = np.load(str(REPO / a.plan).replace(".json", "_masks.npz"))
    pred = np.load(REPO / a.pred, allow_pickle=False)
    data = REPO / a.data_dir
    out = {"ckpt": a.pred, "data": "c1_test", "families": {}, "baseline": "WorldParticle (arXiv 2605.15305)"}
    for fam in FAMILIES:
        pl = plan[fam]
        eps = {f"{e.shard}_w{int(e.w):04d}": e for e in by[fam]}
        acc = {"moving": [0.0, 0.0, 0], "free": [0.0, 0.0, 0]}
        per_window = []
        for i, (eid, t) in enumerate(zip(pl["win_ep"], pl["win_t"])):
            if f"{fam}_win{i:03d}" not in pred.files:                      # smoke tests predict a few windows only
                continue
            x = np.load(data / f"{eid}.npz")["x"]
            m_mov, m_free = masks[f"{fam}_mov{i:03d}"], masks[f"{fam}_free{i:03d}"]
            Vc = len(m_mov)
            tgt, cv = x[t + 1, :Vc], 2 * x[t, :Vc] - x[t - 1, :Vc]
            xp = pred[f"{fam}_win{i:03d}"]
            pw = []
            for k, sel in (("moving", m_mov), ("free", m_free)):
                em = float(np.linalg.norm(xp - tgt, axis=-1)[sel].sum()); ec = float(np.linalg.norm(cv - tgt, axis=-1)[sel].sum())
                acc[k][0] += em; acc[k][1] += ec; acc[k][2] += int(sel.sum())
                pw.append([em, ec, int(sel.sum())])
            per_window.append(pw)
        one = {k: {"ratio": v[0] / max(v[1], 1e-12), "model_mm": 1e3 * v[0] / max(v[2], 1), "cv_mm": 1e3 * v[1] / max(v[2], 1),
                   "n": v[2]} for k, v in acc.items()}
        rows = []
        for eid in pl["roll_ep"]:
            if f"{fam}_roll_{eid}" not in pred.files:
                continue
            e = eps[eid]
            T0, n = 7, min(113, e.n_frames - 8)
            free = ~e.held[T0 + 1:T0 + 1 + n]
            Pr = pred[f"{fam}_roll_{eid}"][:n]
            Tr = e.x(np.arange(T0 + 1, T0 + 1 + n))
            err = np.array([np.median(np.linalg.norm(Pr[k] - Tr[k], axis=-1)[free[k]]) for k in range(n)])
            am, _, sm = shape_stats(Pr[-1], e)
            at, _, st = shape_stats(Tr[-1], e)
            rows.append({"shard": e.shard, "w": int(e.w), "seed": int(e.wd["seed"]),
                         "e30": 1e3 * float(err[min(29, n - 1)]), "e60": 1e3 * float(err[min(59, n - 1)]),
                         "end": 1e3 * float(err[-1]), "strain_pct": 100 * sm, "teacher_strain_pct": 100 * st,
                         "area_ratio": am / at})
        roll = {k: float(np.mean([r[k] for r in rows])) if rows else float("nan") for k in ("e30", "e60", "end", "strain_pct")}
        roll["area_ratio_median"] = float(np.median([r["area_ratio"] for r in rows])) if rows else float("nan")
        out["families"][fam] = {"episodes": len(by[fam]), "one_frame": one, "per_window": per_window, "rollout": roll,
                                "rollout_rows": rows}
        print(f"{fam:10s} {len(per_window)} windows, {len(rows)} rollouts; one frame {one['moving']['ratio']:.3f} (in the air {one['free']['ratio']:.3f})  rollout 1 s "
              f"{roll['e30']:5.1f}  2 s {roll['e60']:5.1f}  end {roll['end']:6.1f} mm  strain {roll['strain_pct']:.1f}%  "
              f"area {roll['area_ratio_median']:.2f}", flush=True)
    (REPO / f"docs/results/l2_coverage_{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("export")
    x.add_argument("--data", default="c1")
    x.add_argument("--subset", default="")
    x.add_argument("--out", required=True)
    x.add_argument("--plan", default="", help="also write the scorer's windows and rollouts (c1_test)")
    x.add_argument("--n", type=int, default=200)
    x.add_argument("--n_roll", type=int, default=48)
    s = sub.add_parser("score")
    s.add_argument("--plan", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--data_dir", default="outputs/lsim/wp/data/test")
    s.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"export": export, "score": score}[a.cmd](a)
