"""L1 -- real cloth recordings -> cloth episodes the teacher and the model can be measured on (lsim.real.cloth).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l1_real_cloth.py                     # all 8 PhysTwin cloth recordings
    $L scripts/l1_real_cloth.py single_lift_cloth_1

Per recording: the rest frame (the flattest of the first 10), a grid of ~1 cm cells over the cloth there, every
tracked point embedded in it, every frame's vertices tracked frame by frame (lsim.real.cloth.track_mesh), the grasp
(held vertices) from the recording's grasp type, and quality numbers:

  residual    distance of each visible tracked point to its embedded position on the fitted mesh
  annotated   the same for the 9 hand-annotated tracks (embedded at the rest frame), an independent check
  drift       how far the tracked points around each annotated point wander from it (the tracker leaving the
              material: what no fit can undo)
  strain      the fitted mesh's edge strain (the fit holds edge lengths at 0.5% tolerance)

Writes data/real/cloth/phystwin/<case>.npz (+ .json) and docs/results/l1_real_cloth_phystwin.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lsim.real import cloth as rc

REPO = Path(__file__).resolve().parents[1]
# what each video shows (cam 0 frames, read by eye): the fabric, and what the hands do
PHYSTWIN_NOTES = {
    "single_lift_cloth": "red brocade, ~45 x 50 cm; one hand pinches near the centre and lifts it into a hanging bundle",
    "single_lift_cloth_1": "green cloth, 33 cm square; one hand pinches near the centre and lifts it clear of the table",
    "single_lift_cloth_3": "grey woven cloth; one hand pinches near the centre and lifts",
    "single_lift_cloth_4": "light blue cloth; one hand pinches near the centre and lifts",
    "single_clift_cloth_1": "green cloth; one hand lifts a corner diagonally",
    "single_clift_cloth_3": "grey woven cloth; one hand lifts a corner diagonally",
    "double_lift_cloth_1": "green cloth; two hands take the two far corners and raise that edge, the cloth hangs like a curtain",
    "double_lift_cloth_3": "grey woven cloth; two hands take the two far corners and raise that edge, the cloth folds over",
}


def drift(P, vis, gt, ref):
    """Median distance (m) of the tracked points within 1 cm of each annotated point at rest to it, per frame."""
    tree = cKDTree(np.where(vis[ref][:, None], P[ref], 1e3))
    D = []
    for k in range(gt.shape[1]):
        if np.isnan(gt[ref, k]).any():
            continue
        nb = tree.query_ball_point(gt[ref, k], 0.01)
        if not nb:
            continue
        row = np.full(len(P), np.nan)
        for t in range(len(P)):
            m = [i for i in nb if vis[t, i]]
            if m and not np.isnan(gt[t, k]).any():
                row[t] = np.median(np.linalg.norm(P[t, m] - gt[t, k], axis=-1))
        D.append(row)
    return np.array(D)


def convert(case, out_dir, iters=60):
    t_start = time.time()
    d = rc.phystwin_case(case)
    P, vis, C, gt = d["P"], d["vis"], d["C"], d["gt"]
    T = len(P)
    ref, spread = rc.rest_frame(P, vis)
    g, _ = rc.rest_grid(P[ref, vis[ref]])
    q = P[ref] - g["origin"]
    uv = np.stack([q @ g["u"], q @ g["v"]], 1)
    span = np.array([g["nx"], g["ny"]]) * g["h"]
    inside = ((uv > -0.01) & (uv < span + 0.01)).all(1)
    tri, bary = rc.embed(uv, g)
    rest_xyz, tris = rc.grid_world(g)
    use = vis & vis[ref][None] & inside[None]                          # points seen at rest, on the cloth
    X = rc.track_mesh(P, use, tri, bary, rest_xyz, tris, ref, iters=iters)
    r = rc.residuals(X, P, use, tri, bary)
    s = np.abs(rc.strains(X, tris, rest_xyz))
    grasp = rc.PHYSTWIN_GRASP[case]
    held, ginfo = rc.held_from_hands(X, C, rest_xyz, grasp, g)
    seen = ~np.isnan(gt[ref]).any(1)
    qg = np.where(seen[:, None], gt[ref], 0.0) - g["origin"]
    gtri, gbary = rc.embed(np.stack([qg @ g["u"], qg @ g["v"]], 1), g)
    gtri[~seen] = -1
    pg = (X[:, gtri[seen]] * gbary[seen][None, :, :, None]).sum(2)
    eg = np.linalg.norm(pg - gt[:, seen], axis=-1)
    dr = drift(P, vis, gt, ref)
    tr_end = int(d["split"]["train"][1])

    def stats(a, frames=slice(None)):
        a = a[frames]
        return {"median_mm": float(1e3 * np.nanmedian(a)), "p90_mm": float(1e3 * np.nanpercentile(a, 90))}

    qc = {"residual": stats(r), "residual_train": stats(r, slice(0, tr_end)), "residual_test": stats(r, slice(tr_end, T)),
          "residual_median_mm_every_10th": [float(1e3 * np.nanmedian(r[t])) for t in range(0, T, 10)],
          "annotated": stats(eg.T), "annotated_test": stats(eg.T, slice(tr_end, T)),
          "drift_train_mm": float(1e3 * np.nanmedian(dr[:, :tr_end])), "drift_test_mm": float(1e3 * np.nanmedian(dr[:, tr_end:])),
          "strain_median": float(np.median(s)), "strain_p90": float(np.percentile(s, 90)), "strain_p99": float(np.percentile(s, 99)),
          "rest_height_spread_mm": 1e3 * spread, "tracked_points": int(inside.sum()), "visible_share": float(use.mean() / max(use[ref].mean(), 1e-9))}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{case}.npz", x=X.astype(np.float32), held=held, hand=C.astype(np.float32),
                        obs=P.astype(np.float32), obs_vis=use, obs_tri=tri.astype(np.int32), obs_bary=bary.astype(np.float32),
                        gt=gt.astype(np.float32), gt_tri=gtri.astype(np.int32), gt_bary=gbary.astype(np.float32))
    rec = {"source": "phystwin", "case": case, "notes": PHYSTWIN_NOTES[case], "fps": d["fps"], "frames": T,
           "split": {"train_end": tr_end, "test_end": int(d["split"]["test"][1])}, "rest_frame": ref,
           "grid": {"nx": g["nx"], "ny": g["ny"], "h": g["h"], "origin": g["origin"].tolist(), "u": g["u"].tolist(),
                    "v": g["v"].tolist()},
           "grasp": ginfo, "qc": qc, "frame": "z up, table top at z = 0 (PhysTwin's frame turned half about x)"}
    (out_dir / f"{case}.json").write_text(json.dumps(rec, indent=1))
    print(f"{case}: {T} frames, grid {g['nx']} x {g['ny']} ({1e3 * g['h']:.1f} mm), rest frame {ref}; residual "
          f"{qc['residual']['median_mm']:.1f} (p90 {qc['residual']['p90_mm']:.1f}) mm, test {qc['residual_test']['median_mm']:.1f}; "
          f"annotated {qc['annotated']['median_mm']:.1f} (p90 {qc['annotated']['p90_mm']:.1f}); drift test {qc['drift_test_mm']:.1f}; "
          f"strain {qc['strain_median']:.1e}; grasp {grasp} {[(x['vertex'], x['from_frame']) for x in ginfo]} "
          f"({time.time() - t_start:.0f} s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=list(rc.PHYSTWIN_CLOTH))
    ap.add_argument("--iters", type=int, default=60)
    a = ap.parse_args()
    out = rc.OUT / "phystwin"
    recs = {c: convert(c, out, a.iters) for c in a.cases}
    res = REPO / "docs/results/l1_real_cloth_phystwin.json"
    old = json.loads(res.read_text()) if res.exists() else {}
    old.update({c: {"notes": r["notes"], "frames": r["frames"], "grid": [r["grid"]["nx"], r["grid"]["ny"], r["grid"]["h"]],
                    "grasp": r["grasp"], "qc": r["qc"]} for c, r in recs.items()})
    res.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
