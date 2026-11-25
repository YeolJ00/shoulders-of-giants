"""L2 -- the cloth teacher against real cloth: where is the simulator wrong? (lsim.real.replay)

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_real_cloth_teacher.py [cases...] --tag <tag>

Per converted recording (scripts/l1_real_cloth.py), PhysTwin's protocol: from the first frame, the held patch
following the recording, the teacher's material is fitted on the training frames (the first 70%) by cross-entropy
search over its own material range (areal density, stretch modulus, bending length, friction:
lsim.data.cloth.sample_cloth's prior), then the best material is run on to the end and scored on the test frames it
never saw. The scores that count are against the 9 hand-annotated points (PhysTwin's track error; all of them, and
those that have moved over 1 cm): the reconstruction drifts from them on moving cloth (the tracker slides off folding
cloth: 10 mm by frames 29-73, up to 27 cm late), so it supplies the start and the held patch's path, and its own
annotated error is the reference beside each score. Error: the median distance of the visible tracked points that have moved
over 1 cm since the first frame (the lifted cloth; over every point the cloth lying still on the table dominated) to
their embedded positions (what the cameras saw), against the floor (the fitted mesh's own) and the prior's centre
material.

Writes docs/results/l2_real_cloth_teacher_<tag>.json and outputs/lsim/real_teacher/<case>.npz (the best run).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from lsim.real import replay
from lsim.real.cloth import PHYSTWIN_CLOTH, load_real_cloth

REPO = Path(__file__).resolve().parents[1]
CENTRE = {"w": 0.16, "Y": 3e3, "c": 0.027, "mu": 0.35}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=list(PHYSTWIN_CLOTH))
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out_dir = REPO / "outputs/lsim/real_teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for ep in load_real_cloth("phystwin", a.cases):
        t_start = time.time()
        case = ep.meta["case"]
        # PhysTwin's protocol: fit on its training frames, predict the rest. Every frame: the reconstruction itself
        # drifts from the annotations on the moving cloth (the tracker slides off folding cloth), so the score is
        # against the annotations, and the reconstruction's own annotated error is reported beside it
        last = ep.n_frames - 1
        train_end = int(ep.meta["split"]["train_end"])
        n_fit = train_end - 1
        n_all = last
        print(f"== {case}: fit on frames 0-{n_fit}, predict to {last}", flush=True)
        best, s_best, hist = replay.fit_material(ep, 0, n_fit, pop=a.pop, gens=a.gens)
        Xw = replay.simulate(ep, [best, CENTRE], 0, n_all)
        err = replay.track_errors(ep, Xw, 0)                                   # (2, n_all + 1)
        floor = ep.track_error(ep.x(np.arange(0, n_all + 1)), np.arange(0, n_all + 1), moved_from=0)
        tr = slice(1, min(train_end, n_all + 1))
        te = slice(train_end, n_all + 1)

        def mm(v):
            with np.errstate(all="ignore"):
                return float(1e3 * np.nanmean(v)) if np.size(v) and np.isfinite(v).any() else None
        fr = np.arange(0, n_all + 1)                                             # PhysTwin's metric: annotated points
        gt = {k: ep.gt_error(X, fr) for k, X in (("best", Xw[0]), ("centre", Xw[1]), ("floor", ep.x(fr)))}
        gtm = {k: ep.gt_error(X, fr, moved_from=0) for k, X in (("best", Xw[0]), ("centre", Xw[1]), ("floor", ep.x(fr)))}
        r = {"material": best, "fit_score_mm": 1e3 * s_best, "n_fit": n_fit, "trusted_to": last, "train_end": train_end,
             "train_mm": {"best": mm(err[0, tr]), "centre": mm(err[1, tr]), "floor": mm(floor[tr])},
             "test_mm": {"best": mm(err[0, te]), "centre": mm(err[1, te]), "floor": mm(floor[te])},
             "gt_train_mm": {k: mm(v[tr]) for k, v in gt.items()}, "gt_test_mm": {k: mm(v[te]) for k, v in gt.items()},
             "gt_moving_train_mm": {k: mm(v[tr]) for k, v in gtm.items()}, "gt_moving_test_mm": {k: mm(v[te]) for k, v in gtm.items()},
             "per_frame_mm": {"best": (1e3 * err[0]).round(2).tolist(), "centre": (1e3 * err[1]).round(2).tolist(),
                              "floor": (1e3 * floor).round(2).tolist()},
             "history": hist, "seconds": time.time() - t_start}
        res[case] = r
        np.savez_compressed(out_dir / f"{case}.npz", x_best=Xw[0], x_centre=Xw[1])
        print(f"   annotated (all / moved): train {r['gt_train_mm']['best']} / {r['gt_moving_train_mm']['best']} (floor "
              f"{r['gt_train_mm']['floor']} / {r['gt_moving_train_mm']['floor']}); test {r['gt_test_mm']['best']} / "
              f"{r['gt_moving_test_mm']['best']} (floor {r['gt_test_mm']['floor']} / {r['gt_moving_test_mm']['floor']})", flush=True)
        print(f"   best {best}: train {r['train_mm']['best']:.1f} mm (floor {r['train_mm']['floor']:.1f}, centre "
              f"{r['train_mm']['centre']:.1f}); test {r['test_mm']['best'] if r['test_mm']['best'] is not None else '-'} "
              f"(floor {r['test_mm']['floor']}) [{r['seconds']:.0f} s]", flush=True)
        path = REPO / f"docs/results/l2_real_cloth_teacher_{a.tag or 'default'}.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update({case: r})
        path.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
