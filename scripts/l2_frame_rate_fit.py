"""Does the cloth network fit the teacher better at a higher frame rate (2026-11-18; the frame-rate study)?

The same episodes recorded at 30, 60 and 120 fps (scripts/l1_generate_cloth.py --sub 4 -> data/datasets/c1_fps<rate>),
one matched run per rate (cloth_train --fps, LSIM_FPS). This scores one run at its rate on held-out windows that start
at the SAME instants at every rate (episode e, 30 fps frame t30 -> frame t30 * rate / 30), clean history:
  one step      the model's mean free-vertex error against the teacher's next frame, constant velocity's on the same
                vertices, their ratio (the share of a frame's acceleration left unexplained), the shape error (mean
                |edge length - teacher's| / rest length over the 8 teacher-forced steps) and the teacher's own
                per-frame change of edge length
  1/30, 4/30 s  the model rolled out on its own predictions for rate / 30 (and 4 x) steps: the error at the same
                physical time for every rate, against constant velocity at 30 fps from the same instant
One process per rate (the frame time is read at import):

    LSIM_FPS=120 CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_frame_rate_fit.py --run cloth_fps120
    $L scripts/l2_frame_rate_fit.py --merge          # docs/results/l2_frame_rate_fit.json, a table
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
RATES = (30, 60, 120)


def merge():
    out = {}
    for r in RATES:
        p = REPO / f"docs/results/l2_frame_rate_fit_{r}.json"
        if p.exists():
            out[str(r)] = json.load(open(p))
    json.dump(out, open(REPO / "docs/results/l2_frame_rate_fit.json", "w"))
    for r, d in out.items():
        for c in d["checkpoints"]:
            s = c["summary"]
            line = (f"{r:>4s} fps step {c['step']:5d}: one step model {s['pos_mm']:.4f} mm, constant velocity {s['cv_mm']:.4f} mm, "
                    f"ratio {s['ratio']:.3f} (moving windows, median {s['ratio_moving_median']:.3f}); shape {s['shape_pct']:.4f} % "
                    f"(teacher's change {s['teacher_edge_pct']:.4f} %)")
            if "h1_mm" in s:
                line += (f"; after 1/30 s {s['h1_mm']:.3f} mm (30 fps constant velocity {s['h1_cv_mm']:.3f}; shape {s['h1_shape_pct']:.3f} %), "
                         f"after 4/30 s {s['h4_mm']:.2f} mm ({s['h4_cv_mm']:.2f}; shape {s['h4_shape_pct']:.3f} %)")
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="outputs/lsim/<run>/step*.pt, trained at LSIM_FPS")
    ap.add_argument("--test", default="", help="held-out dataset (default data/datasets/c1_fps<rate>_test)")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tag", default="", help="results file suffix (l2_frame_rate_fit_<rate><tag>.json)")
    a = ap.parse_args()
    if a.merge:
        return merge()
    import torch
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import rollout, step_loss
    from lsim.train.cloth_windows import FPS, collate_cloth, load_shard, make_cloth_window
    rate = int(round(FPS))
    k = rate // 30
    assert rate in RATES and float(os.environ.get("LSIM_FPS", 30)) == rate
    dev = "cuda"
    test = a.test or f"data/datasets/c1_fps{rate}_test"
    eps = [e for p in sorted(glob.glob(str(REPO / test / "shard_*.npz"))) for e in load_shard(p[:-4])]
    rng = np.random.default_rng(4)
    F30 = (eps[0].n_frames - 1) // k
    picks = [(int(rng.integers(len(eps))), int(rng.integers(8, F30 - 8))) for _ in range(a.n)]   # the same at every rate
    ck = sorted(glob.glob(str(REPO / f"outputs/lsim/{a.run}/step*.pt")), key=lambda p: int(re.findall(r"step(\d+)", p)[-1]))
    out = {"rate": rate, "run": a.run, "test": test, "n": a.n, "checkpoints": []}
    for ci, p in enumerate(ck):
        step = int(re.findall(r"step(\d+)", p)[-1])
        m = load_model(p, dev)
        last = ci == len(ck) - 1
        rows = []
        with torch.no_grad():
            for i, (ei, t30) in enumerate(picks):
                ep, t = eps[ei], t30 * k
                b = collate_cloth([make_cloth_window(ep, t, 8, 8, np.random.default_rng(i))], dev, np.random.default_rng(i),
                                  p_hidden=0.0, augment=False, all_past=True)
                torch.manual_seed(i)
                _, info = step_loss(m, b, np.random.default_rng(i), 8, dev, sig=0.0)
                x, P = b["pos"], 8
                fr = (b["is_cloth"][:, None] & ~b["held"])[:, P - 1]
                cv = (x[fr, P] - 2 * x[fr, P - 1] + x[fr, P - 2]).norm(dim=-1).mean().item() * 1e3
                E = b["edges"]
                L0 = (b["rest"][E[:, 0]] - b["rest"][E[:, 1]]).norm(dim=-1) * b["size"][b["sample_of"][E[:, 0]]]
                ln = (x[E[:, 0], P - 1:P + 8] - x[E[:, 1], P - 1:P + 8]).norm(dim=-1)
                row = dict(ep=ei, t30=t30, pos_mm=info["clean_mean_mm"], cv_mm=cv, shape_pct=info["strain_err_pct"],
                           teacher_edge_pct=100 * ((ln[:, 1:] - ln[:, :-1]).abs() / L0[:, None]).mean().item())
                if last:                                              # the same physical horizon at every rate
                    pred, true, _ = rollout(m, ep, t, 4 * k, dev=dev)
                    free = ~ep.held[t:t + 4 * k + 1].any(0)
                    x0, xm = ep.x([t])[0], ep.x([t - k])[0]            # 30 fps constant velocity from the same instant
                    e, L = ep.edges, L0.cpu().numpy()
                    for h, name in ((1, "h1"), (4, "h4")):
                        pr, tr = pred[h * k - 1], true[h * k - 1]
                        row[f"{name}_mm"] = 1e3 * float(np.linalg.norm(pr - tr, axis=-1)[free].mean())
                        row[f"{name}_cv_mm"] = 1e3 * float(np.linalg.norm(x0 + h * (x0 - xm) - tr, axis=-1)[free].mean())
                        dl = np.abs(np.linalg.norm(pr[e[:, 0]] - pr[e[:, 1]], axis=-1) - np.linalg.norm(tr[e[:, 0]] - tr[e[:, 1]], axis=-1))
                        row[f"{name}_shape_pct"] = 100 * float((dl / L).mean())
                rows.append(row)
        R = {q: np.array([r[q] for r in rows], float) for q in rows[0]}
        ok = np.isfinite(R["pos_mm"])
        mv = ok & (R["cv_mm"] > np.median(R["cv_mm"][ok]))            # the windows that move more than the median one
        s = {q: float(np.nanmean(R[q][ok])) for q in R if q not in ("ep", "t30")}
        s["ratio"] = s["pos_mm"] / s["cv_mm"]
        s["ratio_moving_median"] = float(np.median(R["pos_mm"][mv] / R["cv_mm"][mv]))
        out["checkpoints"].append({"step": step, "summary": s, "rows": rows})
        print(f"{rate} fps step {step}: model {s['pos_mm']:.4f} mm  constant velocity {s['cv_mm']:.4f} mm  ratio {s['ratio']:.3f}  "
              f"shape {s['shape_pct']:.4f} %  teacher {s['teacher_edge_pct']:.4f} %"
              + (f"  1/30 s {s['h1_mm']:.3f} (cv30 {s['h1_cv_mm']:.3f})  4/30 s {s['h4_mm']:.2f} (cv30 {s['h4_cv_mm']:.2f})" if last else ""),
              flush=True)
        del m
        torch.cuda.empty_cache()
        json.dump(out, open(REPO / f"docs/results/l2_frame_rate_fit_{rate}{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
