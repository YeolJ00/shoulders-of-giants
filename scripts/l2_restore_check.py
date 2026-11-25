"""The exact layer's shape restoration (lsim.model.cloth_net.restore_shape, 2026-11-03) on the teachers' own data, per
material (c1_test cloth, r1_test rope, s1_test soft):

  exact   the force that makes the layer WITHOUT the restoration reproduce the teacher's next frame (fixed point:
          f += m (x_teacher - x_layer) / dt^2, 40 rounds), then the same force with the restoration on: what the
          restoration changes on frames the teacher made (the layer given the teacher's exact forces reproduced it to
          p99 0.05 mm, DIRECTION.md "Exact layer"); free vertices of 48 one-step windows per material
  rest    the network's outputs at zero on resting starts (nothing held, nothing moving), 30 frames, with and
          without the restoration: drift of every vertex from the start (CLAUDE.md: run every change to the layer
          with the network's outputs at zero on resting starts)

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_restore_check.py
Writes docs/results/l2_restore_check.json.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def q(a):
    a = np.asarray(a, np.float64)
    return {"p50": float(np.median(a)), "p99": float(np.quantile(a, 0.99)), "max": float(a.max()), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="c1_test,r1_test,s1_test")
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--rest", type=int, default=16)
    ap.add_argument("--restore", default="1")
    ap.add_argument("--out", default="docs/results/l2_restore_check.json")
    a = ap.parse_args()
    from lsim.model.cloth_net import PUSH_OUT, cloth_step, decode, restore_config
    from lsim.train.cloth_train import ZeroForces, resting_starts, rollout
    from lsim.train.cloth_windows import collate_cloth, load_shard, make_cloth_window
    dev = "cuda"
    cfg = restore_config(a.restore)
    res = {"restore": cfg}
    for d in a.data.split(","):
        eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{d}/shard_*.npz")))[:4] for e in load_shard(p[:-4])]
        kind = getattr(eps[0], "material", "cloth")
        rng = np.random.default_rng(0)
        base_err, rest_err, change, t_on, t_off = [], [], [], [], []
        for k in range(a.windows):
            e = eps[int(rng.integers(len(eps)))]
            t = int(rng.integers(8, e.n_frames - 2))
            b = collate_cloth([make_cloth_window(e, t, 8, 1, rng)], dev, rng, p_hidden=0.0, augment=False, all_past=True)
            pos = b["pos"]                                               # fp32, as the models run
            xp, xn, xt = pos[:, 6], pos[:, 7], pos[:, 8]
            free = b["is_cloth"] & ~b["held"][:, 7]
            w, mu = decode(b["params"], b)
            m = (w[b["body_of"]] * b["area"]).clamp_min(1e-9)
            f = torch.zeros_like(xn)
            kw = dict(push_out=PUSH_OUT)
            for _ in range(40):
                x0 = cloth_step(b, xp, xn, f, w, ~free, xt, mu, f_contact=f, **kw)
                f = f + torch.where(free[:, None], m[:, None] * (xt - x0) / (1.0 / 30.0) ** 2, torch.zeros_like(f))
            torch.cuda.synchronize(); t1 = time.time()
            x0 = cloth_step(b, xp, xn, f, w, ~free, xt, mu, f_contact=f, **kw)
            torch.cuda.synchronize(); t2 = time.time()
            x1 = cloth_step(b, xp, xn, f, w, ~free, xt, mu, f_contact=f, restore=cfg, **kw)
            torch.cuda.synchronize(); t3 = time.time()
            t_off.append(t2 - t1); t_on.append(t3 - t2)
            base_err.append(1e3 * (x0 - xt).norm(dim=-1)[free].cpu().numpy())
            rest_err.append(1e3 * (x1 - xt).norm(dim=-1)[free].cpu().numpy())
            change.append(1e3 * (x1 - x0).norm(dim=-1)[free].cpu().numpy())
        r = {"exact": {"layer_mm": q(np.concatenate(base_err)), "with_restore_mm": q(np.concatenate(rest_err)),
                       "restore_moved_mm": q(np.concatenate(change)),
                       "ms_per_step_without": 1e3 * float(np.median(t_off)), "ms_per_step_with": 1e3 * float(np.median(t_on))}}
        E = r["exact"]
        print(f"[{kind}] exact forces, free vertices: layer {E['layer_mm']['p50']:.4f} / p99 {E['layer_mm']['p99']:.4f} / max "
              f"{E['layer_mm']['max']:.3f} mm; with the restoration {E['with_restore_mm']['p50']:.4f} / p99 "
              f"{E['with_restore_mm']['p99']:.4f} / max {E['with_restore_mm']['max']:.3f} mm; moved p99 "
              f"{E['restore_moved_mm']['p99']:.4f} max {E['restore_moved_mm']['max']:.3f} mm; "
              f"{E['ms_per_step_without']:.1f} -> {E['ms_per_step_with']:.1f} ms a step", flush=True)
        starts = resting_starts(eps, n=a.rest)
        zr = {}
        for name, rc in (("without", None), ("with", cfg)):
            z = ZeroForces()
            z.restore = rc
            mean, mx = [], []
            for e, t0 in starts:
                with torch.no_grad():
                    pred, _, _ = rollout(z, e, t0, 30, dev=dev)
                dd = np.linalg.norm(pred[-1] - e.x([t0])[0], axis=-1)
                mean.append(1e3 * dd.mean()); mx.append(1e3 * dd.max())
            zr[name] = {"mean_mm": [float(x) for x in mean], "max_mm": [float(x) for x in mx],
                        "median_of_mean_mm": float(np.median(mean)), "median_of_max_mm": float(np.median(mx))}
            print(f"[{kind}] zero output, {len(starts)} resting starts, 30 frames, {name} the restoration: drift median "
                  f"{zr[name]['median_of_mean_mm']:.3f} mm (mean over vertices), {zr[name]['median_of_max_mm']:.3f} mm (max); "
                  f"worst start {max(mean):.2f} / {max(mx):.2f} mm", flush=True)
        r["rest"] = zr
        res[kind] = r
    (REPO / a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
