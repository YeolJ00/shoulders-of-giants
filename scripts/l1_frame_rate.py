"""Is 30 fps too coarse for the cloth's dynamics (2026-11-18; the user: test the 30 fps hypothesis)?

Model-free. One batch of c1-style episodes with the cloth also recorded at `--sub` instants inside every 30 fps frame
(ClothTeacher.sub), then the same motion read at 30, 60, ... 30 * sub fps. At each rate, over free (unheld) moving
cloth vertices, with a_t = x_{t+1} - 2 x_t + x_{t-1} the per-frame acceleration:
  cv     mean |a_t|                    constant velocity's one-step error
  ca     mean |a_t - a_{t-1}|          the error of repeating the last acceleration
  new    ca / cv                       how much of each frame's acceleration is new: << 1 where the rate resolves the
                                       motion (smooth), ~1.4 where consecutive accelerations are unrelated (aliased)
  rho    correlation of a_t with a_{t-1}
and the share of the acceleration's power above 15 Hz (30 fps' Nyquist) in the finest recording.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_frame_rate.py --sub 16
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=16)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=4.0)
    a = ap.parse_args()
    import warp as wp
    from lsim.data import library
    from lsim.data.cloth import run_batch, sample_spec
    wp.init()
    _, test = library.object_split("gso")
    rng = np.random.default_rng(a.seed)
    specs = [sample_spec(rng, test, seed=a.seed * 10000 + k) for k in range(a.worlds)]
    res = run_batch(specs, seed=a.seed, record=a.seconds, sub=a.sub)
    X, S, settle = res["x"].astype(np.float64), res["x_sub"].astype(np.float64), res["settle"]
    F = S.shape[0]
    full = np.concatenate([X[:F, None], S], 1).reshape(F * a.sub, -1, 3)       # (F * sub, V, 3) at 30 * sub fps
    full = np.concatenate([full, X[F:F + 1]], 0)
    held = np.zeros((F, full.shape[1]), bool)                                 # pinned during frame f (30 fps)
    cloth = np.zeros(full.shape[1], bool)
    for w, wd in enumerate(res["worlds"]):
        p0, n = wd["particles"]
        cloth[p0:p0 + n] = True
        cur = np.zeros(n, bool)
        ev = sorted(res["events"][w], key=lambda e: e[0])
        k = 0
        for f in range(settle + F):
            while k < len(ev) and ev[k][0] <= f:
                cur = np.zeros(n, bool)
                if ev[k][1] == "grasp" and len(ev[k]) > 3:
                    cur[np.asarray(ev[k][3], int)] = True
                k += 1
            if f >= settle:
                held[f - settle, p0:p0 + n] = cur
    np.savez_compressed(REPO / "outputs/lsim/frame_rate_x.npz", x=full.astype(np.float32), held=held, cloth=cloth, sub=a.sub)
    out = {"sub": a.sub, "substeps": int(res["substeps"]), "worlds": a.worlds, "rates": {}}
    heldf = np.repeat(held, a.sub, 0)                                         # per fine frame
    print(f"{a.worlds} episodes x {a.seconds} s, {res['substeps']} substeps per 30 fps frame, recorded at {30 * a.sub} fps")
    s = 1
    steps = []
    while s <= a.sub:
        steps.append(s); s *= 2
    for s in steps[::-1]:
        x = full[::s]
        h = heldf[::s][:len(x) - 1]
        acc = x[2:] - 2 * x[1:-1] + x[:-2]                                    # (T - 2, V, 3): a_t at frame t = 1..
        free = cloth[None] & ~h[1:] & ~h[:-1]                                 # free in the steps into and out of t
        a1, a0 = acc[1:], acc[:-1]
        ok = free[1:] & free[:-1]
        fps = 30 * a.sub // s
        speed = np.linalg.norm(x[2:-1] - x[1:-2], axis=-1) * fps
        mv = ok & (speed > 0.02)                                              # moving: > 2 cm / s
        n1, n0, d = np.linalg.norm(a1, axis=-1)[mv], np.linalg.norm(a0, axis=-1)[mv], np.linalg.norm(a1 - a0, axis=-1)[mv]
        rho = float((a1[mv] * a0[mv]).sum() / np.sqrt((a1[mv] ** 2).sum() * (a0[mv] ** 2).sum()))
        row = dict(cv_mm=1e3 * float(n1.mean()), ca_mm=1e3 * float(d.mean()), new=float(d.mean() / n1.mean()), rho=rho,
                   acc_ms2=float(n1.mean()) * fps * fps, n=int(mv.sum()))
        out["rates"][fps] = row
        print(f"  {fps:4d} fps: constant velocity {row['cv_mm']:.4f} mm   repeat last acceleration {row['ca_mm']:.4f} mm   "
              f"new share {row['new']:.2f}   corr(a_t, a_t-1) {rho:+.2f}   |a| {row['acc_ms2']:.1f} m/s^2")
    v = (full[1:] - full[:-1]) * 30 * a.sub                                   # spectrum of free vertices' velocity
    okv = cloth & ~held.any(0)
    idx = np.flatnonzero(okv)[::7]
    sp = (np.abs(np.fft.rfft(v[:, idx], axis=0)) ** 2).sum((1, 2))
    fr = np.fft.rfftfreq(v.shape[0], 1 / (30 * a.sub))
    pa = sp * fr ** 2                                                         # acceleration power
    out["freq"], out["acc_power"] = fr.tolist(), pa.tolist()
    for cut in (15, 30, 60, 120):
        if cut < fr[-1]:
            out[f"acc_power_above_{cut}"] = float(pa[fr > cut].sum() / pa.sum())
            print(f"  acceleration power above {cut:3d} Hz: {100 * out[f'acc_power_above_{cut}']:.1f} %")
    json.dump(out, open(REPO / "docs/results/l1_frame_rate.json", "w"))


if __name__ == "__main__":
    main()
