"""F12 -- MYRIAD as a LIKELIHOOD over mu: is its sample density lopsided even if wide?

f8's band test found MYRIAD's 10-90% range admits almost every mu over 20x. A wide band can
still carry a peaked density. Here each asset x shove speed gets ONE shared conditioning --
the frictionless launch (a rigid shift of v0 * 1/12 s, the same for every candidate) -- so
every mu is scored under the SAME distribution: a proper likelihood ratio, not the
across-distribution comparison that sank the FPT mu-sweep.

  256 samples per condition (4 chunks of 64), their protocol (k = 1, camera_static) at
  fp32, the precision OWM was reproduced at (f10). Each of the 18 simulated mu is scored by the log-density of its object-
  centre path: a 2-D Gaussian KDE (Scott bandwidth) at each of the 12 steps, summed
  (steps treated as independent -- a pseudo-likelihood).

A usable signal: an interior maximum, the same argmax at every speed (a material property
is speed-independent), different argmax between objects, and a log-likelihood gap that
beats the sampling noise (bootstrap over samples).

    C scripts/f12_likelihood.py run
    W scripts/f12_likelihood.py fit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

ENS = 256
OUT = Path("outputs/scene/f12_samples.npz")
RES = Path("docs/results/f12_likelihood.json")


def do_run(args):
    import time
    import imageio.v2 as imageio
    import f4_fpt_joint as F
    import f5_myriad as M
    import f7_shove as S
    import o13_tilted as T
    from src.render.crop import project_points
    # .float() also converts the attention KV cache, which MYRIAD allocates in bf16 -- the
    # same thing their benchmark does with model.to(float32) (f10).
    m = M.load_model().float()
    W, H = F.wh()
    l, t, s = M.crop_sq(int(W), int(H))
    kb = len(M.BG)
    store, t0 = {}, time.time()
    for tag, _ in T.ASSETS:
        d = S.lab(tag)
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        k0 = spec[0]["key"]
        idx = F.query_idx(meta[k0][0])
        P0 = np.asarray(meta[k0][0], float)[idx]
        im = imageio.imread(d / "sim_s0@a" / "f0000.png")[..., :3][t:t + s, l:l + s]
        u0, _ = project_points(F.cam_at(0), P0)
        for v0 in S.V0:
            # the frictionless launch: every point shifted by v0 * (1/12 s) along +x
            u1, _ = project_points(F.cam_at(0), P0 + np.array([v0 / 12.0, 0.0, 0.0]))
            Pm = np.concatenate([M.simulate_batch(m, [im], [(u0 - [l, t]) / s],
                                                  [(u1 - [l, t]) / s], n_ens=ENS // 4,
                                                  seed=M.SEED + c)[0] for c in range(4)])
            my = Pm[:, :, kb:].mean(2) * s + [l, t]                       # E,T,2 centroid px
            sims = []
            for mu in F.MU:
                key = S.mkey(v0, mu)
                pr = np.stack([project_points(F.cam_at(0),
                                              np.asarray(meta[key][2 * j], float)[idx])[0]
                               for j in range(M.STEPS)]).mean(1)       # T,2
                sims.append(pr)
            store[f"{tag}_{S.vkey(v0)}_my"] = my
            store[f"{tag}_{S.vkey(v0)}_sim"] = np.stack(sims)
            print(f"  {tag} v0={v0}  {time.time()-t0:5.0f} s", flush=True)
    store["mu"] = np.asarray(F.MU)
    np.savez(OUT, **store)
    return 0


def kde_logp(samples, x):
    """2-D Gaussian KDE, Scott bandwidth (full covariance), log-density at x."""
    n = len(samples)
    C = np.cov(samples.T) * n ** (-2.0 / 6.0) + np.eye(2) * 1e-6
    Ci = np.linalg.inv(C)
    d = x[None] - samples
    q = np.einsum("ni,ij,nj->n", d, Ci, d)
    lk = -0.5 * q - 0.5 * np.log(np.linalg.det(2 * np.pi * C))
    mx = lk.max()
    return mx + np.log(np.exp(lk - mx).mean())


def path_logp(my, sim, steps):
    return sum(kde_logp(my[:, j], sim[j]) for j in steps)


def do_fit(args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import o13_tilted as T
    import f7_shove as S
    Z = np.load(OUT)
    mu = Z["mu"]
    steps = range(2, 13)                 # step 1 is the given poke
    rng = np.random.default_rng(0)
    res = {}
    tags = [t for t, _ in T.ASSETS]
    fig, axs = plt.subplots(1, len(tags), figsize=(3.2 * len(tags), 3.2), dpi=100, sharey=False)
    print("log-likelihood of each simulated mu's path under MYRIAD (shared launch poke)")
    print(f"{'asset':<7}{'v0':>5}{'argmax mu':>11}{'interior':>10}{'gap max-min':>13}"
          f"{'boot sd':>9}{'self-typical':>14}")
    for a, tag in enumerate(tags):
        res[tag] = {}
        for v0 in S.V0:
            k = f"{tag}_{S.vkey(v0)}"
            my, sim = Z[k + "_my"], Z[k + "_sim"]
            L = np.array([path_logp(my, s_, steps) for s_ in sim])
            i = int(np.argmax(L))
            # sampling noise of the argmax: bootstrap MYRIAD's samples
            bi = []
            for _ in range(30):
                bs = my[rng.integers(0, len(my), len(my))]
                bi.append(int(np.argmax([path_logp(bs, s_, steps) for s_ in sim])))
            # how typical is the BEST sim path compared with MYRIAD's own samples
            # (leave-one-out log-likelihood of 40 of its own samples)
            own = [path_logp(np.delete(my, j, 0), my[j], steps) for j in range(40)]
            typ = float((np.array(own) < L[i]).mean())
            res[tag][S.vkey(v0)] = {"logL": L.tolist(), "argmax_mu": float(mu[i]),
                                    "interior": bool(0 < i < len(mu) - 1),
                                    "boot_argmax_mu": [float(mu[b]) for b in bi],
                                    "typical_pct": typ}
            bsd = float(np.std(np.log(mu[bi])))
            print(f"{tag:<7}{v0:>5}{mu[i]:>11.3f}{str(0 < i < len(mu)-1):>10}"
                  f"{L.max()-L.min():>13.1f}{np.exp(bsd):>8.2f}x{100*typ:>12.0f}%")
            axs[a].plot(mu, L - L.max(), "-o", ms=3, label=f"v0={v0}")
        axs[a].set_xscale("log"); axs[a].set_title(tag, fontsize=10)
        axs[a].set_xlabel("mu"); axs[a].set_ylim(-60, 5)
    axs[0].set_ylabel("log L - max"); axs[0].legend(fontsize=7)
    fig.suptitle("Shove: log-likelihood of the simulated path under MYRIAD's 256 samples, "
                 "per mu (shared frictionless launch poke)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("docs/results/f12_likelihood.png")
    print("\nACROSS SPEEDS -- argmax mu per asset (a material property must not depend on v0)")
    for tag in tags:
        am = [res[tag][S.vkey(v)]["argmax_mu"] for v in S.V0]
        print(f"  {tag:<7} " + "  ".join(f"{x:.3f}" for x in am) + f"   spread {max(am)/min(am):.2f}x")
    RES.write_text(json.dumps(res, indent=1))
    print("wrote", RES, "and docs/results/f12_likelihood.png")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(f=do_run)
    sub.add_parser("fit").set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
