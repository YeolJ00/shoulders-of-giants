"""F8 -- how many observed steps does MYRIAD need? k-sweep on the ramp and the shove.

MYRIAD's condition is one image (frame 0) plus KNOWN positions of every track for steps
1..k AFTER it; it predicts the rest. So far k = 1 (1/12 s). On the ramp that is 0.3-3 px
-- nothing to go on. Here each candidate mu is conditioned on its OWN first k simulated
steps (k = 1, 2, 4 -> 0.08 / 0.17 / 0.33 s) and MYRIAD predicts the remainder of 1 s.

Only the CONTINUATION is compared -- positions from step k to step 12 -- since steps 1..k
are handed over and would match trivially. No external motion enters: every conditioning
step comes from the candidate being judged.

What a usable target looks like: for some k, MYRIAD's continuation agrees with the
simulator's at ONE mu (a crossing), that mu is the same across conditions (tilts / shove
speeds -- a material property), and it differs between objects. The trade-off being
measured: more given steps = more for MYRIAD to extrapolate kinematically, which fits every
candidate equally and removes discrimination.

    C scripts/f8_ksweep.py run --probe ramp|shove
    W scripts/f8_ksweep.py fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

KS = (1, 2, 4)
OUT = Path("docs/results/f8_ksweep.json")


def cases(probe):
    """(tag, cond, image path, spec rows of that cond, meta, query idx) per asset x cond."""
    import f4_fpt_joint as F
    import o13_tilted as T
    for tag, _ in T.ASSETS:
        if probe == "ramp":
            d = Path(f"outputs/scene/f6_{tag}")
            spec = json.loads((d / "spec.json").read_text())
            meta = json.loads((d / "meta.json").read_text())
            fpt = json.loads((d / "fpt.json").read_text())
            for tk, c in fpt.items():
                rows = sorted([r for r in spec if F.tkey(r["tilt"]) == tk], key=lambda r: r["mu"])
                yield tag, tk, d / f"sim_{tk}@a" / "f0000.png", rows, meta, np.array(c["idx"])
        else:
            import f7_shove as S
            d = Path(f"outputs/scene/f7_{tag}")
            spec = json.loads((d / "spec.json").read_text())
            meta = json.loads((d / "meta.json").read_text())
            idx = F.query_idx(meta[spec[0]["key"]][0])
            for v0 in S.V0:
                rows = sorted([r for r in spec if r["v0"] == v0], key=lambda r: r["mu"])
                yield tag, S.vkey(v0), d / "sim_s0@a" / "f0000.png", rows, meta, idx


def do_run(args):
    import time
    import torch
    import imageio.v2 as imageio
    import f4_fpt_joint as F
    import f5_myriad as M
    from src.render.crop import project_points
    m = M.load_model()
    W, H = F.wh()
    l, t, s = M.crop_sq(int(W), int(H))
    mus = F.MU[::3]
    jobs, keep = [], {}
    for tag, cond, imp, rows, meta, idx in cases(args.probe):
        im = imageio.imread(imp)[..., :3][t:t + s, l:l + s]
        byk = {r["mu"]: r for r in rows}
        for mu in mus:
            r = min(rows, key=lambda r_: abs(np.log(r_["mu"] / mu)))
            proj = [project_points(F.cam_at(0), np.asarray(meta[r["key"]][2 * j], float)[idx])[0]
                    for j in range(M.STEPS)]                      # MYRIAD grid, px
            proj = np.stack(proj)
            q = (proj[0] - [l, t]) / s
            for k in KS:
                jobs.append({"tag": tag, "cond": cond, "key": r["key"], "mu": r["mu"], "k": k,
                             "im": im, "q": q, "known": (proj[1:k + 1] - [l, t]) / s,
                             "sim": proj})
    print(f"{len(jobs)} jobs ({args.probe})", flush=True)
    groups = {}
    for j in jobs:
        groups.setdefault((len(j["q"]), j["k"]), []).append(j)
    out, t0, done = [], time.time(), 0
    kb = len(M.BG)
    for (nq, k), js in groups.items():
        for c0 in range(0, len(js), M.CHUNK):
            ch = js[c0:c0 + M.CHUNK]
            P = M.simulate_batch(m, [j["im"] for j in ch], [j["q"] for j in ch],
                                 [j["known"] for j in ch], seed=M.SEED + 100 * k + c0)
            for j, Pj in zip(ch, P):
                px = Pj[:, :, kb:] * s + np.array([l, t])          # E,T,N,2
                S_ = j["sim"]                                        # T,N,2
                dh = (S_[-1] - S_[0]).mean(0)
                dh = dh / max(np.linalg.norm(dh), 1e-9)
                cont_my = (px[:, -1] - px[:, k]).mean(1)             # E,2 continuation
                cont_sim = (S_[-1] - S_[k]).mean(0)
                out.append({"tag": j["tag"], "cond": j["cond"], "key": j["key"], "mu": j["mu"],
                            "k": k, "my_along": np.median(cont_my @ dh).item(),
                            "my_q": np.percentile(cont_my @ dh, [10, 90]).tolist(),
                            "sim_along": float(cont_sim @ dh),
                            "my_path": np.median(px.mean(2), 0).tolist(),     # T,2 centroid
                            "sim_path": S_.mean(1).tolist()})
            done += len(ch)
            print(f"  {done}/{len(jobs)}  {time.time()-t0:6.0f} s  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    Path(f"outputs/scene/f8_{args.probe}.json").write_text(json.dumps(out))
    return 0


def crossing(mus, my, sm):
    g = np.log(np.maximum(my, 1e-2)) - np.log(np.maximum(sm, 1e-2))
    for i in range(len(g) - 1):
        if g[i] * g[i + 1] <= 0 and g[i] != g[i + 1]:
            f = g[i] / (g[i] - g[i + 1])
            return float(np.exp(np.log(mus[i]) + f * (np.log(mus[i + 1]) - np.log(mus[i]))))
    return None


def do_fit(args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    for probe in ("ramp", "shove"):
        p = Path(f"outputs/scene/f8_{probe}.json")
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        tags = list(dict.fromkeys(r["tag"] for r in rows))
        conds = list(dict.fromkeys(r["cond"] for r in rows))
        res[probe] = {}
        print(f"\n=== {probe.upper()}: crossing mu (MYRIAD continuation = simulator's), "
              f"per condition; spread = max/min over conditions")
        print(f"{'asset':<8}{'k':>3}  " + "".join(f"{c:>8}" for c in conds) + f"{'spread':>9}"
              f"{'  MYRIAD/sim at each mu (median over conditions)':>0}")
        for tag in tags:
            res[probe][tag] = {}
            for k in KS:
                cr, ratios = [], []
                for c in conds:
                    rr = sorted([r for r in rows if r["tag"] == tag and r["cond"] == c
                                 and r["k"] == k], key=lambda r: r["mu"])
                    if not rr:
                        cr.append(None); continue
                    mus = [r["mu"] for r in rr]
                    my = [r["my_along"] for r in rr]; sm = [r["sim_along"] for r in rr]
                    cr.append(crossing(mus, my, sm))
                    ratios.append([m_ / max(s_, 1e-2) for m_, s_ in zip(my, sm)])
                good = [c for c in cr if c is not None]
                spread = (max(good) / min(good)) if len(good) >= 2 else None
                rmed = np.median(np.array(ratios), 0) if ratios else []
                res[probe][tag][k] = {"crossings": cr, "spread": spread,
                                      "ratio_by_mu": list(map(float, rmed))}
                print(f"{tag:<8}{k:>3}  " + "".join(f"{'-':>8}" if c is None else f"{c:>8.3f}"
                                                    for c in cr)
                      + (f"{spread:>8.2f}x" if spread else f"{'-':>9}") + "   "
                      + " ".join(f"{x:.2f}" for x in rmed))
        # figure: one condition, rows = assets, cols = k
        cmid = conds[len(conds) // 2]
        fig, axs = plt.subplots(len(tags), len(KS), figsize=(4.6 * len(KS), 2.9 * len(tags)),
                                dpi=90, squeeze=False)
        cmap = plt.get_cmap("viridis")
        for a, tag in enumerate(tags):
            for b, k in enumerate(KS):
                ax = axs[a, b]
                rr = sorted([r for r in rows if r["tag"] == tag and r["cond"] == cmid
                             and r["k"] == k], key=lambda r: r["mu"])
                for i, r in enumerate(rr):
                    col = cmap(i / max(len(rr) - 1, 1))
                    sp, mp = np.array(r["sim_path"]), np.array(r["my_path"])
                    ax.plot(sp[:, 0], -sp[:, 1], "-", color=col, lw=2)
                    ax.plot(mp[:, 0], -mp[:, 1], "--", color=col, lw=1.5)
                    ax.plot(mp[:k + 1, 0], -mp[:k + 1, 1], "o", color=col, ms=2.5)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{tag}  {probe} {cmid}  k={k}", fontsize=9)
        fig.suptitle(f"{probe}: SOLID = simulator, DASHED = MYRIAD given the first k steps "
                     f"(dots), one colour per mu (dark = low mu). Object centre, 1 s, px",
                     fontsize=10)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.98))
        fig.savefig(f"docs/results/f8_ksweep_{probe}.png")
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT} and docs/results/f8_ksweep_*.png")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s_ = sub.add_parser("run"); s_.set_defaults(f=do_run)
    s_.add_argument("--probe", choices=("ramp", "shove"), required=True)
    s_ = sub.add_parser("fit"); s_.set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
