"""F9 -- MYRIAD per POINT, every sample kept: our probes next to MYRIAD's own demo images.

f8 stored only the median path of the object's centre, which hides what the model does to
individual points (rigid motion? scatter? points diverging?). Here: 4 assets x {ramp 24deg,
shove 1.2 m/s} x k = 1/2/4, one mu each, all 16 points x ENS samples; plus MYRIAD's own
qual.py images with their own pokes as the reference for in-distribution behaviour.

    C scripts/f9_points.py run
    W scripts/f9_points.py plot
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

ASSETS = ("box", "sorter", "books", "teapot")
PICK = {"ramp": ("t24", 0.404), "shove": ("v12", 0.147)}
KS = (1, 2, 4)
OUT = Path("outputs/scene/f9_points.json")
DEMO = {  # MYRIAD's own qual.py examples: object queries + their pokes (fraction of frame)
    "ball_roll": ([(0.27, 0.72)], [(0.05, 0.0)]),
    "falling_ball": ([(0.31, 0.61), (0.28, 0.60), (0.33, 0.64), (0.34, 0.57), (0.28, 0.68)],
                     [(-0.02, 0.05)] * 5),
    "egg_roll": ([(0.27, 0.72)], [(0.05, 0.0)]),
}


def do_run(args):
    import imageio.v2 as imageio
    from PIL import Image
    import f4_fpt_joint as F
    import f5_myriad as M
    import f8_ksweep as K
    from src.render.crop import project_points
    m = M.load_model()
    W, H = F.wh()
    l, t, s = M.crop_sq(int(W), int(H))
    out = {"ours": [], "demo": {}}
    for probe, (cond, mu) in PICK.items():
        for tag, cnd, imp, rows, meta, idx in K.cases(probe):
            if tag not in ASSETS or cnd != cond:
                continue
            r = min(rows, key=lambda r_: abs(np.log(r_["mu"] / mu)))
            proj = np.stack([project_points(F.cam_at(0),
                                            np.asarray(meta[r["key"]][2 * j], float)[idx])[0]
                             for j in range(M.STEPS)])
            im = imageio.imread(imp)[..., :3]
            for k in KS:
                P = M.simulate_batch(m, [im[t:t + s, l:l + s]], [(proj[0] - [l, t]) / s],
                                     [(proj[1:k + 1] - [l, t]) / s], seed=M.SEED + k)[0]
                out["ours"].append({"probe": probe, "tag": tag, "cond": cond, "mu": r["mu"],
                                    "k": k, "img": str(imp),
                                    "my": (P[:, :, len(M.BG):] * s + [l, t]).tolist(),
                                    "sim": proj.tolist()})
                print(f"  {probe} {tag} k={k}", flush=True)
    for nm, (q, pk) in DEMO.items():
        img = Image.open(REPO / f"models/myriad/qual_examples/{nm}.jpg").convert("RGB")
        w, h = img.size; sd = min(w, h); ld, td = (w - sd) // 2, (h - sd) // 2
        sq = np.asarray(img.crop((ld, td, ld + sd, td + sd)).resize((768, 768)))
        qq = (np.array(q) * [w, h] - [ld, td]) / sd
        q1 = qq + np.array(pk) * [w, h] / sd
        P = M.simulate_batch(m, [sq], [qq], [q1])[0]
        out["demo"][nm] = {"my": (P[:, :, len(M.BG):] * 768).tolist(), "q0": (qq * 768).tolist(),
                           "sq": sq.tolist() if False else None}
        imageio.imwrite(f"outputs/scene/_f9_{nm}.png", sq)
        print(f"  demo {nm}", flush=True)
    OUT.write_text(json.dumps(out))
    return 0


def do_plot(args):
    import imageio.v2 as imageio
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = json.loads(OUT.read_text())
    NS = 6                                     # samples drawn per point
    for probe in PICK:
        rows = [r for r in R["ours"] if r["probe"] == probe]
        fig, axs = plt.subplots(len(ASSETS), len(KS), figsize=(5.0 * len(KS), 4.2 * len(ASSETS)),
                                dpi=95, squeeze=False)
        for a, tag in enumerate(ASSETS):
            for b, k in enumerate(KS):
                ax = axs[a, b]
                r = next(x for x in rows if x["tag"] == tag and x["k"] == k)
                my, sim = np.array(r["my"]), np.array(r["sim"])     # E,T,N,2 / T,N,2
                ax.imshow(imageio.imread(r["img"])[..., :3])
                for j in range(sim.shape[1]):
                    ax.plot(sim[:, j, 0], sim[:, j, 1], "-", color="#2b6fd6", lw=1.3, alpha=0.9)
                    for e in range(NS):
                        ax.plot(my[e, k:, j, 0], my[e, k:, j, 1], "-", color="#ff9f1a",
                                lw=0.7, alpha=0.55)
                    ax.plot(my[0, :k + 1, j, 0], my[0, :k + 1, j, 1], "-", color="w", lw=1.6)
                ax.scatter(sim[0, :, 0], sim[0, :, 1], s=7, color="w", zorder=5)
                c = sim[0].mean(0)
                span = max(np.ptp(sim[:, :, 0]), np.ptp(sim[:, :, 1]), 120) * 0.75 + 40
                ax.set_xlim(c[0] - 60, c[0] + span); ax.set_ylim(c[1] + span, c[1] - 70)
                ax.axis("off")
                ax.set_title(f"{tag}  {probe} {r['cond']}  mu={r['mu']:.2f}  k={k}", fontsize=10)
        fig.suptitle(f"{probe.upper()}, per point.  BLUE = simulator (16 points, 1 s).  "
                     f"WHITE = the k steps handed to MYRIAD.  ORANGE = MYRIAD's continuation, "
                     f"{NS} samples per point", fontsize=11)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.985))
        fig.savefig(f"docs/results/f9_points_{probe}.png")
    # MYRIAD's own images, same drawing
    fig, axs = plt.subplots(1, len(R["demo"]), figsize=(5.2 * len(R["demo"]), 5.4), dpi=95)
    for ax, (nm, d) in zip(axs, R["demo"].items()):
        my = np.array(d["my"])
        ax.imshow(imageio.imread(f"outputs/scene/_f9_{nm}.png"))
        for j in range(my.shape[2]):
            for e in range(12):
                ax.plot(my[e, 1:, j, 0], my[e, 1:, j, 1], "-", color="#ff9f1a", lw=0.8, alpha=0.6)
            ax.plot(my[0, :2, j, 0], my[0, :2, j, 1], "-", color="w", lw=2)
        ax.axis("off"); ax.set_title(f"MYRIAD's own example: {nm} (k=1, its own poke)")
    fig.tight_layout()
    fig.savefig("docs/results/f9_points_demo.png")
    print("wrote docs/results/f9_points_{ramp,shove,demo}.png")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(f=do_run)
    sub.add_parser("plot").set_defaults(f=do_plot)
    a = ap.parse_args()
    sys.exit(a.f(a))
