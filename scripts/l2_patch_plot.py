"""One figure for the stencil input (2026-11-20, `--patch 4`): the same 64M network and recipe with and without each
vertex's 9x9 grid stencil in its own axes as input -- one frame on two test sets (paired over windows), against the
local rule's learning curve on its vertex set, and over seconds (paired over episodes).

    $L scripts/l2_patch_plot.py --out <png>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from l2_oracle_v_plot import paired  # noqa: E402

NET, PATCH, LOCAL, CV, TEACHER, INK = "#4c78a8", "#b2182b", "#f58518", "#9aa0a6", "#54a24b", "#333333"


def boot_mean_diff(a, b, n_boot=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = REPO / "docs/results"
    J = lambda n: json.load(open(R / f"{n}.json"))
    have = all((R / f"l2_cloth_long_{n}.json").exists() for n in ("ck1_step7999_n256", "patch4_step7999_n256"))
    if have:
        fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.15, 1.1, 1]})
    else:                                                 # the rollouts over seconds not in yet
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) the fit against training windows seen, on the same 300 c1_test windows (paired across all points)
    ax = axs[0]
    q = [(64, "q64_8000_on_c1_R6"), (96, "q64_12000_on_c1_R6"), (128, "q64_16000_on_c1_R6"),
         (160, "q64_20000_on_c1_R6"), (192, "q64_on_c1_R6")]
    qx, qy = [k for k, _ in q], [J(f"l2_error_where_{n}")["free"]["ratio"] for _, n in q]
    ax.plot(qx, qy, color=NET, lw=2, marker="o", ms=7, zorder=3)
    ck, pt = J("l2_error_where_ck1_on_c1_300R6")["free"]["ratio"], J("l2_error_where_patch4_on_c1_R6")["free"]["ratio"]
    ax.scatter([32], [ck], s=70, color=NET, marker="s", zorder=3)
    ax.scatter([32], [pt], s=70, color=PATCH, marker="s", zorder=3)
    for x_, y_ in zip(qx, qy):
        ax.annotate(f"{y_:.2f}", (x_, y_), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
    ax.annotate(f"{ck:.2f}", (32, ck), textcoords="offset points", xytext=(10, -3), ha="left", fontsize=8.5, color=INK)
    ax.annotate(f"{pt:.2f}", (32, pt), textcoords="offset points", xytext=(10, -3), ha="left", fontsize=8.5, color=INK)
    tr = {n: J(f"l2_error_where_{n}_on_train")["free"]["ratio"] for n in ("q64", "ck1", "patch4")}
    ax.scatter([192], [tr["q64"]], s=70, facecolor="white", edgecolor=NET, lw=1.8, zorder=4)
    ax.scatter([32], [tr["ck1"]], s=60, facecolor="white", edgecolor=NET, lw=1.8, marker="s", zorder=4)
    ax.scatter([32], [tr["patch4"]], s=60, facecolor="white", edgecolor=PATCH, lw=1.8, marker="s", zorder=4)
    ax.annotate(f"training windows {tr['q64']:.2f}", (192, tr["q64"]), textcoords="offset points", xytext=(-8, -16),
                ha="right", fontsize=8.5, color=INK)
    ax.text(36, 0.93, "1 GPU, 8,000 steps", fontsize=8.5, color=INK)
    ax.text(135, 0.80, "64M: 2 GPUs, one 24k-step run\n(checkpoints mid-schedule)", fontsize=8.5, color=INK)
    # the 509M network at the same windows seen (4 GPUs x 4, 12k steps), the stencil at full compute, the continuation
    big = [(64, "q509_4000_on_c1_R6"), (128, "q509_8000_on_c1_R6"), (192, "q509_11999_on_c1_R6")]
    big = [(x_, J(f"l2_error_where_{n}")["free"]["ratio"]) for x_, n in big if (R / f"l2_error_where_{n}.json").exists()]
    if big:
        ax.plot(*zip(*big), color=NET, lw=1.5, ls="--", marker="^", ms=7, zorder=3, label="509M network (8x the parameters)")
    pf = [(x_, f"q64_patch4_{s}_on_c1_R6") for x_, s in ((64, 8000), (96, 12000), (128, 16000), (192, 23999))]
    pf = [(x_, J(f"l2_error_where_{n}")["free"]["ratio"]) for x_, n in pf if (R / f"l2_error_where_{n}.json").exists()]
    if pf:
        ax.plot(*zip(*pf), color=PATCH, lw=2, marker="o", ms=6, zorder=3)
    co = [(192 + x_, f"q64_cont_{s}_on_c1_R6") for x_, s in ((64, 8000), (96, 12000), (128, 16000), (192, 23999))]
    co = [(x_, J(f"l2_error_where_{n}")["free"]["ratio"]) for x_, n in co if (R / f"l2_error_where_{n}.json").exists()]
    if co:
        ax.plot([192] + [c[0] for c in co], [qy[-1]] + [c[1] for c in co], color=NET, lw=2, ls=":", marker="o", ms=6,
                zorder=3)
        ax.annotate("continued for 24k more steps\n(new schedule)", co[-1], textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.5, color=INK)
    ax.axhline(1.0, color=CV, lw=1, ls=":")
    ax.text(25, 1.01, "constant velocity", fontsize=8.5, color="#555", ha="left")
    ax.set_xlim(20, 400 if co else 205)
    ax.set_ylim(0.55, 1.16)
    ax.set_xlabel("training windows seen (thousands; c1 holds 20k episodes)", fontsize=9)
    ax.set_ylabel("one-step error / constant velocity's\n(moving free cloth, 300 c1_test windows)", fontsize=9)
    ax.set_title("the fit improves with training, not with 8x the parameters;\ntraining windows score like held-out ones", fontsize=10)
    ax.scatter([], [], s=50, color=NET, label="64M network (positions, 1-ring)")
    ax.scatter([], [], s=50, color=PATCH, marker="s", label="same + each vertex's 9x9 stencil in its own axes")
    ax.scatter([], [], s=50, facecolor="white", edgecolor="#666", label="filled: held out; hollow: training windows")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    # (b) against the local rule, on its vertex set
    ax = axs[1]
    sc = J("l1_local_probe_scale")
    base = dict(R=4, local=True, material=True, width=1024, depth=3)
    curve = sorted([r for r in sc["rows"] if all(r[k] == v for k, v in base.items())], key=lambda r: r["episodes"])
    x = [r["episodes"] for r in curve]
    ax.plot(x, [r["ratio"] for r in curve], color=LOCAL, lw=2, marker="o", ms=7, zorder=2)
    for r in curve:
        ax.annotate(f"{r['ratio']:.2f}", (r["episodes"], r["ratio"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color=INK)
    ax.text(50, 0.62, "small MLP on each vertex's 9x9\nneighbourhood (own axes), held out", fontsize=8.5, color=INK)
    nets = [("ck1", NET, "D", "8,000 steps"), ("patch4", PATCH, "D", "+ stencil, 8,000 steps"),
            ("q64", NET, "*", "24k steps x 2 GPUs"), ("q64_patch4", PATCH, "*", "+ stencil, 24k x 2"),
            ("q64_cont", NET, "P", "trained twice as long")]
    full = [n for n in ("q64", "q64_patch4", "q64_cont") if (R / f"l2_error_where_{n}_on_c1_2000R6.json").exists()]
    for n, c, mk, lab in nets:
        f = R / f"l2_error_where_{n}_on_c1_2000R6.json"
        if not f.exists():
            continue
        v = J(f"l2_error_where_{n}_on_c1_2000R6")["probe"]["ratio"]
        ax.scatter([20032], [v], s=80 if mk != "*" else 110, color=c, marker=mk, zorder=4)
        if n in full and len(full) > 1:                   # the fully trained ones sit within 0.03: one label block
            continue
        ax.annotate(f"{lab}: {v:.2f}", (20032, v), textcoords="offset points", xytext=(10, 0), ha="left",
                    fontsize=8.5, color=INK, va="center")
    if len(full) > 1:
        lines = [f"{lab}: {J(f'l2_error_where_{n}_on_c1_2000R6')['probe']['ratio']:.3f}" for n, _, _, lab in nets if n in full]
        ax.annotate("\n".join(lines), (20032, 0.66), textcoords="offset points", xytext=(12, -14), ha="left", va="top",
                    fontsize=8.5, color=INK)
    ax.axhline(1.0, color=CV, lw=1, ls=":")
    ax.text(45, 1.015, "constant velocity", fontsize=8.5, color="#555")
    tf = J("l1_teacher_from_inputs")["classes"]["off"]["back2"]["ratio"]
    ax.axhline(tf, color=TEACHER, lw=1, ls=":")
    ax.text(45, tf + 0.015, f"the teacher restarted from 3 frames: {tf:.2f}", fontsize=8.5, color="#555")
    ax.set_xscale("log")
    ax.set_xlim(30, 400000)
    ax.set_ylim(0, 1.2)
    ax.text(20032, 0.95, "networks\n(20k episodes)", fontsize=8.5, color=INK, ha="center")
    ax.set_xlabel("training episodes (c1, all materials)", fontsize=9)
    ax.set_ylabel("one-step error / constant velocity's", fontsize=9)
    ax.set_title("the 64M network against a local rule on its vertices\n(interior, in the air; 2,000 c1_test windows)", fontsize=10)

    # (c) over seconds, paired over episodes
    T = [("1 s", "e30"), ("2 s", "e60"), ("end (3.8 s)", "end")]
    if have:
        ax = axs[2]
        lines = [("ck1_step7999_n256", NET, "-", 1.2, "o", "network, 8,000 steps (1 GPU)"),
                 ("patch4_step7999_n256", PATCH, "-", 1.2, "o", "+ stencil, 8,000 steps (1 GPU)"),
                 ("q64_step23999_n256", NET, "-", 2.4, "*", "network, 24k steps x 2 GPUs"),
                 ("q64_patch4_step23999_n256", PATCH, "-", 2.4, "*", "+ stencil, 24k steps x 2 GPUs"),
                 ("q64_cont_step23999_n256", NET, ":", 2.0, "D", "network, trained twice as long")]
        for n, c, ls, lw, mk, lab in lines:
            if not (R / f"l2_cloth_long_{n}.json").exists():
                continue
            E = J(f"l2_cloth_long_{n}")["episodes"]
            ax.plot(range(3), [np.mean([e[k] for e in E]) for _, k in T], color=c, ls=ls, lw=lw, marker=mk,
                    ms=10 if mk == "*" else 5, label=lab, alpha=0.55 if lw < 2 else 1)
        full = (R / "l2_cloth_long_q64_patch4_step23999_n256.json").exists()
        Eb = J("l2_cloth_long_q64_step23999_n256" if full else "l2_cloth_long_ck1_step7999_n256")["episodes"]
        Ep = J("l2_cloth_long_q64_patch4_step23999_n256" if full else "l2_cloth_long_patch4_step7999_n256")["episodes"]
        ticks = []
        for i, (t, k) in enumerate(T):
            d, lo, hi = boot_mean_diff([e[k] for e in Ep], [e[k] for e in Eb])
            ticks.append(f"{t}\nstencil {d:+.0f} mm\n[{lo:+.0f}, {hi:+.0f}]")
        if full:                                            # shape at the end, fully trained (median over episodes)
            sh = {n: np.median([e["strain_pct_end"] for e in J(f"l2_cloth_long_{n}_step23999_n256")["episodes"]])
                  for n in ("q64", "q64_patch4")}
            ax.text(0.98, 0.04, f"edge strain at the end, fully trained:\nnetwork {sh['q64']:.0f}%, + stencil {sh['q64_patch4']:.1f}%"
                    f" (teacher 0.01%)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        ax.set_xticks(range(3))
        ax.set_xticklabels(ticks, fontsize=8.5)
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("free-vertex distance from the teacher (mm,\nmedian per episode, mean of 256 episodes)", fontsize=9)
        ax.set_title("over seconds: rolled out from frame 7 alone, material\nhidden (256 c1_test episodes; ticks: fully trained, paired)", fontsize=10)
    for ax in axs:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("The 64M cloth network: how long it trains, the stencil input, a local rule, and seconds", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
