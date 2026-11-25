"""One figure from l1_local_probe_scale (2026-11-20): the local rule's learning curve over c1's episodes, with the 64M
network on the same kind of vertices, and what the fit at all episodes depends on.

    $L scripts/l1_local_probe_scale_plot.py --out <png>
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL, NET, TEACHER, CV, DENSE = "#f58518", "#4c78a8", "#54a24b", "#9aa0a6", "#a14f00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = REPO / "docs/results"
    sc = json.load(open(R / "l1_local_probe_scale.json"))
    net = json.load(open(R / "l2_error_where_ck1_on_c1_R6.json"))["probe"]["ratio"]
    teach = json.load(open(R / "l1_teacher_from_inputs.json"))["classes"]
    rows = sc["rows"]
    base = dict(R=4, local=True, material=True, width=1024, depth=3)
    curve = sorted([r for r in rows if all(r[k] == v for k, v in base.items())], key=lambda r: r["episodes"])
    full = curve[-1]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    x = [r["episodes"] for r in curve]
    ax.plot(x, [r["ratio"] for r in curve], color=LOCAL, lw=2, marker="o", ms=8, zorder=3)
    ax.plot(x, [r["train_ratio"] for r in curve], color=LOCAL, lw=1.2, ls="--", marker="o", ms=5, mfc="white", zorder=3)
    for r in curve:
        ax.annotate(f"{r['ratio']:.2f}", (r["episodes"], r["ratio"]), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color="#333")
    ax.text(x[1] * 1.1, curve[1]["train_ratio"] - 0.06, "on its own training samples", fontsize=8.5, color=LOCAL)
    ax.text(36, 0.70, "small MLP on each vertex's\n9x9 neighbourhood, held out", fontsize=9, color=LOCAL, ha="left")
    pe = R / "l1_local_probe_per_ep.json"
    if pe.exists():                                       # the 3x denser draw from the same episodes
        dense = sorted([r for r in json.load(open(pe))["rows"] if r["samples"] == "all"], key=lambda r: r["episodes"])
        ax.plot([r["episodes"] for r in dense], [r["ratio"] for r in dense], color=DENSE, lw=1.5, marker="s", ms=6,
                zorder=3)
        ax.annotate(f"3x the picks from the same\nepisodes: {dense[-1]['ratio']:.2f}", (dense[-1]["episodes"],
                    dense[-1]["ratio"]), textcoords="offset points", xytext=(0, -30), ha="center", fontsize=8.5,
                    color=DENSE)
    ax.scatter([20032], [net], s=90, color=NET, zorder=4, marker="D")
    ax.annotate(f"64M network, trained on\nall 20k episodes: {net:.2f}", (20032, net), textcoords="offset points",
                xytext=(-10, 16), ha="right", fontsize=9, color="#333")
    ax.axhline(1.0, color=CV, lw=1, ls=":")
    ax.text(45, 1.015, "constant velocity", fontsize=8.5, color="#555")
    tf = teach["off"]["back2"]["ratio"] if "back2" in teach.get("off", {}) else None
    if tf is not None:
        ax.axhline(tf, color=TEACHER, lw=1, ls=":")
        ax.text(45, tf + 0.015, f"the teacher restarted from 3 frames (all cloth in the air): {tf:.2f}", fontsize=8.5, color="#555")
    ax.set_xscale("log")
    ax.set_xlim(30, 40000)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("training episodes (c1, all materials)", fontsize=9)
    ax.set_ylabel("one-step error / constant velocity's", fontsize=9)
    ax.set_title("a local rule keeps improving with episodes;\nthe network is where ~200 episodes put it", fontsize=10)
    ax = axs[1]
    var = [("9x9, own axes, material given", full),
           ("5x5 neighbourhood", next(r for r in rows if r["R"] == 2)),
           ("13x13 neighbourhood", next(r for r in rows if r["R"] == 6)),
           ("material hidden", next(r for r in rows if not r["material"])),
           ("world axes (as the network sees them)", next(r for r in rows if not r["local"])),
           ("4x wider-and-deeper MLP", next(r for r in rows if r["width"] > 1024))]
    y = list(range(len(var)))[::-1]
    ax.barh(y, [v[1]["ratio"] for v in var], color=[LOCAL] + [LOCAL] * 5, alpha=1.0, height=0.62)
    for yi, (name, r) in zip(y, var):
        ax.text(r["ratio"] - 0.012, yi, f"{r['ratio']:.3f}", va="center", ha="right", fontsize=9, color="white")
    ax.axvline(net, color=NET, lw=1.5, ls="--")
    ax.text(net + 0.01, len(var) - 0.45, f"64M network {net:.2f}", fontsize=8.5, color=NET)
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in var], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("one-step error / constant velocity's, all 13.7k episodes", fontsize=9)
    ax.set_title("what the local fit at all episodes depends on", fontsize=10)
    for ax in axs:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Moving cloth in the air, interior vertices; held-out c1_test ({sc['n_test']:,} vertex-steps)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
