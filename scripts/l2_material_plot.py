"""Does the one-frame map need the material, and does the network use it (2026-11-19)? One figure from
l1_teacher_sensitivity (the teacher restarted from its full state with the material changed) and l2_error_where
--material (the networks with their given material hidden or another episode's), moving cloth in the air.

    $L scripts/l2_material_plot.py --out <png>
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEACHER, NET, NET_V = "#54a24b", "#4c78a8", "#e45756"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", default="off", help="teacher class (off = not on a flat surface); networks use 'free'")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = REPO / "docs/results"
    ts = json.load(open(R / "l1_teacher_sensitivity.json"))["classes"][a.cls]
    ctl = json.load(open(R / "l2_error_where_ck1_on_c1v_mat.json"))["free"]
    orc = json.load(open(R / "l2_error_where_oracle_v_7999_mat.json"))["free"]
    moves = [("teacher: twice the substeps", ts["substeps x2"]["ratio"], TEACHER),
             ("teacher: stretch stiffness x0.9", ts["Y x0.9"]["ratio"], TEACHER),
             ("teacher: density x1.1", ts["w x1.1"]["ratio"], TEACHER),
             ("teacher: stretch stiffness x0.5", ts["Y x0.5"]["ratio"], TEACHER),
             ("teacher: bending stiffness x2", ts["B x2"]["ratio"], TEACHER),
             ("teacher: another episode's material", ts["swap"]["ratio"], TEACHER),
             ("network: material hidden", ctl["material"]["hidden"]["response"], NET),
             ("network: another episode's material", ctl["material"]["swap"]["response"], NET),
             ("network + true velocity: material hidden", orc["material"]["hidden"]["response"], NET_V),
             ("network + true velocity: another episode's", orc["material"]["swap"]["response"], NET_V)]
    errs = [("teacher from 3 frame positions", ts["back2"]["ratio"], ts["back2 + swap"]["ratio"], TEACHER),
            ("network, positions only", ctl["ratio"], ctl["material"]["swap"]["ratio"], NET),
            ("network + true velocity", orc["ratio"], orc["material"]["swap"]["ratio"], NET_V)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    y = list(range(len(moves)))[::-1]
    ax.barh(y, [m[1] for m in moves], color=[m[2] for m in moves], height=0.7)
    for yi, m in zip(y, moves):
        ax.text(m[1] + 0.015, yi, f"{m[1]:.2f}", va="center", fontsize=9, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in moves], fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("how far the predicted next frame moves / constant velocity's error", fontsize=9)
    ax.set_title("change the material: the teacher's next frame moves, the network's does not", fontsize=10)
    ax = axs[1]
    y = list(range(len(errs)))[::-1]
    for yi, (name, right, wrong, col) in zip(y, errs):
        ax.barh(yi + 0.18, right, height=0.34, color=col)
        ax.barh(yi - 0.18, wrong, height=0.34, color=col, alpha=0.4, hatch="///", edgecolor="white", linewidth=0)
        ax.text(right + 0.015, yi + 0.18, f"{right:.2f}  its own material", va="center", fontsize=8.5, color="#333")
        ax.text(wrong + 0.015, yi - 0.18, f"{wrong:.2f}  another episode's", va="center", fontsize=8.5, color="#333")
    ax.axvline(1.0, color="#666", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in errs], fontsize=9)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("one-step error / constant velocity's (lower is better)", fontsize=9)
    ax.set_title("error with the right and a wrong material", fontsize=10)
    for ax in axs:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Moving cloth in the air, 300 held-out windows (teacher: 64 episodes x 16 instants)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
