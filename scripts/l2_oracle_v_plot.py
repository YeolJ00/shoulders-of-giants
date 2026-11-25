"""The oracle-velocity test as one figure (2026-11-19): one-step error / constant velocity's on moving free cloth and on
cloth lying on a surface, for the teacher and the networks.

    $L scripts/l2_oracle_v_plot.py --out <png> [--oracle oracle_v_7999 --control ck1_on_c1v]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def paired(a, b, cls, n_boot=10000, seed=0):
    """Bootstrap over windows of ratio(a) - ratio(b) on one class (both scored on the same windows): (diff, lo, hi)."""
    import numpy as np
    A, B = np.asarray(a["per_window"][cls], float), np.asarray(b["per_window"][cls], float)
    assert len(A) == len(B) and np.array_equal(A[:, 2], B[:, 2]), "not the same windows"
    rng = np.random.default_rng(seed)
    r = lambda X, i: X[i, 0].sum() / X[i, 1].sum()
    i0 = np.arange(len(A))
    d = [r(A, i) - r(B, i) for i in (rng.integers(0, len(A), len(A)) for _ in range(n_boot))]
    return r(A, i0) - r(B, i0), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle", default="oracle_v_7999")
    ap.add_argument("--control", default="ck1_on_c1v")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = REPO / "docs/results"
    tf = json.load(open(R / "l1_teacher_from_inputs.json"))["classes"]
    ctl = json.load(open(R / f"l2_error_where_{a.control}.json"))
    orc = json.load(open(R / f"l2_error_where_{a.oracle}.json"))
    rows = [("constant velocity", 1.0, 1.0, "#9aa0a6"),
            ("network, positions only", ctl["free"]["ratio"], ctl["table"]["ratio"], "#4c78a8"),
            ("network + true velocity", orc["free"]["ratio"], orc["table"]["ratio"], "#e45756"),
            ("x + true velocity x dt (no network)", ctl["free"]["true_cv_ratio"], ctl["table"]["true_cv_ratio"], "#c7c7c7"),
            ("teacher from 3 frame positions", tf["off"]["back2"]["ratio"], tf["surface"]["back2"]["ratio"], "#54a24b"),
            ("teacher from its full state", tf["off"]["true"]["ratio"], tf["surface"]["true"]["ratio"], "#2f6f2a")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, k, title in ((axs[0], 1, "moving cloth in the air (81% of the error)"), (axs[1], 2, "moving cloth on a surface")):
        y = range(len(rows))[::-1]
        ax.barh(list(y), [r[k] for r in rows], color=[r[3] for r in rows])
        for yi, r in zip(y, rows):
            ax.text(r[k] + 0.02, yi, f"{r[k]:.2f}", va="center", fontsize=9)
        ax.axvline(1.0, color="#666", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(1.3, max(r[k] for r in rows) + 0.15))
        ax.set_xlabel("one-step error / constant velocity's (lower is better)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_yticks(list(range(len(rows)))[::-1])
    axs[0].set_yticklabels([r[0] for r in rows])
    title = "Handing the network the teacher's true velocity (same network and recipe, 8,000 steps)"
    if "per_window" in ctl and "per_window" in orc:
        for cls in ("free", "table"):
            d, lo, hi = paired(orc, ctl, cls)
            print(f"{cls}: true velocity - positions only = {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (paired bootstrap over windows)")
        d, lo, hi = paired(orc, ctl, "free")
        title += f"\nin the air: true velocity minus positions only {d:+.2f} [95%: {lo:+.2f}, {hi:+.2f}]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
