"""Plot the V0 calibration. Runs in the WARP env -- matplotlib is not in `cosmos`,
and rule 5 keeps the scoring env pinned to what the model needs.

    /home/jooyeolyun/anaconda3/envs/warp/bin/python scripts/v0_report.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path("docs/results/v0_surprise.json")
FIG = Path("docs/results/v0_speed_dependence.png")
COL = {"archery": "#b4451f", "bowling": "#2f6f8f"}


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} not found -- run scripts/v0_surprise.py first")
    r = json.loads(SRC.read_text())["results"]
    sweep, rho = r["motion_sweep"], r["motion_rho"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.3),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # left: within each clip, discrimination falls as the clip is slowed
    ax.axhspan(-6, 0, color="0.93", zorder=0)
    ax.axhline(0, color="0.4", lw=0.8)
    for name in COL:
        rows = [x for x in sweep if x["clip"] == name]
        ax.plot([x["motion"] for x in rows], [x["shuffled_pct"] for x in rows],
                "o-", color=COL[name], ms=4,
                label=f"{name}  (rho = {rho['per_clip'][name]:+.2f})")
    for p, mo in r["our_clips"].items():
        ax.axvline(mo, color="#8a8a8a", ls=":", lw=1)
        ax.text(mo, ax.get_ylim()[1], " " + Path(p).stem[:20], rotation=90,
                va="top", ha="left", fontsize=6.5, color="#666")
    ax.set_xscale("log")
    ax.set_xlabel("clip motion   (mean |frame difference|, 0-255)")
    ax.set_ylabel("frame-shuffled vs real   (%)")
    ax.set_title("Slowing a clip destroys its own discrimination\n"
                 "same content, stretched over 64 frames; dotted lines are our renders",
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # right: the same points pooled -- the across-content confound, in one picture
    bx.axhline(0, color="0.4", lw=0.8)
    for name in COL:
        rows = [x for x in sweep if x["clip"] == name]
        bx.scatter([x["motion"] for x in rows], [x["shuffled_pct"] for x in rows],
                   s=22, color=COL[name], label=name)
    bx.set_xscale("log")
    bx.set_xlabel("clip motion")
    bx.set_title(f"Pooled across clips: rho = {rho['pooled']:+.2f}\n"
                 "within each clip it is +0.86 and +0.74", fontsize=9.5)
    bx.legend(fontsize=8, frameon=False)

    fig.suptitle("V0 -- what the surprise signal can and cannot see", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
