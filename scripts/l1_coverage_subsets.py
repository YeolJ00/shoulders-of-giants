"""The coverage study's training sets (claim 1, 2026-11-22): which episodes each arm trains on.

Claim 1: generalisation to new kinds of interaction follows the diversity of interactions seen in training, not the
number of trajectories. c1's episodes each run one to three gripper families (lift_drop, lift_place, drag, fold, shake,
push, slide; "held" -- the cloth in the gripper from the start -- is a start, not a family). Two families are held out
of every arm but the last: fold (cloth laid on itself) and shake (fast waves in the air). An episode belongs to an arm
when its families (passive plank slides, with none, are left out) are all in the arm's set. Every arm trains with the
same recipe and compute from scratch; only the episodes change:

    drag          {drag}                                         2,000 episodes
    place         {lift_place}                                   2,000
    drag_place    {drag, lift_place}                             2,000
    five          {drag, lift_place, lift_drop, push, slide}     2,000
    five_8k       the same five                                  8,000   (4x the trajectories, the same diversity)
    seven         all seven (fold and shake seen)                2,000   (the ceiling: what seeing them is worth)

Prediction (claim 1): on fold and shake, never seen by the first five arms, error falls from one family to two to five
at 2,000 episodes; 8,000 of the same five help less than going from one to five. Scored by scripts/l2_coverage_score.py.

    python3 scripts/l1_coverage_subsets.py      # writes outputs/lsim/coverage_subsets/<arm>.json
"""
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXPLODED_ABOVE = 200.0                                                  # lsim.train.cloth_windows.EXPLODED_ABOVE
FIVE = ("drag", "lift_place", "lift_drop", "push", "slide")
ARMS = {"drag": (("drag",), 2000), "place": (("lift_place",), 2000), "drag_place": (("drag", "lift_place"), 2000),
        "five": (FIVE, 2000), "five_8k": (FIVE, 8000), "seven": (FIVE + ("fold", "shake"), 2000)}


def families(wd):
    """An episode's gripper families: the actions it runs (not "held", the start; not a family skipped for lack of time)."""
    return frozenset(f["family"] for f in wd["families"] if f["family"] != "held" and not f.get("skipped"))


def episodes(data):
    """(shard stem, world, families) of every usable episode of a dataset (cloth_windows.ClothEpisode.ok's test)."""
    out = []
    for p in sorted(glob.glob(str(REPO / f"data/datasets/{data}/shard_*.json"))):
        for w, wd in enumerate(json.loads(Path(p).read_text())["worlds"]):
            q = wd.get("qc", {})
            if q and not q.get("exploded", False) and max(q.get("contact_over_weight", [0.0])) < EXPLODED_ABOVE:
                out.append((Path(p).stem, w, families(wd)))
    return out


def main():
    from lsim.train.cloth_windows import EXPLODED_ABOVE as E
    assert E == EXPLODED_ABOVE, "the quality threshold changed"
    eps = episodes("c1")
    out = REPO / "outputs/lsim/coverage_subsets"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026_11_22)
    for arm, (fams, n) in ARMS.items():
        pool = [(s, w) for s, w, f in eps if f and f <= set(fams)]
        pick = [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]
        ps = set(pick)
        count = {f: sum(f in fs for (s, w, fs) in eps if (s, w) in ps) for f in fams}
        (out / f"{arm}.json").write_text(json.dumps({"families": list(fams), "n": n, "pool": len(pool), "episodes": pick}))
        print(f"{arm:11s} {n:5d} of {len(pool):5d} eligible; episodes running each family: {count}")


if __name__ == "__main__":
    main()
