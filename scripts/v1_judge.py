"""The judge baseline, on the SAME clips V1 scores with surprise.

CLAUDE.md item 1 asked for a one-bit threshold question. This is the stronger form of
it: identical data, identical framing, two supervision signals side by side, so the
comparison cannot be blamed on the clips. No new renders -- judge forwards only.

Run (cosmos env, one polled GPU):
    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/v1_judge.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization")

SRC = Path("docs/results/v1_surprise.json")
OUT = Path("docs/results/v1_judge.json")

# One-bit questions, one per probe: each is a threshold the simulator already knows the
# answer to, which is the only kind of question the judge has ever answered reliably.
QUESTION = {
    "bounce": "Does the bowl bounce back up off the table after it lands?",
    "settle": "Does the bowl rock back and forth several times before coming to rest?",
    "tilt": "Does the bowl slide down the tilting table?",
    "null": "Does the bowl slide down the tilting table?",
    "violation": "Does the bowl move in a physically possible way?",
}


def main():
    from src.judge.cosmos import CosmosJudge

    rows = json.loads(SRC.read_text())["rows"]
    rows = [r for r in rows if r["probe"] != "floor"]
    j = CosmosJudge(device="cuda:0")
    out = []
    for r in rows:
        q = QUESTION.get(r["probe"], QUESTION["violation"])
        margin, lp_yes, lp_no = j.score_one(r["clip"], q)
        out.append({**{k: r[k] for k in ("key", "probe", "param", "value", "ref", "clip")},
                    "question": q, "margin": float(margin),
                    "p_yes": float(__import__("math").exp(lp_yes))})
        print(f"  {r['key']:<24} margin {margin:+7.3f}  p(yes) {out[-1]['p_yes']:.3f}",
              flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}  ({len(out)} clips)")


if __name__ == "__main__":
    main()
