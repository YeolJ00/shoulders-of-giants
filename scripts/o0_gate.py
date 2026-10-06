"""G-0 -- does the SSv2 head respond to hand-free synthetic renders at all?

The cheapest decisive test in the O-track, and it runs before anything is built: no new
renders, no dataset, no new environment. ~144 clips are already on disk from the V-track.

WHY THIS GATE EXISTS. SSv2 templates are overwhelmingly hand-performed -- "poking
something so it falls over", "tilting something with something on it until it falls off".
Our scenes contain no hand, no pusher and no actuator of any kind: excitation is initial
velocity kicks and gravity ramps, so the object moves on its own. A head trained on that
footage has very likely learned hand presence as a class cue. If it returns near-uniform
output on agentless tabletop renders, the O-track's premise fails here for about an hour of
compute, rather than after the render budget is spent.

PASS   median top-1 probability >= 0.15 (~26x uniform, since uniform is 1/174 = 0.0057)
       AND the top-5 of a rolling/sliding clip contains a motion-appropriate template.
FAIL   near-uniform output (median top-1 < 0.05) => the head does not accept our renders.
       Recovery is a hand/pusher proxy, or the track stops and is written up.

Run (cosmos env, one polled GPU, released on exit):
    G=$(warp-python src/gpu.py | tail -1 | sed 's/.*-> //')
    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/o0_gate.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.outcome.head import OutcomeHead                              # noqa: E402

OUT = Path("docs/results/o0_gate.json")
UNIFORM = 1.0 / 174

# Templates whose OUTCOME phase involves no hand. If the head is reading physics rather
# than an actor, these are the ones our clips could plausibly land in. Chosen from the
# label list before looking at any output.
MOTION_WORDS = ("rolling", "roll", "falling", "fall", "tipping", "tips", "spinning",
                "sliding", "slide", "collides", "bouncing", "knocking", "tilting",
                "dropping", "moves", "stops")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labs", default="outputs/scene/v1,outputs/scene/v1b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    clips = []
    for lab in args.labs.split(","):
        clips += sorted(Path(lab).glob("clips/*.mp4"))
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        sys.exit("no clips found -- expected outputs/scene/v1{,b}/clips/*.mp4")

    m = OutcomeHead(device=args.device)
    print(f"{m.model_id}\n{json.dumps(m.config())}\n{len(clips)} clips", flush=True)

    # determinism first (rule 7) -- a score that is not reproducible is not a measurement
    a = m.score_clip(clips[0], use_cache=False)
    b = m.score_clip(clips[0], use_cache=False)
    det = bool(np.array_equal(a, b))
    print(f"determinism: {'PASS' if det else 'FAIL'}  "
          f"top-1 {a.max():.9f} vs {b.max():.9f}\n", flush=True)

    rows, tops = [], Counter()
    for i, c in enumerate(clips):
        p = m.score_clip(c)
        t5 = m.top(p, 5)
        rows.append({"clip": str(c), "top1_p": float(p.max()),
                     "top1": t5[0][0], "entropy": float(-(p * np.log(p + 1e-12)).sum()),
                     "top5": [[n, v] for n, v in t5]})
        tops[t5[0][0]] += 1
        if i < 6 or i % 40 == 0:
            print(f"  {Path(c).stem:<26} p={p.max():.3f}  {t5[0][0][:58]}", flush=True)

    t1 = np.array([r["top1_p"] for r in rows])
    ent = np.array([r["entropy"] for r in rows])
    med = float(np.median(t1))
    # is any motion-appropriate template in the top-5 anywhere?
    hits = sum(any(any(w in n.lower() for w in MOTION_WORDS) for n, _ in r["top5"])
               for r in rows)

    print(f"\n{'='*64}\nmedian top-1 probability : {med:.4f}   (uniform = {UNIFORM:.4f}, "
          f"{med/UNIFORM:.0f}x)")
    print(f"top-1 range              : {t1.min():.4f} .. {t1.max():.4f}")
    print(f"median entropy           : {np.median(ent):.3f} nats "
          f"(uniform = {np.log(174):.3f})")
    print(f"clips with a motion word in top-5: {hits}/{len(rows)} = {100*hits/len(rows):.0f}%")
    print(f"\nmost frequent top-1 classes:")
    for n, k in tops.most_common(6):
        print(f"   {k:>4}x  {n}")

    passed = det and med >= 0.15 and hits > 0
    verdict = "PASS" if passed else ("FAIL" if med < 0.05 else "MARGINAL")
    print(f"\nG-0 {verdict}  (pass: median top-1 >= 0.15 and >0 motion-word hits; "
          f"fail: median < 0.05)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": m.model_id, "config": m.config(),
                               "deterministic": det, "median_top1": med,
                               "median_entropy": float(np.median(ent)),
                               "motion_word_hits": hits, "n": len(rows),
                               "verdict": verdict,
                               "top1_counts": tops.most_common(20),
                               "rows": rows}, indent=1))
    print(f"wrote {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
