# Surprise (V-JEPA 2) — the V-track

Working notes. The user-facing writeup is `docs/V1_SURPRISE.html`; the numbers live in
`docs/results/v0_surprise.json`, `v1_surprise.json`, `v1_verdict.json`.

## What it is

```
surprise = L1( predictor(context tokens) , layer_norm(encoder(target tokens)) )
```

V-JEPA 2's own pretraining objective, run **causally**: context = every token before
temporal slot `s`, target = slot `s` (2 frames), slid across the clip. The mean over
slots is the reading; the per-slot curve says *where* the clip became unpredictable.

Nothing is generated and nothing is prompted. The simulator makes every motion; the
model only reacts.

## Environment

Scoring runs in the **cosmos** env (`torch 2.11.0+cu128`, `transformers 5.14.1`) —
`VJEPA2Model` is already there, no install was needed. The **warp** env has no torch at
all, so the pipeline keeps the existing split: warp = sim + plots, blender = render,
cosmos = scoring. Weights are `facebook/vjepa2-vitl-fpc64-256`, fetched to `$HF_HOME`.

## Architecture facts the code depends on

All read out of `transformers/models/vjepa2/modeling_vjepa2.py` and re-asserted at
runtime by `scripts/v0_surprise.py`:

- Patch embedding is `Conv3d(...).flatten(2).transpose(1,2)`, so token index `n` decodes
  as `t = n // 256`, `h = (n % 256) // 16`, `w = n % 16`. `VJEPA2RopeAttention._get_frame_pos`
  uses exactly that arithmetic.
- `context_mask` / `target_mask` are **index** tensors, not booleans. `VJEPA2Predictor.forward`
  gathers encoder states at the context indices, appends learned mask tokens at the target
  indices, and derives its RoPE positions from those same indices.
- **64 frames is not optional.** `grid_depth = frames_per_clip // tubelet_size` is baked
  into the position grid; a different frame count makes every position id silently wrong.
- The predictor consumes the encoder's **last** hidden state. CLAUDE.md's "middle encoder
  layers" note therefore cannot apply to the predictor target — feeding it a layer it never
  saw is off-distribution. `layer` is kept only for a secondary, predictor-free variant.

## Three things that were wrong first

1. **The target must be layer-normed.** V-JEPA's loss is `smooth_l1(z, F.layer_norm(h))`,
   no affine; `VJEPA2Model` returns the target raw. Raw encoder output has `mean|·| = 2.15`
   against a predictor emitting `0.32`, so the un-normalised L1 is a large constant plus a
   rounding error: a **static** clip scored as *more* surprising than a real one and every
   variant sat within 2.4% of 2.03.
2. **Sliding causal prefix, not one long jump.** Context = first 16 frames, target = the
   other 48 separates real from frame-shuffled by **+0.12%**. Sliding a 2-frame horizon
   gives **+2.54%**; the pretrained spatial-block regime gives +1.49%.
3. **A motion "floor" inferred from one clip did not survive a second one.** See below.

## V0 results (7/7 gates)

| gate | result |
|---|---|
| determinism | same clip twice, bit-identical |
| two routes | `encode()+predict()` == `VJEPA2Model.forward`, `max abs delta = 0` |
| token order | exactly the predicted ids move, temporal and spatial |
| beats zero-prediction | 0.578 vs 0.665 — 13.1% better |
| shuffled > real | on both reference clips |
| speed dependence | per-clip rho **+0.95 / +0.83** |

**The finding that shaped V1.** Stretch one natural clip over more frames so only its
speed changes, and its discrimination collapses monotonically and eventually inverts.
But the absolute threshold does **not** transfer: one reference still discriminates at
motion 2.0 where the other has collapsed. Pooled across the two clips that correlation
is **-0.01** — the across-content confound the probe table already warns about,
reproduced in the new signal. So clip motion is a render design target compared *within*
one scene, never a pass/fail bar applied across scenes.

Every clip in `outputs/judge/` sits at motion 0.4–2.3; natural video is ~17.

## V1 design

`scripts/v1_killtest.py`, staged `sim` → render → `clip` → `score`, then
`scripts/v1_report.py`. One object (`rigid/wooden_bowl_01`), fixed appearance, one
parameter at a time.

| axis | probe | sweep |
|---|---|---|
| damping | bounce (8 cm centre drop) | cd, 9 points, 400–12800 |
| damping | settle (tipped onto its dome) | cd, 9 points |
| friction | tilt (continuous ramp 0→30°) | mu, 9 points, 0.10–0.50 |
| **mass, null control** | tilt | mass, 5 points, 32× range |
| **gross violations** | authored on the bounce reference | teleport, through, vanish, freeze, rewind |
| **noise floor** | bounce reference re-rendered | 5 Cycles seeds + 5 camera jitters |

Render choices, all measured rather than guessed:

- `CAM_PULL = 0.65` — the closest camera that still contains the whole drop; at 0.45 only
  91.5% of the object's projected points stay in frame.
- `BOUNCE_Z = 0.08` m, not 0.30 — at 30 cm the fall dominates the crop box and the clip
  reads motion 0.5–1.3; at 8 cm the same sweep reads 1.5–4.1.
- **One crop box per probe**, taken from that probe's reference θ, and violations and
  floor clips borrow the bounce box. A per-clip box would grow with whatever the object
  did, so surprise would be reading the framing — which varies with θ.
- Zooming does *not* help on its own: cropping to the trajectory normalises the zoom away.
  Motion in the crop is set by object size relative to **trajectory extent**.

## The caveat to carry into the verdict

Motion itself varies with cd (4.05 → 1.54 across the sweep) and V0 showed surprise tracks
motion within a scene. A monotone surprise-vs-θ curve may therefore be reading *how much
moved* rather than *how implausible it was*. The two are separated by **shape**: a
plausibility signal has a minimum inside the sweep; a motion-mediated artefact is monotone
and tracks `rho(surprise, motion)`, which the report prints.
