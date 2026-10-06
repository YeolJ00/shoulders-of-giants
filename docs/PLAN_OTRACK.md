# O-track — outcome-classifier supervision with multi-view soft targets

## Context

Three supervision signals have been measured and retired for optimisation:

| signal | verdict |
|---|---|
| video model as **witness** (fit to generated video) | hallucination enters θ; abandoned by design |
| chat VLM as **judge** (Cosmos 3 yes/no logits) | ρ ≈ 0 with density over 10× sweeps; wrong 9/9 on a one-bit question the simulator knows the answer to |
| V-JEPA **surprise** (prediction error) | **positive control at chance** — see below |

The simulator side is validated: exact mesh contact, per-body materials, and six probes that
pass a parameter-dependence sweep (tilt 0.40° residual, balance 7/7, bounce ρ −0.995,
settle ρ +0.973, spin ρ −0.952, stack ρ +0.994). **We can manufacture excellent evidence
and cannot read it off a video.** That is still the gap.

**Correcting the brief's account of the surprise result.** There is no "V0.5" stage in this
repo and nothing named a positive control. What served that role was the authored-violation
tier of V1 — teleport, pass-through, vanish, freeze, time-reversal — and after the two
defects were fixed (a bowl spawning 11.8 mm inside the table, and a context leak in which
context tokens were gathered from a full-clip encode and had therefore already attended to
the region being predicted), the corrected runs give:

- **47%** (wooden bowl), **47%** (brass pot, held out), **40%** (V-JEPA 2.1 ViT-g) on 30
  matched pairs. Chance.

So the accurate statement is stronger than the brief's: surprise is at chance on
**categorically impossible** motion in our renders, not merely on wrong-but-legal θ. For
wrong-but-legal θ we never established chance because there is no ground truth to score
against. This is consistent with IntPhys 2, which reports *"most models perform at chance
levels (50%)"* on realistic scenes — the 98%/96.7% figures belong to IntPhys **1**, coloured
shapes on plain backgrounds.

One structural finding from that work carries directly into this one and shapes the design:
the signal was **reproducible across models (ρ = +0.891) and not across objects
(ρ = +0.104)**. That is the third reproduction of this project's across-object confound.
**Every O-track comparison must therefore be within-object.**

`CLAUDE.md` still records V1 as "RETRACTED AND RE-RUNNING"; updating it with the final
corrected verdict is a task in O0.

## Corrections to the brief, verified against the artifacts

1. **The SSv2 number is 77.3%, not 77.7%, and it belongs to V-JEPA 2 ViT-g/16₃₈₄ — not 2.1
   ViT-G.** Meta's README: ViT-L/16 → 73.7% (`ssv2-vitl-16x2x3.pt`), ViT-g/16₃₈₄ → 77.3%
   (`ssv2-vitg-384-64x2x3.pt`).
2. **No V-JEPA 2.1 SSv2 head exists.** `configs/eval_2_1/vitG-384/ssv2.yaml` is shipped but
   no trained probe is. Training one per that config needs 16 nodes × 8 GPUs, 20 epochs, a
   20-point lr×wd grid. Out of scope, and unnecessary — see 3.
3. **The V-JEPA 2 head is released and one call away.** Both checkpoints return HTTP 200
   (374 MB / 198 MB), and HuggingFace hosts `facebook/vjepa2-vitg-fpc64-384-ssv2` as
   `VJEPA2ForVideoClassification`, 174 labels, 64 frames, 384 crop, hidden 1408. **The
   "train the head" milestone is deleted.**
4. **There is no differentiable renderer.** `src/render/diff_preprocess.py` is a
   preprocessing bridge — its own docstring says *"This is not a differentiable RENDERER"*.
   `diff_splat` / `diff_soft` / `diff_collide*` are differentiable **physics** producing
   trajectories, not pixels. nvdiffrast is not installed. `warp.render` exposes exactly
   `OpenGLRenderer` and `UsdRenderer`, neither of which touches the autodiff graph.
   **The brief's gradient path cannot be run today.**
5. **Our scenes contain no hand, pusher or actuator.** Excitation is initial velocity kicks
   and gravity ramps. SSv2 templates are overwhelmingly hand-performed, so the head has very
   likely learned hand presence as a class cue. This is a candidate blocker distinct from
   the general render gap, and G-0 exists to settle it before anything is built.

## Decisions taken

- **Head:** `facebook/vjepa2-vitg-fpc64-384-ssv2`. `frames_per_clip = 64` matches our
  existing clip length exactly; 384 → 24×24 patches.
- **Hand problem:** test empirically first (G-0) on clips we already have, then choose the
  class vocabulary from the evidence.
- **G-C:** skipped. Take 73.7 / 77.3% as established; do not re-verify a released
  checkpoint against its own published number using licence-gated video.
- **Optimiser:** SPSA/CEM on Cycles renders. **No differentiable renderer is built now.**
  nvdiffrast was considered and rejected: it is a rasterizer, so its frames do not look like
  Cycles, and the classifier accepting our renders is the whole premise. If gradients become
  worth having after G-A, the choice is **Mitsuba 3** — the only differentiable renderer that
  is also physically based.
- **Renderer-agnostic via USD:** the rollout is exported to USD so the renderer is a swappable
  back end rather than a commitment. See below.

### Why Warp alone does not give us this

Warp *is* differentiable, and θ → trajectory gradients work today: `diff_collide_6dof_recover.py`
does momentum check → tape-grad vs finite difference → gradient-descent recovery of density
on a real mesh asset. The broken link is elsewhere:

```
θ ──Warp sim──▶ trajectory ──RENDER──▶ pixels ──V-JEPA──▶ logits
   differentiable            NOT        differentiable
```

Our images come from **Blender Cycles**, a path tracer with no gradients, and Warp is not in
that link at all — `warp.render` exposes only `OpenGLRenderer` and `UsdRenderer`, neither of
which touches the autodiff graph. nvdiffrast fills exactly this one arrow.

## The loop

```
θ → scripted probe (Warp) → render N views (Cycles) → frozen V-JEPA 2 ViT-g
  → SSv2 attentive probe → 174-way distribution, averaged over views
  → cross-entropy vs a-priori target  +  λ · equilibrium residual → SPSA step
```

Nothing is generated; the model only names the outcome of a motion we made. No sampling
anywhere, so the forward path is deterministic and cacheable.

## Class vocabulary

Rule from the brief, adopted: **the simulator reads binary outcomes exactly** (did it topple
— a sign test); the **classifier reads manner and degree** ("slightly", "quickly"), where the
boundary is a real-world calibration the simulator cannot know. Never route a sign through a
network.

| probe | existing code | candidate SSv2 templates | constrains | hand in outcome phase? |
|---|---|---|---|---|
| roll / slide down slope | `probe_spin_stack.py`, `bigscene_sim.py` tilt ramp | *letting something roll down a slanted surface*; *something falling like a rock / like a feather* | μ, restitution | **no** — safest |
| spin | `probe_spin_stack.py::spin(mu)` → `tau_s` | *spinning something that quickly stops spinning* / *…so it continues spinning* | μ, damping | no |
| drop | `probe_bounce_settle.py::bounce(cd)` | *something falling like a rock / like a feather*; *dropping something onto something* | density, restitution | no |
| tilt-with-object-on-top | `probe_spin_stack.py::stack(mu_pair)` → `collapse_deg` | *tilting something with something on it until it falls off* / *…but it doesn't fall off* | μ between objects, CoM | **yes** — a hand tilts |
| poke | **does not exist**; `shove` in `joint_sim.py` is unverified and reads stopping distance | *poking something so it falls over* / *…so it slightly moves* / *…but it doesn't move* | mass, CoM | **yes** |

Vocabulary is selected **after G-0**, weighted toward the hand-free rows. `poke` requires new
scene code and is deferred to O3 at the earliest.

## Multi-view measurement

`src/render/views.py` is a plain dict of camera configs consumed by
`scripts/blender_render_scene.py` (loops `VIEWS.items()`, one output dir per
`sim_{key}@{view}`). Adding views is a dict entry; **the rollout is computed once and only
pixels are re-rendered**, so N views multiply render cost only, never sim cost.

- **N = 4** to start: the existing `a`/`b`/`c` plus one opposing azimuth. Fixed cameras,
  identical excitation across views and across θ, fixed target-material appearance (rule 4).
- **Measurement** = class distribution averaged over views, matching the head's own
  `64x2x3` multi-clip logit-averaging protocol.
- **Cross-view agreement** is a free render-gap diagnostic: a real outcome is view-invariant,
  so disagreement means the classifier is reacting to rendering. Clips below the agreement
  threshold are down-weighted to zero rather than discarded, so the count is reported.

## Targets, decided a priori

- **Primary — mine SSv2 itself.** `validation.json` (3.7 MB) and `train.json` (26 MB) carry
  per-video `template` + `placeholders` (object nouns). Group by noun × template to get
  per-object-category outcome distributions ("cups, poked: p(slightly moves), p(falls
  over)"). The 174-template list is freely available from
  `huggingface/label-files/something-something-v2-id2label.json` (10 KB, non-gated, and what
  Meta's own demo uses). **Licensing caveat:** the per-video annotations are on unofficial
  community mirrors of Qualcomm-gated data. Flagged for your decision at O2, not needed
  before it.
- **Fallback** — identity-based: material/object type → outcome distribution, asserted.
- **Always** — **equilibrium residual** on the authored rest pose: the asset must not move
  when released at its authored pose. Target zero, no model involved, and the one term in
  the loss that cannot be hacked by any classifier.

Targets are **categorical**. They yield feasible *sets*, not point estimates. Precision comes
from intersecting probes plus equilibrium, never from a single label.

## Loss and optimisation

```
L(θ) = Σ_p CE( mean_views softmax(head(render(sim(θ, p)))) , target_p )  +  λ · ‖equilibrium residual‖²
```

- θ in log-space with prior bounds, per object — the fit already carries `[μ, cd, mass]`
  per object (`scripts/joint_fit.py`, `PROBE_MASK` gates which probes move which component).
- **SPSA** (2 evaluations per step regardless of dimension) reusing `scripts/spsa_fit.py`
  and `scripts/joint_fit.py`; CEM as the cross-check.
- Classifier probabilities are treated as **ordered, not calibrated** — all conclusions rest
  on ranking across θ, never on an absolute probability.

## Renderers, and what is actually differentiable

Within NVIDIA's stack, "differentiable **and** realistic" does not exist — there is no
NVIDIA differentiable path tracer. Pick two of three:

| renderer | differentiable | realistic | NVIDIA stack |
|---|---|---|---|
| Omniverse RTX (Isaac Sim) | no | yes | yes |
| nvdiffrast | yes | **no** — rasterizer, no GI or soft shadows | yes |
| Cycles (today) | no — zero gradient properties on the API | yes | no |
| **Mitsuba 3** (3.9.1, pip) | **yes** — differentiable path tracer | **yes** | no |

Cycles is a Monte Carlo path tracer: sampling is stochastic and visibility is discontinuous
(a ray hits a triangle or it does not), which is the actual obstacle — the derivative at a
silhouette does not exist in the naive formulation. Mitsuba 3 solves this properly via edge
sampling; that is what makes it the eventual gradient option rather than a rasterizer.

**USD is the pivot, and it is what makes this fit the Isaac ecosystem.** Newton is the
direction Isaac Lab's physics is going, `pxr` is *already installed* in the `warp` env,
`warp.render.UsdRenderer` exists, and Blender imports and exports USD (verified). So:

```
Newton rollout ──▶ USD ──▶ { Cycles (now) · Omniverse RTX (later) · Mitsuba 3 (if gradients) }
```

The renderer stops being a commitment and becomes a back end. Cost is small and it is paid
once. **No new conda env is required** — the earlier `otrack` proposal is dropped along with
nvdiffrast, so `warp` (sim + USD) and `cosmos` (scoring) stay untouched and validated.

## Gates — kill-first, nothing downstream counts

**G-0 · does the head fire on hand-free synthetic renders at all?** *(new, first, cheapest)*
Run the head over the ~144 clips already on disk (`outputs/scene/v1/clips`,
`outputs/scene/v1b/clips`). No new renders, no dataset, ~1 hour.
- **Pass:** median top-1 probability ≥ 0.15 (≈26× uniform) **and** the top-5 for a rolling
  clip contains at least one motion-appropriate template.
- **Fail:** near-uniform output (top-1 < 0.05) ⇒ the head does not accept agentless
  tabletop renders. Recovery path: hand/pusher proxy, or the track stops here. **This is the
  cheapest decisive test available and it runs before anything is built.**

**G-A · does the decision boundary sit where the physics threshold sits?** *(the real kill
test)* Render ground-truth sweeps — tilt at known μ, spin at known damping, drop at known
restitution — and check the class flip tracks the simulator's own threshold.
- **Pass:** for ≥2 probes, |log θ_boundary(classifier) − log θ_threshold(sim)| ≤ 0.35
  (a factor of 1.4), and the flip is monotone in θ.
- **Fail:** boundary does not track threshold ⇒ recovery is a **probe-head-only** fine-tune
  on our labelled renders (encoder stays frozen — the brief's constraint and rule 5's).

**G-B · view agreement.** On known-outcome clips, cross-view top-1 agreement must be high
and must not depend on θ.
- **Pass:** ≥ 3 of 4 views agree on top-1 for ≥ 80% of clips, and Spearman ρ(θ, agreement
  rate) with |ρ| ≤ 0.3.
- **Fail:** agreement varies with θ ⇒ the measurement is reading render artefacts, and the
  view-average is not a valid soft target.

**G-R · render parity** — deferred with the gradient path. If a second renderer is ever
introduced (Omniverse RTX or Mitsuba 3), the same rollouts go through both and the head's
top-1 must agree on ≥ 80% of clips with ρ ≥ 0.7 between per-clip class probabilities, or the
gates were calibrated on a domain the optimiser never sees.

**G-C · positive control on real SSv2** — **skipped by decision.** Consequence recorded: if
G-A fails we cannot separate "head is broken" from "render gap". Mitigated by G-0, which
uses the same head on our own footage.

## Milestones, definition of done, cost

Measured render cost: ~1.6 s/frame per Blender worker at 544×448/48 samples; ~6 workers
saturate one card (`SHARD=k/N` already implemented), giving ~0.27 s/frame aggregate.
Classifier: ~2 s per 64-frame clip on ViT-g/384.

| | done when | cost |
|---|---|---|
| **O0** | head loads, deterministic (same clip twice ⇒ identical logits), id2label verified against config, **G-0 reported**; CLAUDE.md V1 verdict updated | ~1 h, no renders |
| **O1** | 4-view render of 2 existing probe scenes; **G-B** reported | 4 views × 2 probes × 64 frames ≈ 512 frames ≈ 3 min render |
| **O2** | ground-truth sweeps rendered and scored; **G-A** reported per probe | 3 probes × 9 θ × 4 views × 64 frames ≈ 6,900 frames ≈ 31 min render + ~14 min classify |
| **O3** | single-object, single-probe fit vs ground truth and vs the CEM/SPSA baselines | ~10 min per SPSA iteration (4 views × 4 probes × 2 candidates × 64 frames ≈ 2,048 frames); 50 iterations ≈ 8 h |
| **O4** | joint multi-probe fit with mined targets + equilibrium; scatter report | 2–3× O3 |

**Caching** is the difference between O3 being 8 hours and being a week: key on
`sha1(θ, probe, view, seed)` for renders — generalising `src/render/clip.py::theta_key`,
which already hashes sorted-key JSON — and on `sha256(clip bytes) + head config` for
classifier calls, reusing the `src/surprise/cache.py` pattern. SPSA revisits nearby θ
constantly.

## Risks

- **The hand gap** — G-0, first, before anything else.
- **Render domain gap** — smartphone footage vs clean tabletop. G-A is the measurement;
  probe-head fine-tune is the recovery.
- **Class granularity too coarse for fine θ.** Mitigation is structural: categorical targets
  give feasible sets, and precision comes from intersecting probes with the equilibrium
  residual. If a probe's class flip is coarser than the prior, that probe adds nothing and
  should be dropped rather than tuned.
- **Classifier hacking.** Bounded by the physical bottleneck: θ is 3–5 numbers and every
  candidate is a real Warp rollout, so there is no pixel-space adversarial direction to
  exploit. Probe stacking plus equilibrium is the defence.
- **Across-object confound.** Mined targets are category-level; our comparisons are
  within-object. Never compare a score between objects — the previous track produced
  ρ = +0.891 within and +0.104 across.
- **Frame/stride mismatch** — head wants 64 frames; our clips are 64. Matched, but any probe
  needing a longer window must be resampled, not extended.
- **VRAM / two envs.** `warp` has no torch; `cosmos` has no Warp. The pipeline stays staged
  (sim → render → cache clips → separate scoring pass), which is what this repo already does
  everywhere. A single-process Warp-tape-into-torch loop is not available without env
  surgery, and is only needed if the gradient path is later built.

**Noted, not planned:** extracting *numeric* outcome statistics from real SSv2 footage with
our tracker (slope angles, bounce counts, spin-down times). Real video, so tracker error is
noise rather than hallucination. Revisit only after O2.

## Files

| path | change |
|---|---|
| `src/outcome/head.py` | new — load `vjepa2-vitg-fpc64-384-ssv2`, deterministic forward, 174-way distribution, view averaging, cache. Mirrors `src/surprise/` conventions (module constants, `device=` arg, guard-once check, imageio decode). |
| `src/outcome/vocab.py` | new — probe → template-id map, chosen after G-0 |
| `src/render/usd_export.py` | new — Newton rollout → USD via `warp.render.UsdRenderer` (`pxr` already installed), so the renderer is swappable |
| `scripts/o0_gate.py` | new — G-0 on existing clips |
| `scripts/o1_views.py` | new — N-view render prep + G-B |
| `scripts/o2_boundary.py` | new — ground-truth sweeps + G-A |
| `scripts/o3_fit.py` | new — SPSA on the outcome loss |
| `src/render/views.py` | add the 4th camera |
| `scripts/blender_render_scene.py` | unchanged — `VIEWS`, `SHARD`, `SEED`, `CAM_JITTER` already sufficient |
| `scripts/joint_fit.py`, `scripts/spsa_fit.py` | reuse the optimiser; swap the score term |
| `CLAUDE.md` | record the final V1 verdict (currently says "RETRACTED AND RE-RUNNING") and the O-track rules |

## Verification

1. **Determinism first** — same clip twice ⇒ identical logits; fp32; TF32 off. Rule 7.
2. **G-0 on existing clips** — no new renders. If it fails, stop and report.
3. **G-B before G-A** — a view-average that varies with θ invalidates every later
   measurement, so it is checked before the boundary test is interpreted.
4. **G-A per probe**, with the simulator's own threshold as the reference, reported as a
   plot in `docs/results/` with the sweep drawn on.
5. **O3 against two baselines** — ground truth where known (balance gives 0.182 kg) and the
   existing CEM/SPSA fits on the same probe.
6. Any change of head, resolution, frame count or view set ⇒ **rerun G-A and G-B** (rule 5).

## Decisions still needed from you

1. **SSv2 annotation licensing** — needed at O2, not before. `HuggingFaceM4/something_something_v2`
   turns out to be **loader-only** (five files: `classes.py`, the loader, `dataset_infos.json`,
   README, gitattributes). Its `classes.py` gives the 174 template strings — which we already
   get free from the head's own `config.json` `id2label` — but not the per-video `template` +
   `placeholders` that target-mining needs. Those live in `train.json` / `validation.json`,
   available only from Qualcomm (gated) or unofficial mirrors. Choose: mirror, official
   registration, or identity-based targets with no mining.
2. **If G-0 fails**, which recovery: build a hand/pusher proxy, or stop the track and write it
   up. Worth deciding now, while the answer is cheap and unmotivated by the result.

**Settled:** head = ViT-g/384; hand question tested first via G-0; G-C skipped; camera = one
added opposing azimuth (any placement acceptable); gradient path = nvdiffrast.
