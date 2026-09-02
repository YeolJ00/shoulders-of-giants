# CLAUDE.md

## What this project is

We make authored 3D assets **simulation-ready**: recover physical parameters **θ**
(mass, friction, restitution/damping, centre of mass, stiffness) so an asset moves the way
its material should. Authored assets never physically existed — there is no footage and no
ground truth — so the definition of "moves right" must come from a model that has watched the
real world, and the simulator guarantees every output is physical.

**Core design: video models never generate our targets.** We never fit to generated pixels.
The simulator (Newton/Warp) produces candidate rollouts under different θ — every one physical
by construction — and an external plausibility signal scores them. Hallucination cannot enter
θ because no external model ever manufactures a motion; it only reacts to motions we made.

**Which external signal, as of now:**
- **Cosmos 3 as a chat judge (yes/no logits, absolute or pairwise) is retired as the
  optimisation signal.** Measured: approves 52.8% of clips at p > 0.90; ρ = −0.02 with
  density, −0.07 damping, −0.25 friction over 10× sweeps; bounce counts 1–2 against a true
  1–21; three fits of one object gave densities 244 / 188 / 356. This matches the literature:
  multimodal LLMs score near untrained networks on intuitive-physics benchmarks. It is kept as
  a **baseline** and for **one-bit threshold questions** (see Next, item 1).
- **Candidate replacement: predictive-model surprise (V-JEPA 2).** A self-supervised
  *predictive* video model — not a generator, not a chat model — sees part of one of our
  clips, predicts what happens next in representation space, and the prediction error is
  "surprise." Plausible dynamics ⇒ low surprise. We would optimise θ to **minimise surprise.**
  No prompt, no language, no sampling: a deterministic network, so `d(surprise)/d(pixels)`
  exists and, through the differentiable render + contact path, gives true gradient MAP.
  **The harness is built and passes V0** (`src/surprise/`, 7/7 gates, `docs/PROBE_SURPRISE.md`).
  Whether it depends on θ is what V1 answers. Rule 1 applies to it exactly as to any probe.
- **Absolute scale** never comes from a video model (they see ratios and mismatches). It comes
  from the **balance probe** against a reference block, plus the prior.

## Non-negotiable rules (each bought with a failure)

1. **A probe must be shown to depend on its parameter before it is trusted.** Sweep the
   parameter, report the correlation or threshold shift, *then* report a recovery. The
   `collide` probe was built to recover mass and measures nothing — a body shoved at fixed
   velocity slides `v²/(2μg)`, independent of mass. No amount of tuning could have fixed it.
   **This rule applies to supervision signals too**: the judge failed it; surprise must pass it.
2. **Check against invariants, not reference values.** Scale invariance, energy conservation,
   "does the reading respond to its input", two routes to one quantity agreeing. Every
   diagnosis made this way held; every one made by comparing against a handbook number had to
   be withdrawn. With no ground truth available, invariants are what we have.
3. **Detect divergence by a spatial bound, not `isfinite`.** An explicit solver reaches 10⁵ m
   long before `inf`. A cloth probe once reported ρ = −0.824 on a run that had exploded to a
   455 m footprint.
4. **Fixed target-material appearance in renders.** Magenta missing-texture surfaces are a
   different material than the one being asked about, not merely ugly.
5. **Any change to the scoring model invalidates calibration.** Model, version, layer, prompt,
   thinking mode or render style ⇒ rerun the relevant kill test before trusting a fit. (Applies
   to the judge and to the surprise model alike.)
6. **Run, don't assume.** Print the trajectory before theorising about it. Five physically
   sensible hypotheses about a scale bug failed in a row; looking at the numbers found it.
7. **Deterministic, seeded, cached.** Renders and scoring calls are the cost centres.

## State

### The simulator side is strong

- **Exact mesh contact** (`src/sim/mesh_contact.py`, `mesh_scene.py`). Ground contact tests
  mesh *vertices* against the plane — exact, since a linear function over a triangle is
  minimised at a vertex. Body–body uses `wp.mesh_query_point_sign_normal` in the *other*
  body's local frame, so each `wp.Mesh` is built once and never refitted. Replaced a sphere
  cover that was 2.28 mm from the true surface.
- **Per-body everything**: friction, mass, damping, contact stiffness. The **table is an
  object** with its own μ and cd, combined by geometric mean.
- Bounded tables, arbitrary initial orientation (`rot0`), exact reduced-coordinate revolute
  joints with per-body damping, real-asset tetrahedralisation.

### Probes

| probe | reads | result | script |
|---|---|---|---|
| tilt | friction (vs table) | 0.40° residual, 3 cm–116 cm | `bigscene_sim.py` |
| balance | **mass** of a real asset | 7/7, brackets 0.182 kg at [0.18, 0.22] | `probe_balance.py` |
| bounce | damping | ρ = −0.995 | `probe_bounce_settle.py` |
| settle | damping | ρ = +0.973 | `probe_bounce_settle.py` |
| spin | friction (vs table) | ρ = −0.952 | `probe_spin_stack.py` |
| stack | friction **between objects** | ρ = +0.994, residual 2.41° | `probe_spin_stack.py` |
| cloth drape | stiffness | ρ = −0.786, saturated at soft end | `probe_deformable.py` |
| soft press | stiffness | **diverges** | `probe_deformable.py` |

`collide` (×3) and `drop` are **retired** — see `docs/PROBE_SPIN_STACK.md`. Collide's
observable is provably mass-independent; drop is superseded by bounce. Both are absent from
`PROBE_MASK`, not zero-weighted, because a retired probe still costs a sim + render + scoring
call per iteration. `shove` remains but predates the mesh solver and is unverified.

**`PROBE_W` is provisional**: the two retired probes held the only measured Fisher weights, so
every surviving probe now starts at the unmeasured mean until the yes-region sample is re-run.

### The reading side is the bottleneck

Every number in the probe table above was **read by the simulator** (a slip angle, a pan
going down, a decay rate). The judge contributed to none of them. **We can manufacture
excellent evidence and still cannot read it off a video** — that is the problem the V-track
attacks.

## Findings that changed the plan

- **The published 98% is IntPhys 1, and IntPhys 1 is coloured shapes on plain
  backgrounds.** IntPhys **2** — realistic objects, complex backgrounds, moving cameras —
  reports that "most models perform at chance levels (50%)", against near-perfect human
  performance (arXiv 2506.09849). Our scenes are a textured object on a wooden table, i.e.
  squarely the regime where the benchmark's own authors measure the capability as absent.
  Any expectation set from the 98% figure is set against the wrong benchmark.
- **The context must be encoded WITH the mask, not gathered from a full-clip encode.**
  The reference does two forwards — `context = encoder(pieces, masks_enc)` and
  `h = target_encoder(pieces, full_mask)` — because a context token computed with full
  self-attention has already seen the region it is being asked to predict. HF's
  `VJEPA2Model` gathers instead, and its `VJEPA2Encoder` accepts no mask argument, so the
  HF path *cannot* do this correctly. Measured leak on the 2.1 ViT-g checkpoint:
  `max |masked - gathered| = 0.043`, against an effect size of ~0.4%. Every V-track number
  produced through HF before 2026-09-02 is contaminated, including a null control that
  came out at rho = +1.000 — which the leak plausibly explains, since context tokens that
  absorb target-region information make the context itself vary with theta.
- **Two checkpoints, not one.** Checkpoints ship `encoder` AND `target_encoder`; context
  uses the former, targets the latter. Using one for both is another silent contaminant.
- **Sweep the context length densely and slide the window.** The IntPhys2 evaluator does
  both, for the reason its own comment gives: "in case the physics breaking event happens
  before CONTEXT + 1". We do not know when a clip goes wrong, so committing to one split
  throws away the localisation.
- **A published effect size is part of the calibration, not background reading.** The
  intuitive-physics result rests on a within-scene gap of ~0.4%, and reverses sign when
  pooled across scenes. Any experiment aimed at it needs matched pairs and tens of them;
  a correlation over nine points against a 0.29% noise floor cannot see it. Check the
  reference implementation's *statistical design*, not just its loss function.

- **V-JEPA's target must be layer-normed before the distance is taken.** Its training loss
  is `smooth_l1(z, F.layer_norm(h))`, no affine; `VJEPA2Model` hands the target back raw,
  and the raw encoder output has `mean|·| = 2.15` against a predictor emitting `0.32`. So
  an un-normalised L1 is a large constant plus a rounding error: measured that way, a
  **static** clip scored as *more* surprising than a real one and every variant sat within
  2.4% of 2.03.
- **Surprise must be read causally, on a short horizon that slides.** CLAUDE.md's own V0
  sketch — context = first K frames, target = all the rest — is far outside what the
  predictor was trained on and separates a real clip from a frame-shuffled one by **+0.12%**.
  Sliding a 2-frame horizon across the clip gives **+2.54%** on the same clips.
- **The across-object confound reappeared in the new signal, unchanged.** Slowing one clip
  down monotonically destroys the harness's ability to tell it from a shuffled version of
  itself (ρ **+0.95** and **+0.83** on two reference clips). Pooled across those two clips
  the same correlation is **−0.01**. Clip motion is therefore a render design target
  compared *within* one scene, never a threshold applied across scenes — and every clip we
  had rendered before V1 sat at motion 0.4–2.3 against natural video's ~17.
- **Framing is part of the calibration, not a presentation choice.** A crop box computed
  per clip grows with whatever the object did, so surprise would read the framing — which
  varies with θ. One box per probe, from that probe's reference θ. And zooming buys nothing
  on its own: cropping to the trajectory normalises the zoom away, so motion-in-crop is set
  by object size against *trajectory extent*. The lever is a shorter trajectory.

- **Magnitude observables are not inherently confounded.** The rule that only thresholds work
  came from measurements taken *across objects*, where shape dominates (a drop probe gave
  ρ = −0.186 over 14 objects). Holding object and excitation fixed and sweeping only the
  parameter, the same observable gives −0.995. Magnitudes are fine *within* an object — which
  is the regime per-object θ already puts us in, and the regime every V-track sweep must use.
- **Solver parameter sets are validated jointly, not individually.** `configs/flag.yaml` is a
  stable point; changing mass alone (0.05 → 0.02) diverges. Stability couples
  `dt·sqrt(ke/m) < 0.3` and `kd/m·dt << 1`.
- **`default_particle_radius` must be set** for cloth and soft bodies. Unset, neighbouring
  particles overlap in the *rest* configuration and the body explodes — 253.7 m of travel in
  free fall with no contacts, against a correct 1.228 m.
- **A scan is a crumpled snapshot, not a rest state.** Cloth rest shape sets the strain in
  every triangle, so simulating a scan's own triangles bakes the folds into the zero-energy
  configuration and it can never drape. The scan gives dimensions and appearance; the sim
  mesh should be a regular grid.
- **Coarsening a timestep to speed a render changes results.** Balance went 7/7 → 4/7 when
  substeps were cut 60 → 24 for a render pass.
- **Inertia carried mass twice.** `set_masses` stored `G = G_unit · m` while `integrate_6dof`
  forms `Iw = R (m·G) Rᵀ`, so inertia was `m²·G_unit` — for a 0.1 kg body, ten times too
  little, and ten times the angular acceleration. Fixed; it also improved the tilt probe
  (residual 0.79° → 0.40°).
- **Relative motion must be measured in the moving body's own frame.** The stack probe took
  `top − base` in world coordinates, so the base rotating with the ramp read as sliding and
  every case collapsed the instant the table moved (ρ +0.933 → +0.994).
- **Contact coefficients combine as a geometric mean**, so a body slips at
  `atan(sqrt(μ_body·μ_table))`, not `atan(μ_body)`.
- **A body pinned at its own centre of mass is in neutral equilibrium.** `MeshProbeScene` puts
  the origin at the vertex mean, so `set_hinge` without `pivot_local` gives a balance that
  slams to its stop on any imbalance and reports no magnitude. The pivot rise above the CoM
  *is* the instrument's sensitivity.
- **A bowl is a shell.** Near its axis the lowest vertices are the outside of its base; a
  payload seated there starts inside the floor and gets ejected on contact.

## Next, in order

1. ~~**Ask the judge a one-bit threshold question**~~ **DONE — FAILED**, on the same 57 V1
   clips (`docs/results/v1_judge.json`), which is the strongest form of the test: identical
   data, identical framing, both signals side by side.
   *"Does the bowl bounce back up off the table?"* — p(yes) **0.016–0.110 on all nine**
   bounce clips, every one of which rebounds; wrong 9/9, and the bounciest clip in the sweep
   scores p(yes) = 0.026. *"Does it slide down the tilting table?"* — p(yes) **0.78–0.94 on
   all nine**, from a clip that slides 57 cm to one that slides 1.9 cm; ρ with μ is −0.32.
   *"Does the bowl move in a physically possible way?"* — p(yes) 0.57–0.78 on five authored
   violations, with the **teleport rated the most plausible of them**.
   The judge does not answer one-bit threshold questions either. It keeps only the baseline
   row it already had.

2. ~~**V0 — surprise harness.**~~ **DONE** — `src/surprise/`, `scripts/v0_surprise.py`,
   7/7 gates, `docs/PROBE_SURPRISE.md`. No install was needed: the **cosmos** env already
   has `VJEPA2Model`. Two gates are exact — `encode()+predict()` reproduces
   `VJEPA2Model.forward` bit for bit, and perturbing a known frame window moves exactly the
   token ids the code claims. Three parts of the sketch above were wrong and are corrected
   in the findings section: the target normalisation, the context regime, and the
   middle-layer knob (the predictor consumes the encoder's **last** hidden state by
   construction, so that knob cannot apply to it).

3. **V1 — KILL TEST: RETRACTED AND RE-RUNNING.** The first run reported FAIL. That
   verdict is withdrawn: two independent defects made it uninterpretable, and both were
   found by someone looking at the clips and at the reference implementation rather than
   at my numbers.

   **Defect 1 — the scenes were wrong.** `rest_height()` was never called, so the tilt and
   null scenes spawned the bowl **11.8 mm inside the table** (its lowest vertex sits 31.8 mm
   below its body origin; it was placed with only 20 mm of clearance). The solver ejected
   it, and the ejection scaled with mass: penetration ran 13.35 mm at 0.05 kg to 21.98 mm
   at 1.6 kg. So the mass "null control" — the headline evidence that surprise reads noise
   — was measuring a bowl visibly sinking further into a table. It also caused the
   divergence at high stiffness that I had blamed on the solver. CLAUDE.md already recorded
   this exact failure from the balance probe ("a bowl is a shell … a payload seated there
   starts inside the floor and gets ejected") and it was repeated in a new scene.
   Fixed: rotation-aware seating, `PEN_FRAC=1e-4`, `SUBSTEPS=420`. Tilt penetration is now
   **4.57 mm at 0.05 kg and 4.59 mm at 1.6 kg** — actually independent of the swept parameter.

   **Defect 2 — the experiment was underpowered by an order of magnitude, and the
   statistic was on the wrong axis.** Measured from the reference repo's own
   `raw_surprises`: ViT-L on IntPhys O1–O3 reaches 96.7–100% paired accuracy on a
   **within-scene gap of just +0.29% to +0.61%**, and the *same data pooled across scenes*
   gives **−1.77%** — the wrong sign. Their result depends entirely on never crossing a
   scene boundary, over 30 matched groups per property. Against that: our render noise
   floor is 0.29% of value, and V1 ran a 9-point correlation with 5 violation pairs from a
   single source. A 3/5 result has a 95% CI of about [23%, 88%]; it never distinguished
   anything from chance. Our own measured gaps (+1.0% pass-through, +0.86% teleport) are
   **larger** than the published effect — the signal was never the problem, the design was.
   Also: `eval.py` reduces with `loss.max(1)` over a sweep of **context lengths**, and V1
   averaged over **time slots** instead.

   **The re-run** (`scripts/v1_pairs.py`) copies the published statistical design and
   ladders the pairs by how defensible the ground truth is:
   **tier 1 violations** (30 pairs, categorically impossible), **tier 2 gravity**
   (g ∈ 1.62–40 m/s² vs Earth — wrong but internally consistent, nothing teleports, and
   the bridge between their regime and ours), **tier 3 θ** (no ground truth; θ_ref is an
   assumed anchor and is reported as such). Paired accuracy with Wilson CIs, on two objects.

4. **V2 — gradient path** (only after V1 passes). `d(surprise)/d(pixels)` on one clip: finite,
   and spatially concentrated on the moving object. Route through the **differentiable render
   path** (`diff_preprocess`, `diff_splat`), **not** Blender Cycles (no gradients), and through
   `SolverSemiImplicit` — `SolverVBD` gradients are exactly 0.0 in newton 1.4.0. Verify
   `d(surprise)/dθ` against finite differences per parameter. The two render paths differ in
   appearance, so compare V3's optimum to V1's Cycles-based minimum rather than assuming they
   coincide.

5. **V3 — gradient MAP on one probe**: `surprise(render(sim(θ))) − log p(θ)`, one object, one
   probe with known θ_true (balance for mass, tilt for μ). Compare against ground truth and
   against the CEM/SPSA fits on the same probe. Multi-start; report scatter (pinned vs
   unconstrained). **V4** — joint multi-probe via `joint_fit.py`, surprise replacing the judge
   margin as `s_p`, weights from measured sign-consistency as before.

6. **Instrument work, independent of the V-track verdict** (can interleave):
   - **Fix the soft-body probe.** Copy `DiffSoft`'s validated set (`k_mu=6000, lam_ratio=2,
     k_damp=10, density=800, substeps=64, radius = spacing/6`) verbatim, then sweep only `k_mu`.
   - **Finer cloth grid.** ρ = −0.786 saturates because a 7×5 grid cannot represent a fold.
     Needs a re-derived stable (mass, cell, ke, kd) point, not a changed cell alone.
   - **Render bounce and settle** via a `*_render_prep.py` in `blender_render_scene.py`'s
     format — `balance_render_prep.py` shows the pattern, including the COM-offset convention.
   - **Per-object focus in the fit**: full-frame global term alongside per-object crops.
     `multiprobe_build.py` already writes `_FULL.mp4`; `joint_fit.py` ignores it.

## Rules specific to the V-track

- Surprise is a **plausibility** signal, never a mass gauge. Absolute scale = balance + prior.
- If V1 passes for some probes only, use surprise for those and keep sim-side threshold
  reading for the rest. Mixed supervision is fine; unvalidated supervision is not.
- Every V1 plot goes in `docs/results/` with the noise floor drawn on it.
- **Score through Meta's own codebase (`models/vjepa2_repo`), never HF transformers.** HF
  cannot pass masks to the encoder, so it structurally cannot reproduce the protocol. Note
  the repo ships `VJEPA_BASE_URL = "http://localhost:8300"` as a dev override; the real
  host is `https://dl.fbaipublicfiles.com/vjepa2`.
- **Only NON-DISTILLED checkpoints can score surprise.** `vjepa2_1_vitb/vitl_dist_vitG_384`
  predict into a 1664-d teacher space they do not contain. `vjepa2_1_vitg_384` (1B) and
  `vjepa2_1_vitG_384` (2B) are the teachers and are self-consistent: verified by loading,
  encoder 1408 and predictor out 5632 = 4 x 1408 Deep-Self-Supervision levels.
- **Clip framing is fixed per probe and reported.** A crop box that varies with θ, or a
  clip whose object leaves the frame, is a confound and not a reading.
- **Carry the motion caveat into every verdict.** A θ that damps harder also moves less, and
  surprise tracks motion within a scene. A monotone surprise-vs-θ curve is separated from a
  motion-mediated artefact by *shape*: a plausibility signal has a minimum inside the sweep.
- Keep all judge code and results; they are the baseline rows of the paper.

## Environment & layout

NVIDIA GPU (use one, don't hold idle ones); Newton 1.4.0 / Warp 1.15.0 (alpha, expect churn)
+ Cosmos 3 (+ V-JEPA 2 for the V-track); VRAM-tight ⇒ scoring as a separate pass over cached
clips. **Two envs, and it matters**: sim and plots run in `warp`
(`/home/jooyeolyun/anaconda3/envs/warp/bin/python`), which has **no torch and no
matplotlib-free scoring stack**; all scoring runs in `cosmos`
(`/home/jooyeolyun/anaconda3/envs/cosmos/bin/python`, torch 2.11.0+cu128,
transformers 5.14.1, `VJEPA2Model` present). `matplotlib` is in `warp` only, so every
pipeline ends with a separate report stage. `src/gpu.py::wait_for_gpu` polls for a free
card and pins one stage to it — the box is shared and a card is held only while computing.
Blender at `/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender`.
`SolverVBD` gradients are exactly 0.0 in newton 1.4.0 — use `SolverSemiImplicit` for anything
differentiable.

```
configs/
src/
  sim/        mesh_contact · mesh_scene · deformable · probe_scene (legacy sphere cover)
  render/     Cycles harness (RENDER_SCALE, CAM_PULL, TABLE_SX/SY) · crop · camera · views
              · diff_preprocess (differentiable path, for V2+)
  judge/      Cosmos 3 pairwise + absolute; fp32 head, full-frame decode  [baseline]
  surprise/   V-JEPA 2 harness: clips · vjepa · regions · cache  [V0 done, 7/7 gates]
  optimize/   CEM · SPSA with per-object θ · gradient MAP  [V3 — to build]
scripts/      probe_*.py · bigscene_sim.py · joint_fit.py · iter_report.py · v0_*.py v1_*.py
docs/         PROBE_*.md · SCALE_RANGE.md · MESH_CONTACT.md · BIGSCENE.md · iters/ · results/
```

`scripts/iter_report.py [RUN] --max-clips N` emits one page per object: θ per iteration, the
per-probe plus/minus scores, and the clip pairs that produced them.
