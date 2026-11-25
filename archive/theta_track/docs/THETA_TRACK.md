# The theta-recovery track (2026-07 to 2026-10), archived from CLAUDE.md

Until 2026-10-16 this was CLAUDE.md's project description and state. The direction then
changed to a learned household simulator (`docs/DIRECTION.md`). Kept verbatim: the rules and
findings are still evidence, and CLAUDE.md's "Carried over from the theta track" section
points at the parts the new track uses. Every script and result path named here still exists.

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

**Retired 2026-10-06: the O-track video friction readout (`o13_tilted.py`)** -- a classifier
calibrated against the simulator can only say what the simulator did, the simulator it
was calibrated on has since been corrected (slides move 28–63%), and friction now comes
from real footage fitted through Newton (p1–p5). The sibling video readouts -- damping
(`o12`), centre of mass (`o14`), mass (`o10`) -- rest on the same construction and were
computed on the pre-fix simulator; treat their numbers as superseded, not re-validated.

`collide` (×3) and `drop` are **retired** — see `docs/PROBE_SPIN_STACK.md`. Collide's
observable is provably mass-independent; drop is superseded by bounce. Both are absent from
`PROBE_MASK`, not zero-weighted, because a retired probe still costs a sim + render + scoring
call per iteration. `shove` remains but predates the mesh solver and is unverified.

**`PROBE_W` is provisional**: the two retired probes held the only measured Fisher weights, so
every surviving probe now starts at the unmeasured mean until the yes-region sample is re-run.

### Which probe reads which parameter (`o7_sensitivity.py`, `docs/results/o7_sensitivity.json`)

Every parameter swept through every probe, on the simulator alone. **`range` is the ratio the
observable spans over the sweep and is reported alongside ρ**, because ρ = +1.000 with a range
of 1.02× is a dead probe that passes a monotonicity test.

| probe | mass | com | μ | cd | built for |
|---|---|---|---|---|---|
| balance | **4.94×** | 3.35× | 1.00× | 1.00× | mass |
| tilt→slide | 1.31× | 2.07× | **13.27×** | 1.42× | μ |
| tilt→tip | 1.00× | **6.67×** | 1.08× | 1.00× | com |
| spin-down | 3.89× | 1.68× | 1.12× | 3.19× | cd |
| bounce | 4.05× | 1.11× | 1.08× | **5.73×** | cd |
| settle | 1.12× | 1.29× | 1.00× | 1.18× | cd |

- **Balance is the mass gauge and is immune to μ and cd** (both exactly 1.00×). Read as a
  **threshold** — the reference mass at which the beam flips — it recovers mass to **±3%**
  over 0.08–0.40 kg. Read as a magnitude it saturates at its ±12° stops and returns −12° for a
  payload both far lighter *and* far heavier than the reference, which looks like a broken
  probe and is a broken readout. The rig is calibrated for ~0.2 kg; at 1.6 kg the payload
  leaves the pan mid-run. **It is confounded by CoM (3.35×)**, so mass and CoM must be fit
  together.
- **Tilt→tip is the cleanest probe in the project**: only CoM moves it (6.67×), with mass and
  cd at exactly 1.00×. Toppling is mass-independent physics and the matrix shows it. Its
  resolution is the frame step, so the reading is quantised.
- **Spin-down does NOT read contact damping.** Under the bounce/settle contact set mass moves
  it more than cd does (3.89× vs 3.19×) — it reads roughly cd/m. Under the tilt rig's stiffer
  contact it reads **friction** instead, which is what the probe table already says
  (ρ = −0.952). **Bounce is the damping probe**: cd 5.73× and μ-immune at 1.08×.
- **Settle is dead as a magnitude** — nothing moves it more than 1.29×. Its recorded
  ρ = +0.973 is real and was reported without a range.
- **Which parameter a probe reads depends on the contact calibration.** Spin reads μ at
  `pen_frac = 1e-4` and cd/m at `5e-3`; bounce's rebound spans 5.47× over cd at `5e-3` and
  only **1.03×** at `1e-4`, where the contact is 50× stiffer and cd stops mattering. Each
  probe must be run at the solver set it was validated at.
- **Fit order follows from the matrix**: (mass, com) jointly from balance + tip, then μ from
  tilt→slide, then cd from bounce. `MeshProbeScene(com_offset=...)` re-centres the local
  vertex cloud, so the frame origin *is* the centre of mass and gravity, inertia and contact
  stay consistent with no solver change.

### Joint fit of the whole θ on simulator observables (`o8_joint_sim.py`)

θ = [mass, com, μ, cd] recovered together, Levenberg–Marquardt on log θ, observables from
balance + tip + tilt at 24/30/36° + bounce. Observations come from the simulator, so this
tests **identifiability**, not model error.

| | mass | com | μ | cd |
|---|---|---|---|---|
| target A (0.240, +12 mm, 0.220, 6000) | **+3%** | **+1.8 mm** | **+2%** | **+7%** |
| target B (0.130, −8 mm, 0.550, 1500) | +10% | +4.5 mm | −12% | +41% |

(Both re-run after the ramp-normal fix below. The pre-fix numbers — A at +0.3 mm and B at
+7.5 mm / +144% — were produced by a broken tip observable and are withdrawn.)

Target A is essentially exact. **Target B is an identifiability failure, not an optimiser
one**: its starting residual was 0.39 because μ = 0.55 sits above μ* = 0.441 at 24°, so the
bowl barely slides and 0.55 and 0.35 give the same 6 cm. Adding 30/36° (μ* = 0.74, 1.17)
took μ from −40% to −13%. Its balance residual then sticks at +0.374 and never falls,
because mass is fit against a 6-step bisection whose output is quantised to ~5%.

Three optimiser defects, each of which looked like an unidentifiable system:
- **Store the PROJECTED iterate.** `to_th` clips to bounds; keeping the unclipped `x` let it
  drift outside the box while `to_th` returned the same θ, and the residual stopped
  responding to `dx` at all.
- **Trust region.** The tip column of J is ~40 against 1–15 for the rest, so an undamped
  step was almost entirely com and left the other three parameters at their start.
- **Adaptive damping.** Fixed λ plus four backtracks gave up on the first step and returned
  the starting point unchanged.

### Rendering the two new probes (`o9_probe_render.py`, `docs/results/o9_new_probes.json`)

| probe | sim observable vs its parameter | classifier reads the observable |
|---|---|---|
| bounce → cd | ρ = −1.000, range 4.74× | **ρ = +0.893**, LOO sd 0.087 (factor 1.22) |
| tip → com | range 4.61× (vase), clean | **ρ = +0.961**, ±4.7 mm over a ±18 mm sweep |

- **Bounce works through the classifier**, on 7 clips, with 117 of 174 classes at |ρ| > 0.5.
- **Mass works too — the fourth and last parameter** (`o10_balance_render.py`,
  `docs/results/o10_balance_read.json`). The balance rig rendered as a **threshold**: 3
  subject masses × 12 references, 36 clips, view b at `CAM_PULL=1.00`, 100% containment.
  Holding out a whole subject mass, the classifier calls the beam direction correctly on
  **34/36** clips and the flip point it implies recovers mass to **+9% / +18% / −3%,
  median 9%** — against 86% for a flat prior and ±3% simulator-side. Read as a magnitude it
  would fail: the beam saturates at its ±12° stops, so what carries the answer is *which way*
  it went against a reference we set.
  Note `CLIP_NF`: the balance rig runs 56 frames (`probe_balance.NF`) and `do_clip`'s hard
  `V.NF = 64` silently skipped every one of its scenes. Its clips also land in `clips_b`,
  not `clips`, because the view is b.
- **CoM works too, once the sweep is big enough.** On 7 clips it read ρ = +0.107 and ±11 mm,
  which is no better than guessing zero; on **30** clips (15 offsets × 30/36° ramps) it reads
  **ρ = +0.961 and ±4.7 mm** against 10.3 mm for always guessing zero, with the sign right at
  every offset and a whole offset held out so nothing in the calibration has seen it. **7
  points cannot fit a 174-feature readout** — the first result was underpowered, not a
  verdict. The readout does compress toward the mean (±18 mm reads as ±11 mm), so the
  ordering is far better than the scale, which is the same pattern as the across-object
  offset.
- **A bowl cannot support a topple probe.** It is 0.31 m wide and 0.09 m tall, tips at ~72°,
  and at μ = 10 it still slid 71 cm without gripping — the "topple angle" was measuring a
  sliding bowl tumbling. The vase (0.20 × 0.40 m, tips at ~34°) does topple and is exactly
  immune to mass, μ and cd (1.01× / 1.00× / 1.01×). Its CoM response **saturates past
  ±15 mm**, where it topples immediately whichever way the offset points and the observable
  goes even in com, losing the sign.
- **The camera pull must match at RENDER, CLIP and ANALYSIS time.** Third occurrence:
  `o4_joint.containment()` hardcoded 1.20 and reported 1 usable clip for a probe the clip
  stage had just measured at 100%. Now `LAB_PULL` per lab.
- **Read poses at the stride the scene was BUILT at.** The bounce rig runs 60 substeps and
  the tilt rig 420; reading a 60-substep rollout at stride 420 returns 9 frames of 64 and
  the rebound came out non-monotone in cd. `v1_killtest.track(..., stride=)`.

### Friction on a TILTED ramp, 7 assets (`o13_tilted.py`, `docs/results/o13_tilted.json`)

Five fixed tilts (15/19/24/30/37°), the μ grid **per tilt** at μ*(tilt)·[0.85 … 1.55], one
crop box per asset over its clean conditions, rendered per asset (the grid was dropped —
see below). 30 clips per asset, 11–23 of them clean.

| asset | clean | tilts | μ range | read error | ρ | held-out μ | (tilting ramp: read) |
|---|---|---|---|---|---|---|---|
| sorter | 23 | 5 | 0.15–1.96 (12.8×) | **1.13×** | 0.987 | **2%** | 2.92× |
| shark | 20 | 5 | 0.15–1.54 | 1.15× | 0.977 | **2%** | 3.46× |
| box | 20 | 4 | 0.14–1.15 | 1.13× | 0.982 | 3% | 2.01× |
| teapot | 15 | 3 | 0.15–0.68 | 1.11× | 0.961 | 2% | 1.37× |
| books | 12 | 4 | 0.14–0.71 | 1.14× | 0.944 | 4% | 1.16× |
| bowl | 11 | 2 | 0.14–0.41 (3.0×) | 1.18× | 0.945 | 2% | 1.75× |
| pot | 11 | 2 | 0.14–0.41 | 1.12× | 0.900 | 8% | 1.10× |

**Held-out μ to 2–8%, against 16–23% on the tilting ramp.** Readouts are also uniformly
tight — every asset 1.11–1.18×, where the tilting ramp ran 1.10–3.46×. The result is
internally coherent: predicting the recovery as (readout sd)/(leverage) gives 3/7/3/3/3/3/4%
against an observed 2/8/2/2/3/4/2%, so it is the instrument, not a leak.

**Why it is better**: leverage is 1.75–4.26 decades of slide per decade of μ, against
0.93–2.49 for the tilting ramp, because every clip sits near its own tilt's threshold where
the slide is most sensitive.

**Three caveats, all real:**
- **The comparison is confounded.** Both the ramp (tilting → tilted) *and* the μ placement
  (fixed grid → threshold-centred) changed. The gain may be mostly the placement, and that
  is not separated.
- **The single-vs-joined test is vacuous here.** Because μ = μ*(tilt)·factor, **no μ appears
  at more than one tilt** (verified: 0 of 11–23 per asset), so "one tilt" and "tilts joined"
  are literally the same number. This design cannot test the multi-condition effect — it is
  not evidence against it.
- **Coverage is uneven and asset-dependent.** The sorter is clean over 5 tilts and a 12.8×
  μ range; the bowl and pot only over 2 tilts and 3.0×, because a sliding body's base
  friction torques it forward and they tumble above 19°. Recovering μ to 2% inside a 3×
  range is a much weaker claim than inside 12.8×.

**The grid renderer was measured, prototyped, and not shipped** (`src/render/grid.py`,
`RENDER_SCALE_X`). Four cells in one frame measured a real **1.71×** (0.72 s vs 1.24 s per
cell-frame) and the camera-right layout works (1 px image-y spread, zero overlaps). But it
does not extend to 7 cells in a row: with `sensor_fit=HORIZONTAL`, widening `resolution_x`
at fixed `angle` narrows the *vertical* FOV rather than adding horizontal extent, and fitting
a row of 7 needs the camera ~3 m back where one cell is 14% of the frame width — matching
single-cell resolution would cost **8.6 Mpx against 1.7 Mpx rendered separately**. Cells must
lie in one row (a second row on the table plane would put them at different depths), so the
aspect mismatch is structural. Per-asset framing also gave much better boxes: **344–448 px**
against the grid's 164–232 px.

### CoM on a tilted ramp (`o14_com.py`, `docs/results/o14_com.json`)

Four tilts (16/20/24/28°) × 9 CoM offsets (±18 mm), vase, μ = 1.30 so it topples rather than
slides. 36 clips, 32 frames each.

| | tilted ramp | previous (pre-fix render) |
|---|---|---|
| one tilt alone | 3.6 mm | — |
| tilts averaged | **2.6 mm** | 4.7 mm |
| always guessing zero | 9.0 mm | 10.3 mm |
| ρ(read, true) | +0.867 | +0.961 |

Unlike the tilted friction sweep, **the multi-condition test is real here** — every CoM offset
appears at all four tilts — and averaging tilts helps, 3.6 → 2.6 mm.

Two design changes were needed, both from measurements:
- **The observable is TIME-TO-TOPPLE, not final orientation.** A fixed tilt has no topple
  angle to read, and the final orientation is the end state of a tumble: at 24° it ran
  4.5 / 82.9 / 158.2 / 34.7 / 164.6 / 66.3 / 151.7° across a monotone CoM sweep, i.e. noise.
  Time-to-topple is monotone at every tilt (24°: 1.33 → 0.21 s).
- **32 frames, not 64.** The observable completes by 0.54 s; the rest is the body rolling off,
  which the crop box then has to cover — over 64 frames no pull contained every clip (best
  55.8%). At 32 the box closes at 290 px, 100%. Cost: 7 of 36 clips saturate at "never
  toppled", and the time is quantised to 1/24 s so several offsets read 0.21 s alike.
- **`off_table` must be measured only up to the topple.** Flagging any departure marked all 36
  clips bad, because every one rolls off eventually — that is aftermath, not a different event.

### No asset carries all four probes, and the reason is geometric

| probe | assets that support it (of 13) |
|---|---|
| damping (drop) | **all** — a drop has no orientation to lose |
| friction (tilted slide) | 7 confirmed |
| mass (balance) | 8 — must fit the 25 cm pan arm |
| CoM (topple) | vase confirmed; 3 other candidates, 2 of them unusable (sphere rolls, dino topples always) |

**The probes pull against each other.** A body that topples readily (tall, narrow) is good for
CoM and bad for friction; one that slides stably (wide, flat) is the reverse; and mass needs it
small enough to weigh. Measured on the two best candidates:
- **Vase** — strong CoM and mass, but it only slides at **12°** (at 15° and 18° it topples),
  giving friction a single tilt over a 1.8× μ window.
- **Teapot** — strong friction and mass, but its CoM signal is 0.29/0.21/0.21 s at 40°,
  a 1.7× range heavily quantised by the frame rate.

So **θ can be recovered per parameter, but a joint four-parameter fit from video has no asset
to run on yet.** The joint fit itself works (+3% / +1.8 mm / +2% / +7% on simulator
observables); what is missing is one object that supports all four probes well. Either the
CoM probe needs a form that does not require toppling (a poke, which does not exist), or the
fit accepts per-parameter coverage: most assets give friction + damping, some add mass, few
add CoM.

### F0 — motion prior (Flow Poke Transformer): FAILED rule 1

`CompVis/flow-poke-transformer`, open-set base (233 M, CC BY-NC), weights on HuggingFace
despite the README saying otherwise. Conditioning is **image + observed flow at "poke"
points + a `camera_static` flag — no text**. `forward()` returns a
`torch.distributions.MixtureSameFamily` (4-component GMM over 2-D flow), so `.log_prob()`
scores our own motion directly: the simulator supplies the motion, the prior supplies what
motion is expected. That would have removed the classifier-as-ruler circularity *and*
restored end-to-end gradients, since the render collapses to one conditioning frame nothing
backprops through.

Scored on the tilted-ramp clips, poke = the object's own frames 0→4, continuation 4→20:

| | result |
|---|---|
| log p spread over a μ sweep | median 2.14 nats — not flat |
| interior maximum | 3/9 sweeps |
| \|ρ(log p, motion magnitude)\| | **0.95** |

The predicted mean flow tracked the truth at ρ ≈ 0.92 — and that was **entirely an artefact
of the poke**. Re-run with a poke that carries no μ information (two static table corners,
zero flow), the prediction is *identical across the whole sweep*: **`pred range = 1.00×` in
all 9 sweeps**, ρ(pred, true) scattered −0.90 to +0.77. It was extrapolating the momentum we
handed it.

**Direction test, and the leak that nearly produced a false positive.** Score the same
motion with the magnitude held identical and only the direction reversed — downhill, which
gravity does, against uphill, which is impossible. Result depends entirely on what the poke
carries:

| poke | downhill preferred | mean log p gap |
|---|---|---|
| the object's own frames 0→4 | **43/43** | +0.555 nats |
| two static table corners, zero flow | **1/43** | −0.580 |
| no poke at all | **0/43** | −0.637 |

Given only the image, the model systematically prefers the **physically impossible**
direction. The 43/43 was momentum extrapolation: the poke already showed the object moving
downhill, so continuing the line needs no knowledge of gravity. This was recorded as "FPT
knows physics" for one revision on the strength of that leaky test — **the same leak, in the
same session, that had already inflated ρ(pred,true) to 0.92.** A poke that carries the
answer invalidates any conclusion drawn downstream of it.

**Where the probability comes from, and its one valid use.** `predict_parallel` returns a
4-component Gaussian mixture over flow at each query point; `.log_prob(our_flow)` is that
mixture's DENSITY at our value — not a probability mass. It is only meaningful as a RATIO
between two flows scored under the SAME distribution (same image, same poke). The direction
test is such a ratio and is valid. The μ sweep is not: every μ is a different clip, hence a
different image and poke, so those log p values come from different distributions — which is
part of why magnitude dominated there.

**Using the prediction as a TARGET (the Cosmos-3 architecture, motion model in place of a
judge) does not work either, and the number says why.** Let the model predict the future
trajectory, then fit theta so the simulated trajectory matches it. FPT's predicted mean
motion over 16 frames is **0.54-0.60 px** (tight crop) or **0.70-0.77 px** (full frame),
against a true **16-43 px** -- a ratio of 0.01-0.03. The target is "nothing moves", so
fitting theta to it drives friction to its maximum for every object. Feeding it the whole
tilted table instead of a crop changes this by 30% and not by the 25x needed.

This is the model being RIGHT, not wrong: in real video an object sitting on a table does
not spontaneously slide, and a still frame does not say the tilt is about to matter. Our
scene is the out-of-distribution one. For this architecture to work the model must know that
motion IS occurring, which needs a poke carrying real observed motion (then it extrapolates
momentum and the target is whatever we poked with), real footage (the standing gap), or
conditioning on the interaction -- which FPT structurally cannot express, having no text
input.

**The self-consistency fit, which needs no real footage, has no solution.** Poke with the
CANDIDATE's own early motion (observable from our own render), take FPT's predicted
continuation as the target, and look for the mu where the simulated continuation agrees.
Across 9 sweeps: **0 crossings**. log(pred/true) runs −2.4 to −0.66 everywhere — FPT
under-predicts motion by **1.9x to 11x at every mu tested**. The gap narrows as mu rises, so
agreement lies above the top of every sweep: the fit drives friction to its maximum.

**Two design suggestions ruled out by inspection, not opinion.** (1) There is no
video-conditioned variant: FPT and MYRIAD both take `"b c h w"`, a single frame, and
`predict_simulate` rolls out autoregressively from it rather than observing real frames.
(2) Starting from an already-tilted ramp is what the tilted design does — the table is at
full angle in frame 0 and the full-frame test confirms the whole ramp is visible; the
prediction is still 0.7 px.

**Verdict: no physics extractable from our renders through this model.** It joins the judge
(ρ ≈ 0, wrong 9/9 on one-bit questions) and surprise (47% on categorically impossible
motion). What is NOT established is any claim about the model itself: μ is genuinely not in a
static frame, and the direction result is on OUR renders, which real-video models have failed
on three times now — that pattern points at the render gap as much as at any model.
Script `f0_fpt_gate.py`, results `docs/results/f0_fpt_gate.json`.

### The balance rig weighs COMPACT payloads only (`probe_balance.py`)

Extending the balance beyond the apple exposed two real defects and one hard limit. The
classifier was faithfully reading a broken instrument in both failing cases — **the sim-side
gate should have run before the renders, and did not.**

- **Seating used half the LONGEST extent.** `p0[bi,2] = z_floor + s.sizes[bi]*0.5` lifts an
  elongated payload by half its *length*: the vase (0.40 m tall) started 0.20 m above the pan
  floor, the shark (0.167 m long) 0.084 m, and both were dropped in. Now seated on the
  payload's own lowest vertex, as `seat_on_plane` does. Same class as the documented shell
  bug, opposite direction.
- **The payload must STAY IN THE PAN, and that is what actually broke it.** Printing the
  subject's displacement settles it: the apple moves 2–8 cm, the **shark 16–25 cm and the
  vase 20–26 cm** — they fall out, after which the beam slams to a stop and reads the same
  angle at every reference. That is why the shark never flipped. Two wrong hypotheses were
  chased first (seating depth, then `PIVOT_RISE` — swept 0.030→0.150 with no setting
  working) before printing the trajectory.
- **Shrinking fixes an over-wide payload, not a tall one.** The shark at scale 0.455
  (0.076 m) stays in the pan (6.2 cm) and reads **+14% / +12% / +14%**. The vase still
  travels 25 cm at a *quarter* size, because scaling preserves aspect: it is 2:1 tall at any
  scale. Laying it on its side (`BAL_ROT=90`) does not help either — then it is a cylinder
  and **rolls** out, 17–19 cm.

| payload | status | flip error |
|---|---|---|
| teapot | works | −13% / −6% / −5% |
| apple | works | +14% / +12% / −5% |
| shark | works **at scale 0.455** | +14% / +12% / +14% |
| vase | **cannot be weighed** — topples upright, rolls on its side | — |

So the rig needs payloads that are **compact and non-rolling**, and its accuracy is a
per-asset bias of order ±15%. `BAL_PIVOT_RISE` and `BAL_ROT` are now env-settable for
re-tuning. Pixel-side mass currently stands at **teapot 5%** (35/36 beam directions) and
**apple 9%** (34/36); the shark is re-renderable at the corrected scale, the vase is not.

### Is it a render gap? Tested against real photographs (n=1 per cell)

The motion prior's "nothing moves" behaviour was tested on real photos, since arguing from
how realistic our scenes look settles nothing.

| scene | weight on the static mode | largest mode | its weight |
|---|---|---|---|
| REAL photo, ball in mid-air | **0.953** | 45.6 px | **0.015** |
| REAL photo, objects on a table | 0.997 | 29.3 px | 0.000 |
| OUR render, bowl in mid-air | 0.990 | 25.4 px | 0.001 |
| OUR render, sorter on a ramp | 0.996 | 30.8 px | 0.000 |

**The model is conservative everywhere** — even a real photograph of a ball in mid-air puts
95% on "nothing moves", so the dominant effect is its prior, not our renders. **But there is
a residual**: real separates moving from static by 0.044 and ours by 0.006. One image per
cell, so suggestive only.

**A specific candidate mechanism, and it is not scene realism.** Mean edge sharpness: a real
moving photo is **4.23** against **18.95** for a real static one (4.5×), while our moving and
static renders are **5.51** and **6.20** (1.1×). A real photograph of a fast object is
BLURRED, and that blur is the strongest single-frame cue that something is in motion. Our
renders are perfectly sharp — a bowl mid-fall looks like a bowl at rest.

**Motion blur is off and cannot simply be switched on.** `use_motion_blur` was never set
(Cycles defaults off). Adding `MOTION_BLUR=` changes nothing — edge sharpness 5.29 either
way — because `blender_render_scene.py` sets object transforms manually and calls `render()`
per pose without advancing the frame counter or keyframing, so Cycles has no motion vectors
to blur along. Producing it means keyframing the poses and rendering an animation: a real
change to the render path, and one that invalidates calibration under rule 5. Worth more
real images to confirm the blur signal before paying that.

### A MOVING CAMERA unlocks the motion prior (`CAM_ORBIT`)

Under `camera_static=True` — which our locked camera honestly is — FPT puts ~0.99 of its
weight on "nothing moves", **for real photographs too** (0.954 on a real ball in mid-air).
It is answering "does a randomly chosen point in a locked-off shot move?", where the honest
answer is usually no. That is not a render gap and not a failure to know physics.

`CAM_ORBIT=<deg>` swings the camera around its target across the clip (per-frame, unlike
`CAM_JITTER`'s constant offset). Three things then hold together: the camera really moves,
`camera_static=False` is truthful, and the induced background flow is measurable from the
render (**−40 px of 544 = −0.0735** over 24 frames; an earlier guess of 0.012 was badly
wrong). With all three:

| conditioning | static weight |
|---|---|
| locked camera, `camera_static=True` | 0.994 |
| locked camera, `camera_static=False` | 0.711 |
| **14° orbit, `camera_static=False`, measured background poke** | **0.001** |

**And it reads the physical situation, not just "an object is here."** Same bowl, same
scene, same orbit, varying only the frame:

| frame | height above table | predicted flow | static weight |
|---|---|---|---|
| 0 | **+29.0 cm, airborne** | **44.3 px** | **0.091** |
| 6 | −1.7 cm, settled | 2.6 px | 0.529 |
| 40 | −1.6 cm, settled | 15.1 px | 0.267 |
| 56 | −0.2 cm, settled | 16.5 px | 0.439 |
| background (table) | — | 20.9 px | 0.472 |

The airborne frame gets 2.7–17× the predicted motion of settled frames, and is the only
case where the object beats the background. **Why it works: parallax.** With a locked camera
a bowl in mid-air and a bowl at rest are visually ambiguous; a moving camera separates them
in depth. This is the first positive physics signal from any external model in this project.

**But it does NOT discriminate theta, tested properly.** The architecture would be:

    input : one rendered frame + a background poke carrying the MEASURED camera flow
            + camera_static=False + query points on the object
    output: a Gaussian mixture over the object's flow for the next second
    fit   : theta* = argmax_theta  log p( flow_sim(theta) | image, poke )

Because the conditioning is then IDENTICAL for every candidate, this is a proper
within-distribution likelihood ratio -- unlike the earlier mu sweep, which used a different
poke and query point per clip and was therefore comparing across distributions.

Run on the damping sweep (9 cd values, one shared conditioning frame), two windows:

| window | argmax | spread | rho(log p, cd) | rho(log p, flow) |
|---|---|---|---|---|
| frames 0→24 (contains the fall) | cd = 400, at an edge | 1.03 nats | **−0.089** | **−0.972** |
| frames 8→32 (post-impact, where cd lives) | cd = 9051, interior | 0.68 nats | +0.638 | **−0.856** |

The first window is a PROBE DESIGN error worth keeping: flow was 189–234 px while the
rebound varied 6× (186 → 30 mm), because over 24 frames the motion is dominated by the fall,
which is cd-independent. Bracket the parameter.

The second window has an interior maximum, but it is not damping: the flow magnitudes are
themselves **non-monotone** in cd (39.7 / 68.6 / 72.0 / 77.3 / 45.5 / 38.6 / 39.1 / 39.2 /
39.2 px), log p tracks flow magnitude at −0.856, and the "interior max" is simply where the
flow is smallest. The fit would select whichever theta moves least.

**So the moving camera buys a SUPPORT judgement, not a measurement.** It makes the model
commit to motion (static weight 0.994 → 0.001) and separate a supported object from an
unsupported one (2.6×). That is a depth cue. It is not sensitivity to how much friction or
damping, and the two must not be conflated.

**Fitting theta to the prior's TRAJECTORY (not scoring theta under its density).** The
prior does not need to respond to theta: it only needs to state what the motion should be,
and we invert our own monotone forward model to reach it. Under the orbit conditioning the
target is substantial (32 px, static weight 0.129, against 0.7 px with a locked camera), and
on a single 24 deg ramp the fit is well posed -- sim flow monotone in mu (96 -> 18 px), a
clean interior minimum, fitted mu = 0.605 inside a 0.374-0.683 sweep.

**It fails the same-question-twice test.** Asked from four ramp angles, where a real friction
reading must agree:

| tilt | mu* | target px | fitted mu |
|---|---|---|---|
| 15° | 0.160 | 25.3 | 0.247 |
| 19° | 0.263 | 24.0 | 0.321 |
| 24° | 0.441 | 32.0 | 0.605 |
| 30° | 0.741 | 29.5 | 1.148 (at the sweep's edge) |

Fitted mu spreads **4.64x** and tracks mu* at a near-constant 1.2-1.55x. The cause is in the
third column: **the target is 24-32 px whatever the tilt.** A scene-independent target
inverted through a tilt-dependent forward model returns
`mu ~ 1.4 x tan^2(tilt)/mu_table` -- a restatement of "barely slides", not a property of the
object. The architecture is sound and the prior's output is the problem.

**Joint fit over tilts, 7 assets (`f4_fpt_joint.py`, `docs/results/f4_fpt_joint.json/.png`).**
One μ per asset fitted to 5 tilts (15–37°) at once, on a COMMON μ grid (0.08–2.5, 18 values,
sim only), one camera pull (2.0) for every asset, 6° orbit, background poke measured in both
axes. (The earlier run set the poke's x and y to the same value -- a diagonal camera flow
that does not exist.) FPT is exactly deterministic (repeat 0.0).

| | result |
|---|---|
| target, px over 1 s | **4.6–21.6**, rising with tilt in every asset |
| variance of log target | **tilt 88%, asset 5%**, interaction 8% |
| E\|f\| vs \|E f\| | equal -- no cancellation between modes; largest mode 25–34 px at weight ~0.01 |
| fitted μ vs const-target null / tilt-only null | **1.00× / 1.00×** median |

**Per point, the prediction contains NO downhill motion at all** (`f4_fpt_points.png`).
Decomposing each of the 16 per-point predicted flows into the background (camera) direction
and the image-space downhill direction: **downhill −0.6 to +1.3 px in all 35 asset×tilt
cells; background direction 4.6–21.5 px**, residual 1–5 px. The "rise with tilt" is the
camera-copy component growing, not a physics response. It is also the wrong camera motion:
the orbit pivots near the object, so a static object really moves only **1.3–2.3 px**; FPT
paints it with a large share of the far background's +27 px, as if the object were part of
the backdrop. Magnitude-only fitting hid this -- a 15 px sideways drift and a 15 px slide
have the same length. **So there is no object-motion target here to fit to**, and the
fitted μ (joint 0.40–2.2) is the creep floor matched against a camera artefact. The earlier
orbit results (static weight 0.994 → 0.001, airborne vs settled 2.6×) were also magnitude
readings and must be re-checked for this before they are cited: "commits to motion" may be
"copies the camera poke".
**Lesson: compare trajectories as per-point VECTORS, decomposed against known directions
(camera, gravity/downhill), never as a magnitude.**

**MYRIAD (paths, not one displacement) — `f5_myriad.py`, `docs/results/f5_myriad*.{json,png}`.**
Same group and repo as FPT; `predict_simulate` samples each point's path step by step
(1/12 s steps). Weights `CompVis/myriad` (CC BY-NC); DINOv3 config is HF-gated, rebuilt as
ViT-L/16 and verified by a strict load (408 keys renamed for transformers 5 nesting).
Static camera, `camera_static=True`, background points poked with zero motion.
- **Positive control, MYRIAD's own images**: object UNPOKED, a basketball in mid-air moves
  3.5 px of 3685 in 1 s. Poked with its own example poke, it moves 2 steps and **stops in
  mid-air** (no gravity). A poked rolling ball DECELERATES smoothly (42 → 12 px/step) --
  the one physical behaviour it shows.
- **Our 35 asset×tilt cases, unpoked: −0.2 to −0.5 px in 1 s.** Nothing moves.
- **Self-poked** (each candidate μ pokes with its OWN first simulated 1/12 s -- no external
  motion enters): MYRIAD continues **0–11 px whatever μ is** (box 10.8 / 10.9 / 10.8 at
  0.7 / 1.0 / 1.5 μ*) against the simulator's 12–490 px; some bowl/teapot samples fly
  upward. From rest, 1/12 s of motion is 0.3–3 px for every μ, so the poke carries almost
  no μ information -- a ramp started from rest is the wrong probe for a poke-driven model.
**Verdict for the tilted ramp: no path target.** What MYRIAD does model is deceleration of
something already moving, which points at a shove probe (initial velocity, stopping
distance v²/2μg), where the self-poke carries v.

**A real ramp changes nothing (`F4_RAMP=1`, labs `f6_*`, `f5_myriad_paths_ramp.png`).**
Hypothesis tested: a table on tilted legs reads as a table falling over. Replaced by a 1.0 ×
0.4 m plank lying on the LEVEL table, low end on the tabletop, high end on a block; the
plank's top face is the sim plane (pivot and extent passed per scene, so the old 5–6 cm
table-pivot mismatch is gone too). Same 7 assets × 5 tilts. **MYRIAD unpoked: −0.2 to
−0.6 px in 1 s. Self-poked: 0–11 px whatever μ is** (box 11.3 / 11.0 / 11.0 against the
sim's 62 / 43 / 14). **FPT: downhill +0.1 to +0.8 px**, along-camera 8.8–20.5 px. Both
models behave as on the tilted table, so the scene's plausibility was not the blocker.
Batching cases into one `predict_simulate` call (`MYRIAD_CHUNK=8`) took MYRIAD from ~40 s
to ~3.5 s per case: the rollout is ~290 sequential tiny rounds, launch-bound on one CPU
core at a 33% GPU.

**Shove probe + k-sweep (`f7_shove.py`, `f8_ksweep.py`, `docs/results/f7_*`, `f8_*`).**
MYRIAD's condition is one image (frame 0) plus KNOWN positions of every track for steps
1..k after it; it fills in the rest. Each candidate μ is conditioned on its OWN first k
simulated steps (k = 1/2/4 = 0.08/0.17/0.33 s); only the continuation (step k → 1 s) is
compared. Shove: level table, v0 = 0.8/1.2/1.6 m/s, sim slide matches v0²/2μ_eff g to ~3%
below μ ≈ 0.5 (rule 1); the box trips over above μ ≈ 0.6.
- **The poke is MYRIAD's only μ-dependent input.** At k = 1 friction enters solely as a
  shorter first step (box 0.8 m/s: 12.0 px at μ = 0.08 → 7.1 at 1.67). Its "response to
  friction" is a response to the speed it was handed.
- **Shove, k = 1**: the first real motion from MYRIAD -- the right direction, plausible
  slides -- but longer than physics at nearly every μ, including 0.08, and growing slower
  than v0² (box 119/130/191 px against the sim's 36/84/161). Crossings in 9/21
  asset×speed cells, spread up to 8.8× (teapot 0.57/0.84/0.10).
- **Shove, k = 2/4**: for low-μ candidates MYRIAD's continuation ≈ the sim's (sorter k=4:
  1.02/1.11/1.06) -- kinematic extrapolation of what it was shown, which matches every
  candidate alike. Crossing spread across speeds 1.8–16×. The one consistent case, pot
  k=4 (0.085/0.094/0.099), sits at the grid's bottom edge where the pot sim tumbles.
- **Ramp, even k = 4** (0.33 s of observed downhill motion): MYRIAD's continuation is
  10–40% of the sim's -- it brakes the object right after the given steps. Crossings
  appear only where the sim itself barely moves and rise with tilt (box 0.23 → 0.48 at
  15 → 19°, i.e. ~1.5–1.8 μ*): the "barely slides" answer FPT gave, again set by the tilt.
**Verdict: no friction reading from MYRIAD at any k, on either probe.** Unconditioned it
predicts nothing; conditioned it either extrapolates what it is given (shove) or stops
(ramp). Neither varies with the object.

**MYRIAD setup check on its own benchmark: REPRODUCED (`f10_owm.py`, `f11_dino_versions.py`,
`docs/results/f10_owm.json`).** Their `openset_prediction.py` run unchanged on OWM-95 with
our loader (their protocol: 1 known step = true first-step motion of every track, video
fps, fp32, ensemble 5, 32 frames). Two local changes, neither arithmetic: the fp32
flex-attention tile is shrunk to fit an A6000's 101 KB shared memory (patch marked in
`models/myriad/myriad/model.py`), and pandas is vendored in `models/myriad/_deps`.

| best-of-5 mean SQUARED distance, [-1,1] coords | OWM-95 |
|---|---|
| static (nothing moves) | 0.066 |
| constant velocity from the poke | 0.279 |
| **our MYRIAD** | **0.0294 ± 0.005** (median 0.012; beats static on 76% of videos) |
| paper (arXiv 2604.09527, Table 1, Best-5) | **0.029** |

**The paper's "minADE" squares the distance** — min_k 1/(KT) ΣΣ‖x̂ − x‖²₂ (Sec. 4.2) — so
it equals their script's `top_k_MSE`. Read as a plain distance, the same run gives 0.053
and looked 1.8× off; that was a metric misreading, not a bug. It was chased first through
the loader: DINOv3 features are **bit-identical** under transformers 5.14 and 4.56 (max
|Δ| = 0.0 on three images, weights load natively under 4.56), and the checkpoint has no
EMA copy (`model` + optimizer + scheduler, 665 M values). **So every MYRIAD result in
f5–f9 was obtained with a correctly loaded model**, and stands — with one caveat of
protocol: k = 2/4 conditioning is outside what the authors evaluate (their script asserts
`context_length == 1`). Lesson, same class as the V-JEPA layer-norm finding: **read the
metric's formula, not its name.**

**MYRIAD as a likelihood over μ (`f12_likelihood.py`, `docs/results/f12_likelihood.*`).**
Shove, one SHARED conditioning per asset × speed (the frictionless launch: a rigid shift of
v0·1/12 s), so all 18 μ are scored under the same distribution -- a proper likelihood
ratio. 256 samples, fp32 (the reproduced precision; `.float()` also converts MYRIAD's bf16
KV cache), per-step 2-D KDE of the object-centre path.
- **The density is strongly lopsided -- toward the lowest friction.** Argmax at μ = 0.08,
  the grid floor, in **15/21** asset×speed cells; log-likelihood gaps 9–85 nats,
  bootstrap-stable. MYRIAD's prior is that a shoved object keeps sliding.
- **Where it is interior, it moves with speed, not object**: only at v0 = 1.6 m/s
  (0.12–0.33), where the lowest-μ sim paths outrun MYRIAD's sublinear growth.
- **Identical across objects**: 0.08 for all seven at 0.8 and 1.2 m/s -- porcelain, brass,
  cardboard alike.
The band test (f8) said "too wide to exclude anything"; the likelihood says "prefers ice,
for everything". Both are the same fact: MYRIAD's sense of a slide is not conditioned on
material. Same pattern as every earlier signal: a monotone curve set by motion magnitude.

**Wan 2.2 i2v as the motion source -- PILOT (`f13_wan.py`, `docs/results/f13_wan.*`).**
A generated video as the target, revisited by decision; guarded by tracking. Wan2.2-I2V-A14B
(Physics-IQ Verified 32.2, best open i2v with HunyuanVideo 1.5 -- API models are not better:
Sora 2 26.5), from our real-ramp frame 0, prompt NEUTRAL on the outcome, 81 frames at
16 fps, 40 steps, guidance 3.5; CoTracker3 on the same 16 object points + 8 background.
2 assets × 19/24/30° × 2 seeds, ~15 min per clip on an A6000 with CPU offload.
- **11/12 clips: the object does not move** -- |downhill| ≤ 2 px over 5 s, box and sorter,
  including 30° (tan 30° = 0.58), where a real box slides.
- **The one slide is at the SHALLOWEST tilt** (sorter 19°, seed 0: 6 px in 1 s, 45 px by
  5 s); the same object at 24° and 30° stays put. Not monotone in tilt.
- Objects stay rigid (0–3% change in point-pair distance); the camera drifts 10–12 px on
  one seed despite "static camera" in the prompt.
- Inverted through the sim, "no motion" lands in the creep floor: μ = 0.74–2.5, three of
  four at or near the grid top -- the same "nothing slides" answer as FPT and MYRIAD.
Wan's prior from a neutral prompt is a still scene. Untested: a prompt that states THAT it
slides, leaving HOW fast to the model -- the prompt then fixes the outcome and only the
speed could carry μ, which must still scale with tilt and differ by object.

**Wan 2.2, prompt "slides down" (`F13_PROMPT=slide`, `docs/results/f13_wan_slide.*`).**
The prompt fixes THAT it slides; only how it slides could carry μ. All 12 clips slide
downhill, camera still (0.3–1.3 px), objects mostly rigid (seed 0: 2–8%; seed 1: 12–34%).
- **No acceleration.** Fitting down(t) ∝ t^p over 0.25–2 s gives **p = 0.91–1.12 in 10/12
  clips**: the object is already at full speed in the first frame and moves at constant
  velocity. A body released from rest on a ramp has p = 2. Only one clip reaches 1.76, and
  it flies off the plank and out of frame (780 px).
- **The seed sets the speed, not the tilt or the object.** At 1 s: box seed 0 = 10/13/16 px
  at 19/24/30°, seed 1 = 59/69/71 px -- a 4–6× seed effect against a ≤1.5× tilt effect;
  sorter seed 1 even slows with tilt (105/92/23 px).
- Inverted through the sim, fitted μ differs 4–14× between seeds at the same tilt and
  spreads 1.8–9.3× across tilts.
**Verdict: Wan generates a plausible-looking slide, but with the wrong kinematics
(constant velocity from rest) and a speed drawn by the random seed.** Without the prompt it
does not move; with it, the motion carries no μ. Same end state as FPT and MYRIAD.

**LLM outcome prediction -- rule 1 for TEXT supervision (`f14_llm_outcomes.py`,
`docs/results/f14_llm_outcomes.json`).** 6 assets (shark excluded: rests on a fin tip),
incline 10–45° and shove 0.5–3 m/s, image + close-up + dimensions; Claude Opus 5.5,
GPT-6 Astra, Gemini 3.1 Pro via the Letsur gateway. Ground truth ANALYTIC (incline
stay/slide/topple and shove distance from mesh b/h with a shell or solid CoM), kept only
where the outcome is the same for every μ_eff in [0.30, 0.60].
- **On the 48 unambiguous questions: Claude 100%, GPT-6 100%, Gemini 96%** (chance 28%,
  majority-per-kind 50%). They apply textbook mechanics with typical friction correctly.
  **Partly circular**: the truth was built from that same typical range, so this shows
  coherent reasoning with a generic prior -- not object knowledge.
- **On the 24 ambiguous questions -- where supervision would actually act -- the implied
  μ_eff is ~0.4 for everything**: 0.31–0.47 across objects and models, i.e. the middle of
  the prior. Each model is roughly self-consistent between two routes (incline threshold
  vs shove distance; Claude 0.41/0.41, 0.44/0.40, 0.47/0.45), but the models disagree on
  which object is grippier: teapot is Claude's highest (0.47) and Gemini's lowest (0.32).
**Verdict: LLMs are reliable textbook reasoners and plausibility checkers, not measurers.**
Their object-specific numbers are a generic prior plus model-dependent noise -- the same
conclusion CoDimRecon (arXiv 2609.36024) reaches by construction: "accepted values are
effective properties under the assumed model, not recovered material parameters."

**Not yet established:** n = 1 scene and 4 frames. It must replicate across objects, and it
must discriminate theta (does it separate high from low damping?) before anything is built on
it. Rule 1 applies unchanged.

### FIRST REAL-WORLD MEASUREMENT: Physics 101 (`p1_ramp_friction.py`, `docs/results/p1_*`, `p101_density.json`)

Physics 101 (Wu et al., BMVC 2016; `data/phys101/`, 4.5 GB, git-ignored): 101 real objects,
15 materials, measured mass + volume, ramp at 10°/20° (white board onto a table), drops onto
5 surfaces, springs; side + top DSLR and Kinect views, ~30 fps. **Precedent to state up
front: Galileo (NeurIPS 2015) and Physics 101 already fit physics to these real videos**;
ours differs in the exact-mesh simulator, the transfer to never-real assets, and the
invariant / repeat-trial checks.

- **Density per material, measured directly** (mass/volume, no model): foam 0.006,
  cardboard 0.070, hollow rubber 0.37, wooden block 0.68, plastic block 0.86, porcelain
  1.06, dough 1.38, metal pole 7.8 g/cm³. These are EFFECTIVE bulk densities (hollow
  cardboard boxes, porcelain cups at 1.06 vs 2.4 solid) -- what an authored asset of the
  same build needs.
- **Kinetic friction of cardboard on the ramp board, 20°: μ = 0.229 (IQR 0.210–0.258),
  93 trials, 9 cubes.** Side view, CoTracker3, s = s₀ + ½at² on the ramp segment (fit RMS
  0.3–3 mm), μ = tan α − a/(g cos α). Per-cube IQR 0.01–0.03 across 5–14 trials.
- **Three invariants pass**: (1) every cube STAYS at 10° (μ > tan 10° = 0.18; checked by
  eye -- the automatic fit tracks the hand when the cube does not move, giving impossible
  negative μ, so the 10° fits are void); (2) the camera scale from each cube's own edge
  agrees across all nine, 1633–1724 px/m (±3%); (3) the target (cardboard vs foam) does not
  change the ramp-segment μ.
- **μ is insensitive to scale and sensitive to angle**: at 20° the measured term
  a/(g cos α) is only 0.078, so 10% scale error moves μ by 0.008, while 1° of ramp-angle
  error moves it by 0.02. The nominal 20° is the dominant uncertainty.
- **Unexplained**: μ falls with cube size (Spearman −0.80, 0.28 → 0.21), not with contact
  pressure (mass/area, +0.07). Cubes were filmed in separate sessions; finish or setup may
  differ.
- Cardboard first because its volumes are exact cubes (343.0 = 7.0³), giving metric scale
  from the object itself; the Kinect depth is 8-bit grey video with no recoverable scale.

**All 15 materials on the real ramp (`p2_ramp_all.py`, `docs/results/p2_ramp_all.*`,
`p2_two_angle.json`).** 643 clips (≤5 per object at 20°, 2 at 10°), one camera scale
(1661 px/m from p1), each clip classified SLIDES / ROLLS (rigid-fit rotation > 20°,
excluded: a rolling cylinder's ~⅔ g sin α would read as a false μ ≈ 0.12) / STAYS.
- **Kinetic μ vs the ramp board, 20°**: dough 0.159, plastic doll 0.163, porcelain 0.166,
  hollow wood 0.197, metal pole 0.201, wooden pole 0.211, cardboard 0.229, wooden block
  0.230, plastic block 0.246, plastic toy 0.252, plastic ring 0.330 (wide IQR).
  **Never slide at 20° (μ_static > 0.36)**: foam 35/35, rubber 23/23, hollow rubber 36/36.
  Metal coins 20/20 "stay" but are a few px thick in this view -- unverified.
- **Two-angle invariant passes (rule 1b)**: at 10° only the two materials with μ_k <
  tan 10° = 0.176 slide -- plastic doll (10/12) and porcelain (4/8) -- and their 10° μ
  matches their 20° μ: doll 0.155 vs 0.163, porcelain 0.146 vs 0.166. Dough (0.159) stays at
  10° -- static above kinetic, plausible for a sticky material.
- **Two pipeline faults found by looking, not by the numbers**: (1) at a 48 px tracking
  grid, thin objects fell between points and defaulted to STAYS (hollow-wood plates and
  toys visibly slid); a 19 px grid fixed it. (2) `torch.hub.load` contacts GitHub on load;
  an HTTP 504, retried per clip, failed 52 clips -- now loaded from the local cache.

**THE LOOP CLOSES: Newton fitted to real footage (`p4_sim_loop.py`, `docs/results/p4_*`).**
Each real cardboard cube simulated at its measured size and mass on a 20° plane, μ swept
0.10–0.40, every real trial aligned in release time and offset to the simulated curve.
- **Simulator-fitted μ = analytic μ**: 93 trials, median |Δμ| = 0.005, correlation 0.84
  (sim 0.235, analytic 0.229 -- a small upward offset; the outliers are trials where the
  parabola fit was poor and the time-aligned simulator fit holds up better).
- **Newton reproduces the real trajectories to 0.4–1.0 mm RMS per cube** -- as close as, or
  closer than, the parabola (0.4–3.2 mm). First direct check of the simulator against
  reality in this project, and it passes.
- **Sim acceleration matches the textbook law within 1%** at μ = 0.15/0.25/0.30 (effective
  0.150/0.249/0.298), so the V_EPS = 1e-3 friction regularisation costs nothing for sliding.
  An apparent 11% overshoot was a constant ~6 cm/s HEAD START: `seat_on_plane` leaves 2 mm
  of clearance and the cube drops onto the slope at release. Harmless for release-aligned
  fits; matters for anything timed from t = 0.

**Seating gap fixed (`MeshProbeScene.seat_on_plane`, clearance 2 mm → 0).** Verified: the
sliding cube now matches ½at² to 2.5% at 1 s (was 11%); a grippy cube (μ 0.8, 20°) creeps
8.8 mm in 1 s instead of 41.8 mm; real assets at rest wobble 2.1–5.7 mm normal to the
plane instead of 4.7–12.9 mm. **The "creep floor" recorded for the tilted probe (8–23 cm)
was largely this 2 mm drop**; ~9 mm/s of true no-static-friction creep remains. Results
computed before the fix (o13, o14, f4/f6, f7, p4) carry the old drop. `probe_balance.py`'s
own +2 mm pan seating is separate and was left as measured.

**Real friction and density onto our assets (`p5_transfer.py`, `docs/results/p5_transfer.json`).**
- **The friction rule factorises on real data.** Every slide gives μ_pair(object, ramp
  board) and, after the ramp, μ_pair(object, table). Their ratio is one constant across
  block-like materials, 0.75–1.05 (cardboard 0.83, plastic block 0.96, wooden block 0.88,
  wooden pole 0.85, porcelain 1.01, dough 1.05, plastic toy 0.93, metal pole 0.75); the
  outliers are shape, not friction (ring 0.31 rolls on the table, doll 1.36 tumbles,
  hollow wood 1.89 on n = 3). This supports any product-form rule, not the geometric mean
  specifically.
- **One anchor is unavoidable** (pair frictions fix per-material μ only up to a common
  scale) and is stated once: our scene table ≈ the ramp board, so body μ = μ_pair² / 0.45.
- **Hollow objects scale with AREA.** Cardboard cubes: mass/volume varies 35% with size,
  mass/area is constant at 0.106 g/cm² (CV 0.03). Volumetric transfer made our big box
  4.1 kg; areal gives 939 g. Mass now transfers by build: areal (cardboard, measured),
  solid (toys 49–97 g, C-clamp 1156 g, wooden bowl 1778 g from solid-wood density 0.68),
  shell (teapot, vase, brass pot, sorter -- NOT measured: assumed wall × handbook density,
  reported as ranges; open containers' meshes may double-count inner and outer area).
  Books, baseball, apple have no Physics 101 class.

**Centre of mass and inertia now come from the object's BUILD (`MeshProbeScene(mass_model=)`,
`src/data/assets.MASS_MODEL`).** The old convention -- centre of mass = plain vertex mean,
inertia = area-weighted cloud about it -- disagreed with itself and put a hemisphere's
centre of mass at 0.0624 against a true 0.0375 (solid) or 0.0333/0.0500 (closed/open
shell). Now one mass point cloud sets both: "shell" (area-weighted surface) or "solid"
(interior voxels), chosen per asset ("auto", the default); "vertex" keeps the legacy
behaviour for reproducing old runs. **Checked against textbook values: shell cube inertia
+0.3%, open-bowl shell centre of mass exact (r/2), closed-hemisphere shell exact (r/3);
solid cube +0.2%, solid hemisphere +0.6%, solid cylinder +0.4/+1.0%.** Getting the solid
voxel fill right took three tries, each caught by those numbers: a face lying exactly on a
voxel boundary leaked the exterior fill inside (a solid cube came out with its shell
inertia); thickening the wall to close the leaks grew the body outward (+10%, +3%);
offsetting the grid by an irrational fraction of a voxel, plus half weight on wall voxels,
fixed both. **At rest on a level plane the wooden bowl stopped rocking: max tilt 19.4° →
2.5°**, drift 6.6 → 1.4 mm; shark 7.2° → 4.6°; the rest 0.4–2.6° either way.

**What the two fixes do to the probes already run (`p6_recheck.py`, `docs/results/p6_recheck.json`).**
Every stored o13 / o14 condition re-simulated in memory three ways (legacy / seating fix /
both). Legacy reproduces the stored numbers (0.006–0.13 cm for 4 assets; 0.6–2.2 cm for
books, shark, sorter -- GPU nondeterminism on long slides).
- **o13 slide magnitudes move 28–63%** with both fixes (seating alone 14–42%). Friction
  ordering survives at the informative tilts (ρ(μ, slide) ≈ −1 at 15–30° for most assets)
  but breaks at the steepest tilt, where bodies tumble (pot 37° flips to +1.00, box 37°
  −0.14). The o13 μ readout inverted the OLD forward curve against clips rendered from the
  old motion, so its reported 2–8% is a statement about the old simulator; re-validating
  it needs re-simulation, re-render and re-scoring.
- **o14 vase: unchanged** (time-to-topple identical; the vase's shell centre of mass sits
  where the vertex mean did). **o14 pot: broken** -- with its true (lower) shell centre of
  mass it no longer topples in 21 of 36 conditions (was 5), and ρ(com, t) at 16°/20° goes
  from −0.90/−0.97 to +1.00/−0.27. **The pot's 1.8 mm CoM result was measured on a wrong
  mass model**; the probe needs steeper tilts to work on the real pot.

**The simulator's restitution, measured (`p7_cd_restitution.py`, `docs/results/p7_cd_restitution.json`).**
Solid 7 cm cube dropped 0.20 m; e = rebound/impact normal speed every substep.
- **Bounce-rig settings (pen_frac 5e-3, 60 substeps): e spans 0.97 → 0.04** over cd 50 →
  30000, and **e is a function of cd / mass, not cd**: at cd 1566 a 0.1 kg cube gets 0.52,
  a 0.4 kg cube 0.82; at equal cd/m they coincide (0.52/0.50, 0.26/0.24). A probe that
  sets one cd for bodies of different mass gives them different restitution.
- **Tilt-rig settings (pen_frac 1e-4, 420 substeps) cannot represent realistic restitution:
  e = 0.77–1.00 over a 600× range of cd.** Every impact in o13 / o14 / p4 is near-elastic --
  irrelevant to sliding, wrong for toppling and tumbling. Same root cause as the old note
  that bounce goes dead at the stiff setting.
- `p5_transfer.cd_for_restitution(e, m)` turns a real e into a bounce-rig cd through this
  curve. Real material restitution is still unmeasured: Physics 101's 30 fps drops failed;
  "Bounce and Learn" (CVPR 2019, 100 fps) bounces one foam ball off many SURFACES, not many
  materials. A 240 fps phone recording of real counterparts would serve the p3 pipeline.
- References for uncovered assets, labelled as such: baseball mass 142–149 g and COR
  0.546 ± 0.032 (rulebook specification, against ash wood at 58 mph) → cd 1837–2341;
  apple 333–377 g and books 11.6–17.4 kg from published bulk densities, not measured here.

**Tilt-rig settings changed: restitution by Hunt-Crossley α (`src/sim/rigs.py`,
`MeshProbeScene.set_restitution_alpha`, `scripts/p8_alpha.py`).** The contact law is
Hunt–Crossley, fn = k·w·pen − HC·cd·w·pen·v, whose restitution is ≈ 1 − α·v with
α = 2·HC·cd / (3k): it is set by cd/k, not cd -- which is why the stiff tilt rig could not
bounce (k 50× larger) and why one cd gave different masses different e (k ∝ mass). Damping
is now specified as α and cd = 1.5·α·k/HC per body, re-applied whenever stiffness changes.
- **Measured: identical e for 0.1 and 0.4 kg** at every α; the two rigs agree within
  0.04–0.07 (bounce 0.50 vs tilt 0.54 at α = 0.4, 2 m/s); faster impacts bounce less
  (0.54 at 2 m/s, 0.70 at 1 m/s), a trend real materials share, more weakly.
- Rig definitions live in `src/sim/rigs.py` (TILT_RIG: pen_frac 1e-4, 24 fps × 420,
  **α = 0.4 -- an assumed mid-range default** until real restitution is measured;
  BOUNCE_RIG: pen_frac 5e-3, 24 × 60, raw cd as validated). o14, p4, f7 now call
  `apply_tilt_rig`.
- **Sliding unchanged** (cube acceleration 1.054 vs analytic 1.051). **Topple times
  unchanged** (decided before landing: vase ρ(com, t) −0.97/−0.97/−0.92/−0.95; pot still
  broken by its true centre of mass, 23/36 never topple). **Landing changed**: a toppled
  vase used to keep bouncing and flipping (centre 4–14 cm above the plane, rebounds to
  0.72 m/s); now it lands, rebounds once at 0.24 m/s and settles on its side at 3.3 cm.

**Pot topple probe redesigned (`p9_pot_topple.py`, `docs/results/p9_pot_topple.json`).**
Why it broke: with its true shell centre of mass the pot's base leverage is b/h ≈ 1.09, so
it can only tip once tan(tilt) > 1.09 -- but at the probe's grip (body μ 1.30, μ_eff 0.76)
it starts SLIDING ~10° earlier; sliding won and 23/36 conditions never toppled. The old
vertex-mean centre of mass (b/h 0.74) had let it tip inside the sliding window: the probe
worked for the wrong reason. Toppling must beat sliding: μ_eff > b/h, i.e. body μ > ~2.6.
- **Design: body μ 4.0 (μ_eff ≈ 1.34), tilts 40/42/44/46°, tilt rig incl. α = 0.4,
  time-to-topple read every 2 ms off the simulator** (the 1/24 s frame grid flattened every
  condition above 44° to 0.17–0.21 s; with the video readout retired there is no frame
  grid to respect).
- **Sim-side gate**: ρ(offset, time) = −1.00 at every tilt 38–50°; range 2.75× / 2.29× /
  2.09× / 1.89× at the chosen tilts; no "never toppled"; ≤6 cm of slide before tipping.
  Below 38° negative offsets never topple; above 46° the range shrinks toward 1.6×.
- **Rule 1 passes**: time-to-topple moves 2.19× with the ±18 mm offset and 1.00× with mass
  (0.1–1.6 kg), 1.02× with restitution α (0.05–1.2), 1.01× with friction above the grip
  threshold (μ 3–15).
- **Held-out offset recovered to 0.61 mm** (one tilt) / 0.71 mm (tilts joined; max 1.0 mm)
  on a 4.5 mm grid -- deterministic, so this is interpolation error, not noise; joining
  tilts cannot help without noise to average.
- **Caveat for a physical version**: μ_eff 1.34 is not an ordinary surface -- a real rig
  would need a high-grip (rubber) ramp, and its measured μ would replace the assumed 4.0.

**Restitution from the real drops: FAILED, not reported (`p3_restitution.py`).** 30 fps
DSLR with long exposure: the fall is ~5 blurred frames. The scale check fails (g spans
5,750–15,568 px/s², a 2.7× spread on a fixed camera), the two routes (height vs time to
apex) correlate at only 0.38, and the physics is inverted -- a wooden block "rebounds"
more on foam (0.46) than on a table (0.24), because on a soft surface the height route
measures the foam springing back. Restitution needs a higher frame rate than Physics 101.

### THE HELD-OUT TEST: parameters for an unseen real object (`e1_heldout.py`, `docs/results/e1_heldout.*`)

The claim everything rests on -- parameters assigned by material make a never-seen object
move like the real thing -- tested on Physics 101 with each real object held out in turn,
its REAL tracked slides as the judge. Only release time and offset are aligned, never the
acceleration. (Newton reproduces the sliding law to <1 mm on these clips, p4.)

| friction predictor, held-out object (70 objects, 359 trials, 20°) | traj. error median / p90 | stays-or-slides (198 cases) |
|---|---|---|
| oracle: the object's own other trials | 1.8 / 6.2 mm | — |
| **material, median over OTHER objects** | **2.6 / 12.4 mm** | **94%** |
| **pipeline: LLM names the material from the photo (80% right), then the same** | **2.7 / 16.7 mm** | **92%** |
| dataset-wide median, any material | 4.4 / 15.4 mm | 80% |
| LLM's direct number, told the exact surface, mass and volume | 7.5 / 44.9 mm | 85% |
| LLM's generic 0.40 (f14; asked about wood, so unfair here) | 86.9 / 104 mm | 62% |

Paired per object (sign test): material beats the dataset median on 52/70 (p = 6e-5) and
the LLM's direct number on 55/67 (p = 1e-7); even the dataset median beats the LLM's
number, 46/67 (p = 0.003); pipeline vs material is indistinguishable (2/10, p = 0.11).
**Mass** of the held-out object from material density × its real volume: within 25% for
63% of 101 objects; **build-aware (areal for the hollow cardboard) 69%, median error
×1.11**; the LLM's direct guess (given the photo and volume) 46%, ×1.30.

So: **an LLM is worth using to NAME the material (80% right, costing ~0.1 mm), and not
worth using for the NUMBER** -- told the surface, mass and volume it still does 3× worse
than a measured material table and worse than a dataset-wide median.
Limits of this test: friction is against the same board for train and test (the asset
transfer additionally needs the stated table anchor); a sliding block's law is
shape-independent, so shape effects and centre of mass are not tested here.

### Gradients through the validated simulator (`g0_grad_check.py`)

`MeshProbeScene` reuses its force / torque / world-vertex buffers every step -- right for a
forward rollout, wrong for a Warp tape (the adjoint of a nonlinear kernel needs that step's
own forward values). A per-step-buffer variant with `requires_grad` (`DiffMeshScene`)
**reproduces the validated rollout exactly and gives d(slide)/d(μ_body) = −0.5448 against
−0.5453 by central finite differences** (cube on a 20° plane, 0.5 s, 1200 steps, 386
vertices; 0.4 s forward + backward). A first run read "exactly half": the finite difference
had varied the ground friction along with the body's, and the pair rule √(μ_b·μ_g) halves
the sensitivity at μ_b = μ_g -- the gradient was right, the test was not. So the exact-mesh
contact simulator that reproduces real footage to <1 mm is differentiable today, through
plane contact.

**Body–body contact (`g0b_grad_pair.py`, `g0c_grad_diag.py`, `g0d_grad_free.py`): partly.**
- The mesh-query adjoint (`wp.mesh_query_point_sign_normal` → `mesh_eval_position`) is
  exact away from corners (Test A); at a corner the tape picks one face where the finite
  difference averages three -- a true non-differentiability, not a bug.
- A body's motion w.r.t. its OWN parameters in a two-body scene: tape −0.0456 vs FD
  −0.0434 (5%).
- One body's motion w.r.t. the OTHER body's parameter, through contact: brief collisions
  in free space are right to within 1–2× (glancing hit at 3 cm offset: +0.0007 vs +0.0007;
  head-on damping −3.0e−5 vs −1.8e−5); but under SUSTAINED pushing (B sliding into A and
  shoving it for 0.17 s, soft contact, deterministic forward) the tape gives
  d(A)/dμ_B = −0.032 against FD −0.385 while d(B)/dμ_B comes out −0.666 against −0.315 --
  **the total (−0.70) is right, its split between the bodies is not**: the adjoint
  under-transmits the push through the contact. At the stiff tilt-rig setting the forward
  itself is nondeterministic at 1.5 mm (atomic-add order amplified by the collision), the
  finite difference is step-size dependent, and one adjoint component explodes (5e4).
- **So: per-object gradients against a surface are usable now; gradients through
  sustained multi-body contact are not, and fixing them (adjoint accuracy through stiff
  penalty contact; cf. "soft gradients for hard contacts", 2025) is research, not setup.**
  Any direction that needs gradients through long multi-body interactions must budget
  for it; directions that identify one object against known surfaces can proceed today.

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
- **`MeshProbeScene`'s centre of mass is the VERTEX MEAN, which is neither physical model.**
  Measured against an area-weighted shell (right for thin-walled hollow objects) and a
  voxel-filled solid (right for solid ones), base-edge leverage b/h differs by up to 2.2×:
  sorter 1.54 (vertex mean) vs 3.34 (shell), pot 0.74 vs 1.09, bowl 1.57 vs 2.55, books
  2.62 vs 2.22 (solid). Vertex density follows modelling detail, not mass, so every tipping
  threshold in the simulator inherits this bias. Uniform density is itself an assumption
  (`scripts/f14_llm_outcomes.py::geometry` computes both shell and solid).
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
- **The camera pull used to CLIP must match the pull used to RENDER.** `do_clip` reads
  `CAM_PULL` from the environment and falls back to `V1_KILLTEST.CAM_PULL = 0.65`; rendering
  at 1.20 and clipping without the variable computes the crop box against the wrong camera
  and the object leaves frame — worst containment 41.3% against a correct 99.5%. This is the
  second time bad framing has inverted a result. Containment is now checked per clip in
  `o4_joint.containment()` and out-of-frame clips are dropped, not scored.
- **A readout picked by reading class names is worse than one fitted over the vocabulary.**
  The SSv2 templates whose wording matches the probe track the observable at ρ = +0.343 /
  −0.462; a ridge over all 174 log-probabilities cuts the read error by a third. Name
  matching is a hypothesis about what the network encodes, not a measurement of it.
- **Coarsening a timestep to speed a render changes results.** Balance went 7/7 → 4/7 when
  substeps were cut 60 → 24 for a render pass.
- **Inertia carried mass twice.** `set_masses` stored `G = G_unit · m` while `integrate_6dof`
  forms `Iw = R (m·G) Rᵀ`, so inertia was `m²·G_unit` — for a 0.1 kg body, ten times too
  little, and ten times the angular acceleration. Fixed; it also improved the tilt probe
  (residual 0.79° → 0.40°).
- **The ground plane can now be TILTED, and that is the preferred incline.**
  `ground_contact_mesh` took a horizontal z-test (`pen = ground_z - world.z`) with an
  axis-aligned bounds rectangle, so the only way to make an incline was to rotate gravity.
  It now takes a unit normal `gn`, an in-plane tangent `gt` and a point `gp`; passing
  (0,0,1)/(1,0,0)/(0,0,ground_z) reproduces the old plane exactly. `MeshProbeScene(tilt_deg=,
  pivot=)` wires it up. Why switch:
  - **One frame for sim and render.** Rotating gravity split the world in two — a level
    simulator frame and a tilted render frame — and produced **four** wrong-frame bugs: the
    stack probe subtracting in world coordinates, the crop box, the ramp normal, and object
    poses handed to the renderer untransformed so the object visibly floated free of the ramp.
  - **No rotating-frame approximation.** A *ramping* gravity omits the centrifugal/Euler terms
    a real rotating table has (0.05–0.3% of g at our rates, ~0.8% at 60°). A static tilt is
    exactly equivalent, with nothing omitted.
  - **The rate dimension disappears.** At fixed μ and fixed final angle, ramp rate alone moved
    slide 374 → 41 cm (μ = 0.15) and 134 → 12 cm (μ = 0.30) — a 9–11× lever, and a nuisance
    unless controlled. A fixed angle has no such dimension.
  - **More leverage.** 57× slide range over a 4.4× μ sweep, against 13–22× for the tilting ramp.
  **Validated against the analytic threshold**: sliding at three tilts collapses onto one curve
  in μ/μ* where μ* = tan²(tilt)/μ_table — at μ/μ* = 1.0 the slide is 62/72/82 cm for
  18/24/30°, at 1.5 it is 15.5/17.5/13.8, at 2.2 it is 8.5/13.0/23.3.
- **On a real tilted plane the body must be rotated to the plane, and seated along its
  normal.** With gravity rotated, an identity-orientation body was automatically flat on the
  ramp. On a tilted plane the same body starts tipped by the tilt angle and rests on the
  *edge* of its base: only a sliver of weighted contact area engages, effective stiffness
  collapses, and a 0.4 kg bowl sank **14.5 mm**, was ejected, and slid 4 m at a friction
  **2.2× above** its slip threshold. Both steps are inseparable and live in
  `MeshProbeScene.seat_on_plane()` — never re-derive them in a script.
- **There is no static friction.** `ft = -mue*fn*vt/(|vt| + V_EPS)` can resist motion but
  cannot stick, so a tilted probe has a **creep floor**: 8–23 cm of slide even at 2.2× the
  slip threshold. Read a tilted probe as a magnitude against that floor, not as "did it
  move at all". **Revised 2026-10-05: most of that floor was the 2 mm seating drop**, now
  removed; ~9 mm/s of genuine no-static-friction creep remains.
- **Rendering: per-frame FIXED cost dominates sampling 2.5:1.** Measured at 544×448/48
  samples: startup 0.8 s, then **2.01 s per frame**, of which only **0.58 s** scales with
  pixels — the rest is CPU-side scene update and BVH rebuild, because objects move every
  frame. So halving samples saves 14%, and the real lever is **fewer render passes**: N
  probe cells in one big frame cost `1.43 + 0.58N`, i.e. 2.1× faster at 4 cells and 2.6× at
  8. The constraint is that each cell sees the camera differently, so **one cell per asset**
  (constant per asset, absorbed into its own calibration) — never different θ of one asset
  in different cells.
- **The simulator tilts GRAVITY; the renderer tilts the TABLE.** `ramp_gravity_seq` keeps
  the ground plane flat and rotates g, so in simulator coordinates the ramp normal is
  **(0, 0, 1)** — anything comparing a body's orientation to "the ramp" must use that, not a
  rotated normal. `o7_sensitivity.tip_angle` compared against a rotated normal *and* got its
  sign backwards, measuring roughly ramp + body tilt: 39.6° at a 27° ramp against a true
  22.2°, so every clip of a merely sliding bowl flagged as toppling, including one that moved
  3 cm. Fixed, the vase's CoM range *improves* 4.61× → **6.50×** and stays exactly immune to
  mass/μ/cd. Third appearance of a wrong-frame error, after the stack probe and the crop box.
- **The tip probe needs an object that topples, and the matrix now says so.** On the bowl the
  corrected observable is a one-sided cliff — 60/60/56/3.1/2.8° across CoM −30…+30 mm, i.e.
  blind to an uphill offset — and it picks up μ at 2.18× non-monotonically, because whether
  the bowl topples depends on whether it slid first. On the vase the μ cell is 1.00×. This is
  what made the joint fit recover target A's CoM (+12 mm, on the informative side) and miss
  target B's (−8 mm, in the flat region).
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

3. **V1 — KILL TEST: FAIL, final.** Surprise sits at **chance on categorically impossible
   motion** in our renders — 47% (wooden bowl), 47% (brass pot, held out), 40% (2.1 ViT-g)
   over 30 matched pairs. This is the corrected verdict after two defects were fixed: the
   bowl spawning 11.8 mm inside the table (so the mass "null control" was measuring a bowl
   sinking), and a context leak in which context tokens were gathered from a full-clip
   encode and had therefore already attended to the region being predicted. Both were found
   by looking at the clips and at the reference implementation. Consistent with **IntPhys 2**,
   which reports models at chance on realistic scenes; the famous ~98% is IntPhys **1**,
   coloured shapes on plain backgrounds. One thing works: **localisation** — sliding a block
   mask through time recovers *when* the clip went wrong, median error 1.0 slot, 62% within
   4 frames against ~16% chance, exact on all six teleport sources. Write-up `docs/V_TRACK.html`.
   Do not build V2/V3 on surprise.

3b. **O-track — outcome classification. The measurement half PASSES.** A frozen V-JEPA 2
   ViT-g/384 with the released SSv2 head names the *outcome* of a motion we made, over 174
   templates. Every single-sweep gate (G-A, G-A2, G-B) failed, and **those were the wrong
   test**: within one ramp angle, friction and slide distance are the same column of numbers
   wearing two labels, so no single sweep can separate them. Ramp angle and rate are
   conditions we *set*. Constraining several at once:

   | held-out μ recovered from | bowl (74 clips, 7 angles) | pot (33 clips, 3 angles) |
   |---|---|---|
   | one condition alone | 29% median error | 20% |
   | conditions joined | **16%** | **15%** |
   | flat prior on [0.10, 1.20] | 86% | 86% |

   Two corrections were needed first, both worth keeping:
   - **Read the whole distribution, not classes picked by name.** The two templates chosen
     by their wording track true slide at ρ = +0.343 / −0.462; the best available reach
     +0.660 / −0.880 and 92 of 174 carry |ρ| > 0.5. A cross-validated ridge over all 174
     log-probabilities cuts the read error by a third (bowl 0.294 → 0.220, pot 0.105 → 0.057).
   - **Bracket the parameter.** Above μ ≈ 0.7 the bowl barely moves at 21–27°, so those
     clips carry no information and the estimate regresses to the middle. Adding 33/36°
     (202 → 9 cm span) helps but does not fix it: the high-μ band stays at 61% (bowl) /
     144% (pot). That is the tilt probe's identifiability limit, not a readout defect.

   **The blind fit works.** Three frictions off the calibration grid (0.175 / 0.245 / 0.390),
   simulated, rendered and scored separately, recovered with nothing in the calibration
   having seen them:

   | blind μ recovered | brass pot | wooden bowl |
   |---|---|---|
   | median error | **3%** | 23% |
   | truth inside the reported error bar | 3/3 | 1/3 |

   The fit needs no external ground truth: the readout calibrates against the **simulator's
   own** slide, which we always have for our own rollout. `μ̂ = argmin_μ Σ_t ‖readout(clip_t)
   − log slide_sim(μ,t)‖²` — fitting all conditions at once rather than inverting each and
   averaging, which also removes the monotone-prefix truncation that inversion required.
   Uncertainty propagates the readout's LOO residual through the local leverage and is
   honest on the pot: the one hard case (μ = 0.390, where the pot's slide flattens) reports
   ±51% rather than a confident wrong answer. The bowl's bar is too tight because its
   readout is 4× noisier (LOO sd 0.211 vs 0.057) and under-reads slide.
   Baseline for the same question on the same probe: the judge gives ρ = −0.32.
   Scripts `o5_fit.py` (uncalibrated objective, 32%/28% — worse) · `o6_blind.py`.

   **Two framing defects found and fixed, both of the kind already on this list.** The crop
   box was computed **per lab**, so `ga` was 342 px and `ga3` 448 px at a different offset —
   framing varied across the very conditions being compared. Forced to one box
   (`CROP_BOX` in `o2_boundary.do_clip`); the bowl moved 16% → 18%, the pot was single-lab
   and unchanged. And a bash timeout SIGTERMs the whole process group, killing backgrounded
   Blender workers mid-scene — launch long renders with `setsid nohup`.

   **Ten assets, per-asset fits (`o11_assets.py`, `docs/results/o11_assets.json`).**
   One friction probe (3 ramp angles × 9 μ, 27 clips each), one crop box and one camera pull
   across every asset, every asset scaled to a common 0.19 m width and given identical mass.

   **Per-asset readouts work on all seven valid assets** — each calibrated on its own sweep
   against the simulator's own slide, leave-one-out: pot 1.10×, books 1.16×, teapot 1.37×,
   bowl 1.75×, box 2.01×, sorter 2.92×, shark 3.46× (vase 1.15× but excluded, see below).
   ρ(read, true) is 0.67–0.97. No per-asset tuning was needed.

   **Three assets of ten cannot support the friction probe, and the SIM tells you which.**
   The vase topples on 12/27 clips, the lion and triceratops on 27/27 — they read a topple,
   not a slide. The geometric estimate `atan(halfwidth/com_z)` **under-predicts** toppling
   for irregular shapes (it gave the lion 59° and the dino 41°), so it is a hint, not a
   filter. The **sim-side topple count is the filter and it costs no renders** — run the
   sweep, count topples, render only what slides. The dino also confirms it another way: its
   slide spans 294–317 cm with a μ leverage of exactly 0.00.

   **The offset is NOT predictable from geometry, and the first look said it was.** Over 42
   ordered asset pairs the scale offset runs 1.02× to 54.1×, median 2.21×. At four assets
   `footprint` correlated with the per-asset level at ρ = 1.000 — but an exact permutation
   test over all orderings, taking the best of six features each time, puts that at **25% by
   chance**. At seven assets the best feature falls to ρ = 0.429 at **p = 0.78**. A cheap
   geometric shortcut for calibrating a new asset does not exist on this evidence, so
   **per-asset calibration is the cost, ~27–100 clips an asset.**

   **What does survive is the ordering.** Borrowing another asset's readout: Spearman median
   **+0.804**, and **39/42** pairs above +0.5, against only 25/42 pairs beating the mean on
   R². Removing one constant per pair recovers **38/42** — the error is a single scale factor,
   not scrambled structure. Since a fit compares candidate θ for one fixed object and never
   needs a calibrated distance, ordering is what a fit actually consumes.

   **Where it stops: the reading does not transfer across objects, only the ordering does.**
   Bowl-calibrated readout on the pot gives R² = −4.22 — worse than the mean. Almost all of
   it is one constant: the readout is off by a factor of **3.4** on the new object. Remove
   that single offset and R² = +0.61; Spearman survives at +0.734. Survivable for a fit,
   which compares candidate θ for one fixed object and never needs a calibrated distance.
   **Not** survivable for comparing a render against real footage — and that comparison is
   the entire reason for using a model that has watched the real world. Write-up
   `docs/O_TRACK.html`, scripts `o0_gate.py` · `o2*_*.py` · `o4_joint.py`.

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

