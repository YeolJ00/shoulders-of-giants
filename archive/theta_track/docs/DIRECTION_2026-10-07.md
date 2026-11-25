# Research direction review, 2026-10-07 (SUPERSEDED 2026-10-16 -- see docs/DIRECTION.md)

Record of the direction review run on 2026-10-07: six literature scouts (91 works), five
proposers with distinct stances (10 proposals), three adversarial reviewers per proposal
(30 verdicts: novelty, feasibility, impact), a ranked synthesis, a completeness critic, and
a second round of three gap scouts (post-cutoff scoop sweep, consumer/impact evidence,
engine/contact-law). Full structured output: `docs/results/direction_workflow.json`.
Scripts referenced: `scripts/e1_heldout.py`, `scripts/g0_grad_check.py`,
`scripts/g0b_grad_pair.py`, `scripts/g0c_grad_diag.py`, `scripts/g0d_grad_free.py`.

## The question asked

Find an actionable direction that (a) uses a differentiable physics engine, (b) yields a
practical system for robots or simulations, (c) possibly at very large scale beyond rooms
("interactable worlds at scale") -- and establish whether the project is reinventing the
wheel.

## The verdict in one paragraph

The method the project converged on -- a vision model names the material, a table supplies
density and friction, the simulator integrates mass, centre of mass and inertia over the
shape -- is the field's standard (PhysX-3D/-Anything/-Omni, UniPhysGen, VoMP, NeRF2Physics,
Asset2Sim, Kaedim). Identifying physics from uninstrumented consumer video is also done
(SMASH 2016, Vid2Param 2020, Hofherr 2023, DiffSDFSim 2021, Delta-ynamics 2026), and
held-out-object evaluation on ramp footage goes back to Physics 101 (2016) and Galileo
(2015). Every one of the ten proposals was refuted on novelty; the best total score was
4.7/10. What survives is narrower than the ask: (1) measured truth at a scale nobody has
published for everyday objects (friction, restitution at a stated impact speed, centre of
mass by build; ~40 weighed objects x 3 surfaces x 3 speeds, with repeat-trial uncertainty
and invariant gates), (2) a benchmark that scores every parameter-assignment route the
field ships (released generators, LLM numbers, handbook tables, library defaults) against
the real motion of held-out objects, in the unit a maintainer acts on -- the fraction of
outcomes (stays / slides / topples / bounces) that flip -- which no source reports, and
(3) one gated engine claim: FD-validated gradients through sustained frictional mesh--mesh
contact, which no shipped differentiable engine provides (Newton 1.4's differentiable
contacts freeze the narrow phase; DiffMJX covers primitive pairs; Nimble freezes mesh
normals and has no friction gradient; Ostrich/Jade code unreleased) and where this
project's pair kernel has a measured, localisable defect.

## Ranking (final round)

| total | proposal (merged surviving form) | novelty | feasibility | impact | scale | diff. physics |
|---|---|---|---|---|---|---|
| 4.72 | Phone instrument: per-object (mu, e at a stated speed, CoM-by-build) through the validated plane-contact tape | 4 | 6.5 | 4 | 3.5 | 5.5 |
| 4.55 | ContactAtlas-240: multi-object tagged tilt-board + drop station, object x surface atlas | 4 | 6 | 4 | 4 | 4.5 |
| 4.25 | Measured table + assignment benchmark/audit on Physics 101 (no capture, no gradients) | 4.5 | 6 | 4 | 3 | 1 |
| 3.95 | Impact atlas: restitution only, 240 fps drops, audio second route | 3.5 | 6.5 | 2.5 | 3 | 3.5 |
| 3.88 | Phone-to-twin, scene level: every movable object on one tape, persistent world API | 3.5 | 4 | 3.5 | 4.5 | 5 |
| 3.85 | M2P: photo -> (mu, e, build) model trained through the simulator on real motion | 3 | 6 | 3 | 2.5 | 4.5 |
| 3.48 | PhysReady: measured material layer + 200--1000-object Newton street scene | 3 | 4 | 3.5 | 5 | 2 |

The scores are low on purpose: the reviewers compared *surviving forms* after refutation.
The first-round recommendation (before the gap scouts) was the same instrument-plus-
benchmark; the gap round added the engine claim and the consumer-side facts below.

## Recommendation

**The instrument, the benchmark, and one engine claim.** Phone-measured (mu, e at a stated
impact speed, CoM-by-build) of everyday objects on everyday surfaces through the exact-mesh
simulator, scored on held-out real motion as outcome flips for every assignment route the
field ships, with the mesh--mesh contact adjoint as the single, gated differentiable-physics
contribution.

Thesis: the physical parameters of simulation assets have to be measured, not asked for;
every external model this project tested returns a generic prior, and independent 2026
results say the same (PhysVid: V-JEPA-2 friction r 0.21--0.28; PokeWorld: mass R^2
0.21--0.25; WRBench: 0/23 world models persist state; GAUGE/WorldBench: right law form,
wrong accelerations). The cheapest validated instrument is a 240 fps phone plus this
project's simulator (real slides reproduced to 0.4--1.0 mm; plane-contact adjoint exact).

### What is built (four faces of one thing)

- **Instrument.** Phone on a tripod at 240 fps, inclinometer angle, the object's own
  measured edge for scale, a reference board + laminate + concrete paver, kitchen-scale
  mass, a reference pair re-slid every session (MIT Push: surfaces drift 11--22% with
  wear). CoTracker3 tracks -> (mu_pair, alpha at a stated impact speed, CoM offset where
  the object topples) through `DiffMeshScene`, with per-clip nuisances (release time,
  offset, initial pose), tape gradients where they pass a per-class FD gate, FD/CEM where
  they do not, and a per-clip Fisher matrix stating what the clip identifies. Slides and
  sphere rebounds are closed form and read as such.
- **Library and delivery path**, authored in the consumers' own conventions: rows with n,
  IQR and provenance (measured / inferred / default); friction as sqrt(mu) under PhysX
  "multiply" or explicit MuJoCo `<pair>` elements, because no engine combines by geometric
  mean; restitution as a scalar at a stated reference speed, because no consumer ingests
  e(v); every exported value re-simulated in Newton-MuJoCo and PhysX with the per-engine
  disagreement reported (Acosta et al. 2022 identified one cube at mu 0.10 / 0.22 / 0.36
  across three engines).
- **Benchmark.** The `e1` harness generalised to a scorer for any f(image, mesh) ->
  (mu, rho, e): re-simulated trajectory error, outcome accuracy and outcome-flip rate on
  held-out real objects and held-out surfaces; routes = own-trials oracle, measured
  table, LLM-name + table, LLM direct number, handbook table by material pair (never run),
  VLM-range midpoints in the generators' prompt formats, PhysX-Anything density -> mass,
  PhysX-Anything -> UniPhysGen friction, BEHAVIOR-1K lookup, Isaac Lab / SimReady defaults.
- **Engine track, gated.** Week 1: the FD regime map of the pair adjoint (sustained push,
  glancing 3/5 cm, head-on strike, stacked settle; both rigs; vs contact duration and
  approach angle), the per-launch isolation of the lost term, and the symmetric-pair
  invariant (two identical bodies, equal mu, no plane: d/dmu_A must equal d/dmu_B).
  Weeks 2--4: the fix attempt (templates: DiffMJX's contacts-from-distance, Ostrich's IFT
  adjoint; fallback: Suh et al.'s bundled gradients). Only if sustained-contact gradients
  pass FD within 10--20% do strike / stack / push clips enter the capture.

### The honest role of differentiable physics

- Tier 1, certified: one body on a plane, friction (g0: tape -0.5448 vs FD -0.5453) and
  damping through a plane impact (reviewers' runs: tape/FD 0.998--1.05 flat landings,
  0.97--1.17 edge landings at >= 420 substeps; sign flips at 100 substeps). Load-bearing
  uses: per-clip inversion with 6--10 unknowns including nuisances, joint fits with shared
  per-session / per-material parameters over hundreds of clips, Fisher identifiability.
  Caveat: FD at 6--10 parameters costs seconds per clip, so here the tape is convenience
  and bookkeeping, not capability.
- Tier 2, gated: gradients through sustained frictional mesh--mesh contact (the hole).
  Measured defect (g0c): FD d(x_A)/d(mu_B) = -0.385, tape -0.032, with the total over both
  bodies right (-0.70), so the adjoint under-transmits the push; brief collisions are within
  1--2x (g0d); at the stiff setting one adjoint component explodes and the forward itself
  is nondeterministic at 1.5 mm.
- Tier 3, never: a sliding block (closed form), a sphere's rebound (two closed-form routes),
  mass (a scale).
- Carried caveats: the forward rotational response to corner impacts is not converged in
  substeps (angular velocity -4.29 / -3.98 / -3.81 rad/s at 60 / 120 / 240) and no
  convergence criterion exists for polyhedral corner impacts under penalty contact (DEM
  rules are sphere-only), so 6-DoF fits are restricted to clip classes shown to converge;
  the velocity exponent in e = 1 - alpha v^p is a kernel choice, not a measurement
  (Hunt--Crossley linear, Kuwabara--Kono v^1/5 for a different kernel, PFC constant), so
  restitution is reported at a stated impact speed as DEM practice does.

### First experiment (week 1, on disk, zero capture)

(A) Does measuring beat looking up? Add a handbook-friction-table route (material pair
named by the LLM's existing labels) to `e1_heldout.py` and score it on the 70 held-out
objects / 359 trials against the measured table's 2.55 mm and the dataset median's
4.40 mm, with a paired sign test. If p > 0.05 and medians within 0.5 mm, the friction
half of the capture is cancelled before a clip is shot.
(B) Where does the pair adjoint fail, and is it a bug? The regime map and the per-launch
isolation above; the numbers: per-launch d(force on A)/d(mu_B) ratio to FD, the
symmetric-pair asymmetry (must be 1.00), the full ratio table by regime.
(C) Record the impact-adjoint numbers that exist only in reviewer scratchpads into
`docs/results` (`scripts/g1_impact_grad.py`).
The first capture experiment (weeks 2--3): the 10-object instrument gate -- sphere e repeat
IQR <= 0.05, two-route agreement <= 0.03, free-fall g spread <= 5%, e span >= 0.3 across
materials, two-angle mu agreement <= 0.03, the Physics 101 bridge ratios.

### Kill tests, pre-registered

0. Engine: if within three weeks the pair adjoint under sustained contact is not within
   10--20% of FD and passing the symmetric-pair invariant, or the lost term is the
   contact-assignment non-differentiability rather than a bug, the engine claim is dropped,
   the regime map is published as a finding, and "differentiable" leaves the title.
1. Instrument: sphere repeat IQR of e > 0.10, routes > 0.05 apart, or free-fall g spread
   > 10% within a session => a phone at 240 fps is not a restitution instrument; the
   direction collapses to the runner-up.
2. Value of measuring: handbook table matches the measured table on held-out Physics 101
   motion (p > 0.05, within 0.5 mm) => measured friction buys nothing over a handbook.
3. Transfer: fitted values are exported only if held-out-trial-type prediction (fit drops,
   predict slides; fit slides, predict stay/slide at the second tilt) lands within 1.5x the
   object's trial spread for >= 2/3 of objects.
4. Differentiable claim per clip class: tape within 20% of FD at the fitting substep count
   and a substep-converged forward observable; otherwise FD/CEM and no "differentiable".
5. Field-level null: if replacing VLM/handbook values with measured values flips < ~10% of
   open-loop push/drop/topple outcomes over 50--200 props, parameter accuracy does not
   matter at the outcome level; publish the negative, do not scale the library.
6. Engine portability: exported values reproduce the measured drops and slides in
   Newton-MuJoCo / PhysX no better than defaults => the table is engine-specific-effective.

### 90-day plan

Weeks 1 (on disk) -> 2--3 (10-object instrument gate; pair-adjoint fix attempt in parallel,
decision day 28) -> 3--4 (contact-law check: alpha vs drop height and object size) ->
5--8 (~40 weighed objects x 3 surfaces, ~1,000 clips; +~150 strike/stack/push clips only if
the engine gate passed; PhysX-Anything and UniPhysGen environments installed in parallel)
-> 9--10 (library v1 with IQR and provenance, held-out-object / -surface / -trial-type
tests, USD/MJCF export, per-engine re-simulation) -> 11--12 (the benchmark and the
outcome-flip audit on open-loop primitives) -> 13 (write-up with every kill-test number
stated whichever way it fell).

## Runner-up

`e1`-bench on disk: score every assignment route (released generators, LLM numbers,
handbook tables, VLM-range midpoints, library defaults) on Physics 101 held-out real
motion, with outcome accuracy as the unit. Cheapest and most certain; its novelty claim
was the best-defended. It cannot be the direction because Physics 101's 20 deg board is
flat above mu = tan 20 deg = 0.364 (every route emitting 0.37--0.60 scores the identical
87.8 mm), so generic routes cannot be ranked without a steeper, self-captured ramp; it has
no restitution or centre-of-mass column; and it uses no differentiable physics. Prefer it
if no 240 fps phone or capture time is available, if the day-21 instrument gate fails, or
if a workshop-length audit in 4--6 weeks is wanted.

## Parts of the ask the evidence says to drop or defer

- **City scale / "beyond rooms" / "interactable worlds at scale": drop as a target.** No
  street- or city-scale pipeline validates any physics (UrbanVerse: GPT-4.1 mass vs spec
  sheets, friction unchecked; URBAN-SIM/MetaUrban: per-texture friction labels; HUGSIM/
  OmniRe/WOSAC: no contact physics); "interaction" at that scale means collision avoidance.
  The physics content of a city is O(materials x surfaces), which the recommendation
  measures and exports; the geometry is a solved, separate problem. The critic's correction
  stands: the honest blocker is validation and headcount, not the missing broadphase (a
  day of Warp work). "Beyond rooms" survives as outdoor surfaces measured as rows, and the
  one scale experiment reachable in 90 days is delivering the measured material layer into
  an existing street simulator with a sensitivity test.
- **Neural world models as the world or as supervision: drop.** Independently reproduced
  negatives (PhysVid, PokeWorld, ACWM-Phys, WRBench, GAUGE, WorldBench). Keep one baseline
  row.
- **Robots: defer, with the test specified.** No robot; SIMPLER's closed-loop evaluation is
  reportedly insensitive to an 8x mass x 8x friction sweep, while open-loop tasks are
  parameter-set (BayesSim Fetch Slide, DROPO puck, ASID). The robot-facing deliverable is
  USD/MJCF assets with measured values and provenance plus the in-sim open-loop
  outcome-flip test; a policy claim needs a partner lab.
- **"Differentiable physics" as the headline: demote** to a per-clip-class role with one
  gated engine claim.
- **"First / nobody has done X": drop throughout** (Physics 101, Galileo, SMASH, Vid2Param,
  Hofherr, Delta-ynamics, DiffSDFSim, NeRF2Physics/Asset2Sim/UniPhysGen/Kaedim, GAUGE,
  MIT Push, Omnipush, DROPO, Acosta 2022).
- **A measured table as a universal physics layer: drop that framing**; deliver
  engine-authored values with a per-engine portability check.
- **A learned appearance-to-physics model: drop** (Physics 101 built it in 2016; the
  no-model table already sits within 0.8 mm of the own-trials oracle).
- **Persistent deterministic world API, soft bodies/cloth: drop.**

## What to stop doing

- Any use of generative/predictive video models or LLM numbers as supervision (keep as
  baseline rows only).
- The render-and-read-off-video O-track readouts (retired; do not re-validate).
- Assuming body--body contact gradients either way: a bounded three-week diagnosis with a
  pre-registered FD gate, and the number decides.
- Persistent-world / 1,000-object street deliverables; M2P; soft-body probes; fitting
  more Physics 101 slides for their own sake; exporting physics consumers cannot ingest
  (geometric-mean pair rule, e(v), unstated engine, invented static = 1.2 x dynamic);
  "first" claims; closed-loop policy evaluation as the impact test.

## What would change the recommendation

(a) the handbook route matches the measured table -> narrow to restitution + CoM + engine
+ benchmark; (b) the day-21 instrument gate fails -> runner-up; (c) the pair adjoint passes
its FD gate -> the engine result becomes the method contribution and the measurement its
validation; (d) a robot lab swaps in the exported values on open-loop tasks -> a robot
claim; (e) a 2025--26 paper already provides a phone-video multi-material multi-surface
(mu, e, CoM) table with held-out-motion validation -> narrow to the benchmark; (f) the
outcome-flip audit returns < ~10% -> publish the negative, do not scale the library.

## Caveats on the review itself (from the completeness critic)

- Three load-bearing numbers came from reviewers running experiments in their own
  sessions, not from `docs/results`: the impact-adjoint tape-vs-FD agreement, the
  corner-impact non-convergence, and the 20 deg-board flatness. Week 1 records them.
- Several refuting citations are post-cutoff and were reached by scouts, not independently
  by the critic (GAUGE 2608.05948, Kaedim 2610.00731, Wang & Wang 2604.24188, PhysVid
  2510.02311, 2606.20712, WRBench, PokeWorld, ACWM-Phys); the Jul--Oct 2026 proceedings
  sweep was only partly done. A targeted day-1 search is part of week 1.
- Camera metrology (true 240 fps, rolling-shutter skew, exposure) was not scouted and is
  the whole instrument gate's precondition.
- Phone-native restitution tools (phyphox acoustic collision, OSP Tracker) have published
  repeatability numbers the gate thresholds should be set against.

## Open questions carried forward

Is the pair-adjoint defect a bug or the contact-assignment non-differentiability? Does a
handbook table already match the measured one? Are measured values engine-portable at all?
Is restitution a material or a pair/surface property (Bounce and Learn says surface)? Is
e(v) monotone over 1.4--2.8 m/s for everyday materials? Does parameter accuracy change
open-loop outcomes? Is the corner-impact forward response convergent at any affordable
substep count? Is GAUGE's data reachable? How large is within-session friction drift on
the user's surfaces? Can the user's board be bridged to the Physics 101 rows (p1's size
dependence says maybe not)? What are Physics 101's redistribution terms?
