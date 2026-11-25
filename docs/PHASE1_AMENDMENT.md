# Phase 1 amendment: identifiability-first data design

Amends `docs/DIRECTION.md` sections *Slots*, *Actions*, *Episodes*, *Real data* and the
Phase 1 gate. Motivation: four defects that are cheap to fix before the data engine exists and
expensive after 50k episodes do. All four are the theta track's lessons re-applied to the
dataset: a parameter nobody's interaction can observe is not a model failure (rules 1, 1b).

## 1. The principle: every parameter slot needs an interaction that makes it observable

Claim 2 (in-context material inference) is only testable for a parameter if the training
distribution contains motion whose shape depends on that parameter. Free flight is mass-free
(equivalence principle); a position-controlled gripper imposes the object's acceleration
regardless of mass; sliding deceleration is mu*g, mass-free. With the current action set,
**mass is unobservable** and claim 2 would return "not inferred" for the wrong reason.

Identifiability map (which interaction exposes which parameter):

| parameter | observable through | requires |
|---|---|---|
| friction mu | sliding deceleration after a push; incline slip | nothing extra (already covered) |
| restitution e | bounce after drop/strike; velocity jump at impact | drops, strikes (covered) |
| **mass** | acceleration under a **known** force; momentum exchange in dynamic-dynamic collisions; load on a hinged part or on a stack | **gripper contact force recorded as a given slot**; collisions; loading |
| **centre of mass / inertia** | topple threshold under off-centre push; rocking frequency after release; which way it falls | off-centre pushes, tilt-release, rolling |
| damping (contact) | settling time; bounce decay | settling recorded (covered) |
| joint damping | open/close velocity decay after release | release the handle mid-motion |
| cloth stretch / bend | drape shape; drag response | covered in Phase 3 |

## 2. Slot table changes

Added or changed rows (everything else as in DIRECTION.md):

| on | slot | state | change |
|---|---|---|---|
| gripper/hand | **contact force on each touched vertex, every frame** | given (teacher) / zero / empty | **new.** The teacher knows it; exposing it makes mass readable via F = ma. Hidden with p = 0.3 so the model also learns position-only grippers (real hands). |
| object | **centre-of-mass offset** (3, object frame, / object size) | given or empty | **new.** Shell/solid mass model becomes a *prior*, not a fact (repo: vertex-mean CoM off by up to 2.2x; roly-poly). |
| object | **inertia scale** (log, per principal axis or scalar) | given or empty | **new.** Same reason. The physics layer uses mesh inertia x scale about the offset CoM. |
| object | mass-model flag (shell/solid) | given -> **given or empty** | real objects: unknown. Inferable as a categorical with UNKNOWN. |
| vertex | external force | given / zero / empty | **policy change** (section 5): *empty* only where the teacher applied an unseen poke; *zero* elsewhere. |

Teacher generation: sample CoM offset and inertia scale from priors (section 4) by building
objects with non-uniform density (weighted base / lid / handle region at 2-8x density), so the
labels are physical, not solver settings. Record the resulting true CoM and inertia tensor.

## 3. Action set, amended

Each episode draws actions so that every dynamic object experiences at least one
mass-identifying and one CoM-identifying interaction with probability >= 0.6. Target
fractions of *episodes* containing each family (not exclusive):

| family | fraction | parameters it identifies | notes |
|---|---|---|---|
| push along a line, release, slide to rest | 0.6 | mu, (mass via recorded force) | gripper force recorded |
| **off-centre push / nudge near the top** | 0.4 | CoM, inertia | topple vs rock; direction of fall |
| **tilt-and-release** (gripper tilts object to 10-40 deg, lets go) | 0.3 | CoM, damping | rights or topples; rocking frequency |
| pick-lift-drop (5-25 cm) | 0.4 | e, damping, mass (via lift force) | lift force recorded |
| strike / poke (0.5-1.5 m/s) | 0.3 | e, mass | impulse known |
| **dynamic-dynamic collision** (push A into B) | 0.3 | mass ratio, e | momentum exchange; no gripper force needed |
| **stack / load** (place A on B; B on a hinged lid or drawer) | 0.2 | mass via load, inter-object mu | hinged deflection / drawer motion under load |
| open/close hinged part, **release mid-motion** | 0.3 (hinged scenes) | joint damping, limits | coast-to-stop is the damping readout |
| sweep several objects | 0.15 | mu, collisions | |
| random smooth waypoints | 0.2 | coverage | unchanged |
| external pokes (unseen) | 0.3 | force-slot prior | unchanged; recorded as empty force slots |

Gripper control: kinematic as before, **but the teacher records the contact wrench** it
exerts on each touched vertex per frame. Optionally 20% of pushes are **force-limited**
(gripper slips if the required force exceeds a cap), which yields mass-dependent kinematics
even without the force slot.

## 4. Priors for the new slots (log-uniform unless stated)

| parameter | range |
|---|---|
| CoM offset | uniform in a ball of radius 0.25 x object size, biased to lie inside the mesh; 40% of objects exactly at the uniform-density CoM |
| inertia scale | 0.5-2.0 per axis (built by density regions, labels measured) |
| density regions | 1-3 regions at 2-8x base density (base, lid, handle, interior weight) |

## 5. Force-slot policy (closes risk 3; required for claim 3)

- **Pretraining:** force slot = *empty* only on vertices where the teacher applied an unseen
  poke; *zero* everywhere else. The model learns that unexplained forces are rare and local.
- **Real fine-tuning and all claim-3 evaluation:** force slots **fixed to zero** on dynamic
  objects, *empty* permitted only on vertices within 2 cm of a kinematic body (hand). A
  student that beats the teacher only when free force slots are allowed has fitted phantom
  forces, not physics. Report claim 3 **both ways**; the zero-slot number is the headline.

## 6. Recording additions

Per frame, in addition to DIRECTION.md: gripper contact wrench per touched vertex (sparse);
object-object contact impulse per pair (sparse, for the collision family); true CoM and
inertia tensor per object (per episode). Per episode: which action families occurred and
which objects were touched (for the identifiability gate below).

## 7. Phase 1 gate, extended (go / no-go before training)

In addition to invariants on a 1% sample and seed reproducibility:

- **Identifiability check (rule 1 applied to the dataset).** For each parameter and each
  action family that claims to identify it: take 50 episodes, re-simulate with the parameter
  perturbed by its prior's half-width, measure trajectory divergence. The family counts as
  identifying only if median divergence exceeds the measured GPU nondeterminism (1.5 mm)
  by 5x. Report the table. A parameter with no identifying family does not enter claim 2.
- **Free-dynamics fraction.** Fraction of dynamic-object frames with no kinematic-body
  contact: target >= 0.5 for rigid episodes.
- **Coverage per parameter.** Fraction of episodes in which each parameter is identifiable
  by the check above: target >= 0.6 for mu, e, mass, CoM; >= 0.3 for damping.

## 8. Real-data audit (do in Phase 1, decides Phase 4's scope)

For HOT3D, ARCTIC, OakInk2, PhysTwin, Physics 101: measure the fraction of dynamic-object
frames that are **not** in hand contact (free sliding, falling, settling, toppling). If it is
below ~0.1 for the hand-object sets, claim 3 as written cannot be tested on them: either
(a) run a targeted real capture (10-20 household objects: pushed, slid, dropped, toppled on a
table under mocap or multi-view tracking; one weekend), or (b) scope claim 3 to
hand-object dynamics explicitly and say so in the paper.

## 9. Housekeeping

- Pin one teacher version before any episode is generated (CLAUDE.md says newton 1.7,
  DIRECTION.md says newton 1.5.2 / warp 1.16). T0 numbers belong to a specific version.
- Write the related-work paragraph against PhysiFormer and RiCo now, so the design
  differentiates on purpose: the delta is given/empty parameter slots, one model across
  rigid/hinged/cloth, distil-then-fine-tune-on-real, and the coverage law, not the backbone.

## What this changes in the claims

- Claim 2 becomes testable for mass and CoM, not only friction. State the parameter list.
- Claim 3 becomes falsifiable: the zero-force-slot result is the claim; the free-slot result
  is a diagnostic.
- Claim 1 is unchanged.
