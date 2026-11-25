# NVIDIA SimReady asset inventory (2026-10-30)

Metadata only: Hugging Face API (dataset info, recursive trees), dataset cards, the SimReady Central
index (github.com/NVIDIA-Omniverse/simready-central, `submissions/`), a text crawl of all 2,669
`payloads/Physics/physics.usda` layers in `nvidia/simready-assets`, and 7 sampled packages opened with
pxr (USD and JSON only, textures skipped). Per-repo rows: `simready_inventory.json`.

Where things live: `nvidia/simready-catalog` does not exist on the Hub (401). The index is the GitHub repo
above: 22 registered datasets from 17 owners (3,872 package references). The HF Space `nvidia/simready-central` is a semantic-search
front end over `nvidia/simready-assets`, `standardbots/simready-core` and two private NVIDIA datasets.

## Libraries

Counts are SimReady packages (folders holding `com.nvidia.simready.packaging.json`) unless marked.
Size: deduplicated Hub storage, with the listed sum in brackets where textures are duplicated.

| library | assets | categories | license | size | physics |
|---|---|---|---|---|---|
| nvidia/simready-assets (indexed) | 2,677 (2,536 indexed) | food 538, kitchen/dining 222, household goods 424, furniture/appliances 108, tools 223, medical 171, automotive 234, warehouse/industrial 757 | CC-BY-4.0 | 83 GB (490 GB listed) | rigid body on all 2,669 physics layers; MassAPI (mass, CoM, inertia); PhysicsMaterialAPI (density, mu_s, mu_d, e) per material class; colliders in per-solver variants (PhysX sdf, MuJoCo convexHull, Newton triangle mesh + rolling/torsional friction); **678 with revolute/prismatic joints** (limits + drives) |
| PalatialSim/palatial-nv-asset-library (indexed) | 1,013 | kitchenware, food, household goods | CC-BY-4.0 | 56 GB | mass, density, collision, joints (90 with moving joints); some mass geometry-derived |
| ALLSIDES/simready-scanned-objects (indexed) | 69 | household, kitchen, toys/sports, tools (3D scans) | CC-BY-4.0 | 34 GB | **weighed real mass**; friction and restitution from material class; CoACD convex hulls (default) or SDF (PhysX) |
| ALLSIDES/simready-halloween | 26 | decor, dolls, figurines | CC-BY-4.0 | 8.4 GB | as ALLSIDES |
| ForgeCAD/simready-assets (indexed) | 121 (76 indexed) | appliances, kitchen, furniture, tools, office, warehouse, assembly kits | CC-BY-4.0 | 21 GB | ArticulationRoot, mass/CoM/inertia, materials, convex hulls, joints |
| LGElectronics/simready-assets (indexed) | 21 | fridges, washers, dishwashers, ovens, TVs | CC-BY-4.0 | 1.8 GB | SimReady (not sampled) |
| projectsim/simready-articulated-objects-sample | 8 (100k+ on request, commercial) | kitchen tools, lamp, tools, industrial | CC-BY-4.0 | 0.3 GB | joints with limits, mass per part, convex decomposition |
| imagine-io/PhysicalAI-Simready-Kitchen-Assets | ~102 usdz packages + 453 glb | appliances 27, kitchen clutter 73 | CC-BY-4.0 | 13 GB | physics APIs + collision approximations (card) |
| nvidia/PhysicalAI-SimReady-Warehouse-01 | 753 (card) | warehouse + graspable props | CC-BY-4.0 | 11 GB | colliders only, **no rigid bodies**; CSV of approximate mass |
| imagineio/PhysicalAI-SimReady-Assets | 892 | furniture, household, "linens" | **all rights reserved, no training use** | 7 GB | full (card) |
| imagineio Homes / Kitchens-v1 / Kitchens-v2 | 1,000 / 800 / 7,556 scenes | home and kitchen scenes | CC-BY-NC (v2 CC-BY, manual gate) | 1.6 GB / 1.6 TB / 3.0 TB | articulated doors/drawers, mass, colliders |
| Yootta/World-SimReady-Home | 500 scenes (31,711 placements) + 1,000 robot episodes | household scenes | CC-BY-NC-SA | 3.0 TB | mass, density, friction, restitution |
| Aperdata-tech/Aperdata-SimReady-Kitchen-01 | 1 scene + 22 props | kitchen | YAML says CC-BY, gate says non-commercial | 2 GB | not inspected |
| valtsblukis/robolab_longhorizon_simready_assets | 34 | dishware, food, storage | NVIDIA Omniverse license (Isaac 6.0 copies) | 0.75 GB | 14 with recalibrated mass |
| Reger-Chen/simready-assets | 2,575 | mirror of nvidia/simready-assets | none | 74 GB | the same |
| robots/grippers (13 OEM repos, nvidia/isaacsim-assets 137) | 1-137 each | robots | mostly CC-BY | < 2 GB | articulations |
| others: Vertiv 41, nvidia/simready-dsx 1 (data-centre), atechdev 24 (electronics), halloween-candy 22, hcltech simreadybench (a benchmark, not a library) | | | | | |

**Household-relevant total with a permissive license: about 2,550 distinct assets.** That is nvidia 1,292
(prefix-based), Palatial 1,013, ALLSIDES 69, ForgeCAD ~45, LG 21, imagine-io kitchen packages ~102 and
projectsim 4. Not counted: the unlicensed mirror, the all-rights-reserved imagine.io objects, the
non-commercial scenes and the copied robolab set.

## What the sampled USD actually holds

7 packages in `data/external/simready_sample/<owner>__<repo>/`, 82 MB in total (textures and MDL skipped).
All of them: `metersPerUnit = 1`, `upAxis = Z`, `kilogramsPerUnit = 1`.

| asset | bodies / joints | mass, CoM, inertia | material | collider (approximation; watertight) |
|---|---|---|---|---|
| nvidia kitchenware_bowl_b01 | 1 / none | 1.145 kg, CoM, diag inertia, principal axes | rho 2600, mu_s 0.35, mu_d 0.30, e 0.05 | PhysX sdf / MuJoCo convexHull / Newton mesh; 1 of 1 watertight |
| nvidia cleaning_towel_a01 | 1 **rigid** / none | 0.238 kg | rho 1000, mu 0.5, e 0.3 | 48k-face mesh; watertight |
| nvidia cabinet_file_a01 | 13 / 6 prismatic (drawers, 0-0.294 m), 4 revolute, 2 fixed; linear and angular drives | 0.88-100 kg per part | rho 950/2500/7850 | 2 of 13 watertight (4,600 open edges) |
| nvidia appliance_toaster_a01 | 5 / 2 prismatic, 1 revolute, 1 fixed | 0.0016-1.46 kg | plastic and metal classes | 2 of 5 watertight |
| ALLSIDES mug_A07RDPW843 | 1 / none | 0.150 kg weighed; CoM and inertia not authored | mu_s 0.363, mu_d 0.273, e 0.283 (measured class) | 64 CoACD hulls, all watertight; PhysX variant: SDF at 256 on a 200k-tri mesh |
| ForgeCAD air_fryer | 4 + ArticulationRoot / prismatic basket 0-0.24 m, revolute dial 0-270 deg | 0.045-4.6 kg | rho 1250/2700, mu 0.45-0.65, e 0 | 29 convex hulls, all watertight |
| projectsim garlic_press | 2 + ArticulationRoot / revolute -5.7 to 91.7 deg, filtered pair | 0.257 / 0.288 kg | rho 2700, mu_s 0.45, mu_d 0.35, e 0.25 | convexDecomposition, both watertight; CCD on |

Across all 2,669 nvidia physics layers the values are class defaults, not per-object measurements:
densities cluster on 950 / 7850 / 7800 / 689 / 900 kg/m^3 and restitution on 0.4 / 0.1. Only the ALLSIDES
masses are measured.

## Usefulness for this project

**(a) Scene objects for cloth, rope and soft bodies to interact with (needs closed collision meshes).**
- The best fit is the **ALLSIDES scans**: real shapes, weighed mass, CoACD hulls that are all closed and
  the same decomposition `scene.add_convex_parts` already uses. Only 69 of them, but household.
- The **nvidia kitchen/dining (222) and household goods (424)** collide with their visual mesh. Single-shell
  props are closed (the bowl and the towel are), but multi-part appliances and furniture are not (toaster
  3 of 5 parts open, cabinet 11 of 13). Run a closedness check per asset, or CoACD them as we already do.
- Palatial (1,013 kitchen items) is the same kind of content at larger count. Not sampled.

**(b) A future rigid/hinged library (needs mass, friction and joints).**
- nvidia has 678 assets with joints, 202 of them household (file and drug cabinets, dishwasher,
  refrigerator, toaster, kettles, bins with lids, faucets, soap dispensers). Joints carry limits and drives.
- ForgeCAD (kitchen cabinets, chest of drawers, microwave, fridge, washer, doors) and projectsim also
  ship ArticulationRoot assets.
- Mass, CoM and inertia are authored. Friction, restitution and density come per material class, so they
  are labels drawn from a small set, not measurements. That suits theta slots sampled around a class
  prior, not a real-material ground truth.

**(c) Deformables.** There are none. No deformable, cloth, particle or soft-body schema appears in any of
the 2,669 nvidia physics layers (the one token hit is a prim named `cleaningCloth_01_obj_00`). Towels,
cleaning cloths, sponges, surgical drapes, bandages, hoses, the ALLSIDES rope and garden hose are all
single rigid bodies. Their meshes could still serve as rest shapes or targets (folded and wrinkled
microfiber towels) for our own cloth teacher. imagine.io has a "linens" folder, but its license forbids
training use.

## Recommended subset

| part | assets | download (USD/JSON, no textures) |
|---|---|---|
| (a) nvidia kitchen/dining | 222 | 0.25 GB |
| (a) ALLSIDES scanned, lod 1 + colliders | 69 | 1.33 GB |
| (b) nvidia articulated furniture/fixtures/appliances | 72 | 2.21 GB |
| (b) ForgeCAD household subset | 45 | 1.79 GB |
| (b) projectsim articulated sample | 8 | 0.06 GB (0.30 GB with textures) |
| **total** | **416** | **5.6 GB** |

Textures dominate the full packages: the nvidia median package is 70 MB with textures, against 1.4 MB
for `geometries.usd`. Renders need the textures or a fixed material (rule 4).

## Notes
- `data/external/` was not git-ignored. It is added to `.gitignore` now, and nothing is committed.
- Downloaded USD was opened only from scratchpad scripts, with `python -I`, using the `pxr` already in the
  `lsim` env. Nothing was installed and nothing is imported from project code.
