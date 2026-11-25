"""SimReady scene objects (NVIDIA SimReady, ALLSIDES scans, ForgeCAD, projectsim), converted to the GSO format.

Built by scripts/l1_simready_extract.py (reads the downloaded USD with pxr; this module never touches USD or any
downloaded file) and scripts/l1_simready_convert.py. Layout under data/assets/simready/:

  proc/<name>.npz     the same keys as data/assets/proc/gso/<name>.npz (lsim.data.assets.process_mesh): `collide_v/f`
                      (what the cloth/rope/soft teachers collide with: closed, <= ~2,000 faces), `convex_*` (our CoACD
                      parts, for Kamino), learner mesh and points, labels -- in the object frame: Z-up, metres, the
                      object resting on z = min(collide_v[:, 2]) in its chosen stable pose, origin at the uniform-solid
                      centre of mass. Plus `rest_T` (4x4: authored USD frame -> object frame) and the authored physics.
  parts/<name>.npz    jointed assets: each rigid part's closed collision mesh in the object frame.
  manifest.json       one row per asset: library, package, pinned revision, source URL, license, category, extents,
                      checks (closed / size / stable), authored mass, CoM, inertia, friction, restitution with their
                      provenance, split ("train" / "test"), joints summary, and `scale_evidence`: how large the
                      authored geometry is against the package's own masses (projectsim's meat slicer and forklift
                      are authored ~1.9-2.0x their real size). Those two also have a copy "<name>_rescaled" (by 1/1.90
                      and 1/2.03, from the package's recorded geometry masses; row field `rescaled`) beside the
                      authored one, in the same split as it.
  joints.json         jointed assets: parts (mass, CoM, inertia, material) and joints (type, bodies, frames, axis,
                      limits, drives) in the object frame.

Names are "<lib>__<package>" (nv__, allsides__, forgecad__, projectsim__); they never collide with GSO names.
Physics provenance: NVIDIA, ForgeCAD and projectsim values are authored class defaults (a material class's
density / friction / restitution, mass from density x volume or set by the author); ALLSIDES masses are the weighed
objects (measured), its friction and restitution come from a material class.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "data/assets/simready"
PROC = ROOT / "proc"

TABLE_SIZE = (0.03, 0.60)          # m: a table object's largest extent
MIN_THICK = 0.008                  # m: its smallest extent (as lsim.data.library.tabletop)

# coarse categories (NVIDIA by name prefix, the others by name)
_NV = {"kitchenware": "kitchenware", "dishware": "kitchenware", "utensil": "kitchenware", "utensils": "kitchenware",
       "appliance": "appliance", "refrigerator": "appliance", "climate": "appliance", "purifier": "appliance",
       "cabinet": "furniture", "chair": "furniture", "table": "furniture", "stool": "furniture", "bed": "furniture",
       "furniture": "furniture", "fixture": "fixture", "fauset": "fixture", "sink": "fixture", "toilet": "fixture",
       "pendant": "fixture", "unit": "fixture"}
_WORDS = [  # first match wins
    ("kitchenware", r"mug|cup|bowl|plate|jar|steamer|stein|cutting_board|dish_rack|rolling_pin|scoop|spatula|whisk|"
                    r"spoon|tongs|cookware|cutlery|glass|french_press|tagine|bottle|drinks_can|latching_food|tray|lid|"
                    r"soap_dish|pill_organizer|bottle_holder"),
    ("food", r"bread|lemon|pumpkin|sausage|zucchini|doughnut"),
    ("appliance", r"fryer|kettle|coffee_maker|food_processor|microwave|oven|range_hood|rice_cooker|stand_mixer|toaster|"
                  r"dishwasher|refrigerator|freezer|washer"),
    ("furniture", r"cabinet|chest_of_drawers|bookshelf|chair|footstool|stool|interior_door"),
    ("fixture", r"towel_rail|wall_bracket|faucet"),
    ("tools", r"clamp|cultivator|metal_file|shovel|toolbox|scissors|earth_anchor|sheath|stapler|hole_punch|garlic_press|"
              r"utility_knife|shears|slicer|valve|forklift"),
    ("toys_sports", r"balance_board|dartboard|dollhouse|dumbbell|football|inline_skate|kickboard|racket|toy_block|"
                    r"figurine|skull|drum"),
    ("decor", r"candle|diya|lantern|flowerpot|picture_frame|jewelry_box|lamp"),
    ("household", r"basket|bucket|crate|computer_mouse|electronics_housing|garden_hose|rope|rain_boot|stone_tile|"
                  r"trash_can|tote|spray_bottle|bathroom_scale"),
]


def category(library, package):
    if library.startswith("nvidia/"):
        return _NV.get(package.split("_")[0], "household")
    for cat, pat in _WORDS:
        if re.search(pat, package):
            return cat
    return "household"


def manifest():
    return json.loads((ROOT / "manifest.json").read_text())


def joints():
    return json.loads((ROOT / "joints.json").read_text())


def tabletop(require=("closed", "size", "stable"), libraries=None):
    """Names of converted assets that pass the checks (default: all three: a closed collision mesh, a table
    object's size, a stable resting pose)."""
    return sorted(r["name"] for r in manifest()
                  if all(r["checks"].get(k) for k in require) and (libraries is None or r["library"] in libraries))


def split(names=None):
    """(train, test) by asset, fixed in the manifest (about 10% held out, stratified by category, seed 0)."""
    rows = manifest()
    keep = None if names is None else set(names)
    pick = lambda s: sorted(r["name"] for r in rows if r["split"] == s and (keep is None or r["name"] in keep))
    return pick("train"), pick("test")


def stratified_split(rows, frac_test=0.1, seed=0):
    """name -> "train" / "test": round(frac_test x n) of each category held out (at least one where n >= 5)."""
    rng = np.random.default_rng(seed)
    out = {}
    cats = sorted({r["category"] for r in rows})
    for c in cats:
        names = sorted(r["name"] for r in rows if r["category"] == c)
        k = int(round(frac_test * len(names)))
        if k == 0 and len(names) >= 5:
            k = 1
        test = set(rng.permutation(names)[:k].tolist())
        out.update({n: ("test" if n in test else "train") for n in names})
    return out


@lru_cache(maxsize=2048)
def load(name):
    """The asset as lsim.data.library.load returns a GSO asset (plus the SimReady extras)."""
    d = np.load(PROC / f"{name}.npz")
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=512)
def load_parts(name):
    d = np.load(ROOT / "parts" / f"{name}.npz")
    return {k: d[k] for k in d.files}
