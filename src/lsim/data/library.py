"""Processed asset library (data/assets/proc/<source>/): selection and cached loading."""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PROC = REPO / "data/assets/proc"


def index(source="gso"):
    return json.loads((PROC / source / "index.json").read_text())


def tabletop(source="gso", min_extent=0.03, max_extent=0.30, max_shoes=40, seed=0):
    """Names of watertight, tabletop-sized objects. GSO is ~25% shoes; cap them so the set is a
    household mix rather than a shoe catalogue."""
    rows = [r for r in index(source) if r.get("ok") and r["watertight"]
            and min_extent <= max(r["extents"]) <= max_extent and min(r["extents"]) >= 0.008]
    shoe = re.compile(r"shoe|sneaker|boot|sandal|slipper|loafer|oxford|moccasin", re.I)
    shoes = [r["name"] for r in rows if shoe.search(r["name"])]
    keep = set(np.random.default_rng(seed).permutation(shoes)[:max_shoes]) if shoes else set()
    return sorted(r["name"] for r in rows if not shoe.search(r["name"]) or r["name"] in keep)


SIMREADY_PREFIXES = ("nv__", "allsides__", "forgecad__", "projectsim__")   # lsim.data.simready names; no GSO name has "__"


def source_of(name):
    """The library an object name belongs to: "gso" or "simready"."""
    return "simready" if name.startswith(SIMREADY_PREFIXES) else "gso"


@lru_cache(maxsize=2048)
def load(name, source="gso"):
    """An object by name. SimReady names (lsim.data.simready) load from data/assets/simready, in the same format."""
    if source_of(name) == "simready":
        from lsim.data import simready
        return simready.load(name)
    d = np.load(PROC / source / f"{name}.npz")
    return {k: d[k] for k in d.files}


def object_split(objects="gso"):
    """(train, test) obstacle names for the scene generators' --objects: "gso" (exactly split(tabletop()), today's
    pools), "simready" (the SimReady table objects passing every check, split by their manifest), or "both" (the two
    lists concatenated: a uniform draw then picks GSO about 3:1)."""
    if objects not in ("gso", "simready", "both"):
        raise ValueError(f"objects must be gso, simready or both, not {objects!r}")
    tr, te = [], []
    if objects in ("gso", "both"):
        a, b = split(tabletop())
        tr, te = tr + a, te + b
    if objects in ("simready", "both"):
        from lsim.data import simready
        a, b = simready.split(simready.tabletop())
        tr, te = tr + a, te + b
    return tr, te


def split(names, frac_test=0.1, seed=0):
    """Held-out objects are fixed by name before any training (DIRECTION.md, splits)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sorted(names))
    n = int(round(frac_test * len(perm)))
    return sorted(perm[n:].tolist()), sorted(perm[:n].tolist())
