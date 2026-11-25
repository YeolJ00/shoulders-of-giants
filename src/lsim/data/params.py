"""Physical-parameter priors (DIRECTION.md "Parameter priors", PHASE1_AMENDMENT.md section 4).

All labels are physical quantities: friction coefficient (pair rule = Kamino's mix mode, recorded),
restitution at impact (Kamino applies it at impacts, independently of static friction), mass and
inertia measured from the built body (base build + hidden density regions), never solver knobs.
"""
from dataclasses import dataclass, field

import numpy as np


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


@dataclass
class ObjectParams:
    mu: float
    restitution: float
    density: float                     # base density of the solid or of the shell wall, kg/m^3
    build: str                         # "solid" or "shell"
    wall: float                        # shell wall thickness, m (0 for solid)
    regions: list = field(default_factory=list)   # [(xyz in object frame, radius, density)]


def sample_object(rng, interior_points, extents):
    """interior_points: candidate points inside the mesh (object frame) for density regions."""
    mu = log_uniform(rng, 0.1, 1.2)
    e = float(rng.uniform(0.05, 0.85))
    build = "shell" if rng.random() < 0.4 else "solid"
    if build == "solid":
        density = 7800.0 if rng.random() < 0.05 else log_uniform(rng, 50.0, 3000.0)
        wall = 0.0
    else:
        density, wall = log_uniform(rng, 500.0, 2700.0), log_uniform(rng, 0.001, 0.005)
    regions = []
    if rng.random() < 0.6 and len(interior_points):                     # 40% stay uniform
        for _ in range(int(rng.integers(1, 4))):
            p = interior_points[rng.integers(len(interior_points))]
            r = float(min(extents) * rng.uniform(0.08, 0.2))
            regions.append((p.tolist(), r, density * rng.uniform(2.0, 8.0) if build == "solid"
                            else log_uniform(rng, 500.0, 8000.0)))
    return ObjectParams(mu, e, density, build, wall, regions)


@dataclass
class HingeParams:
    kind: str                          # "drawer" (prismatic) or "lid" (revolute)
    damping: float                     # N s/m (drawer) or N m s/rad (lid)
    mu: float
    restitution: float


def sample_hinge(rng, kind):
    damping = log_uniform(rng, 0.5, 20.0) if kind == "drawer" else log_uniform(rng, 0.005, 0.5)
    return HingeParams(kind, damping, log_uniform(rng, 0.2, 0.8), float(rng.uniform(0.05, 0.5)))
