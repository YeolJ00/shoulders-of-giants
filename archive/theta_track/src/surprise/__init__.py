"""Predictive-model surprise as a plausibility signal (the V-track).

The chat judge is retired as the optimisation signal: it approved 52.8% of clips at
p > 0.90 and correlated -0.02 with density. This package tries the other kind of
video model -- a *predictive* one. V-JEPA 2 sees the first part of one of our clips,
predicts the continuation in representation space, and the prediction error is
"surprise". No prompt, no language, no sampling.

Nothing here is trusted until scripts/v1_killtest.py shows it moves with theta.
Rule 1 applies to a supervision signal exactly as it applies to a probe.
"""
from .clips import (  # noqa: F401
    FRAMES, MOTION_TARGET, SIZE, clip_hash, motion, prepare, read_frames,
)
from .vjepa import MODEL, SurpriseModel                                    # noqa: F401
