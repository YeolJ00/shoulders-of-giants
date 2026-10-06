"""Outcome-classifier supervision (the O-track).

Ask the video model the one thing it is measurably best at: WHICH PHYSICAL OUTCOME
happened. Not "is this plausible" (the judge: rho ~ 0 with every parameter) and not "how
surprising is this" (V-JEPA surprise: at chance on categorically impossible motion in our
renders). SSv2's 174 templates are outcome labels on real objects -- "poking something so
it falls over" vs "so it slightly moves" -- and a frozen encoder plus a small attentive
probe reaches 77.3% on them.

Nothing here is trusted until G-0 shows the head responds to our renders at all.
"""
from .head import HEAD, OutcomeHead                                   # noqa: F401
