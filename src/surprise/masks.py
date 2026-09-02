"""Mask families for surprise, and what each one can see.

The prefix mask the reference uses answers "given the past, what happens next" -- but it
averages the whole future into one number, so a local anomaly is diluted by construction.
Measured: the authored teleport moves ONE slot in 30 and its per-slot spike is ~100x the
noise floor, yet it scores 2/6 under a prefix mask while `freeze` and `through`, which
change the entire remaining trajectory, score 5/6.

We do not know in advance WHEN or WHERE a clip will be implausible, so we should not commit
to one mask. Each family answers a different question:

  prefix(n)              given everything before slot n, predict all of the rest.
                         Causal. Sensitive to sustained divergence, blind to brief events.

  temporal_block(t, w)   hide slots [t, t+w) and predict them from BOTH sides. This is
                         closest to V-JEPA's own pretraining (random tubes, not prefixes)
                         and it is the one that should catch a teleport: with the future
                         visible, an object that jumps and continues is inconsistent with
                         both neighbours at once.

  object_tube(t, w)      as temporal_block, but only the object's own patches are hidden;
                         the same-time background stays in context, so the model is told
                         the camera did not move and is asked only where the object went.

Because the encoder runs once per clip and every mask is just another predictor call, a
whole family costs a fraction of one encode. The ARGMAX over masks is as useful as the max:
it says which slot the model found implausible, which is what localises the parameter to
blame -- a peak at impact implicates contact damping, a peak during the slide implicates
friction.
"""
import numpy as np


def prefix(depth, tps, step=2, spatial=None):
    """(name, context_ids, target_ids, centre_slot) for each context length."""
    out = []
    for n in range(step, depth, step):
        ctx = _ids(range(0, n), tps)
        tgt = _ids(range(n, depth), tps, spatial)
        out.append((f"prefix{n:02d}", ctx, tgt, n))
    return out


def temporal_block(depth, tps, widths=(1, 2), spatial=None, per_slot=None):
    """Hide slots [t, t+w) and predict them from both sides."""
    out = []
    for w in widths:
        for t in range(1, depth - w):
            slots = range(t, t + w)
            tgt = (_ids(slots, tps, spatial) if per_slot is None
                   else np.concatenate([s * tps + np.asarray(per_slot[s], np.int64)
                                        for s in slots]))
            ctx = np.setdiff1d(np.arange(depth * tps), tgt)
            out.append((f"block{w}@{t:02d}", ctx, tgt, t + w / 2.0))
    return out


def _ids(slots, tps, spatial=None):
    sp = np.arange(tps) if spatial is None else np.asarray(spatial, np.int64)
    s = np.asarray(list(slots), np.int64)
    return (s[:, None] * tps + sp[None, :]).reshape(-1)


def families(depth, tps, per_slot=None, step=2, widths=(1, 2)):
    fam = {"prefix": prefix(depth, tps, step),
           "block": temporal_block(depth, tps, widths)}
    if per_slot is not None:
        fam["object"] = temporal_block(depth, tps, widths, per_slot=per_slot)
    return fam
