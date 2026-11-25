"""Which tokens cover the object.

A 256x256 frame of a mostly-static table averages the moving object away: it is a
few hundred pixels of a 65k-pixel frame, so a full-frame mean is dominated by
tokens whose content never changes and cannot depend on theta. That does not add
noise -- it adds a large constant -- but it shrinks the fractional signal we are
trying to resolve against the render-seed floor.

So report a second reading restricted to the tokens the object actually occupies.
The box comes from the same machinery the judge crops with (src/render/crop.py:
project the object's points through the known camera, take the union over time),
so it costs nothing and is exact.
"""
import numpy as np

from .vjepa import GRID, PATCH, TOKENS_PER_SLOT, DEPTH


def box_to_spatial_tokens(box, xf, grid=GRID, patch=PATCH, dilate=1):
    """(x0,y0,x1,y1) in RENDER pixels -> spatial token indices in [0, grid*grid).

    xf is the (scale, ox, oy) returned by clips.letterbox -- the box must be moved
    into letterboxed coordinates before it means anything in token space.
    `dilate` pads by whole tokens; 1 is the default because a body's contact edge
    matters and it sits at the boundary of its own box.
    """
    scale, ox, oy = xf
    x0, y0, x1, y1 = box
    x0, x1 = x0 * scale + ox, x1 * scale + ox
    y0, y1 = y0 * scale + oy, y1 * scale + oy
    c0 = max(0, int(np.floor(x0 / patch)) - dilate)
    c1 = min(grid - 1, int(np.ceil(x1 / patch)) + dilate)
    r0 = max(0, int(np.floor(y0 / patch)) - dilate)
    r1 = min(grid - 1, int(np.ceil(y1 / patch)) + dilate)
    if c1 < c0 or r1 < r0:
        return np.zeros(0, np.int64)
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    return (rr * grid + cc).reshape(-1).astype(np.int64)


def spatial_to_full(spatial, slots=None, tokens_per_slot=TOKENS_PER_SLOT, depth=DEPTH):
    """Spatial token ids -> full-clip token ids, over the given temporal slots."""
    slots = np.arange(depth) if slots is None else np.asarray(slots)
    return (slots[:, None] * tokens_per_slot + spatial[None, :]).reshape(-1).astype(np.int64)
