"""mp4 -> the exact array V-JEPA 2 is given.

Two constraints come from the architecture, not from taste:

* **Exactly 64 frames.** `VJEPA2RopeAttention` builds its position grid from
  `frames_per_clip // tubelet_size` = 32 and decodes a token index n as
  `t = n // (grid*grid)`. Feed a different frame count and every position id is
  wrong -- silently, with no error raised. So every clip is resampled to FRAMES.
* **Square.** `crop_size` is 256 over a 16x16 patch grid; our renders are 544x448.
  We LETTERBOX rather than centre-crop: a crop can carry the object out of frame,
  and where the object went is the entire observable. The bars are identical in
  every clip of a sweep, so they cancel in any comparison between clips.

Decoding is hand-rolled on imageio, matching src/judge/cosmos.py. The framework's
own video loader is broken in this environment and fails by returning *nothing* --
which is how the judge once scored the text alone and gave three visibly different
clips byte-identical scores.
"""
import hashlib

import numpy as np

FRAMES = 64
SIZE = 256

# A design target for renders, not a threshold to test against. Measured in
# scripts/v0_surprise.py: slowing a single clip down monotonically destroys the
# harness's ability to tell it from a frame-shuffled version of itself, and the signal
# eventually inverts. But the absolute value does NOT transfer between clips -- one
# reference discriminates well at motion 2.0 and another has already collapsed there.
# That is the same across-content confound the probe table warns about, so this is used
# to compare renders of ONE scene against each other, never as a pass/fail bar.
MOTION_TARGET = 6.0


def read_frames(path):
    """(T, H, W, 3) uint8, every frame, no subsampling."""
    import imageio.v2 as imageio

    rd = imageio.get_reader(str(path))
    frames = [f[..., :3] for f in rd]
    rd.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def resample(frames, n=FRAMES):
    """Exactly n frames, evenly spaced.

    Note this does NOT de-duplicate indices the way
    src/render/motion_budget.sample_frames does: a clip shorter than n must repeat
    frames rather than return fewer, because the rope grid is fixed at n.
    """
    idx = np.linspace(0, len(frames) - 1, int(n)).round().astype(int)
    return frames[idx]


def letterbox(frames, size=SIZE):
    """Aspect-preserving fit into size x size on black. Returns (frames, xf).

    xf = (scale, ox, oy) maps a source pixel to the letterboxed one:
        u' = u * scale + ox,  v' = v * scale + oy
    which is what src/surprise/regions.py needs to turn a crop box (computed in
    render pixels) into token indices.
    """
    from PIL import Image

    t, h, w, _ = frames.shape
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    out = np.zeros((t, size, size, 3), np.uint8)
    for i in range(t):
        im = Image.fromarray(frames[i]).resize((nw, nh), Image.BILINEAR)
        out[i, oy:oy + nh, ox:ox + nw] = np.asarray(im)
    return out, (scale, float(ox), float(oy))


def prepare(path, n=FRAMES, size=SIZE):
    """mp4 -> ((n, size, size, 3) uint8, letterbox transform)."""
    return letterbox(resample(read_frames(path), n), size)


def motion(frames):
    """Mean |frame difference| -- how much the clip actually moves.

    This is not a nicety. scripts/v0_surprise.py measures the harness's
    discrimination against exactly this quantity and finds it collapses, and then
    INVERTS when a clip is slowed: a frame-shuffled version stops scoring as more
    surprising than the real one. Natural video sits near 17; every clip in
    outputs/judge sits between 0.3 and 2.3. Use it to compare renders of the same
    scene -- the same stance src/render/motion_budget.py takes for the judge.
    """
    f = np.asarray(frames, np.float32)
    return float(np.abs(np.diff(f, axis=0)).mean())


def clip_hash(path):
    """sha256 of the file's bytes, truncated -- the cache key for a scored clip.

    Hashing the bytes rather than the theta that produced them means a re-render
    with a different seed is correctly a different clip, which is exactly what the
    noise-floor measurement depends on.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
