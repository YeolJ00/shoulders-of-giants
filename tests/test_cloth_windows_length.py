"""The windows read episodes of any length (long episodes: lsim.data.cloth / rope with --seconds 20): a 4 s shard
tiled in time to 5x its frames loads with that many frames, its held map follows the shifted grasp events, and a
window at the end of the long episode is the matching window of the short one. Runs on CPU.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_cloth_windows_length.py
"""
import glob
import json

import numpy as np
import pytest

from lsim.train.cloth_windows import load_shard, make_cloth_window

CLOTH = sorted(glob.glob("data/datasets/c1_test/shard_*.npz"))[:1]
ROPE = sorted(glob.glob("data/datasets/r1_test/shard_*.npz"))[:1]
REPEAT = 5


def _tile(src, dst):
    """Write `dst` (no suffix): src's shard with its recorded frames repeated REPEAT times and every world's events
    repeated at the matching offsets (the settle frames stay in front)."""
    npz = dict(np.load(src))
    meta = json.loads(open(src[:-4] + ".json").read())
    F = npz["cloth_c"].shape[0]                                        # recorded frames (F - 1 steps)
    for k in ("cloth_c", "cloth_d", "grip_q", "grasp_F", "contact_F"):
        npz[k] = np.concatenate([npz[k]] * REPEAT, 0)
    for wd in meta["worlds"]:
        ev = wd["events"]
        copies = [[[e[0] + r * F] + list(e[1:]) for e in ev] for r in range(REPEAT)]
        for r in range(REPEAT - 1):                                    # a copy's late events must not outlast the next
            if copies[r + 1]:                                          # copy's first (a held start's settling grasp)
                copies[r] = [e for e in copies[r] if e[0] < min(x[0] for x in copies[r + 1])]
        wd["events"] = [e for c in copies for e in c]
    np.savez(str(dst) + ".npz", **npz)
    (dst.parent / (dst.name + ".json")).write_text(json.dumps(meta))
    return F


@pytest.mark.parametrize("shards", [pytest.param(CLOTH, id="cloth"), pytest.param(ROPE, id="rope")])
def test_long_episode_windows(shards, tmp_path):
    if not shards:
        pytest.skip("dataset not on disk")
    src = shards[0]
    F = _tile(src, tmp_path / "shard_00000")
    short = load_shard(src[:-4], only_ok=False)
    long = load_shard(str(tmp_path / "shard_00000"), only_ok=False)
    assert len(short) == len(long)
    settle = json.loads(open(src[:-4] + ".json").read())["settle_frames"]
    compared = 0
    for es, el in list(zip(short, long))[:8]:
        assert es.n_frames == F and el.n_frames == REPEAT * F
        np.testing.assert_array_equal(el.x(np.arange(el.n_frames - F, el.n_frames)), es.x(np.arange(F)))
        # held: tiled copy r's events lie r F frames on, so from the short episode's first event on, the last copy's
        # held map is the short one's (before it, the previous copy's last grasp carries over)
        ev = [e[0] for e in es.wd["events"]]
        j0 = max(min(ev) - settle, 0) if ev else 0
        np.testing.assert_array_equal(el.held[(REPEAT - 1) * F + j0:], es.held[j0:])
        for t_cur in (el.n_frames - 9, (REPEAT - 1) * F + 20):         # the end of the long episode, and inside it
            t_s = t_cur - (REPEAT - 1) * F
            if t_s - 8 < j0 or t_s + 8 > F - 1:
                continue
            wl = make_cloth_window(el, t_cur, 8, 8, np.random.default_rng(0))
            ws = make_cloth_window(es, t_s, 8, 8, np.random.default_rng(0))
            assert wl.keys() == ws.keys()
            compared += 1
            for k in set(wl) - {"frames"}:
                a, b = np.asarray(wl[k]), np.asarray(ws[k])
                if a.dtype.kind in "fiub":
                    np.testing.assert_array_equal(a, b, err_msg=k)
    assert compared >= 4
