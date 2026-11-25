"""--chunk (2026-11-06): the network predicts a window's 8 coming frames from one call and the exact layer integrates
them in order. Checks: off is the default and changes nothing; with the network's forces at zero the chunk integrates
exactly as the frame-by-frame rollout does (same frames, held targets, free vertices); the coming frames' true cloth
positions never reach the prediction (only the given gripper / held / static ones); every parameter trains."""
import glob

import numpy as np
import pytest
import torch

from lsim.model.cloth_net import build_cloth_model, kin_next

SHARDS = sorted(glob.glob("data/datasets/c1_test/shard_*.npz"))[:1]
pytestmark = pytest.mark.skipif(not SHARDS, reason="c1_test not on disk")
A = dict(arch="qwen", d=64, layers=2, heads=4, kv_heads=2, prob_slots=True, unc=True, accel=True, layer=2)
P = K = 8


def _batch(n=2, seed=0, p_hidden=0.0):
    from lsim.train.cloth_train import _load, _windows
    from lsim.train.cloth_windows import collate_cloth
    rng = np.random.default_rng(seed)
    return collate_cloth(_windows(_load(SHARDS, set()), n, P, K, rng), "cpu", rng, p_hidden=p_hidden, augment=False,
                         all_past=True)


def _chunk_predict(m, b):
    from lsim.model.cloth_net import decode
    from lsim.train.cloth_train import chunk_inputs, chunk_steps, contact_kw, forces
    pos = b["pos"]
    free = b["is_cloth"][:, None] & ~b["held"]
    kn = kin_next(b, torch.cat([pos, pos[:, -1:]], 1))
    fi, fc, pr, _ = m(chunk_inputs(b, pos, free, kn, P, K, "cpu"))
    w, mu = decode(pr, b)
    return chunk_steps(b, pos, forces(b, fi, fc, w), contact_kw(m, b, fc, w), w, free, mu, P, K)


def test_off_by_default():
    assert build_cloth_model(A, ckpt=False, bf16=False).chunk == 0
    assert build_cloth_model(dict(A, chunk=8), ckpt=False, bf16=False).chunk == 8


def test_zero_forces_chunk_equals_frame_by_frame():
    import sys
    sys.path.insert(0, "scripts")
    from l2_chunk_fit import predict8
    from lsim.train.cloth_train import ZeroForces
    b = _batch(seed=6)
    z = ZeroForces()
    seq = predict8(z, b, "cpu")                                          # 8 calls, own predictions fed back
    z.chunk = 8
    ch = _chunk_predict(z, b)
    assert ch.shape == seq.shape == (b["pos"].shape[0], K, 3)
    assert torch.allclose(ch, seq, atol=1e-7), (ch - seq).abs().max()


def test_coming_cloth_positions_do_not_leak():
    torch.manual_seed(0)
    m = build_cloth_model(dict(A, chunk=8), ckpt=False, bf16=False).eval()
    for p in m.parameters():                                             # non-zero forces (zero-initialised heads)
        p.data += 1e-2 * torch.randn_like(p)
    b = _batch(seed=6)
    with torch.no_grad():
        x0 = _chunk_predict(m, b)
        free = b["is_cloth"][:, None] & ~b["held"]
        b2 = dict(b, pos=b["pos"].clone())
        b2["pos"][:, P:] += 0.05 * free[:, P - 1:P - 1 + K, None] * torch.randn_like(b2["pos"][:, P:])
        x1 = _chunk_predict(m, b2)
    assert x0.abs().max() > 0 and torch.equal(x0, x1)


def test_chunk_loss_trains_every_parameter():
    from lsim.train.cloth_train import step_loss
    m = build_cloth_model(dict(A, chunk=8), ckpt=False, bf16=False)
    loss, info = step_loss(m, _batch(seed=6, p_hidden=0.5), np.random.default_rng(1), K, "cpu", sig=1e-3, pf=4, edge_w=1.0)
    loss.backward()                                                      # the force heads start at zero: one step on
                                                                         # (seeds 6, 7: held, moving cloth -- a towel at
                                                                         # rest on the table takes no gradient: static
                                                                         # friction absorbs small forces)
    torch.optim.SGD(m.parameters(), lr=1e-2).step()
    m.zero_grad()
    loss, info = step_loss(m, _batch(seed=7, p_hidden=0.5), np.random.default_rng(3), K, "cpu", sig=1e-3, pf=4, edge_w=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)]
    assert not missing, missing
