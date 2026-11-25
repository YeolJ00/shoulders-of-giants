"""The Qwen-style cloth network (lsim.model.cloth_qwen) as a drop-in for ClothSimulator: same outputs, forces zero at
initialisation (the exact layer's default), every parameter trained, old checkpoints still load through the factory."""
import glob

import numpy as np
import pytest
import torch

from lsim.model.cloth_net import build_cloth_model

SHARDS = sorted(glob.glob("data/datasets/c1_test/shard_*.npz"))[:1]
pytestmark = pytest.mark.skipif(not SHARDS, reason="c1_test not on disk")


def _batch(n=2, seed=0):
    from lsim.train.cloth_train import _load, _windows
    from lsim.train.cloth_windows import collate_cloth
    rng = np.random.default_rng(seed)
    return collate_cloth(_windows(_load(SHARDS, set()), n, 8, 8, rng), "cpu", rng)


A = dict(arch="qwen", d=64, layers=2, heads=4, kv_heads=2, prob_slots=True, unc=True, accel=True, layer=2)


def test_outputs_and_zero_init():
    from lsim.train.cloth_train import kin_next
    b = _batch()
    b.update(tau=torch.zeros(b["pos"].shape[1]), frame_given=torch.ones(b["pos"].shape[1]))
    b["kin_next"] = kin_next(b, torch.cat([b["pos"], b["pos"][:, -1:]], 1))
    m = build_cloth_model(A, ckpt=False, bf16=False).eval()
    fi, fc, pr, ec = m(b)
    N, Fw, _ = b["pos"].shape
    assert fi.shape == fc.shape == (N, Fw, 3) and pr.shape == (b["n_bodies"], 4) and ec.shape[0] == 2
    assert fi.abs().max() == 0 and fc.abs().max() == 0                # starts from free fall + Coulomb


def test_every_parameter_gets_a_gradient():
    from lsim.train.cloth_train import step_loss
    m = build_cloth_model(A, ckpt=False, bf16=False)
    loss, _ = step_loss(m, _batch(), np.random.default_rng(1), 8, "cpu", sig=1e-3, edge_w=1.0)
    loss.backward()                                                  # the force heads start at zero: one step on
    torch.optim.SGD(m.parameters(), lr=1e-2).step()
    m.zero_grad()
    loss, _ = step_loss(m, _batch(seed=2), np.random.default_rng(3), 8, "cpu", sig=1e-3, edge_w=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0] == []


def test_old_checkpoint_loads():
    ck = "outputs/lsim/cloth_v3/last.pt"
    if not glob.glob(ck):
        pytest.skip("no v3 checkpoint")
    c = torch.load(ck, map_location="cpu")
    build_cloth_model(c["args"]).load_state_dict(c["model"])


# ---------------------------------------------------------------------------------------------------------------------
# Old checkpoints: outputs bit-for-bit as before the 3D network (fingerprints in tests/data, tests/cloth_ref_outputs.py)
# ---------------------------------------------------------------------------------------------------------------------
def test_old_checkpoints_unchanged_bit_for_bit():
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from cloth_ref_outputs import CKPTS, REF, ROOT, fingerprints
    ref = json.loads(REF.read_text())
    have = [ck for ck in CKPTS if (ROOT / ck).exists() and ck in ref]
    if not have:
        pytest.skip("no old checkpoints on disk")
    for ck in have:
        assert fingerprints(ck) == ref[ck], ck


# ---------------------------------------------------------------------------------------------------------------------
# The 3D network (ClothQwen3D, --arch qwen3d) on cloth, rope and soft windows
# ---------------------------------------------------------------------------------------------------------------------
ROPE = sorted(glob.glob("data/datasets/r1_test/shard_*.npz"))[:1]
SOFT = (sorted(glob.glob("data/datasets/s1/shard_*.npz")) or sorted(glob.glob("outputs/lsim/soft_pilot/shard_*.npz")))[:1]
A3 = dict(A, arch="qwen3d", d=96, heads=4, kv_heads=2)               # head dim 24: rotary sections 8 / 8 / 8


def _mixed(seed=0):
    """One cloth, one rope and one soft window (the smallest soft body with a grasp, to keep the CPU test small)."""
    from lsim.train.cloth_windows import collate_cloth, load_shard, make_cloth_window
    if not (ROPE and SOFT):
        pytest.skip("rope / soft shards not on disk")
    rng = np.random.default_rng(seed)
    c = load_shard(SHARDS[0][:-4])[0]
    r = load_shard(ROPE[0][:-4])[0]
    soft = sorted(load_shard(SOFT[0][:-4]), key=lambda e: (not e.held.any(), len(e.rest)))[0]
    t = int(np.flatnonzero(soft.held.any(1))[0]) + 4 if soft.held.any() else 30
    wins = [make_cloth_window(c, 30, 8, 8, rng), make_cloth_window(r, 30, 8, 8, rng),
            make_cloth_window(soft, min(t, soft.n_frames - 9), 8, 8, rng)]
    return collate_cloth(wins, "cpu", rng, hidden=("density", "bend"))     # slots 0 and 2 hidden in every window


def test_3d_outputs_finite_and_zero_init():
    from lsim.train.cloth_train import kin_next
    b = _mixed()
    assert b["material"].tolist() == [0, 1, 2]
    b.update(tau=torch.zeros(b["pos"].shape[1]), frame_given=torch.ones(b["pos"].shape[1]))
    b["kin_next"] = kin_next(b, torch.cat([b["pos"], b["pos"][:, -1:]], 1))
    m = build_cloth_model(A3, ckpt=False, bf16=False).eval()
    fi, fc, pr, ec = m(b)
    N, Fw, _ = b["pos"].shape
    assert fi.shape == fc.shape == (N, Fw, 3) and pr.shape == (3, 4)
    assert torch.isfinite(pr).all() and torch.isfinite(m.slot_logvar).all() and torch.isfinite(m.log_sigma).all()
    assert fi.abs().max() == 0 and fc.abs().max() == 0                # starts from free fall + Coulomb, every material


def test_3d_every_parameter_gets_a_gradient():
    from lsim.train.cloth_train import step_loss
    m = build_cloth_model(A3, ckpt=False, bf16=False)
    loss, info = step_loss(m, _mixed(), np.random.default_rng(1), 8, "cpu", sig=1e-3, edge_w=1.0)
    loss.backward()
    torch.optim.SGD(m.parameters(), lr=1e-2).step()
    m.zero_grad()
    loss, info = step_loss(m, _mixed(seed=2), np.random.default_rng(3), 8, "cpu", sig=1e-3, edge_w=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(f"{k}_l_pos" in info for k in ("cloth", "rope", "soft"))   # per-material losses are logged
    assert [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0] == []


def test_3d_rotary_and_neighbour_slots_reduce_to_cloth():
    """For a cloth the 3D rotary's third section is the identity and the 26 neighbour slots hold the 8 in-plane ones
    of the 2D network, unchanged."""
    from lsim.model.cloth_qwen import _DIRS3, N_SLOT, SLOT_F, ClothQwen3D, neighbour_slots, neighbour_slots_3d, rope_sections
    assert rope_sections(64) == (22, 22, 20) and rope_sections(24) == (8, 8, 8)
    b = _batch(1)
    m = ClothQwen3D(d=96, layers=1, heads=4, kv_heads=2, mlp_hidden=96, ckpt=False, bf16=False)
    rs = m._rope_space(b, 2)
    cl = b["is_cloth"].repeat_interleave(2)
    assert (rs[4][cl] == 1).all() and (rs[5][cl] == 0).all()
    pos, vel = b["pos"], b["vel"]
    old = neighbour_slots(b, pos, vel).view(len(pos), pos.shape[1], N_SLOT, SLOT_F)
    new = neighbour_slots_3d(b, pos, vel).view(len(pos), pos.shape[1], 26, SLOT_F)
    in_plane = []
    for k in range(N_SLOT):
        a = k * np.pi / 4
        j = int((torch.tensor([np.cos(a), np.sin(a), 0.0], dtype=torch.float32) @ _DIRS3.T).argmax())
        assert torch.allclose(new[:, :, j], old[:, :, k], atol=1e-6), k
        in_plane.append(j)
    rest = [j for j in range(26) if j not in in_plane]
    assert new[:, :, rest].abs().max() == 0


# ---------------------------------------------------------------------------------------------------------------------
# The exact layer per material, network outputs at zero: closed forms
# ---------------------------------------------------------------------------------------------------------------------
DT, G = 1.0 / 30.0, 9.81
TABLE = [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0.6, 0.5, float(np.log(0.5))],
         [0, 0, -0.75, 0, 0, 1, 1, 0, 0, 0, 1, 0, 10.0, 10.0, float(np.log(0.6))]]


def _body(x, area, edges, radius, segs=False, surf_v=None, tris=None):
    """A batch cloth_step reads for one deformable body at positions x (V, 3), float64."""
    V = len(x)
    b = {"is_cloth": torch.ones(V, dtype=torch.bool), "body_of": torch.zeros(V, dtype=torch.long),
         "area": torch.as_tensor(area, dtype=torch.float64), "kind": torch.zeros(V, dtype=torch.long),
         "sample_of": torch.zeros(V, dtype=torch.long),
         "tris": torch.zeros(0, 3, dtype=torch.long) if tris is None else torch.as_tensor(tris),
         "normal": torch.zeros(V, 3, dtype=torch.float64), "surface": torch.zeros(V, 2, dtype=torch.float64),
         "spacing": torch.zeros(V, dtype=torch.float64), "point_contact": torch.zeros(V, dtype=torch.bool),
         "radius": torch.tensor([radius], dtype=torch.float64), "planes": torch.tensor(TABLE, dtype=torch.float64),
         "plane_sample": torch.zeros(2, dtype=torch.long), "n_bodies": 1,
         "surf_v": torch.ones(V, dtype=torch.bool) if surf_v is None else torch.as_tensor(surf_v)}
    E = torch.as_tensor(edges, dtype=torch.long)
    b["segs"] = E if segs else torch.zeros(0, 2, dtype=torch.long)
    b["seg_r"] = torch.full((len(b["segs"]),), radius, dtype=torch.float64)
    return b


def _run(b, x0, v0, slot0, mu, n):
    from lsim.model.cloth_net import cloth_step
    xp, x = x0 - v0 * DT, x0.clone()
    zero = torch.zeros_like(x)
    none = torch.zeros(len(x), dtype=torch.bool)
    for _ in range(n):
        x1 = cloth_step(b, xp, x, zero, torch.tensor([slot0], dtype=torch.float64), none, x,
                        torch.tensor([mu], dtype=torch.float64))
        xp, x = x, x1
    return x, (x - xp) / DT


def _rope(n=41, l=0.01, axis=(1.0, 0.0, 0.0), z=1.0, r=0.002):
    x = torch.arange(n, dtype=torch.float64)[:, None] * l * torch.tensor(axis, dtype=torch.float64)
    x = x - x.mean(0) + torch.tensor([0.0, 0.0, z], dtype=torch.float64)
    area = np.full(n, l)
    area[[0, -1]] = l / 2
    return x, area, np.stack([np.arange(n - 1), np.arange(1, n)], 1), r


def _soft(n=6, h=0.01, z0=None, r=0.001):
    """A cube of n^3 vertices (h apart), its tet-free edges along the lattice; surface = the outer shell."""
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    x = torch.as_tensor(g * h, dtype=torch.float64)
    x = x - x.mean(0)
    x[:, 2] += (r - x[:, 2].min()) if z0 is None else z0
    idx = np.arange(len(g)).reshape(n, n, n)
    e = np.concatenate([np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1), np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                        np.stack([idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel()], 1)])
    surf = ((g == 0) | (g == n - 1)).any(1)
    return x, np.full(len(g), h ** 3), e, r, surf


def test_free_fall_every_material():
    """With the network's outputs at zero a rope falling end-down (no normal velocity: no drag), a soft body (no air
    drag in its teacher) and a cloth falling edge-on all fall at g: x = x0 - g t (t + dt) / 2."""
    from lsim.train.cloth_windows import _grid
    t = 15 * DT
    x, area, e, r = _rope(axis=(0.0, 0.0, 1.0), z=3.0)
    cases = [(_body(x, area, e, r, segs=True), x, 0.01)]
    x, area, e, r, surf = _soft(z0=3.0)
    cases.append((_body(x, area, e, r, surf_v=surf), x, 100.0))
    v, tri, e, area = _grid(10, 10, 0.01)
    x = torch.as_tensor(v, dtype=torch.float64)[:, [0, 2, 1]] + torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
    cases.append((_body(x, area, e, 0.003, tris=torch.as_tensor(tri)), x, 0.2))
    for b, x0, s0 in cases:
        xe, v = _run(b, x0, torch.zeros(3, dtype=torch.float64), s0, 0.5, 15)
        assert abs((xe[:, 2] - x0[:, 2]).mean().item() + G * t * (t + DT) / 2) < 1e-6
        assert abs(v[:, 2].mean().item() + G * t) < 1e-6


def test_rope_terminal_speed():
    """A horizontal rope falling across the air reaches the cylinder's terminal speed sqrt(lam g / (1/2 rho Cd 2 r))
    (lsim.teacher.rope_teacher._air_drag): the exact layer carries the rope teacher's drag. 45 frames: with no
    tension (the network's outputs at zero) a falling chain of free nodes is unstable in shape -- float noise of 1e-11
    m/s at 40 frames grows to m/s by 120 -- which the rope's own tension prevents."""
    from lsim.model.cloth_net import RHO_CD
    lam = 2e-3
    x, area, e, r = _rope(z=20.0, r=0.002)
    _, v = _run(_body(x, area, e, r, segs=True), x, torch.zeros(3, dtype=torch.float64), lam, 0.5, 45)
    want = np.sqrt(lam * G / (0.5 * RHO_CD * 2 * r))
    assert abs(-v[:, 2].mean().item() / want - 1) < 0.002 and v[:, 2].std() < 1e-6


def test_rest_and_slide_on_the_table():
    """Network outputs at zero: a rope lying on the table stays put (to 1e-9 m over a second); one sliding along its
    length stops by Coulomb's law (v^2 / 2 mu g, mu the mean of its and the table's); a soft cube's bottom face stays
    on the table while its inside -- held by internal forces, the network's -- falls freely and never through it."""
    x, area, e, r = _rope(z=0.0)
    x[:, 2] = r
    b = _body(x, area, e, r, segs=True)
    xe, v = _run(b, x, torch.zeros(3, dtype=torch.float64), 0.01, 0.5, 30)
    assert (xe - x).abs().max() < 1e-9 and v.abs().max() < 1e-9
    v0 = 0.6
    xe, v = _run(b, x, torch.tensor([v0, 0.0, 0.0], dtype=torch.float64), 0.01, 0.3, 45)
    a = 0.5 * (0.3 + 0.5) * G                                            # each frame takes mu g dt off its speed
    want = sum(max(v0 - k * a * DT, 0.0) * DT for k in range(1, 46))     # (v^2 / 2 mu g in the limit dt -> 0)
    assert v.abs().max() < 1e-9
    # 0.4% off: the layer's drag acts on the step's velocity before contact, gravity's -g dt included, so the normal
    # impulse friction is capped by is 0.3% under m g dt for this rope (lam 0.01, r 2 mm), as for cloth
    assert abs((xe[:, 0] - x[:, 0]).mean().item() / want - 1) < 0.006
    x, area, e, r, surf = _soft()
    b = _body(x, area, e, r, surf_v=surf)
    x1, _ = _run(b, x, torch.zeros(3, dtype=torch.float64), 100.0, 0.5, 1)
    bottom = x[:, 2] < r + 1e-9
    inner = ~torch.as_tensor(surf)
    assert (x1[bottom] - x[bottom]).abs().max() < 1e-12
    assert torch.allclose(x1[inner, 2] - x[inner, 2], torch.full_like(x1[inner, 2], -G * DT * DT))   # g dt^2 = 1.1 cm:
    assert x1[~inner, 2].min() >= r - 1e-9                               # the second layer's inside passes the table
    x2, _ = _run(b, x, torch.zeros(3, dtype=torch.float64), 100.0, 0.5, 20)
    assert x2[torch.as_tensor(surf), 2].min() >= r - 1e-9                # the surface never passes the table
    assert x2[inner, 2].min() < r - 1e-3              # the inside touches nothing (no internal forces at zero)


def test_noise_scale_per_material_and_val_choice():
    """--noise_rel: a cloth's chunk noise exactly as before (scale 1), a rope's and a soft body's by their median rest
    edge over the cloth's 1 cm; --val alternatives take the first glob that has shards."""
    from lsim.train.cloth_train import CLOTH_EDGE, noise_scale, val_choice
    b = _mixed()
    f = noise_scale(b)
    for k in range(3):
        fk = f[b["body_of"] == k]
        assert (fk == fk[0]).all()
        if k == 0:
            assert fk[0] == 1.0
        else:
            E = b["edges"][b["body_of"][b["edges"][:, 0]] == k]
            L0 = (b["rest"][E[:, 0]] - b["rest"][E[:, 1]]).norm(dim=-1) * b["size"][k]
            assert torch.isclose(fk[0], L0.median() / CLOTH_EDGE)
    assert val_choice("nowhere/shard_*.npz|" + SHARDS[0]) == (SHARDS[0],)
    assert val_choice(SHARDS[0] + ",x/*.npz|y/*.npz") == (SHARDS[0], "y/*.npz")


# ---------------------------------------------------------------------------------------------------------------------
# The generative network (--gen energy --noise_dim): a noise vector per body per frame through conditional RMSNorm
# ---------------------------------------------------------------------------------------------------------------------
def _ready(b):
    from lsim.train.cloth_train import kin_next
    b.update(tau=torch.zeros(b["pos"].shape[1]), frame_given=torch.ones(b["pos"].shape[1]))
    b["kin_next"] = kin_next(b, torch.cat([b["pos"], b["pos"][:, -1:]], 1))
    return b


def _gen_from(c_args, state):
    m = build_cloth_model(dict(c_args, noise_dim=32), ckpt=False, bf16=False).eval()
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("noise_in") or ".mod." in k for k in missing), missing
    return m


def test_gen_init_is_the_deterministic_model_bit_for_bit():
    """A deterministic checkpoint loaded into the generative network: the same outputs, bit for bit, whatever noise
    is drawn (the conditional norms' maps start at zero)."""
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    torch.manual_seed(0)
    for ck in ("outputs/lsim/cloth_q64/step23999.pt", None):
        if ck is not None and not (ROOT / ck).exists():
            continue
        if ck is None:
            det = build_cloth_model(A, ckpt=False, bf16=False).eval()
            args, state = A, det.state_dict()
        else:
            c = torch.load(ROOT / ck, map_location="cpu", weights_only=False)
            args, state = c["args"], c["model"]
            det = build_cloth_model(args, ckpt=False, bf16=False).eval()
            det.load_state_dict(state)
        gen = _gen_from(args, state)
        b = _ready(_batch())
        with torch.no_grad():
            ref = det(b)
            for seed in (1, 2):
                torch.manual_seed(seed)
                out = gen(b)
                for r, o in zip(ref[:3], out[:3]):
                    assert torch.equal(r, o), ck


def test_gen_samples_differ_and_seed_and_zero_noise():
    from lsim.model.cloth_qwen import set_noise
    m = build_cloth_model(dict(A, noise_dim=32), ckpt=False, bf16=False).eval()
    torch.manual_seed(0)
    for blk in m.blocks:                                             # as if trained: the noise acts
        torch.nn.init.normal_(blk.mod.weight, std=0.05)
    m.f_int.weight.data.normal_(0, 0.05)
    b = _ready(_batch())
    with torch.no_grad():
        a1, a2 = m(b)[0], m(b)[0]
        assert not torch.equal(a1, a2)
        set_noise(m, seed=5)
        s1 = m(b)[0]
        set_noise(m, seed=5)
        assert torch.equal(s1, m(b)[0])
        set_noise(m, zero=True)
        z1, z2 = m(b)[0], m(b)[0]
        assert torch.equal(z1, z2) and not torch.equal(z1, s1)
    m2 = build_cloth_model(A, ckpt=False, bf16=False)                # a deterministic model ignores set_noise
    set_noise(m2, seed=1)


def test_gen_energy_losses_train_every_parameter():
    from lsim.train.cloth_real import rollout_loss
    from lsim.train.cloth_train import _load, step_loss
    m = build_cloth_model(dict(A, noise_dim=32), ckpt=False, bf16=False)
    eps = _load(SHARDS, set())
    for k in range(2):                                               # the maps start at zero: one step on
        loss, info = step_loss(m, _batch(seed=k), np.random.default_rng(k), 8, "cpu", sig=1e-3, edge_w=1.0, gen="energy")
        lr, ri = rollout_loss(m, eps, np.random.default_rng(k), 6, "cpu", n=1, detach_every=3, gen="energy")
        (loss + lr).backward()
        assert torch.isfinite(loss) and torch.isfinite(lr) and "es_spread_mm" in info and "roll_spread_mm" in ri
        if k == 0:
            torch.optim.SGD(m.parameters(), lr=1e-2).step()
            m.zero_grad()
    assert [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0] == []
    assert ri["roll_spread_mm"] > 0


def test_gen_rollout_copies_are_identical_without_noise():
    """The long-rollout energy score's two copies of a start: identical inputs, so with zero noise identical
    rollouts (spread exactly 0) -- the spread measures the noise alone."""
    from lsim.model.cloth_qwen import set_noise
    from lsim.train.cloth_real import rollout_loss
    from lsim.train.cloth_train import _load
    m = build_cloth_model(dict(A, noise_dim=32), ckpt=False, bf16=False)
    for blk in m.blocks:
        torch.nn.init.normal_(blk.mod.weight, std=0.05)
    m.f_int.weight.data.normal_(0, 0.05)
    set_noise(m, zero=True)
    eps = _load(SHARDS, set())
    with torch.no_grad():
        for k in range(3):
            _, ri = rollout_loss(m, eps, np.random.default_rng(k), 4, "cpu", n=1, gen="energy")
            assert ri["roll_spread_mm"] <= 1e-3 + 1e-9                  # exactly the 1e-12 under the root: 1e-6 m
