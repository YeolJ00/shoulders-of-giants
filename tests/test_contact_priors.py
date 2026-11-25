"""Invariants of the contact-force scaling and the support prior (rule 2), on real v0 windows.

With a head that pushes every contact pair apart with output 1, per-body-pair scaling must give
exactly m g of support to a body resting on the table, whatever its contact count (per-vertex
scaling gave 0.93-4.6 m g); untrained, the support prior must hold a resting body with m g and
zero torque about its centre of mass; the batched rigid fit B1/B2 use must match Kabsch."""
import glob
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
SHARDS = sorted(glob.glob(str(REPO / "data/datasets/v0/shard_*.npz")))


class Lam(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def _batch():
    from lsim.train.train import collate
    from lsim.train.windows import load_shard, make_window
    eps = [e for e in load_shard(SHARDS[0][:-4]) if any(b.role == "dynamic" for b in e.bodies)][:6]
    rng = np.random.default_rng(0)
    wins = [make_window(e, 5, 8, 0, rng, p_hide_grip=0.0) for e in eps]       # objects at rest after settling
    return collate(wins, torch.device("cpu"), rng, p_hidden=0.0, p_hist_noise=0.0, augment=False, all_past=True)


def _contact_forces(model, b):
    Fw = b["pos"].shape[1]
    with torch.no_grad():
        f, _, pp = model(dict(b, tau=torch.zeros(Fw), frame_given=torch.ones(Fw)), lambda pr: b["body_mass"])
    F = torch.zeros(b["n_bodies"], 3).index_add_(0, b["body_of"], f[:, -1])
    return f, F, pp


@pytest.mark.skipif(not SHARDS, reason="dataset v0 not generated")
def test_pair_norm_unit_head_gives_weight():
    from lsim.model.net import Simulator
    from lsim.train.windows import PARAMS
    b = _batch()
    m = Simulator(128, 2, n_params=len(PARAMS), pair_norm=True, bf16=False, ckpt=False).eval()
    m.edge_in = Lam(lambda x: x[:, :16])                              # raw edge features through
    d = m.d
    m.force = Lam(lambda x: torch.stack([torch.zeros_like(x[:, 0]), torch.zeros_like(x[:, 0]),
                                         torch.sign(x[:, 2 * d + 2])], -1))   # +1 along z, apart
    _, F, _ = _contact_forces(m, b)
    rig = ((b["body_role"] == 2) & (b["body_type"] == 0)).nonzero()[:, 0]
    ratio = F[rig, 2] / (9.81 * b["body_mass"][rig])
    touching = ratio.abs() > 1e-6
    assert touching.sum() >= 10
    assert torch.allclose(ratio[touching], torch.ones_like(ratio[touching]), atol=1e-4)


@pytest.mark.skipif(not SHARDS, reason="dataset v0 not generated")
def test_support_prior_holds_with_zero_torque():
    from lsim.model.net import Simulator
    from lsim.model.physics import qrot
    from lsim.train.windows import PARAMS
    b = _batch()
    m = Simulator(128, 2, n_params=len(PARAMS), pair_norm=True, support=True, bf16=False, ckpt=False).eval()
    f, F, pp = _contact_forces(m, b)
    com = m._com_world(b, b["pos"], pp)[:, -1]
    q = b["body_q_now"]
    rig = ((b["body_role"] == 2) & (b["body_type"] == 0)).nonzero()[:, 0]
    assert (com - (q[:, :3] + qrot(q[:, 3:], b["body_com"])))[rig].norm(dim=-1).max() < 1e-5   # CoM from vertices
    x = b["pos"][:, -1]
    T = torch.zeros(b["n_bodies"], 3).index_add_(0, b["body_of"], torch.cross(x - com[b["body_of"]], f[:, -1], dim=-1))
    mg = 9.81 * b["body_mass"][rig]
    held = F[rig, 2] / mg > 0.5
    assert held.sum() >= 10
    assert torch.allclose(F[rig[held], 2] / mg[held], torch.ones(int(held.sum())), atol=1e-4)
    tq = T[rig[held]].norm(dim=-1) / (mg[held] * b["body_size"][rig[held]])
    assert (tq < 1e-3).float().mean() >= 0.8                           # a CoM outside its contacts keeps a torque


def test_rigid_project_matches_kabsch():
    from lsim.model.rollout import rigid_fit
    from lsim.train.train import rigid_project
    rng = np.random.default_rng(0)
    vl, x, body = [], [], []
    for k in range(3):
        loc = rng.normal(size=(40, 3)) * 0.05
        R, t = Rotation.random(random_state=k).as_matrix(), rng.normal(size=3)
        vl.append(loc); x.append(loc @ R.T + t + rng.normal(size=(40, 3)) * 0.002); body += [k] * 40
    vl, x, body = np.concatenate(vl), np.concatenate(x), np.array(body)
    xp = rigid_project(torch.tensor(x), torch.tensor(vl), torch.tensor(body), 3, torch.ones(len(x), dtype=torch.bool)).numpy()
    for k in range(3):
        q = rigid_fit(vl[body == k], x[body == k])
        ref = Rotation.from_quat(q[3:]).apply(vl[body == k]) + q[:3]
        assert np.abs(ref - xp[body == k]).max() < 1e-6


def test_coulomb_default_slides_at_the_closed_form():
    """Untrained, support='coulomb': a cube resting on a 20 deg plank accelerates down the slope at
    g (sin - mu cos) below tan 20 = 0.364 and sticks above it (support=True holds it at any mu)."""
    from lsim.data.scene import ramp_frame
    from lsim.model.net import Simulator
    from lsim.real.synthetic import ramp_cube
    from lsim.train.train import collate
    from lsim.train.windows import PARAMS, make_window
    th = np.radians(20.0)
    m = Simulator(128, 2, n_params=len(PARAMS), pair_norm=True, support="coulomb", bf16=False, ckpt=False).eval()
    for mu in (0.15, 0.30, 0.50):
        ep, ramp = ramp_cube(0.05, 0.1, mu, 20.0, 0.30, 12)
        w = make_window(ep, 8, 8, 0, np.random.default_rng(0), p_hide_grip=0.0)
        b = collate([w], torch.device("cpu"), np.random.default_rng(0), p_hidden=0.0, p_hist_noise=0.0, augment=False,
                    all_past=True)
        _, F, _ = _contact_forces(m, b)
        a = F[0].numpy() / float(b["body_mass"][0]) + np.array([0.0, 0.0, -9.81])
        _, ax, lat, n = ramp_frame(ramp)
        assert abs(a @ (-ax) - 9.81 * max(0.0, np.sin(th) - mu * np.cos(th))) < 1e-3
        assert abs(a @ n) < 1e-3 and abs(a @ lat) < 1e-3
