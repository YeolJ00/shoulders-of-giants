"""The physics layer against closed-form motion (rule 2: two routes to one quantity)."""
import math

import torch

from lsim.model.physics import prismatic_step, qrot, revolute_step, rigid_step, wrench

DT = 1.0 / 30.0


def test_free_fall():
    pos, q = torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    v, w = torch.zeros(1, 3), torch.zeros(1, 3)
    I = torch.eye(3)[None] * 1e-3
    for _ in range(30):                                               # 1 s
        pos, q, v, w = rigid_step(pos, q, v, w, torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1), I, DT)
    assert abs(v[0, 2].item() + 9.81) < 1e-4
    assert abs(pos[0, 2].item() + 0.5 * 9.81 * (1 + DT)) < 1e-3          # semi-implicit Euler's offset


def test_wrench_pure_torque():
    x = torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    f = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    F, T = wrench(f, x, torch.tensor([0, 0]), torch.zeros(1, 3), 1)
    assert torch.allclose(F, torch.zeros(1, 3)) and torch.allclose(T, torch.tensor([[0.0, 0.0, 0.2]]))


def test_spin_up_matches_tau_over_I():
    pos, q = torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    v, w = torch.zeros(1, 3), torch.zeros(1, 3)
    I = torch.diag(torch.tensor([1.0, 2.0, 4.0]))[None] * 1e-3
    for _ in range(30):
        pos, q, v, w = rigid_step(pos, q, v, w, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 4e-3]]), torch.ones(1), I, DT,
                                  gravity=False)
    assert abs(w[0, 2].item() - 1.0) < 1e-4                           # tau / Izz * t = 1 rad/s
    ang = 2 * math.atan2(q[0, 2].item(), q[0, 3].item())
    assert abs(ang - 0.5 * (1 + DT)) < 2e-3


def test_rotation_moves_vertices():
    q = torch.tensor([[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])   # 90 deg about z
    assert torch.allclose(qrot(q, torch.tensor([[1.0, 0.0, 0.0]])), torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_prismatic_damped_terminal_velocity_and_limit():
    q, qd = torch.zeros(1), torch.zeros(1)
    for _ in range(90):                                               # 3 s at F 2 N, c 5 N s/m, m 1 kg
        q, qd = prismatic_step(q, qd, torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([5.0]), DT, 0.0, 10.0)
    assert abs(qd.item() - 0.4) < 2e-3
    for _ in range(90):
        q, qd = prismatic_step(q, qd, torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([5.0]), DT, 0.0, 1.2)
    assert abs(q.item() - 1.2) < 1e-6 and qd.item() == 0.0


def test_revolute_damped():
    q, qd = torch.zeros(1), torch.zeros(1)
    for _ in range(300):
        q, qd = revolute_step(q, qd, torch.tensor([0.1]), torch.tensor([0.01]), torch.tensor([0.5]), DT, 0.0, 100.0)
    assert abs(qd.item() - 0.2) < 2e-3


def test_torque_free_spin_stays_bounded():
    """An anisotropic body spinning at 30 rad/s at 30 Hz: the explicit gyroscopic term diverged
    (w dt 1.1 -> 5781 in three steps); the implicit one keeps |w| near its start."""
    torch.manual_seed(0)
    B = 32
    Q, _ = torch.linalg.qr(torch.randn(B, 3, 3))
    I = Q @ torch.diag_embed(10 ** (-4 + 1.3 * torch.rand(B, 3))) @ Q.transpose(-1, -2)    # anisotropy up to 20
    pos, q = torch.zeros(B, 3), torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
    v, w = torch.zeros(B, 3), torch.nn.functional.normalize(torch.randn(B, 3), dim=-1) * 30.0
    for _ in range(90):
        pos, q, v, w = rigid_step(pos, q, v, w, torch.zeros(B, 3), torch.zeros(B, 3), torch.ones(B), I, DT,
                                  gravity=False, clamp=False)
        assert torch.isfinite(w).all() and (w.norm(dim=-1) < 30.0 * 20).all()


def test_joint_step_lid_falls_and_stops():
    """A lid hinged at its back edge falls shut under gravity and stops at its limit; a drawer
    pushed along its axis moves exactly F dt^2 / m in one step from rest."""
    from lsim.model.physics import joint_coord, joint_step
    kind = torch.tensor([2, 1])
    axis = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    pivot = torch.zeros(2, 3)
    lim = torch.tensor([[0.0, 1.9], [0.0, 0.2]])
    a = 0.6                                                              # lid open 0.6 rad
    q = torch.tensor([[0.0, 0.0, 0.0, -math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    com = torch.tensor([[0.0, -0.1, 0.0], [0.0, 0.0, 0.0]])
    I = torch.eye(3)[None].repeat(2, 1, 1) * 1e-4
    m = torch.tensor([0.5, 0.3])
    F = torch.tensor([[0.0, 0.0, 0.0], [0.0, -3.0, 0.0]])
    assert abs(joint_coord(kind, axis, pivot, q)[0].item() - a) < 1e-6
    q1 = joint_step(kind, axis, pivot, lim, q, q, F, torch.zeros(2, 3), m, com, I, torch.zeros(2), DT)
    assert abs(joint_coord(kind, axis, pivot, q1)[1].item() - 3.0 / 0.3 * DT * DT) < 1e-6
    prev, cur = q, q
    for _ in range(60):
        nxt = joint_step(kind, axis, pivot, lim, prev, cur, torch.zeros(2, 3), torch.zeros(2, 3), m, com, I, torch.zeros(2), DT)
        prev, cur = cur, nxt
    assert abs(joint_coord(kind, axis, pivot, cur)[0].item()) < 1e-6           # shut, at the stop
