"""The training physics layer starts each chunk from two poses (frame-average velocities): for a
trajectory made by the integrator itself, frame_velocity must return exactly the integrator's
velocity state, for any centre of mass -- as long as a body turns less than half a revolution per
frame (pi rad per 1/30 s = 94 rad/s; beyond that two poses cannot tell the rotations apart)."""
import torch

from lsim.model.physics import qrot, rigid_step
from lsim.train.train import DT, frame_velocity


def test_frame_velocity_recovers_integrator_state():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    try:
        B = 16
        c = torch.randn(B, 3) * 0.02                                  # CoM offsets in the body frame
        Q, _ = torch.linalg.qr(torch.randn(B, 3, 3))                  # inertia about the CoM: principal
        I = Q @ torch.diag_embed(10 ** (-4 + torch.rand(B, 3))) @ Q.transpose(-1, -2)   # moments 1e-4..1e-3
        m = torch.rand(B) + 0.1
        q = torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
        p = torch.randn(B, 3) * 0.1
        v, w = torch.randn(B, 3), torch.randn(B, 3) * 10
        poses = [torch.cat([p - qrot(q, c), q], -1)]                  # origin pose = CoM - R c
        states = [(v, w)]
        for _ in range(6):
            p, q, v, w = rigid_step(p, q, v, w, torch.randn(B, 3), torch.randn(B, 3) * 1e-4, m, I, DT, clamp=False)
            assert (w.norm(dim=-1) * DT < 3.0).all()
            poses.append(torch.cat([p - qrot(q, c), q], -1))
            states.append((v, w))
        for k in range(1, 7):
            vf, wf = frame_velocity(poses[k - 1], poses[k], c)
            assert torch.allclose(vf, states[k][0], atol=1e-9), (vf - states[k][0]).abs().max()
            assert torch.allclose(wf, states[k][1], atol=1e-9), (wf - states[k][1]).abs().max()
    finally:
        torch.set_default_dtype(torch.float32)
