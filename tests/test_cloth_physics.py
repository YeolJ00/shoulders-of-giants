"""The cloth physics layer (lsim.model.cloth_net.cloth_step) against closed forms, with the network's forces at zero:
free fall, the terminal speed of a falling sheet, rest on the table, a Coulomb slide, held vertices; and with a
network that holds a sheet up itself (friction counts it) or a sheet sunk into the table (pushed out)."""
import math

import numpy as np
import torch

from lsim.model.cloth_net import RHO_CD, cloth_step
from lsim.teacher.cloth_teacher import grid
from lsim.train.cloth_windows import KIND, _grid

from lsim.model.cloth_net import DT                    # 1 / 30 s (LSIM_FPS: the layer at another frame rate)
G = 9.81


def sheet(z, n=10, h=0.01, w=0.2, mu_table=0.5, ramp=None):
    """One flat n x n sheet at height z (vertex centres) over the table top, as a batch cloth_step reads."""
    v, t, e, area = _grid(n, n, h)
    x = torch.tensor(v, dtype=torch.float64) - torch.tensor([n * h / 2, n * h / 2, 0.0], dtype=torch.float64)
    x[:, 2] = z
    V = len(v)
    planes = [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0.6, 0.5, math.log(mu_table)],
              [0, 0, -0.75, 0, 0, 1, 1, 0, 0, 0, 1, 0, 10.0, 10.0, math.log(0.6)]]
    b = {"is_cloth": torch.ones(V, dtype=torch.bool), "body_of": torch.zeros(V, dtype=torch.long),
         "area": torch.tensor(area, dtype=torch.float64), "kind": torch.full((V,), KIND["cloth"]),
         "sample_of": torch.zeros(V, dtype=torch.long), "tris": torch.tensor(t), "normal": torch.zeros(V, 3, dtype=torch.float64),
         "surface": torch.zeros(V, 2, dtype=torch.float64), "spacing": torch.zeros(V, dtype=torch.float64),
         "point_contact": torch.zeros(V, dtype=torch.bool), "radius": torch.tensor([0.003], dtype=torch.float64),
         "planes": torch.tensor(planes, dtype=torch.float64), "plane_sample": torch.zeros(2, dtype=torch.long), "n_bodies": 1}
    return b, x, torch.tensor([w], dtype=torch.float64)


def run(b, x0, v0, w, mu, n, given=None, xg=None):
    xp, x = x0 - v0 * DT, x0.clone()
    zero = torch.zeros_like(x)
    none = torch.zeros(len(x), dtype=torch.bool) if given is None else given
    for k in range(n):
        x1 = cloth_step(b, xp, x, zero, w, none, x if xg is None else xg(k + 1), torch.tensor([mu], dtype=torch.float64))
        xp, x = x, x1
    return x, (x - xp) / DT


def test_free_fall_edge_on():
    """A sheet falling edge-on (no normal velocity: no drag) falls at g: x = x0 - g t (t + dt) / 2."""
    b, x, w = sheet(3.0)
    R = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], dtype=torch.float64)   # its plane vertical
    x = (x - x.mean(0)) @ R.T + torch.tensor([0, 0, 3.0], dtype=torch.float64)
    xe, v = run(b, x, torch.zeros(3, dtype=torch.float64), w, 0.5, 15)
    t = 15 * DT
    assert abs((xe[:, 2] - x[:, 2]).mean().item() + G * t * (t + DT) / 2) < 1e-6
    assert abs(v[:, 2].mean().item() + G * t) < 1e-6


def test_terminal_speed_face_down():
    """A sheet falling face-down reaches sqrt(2 w g / (rho Cd)) (the teacher's airfall check)."""
    b, x, w = sheet(8.0)
    _, v = run(b, x, torch.zeros(3, dtype=torch.float64), w, 0.5, 90)
    want = math.sqrt(2 * 0.2 * G / RHO_CD)
    assert abs(-v[:, 2].mean().item() / want - 1) < 0.02


def test_rest_on_table():
    """A sheet at its radius above the table stays: the Coulomb default holds it against gravity."""
    b, x, w = sheet(0.003)
    xe, v = run(b, x, torch.zeros(3, dtype=torch.float64), w, 0.5, 60)
    assert (xe - x).norm(dim=-1).max().item() < 1e-9


def test_lands_without_bouncing():
    """Dropped from 5 cm it ends at its radius above the table, not below, and does not rise again."""
    b, x, w = sheet(0.053)
    xp, xc = x.clone(), x.clone()
    zs = []
    for k in range(30):
        x1 = cloth_step(b, xp, xc, torch.zeros_like(xc), w, torch.zeros(len(x), dtype=torch.bool), xc,
                        torch.tensor([0.5], dtype=torch.float64))
        xp, xc = xc, x1
        zs.append(xc[:, 2].mean().item())
    k = int(np.argmin(zs))
    assert abs(zs[-1] - 0.003) < 1e-9 and max(zs[k:]) - zs[-1] < 1e-9 and min(zs) > 0.003 - 1e-9


def test_coulomb_slide():
    """A sheet sliding on the table at 0.5 m/s stops after v^2 / (2 mu g), mu the mean of the cloth's and the
    table's (0.3 and 0.5): within one frame's travel."""
    b, x, w = sheet(0.003, mu_table=0.5)
    v0 = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    xe, v = run(b, x, v0, w, 0.3, 60)
    want = 0.5 ** 2 / (2 * 0.4 * G)
    got = (xe[:, 0] - x[:, 0]).mean().item()
    assert v.norm(dim=-1).max().item() < 1e-9
    assert abs(got - want) < 0.5 * DT


def test_held_vertices_follow():
    """Held vertices end where they are given; the rest fall."""
    b, x, w = sheet(3.0)
    held = torch.zeros(len(x), dtype=torch.bool)
    held[:3] = True
    path = lambda k: x + torch.tensor([0.0, 0.0, 0.01 * k], dtype=torch.float64)
    xe, _ = run(b, x, torch.zeros(3, dtype=torch.float64), w, 0.5, 10, given=held, xg=path)
    assert torch.allclose(xe[:3], path(10)[:3])
    assert (xe[3:, 2] < x[3:, 2]).all()


def test_friction_counts_the_networks_support():
    """A sheet on the table whose weight the network's contact force carries (95%) and that it pushes sideways at
    0.3 g, below mu = 0.4 (the mean of 0.3 and 0.5): the table's friction holds it, because friction is capped by the
    surface's whole normal impulse. Capped by the default's share alone (layer 1) it slides away."""
    b, x, w = sheet(0.003, mu_table=0.5)
    m = (w[0] * b["area"])[:, None]
    f = m * G * torch.tensor([0.3, 0.0, 0.95], dtype=torch.float64)
    none = torch.zeros(len(x), dtype=torch.bool)
    for f_contact, push_out, still in ((f, 0.01, True), (None, 0.0, False)):
        xp, xc = x.clone(), x.clone()
        for k in range(30):
            x1 = cloth_step(b, xp, xc, f, w, none, xc, torch.tensor([0.3], dtype=torch.float64), f_contact=f_contact,
                            push_out=push_out)
            xp, xc = xc, x1
        moved = (xc - x).norm(dim=-1).max().item()
        assert (moved < 1e-9) if still else (moved > 0.1)


def test_pushed_out_of_the_table():
    """A sheet 3 mm into the table ends at its radius above it after one step, and stays there."""
    b, x, w = sheet(0.0)
    xp, xc = x.clone(), x.clone()
    zs = []
    for k in range(10):
        x1 = cloth_step(b, xp, xc, torch.zeros_like(xc), w, torch.zeros(len(x), dtype=torch.bool), xc,
                        torch.tensor([0.5], dtype=torch.float64))
        xp, xc = xc, x1
        zs.append(xc[:, 2].mean().item())
    assert abs(zs[0] - 0.003) < 1e-9 and max(abs(z - 0.003) for z in zs) < 1e-9

