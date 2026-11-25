"""The exact layer's shape restoration (lsim.model.cloth_net.restore_shape, 2026-11-03), against invariants, with the
network's outputs at zero where the layer alone should be exact (CLAUDE.md "run every change to the layer with the
network's outputs at zero on resting starts"):

  - a constraint inside its band is untouched (bit for bit); outside, it comes back near the band; held vertices fixed
  - no net impulse: sum m dx = 0 over a body no one holds (edge and volume corrections are internal forces)
  - a squashed or inverted tet comes back inside its volume band
  - differentiable (gradcheck)
  - zero output, restoration on: a flat towel and a straight rope on the table stay to 1e-9 m; an overhanging towel
    does not slide off; a hanging strip / rope at its closed-form extension w g L^2 / 2Y is left exactly there (and a
    band tighter than that extension would wreck it: the test can fail)
  - forces right on average but noisy by 3 mm a frame for 10 s (an imprecise network): towel, rope and soft block keep
    their edges and tets near their bands, where without the restoration they stretch 8-18x and tets invert

Not tested, because it does not hold: a soft body with the network's outputs at ZERO. Its inside falls g dt^2 = 11 mm
a frame, more than a tet edge (s1: ~6 mm), and the restoration (10 sweeps) cannot undo inversions that deep (a 5^3
block: tets down to -1.6x volume). Holding a soft body's inside up is the network's job (DIRECTION.md "A gap the cloth
never had"); the restoration catches a few mm per frame, not free fall.
"""
import math

import numpy as np
import torch

from lsim.model.cloth_net import RESTORE_BANDS, cloth_step, restore_config, restore_shape
from lsim.train.cloth_windows import KIND, _grid, greedy_colors

DT, G = 1.0 / 30.0, 9.81
F64 = torch.float64
TABLE = [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0.6, 0.5, math.log(0.5)],
         [0, 0, -0.75, 0, 0, 1, 1, 0, 0, 0, 1, 0, 10.0, 10.0, math.log(0.6)]]


def body(x, area, edges, radius, material=0, tris=None, tets=None, surf_v=None, segs=False):
    """A batch cloth_step and restore_shape read for one deformable body at rest positions x (V, 3)."""
    V = len(x)
    x = torch.as_tensor(x, dtype=F64)
    size = float((x.max(0).values - x.min(0).values).max())
    E = torch.as_tensor(np.asarray(edges), dtype=torch.long)
    b = {"is_cloth": torch.ones(V, dtype=torch.bool), "body_of": torch.zeros(V, dtype=torch.long),
         "area": torch.as_tensor(area, dtype=F64), "kind": torch.full((V,), KIND["cloth"]),
         "sample_of": torch.zeros(V, dtype=torch.long),
         "tris": torch.zeros(0, 3, dtype=torch.long) if tris is None else torch.as_tensor(tris),
         "normal": torch.zeros(V, 3, dtype=F64), "surface": torch.zeros(V, 2, dtype=F64),
         "spacing": torch.zeros(V, dtype=F64), "point_contact": torch.zeros(V, dtype=torch.bool),
         "radius": torch.tensor([radius], dtype=F64), "planes": torch.tensor(TABLE, dtype=F64),
         "plane_sample": torch.zeros(2, dtype=torch.long), "n_bodies": 1,
         "surf_v": torch.ones(V, dtype=torch.bool) if surf_v is None else torch.as_tensor(surf_v),
         "edges": E, "rest": (x - x.mean(0)) / size, "size": torch.tensor([size], dtype=F64),
         "material": torch.tensor([material]), "edge_color": torch.as_tensor(greedy_colors(E.numpy()))}
    b["segs"] = E if segs else torch.zeros(0, 2, dtype=torch.long)
    b["seg_r"] = torch.full((len(b["segs"]),), radius, dtype=F64)
    T = np.zeros((0, 4), np.int64) if tets is None else np.asarray(tets)
    b["tets"] = torch.as_tensor(T, dtype=torch.long)
    b["tet_color"] = torch.as_tensor(greedy_colors(T))
    return b


def sheet(n=10, ny=None, h=0.01, z=0.003):
    ny = n if ny is None else ny
    v, t, e, area = _grid(n, ny, h)
    x = torch.tensor(v, dtype=F64) - torch.tensor([n * h / 2, ny * h / 2, 0.0], dtype=F64)
    x[:, 2] = z
    return x, area, e, t


def rope(n=41, l=0.01, r=0.002):
    x = torch.zeros(n, 3, dtype=F64)
    x[:, 0] = torch.arange(n, dtype=F64) * l
    x[:, 0] -= x[:, 0].mean()
    x[:, 2] = r
    area = np.full(n, l)
    area[[0, -1]] = l / 2
    return x, area, np.stack([np.arange(n - 1), np.arange(1, n)], 1), r


def block(n=5, h=0.01, r=0.001):
    """A cube lattice of n^3 vertices h apart on the table, each cell split into 6 tets (Kuhn); tet edges; the outer
    shell as its surface."""
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    idx = np.arange(len(g)).reshape(n, n, n)
    tets = []
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                for p in perms:
                    c = np.array([i, j, k])
                    path = [c.copy()]
                    for ax in p:
                        c = c.copy(); c[ax] += 1; path.append(c)
                    tets.append([idx[tuple(q)] for q in path])
    tets = np.array(tets)
    e = np.unique(np.sort(np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), 1), axis=0)
    x = torch.as_tensor(g * h, dtype=F64)
    x[:, :2] -= x[:, :2].mean(0)
    x[:, 2] += r
    surf = ((g == 0) | (g == n - 1)).any(1)
    vol = np.full(len(g), h ** 3)                                        # lumped volume share (approximate is fine)
    return x, vol, e, tets, r, surf


def strain(b, x):
    E = b["edges"]
    L0 = (b["rest"][E[:, 0]] - b["rest"][E[:, 1]]).norm(dim=-1) * b["size"][0]
    return (x[E[:, 0]] - x[E[:, 1]]).norm(dim=-1) / L0 - 1


def tet_ratio(b, x):
    T = b["tets"]
    def vol(p):
        a = p[T[:, 0]]
        return torch.linalg.det(torch.stack([p[T[:, 1]] - a, p[T[:, 2]] - a, p[T[:, 3]] - a], -1)) / 6
    return vol(x) / vol(b["rest"] * b["size"][0])


def run(b, x0, w, mu, n, v0=None, f=None, given=None, xg=None, restore=RESTORE_BANDS):
    v0 = torch.zeros(3, dtype=F64) if v0 is None else v0
    xp, x = x0 - v0 * DT, x0.clone()
    f = torch.zeros_like(x) if f is None else f
    none = torch.zeros(len(x), dtype=torch.bool) if given is None else given
    for _ in range(n):
        x1 = cloth_step(b, xp, x, f, torch.tensor([w], dtype=F64), none, x if xg is None else xg,
                        torch.tensor([mu], dtype=F64), restore=restore)
        xp, x = x, x1
    return x, (x - xp) / DT


def test_inside_band_untouched_and_outside_restored():
    x, area, e, t = sheet(20)
    b = body(x, area, e, 0.003, tris=t)
    m = torch.as_tensor(area, dtype=F64) * 0.2
    fixed = torch.zeros(len(x), dtype=torch.bool)
    calm = x * torch.tensor([1.006, 0.995, 1.0], dtype=F64)              # inside the cloth band (1% / 1%)
    assert torch.equal(restore_shape(b, calm.clone(), fixed, m), calm)
    fixed[:21] = True                                                    # the first row held
    x0 = x + torch.as_tensor(np.random.default_rng(0).normal(0, 0.001, x.shape))
    x0[:21] = x[:21]
    assert strain(b, x0).abs().max() > 0.2
    x1 = restore_shape(b, x0.clone(), fixed, m)
    movable = ~(fixed[b["edges"][:, 0]] & fixed[b["edges"][:, 1]])
    s = strain(b, x1)[movable]
    assert torch.equal(x1[:21], x0[:21])
    st, co = RESTORE_BANDS["edge"][0]
    assert ((s > st + 0.01) | (s < -co - 0.01)).float().mean() < 0.01   # 10 sweeps: within 1% of the band
    assert s.abs().max() < 0.06                                          # (a 25% jitter in one call)


def test_no_net_impulse():
    """Nothing fixed: every correction is an internal force pair, so sum m dx = 0 -- for edges and for volumes."""
    for make in ("sheet", "block"):
        if make == "sheet":
            x, area, e, t = sheet(12)
            b = body(x, area, e, 0.003, tris=t)
            m = torch.as_tensor(area, dtype=F64) * 0.2
        else:
            x, vol, e, tets, r, surf = block(4)
            b = body(x, vol, e, r, material=2, tets=tets, surf_v=surf)
            m = torch.as_tensor(vol, dtype=F64) * 100
            m = m * torch.linspace(0.5, 2.0, len(m), dtype=F64)             # unequal masses
        x0 = x + torch.as_tensor(np.random.default_rng(1).normal(0, 0.003, x.shape))
        x1 = restore_shape(b, x0.clone(), torch.zeros(len(x), dtype=torch.bool), m)
        assert (x1 - x0).abs().max() > 1e-4
        assert ((x1 - x0) * m[:, None]).sum(0).abs().max() < 1e-12 * m.sum()


def test_tet_volume_restored():
    x, vol, e, tets, r, surf = block(4)
    b = body(x, vol, e, r, material=2, tets=tets, surf_v=surf)
    m = torch.as_tensor(vol, dtype=F64) * 100
    fixed = torch.zeros(len(x), dtype=torch.bool)
    z0 = x[:, 2].min()
    squashed = x.clone()
    squashed[:, 2] = z0 + 0.3 * (x[:, 2] - z0)                           # every tet at 30% of its volume
    x1 = squashed.clone()
    for _ in range(5):
        x1 = restore_shape(b, x1, fixed, m)
    vr = tet_ratio(b, x1)
    vc, ve = RESTORE_BANDS["vol"][2][1], RESTORE_BANDS["vol"][2][0]
    assert vr.min() > 1 - vc - 0.02 and vr.max() < 1 + ve + 0.02
    inv = x.clone()                                                     # one vertex pushed through its tets' face
    k = int(np.flatnonzero(~surf)[0])
    inv[k, 2] -= 0.015
    assert (tet_ratio(b, inv) < 0).any()
    x2 = inv.clone()
    for _ in range(5):
        x2 = restore_shape(b, x2, fixed, m)
    assert tet_ratio(b, x2).min() > 1 - vc - 0.02


def test_differentiable():
    x, area, e, t = sheet(3)
    b = body(x, area, e, 0.003, tris=t)
    m = torch.as_tensor(area, dtype=F64) * 0.2
    fixed = torch.zeros(len(x), dtype=torch.bool)
    fixed[0] = True
    x0 = (x + torch.as_tensor(np.random.default_rng(2).normal(0, 0.002, x.shape))).requires_grad_(True)
    cfg = restore_config("iters=3")
    assert torch.autograd.gradcheck(lambda y: restore_shape(b, y, fixed, m, cfg), (x0,), eps=1e-7, atol=1e-5)


def test_resting_towel_and_rope_stay():
    x, area, e, t = sheet(20, z=0.003)
    b = body(x, area, e, 0.003, tris=t)
    xe, v = run(b, x, 0.2, 0.5, 60)
    assert (xe - x).norm(dim=-1).max() < 1e-9 and v.abs().max() < 1e-9
    x, area, e, r = rope()
    b = body(x, area, e, r, material=1, segs=True)
    xe, v = run(b, x, 0.01, 0.5, 60)
    assert (xe - x).norm(dim=-1).max() < 1e-9 and v.abs().max() < 1e-9


def drape(n=20, k_on=15, h=0.01, r=0.003):
    """An n x n towel draped over the table's edge (x = 0.6 m): columns 0..k_on flat on the table (column k_on on the
    edge), the rest hanging down its side at x = 0.6 + r. Returns the flat rest sheet's batch, the draped positions
    and which vertices lie on the table."""
    x, area, e, t = sheet(n, z=r)
    col = torch.round((x[:, 0] - x[:, 0].min()) / h).long()
    X = torch.zeros_like(x)
    X[:, 1] = x[:, 1]
    on = col <= k_on
    X[on, 0] = 0.6 - (k_on - col[on]).to(F64) * h
    X[on, 2] = r
    j = (col - k_on).clamp_min(0).to(F64)
    X[~on, 0] = 0.6 + r
    X[~on, 2] = r - (h ** 2 - r ** 2) ** 0.5 - (j[~on] - 1) * h
    return body(x, area, e, r, tris=t), X, on, col


def test_overhanging_towel_holds():
    """A towel a quarter over the table's edge, mu 0.5 (the hanging quarter weighs 0.25 of it, friction holds up to
    0.5 x 0.75), the network's outputs at zero, for 2 s. The layer alone lets the hanging part fall away (no tension:
    740 mm); with the restoration the towel stays a towel and does not slide off: the table part (but the column on
    the edge) stays within 3 mm, its mean within 3 mm, the hanging part within 3 cm of where it hung. The restoration
    runs after the contact with vertices that static friction holds fixed, so it never drags a resting vertex (before
    the contact, friction could not keep up with the lip's pull and the towel slid). Known gap: the layer has no
    table side face, so the edge column and the edges round the corner do not hold exactly with nothing else
    holding the flap (a trained network's tension does)."""
    b, X, on, col = drape()
    xa, _ = run(b, X, 0.2, 0.5, 60, restore=None)
    assert (xa - X).norm(dim=-1).max() > 0.5                            # without: the flap falls away
    xe, v = run(b, X, 0.2, 0.5, 60)
    d = (xe - X).norm(dim=-1)
    inner = on & (col < 14)
    assert d[inner].max() < 3e-3
    assert abs((xe[:, 0] - X[:, 0]).mean()) < 3e-3                       # not sliding off (nor pulled on)
    assert d[~on].max() < 0.03
    assert v[inner].norm(dim=-1).max() < 0.01


def _hanging_rope(EA, n=31, l=0.01, lam=0.01):
    """A rope hanging from its top node at its closed-form equilibrium: segment k (counted from the bottom, carrying the
    nodes below it) at strain lam g s_k / EA, s_k the length share below it; the elastic forces that balance gravity
    exactly (f = m g up on every free node)."""
    area = np.full(n, l)
    area[[0, -1]] = l / 2
    m = lam * area
    below = np.cumsum(m[::-1])[::-1]                                     # weight below each node, itself included
    eps = below[1:] * G / EA                                             # segment k joins node k (top side) and k+1
    z = np.concatenate([[0.0], -np.cumsum(l * (1 + eps))]) + 1.0
    x = torch.zeros(n, 3, dtype=F64)
    x[:, 2] = torch.as_tensor(z)
    rest = torch.zeros(n, 3, dtype=F64)
    rest[:, 2] = -torch.arange(n, dtype=F64) * l
    e = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
    b = body(rest, area, e, 0.002, material=1)
    f = torch.zeros(n, 3, dtype=F64)
    f[:, 2] = torch.as_tensor(m * G)
    held = torch.zeros(n, dtype=torch.bool)
    held[0] = True
    return b, x, f, held, lam, eps


def test_hanging_closed_form_kept():
    """At its closed-form extension (0.1-1.4% here: inside the rope band) a hanging rope stays exactly where it is under
    the restoration; a band tighter than the extension (0.01%) would wreck it -- the test can fail."""
    for EA in (100.0, 2.0):
        b, x, f, held, lam, eps = _hanging_rope(EA)
        assert eps.max() < RESTORE_BANDS["edge"][1][0]
        xe, v = run(b, x, lam, 0.5, 30, f=f, given=held, xg=x)
        xo, _ = run(b, x, lam, 0.5, 30, f=f, given=held, xg=x, restore=None)
        assert torch.equal(xe, xo)                                       # the restoration does nothing
        assert (xe - x).abs().max() < 1e-6                               # (2e-7 m: GRAVITY is float32)
        assert torch.allclose(strain(b, xe), torch.as_tensor(eps), atol=1e-4)
    tight = restore_config("rope=0.0001:0.0001")
    xe, _ = run(b, x, lam, 0.5, 30, f=f, given=held, xg=x, restore=tight)
    assert (xe - x).abs().max() > 1e-3                                   # wrecked: the extension is not allowed
    # a cloth strip held along its top row, hanging at a uniform closed-form strain w g s / Y (s the length below)
    n, ny, h, w, Y = 6, 30, 0.01, 0.2, 200.0
    v, t, e, area = _grid(n, ny, h)
    rest = torch.tensor(v, dtype=F64)
    j = torch.round(rest[:, 1] / h)                                      # row index, 0 at the bottom
    Lb = ny * h
    s = rest[:, 1]                                                       # length below each point
    z = 1.0 + (s - Lb) + w * G * (s ** 2 - Lb ** 2) / (2 * Y)            # integral of (1 + w g s / Y) ds from the top
    x = torch.stack([rest[:, 0], torch.zeros_like(s), z], 1)
    bc = body(torch.stack([rest[:, 0], rest[:, 1], torch.zeros_like(s)], 1), area, e, 0.003, tris=t)
    held = j == ny
    fz = torch.zeros(len(x), 3, dtype=F64)
    fz[:, 2] = torch.as_tensor(area) * w * G
    st0 = strain(bc, x)
    assert st0.max() < RESTORE_BANDS["edge"][0][0] and st0.max() > 1e-3   # 0.3%: real, inside the band
    xe, _ = run(bc, x, w, 0.5, 30, f=fz, given=held, xg=x)
    xo, _ = run(bc, x, w, 0.5, 30, f=fz, given=held, xg=x, restore=None)
    assert torch.equal(xe, xo)
    assert torch.allclose(xe[:, 2], x[:, 2], atol=1e-6)                  # (drag on a still strip: none)
    assert torch.allclose(strain(bc, xe), st0, atol=1e-4)


def _noisy_run(b, x, w, n, f0, sig, restore, seed=0):
    """n frames with the forces f0 (balancing gravity) plus a random acceleration per vertex per frame -- an imprecise
    network: sig m/s^2 is a position error of sig dt^2 per frame (3 m/s^2: 3.3 mm)."""
    g = torch.Generator().manual_seed(seed)
    m = torch.as_tensor(b["area"]) * w
    xp, xc = x.clone(), x.clone()
    none = torch.zeros(len(x), dtype=torch.bool)
    for _ in range(n):
        f = f0 + m[:, None] * sig * torch.randn(x.shape, generator=g, dtype=F64)
        x1 = cloth_step(b, xp, xc, f, torch.tensor([w], dtype=F64), none, xc, torch.tensor([0.5], dtype=F64), restore=restore)
        xp, xc = xc, x1
    return xc


def test_imprecise_forces_keep_their_shape():
    """The failure the restoration is for: forces right on average but wrong by a few mm per frame (the trained
    models' one-step error is 1-4 mm). 300 frames (10 s) of 3 mm noise on a towel and a rope resting on the table and
    a soft block resting on it (its inside held up by f = m g, the network's job): without the restoration edges and
    tets drift without bound (towels and ropes stretch, tets invert); with it, every edge and tet stays near its band.
    Near, not in: an edge whose two ends static friction holds is not restored (the restoration never drags a
    resting vertex), so under noise this large (3 mm on 1 cm rope segments) a rope edge on the table reaches ~5%; soft
    bodies are solved by averaged Jacobi (80 sweeps; colour by colour took 186 ms a step), which leaves tets at
    0.75-1.08 of their volume (band 0.8-1.05) and edges at +-24% (band 20%) under this noise."""
    sig = 3.0 / (DT ** 2) * 1e-3                                         # 3 mm per frame
    x, area, e, t = sheet(16, z=0.003)
    cases = [("towel", body(x, area, e, 0.003, tris=t), x, 0.2, torch.zeros_like(x))]
    x, area, e, r = rope()
    cases.append(("rope", body(x, area, e, r, material=1, segs=True), x, 0.01, torch.zeros_like(x)))
    x, vol, e, tets, r, surf = block(5)
    bs = body(x, vol, e, r, material=2, tets=tets, surf_v=surf)
    f0 = torch.zeros_like(x)
    lifted = x[:, 2] > r + 1e-9                                          # not on the table: held by internal forces
    f0[lifted, 2] = torch.as_tensor(vol, dtype=F64)[lifted] * 100.0 * G
    cases.append(("soft", bs, x, 100.0, f0))
    for name, b, x0, w, f0 in cases:
        mat = int(b["material"][0])
        st, co = RESTORE_BANDS["edge"][mat]
        xa = _noisy_run(b, x0, w, 300, f0, sig, None)
        xe = _noisy_run(b, x0, w, 300, f0, sig, RESTORE_BANDS)
        sa, se = strain(b, xa), strain(b, xe)
        assert sa.abs().max() > 3 * max(st, co), name                    # without: far outside the band
        print(name, f"without: strain {float(sa.min()):.3f}..{float(sa.max()):.3f}; with: {float(se.min()):.4f}..{float(se.max()):.4f}")
        tol = 0.05 if name == "soft" else 0.03                           # (soft: averaged Jacobi, 40 sweeps)
        assert se.max() < st + tol and se.min() > -co - tol, (name, float(se.min()), float(se.max()))
        assert xe[:, 2].min() > -1e-3                                    # nothing pushed into the table
        if name == "soft":
            va, ve = tet_ratio(b, xa), tet_ratio(b, xe)
            print(f"soft tets without: {float(va.min()):.3f}..{float(va.max()):.3f}; with {float(ve.min()):.3f}..{float(ve.max()):.3f}")
            assert (va < 0).any() or (va - 1).abs().max() > 0.5
            assert ve.min() > 1 - RESTORE_BANDS["vol"][2][1] - 0.1 and ve.max() < 1 + RESTORE_BANDS["vol"][2][0] + 0.15
