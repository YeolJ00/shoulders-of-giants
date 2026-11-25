"""Tetrahedral meshes for the soft-body teacher (lsim.teacher.soft_teacher): household soft objects as linear tets.

Linear tets lock in bending (DIRECTION.md "Soft-body teacher gate": a cantilever 0.66 / 0.88 / 0.96 of Timoshenko at
2 / 4 / 8 cells through its thickness), so every shape here has at least N_MIN = 8 cells through its thinnest
section, and the vertex count is held under ~1.5k by the shapes' aspect ratios (soft_teacher.sample_shape).

  box        the gate's structured grid (`box_tets`, moved here from scripts/l0_soft_gate.py): 5 tets per cubic cell,
             alternating, conforming; the thinnest side N_MIN cells
  cylinder   a disk of concentric rings (6k points on ring k: near-equilateral triangles, scipy Delaunay) extruded in
             layers; each prism split into 3 tets by the sorted-index rule (every quad face's diagonal from its
             lowest-index bottom vertex), conforming between neighbours
  ellipsoid  an O-grid ball (`ball`: a rounded core cube grid and prism layers out to the sphere), scaled per axis
  blob       the ellipsoid with smooth radial lumps (von Mises bumps on the direction: a head, ears, limbs' stubs),
             a plush-toy-like rounded body; star-shaped by construction

Every mesh is centred on its bounding box, tets positively oriented; `surface` gives the outward boundary triangles.
"""
import numpy as np

N_MIN = 8                                           # cells through the thinnest section


def box_tets(nx, ny, nz, h):
    """A box of nx x ny x nz cubic cells of side h, 5 tets per cell with alternating orientation (conforming)."""
    v = np.array([(x * h, y * h, z * h) for z in range(nz + 1) for y in range(ny + 1) for x in range(nx + 1)], float)

    def gi(x, y, z):
        return (nx + 1) * (ny + 1) * z + (nx + 1) * y + x
    t = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                v0, v1, v2, v3 = gi(x, y, z), gi(x + 1, y, z), gi(x + 1, y, z + 1), gi(x, y, z + 1)
                v4, v5, v6, v7 = gi(x, y + 1, z), gi(x + 1, y + 1, z), gi(x + 1, y + 1, z + 1), gi(x, y + 1, z + 1)
                if (x & 1) ^ (y & 1) ^ (z & 1):
                    t += [(v0, v1, v4, v3), (v2, v3, v6, v1), (v5, v4, v1, v6), (v7, v6, v3, v4), (v4, v1, v6, v3)]
                else:
                    t += [(v1, v2, v5, v0), (v3, v0, v7, v2), (v4, v7, v0, v5), (v6, v5, v2, v7), (v5, v2, v7, v0)]
    return v, orient(v, np.array(t, int))


def orient(v, t):
    d = signed_volume(v, t)
    t = t.copy()
    t[d < 0] = t[d < 0][:, [0, 2, 1, 3]]
    return t


def signed_volume(v, t):
    a = v[t[:, 1]] - v[t[:, 0]]
    b = v[t[:, 2]] - v[t[:, 0]]
    c = v[t[:, 3]] - v[t[:, 0]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0


def tet_J(x, tets, Dm_inv):
    """Volume ratio of each tet (x against the rest shape whose inverse edge matrices are Dm_inv)."""
    Ds = np.stack([x[tets[:, 1]] - x[tets[:, 0]], x[tets[:, 2]] - x[tets[:, 0]], x[tets[:, 3]] - x[tets[:, 0]]], -1)
    return np.linalg.det(Ds @ Dm_inv)


def _centre(v):
    return v - 0.5 * (v.min(0) + v.max(0))


def box(dims, n_min=N_MIN):
    """dims (m): the thinnest side gets n_min cells; cells cubic."""
    h = min(dims) / n_min
    n = [max(1, int(round(d / h))) for d in dims]
    v, t = box_tets(*n, h)
    v = v * (np.asarray(dims) / (np.asarray(n) * h))           # exact dims (cells within a few % of cubic)
    return _centre(v), t


def _disk(R, h):
    """Concentric rings of near-equilateral triangles: ring k at radius k R / nr with 6k points."""
    nr = max(2, int(round(R / h)))
    pts = [np.zeros((1, 2))]
    for k in range(1, nr + 1):
        m = 6 * k
        a = 2 * np.pi * (np.arange(m) + 0.5 * (k % 2)) / m
        pts.append(R * k / nr * np.stack([np.cos(a), np.sin(a)], 1))
    p = np.concatenate(pts)
    from scipy.spatial import Delaunay
    tri = Delaunay(p).simplices
    a = p[tri[:, 1]] - p[tri[:, 0]]
    b = p[tri[:, 2]] - p[tri[:, 0]]
    area = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    tri = tri[np.abs(area) > 1e-12 * R * R]
    return p, tri


def extrude(p2, tri2, H, nz):
    """A 2D triangulation extruded to height H in nz layers: prisms split into 3 conforming tets each."""
    m = len(p2)
    z = np.linspace(0.0, H, nz + 1)
    v = np.concatenate([np.c_[p2, np.full(m, zk)] for zk in z])
    tets = []
    for k in range(nz):
        for tr in tri2:
            i, j, l = sorted(int(a) for a in tr)
            b0, b1 = k * m, (k + 1) * m
            I, J, L = i + b0, j + b0, l + b0
            I2, J2, L2 = i + b1, j + b1, l + b1
            tets += [(I, J, L, L2), (I, J, J2, L2), (I, I2, J2, L2)]
    return v, orient(v, np.array(tets, int))


def cylinder(R, H, n_min=N_MIN):
    """Upright cylinder of radius R and height H: the thinner of 2R and H gets n_min cells."""
    h = min(2 * R, H) / n_min
    p2, tri2 = _disk(R, h)
    v, t = extrude(p2, tri2, H, max(1, int(round(H / h))))
    return _centre(v), t


def ball(nc=4, nr=3, w=0.7, a=0.55):
    """The unit ball: a core cube grid (nc cells a side, half-size a, rounded towards a sphere by w s^2 of the
    cubed-sphere map) and nr layers of prisms from the core's boundary out to the sphere along its rays (an O-grid:
    a single mapped cube grid cannot mesh a ball -- its 12 edges flatten to 180 deg and their tets to slivers, 4.6-175
    deg measured). Prisms are split by the sorted-index rule, conforming with each other and with the core. nc = 4,
    nr = 3: 10 cells across, 419 vertices, dihedral angles 16.8-137.7 deg, smallest altitude 0.063 of the radius."""
    v, t = box_tets(nc, nc, nc, 2.0 / nc)
    p = v - 1.0
    s = np.abs(p).max(1)
    r = np.linalg.norm(p, axis=1)
    ws = w * s ** 2
    q = np.where(r[:, None] > 1e-12, p * ((1 - ws) + ws * s / np.maximum(r, 1e-12))[:, None], 0.0) * a
    S = surface(t)
    B = np.unique(S)
    rank = -np.ones(len(q), int)
    rank[B] = np.arange(len(B))
    P = q[B]
    d = P / np.linalg.norm(P, axis=1, keepdims=True)
    V, layers, n = [q], [B], len(q)
    for k in range(1, nr + 1):
        u = k / nr
        V.append(P * (1 - u) + d * u)
        layers.append(n + np.arange(len(B)))
        n += len(B)
    T = [t]
    for k in range(nr):
        b0, b1 = layers[k], layers[k + 1]
        for tr in rank[S]:
            i, j, l = sorted(int(x) for x in tr)
            T.append(np.array([(b0[i], b0[j], b0[l], b1[l]), (b0[i], b0[j], b1[j], b1[l]), (b0[i], b1[i], b1[j], b1[l])]))
    V = np.concatenate(V)
    return V, orient(V, np.concatenate(T))


BALL = {"small": (4, 3), "fine": (6, 3), "xfine": (8, 3)}   # 10 / 12 / 14 cells across: 419 / 997 / 1,887 vertices


def ellipsoid(axes, fine=False):
    """Semi-axes (a, b, c) (m) of the O-grid ball: >= 10 cells across every diameter."""
    q, t = ball(*BALL["fine" if fine else "small"])
    return _centre(q * np.asarray(axes)), t


def blob(axes, lumps, fine=True):
    """fine: True ("fine"), False ("small") or a BALL key."""
    """An ellipsoid with radial lumps: lumps [(direction (3,), amplitude, sharpness kappa)], the radius along d
    scaled by 1 + sum A exp(kappa (d . u - 1)) -- smooth bumps (a head, ears, stubby limbs)."""
    q, t = ball(*BALL[fine if isinstance(fine, str) else ("fine" if fine else "small")])
    r = np.linalg.norm(q, axis=1)
    d = np.where(r[:, None] > 1e-12, q / np.maximum(r, 1e-12)[:, None], 0.0)
    f = np.ones(len(q))
    for u, A, kap in lumps:
        u = np.asarray(u, float) / np.linalg.norm(u)
        f += A * np.exp(kap * (d @ u - 1.0))
    v = q * f[:, None] * np.asarray(axes)
    return _centre(v), orient(v, t)


def surface(t):
    """Boundary triangles of a tet mesh, outward (counter-clockwise seen from outside)."""
    faces = np.concatenate([t[:, [1, 2, 3]], t[:, [0, 3, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 1]]])
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[cnt[inv.ravel()] == 1]


def quality(v, t):
    """Smallest altitude, smallest and largest dihedral angle (deg), smallest volume (all of the rest mesh)."""
    hmin = np.inf
    for o in range(4):
        a, b, c = [t[:, (o + k) % 4] for k in (1, 2, 3)]
        n = np.cross(v[b] - v[a], v[c] - v[a])
        hmin = min(hmin, float(np.min(np.abs(((v[t[:, o]] - v[a]) * n).sum(1)) / np.linalg.norm(n, axis=1))))
    fn = []                                          # outward face normals
    for o in range(4):
        a, b, c = [t[:, (o + k) % 4] for k in (1, 2, 3)]
        n = np.cross(v[b] - v[a], v[c] - v[a])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        side = np.sign(((v[t[:, o]] - v[a]) * n).sum(1))
        fn.append(-n * side[:, None])
    dih = []
    for i in range(4):
        for j in range(i + 1, 4):
            dih.append(180.0 - np.degrees(np.arccos(np.clip((fn[i] * fn[j]).sum(1), -1, 1))))
    dih = np.stack(dih, 1)
    return {"h_min": hmin, "dihedral_min": float(dih.min()), "dihedral_max": float(dih.max()),
            "vol_min": float(signed_volume(v, t).min())}
