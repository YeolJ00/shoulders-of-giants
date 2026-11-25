"""PGND's robot cloth recordings -> cloth episodes like lsim.real.cloth's, for the teacher and the model.

PGND (Zhang et al., RSS 2025; CC-BY-4.0; data/real/raw_pgnd_cloth, git-ignored): two xArm grippers (GELLO teleoperation)
hold one cloth by two points of an edge for a whole recording (the grippers never open) and drag, lift, swing and lay
it on the table -- in one set over a box; four RealSense cameras at 30 fps; 11 raw recordings of 0.6-4.3 min, released
as 4 s windows (120 frames) every 2 s (407 readable). Per window: the cloth's point cloud at the window's first frame
(masked depth from the four cameras, fused and cleaned; ~3 mm spacing) and those points carried forward frame by frame
through a velocity field (each point moved by the mean velocity of its four nearest points in the next frame's cloud,
velocities from 2D point tracks lifted with depth) -- only the first frame is a measurement; carried 60 frames (to the
next window's first frame, the same instant) the points sit 7.6 mm (median; p90 24 mm) off the observed surface --
and each gripper's pose and opening every frame. Their world frame has z down, the table near z = 0.

No frame shows the cloth flat, so its rest shape comes from unrolling: a window's first cloud is laid flat by Isomap
(geodesic distances over a 10-nearest-neighbour graph, landmark MDS into the plane), its scale corrected on the part
lying flat on the table (geodesics over a noisy cloud run 7-9% long), turned so the held edge is on top and gripper 0 on
the left. One cloth and one grasp per recording, so a recording's rest rectangle is the median of its flattest windows'
rectangles; each window's cloud is embedded in that grid by its own layout and the grid fitted to the cloud (robust
distance, edge strain, smoothness, the table). The carried points lag fast motion, so the grid follows them only for
the first 10 frames (enough history to start a rollout); after that only the vertices under each fingertip move,
rigidly in the gripper's frame along the robot's recorded path to frame 120. A prediction is scored against what the
cameras saw: the observed clouds at frames 0, 60 and 120 (the next windows' first clouds), by the median distance from
the cloud's points to the predicted surface (one-sided: a fold's hidden layer is not penalised). PgndEpisode.gt_error
is NaN; use cloud_error.
"""
import json
from pathlib import Path

import numpy as np
import torch

from lsim.real.cloth import FLIP, H, R_HELD, embed, grid_world
from lsim.teacher.cloth_mesh import grid, mesh_edges

REPO = Path(__file__).resolve().parents[3]
PGND = REPO / "data/real/raw_pgnd_cloth/experiments/log/data"
OUT = REPO / "data/real/cloth/pgnd"
PGND_SOURCES = ("0114_cloth6_processed", "1223_cloth_gello_cali2_processed", "1223_cloth_gello_obj1_processed")
SHORT = {"0114_cloth6_processed": "cloth6", "1223_cloth_gello_cali2_processed": "cali2",
         "1223_cloth_gello_obj1_processed": "obj1"}
BOX = ("1223_cloth_gello_obj1_processed",)              # the cloth goes over a box the episodes do not model yet
TEST = {("0114_cloth6_processed", 1), ("1223_cloth_gello_cali2_processed", 1)}   # held-out recordings (whole)
EEF_T = np.array([0.0, 0.0, 0.01])                    # PGND's fingertip offset from the flange (cfg.model.eef_t; raw frame)
TOOL = np.diag([1.0, -1.0, -1.0])                     # our gripper's frame (fingers along -z) in the xArm tool frame
INFL = 1.08                                           # kNN geodesic over true distance where none can be calibrated
FLAT_Z = 0.025                                        # raw height under which a point lies on the table
REST_Z = 0.003                                        # the teacher's resting cloth: vertices one radius above the table


def pgnd_windows(root=PGND, sources=PGND_SOURCES):
    """Every readable window directory (one archive member was cut off by the stream: obj1 episode_0063)."""
    import zipfile
    out = []
    for src in sources:
        for s in sorted((Path(root) / src / "sub_episodes_v").glob("episode_*")):
            try:                                                        # a cut-off zip has no central directory
                with zipfile.ZipFile(s / "traj.npz") as z:
                    if "xyz.npy" not in z.namelist():
                        continue
            except (zipfile.BadZipFile, OSError):
                continue
            out.append(s)
    return out


def load_window(path):
    """One PGND window in our frame (z up; the table not yet moved to z = 0): the carried points (T, N, 3), each
    gripper's fingertip (T, 2, 3) and rotation (T, 2, 3, 3; our gripper's frame: fingers along -z), its opening
    (T, 2; 1 open, 0 closed), and which recording and frame it starts at."""
    path = Path(path)
    with np.load(path / "traj.npz", allow_pickle=False) as z:
        x = z["xyz"] @ FLIP.T
    T = len(x)
    tip = (np.loadtxt(path / "eef_traj.txt").reshape(T, -1, 3) + EEF_T) @ FLIP.T
    rot = FLIP @ np.loadtxt(path / "eef_rot.txt").reshape(T, -1, 3, 3) @ TOOL
    grip = np.loadtxt(path / "eef_gripper.txt").reshape(T, -1)
    meta = np.loadtxt(path / "meta.txt").astype(int)
    return {"x": x, "tip": tip, "rot": rot, "grip": grip, "source": path.parts[-3], "episode": int(meta[0]),
            "pivot": int(meta[1]), "window": path.name}


def gripper_points(tip, rot):
    """The teacher's gripper (lsim.train.windows.gripper_mesh: origin 5.5 cm above its fingertips) posed at each
    PGND gripper: (T, 2 * M, 3). Built when an episode is loaded, not by the converter: its modules import warp."""
    from lsim.data.cloth import TIP
    from lsim.train.windows import gripper_mesh
    m = gripper_mesh().astype(np.float64)
    org = tip + rot[..., 2] * TIP                                       # fingertips + TIP along the gripper's up
    pts = np.einsum("tgij,mj->tgmi", rot, m) + org[:, :, None]
    return pts.reshape(len(tip), -1, 3)


def _fps(x, n, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(x)))]
    d = np.linalg.norm(x - x[idx[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(x - x[i], axis=1))
    return np.array(idx)


def unroll(x, k=10, n_land=1200):
    """Landmark Isomap of a cloud (N, 3) into the plane: (N, 2) coordinates (NaN off the largest connected piece of
    the k-nearest-neighbour graph), that piece's mask and the top four MDS eigenvalue shares (a third that is not
    small: the cloud does not lie flat without tearing or folding)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import cKDTree
    dist, nb = cKDTree(x).query(x, k=k + 1)
    G = csr_matrix((dist[:, 1:].ravel(), (np.repeat(np.arange(len(x)), k), nb[:, 1:].ravel())), shape=(len(x), len(x)))
    G = G.maximum(G.T)
    _, lab = connected_components(G, directed=False)
    main = lab == np.bincount(lab).argmax()
    land = np.flatnonzero(main)[_fps(x[main], min(n_land, int(main.sum())))]
    D = dijkstra(G, indices=land)                                        # (L, N); inf off the main piece
    Dl = D[:, land]
    n = len(land)
    J = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(-0.5 * J @ (Dl ** 2) @ J)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    Y = np.full((len(x), 2), np.nan)
    Dm = D[:, main]
    Y[main] = -0.5 * ((Dm ** 2) - (Dl ** 2).mean(0)[:, None]).T @ (vecs[:, :2] / np.sqrt(vals[:2]))
    return Y, main, vals[:4] / vals[:4].sum()


def _min_rect(y, q=0.5):
    best = None
    for th in np.radians(np.arange(0.0, 90.0, 0.5)):
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        a = y @ R.T
        lo, hi = np.percentile(a, q, 0), np.percentile(a, 100 - q, 0)
        if best is None or np.prod(hi - lo) < best[0]:
            best = (np.prod(hi - lo), lo, hi, R)
    return best[1:]


def layout(x, tip, calibrate=True):
    """A first-frame cloud (N, 3, raw heights) laid flat: in-plane coordinates (N, 2; NaN where unrolled) in its
    smallest rectangle, the held edge on top (y = H) and gripper 0 on the left, with the rectangle's size and the
    checks: the geodesic scale correction (calibrated on the part lying flat, else INFL), the third MDS eigenvalue's
    share, whether both grippers sit on one edge, where they are in the layout and how far each fingertip is from
    its nearest cloth point."""
    from scipy.spatial import cKDTree
    Y, main, ev = unroll(x)
    flat = main & (x[:, 2] < FLAT_Z)
    infl = None
    if calibrate and flat.mean() > 0.3:
        fi = np.flatnonzero(flat)
        rng = np.random.default_rng(0)
        a, b = rng.choice(fi, 4000), rng.choice(fi, 4000)
        de = np.linalg.norm(x[a, :2] - x[b, :2], axis=1)
        dl = np.linalg.norm(Y[a] - Y[b], axis=1)
        sel = de > 0.1
        if sel.sum() > 200:
            infl = float(np.median(dl[sel] / de[sel]))
    Y = Y / (infl or INFL)
    lo, hi, R = _min_rect(Y[main])
    A = np.full_like(Y, np.nan)
    A[main] = Y[main] @ R.T - lo
    W, Hh = hi - lo
    near = cKDTree(x[main]).query(tip)
    gi = np.flatnonzero(main)[near[1]]
    g = A[gi]
    for _ in range(4):                                                  # turn the held edge to the top
        if np.all(g[:, 1] > 0.6 * Hh):
            break
        A = np.c_[A[:, 1], W - A[:, 0]]
        g = np.c_[g[:, 1], W - g[:, 0]]
        W, Hh = Hh, W
    if g[0, 0] > g[1, 0]:                                               # gripper 0 on the left (a cloth turned over)
        A[:, 0] = W - A[:, 0]
        g[:, 0] = W - g[:, 0]
    info = {"W": float(W), "H": float(Hh), "infl": infl, "eig3": float(ev[2]), "flat": float(flat.mean()),
            "flat_z": float(np.median(x[flat, 2])) if flat.mean() > 0.3 else None,
            "held_top": bool(np.all(g[:, 1] > 0.6 * Hh)), "grippers_uv": g.tolist(), "tip_gap": near[0].tolist(),
            "main": float(main.mean())}
    return A, info


def recording_rest(infos):
    """A recording's rest rectangle (W, H) and table height from its windows' layouts: medians over the flattest
    windows (at least half the cloth on the table, a flat layout, both grippers on one edge), over every window
    with the grippers on one edge if fewer than three are that flat. Table: the median height of the points lying
    flat, minus the teacher's resting height (REST_Z)."""
    good = [i for i in infos if i["flat"] >= 0.5 and i["eig3"] < 0.03 and i["held_top"]]
    if len(good) < 3:
        good = [i for i in infos if i["held_top"]] or infos
    W = float(np.median([i["W"] for i in good]))
    Hh = float(np.median([i["H"] for i in good]))
    zs = [i["flat_z"] for i in infos if i.get("flat_z") is not None and i["flat"] >= 0.3]
    return {"W": W, "H": Hh, "n_windows": len(good), "table_z": (float(np.median(zs)) - REST_Z) if zs else None}


def rest_grid_of(W, Hh, h=H):
    nx, ny = max(int(round(W / h)), 2), max(int(round(Hh / h)), 2)
    hh = float(np.mean([W / nx, Hh / ny]))
    return {"nx": nx, "ny": ny, "h": hh, "origin": np.zeros(3), "u": np.array([1.0, 0, 0]), "v": np.array([0, 1.0, 0])}


def _laplacian(nx, ny):
    from scipy.sparse import coo_matrix
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    a = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()]
    b = np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
    n = idx.size
    A = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return A, np.asarray(A.sum(1)).ravel()


def init_positions(p, uv, g, radius=0.015):
    """First guess of the grid's vertices (V, 3) from the cloud p (N, 3) at in-plane coordinates uv: each vertex at
    the mean of the points within `radius` of it in the plane, the rest filled harmonically (a membrane over the
    holes the cameras did not see)."""
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree
    verts, _ = grid(g["nx"], g["ny"], g["h"])
    V = len(verts)
    tree = cKDTree(uv)
    X = np.zeros((V, 3))
    known = np.zeros(V, bool)
    for k, nb in enumerate(tree.query_ball_point(verts[:, :2], radius)):
        if nb:
            X[k] = p[nb].mean(0)
            known[k] = True
    A, deg = _laplacian(g["nx"], g["ny"])
    u = np.flatnonzero(~known)
    if len(u) and known.any():
        from scipy.sparse import diags
        L = diags(deg) - A
        Luu = L[u][:, u]
        rhs = -(L[u][:, np.flatnonzero(known)] @ X[known])
        X[u] = np.stack([spsolve(Luu.tocsc(), rhs[:, j]) for j in range(3)], 1)
    return X, known


class _Terms:
    """The fit's terms on one frame, each ~1 at its tolerance: robust (Cauchy) distance of the embedded points to
    the cloud at `delta`; edge strain at `strain_scale`; the umbrella Laplacian (bending) at `bend_scale`; vertices
    below the table at 1 mm; the acceleration against the two frames before at `acc_scale` (tracking only)."""

    def __init__(self, tris, rest_xyz, nx, ny, tri, bary, w_data, dev, delta=0.005, strain_scale=0.005,
                 bend_scale=0.004, acc_scale=0.01, radius=0.002, w_stretch=1.0, w_bend=0.02, w_time=0.05):
        self.e = torch.as_tensor(mesh_edges(tris)[0].astype(np.int64), device=dev)
        rx = torch.as_tensor(rest_xyz, dtype=torch.float64, device=dev)
        self.L0 = (rx[self.e[:, 0]] - rx[self.e[:, 1]]).norm(dim=-1)
        idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        c = idx[1:-1, 1:-1].ravel()
        self.lap = [torch.as_tensor(a, device=dev) for a in (c, idx[1:-1, :-2].ravel(), idx[1:-1, 2:].ravel(),
                                                               idx[:-2, 1:-1].ravel(), idx[2:, 1:-1].ravel())]
        self.ti = torch.as_tensor(tri, device=dev)
        self.bw = torch.as_tensor(bary, dtype=torch.float64, device=dev)
        self.wd = torch.as_tensor(w_data, dtype=torch.float64, device=dev)
        self.nw = self.wd.sum().clamp_min(1.0)
        self.k = dict(delta=delta, strain=strain_scale, bend=bend_scale, acc=acc_scale, radius=radius,
                      w_stretch=w_stretch, w_bend=w_bend, w_time=w_time)
        self.dev = dev

    def __call__(self, x, P, p1=None, p2=None):
        k = self.k
        r = ((x[self.ti] * self.bw[:, :, None]).sum(1) - P).norm(dim=-1) / k["delta"]
        data = (self.wd * torch.log1p(r ** 2)).sum() / self.nw
        L = (x[self.e[:, 0]] - x[self.e[:, 1]]).norm(dim=-1)
        stretch = (((L - self.L0) / self.L0 / k["strain"]) ** 2).mean()
        c, l, rr, d, u = self.lap
        lap = x[c] - 0.25 * (x[l] + x[rr] + x[d] + x[u])
        bend = ((lap / k["bend"]) ** 2).sum(-1).mean()
        table = ((torch.relu(k["radius"] - x[:, 2]) / 1e-3) ** 2).mean()
        loss = data + k["w_stretch"] * stretch + k["w_bend"] * bend + table
        time = torch.zeros((), dtype=x.dtype, device=x.device)
        if p1 is not None:
            time = (((x - 2 * p1 + p2) / k["acc"]) ** 2).sum(-1).mean()
            loss = loss + k["w_time"] * time
        return loss, {"data": data.item(), "stretch": stretch.item(), "bend": bend.item(), "table": table.item(),
                      "time": time.item()}


def _solve(terms, x0, P, held_mask=None, held_pos=None, p1=None, p2=None, iters=60):
    dev = terms.dev
    x = torch.nn.Parameter(torch.as_tensor(x0, dtype=torch.float64, device=dev).clone())
    Pt = torch.as_tensor(P, dtype=torch.float64, device=dev)
    hm = None if held_mask is None else torch.as_tensor(held_mask, device=dev)[:, None]
    hp = None if held_pos is None else torch.as_tensor(held_pos, dtype=torch.float64, device=dev)
    t1 = None if p1 is None else torch.as_tensor(p1, dtype=torch.float64, device=dev)
    t2 = None if p2 is None else torch.as_tensor(p2, dtype=torch.float64, device=dev)
    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14)

    def full():
        return x if hm is None else torch.where(hm, hp, x)

    def closure():
        opt.zero_grad()
        loss, _ = terms(full(), Pt, t1, t2)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        xf = full()
        _, parts = terms(xf, Pt, t1, t2)
    return xf.detach().cpu().numpy(), parts


def _closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p (all (N, 3)): barycentric weights (N, 3) (Ericson's regions)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-18, 1e-18, va + vb + vc)
    v, w = vb / den, vc / den
    out = np.stack([1 - v - w, v, w], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.clip(d1 / np.where(d1 - d3 == 0, 1, d1 - d3), 0, 1)
        out = np.where(r_ab[:, None], np.stack([1 - t, t, 0 * t], 1), out)
        r_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = np.clip(d2 / np.where(d2 - d6 == 0, 1, d2 - d6), 0, 1)
        out = np.where(r_ac[:, None], np.stack([1 - t, 0 * t, t], 1), out)
        r_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1, (d4 - d3) + (d5 - d6)), 0, 1)
        out = np.where(r_bc[:, None], np.stack([0 * t, 1 - t, t], 1), out)
    out = np.where(((d1 <= 0) & (d2 <= 0))[:, None], np.array([1.0, 0, 0]), out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[:, None], np.array([0, 1.0, 0]), out)
    out = np.where(((d6 >= 0) & (d5 <= d6))[:, None], np.array([0, 0, 1.0]), out)
    return out


def project_to_mesh(points, X, tris, k=8):
    """Each point's closest point on the mesh X (V, 3): its triangle's vertices (N, 3), barycentric weights (N, 3)
    and distance (N,) -- among the k triangles whose centroids are nearest."""
    from scipy.spatial import cKDTree
    cen = X[tris].mean(1)
    _, cand = cKDTree(cen).query(points, k=k)
    best_d = np.full(len(points), np.inf)
    best_t = np.zeros(len(points), np.int64)
    best_w = np.zeros((len(points), 3))
    for j in range(k):
        t = tris[cand[:, j]]
        w = _closest_on_triangles(points, X[t[:, 0]], X[t[:, 1]], X[t[:, 2]])
        q = (X[t] * w[:, :, None]).sum(1)
        d = np.linalg.norm(points - q, axis=1)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_t = np.where(better, cand[:, j], best_t)
        best_w = np.where(better[:, None], w, best_w)
    return tris[best_t], best_w, best_d


def mesh_distance(points, X, tris, per_tri=10):
    """Distance (N,) of each point to the mesh surface X (V, 3) (triangles sampled at ~per_tri points: ~3 mm apart on a
    1 cm grid, so distances under ~1.5 mm read as up to that)."""
    from scipy.spatial import cKDTree
    s = int(np.ceil(np.sqrt(2 * per_tri)))
    a, b = np.meshgrid(np.linspace(0, 1, s), np.linspace(0, 1, s))
    m = a + b <= 1.0
    w = np.stack([1 - a[m] - b[m], a[m], b[m]], 1)                      # (S, 3)
    S = np.einsum("sk,tkd->tsd", w, X[tris]).reshape(-1, 3)
    return cKDTree(S).query(points)[0]


def recordings(windows):
    """Windows grouped into chains: one per recording, split where a window is missing (consecutive windows start 60
    frames apart). Returns {(source, episode, first pivot): [window paths in order]}."""
    by = {}
    for p in windows:
        m = np.loadtxt(Path(p) / "meta.txt").astype(int)
        by.setdefault((Path(p).parts[-3], int(m[0])), []).append((int(m[1]), Path(p)))
    chains = {}
    for (src, ep), ws in by.items():
        ws.sort()
        cur = [ws[0]]
        for pv, p in ws[1:]:
            if pv == cur[-1][0] + 60:
                cur.append((pv, p))
            else:
                chains[(src, ep, cur[0][0])] = [q for _, q in cur]
                cur = [(pv, p)]
        chains[(src, ep, cur[0][0])] = [q for _, q in cur]
    return chains


def _track(terms, x2, x1, P, held, HP, iters):
    """Frames after x1 one by one, each started from x1 moved at its last velocity."""
    out = []
    for t in range(len(P)):
        x, _ = _solve(terms, 2 * x1 - x2, P[t], held, HP[t], x1, x2, iters=iters)
        out.append(x)
        x2, x1 = x1, x
    return np.stack(out)


def first_cloud(path):
    """A window's first cloud (an observation) in our frame, before the table shift."""
    with np.load(Path(path) / "traj.npz", allow_pickle=False) as z:
        return z["xyz"][0] @ FLIP.T


def next_windows(path, chains):
    """The windows starting 60 and 120 frames after this one in the same recording (None where there is none)."""
    path = Path(path)
    for ws in chains.values():
        if path in ws:
            i = ws.index(path)
            return (ws[i + 1] if i + 1 < len(ws) else None), (ws[i + 2] if i + 2 < len(ws) else None)
    return None, None


def convert_window(path, rest, table_z, nxt=None, nxt2=None, seg=60, n_rec=10, iters0=400, iters=40, n_obs=1500,
                   delta_obs=0.005, delta_carried=0.01, dev="cpu", log=print):
    """One window -> an episode scored on observations. The grid is fitted to the window's first cloud (unrolled:
    layout) and followed through the carried points for its first `n_rec` frames only (the history a rollout is given:
    over longer spans the carried points lag fast motion -- the gripper hides the cloth it holds, so the velocity field
    there comes from slower cloth around it -- and per-frame fitting loses swinging cloth). The held patches (the
    vertices under each fingertip at frame 0, within R_HELD at rest) move rigidly with their gripper along the robot's
    path, which runs to frame 2 seg (the next window carries it), so a rollout can be driven 4 s and scored against
    the observed clouds at frames 0, seg (`nxt`'s first cloud: the same instant) and 2 seg (`nxt2`'s). After frame
    n_rec - 1 the episode holds its last fit with the held patches moved: only driven vertices may be read there.
    rest: the recording's {"W", "H"}; table_z: its table height in the raw frame (everything moves down by it)."""
    w = load_window(path)
    w1 = load_window(nxt) if nxt is not None else None
    shift = np.array([0.0, 0.0, table_z])
    Tr = 2 * seg + 1

    def path_of(key):
        a = np.concatenate([w[key], (w1[key][seg:seg + 1] if w1 is not None else w[key][-1:])], 0)
        return a[:Tr]

    tip, rot, grip = path_of("tip") - shift, path_of("rot"), path_of("grip")
    g = rest_grid_of(rest["W"], rest["H"])
    rest_xyz, tris = grid_world(g)
    V = len(rest_xyz)
    P = w["x"][:n_rec] - shift                                          # frame 0 observed, then carried
    uv, info = layout(w["x"][0], w["tip"][0], calibrate=w["source"] not in BOX)
    ok = np.isfinite(uv[:, 0])
    uvr = np.where(ok[:, None], uv * np.array([rest["W"] / info["W"], rest["H"] / info["H"]]), 0.0)
    inside = ok & (uvr[:, 0] > -0.01) & (uvr[:, 0] < rest["W"] + 0.01) & (uvr[:, 1] > -0.01) & (uvr[:, 1] < rest["H"] + 0.01)
    tri, bary = embed(np.clip(uvr, 0.0, None), g)
    Xi, known = init_positions(P[0, inside], uvr[inside], g)
    X0, fit0 = _solve(_Terms(tris, rest_xyz, g["nx"], g["ny"], tri, bary, inside.astype(float), dev, delta=delta_obs),
                      Xi, P[0], iters=iters0)
    held = np.zeros(V, bool)
    held_info, HP = [], np.zeros((Tr, V, 3))
    for k in range(tip.shape[1]):
        v = int(np.argmin(np.linalg.norm(X0 - tip[0, k], axis=1)))
        near = np.flatnonzero(np.linalg.norm(rest_xyz - rest_xyz[v], axis=1) <= R_HELD)
        near = near[~held[near]]
        held[near] = True
        off = (X0[near] - tip[0, k]) @ rot[0, k]                         # in the gripper's frame
        HP[:, near] = np.einsum("tij,nj->tni", rot[:, k], off) + tip[:, k, None]
        held_info.append({"vertex": v, "vertices": int(len(near)), "uv": rest_xyz[v, :2].tolist(),
                          "tip_to_vertex_m": float(np.linalg.norm(X0[v] - tip[0, k]))})
    X0[held] = HP[0, held]
    ti, bw, d0 = project_to_mesh(P[0], X0, tris)                        # the carried points' correspondences
    use = d0 < 0.03
    terms = _Terms(tris, rest_xyz, g["nx"], g["ny"], ti, bw, use.astype(float), dev, delta=delta_carried)
    Xs = _track(terms, X0, X0, P[1:], held, HP[1:n_rec], iters)
    X = np.zeros((Tr, V, 3))
    X[0], X[1:n_rec] = X0, Xs
    X[n_rec:] = X[n_rec - 1]
    X[n_rec:, held] = HP[n_rec:, held]
    pe = (X[:n_rec, ti] * bw[None, :, :, None]).sum(2)
    res = np.linalg.norm(pe - P, axis=-1)[:, use]
    e = mesh_edges(tris)[0]
    L0 = np.linalg.norm(rest_xyz[e[:, 0]] - rest_xyz[e[:, 1]], axis=-1)
    strain = np.linalg.norm(X[:n_rec, e[:, 0]] - X[:n_rec, e[:, 1]], axis=-1) / L0 - 1.0
    q = {"cloud0_to_mesh_mm": float(1e3 * np.median(d0)), "cloud0_to_mesh_p90_mm": float(1e3 * np.percentile(d0, 90)),
         "known0": float(known.mean()), "fit0": fit0, "res_mm": (1e3 * np.median(res, axis=1)).round(2).tolist(),
         "strain_abs_median": float(np.median(np.abs(strain))), "strain_p99": float(np.percentile(np.abs(strain), 99))}
    clouds = {0: P[0], seg: (w1["x"][0] - shift) if w1 is not None else None,
              2 * seg: (first_cloud(nxt2) - shift) if nxt2 is not None else None}
    for f in (seg, 2 * seg):                                            # PGND's own carried points there (to seg only)
        if clouds[f] is not None and f < len(w["x"]):
            from scipy.spatial import cKDTree
            q[f"carried_to_cloud{f}_mm"] = float(1e3 * np.median(cKDTree(w["x"][f] - shift).query(clouds[f])[0]))
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(np.flatnonzero(use), min(n_obs, int(use.sum())), replace=False))
    out = {"x": X.astype(np.float32), "held": np.broadcast_to(held, X.shape[:2]).copy(),
           "tip": tip.astype(np.float32), "rot": rot.astype(np.float32),
           "grip": grip.astype(np.float32), "obs": P[:, sub].astype(np.float32), "obs_tri": ti[sub], "obs_bary": bw[sub],
           "carried": (w["x"][:, sub] - shift).astype(np.float32)}
    for f, c in clouds.items():
        if c is not None:
            out[f"cloud{f}"] = c.astype(np.float32)
    case = f"{SHORT[w['source']]}_e{w['episode']}_p{w['pivot']:05d}"
    meta = {"source": "pgnd", "case": case, "pgnd_source": w["source"], "pgnd_window": w["window"], "episode": w["episode"],
            "pivot": w["pivot"], "n_frames": int(Tr), "n_rec": int(n_rec), "fps": 30.0, "rest_frame": 0,
            "clouds": [f for f, c in clouds.items() if c is not None],
            "grid": {"nx": g["nx"], "ny": g["ny"], "h": g["h"]}, "rest": rest, "table_z_raw": table_z, "layout": info,
            "held": held_info, "quality": q, "box": w["source"] in BOX,
            "split": "test" if (w["source"], w["episode"]) in TEST else "train", "grip_max": float(grip.max())}
    return out, meta


def save_episode(out, meta, root=OUT):
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / f"{meta['case']}.npz", **out)
    (root / f"{meta['case']}.json").write_text(json.dumps(meta, indent=1))


def load_pgnd(split=None, box=False, root=OUT):
    """The converted windows (PgndEpisode), optionally one split ("train" / "test"); the box set is left out unless
    asked for (its box is not modelled yet)."""
    eps = []
    for j in sorted(Path(root).glob("*.json")):
        m = json.loads(j.read_text())
        if (split is None or m["split"] == split) and (box or not m["box"]):
            eps.append(PgndEpisode(j.with_suffix("")))
    return eps


from lsim.real.cloth import RealClothEpisode  # noqa: E402


class PgndEpisode(RealClothEpisode):
    """A converted PGND window with lsim.train.cloth_windows.ClothEpisode's interface (as RealClothEpisode): the
    fitted grid for its first n_rec frames, then its last fit with the held patches moved along the robot's path
    (only rollouts' driven vertices may be read there), both grippers as the kinematic tokens (the teacher's gripper
    at each fingertip), the table as the only surface, the material unknown, the grasp force not measured. Scored
    against the observed clouds at frames self.cloud_frames (cloud_error)."""

    def __init__(self, path):
        from lsim.train.cloth_windows import C_CENTRE, _grid
        path = Path(path)
        f = np.load(path.with_suffix(".npz"))
        self.meta = json.loads(path.with_suffix(".json").read_text())
        self.name = f"pgnd/{self.meta['case']}"
        g = self.meta["grid"]
        self._x = f["x"].astype(np.float32)
        self.held = f["held"].astype(bool)
        self.hand = gripper_points(f["tip"].astype(np.float64), f["rot"].astype(np.float64)).astype(np.float32)
        self.n_frames = len(self._x)
        self.n_rec = int(self.meta["n_rec"])
        self.rest, self.tris, self.edges, self.area = _grid(int(g["nx"]), int(g["ny"]), float(g["h"]))
        w, Y, c, mu = np.exp(C_CENTRE)
        self.c = {"nx": g["nx"], "ny": g["ny"], "h": g["h"], "r": 0.003, "w": float(w), "Y": float(Y),
                  "bending_length": float(c), "mu": float(mu), "unknown": True}
        self.wd = {"table_mu": 0.5, "ramp": None, "obstacles": [], "init": "real", "families": [], "events": []}
        self.grasp_F = np.zeros((self.n_frames, 3), np.float32)
        self.contact_F = np.zeros((self.n_frames, 3, 3), np.float32)
        self.gripper_rest = (self.hand[0] - self.hand[0].mean(0)).astype(np.float64)
        self.obs, self.obs_vis = f["obs"], np.ones(f["obs"].shape[:2], bool)
        self.obs_tri, self.obs_bary = f["obs_tri"], f["obs_bary"]
        self.carried = f["carried"]
        self.clouds = {int(k[5:]): f[k] for k in f.files if k.startswith("cloud")}
        self.cloud_frames = sorted(self.clouds)
        self.frame_ok = np.arange(self.n_frames) < self.n_rec

    def cloud_error(self, X, frame, stat="median"):
        """Distance (m) from the observed cloud at `frame` to the surface X (V, 3): what the cameras saw against a
        prediction (one-sided: the cameras see one side of a fold, a mesh's hidden layers are not penalised)."""
        d = project_to_mesh(self.clouds[int(frame)].astype(np.float64), np.asarray(X, np.float64), self.tris)[2]
        return float(np.median(d) if stat == "median" else np.mean(d))

    def gt_error(self, X, frames, moved_from=None, min_move=0.01):
        return np.full(len(np.atleast_1d(frames)), np.nan)


TEACHER = REPO / "data/datasets/c1_pgnd"
WARM = 20                                  # teacher frames dropped after its start (frame 7): it rings 15-20 frames


def load_pgnd_teacher(root=TEACHER, warm=WARM):
    """The teacher's episodes on PGND's training windows (scripts/l1_generate_cloth_pgnd.py), as cloth episodes with
    every vertex known: per window and material, frames 7 + warm .. 120 of the teacher (started from the window's
    fitted frame 7 it rings for 15-20 frames, 25-32 mm per frame at p99 against ~1 mm in the real tracks: its answer
    to a fitted state that is not its own equilibrium, whatever the material), the held patches and the grippers
    sliced to match, the material given as in c1."""
    import copy
    out = []
    for j in sorted(Path(root).glob("*.json")):
        meta = json.loads(j.read_text())
        f = np.load(j.with_suffix(".npz"))
        base = PgndEpisode(OUT / meta["case"])
        s = int(meta["t0"]) + warm
        for x, m in zip(f["x"], meta["materials"]):
            e = copy.copy(base)
            e._x = x[s:]
            e.held, e.hand = base.held[s:], base.hand[s:]
            e.grasp_F, e.contact_F = base.grasp_F[s:], base.contact_F[s:]
            e.n_frames = e.n_rec = len(e._x)
            e.frame_ok = np.ones(e.n_frames, bool)
            e.c = dict(base.c, w=m["w"], Y=m["Y"], bending_length=m["c"], mu=m["mu"], unknown=False)
            e.name = f"c1_pgnd/{meta['case']}"
            out.append(e)
    return out
