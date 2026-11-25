"""Real cloth recordings -> cloth episodes like the teacher's, for the teacher and the model to be measured on.

Sources, each downloaded into its own git-ignored directory under data/real/:
  PhysTwin (Jiang et al., ICCV 2025; MIT; data/real/raw_phystwin): 8 cloth recordings, one or two hands lifting a
    33-49 cm cloth from a table, three RealSense cameras at 30 fps, 80-173 frames. Per frame ~5k CoTracker points on
    the cloth lifted to 3D with depth, with visibility (object_points), and 30 points on the hands nearest the cloth
    (controller_points); 9 hand-annotated 3D tracks (gt_track_3d, NaN where unseen); a train/test frame split
    (the test frames are the last 30%: future prediction).
  PGND (Zhang et al., RSS 2025; CC-BY-4.0; data/real/raw_pgnd_cloth): cloth handled by a robot gripper (to come).

Their world frame has z down and the table at z = 0; ours has z up: (x, y, z) -> (x, -y, -z).

A recording becomes a mesh like the teacher's cloth: a flat grid of ~1 cm cells laid over the cloth at rest (the
flattest of its first frames), each tracked point embedded in a triangle of it (barycentric), and every frame's
vertex positions fitted to the tracked points -- robust distance to the visible points, plus terms that keep the
mesh's edge lengths (the cloth barely stretches), its accelerations small (fills occlusions) and its vertices above
the table. The fit is a reconstruction, not a measurement: errors are reported against the tracked points
themselves where visible, and against the annotated tracks.
"""
import json
from pathlib import Path

import numpy as np
import torch

from lsim.real import safe_pickle
from lsim.teacher.cloth_mesh import grid, mesh_edges

REPO = Path(__file__).resolve().parents[3]
PHYSTWIN = REPO / "data/real/raw_phystwin/extracted"
OUT = REPO / "data/real/cloth"
PHYSTWIN_CLOTH = ("single_lift_cloth", "single_lift_cloth_1", "single_lift_cloth_3", "single_lift_cloth_4",
                  "double_lift_cloth_1", "double_lift_cloth_3", "single_clift_cloth_1", "single_clift_cloth_3")
FLIP = np.diag([1.0, -1.0, -1.0])                     # their z-down frame -> z up (a half turn about x)
H = 0.01                                              # target cell size (the teacher's towels: 1 cm)
R_HELD = 0.015                                        # the held patch: vertices this close to the grasp at rest
PHYSTWIN_GRASP = {"single_lift_cloth": "pinch", "single_lift_cloth_1": "pinch", "single_lift_cloth_3": "pinch",
                  "single_lift_cloth_4": "pinch", "single_clift_cloth_1": "corner", "single_clift_cloth_3": "corner",
                  "double_lift_cloth_1": "corners", "double_lift_cloth_3": "corners"}   # from each video's frames


def phystwin_case(case, root=PHYSTWIN):
    """One PhysTwin recording in our frame: tracked cloth points (T, N, 3) and visibility, hand points (T, M, 3),
    annotated tracks (T, 9, 3; NaN where unseen), the frame split and the frame rate."""
    d = safe_pickle.load(Path(root) / case / "final_data.pkl")
    gt = safe_pickle.load(Path(root) / case / "gt_track_3d.pkl")
    meta = json.loads((Path(root) / case / "metadata.json").read_text())
    split = json.loads((Path(root) / case / "split.json").read_text())
    return {"P": (d["object_points"] @ FLIP.T).astype(np.float64), "vis": d["object_visibilities"].astype(bool),
            "C": (d["controller_points"] @ FLIP.T).astype(np.float64), "gt": (np.asarray(gt, np.float64) @ FLIP.T),
            "split": split, "fps": float(meta["fps"]), "source": "phystwin", "case": case}


def rest_frame(P, vis, first=10):
    """The flattest of the first frames (smallest spread in height of the visible points): the cloth at rest."""
    spread = [np.percentile(P[t, vis[t], 2], 99) - np.percentile(P[t, vis[t], 2], 1) for t in range(min(first, len(P)))]
    return int(np.argmin(spread)), float(min(spread))


def rest_grid(p0, h=H):
    """A flat grid of ~h cells over the cloth at rest (points p0, z up), aligned with the cloth's edges: the
    in-plane rotation whose rectangle (0.2-99.8 percentiles of the points along each axis) is smallest (a square
    cloth has no principal axes to align with). Returns the grid record and every point's in-plane coordinates
    relative to the grid's corner."""
    c = p0.mean(0)
    q = p0 - c

    def box(th):
        a, b = q[:, 0] * np.cos(th) + q[:, 1] * np.sin(th), -q[:, 0] * np.sin(th) + q[:, 1] * np.cos(th)
        lo = np.array([np.percentile(a, 0.2), np.percentile(b, 0.2)])
        hi = np.array([np.percentile(a, 99.8), np.percentile(b, 99.8)])
        return a, b, lo, hi

    ths = np.radians(np.arange(0.0, 90.0, 0.25))
    th = ths[int(np.argmin([np.prod(box(t)[3] - box(t)[2]) for t in ths]))]
    u = np.array([np.cos(th), np.sin(th), 0.0])
    v = np.array([-np.sin(th), np.cos(th), 0.0])                       # in the table plane, u x v = +z
    a, b, lo, hi = box(th)
    nx, ny = (np.maximum(np.round((hi - lo) / h), 2)).astype(int)
    hh = float(np.mean((hi - lo) / np.array([nx, ny])))                # one cell size for both axes, as the teacher's
    mid = 0.5 * (lo + hi)
    corner = mid - 0.5 * hh * np.array([nx, ny])
    origin = c + corner[0] * u + corner[1] * v
    origin[2] = float(np.median(p0[:, 2]))                             # the cloth's mid-surface height at rest
    uv = np.stack([a - corner[0], b - corner[1]], 1)
    return {"nx": int(nx), "ny": int(ny), "h": hh, "origin": origin, "u": u, "v": v}, uv


def grid_world(g):
    verts, tris = grid(g["nx"], g["ny"], g["h"])
    world = g["origin"] + verts[:, :1] * g["u"] + verts[:, 1:2] * g["v"]
    return world, tris.astype(np.int64)


def embed(uv, g):
    """Each in-plane point's triangle in the rest grid and its barycentric weights (points outside the grid are
    clamped onto its border)."""
    nx, ny, h = g["nx"], g["ny"], g["h"]
    s = np.clip(uv / h, 0.0, [nx - 1e-6, ny - 1e-6])
    i, j = np.floor(s[:, 0]).astype(int), np.floor(s[:, 1]).astype(int)
    fx, fy = s[:, 0] - i, s[:, 1] - j
    a = j * (nx + 1) + i
    b, c_, d = a + 1, a + nx + 1, a + nx + 2
    even = (i + j) % 2 == 0
    # the grid's diagonals alternate (cloth_teacher.grid): even cells split along a-d, odd ones along b-c
    t_even1 = fx >= fy                                                  # (a, b, d)
    t_odd1 = fx + fy <= 1.0                                             # (a, b, c)
    tri = np.where(even[:, None],
                   np.where(t_even1[:, None], np.stack([a, b, d], 1), np.stack([a, d, c_], 1)),
                   np.where(t_odd1[:, None], np.stack([a, b, c_], 1), np.stack([b, d, c_], 1)))
    w = np.where(even[:, None],
                 np.where(t_even1[:, None], np.stack([1 - fx, fx - fy, fy], 1), np.stack([1 - fy, fx, fy - fx], 1)),
                 np.where(t_odd1[:, None], np.stack([1 - fx - fy, fx, fy], 1), np.stack([1 - fy, fx + fy - 1, 1 - fx], 1)))
    return tri.astype(np.int64), w.astype(np.float64)


def _init_positions(P, vis, ref, uv, rest_xyz, g, radius=0.02):
    """Start of the fit: every vertex displaced by the mean displacement (from the rest frame) of the visible points
    within `radius` of it at rest, the nearest point's where none is."""
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    verts_uv, _ = grid(g["nx"], g["ny"], g["h"])
    tree = cKDTree(uv)
    rows, cols, vals = [], [], []
    for k, nb in enumerate(tree.query_ball_point(verts_uv[:, :2], radius)):
        if not nb:
            nb = [int(tree.query(verts_uv[k, :2])[1])]
        rows += [k] * len(nb); cols += nb; vals += [1.0] * len(nb)
    W = csr_matrix((vals, (rows, cols)), shape=(len(verts_uv), len(uv)))
    X = np.zeros((len(P), len(verts_uv), 3))
    last = np.zeros((len(verts_uv), 3))
    for t in range(len(P)):
        m = vis[t] & vis[ref]
        disp = np.where(m[:, None], P[t] - P[ref], 0.0)
        num, den = W @ disp, W @ m.astype(float)
        cur = np.where(den[:, None] > 0, num / np.maximum(den, 1e-9)[:, None], last)
        X[t] = rest_xyz + cur
        last = cur
    return X


def fit_mesh(P, vis, tri, bary, rest_xyz, tris, h, iters=300, delta=0.005, strain_scale=0.005, acc_scale=0.01,
             w_stretch=1.0, w_time=0.05, radius=0.002, X0=None, dev="cpu", log=print):
    """Every frame's vertex positions (T, V, 3) fitted jointly to the visible tracked points. Terms, each scaled to
    be ~1 at its tolerance: robust (Cauchy) distance of the points' embedded positions to the tracked ones at
    `delta`; edge strain at `strain_scale`; vertex acceleration at `acc_scale` per frame^2; vertices below the
    table at 1 mm."""
    T, V = len(P), len(rest_xyz)
    e = torch.as_tensor(mesh_edges(tris)[0].astype(np.int64), device=dev)
    rx = torch.as_tensor(rest_xyz, dtype=torch.float64, device=dev)
    L0 = (rx[e[:, 0]] - rx[e[:, 1]]).norm(dim=-1)
    Pt = torch.as_tensor(np.nan_to_num(P), dtype=torch.float64, device=dev)
    Vt = torch.as_tensor(vis, dtype=torch.float64, device=dev)
    ti = torch.as_tensor(tri, device=dev)
    bw = torch.as_tensor(bary, dtype=torch.float64, device=dev)
    X = torch.nn.Parameter(torch.as_tensor(X0 if X0 is not None else np.broadcast_to(rest_xyz, (T, V, 3)).copy(),
                                           dtype=torch.float64, device=dev))
    opt = torch.optim.LBFGS([X], lr=1.0, max_iter=iters, history_size=30, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14)
    nvis = Vt.sum()

    def terms():
        pred = (X[:, ti] * bw[None, :, :, None]).sum(2)                # (T, N, 3)
        r = (pred - Pt).norm(dim=-1) / delta
        data = (Vt * torch.log1p(r ** 2)).sum() / nvis
        L = (X[:, e[:, 0]] - X[:, e[:, 1]]).norm(dim=-1)
        stretch = (((L - L0) / L0 / strain_scale) ** 2).mean()
        acc = X[2:] - 2 * X[1:-1] + X[:-2]
        time = ((acc / acc_scale) ** 2).sum(-1).mean()
        below = torch.relu(radius - X[..., 2]) / 1e-3
        table = (below ** 2).mean()
        return data, stretch, time, table

    def closure():
        opt.zero_grad()
        d, s, t, b = terms()
        loss = d + w_stretch * s + w_time * t + b
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        d, s, t, b = terms()
    log(f"  fit: data {d.item():.3f}  stretch {s.item():.3f}  time {t.item():.3f}  table {b.item():.4f}")
    return X.detach().cpu().numpy()


def track_mesh(P, vis, tri, bary, rest_xyz, tris, ref, iters=60, delta=0.005, strain_scale=0.005, acc_scale=0.01,
               w_stretch=1.0, w_time=0.05, radius=0.002, dev="cpu", log=print):
    """Frame by frame from the rest frame (forwards, then backwards to frame 0), each frame started from the last
    one moved at its last velocity and fitted with fit_mesh's terms, its acceleration against the two frames before
    it: a large deformation is followed in small steps rather than found from one guess."""
    T, V = len(P), len(rest_xyz)
    e = torch.as_tensor(mesh_edges(tris)[0].astype(np.int64), device=dev)
    rx = torch.as_tensor(rest_xyz, dtype=torch.float64, device=dev)
    L0 = (rx[e[:, 0]] - rx[e[:, 1]]).norm(dim=-1)
    ti = torch.as_tensor(tri, device=dev)
    bw = torch.as_tensor(bary, dtype=torch.float64, device=dev)
    X = np.zeros((T, V, 3))
    X[ref] = rest_xyz

    def one(t, prev1, prev2):
        Pt = torch.as_tensor(np.nan_to_num(P[t]), dtype=torch.float64, device=dev)
        Vt = torch.as_tensor(vis[t], dtype=torch.float64, device=dev)
        p1 = torch.as_tensor(prev1, dtype=torch.float64, device=dev)
        p2 = torch.as_tensor(prev2, dtype=torch.float64, device=dev)
        x = torch.nn.Parameter(2 * p1 - p2)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-14)
        nv = Vt.sum().clamp_min(1.0)

        def closure():
            opt.zero_grad()
            r = ((x[ti] * bw[:, :, None]).sum(1) - Pt).norm(dim=-1) / delta
            data = (Vt * torch.log1p(r ** 2)).sum() / nv
            L = (x[e[:, 0]] - x[e[:, 1]]).norm(dim=-1)
            stretch = (((L - L0) / L0 / strain_scale) ** 2).mean()
            time = (((x - 2 * p1 + p2) / acc_scale) ** 2).sum(-1).mean()
            table = ((torch.relu(radius - x[:, 2]) / 1e-3) ** 2).mean()
            loss = data + w_stretch * stretch + w_time * time + table
            loss.backward()
            return loss

        opt.step(closure)
        return x.detach().cpu().numpy()

    for t in range(ref + 1, T):
        X[t] = one(t, X[t - 1], X[t - 2] if t - 2 >= ref else X[t - 1])
    for t in range(ref - 1, -1, -1):
        X[t] = one(t, X[t + 1], X[t + 2] if t + 2 <= ref else X[t + 1])
    return X


def residuals(X, P, vis, tri, bary):
    """Distance of each visible tracked point to its embedded position on the fitted mesh (T, N; NaN where unseen)."""
    pred = (X[:, tri] * bary[None, :, :, None]).sum(2)
    r = np.linalg.norm(pred - P, axis=-1)
    return np.where(vis, r, np.nan)


def strains(X, tris, rest_xyz):
    e = mesh_edges(tris)[0]
    L0 = np.linalg.norm(rest_xyz[e[:, 0]] - rest_xyz[e[:, 1]], axis=-1)
    L = np.linalg.norm(X[:, e[:, 0]] - X[:, e[:, 1]], axis=-1)
    return L / L0 - 1.0


def split_hands(C, apart=0.08):
    """The hand points (T, M, 3) as one or two hands: two when the two k-means clusters of the first frame stay more
    than `apart` from each other in every frame. Returns a label per point."""
    from scipy.cluster.vq import kmeans2
    _, lab = kmeans2(C[0], 2, seed=0, minit="++")
    if lab.min() == lab.max():
        return np.zeros(C.shape[1], int)
    gap = np.linalg.norm(C[:, lab == 0].mean(1) - C[:, lab == 1].mean(1), axis=-1)
    return lab if gap.min() > apart else np.zeros(C.shape[1], int)


def held_from_hands(X, C, rest_xyz, grasp, g, patch=R_HELD, moved=0.003, lifted=0.02):
    """The cloth the hands hold, per frame (T, V). PhysTwin's hand points are the hand's visible surface (the back
    of the hand and the knuckles, 4-10 cm from the cloth during a lift, not a rigid body), so where a hand holds the
    cloth comes from the recording's grasp type (PHYSTWIN_GRASP, read off its video) and the fitted cloth, at the
    first frame its top is `lifted` above its start: a "pinch" holds the apex of the tent it lifts (the median, at
    rest, of the highest vertex over the frames the top is 3-10 cm up: the apex slides along the tent's ridge as it
    rises); "corner" / "corners" hold the grid corner nearest each hand horizontally. Held from the last frame before
    the lift that the held vertex was still (within `moved` of its first-frame position) to the end (no recording lets
    go), with the vertices within `patch` of it at rest."""
    held = np.zeros(X.shape[:2], bool)
    top = X[:, :, 2].max(1)
    up = np.flatnonzero(top > top[0] + lifted)
    if not len(up):
        return held, [{"held": False, "why": "the cloth is never lifted"}]
    t1 = int(up[0])
    lab = split_hands(C)
    hands = [C[t1, lab == h].mean(0) for h in np.unique(lab)]
    nx, ny = g["nx"], g["ny"]
    corners = np.array([0, nx, ny * (nx + 1), ny * (nx + 1) + nx])
    if grasp == "pinch":                                               # the apex slides along the tent's ridge as
        mid = np.flatnonzero((top > top[0] + 0.03) & (top < top[0] + 0.10))   # it rises: its median at rest over
        mid = mid if len(mid) else np.array([t1])                     # the mid-lift (top 3-10 cm up)
        apex = rest_xyz[np.argmax(X[mid, :, 2], axis=1)]
        picks = [int(np.argmin(np.linalg.norm(rest_xyz - np.median(apex, axis=0), axis=-1)))]
    elif grasp in ("corner", "corners"):
        picks = []
        for c in hands[:1 if grasp == "corner" else 2]:
            d = np.linalg.norm(X[t1, corners, :2] - c[:2], axis=-1)
            d[np.isin(corners, picks)] = np.inf
            picks.append(int(corners[np.argmin(d)]))
    else:
        raise ValueError(grasp)
    disp = np.linalg.norm(X - X[:1], axis=-1)
    info = []
    for v in picks:
        still = np.flatnonzero(disp[:t1 + 1, v] < moved)              # the last frame before the lift it was still
        t0 = int(still[-1]) if len(still) else 0
        near = np.linalg.norm(rest_xyz - rest_xyz[v], axis=-1) <= patch
        held[t0:, near] = True
        hd = min(np.linalg.norm(X[t1, v, :2] - c[:2]) for c in hands)
        info.append({"grasp": grasp, "vertex": v, "from_frame": t0, "vertices": int(near.sum()), "lift_frame": t1,
                     "hand_horizontal_m": float(hd), "held": True})
    return held, info


FIT_OK_MM = 6.0                                        # a frame's fit is trusted when its median residual is below this


class RealClothEpisode:
    """A converted real recording with the interface of lsim.train.cloth_windows.ClothEpisode, so that windows,
    rollouts and the evaluation scripts run on it unchanged: the fitted vertices, the held patch, the hand points as
    the kinematic tokens (their rest offsets from their centroid at the rest frame), the table as the only surface.
    The material is unknown (its slots are placeholders at the prior's centre: hide all four) and the grasp force is
    not measured (always EMPTY). frame_ok: frames whose fit is trusted (median residual under FIT_OK_MM)."""

    grasp_measured = False

    def __init__(self, path):
        from lsim.train.cloth_windows import C_CENTRE, _grid
        path = Path(path)
        f = np.load(path.with_suffix(".npz"))
        self.meta = json.loads(path.with_suffix(".json").read_text())
        self.name = f"{self.meta['source']}/{self.meta['case']}"
        g = self.meta["grid"]
        self._x = f["x"].astype(np.float32)
        self.held = f["held"].astype(bool)
        self.hand = f["hand"].astype(np.float32)
        self.n_frames = len(self._x)
        self.rest, self.tris, self.edges, self.area = _grid(int(g["nx"]), int(g["ny"]), float(g["h"]))
        w, Y, c, mu = np.exp(C_CENTRE)
        self.c = {"nx": g["nx"], "ny": g["ny"], "h": g["h"], "r": 0.003, "w": float(w), "Y": float(Y),
                  "bending_length": float(c), "mu": float(mu), "unknown": True}
        self.wd = {"table_mu": 0.5, "ramp": None, "obstacles": [], "init": "real", "families": [], "events": []}
        self.grasp_F = np.zeros((self.n_frames, 3), np.float32)
        self.contact_F = np.zeros((self.n_frames, 3, 3), np.float32)
        ref = int(self.meta["rest_frame"])
        self.gripper_rest = (self.hand[ref] - self.hand[ref].mean(0)).astype(np.float64)
        self.obs, self.obs_vis = f["obs"], f["obs_vis"]
        self.obs_tri, self.obs_bary = f["obs_tri"], f["obs_bary"]
        self.gt, self.gt_tri, self.gt_bary = f["gt"], f["gt_tri"], f["gt_bary"]
        r = residuals(self._x.astype(np.float64), self.obs.astype(np.float64), self.obs_vis, self.obs_tri, self.obs_bary)
        with np.errstate(all="ignore"):
            self.frame_ok = np.nan_to_num(np.nanmedian(r, axis=1), nan=1.0) < FIT_OK_MM / 1e3

    def x(self, frames):
        frames = np.clip(np.asarray(frames), 0, self.n_frames - 1)
        return self._x[frames]

    def gripper_points(self, frames):
        return self.hand[np.clip(np.asarray(frames), 0, self.n_frames - 1)]

    def statics(self):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), np.float32), np.zeros(0), np.zeros(0, bool)

    def planes(self):
        """The table top (as large as needed: the real one's edges are out of reach) and a floor far below."""
        out = [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.0, 2.0, np.log(0.5)),
               ((0.0, 0.0, -0.75), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 10.0, 10.0, np.log(0.5))]
        return np.array([np.r_[np.ravel(o), np.ravel(n), np.ravel(u), np.ravel(v), a, b, m] for o, n, u, v, a, b, m in out],
                        np.float32)

    def ok(self):
        return True

    def track_error(self, X, frames, stat="median", moved_from=None, min_move=0.01):
        """Distance (m) of the visible tracked points to their embedded positions on vertex positions X
        (len(frames), V, 3), per frame: the error of a prediction against what the cameras saw. The median by
        default: a tracked point that left the material it started on (the tracker's drift, up to 3 cm on folded
        cloth) would weigh on a mean whatever the prediction. moved_from: only the points whose cloth has moved more
        than `min_move` since that frame -- in these lifts most of the cloth lies still on the table, and an error over
        every point is mostly theirs (the exact layer alone scored as well as the fitted teacher). Moved by the
        reconstruction, not the raw track: a drifting track "moves" without its cloth, and selecting by the track's
        own motion picked mostly the tracker's failures (a floor of 25-67 mm where the annotated tracks sit 3 mm
        from the reconstruction). NaN where nothing has moved."""
        frames = np.asarray(frames)
        r = residuals(np.asarray(X, np.float64), self.obs[frames].astype(np.float64), self.obs_vis[frames],
                      self.obs_tri, self.obs_bary)
        if moved_from is not None:
            pe = (self._x[np.r_[moved_from, frames]][:, self.obs_tri].astype(np.float64)
                  * self.obs_bary[None, :, :, None]).sum(2)                    # the points on the reconstruction
            r = np.where(np.linalg.norm(pe[1:] - pe[:1], axis=-1) > min_move, r, np.nan)
        with np.errstate(all="ignore"):
            return np.nanmedian(r, axis=1) if stat == "median" else np.nanmean(r, axis=1)

    def gt_error(self, X, frames, moved_from=None, min_move=0.01):
        """PhysTwin's track error: the mean distance of the hand-annotated points (seen at the rest frame, where they
        are embedded) to their embedded positions on X, per frame (NaN where none is seen). moved_from: only the
        annotated points that have moved more than `min_move` since that frame (annotations do not drift)."""
        frames = np.asarray(frames)
        k = self.gt_tri[:, 0] >= 0
        p = (np.asarray(X, np.float64)[:, self.gt_tri[k]] * self.gt_bary[k][None, :, :, None]).sum(2)
        g = self.gt[frames][:, k].astype(np.float64)
        d = np.linalg.norm(p - g, axis=-1)
        if moved_from is not None:
            g0 = self.gt[moved_from][k].astype(np.float64)
            d = np.where(np.linalg.norm(g - g0[None], axis=-1) > min_move, d, np.nan)
        with np.errstate(all="ignore"):
            return np.nanmean(d, axis=1)


def load_real_cloth(source="phystwin", cases=None):
    root = OUT / source
    names = cases or sorted(p.stem for p in root.glob("*.json"))
    return [RealClothEpisode(root / n) for n in names]
