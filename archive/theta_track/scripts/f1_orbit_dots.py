"""Dot trajectories under a MOVING camera, with the background drawn for reference.

With a locked camera an object at 29 cm and one resting on the table project to nearly the
same image, so a motion prior cannot tell them apart and falls back on its marginal ("~0.99
nothing moves"). A gentle orbit separates them by parallax. Drawing the background points
alongside the object makes the distinction visible: background dots trace the CAMERA's
motion, and the object's departure from that trace is its own.
"""
import base64, io, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from src.render.crop import project_points
from t0_tracking import PinholeCam
from src.render.views import VIEWS

S = ("/tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/"
     "b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad")
ORB = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NF = 25


def cam_at(t, n, orbit_deg, view="a", pull=1.0):
    """The camera the RENDERER used at frame t -- same orbit formula as CAM_ORBIT."""
    vc = VIEWS[view]
    eye, tgt = np.asarray(vc["eye"], float), np.asarray(vc["target"], float)
    eye = tgt + (eye - tgt) * pull
    a = np.radians(orbit_deg) * (t / max(n - 1, 1) - 0.5)
    rel = eye - tgt
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return PinholeCam(tgt + R @ rel, tgt, vc["fov_deg"], vc["width"], vc["height"])


meta = json.loads(Path("outputs/scene/_orb/meta.json").read_text())["bowl_h32_cd0000400"]
n_all = len(meta)
# a patch of table, to show what the CAMERA alone does
# TWO reference sets. The orbit pivots about the table centre, so points near it barely
# move (3.8 px) while the FAR background sweeps ~27 px. That difference is the parallax,
# and it is what lets the model tell a suspended object from a resting one.
bg = np.array([[x, y, 0.706] for x in (-0.25, 0.0, 0.25) for y in (-0.15, 0.10)])
far = np.array([[x, 3.0, 1.2] for x in (-1.2, 0.0, 1.2)])

obj_tr, bg_tr, far_tr = [], [], []
for t in range(NF):
    c = cam_at(t, n_all, ORB)
    P = np.asarray(meta[t], float).reshape(-1, 3)
    uv, ok = project_points(c, P); uv = uv[ok]
    obj_tr.append(uv[np.linspace(0, len(uv) - 1, 6).astype(int)])
    uvb, okb = project_points(c, bg)
    bg_tr.append(uvb[okb])
    uvf, okf = project_points(c, far)
    far_tr.append(uvf[okf])
obj_tr = np.stack(obj_tr); bg_tr = np.stack(bg_tr); far_tr = np.stack(far_tr)

frames = sorted(Path(f"{S}/orb_frames_{ORB}").glob("f*.png"))
im = imageio.imread(frames[0])[..., :3]
H, W = im.shape[:2]
fig, ax = plt.subplots(figsize=(6.4, 5.3), dpi=150)
ax.imshow(im, extent=[0, W, H, 0])
for j in range(bg_tr.shape[1]):
    ax.plot(bg_tr[:, j, 0], bg_tr[:, j, 1], "-", color="#8a8f95", lw=1.3, alpha=0.95, zorder=3)
    ax.scatter(bg_tr[::6, j, 0], bg_tr[::6, j, 1], s=8, color="#8a8f95", zorder=4)
for j in range(far_tr.shape[1]):
    ax.plot(far_tr[:, j, 0], far_tr[:, j, 1], "-", color="#c8501d", lw=1.4, alpha=0.9, zorder=3)
    ax.scatter(far_tr[::6, j, 0], far_tr[::6, j, 1], s=8, color="#c8501d", zorder=4)
for j in range(obj_tr.shape[1]):
    ax.plot(obj_tr[:, j, 0], obj_tr[:, j, 1], "-", color="#2b5f86", lw=1.7, zorder=5)
    ax.scatter(obj_tr[::6, j, 0], obj_tr[::6, j, 1], s=9, color="#2b5f86", zorder=6)
ax.scatter(obj_tr[0, :, 0], obj_tr[0, :, 1], s=36, facecolor="none",
           edgecolor="#f2f2f2", lw=1.7, zorder=7)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.tight_layout(pad=0.1)
b = io.BytesIO(); fig.savefig(b, format="png", dpi=150); plt.close(fig)
u = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

bgd = np.linalg.norm(bg_tr[-1] - bg_tr[0], axis=-1).mean()
objd = np.linalg.norm(obj_tr[-1] - obj_tr[0], axis=-1).mean()
Path(f"{S}/orbit_dots_{ORB}.txt").write_text(u)
fard = np.linalg.norm(far_tr[-1] - far_tr[0], axis=-1).mean()
print(f"orbit {ORB} deg / {NF} frames: table {bgd:.1f} px, far background {fard:.1f} px, "
      f"object {objd:.1f} px")
print(f"wrote data URI to {S}/orbit_dots_{ORB}.txt ({len(u)//1024} KB)")
