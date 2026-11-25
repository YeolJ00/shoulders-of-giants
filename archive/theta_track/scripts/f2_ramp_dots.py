"""Dot trajectories for the friction probe: one ramp, several frictions, same start.

This is the observable the friction fit actually inverts. The object starts at the same
place for every mu -- the ramp is already at its full angle -- so the only thing that
differs between traces is how far it gets.
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
from t0_tracking import pulled_camera

TAG = sys.argv[1] if len(sys.argv) > 1 else "sorter"
TILT = int(sys.argv[2]) if len(sys.argv) > 2 else 24
cfg = json.loads(Path("outputs/scene/_t_solo.json").read_text())[TAG]
box, pull = tuple(cfg["box"]), cfg["pull"]
side = box[2] - box[0]
cam = pulled_camera(pull, view="a")
lab = Path(f"outputs/scene/t_{TAG}")
meta = json.loads((lab / "meta.json").read_text())
spec = [r for r in json.loads((lab / "spec.json").read_text())
        if r["key"].startswith(f"t{TILT:02d}")
        and not (r["off_table"] or r["lifted"] or r["tumbled"])]
spec.sort(key=lambda r: r["value"])

im = imageio.get_reader(str(lab / "clips" / f"{spec[0]['key']}.mp4"))
frame0 = next(iter(im))[..., :3]; im.close()
fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=150)
ax.imshow(frame0, extent=[0, side, side, 0])
cmap = plt.get_cmap("viridis")
for i, r in enumerate(spec):
    P = []
    for fr in meta[r["key"]][:25]:
        uv, ok = project_points(cam, np.asarray(fr, float).reshape(-1, 3))
        uv = uv[ok]
        P.append((uv.mean(0) - np.array([box[0], box[1]])))
    P = np.array(P)
    c = cmap(i / max(len(spec) - 1, 1))
    ax.plot(P[:, 0], P[:, 1], "-", color=c, lw=2.0, zorder=4,
            label=f"$\\mu$={r['value']:.3f}  ({abs(r['slide_cm']):.0f} cm)")
    ax.scatter(P[::6, 0], P[::6, 1], s=11, color=c, zorder=5)
ax.scatter([P[0, 0]], [P[0, 1]], s=60, facecolor="none", edgecolor="w", lw=2, zorder=6)
ax.set_xlim(0, side); ax.set_ylim(side, 0); ax.axis("off")
ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
fig.tight_layout(pad=0.1)
b = io.BytesIO(); fig.savefig(b, format="png", dpi=150); plt.close(fig)
u = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
out = Path(f"/tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/"
           f"b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad/ramp_{TAG}_{TILT}.txt")
out.write_text(u)
print(f"{TAG} at {TILT} deg: {len(spec)} frictions, "
      f"{abs(spec[0]['slide_cm']):.0f}-{abs(spec[-1]['slide_cm']):.0f} cm  -> {out.name}")
