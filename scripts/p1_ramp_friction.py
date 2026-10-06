"""P1 -- friction from REAL footage: Physics 101 ramp, cardboard cubes, side view.

The first measurement in this project that comes from reality rather than from our
simulator or a model. Real objects released from rest on a real ramp; their acceleration
along it is a = g (sin a - mu cos a), so mu = tan a - a_meas / (g cos a).

Why cardboard first: the 9 cardboard objects are CUBES -- their measured volumes are exact
cubes (343.0 = 7.0^3, 729.0 = 9.0^3 ...) -- so each cube's face height in the side view gives
pixels per centimetre with no camera calibration. (The Kinect depth is shipped as 8-bit
grey video, so its metric scale is not recoverable.)

Per clip (Camera_1, side view, ~30 fps):
  1. CoTracker3 on a dense grid from frame 0.
  2. Cube tracks = points that travel far along the dominant motion direction (the hand
     releases and stays put); the cube trajectory is their median.
  3. Scale: the cube's face is segmented at the END of the clip by differencing against
     frame 0 (the end position is empty table at frame 0); its height in px = edge length.
  4. Along-ramp distance s(t) on the accelerating segment (release -> peak speed), fit
     s = s0 + 1/2 a (t - t0)^2 with t0 >= release.
  5. mu = tan(alpha) - a / (g cos alpha), alpha = 20 deg (the dataset's nominal setting).

Checks, by the project's rules: the same cube across 3-14 trials (repeatability), across
the 9 cubes (one material), and the 10 deg clips must NOT slide if mu > tan 10 = 0.18.

    C scripts/p1_ramp_friction.py one --clip cardboard_03/20_01/01   (visual check)
    C scripts/p1_ramp_friction.py run
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
ROOT = REPO / "data/phys101/phys101/scenarios/ramp"
G = 9.81
SCR = Path("/tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/"
           "b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad")
OUT = REPO / "docs/results/p1_ramp_friction.json"


def edges_cm():
    import zipfile
    z = zipfile.ZipFile(REPO / "data/phys101/phys101_v1.0.zip")
    v = {l.split()[0]: float(l.split()[1]) for l in z.read("phys101/objects/vol").decode()
         .split("\n") if l.strip()}
    return {k: v[k] ** (1 / 3) for k in v if k.startswith("cardboard")}


def load(clip, view="Camera_1"):
    import imageio.v2 as io
    r = io.get_reader(str(ROOT / clip / f"{view}.mp4"))
    fps = float(r.get_meta_data()["fps"])
    fr = np.stack([f[..., :3] for f in r]); r.close()
    return fr, fps


_COT = None


def track(fr, grid=40):
    import torch
    global _COT
    if _COT is None:
        # from the LOCAL hub cache: the online repo check timed out (HTTP 504) and, retried
        # per clip, failed 52 clips in a row in p2
        _COT = torch.hub.load(str(Path.home() / ".cache/torch/hub/facebookresearch_co-tracker_main"),
                              "cotracker3_offline", source="local").to("cuda:0")
    v = torch.tensor(fr, device="cuda:0").permute(0, 3, 1, 2)[None].float()
    with torch.no_grad():
        tr, vis = _COT(v, grid_size=grid)
    return tr[0].cpu().numpy(), vis[0].cpu().numpy()          # T,N,2 / T,N


def analyse(clip, fr, fps, tr, vis, edge_cm):
    T = len(fr)
    disp = tr[-1] - tr[0]
    dist = np.linalg.norm(disp, axis=1)
    # the cube is what travels furthest; the hand moves little once it lets go
    big = dist > max(0.5 * np.percentile(dist, 99.5), 40)
    if big.sum() < 3:
        return {"clip": clip, "moved": False, "max_px": float(dist.max())}
    d = np.median(disp[big], 0); d /= np.linalg.norm(d)
    # keep only points moving ALONG that direction (rejects the hand's other motions)
    along = (disp[big] @ d) / np.maximum(dist[big], 1e-6)
    sel = np.flatnonzero(big)[along > 0.9]
    path = np.median(tr[:, sel], 1)                           # T,2
    s_px = (path - path[0]) @ d
    # scale: cube face at the end, by differencing against frame 0
    diff = np.abs(fr[-1].astype(np.int16) - fr[0].astype(np.int16)).sum(-1) > 60
    cx, cy = path[-1].astype(int)
    win = 260
    y0, y1 = max(cy - win, 0), min(cy + win, fr.shape[1])
    x0, x1 = max(cx - win, 0), min(cx + win, fr.shape[2])
    m = diff[y0:y1, x0:x1]
    rows = np.flatnonzero(m.sum(1) > 0.15 * m.sum(1).max()) if m.any() else []
    face_px = float(rows.max() - rows.min()) if len(rows) else float("nan")
    px_per_m = face_px / (edge_cm / 100.0)
    s = s_px / px_per_m
    t = np.arange(T) / fps
    v = np.gradient(s, t)
    # accelerating segment: from first sustained motion to peak speed
    start = int(np.argmax(s > 0.005))
    peak = int(np.argmax(v))
    res = {"clip": clip, "moved": True, "fps": fps, "face_px": face_px, "px_per_m": px_per_m,
           "dir_deg": float(np.degrees(np.arctan2(d[1], d[0]))), "n_pts": int(len(sel)),
           "s": s.tolist(), "start": start, "peak": peak}
    if peak - start < 4:
        res["fit"] = None
        return res
    # fit s = s0 + 1/2 a (t - t0)^2 over a grid of release times t0 (<= first motion frame)
    best = None
    for t0 in np.linspace(t[max(start - 6, 0)], t[start], 25):
        tt = t[start - 1:peak + 1] - t0
        ok = tt > 0
        A = np.stack([np.ones(ok.sum()), 0.5 * tt[ok] ** 2], 1)
        coef, *_ = np.linalg.lstsq(A, s[start - 1:peak + 1][ok], rcond=None)
        r = np.sqrt(np.mean((A @ coef - s[start - 1:peak + 1][ok]) ** 2))
        if best is None or r < best[0]:
            best = (r, float(t0), float(coef[0]), float(coef[1]))
    rms, t0, s0, a = best
    alpha = np.radians(20.0 if "/20_" in clip else 10.0)
    mu = float(np.tan(alpha) - a / (G * np.cos(alpha)))
    res["fit"] = {"a": a, "t0": t0, "s0": s0, "rms_m": rms, "mu": mu,
                  "n_frames": int(peak - start + 2), "v_peak": float(v[peak])}
    return res


def do_one(args):
    """Track + fit one clip (cosmos env); the plot is a separate step (`plot1`, warp env)."""
    E = edges_cm()
    fr, fps = load(args.clip)
    tr, vis = track(fr)
    r = analyse(args.clip, fr, fps, tr, vis, E[args.clip.split("/")[0]])
    print(json.dumps({k: v for k, v in r.items() if k != "s"}, indent=1))
    np.savez(SCR / "p1_one.npz", f0=fr[0], tr=tr, res=json.dumps(r), fps=fps)
    return 0


def do_plot1(args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = np.load(SCR / "p1_one.npz", allow_pickle=True)
    f0, tr, r, fps = Z["f0"], Z["tr"], json.loads(str(Z["res"])), float(Z["fps"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), dpi=90)
    ax[0].imshow(f0); d = tr[-1] - tr[0]
    dist = np.linalg.norm(d, axis=1)
    for j in np.flatnonzero(dist > max(0.5 * np.percentile(dist, 99.5), 40)):
        ax[0].plot(tr[:, j, 0], tr[:, j, 1], "-", lw=0.8)
    ax[0].axis("off"); ax[0].set_title(r["clip"])
    if r.get("moved"):
        t = np.arange(len(r["s"])) / fps
        ax[1].plot(t, np.array(r["s"]) * 100, "o-", ms=3, label="tracked")
        f = r.get("fit")
        if f:
            tt = np.clip(t - f["t0"], 0, None)
            ax[1].plot(t, (f["s0"] + 0.5 * f["a"] * tt ** 2) * 100, "-",
                       label=f"fit a={f['a']:.2f} m/s2  mu={f['mu']:.3f}")
            ax[1].axvspan(t[r["start"]], t[r["peak"]], alpha=0.15)
        ax[1].set_xlabel("s"); ax[1].set_ylabel("along-ramp cm"); ax[1].legend()
    fig.tight_layout(); fig.savefig(SCR / "p1_one.png")
    return 0


def do_run(args):
    E = edges_cm()
    clips = sorted(str(p.parent.relative_to(ROOT)) for p in ROOT.glob("cardboard_*/*/*/Camera_1.mp4"))
    res = []
    for c in clips:
        try:
            fr, fps = load(c)
            tr, vis = track(fr)
            r = analyse(c, fr, fps, tr, vis, E[c.split("/")[0]])
        except Exception as e:
            r = {"clip": c, "error": str(e)}
        res.append(r)
        f = r.get("fit")
        print(f"  {c:<26} " + (f"mu {f['mu']:.3f}  a {f['a']:.2f}  rms {100*f['rms_m']:.2f} cm  "
                               f"scale {r['px_per_m']:.0f} px/m" if f else
                               ("no slide" if r.get("moved") is False else r.get("error", "no fit"))),
              flush=True)
    OUT.write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("one"); s.set_defaults(f=do_one); s.add_argument("--clip", required=True)
    sub.add_parser("run").set_defaults(f=do_run)
    sub.add_parser("plot1").set_defaults(f=do_plot1)
    a = ap.parse_args()
    sys.exit(a.f(a))
