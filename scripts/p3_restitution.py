"""P3 -- coefficient of restitution from REAL drops: Physics 101 'fall', side view.

Objects are dropped by hand onto five surfaces (foam, glass, metal, wooden table, rug).
Restitution is scale-free, and is read TWO independent ways, which must agree (rule 2):

  velocity route   e_v = -v_after / v_before   from parabola fits on either side of impact,
                   both extrapolated to the moment the pre-impact parabola reaches the
                   table line;
  height route     e_h = sqrt(h_bounce / h_drop)   drop height from release (where the
                   pre-impact parabola has zero velocity) to the table line; bounce height
                   from the table line to the post-impact apex.

The free-fall parabola also yields g in px/s^2, i.e. the camera scale -- a third check: it
must be about the same for every clip (fixed camera).

Tracking: the objects are small in this view (10-60 px), so CoTracker3 runs on a dense grid
queried at three frames, tracking forward and backward; the object's tracks are those that
drop > 60 px and share one impact frame; its path is their median. 'table line' = the
object's resting image height at the end of the clip.

    C scripts/p3_restitution.py run
    W scripts/p3_restitution.py summary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
ROOT = REPO / "data/phys101/phys101/scenarios/fall"
OUT = REPO / "docs/results/p3_restitution.json"
_COT = None


def track_multi(fr, grid=60):
    import torch
    global _COT
    if _COT is None:
        _COT = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda:0")
    T, H, W = fr.shape[:3]
    xs = np.linspace(0, W - 1, grid); ys = np.linspace(0, H - 1, int(grid * H / W))
    gx, gy = np.meshgrid(xs, ys)
    q = []
    for t in (0, T // 3, 2 * T // 3):
        q.append(np.stack([np.full(gx.size, t), gx.ravel(), gy.ravel()], 1))
    q = torch.tensor(np.concatenate(q)[None], dtype=torch.float32, device="cuda:0")
    v = torch.tensor(fr, device="cuda:0").permute(0, 3, 1, 2)[None].float()
    with torch.no_grad():
        tr, vis = _COT(v, queries=q, backward_tracking=True)
    return tr[0].cpu().numpy(), vis[0].cpu().numpy()


def locate_rest(fr):
    """The object at rest at the END of the clip: the lowest sizeable connected blob of
    |last - first| (frame 0 shows empty table there; the hand is the higher blob)."""
    import cv2
    d = (np.abs(fr[-1].astype(np.int16) - fr[0].astype(np.int16)).sum(-1) > 60).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(d)
    cands = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 150]
    if not cands:
        return None
    i = max(cands, key=lambda i: st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT])
    x, y, w, h = st[i, :4]
    return float(cen[i][0]), float(cen[i][1]), int(w), int(h)


def track_object(fr, loc, n=4):
    """Query points inside the resting blob at the LAST frame, tracked backward."""
    import torch
    global _COT
    if _COT is None:
        _COT = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda:0")
    cx, cy, w, h = loc
    T = len(fr)
    q = np.array([[T - 1, cx + dx * w * 0.25, cy + dy * h * 0.25]
                  for dx in np.linspace(-1, 1, n) for dy in np.linspace(-1, 1, n)], float)
    v = torch.tensor(fr, device="cuda:0").permute(0, 3, 1, 2)[None].float()
    with torch.no_grad():
        tr, vis = _COT(v, queries=torch.tensor(q[None], dtype=torch.float32, device="cuda:0"),
                       backward_tracking=True)
    return tr[0].cpu().numpy(), vis[0].cpu().numpy()


def analyse(clip, fr, fps, tr):
    """y(t) of the object (median of its backward-tracked points).
    release = highest point before the fall; impact k = where the downward speed collapses
    (NOT the final resting height: a block lands on a corner, then settles lower);
    height route e_h = sqrt(h_bounce / h_drop); timing route e_t = g * tau / (2 v_before),
    tau = frames airborne after impact; g_px from the free-fall parabola (scale check)."""
    T = len(fr)
    y = np.median(tr[:, :, 1], 1)
    t = np.arange(T) / fps
    dy = np.diff(y)
    fall = np.flatnonzero(dy > 8)
    if len(fall) < 3:
        return {"clip": clip, "ok": False, "why": "no fall"}
    f0 = int(fall[0])
    k = None
    for i in range(f0 + 2, T - 1):
        if dy[i] < 0.3 * dy[i - 1] or dy[i] <= 0:
            k = i; break
    if k is not None:                     # the impact is the LOCAL MAXIMUM around the collapse
        k = k + int(np.argmax(y[k: min(k + 3, T)]))
    if k is None or k - f0 < 3:
        return {"clip": clip, "ok": False, "why": "no impact"}
    y_rel = float(y[: f0 + 1].min())
    h_drop = float(y[k] - y_rel)
    pre = list(range(f0, k + 1))
    c2, c1, c0 = np.polyfit(t[pre], y[pre], 2)
    g_px = 2 * c2
    v_before = float(c1 + 2 * c2 * t[k])
    # bounce: up from the impact point, until it is back down at the impact height
    # bounce: the first rise after impact; its apex = the first local minimum of y
    seg = y[k + 1: min(k + 10, T)]
    h_b, tau = 0.0, 0
    if len(seg) and seg.min() < y[k] - 3:
        a = 0
        while a + 1 < len(seg) and seg[a + 1] < seg[a]:
            a += 1
        h_b = float(y[k] - seg[a])
        tau = a + 1                          # frames from impact to apex
    # timing route: v_after = g * t_apex  ->  e_t = g t_apex / v_before
    e_t = float(g_px * (tau / fps) / v_before) if tau and v_before > 0 else 0.0
    return {"clip": clip, "ok": True, "fps": fps, "g_px": float(g_px), "h_drop_px": h_drop,
            "v_before": v_before, "k": k, "f0": f0, "h_bounce_px": h_b, "tau_frames": tau,
            "e_h": float(np.sqrt(max(h_b, 0) / h_drop)) if h_drop > 0 else None,
            "e_t": e_t, "y": y.tolist()}


def do_run(args):
    import imageio.v2 as io
    clips = sorted(str(p.parent.relative_to(ROOT)) for p in ROOT.glob("*/*/*/Camera_1.mp4"))
    print(f"{len(clips)} clips", flush=True)
    res = []
    for i, c in enumerate(clips):
        try:
            r = io.get_reader(str(ROOT / c / "Camera_1.mp4"))
            fps = float(r.get_meta_data()["fps"])
            fr = np.stack([f[..., :3] for f in r]); r.close()
            loc = locate_rest(fr)
            if loc is None:
                raise RuntimeError("object not found at rest")
            tr, _ = track_object(fr, loc)
            out = analyse(c, fr, fps, tr)
        except Exception as e:
            out = {"clip": c, "ok": False, "why": f"error {e}"}
        res.append(out)
        msg = (f"e_h {out['e_h']:.3f}  e_t {out['e_t']:.3f}  drop {out['h_drop_px']:.0f}px  "
               f"g_px {out['g_px']:.0f}" if out.get("ok") else out["why"])
        print(f"  [{i+1}/{len(clips)}] {c:<26} {msg}", flush=True)
        if (i + 1) % 25 == 0:
            OUT.write_text(json.dumps(res))
    OUT.write_text(json.dumps(res))
    return 0


def do_summary(args):
    import collections
    R = [r for r in json.loads(OUT.read_text())]
    ok = [r for r in R if r.get("ok")]
    g = np.array([r["g_px"] for r in ok])
    print(f"{len(ok)}/{len(R)} clips analysed.  g in px/s^2 (camera scale check): median "
          f"{np.median(g):.0f}, IQR {np.percentile(g, 25):.0f}-{np.percentile(g, 75):.0f} "
          f"-> {np.median(g)/9.81:.0f} px/m")
    both = [r for r in ok if r["tau_frames"] >= 3 and r["h_bounce_px"] > 5]
    if len(both) > 3:
        et = np.array([r["e_t"] for r in both]); eh = np.array([r["e_h"] for r in both])
        print(f"two routes on {len(both)} clearly bouncing clips: median |e_t - e_h| "
              f"{np.median(np.abs(et-eh)):.3f}, corr {np.corrcoef(et, eh)[0,1]:.2f}")
    by = collections.defaultdict(list)
    for r in ok:
        mat = r["clip"].split("/")[0].rsplit("_", 1)[0]; surf = r["clip"].split("/")[1]
        by[(mat, surf)].append(r)
    surfs = ["table", "glass", "metal", "foam", "rug"]
    mats = sorted({m for m, _ in by})
    print(f"\nrestitution e (height route, median over trials; n)")
    print(f"{'material':<11}" + "".join(f"{s:>12}" for s in surfs))
    out = {}
    for m in mats:
        row = f"{m:<11}"
        for s in surfs:
            v = [r["e_h"] for r in by.get((m, s), [])]
            out.setdefault(m, {})[s] = {"n": len(v), "e_h": float(np.median(v)) if v else None}
            row += f"{np.median(v):>8.2f} ({len(v)})" if v else f"{'-':>12}"
        print(row)
    (REPO / "docs/results/p3_restitution_summary.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(f=do_run)
    sub.add_parser("summary").set_defaults(f=do_summary)
    a = ap.parse_args()
    sys.exit(a.f(a))
