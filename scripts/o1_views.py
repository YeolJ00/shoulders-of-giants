"""G-B -- multi-view measurement and cross-view agreement.

A real physical outcome is VIEW-INVARIANT. So disagreement between cameras about what
happened is a free render-gap diagnostic: it means the classifier is reacting to the
rendering rather than to the motion. Three of our four views (a, b, c) sit on the same side
of the table and are nearly coplanar, which could never have shown this -- "d" is the
opposing azimuth added for exactly that reason.

Two things come out of one render pass:
  G-B   cross-view top-1 agreement, and whether agreement depends on theta. If agreement
        varies with theta, the view-average is not a valid soft target, because the
        measurement's reliability would itself be a function of the thing being measured.
  the MEASUREMENT the O-track actually proposes: the class distribution averaged over
        views, which mirrors the head's own 64x2x3 multi-clip logit averaging.

PASS  >= 3 of 4 views agree on top-1 for >= 80% of clips, and |rho(theta, agreement)| <= 0.3
FAIL  agreement varies with theta => view-averaging is invalid

    LAB=outputs/scene/ga  W scripts/o1_views.py clip
    CUDA_VISIBLE_DEVICES=$G C scripts/o1_views.py score
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

LAB = Path(os.environ.get("LAB", "outputs/scene/ga"))
OUT = Path("docs/results/o1_views.json")
SLIDE = "Lifting a surface with [something] on it until it starts sliding down"
NOSLIDE = "Lifting a surface with [something] on it but not enough for it to slide down"


def do_clip(args):
    """One clip per (scene, view). The crop box is computed with THAT view's camera --
    using view a's box for view d would frame empty table."""
    import imageio.v2 as imageio
    from src.render.crop import crop_clip
    from src.render.views import VIEWS
    import v1_killtest as V
    from t0_tracking import pulled_camera

    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    meta = json.loads((LAB / "meta.json").read_text())
    n = 0
    for view in VIEWS:
        cam = pulled_camera(float(os.environ.get("CAM_PULL", V.CAM_PULL)), view=view)
        W, H = VIEWS[view]["width"], VIEWS[view]["height"]
        boxes = V.shared_boxes(cam, meta, spec, W, H)
        dst = LAB / f"clips_{view}"
        dst.mkdir(parents=True, exist_ok=True)
        for key in sorted(spec):
            d = LAB / f"sim_{key}@{view}"
            ps = sorted(d.glob("f*.png"))[:V.NF]
            if len(ps) < V.NF:
                continue
            frames = np.stack([imageio.imread(p)[..., :3] for p in ps])
            box = boxes.get(key, boxes.get("__floor__"))
            w = imageio.get_writer(str(dst / f"{key}.mp4"), fps=int(V.FPS),
                                   codec="libx264", quality=9, macro_block_size=1)
            for f in crop_clip(frames, box):
                w.append_data(f)
            w.close(); n += 1
        print(f"  view {view}: {len(list(dst.glob('*.mp4')))} clips", flush=True)
    print(f"{n} clips total")
    return 0


def do_score(args):
    from src.outcome.head import OutcomeHead
    from src.render.views import VIEWS
    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    m = OutcomeHead(device="cuda:0")
    li = {n: i for i, n in enumerate(m.labels)}
    views = [v for v in VIEWS if (LAB / f"clips_{v}").exists()]
    print(f"views: {views}")

    rows = []
    for key in sorted(spec, key=lambda k: spec[k]["value"]):
        per, tops = {}, []
        for v in views:
            c = LAB / f"clips_{v}" / f"{key}.mp4"
            if not c.exists():
                continue
            p = m.score_clip(c)
            per[v] = p
            tops.append(m.labels[int(np.argmax(p))])
        if len(per) < 2:
            continue
        avg = np.mean(list(per.values()), axis=0)
        cnt = Counter(tops)
        agree = cnt.most_common(1)[0][1] / len(tops)
        rows.append({**spec[key], "n_views": len(per), "agreement": agree,
                     "top1_per_view": tops,
                     "avg_top1": m.labels[int(np.argmax(avg))],
                     "avg_margin": float(avg[li[SLIDE]] - avg[li[NOSLIDE]]),
                     "per_view_margin": {v: float(p[li[SLIDE]] - p[li[NOSLIDE]])
                                         for v, p in per.items()}})
        print(f"  mu={spec[key]['value']:5.3f}  agree {agree:.2f}  "
              f"avg margin {rows[-1]['avg_margin']:+.3f}  {rows[-1]['avg_top1'][:46]}",
              flush=True)

    ag = np.array([r["agreement"] for r in rows])
    v = np.array([r["value"] for r in rows])
    mg = np.array([r["avg_margin"] for r in rows])
    def rho(a, b):
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / d) if d else float("nan")
    frac = float((ag >= 0.75).mean())
    r_ag = rho(v, ag)
    cross = None
    for i in range(len(v) - 1):
        if mg[i] > 0 >= mg[i + 1]:
            f = mg[i] / (mg[i] - mg[i + 1])
            cross = float(np.exp(np.log(v[i]) + f * (np.log(v[i + 1]) - np.log(v[i])))); break
    star = float(rows[0]["ref"])

    print(f"\n>= 3/4 views agree on top-1: {100*frac:.0f}% of clips  (pass >= 80%)")
    print(f"rho(theta, agreement)      : {r_ag:+.3f}  (pass |rho| <= 0.3)")
    print(f"view-averaged boundary     : mu = {'n/a' if cross is None else f'{cross:.3f}'}"
          f"   vs analytic mu* = {star:.3f}")
    print(f"rho(mu, view-avg margin)   : {rho(v, mg):+.3f}")
    ok = frac >= 0.8 and abs(r_ag) <= 0.3
    print(f"\nG-B {'PASS' if ok else 'FAIL'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": m.model_id, "views": views, "rows": rows,
                               "agree_frac": frac, "rho_theta_agreement": r_ag,
                               "avg_boundary": cross, "mu_star": star,
                               "verdict": "PASS" if ok else "FAIL"}, indent=1))
    print(f"wrote {OUT}")
    return 0 if ok else 1


MODES = {"clip": do_clip, "score": do_score}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=list(MODES))
    raise SystemExit(MODES[ap.parse_args().mode](None))
