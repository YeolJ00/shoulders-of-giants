"""Break the collinearity: vary RAMP RATE at fixed mu and fixed final tilt.

Every earlier sweep confounded three variables. Measured over the 49 clips of the 5-angle
sweep: rho(mu, slide) = -0.844, and inside every "fixed slide" band rho(mu, tilt) ran from
+0.797 to +1.000. At +1.000 there is no way to tell mu from tilt, so the claim that the
head reads displacement rather than friction was not supported by that data -- and neither
was the opposite. rho(tilt, margin) = +0.671 was in fact the strongest single correlate and
I had not tested it at all.

This design fixes mu AND the final tilt angle, and varies only how fast the table gets
there. The ramp reaches 24 deg at frame R and then holds. Longer above the slip angle means
further sliding, so:

    margin follows slide      -> displacement is the driver
    margin unchanged          -> not displacement; friction and geometry stay live
    margin follows rate itself-> it is reading the tilting motion, not the object

mu = 0.15 (slides freely), 0.30 (marginal), 0.60 (above mu* = 0.441, should not slide).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

LAB = Path(os.environ.get("LAB", "outputs/scene/ga4"))
OUT = Path("docs/results/o2c_rate.json")
TILT_END = 24.0
RATES = (24, 40, 64)          # frame at which the ramp reaches TILT_END, then holds
MUS = (0.15, 0.30, 0.60)
SLIDE = "Lifting a surface with [something] on it until it starts sliding down"
NOSLIDE = "Lifting a surface with [something] on it but not enough for it to slide down"


def angle_seq(rate, n_frames):
    """Reach TILT_END at frame `rate`, then hold. Same final geometry for every rate."""
    a = np.minimum(np.arange(n_frames) / float(rate), 1.0) * TILT_END
    return a


def gravity_from_angles(ang_per_frame, substeps, settle_frames=6, g=9.81):
    """Per-substep gravity for an arbitrary angle schedule (tilt_probe only ramps linearly)."""
    from src.sim.tilt_probe import tilt_gravity
    out = []
    for i, a in enumerate(ang_per_frame):
        th = 0.0 if i < settle_frames else np.radians(a)
        out += [tilt_gravity(th, g)] * substeps
    return np.asarray(out, np.float32)


def do_sim(args):
    import warp as wp
    import v1_killtest as V
    LAB.mkdir(parents=True, exist_ok=True)
    wp.init()
    scenes, meta, spec = {}, {}, []
    with wp.ScopedDevice("cuda:0"):
        for mu in MUS:
            for rate in RATES:
                ang = angle_seq(rate, V.NF)
                gseq = gravity_from_angles(ang, V.SUBSTEPS)
                r = V.track(V.build(mu=float(mu), pos=(V.TILT_X0, 0.0, V.GZ + 0.002),
                                    gseq=gseq), tilt_seq=list(ang))
                if r is None:
                    print(f"  mu={mu:.2f} rate={rate} DIVERGED"); continue
                key = f"mu{mu:04.2f}_r{rate:02d}"
                scenes[key] = {"tilt_deg": list(ang), "objects": {"subject": r[0]}}
                meta[key] = r[1]
                dx = 100 * (r[0][-1]["loc"][0] - r[0][0]["loc"][0])
                spec.append({"key": key, "probe": "tilt", "param": "rate",
                             "value": float(rate), "ref": float(rate), "mu": float(mu),
                             "rate": int(rate), "slide_cm": dx})
                print(f"  mu={mu:.2f}  reaches {TILT_END:.0f}deg at frame {rate:2d}  "
                      f"slid {dx:6.2f} cm")
    (LAB / "scene_poses.json").write_text(json.dumps(scenes))
    (LAB / "scene.json").write_text(Path("outputs/scene/v1/scene.json").read_text())
    (LAB / "meta.json").write_text(json.dumps(meta))
    (LAB / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote {len(scenes)} scenes")
    return 0


def do_clip(args):
    import o2_boundary
    o2_boundary.LAB = LAB
    return o2_boundary.do_clip(args)


def do_score(args):
    from src.outcome.head import OutcomeHead
    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    m = OutcomeHead(device="cuda:0")
    li = {n: i for i, n in enumerate(m.labels)}
    rows = []
    for key in sorted(spec):
        c = LAB / "clips" / f"{key}.mp4"
        if not c.exists():
            continue
        p = m.score_clip(c)
        rows.append({**spec[key],
                     "margin": float(p[li[SLIDE]] - p[li[NOSLIDE]])})
    def rho(a, b):
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / d) if d else float("nan")

    print(f"\n{'mu':>6}{'rate':>6}{'slide cm':>10}{'margin':>9}")
    for r in sorted(rows, key=lambda r: (r["mu"], r["rate"])):
        print(f"{r['mu']:>6.2f}{r['rate']:>6d}{r['slide_cm']:>10.2f}{r['margin']:>+9.3f}")

    sl = np.array([r["slide_cm"] for r in rows]); mg = np.array([r["margin"] for r in rows])
    rt = np.array([float(r["rate"]) for r in rows]); mu = np.array([r["mu"] for r in rows])
    print(f"\nAT FIXED FINAL TILT (24 deg), pooled over the 3 mu:")
    print(f"  rho(slide, margin) = {rho(sl, mg):+.3f}")
    print(f"  rho(rate,  margin) = {rho(rt, mg):+.3f}")
    print(f"  rho(mu,    margin) = {rho(mu, mg):+.3f}")
    print(f"\nWITHIN each mu -- only the rate (and hence the slide) changes:")
    for u in sorted(set(mu)):
        k = mu == u
        if k.sum() >= 3:
            sp = mg[k].max() - mg[k].min()
            print(f"  mu={u:.2f}  slide {sl[k].min():.1f}-{sl[k].max():.1f}cm  "
                  f"margin {mg[k].min():+.3f}..{mg[k].max():+.3f}  spread {sp:.3f}  "
                  f"rho(slide,margin) = {rho(sl[k], mg[k]):+.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": m.model_id, "rows": rows,
                               "rho_slide": rho(sl, mg), "rho_rate": rho(rt, mg),
                               "rho_mu": rho(mu, mg)}, indent=1))
    print(f"\nwrote {OUT}")
    return 0


MODES = {"sim": do_sim, "clip": do_clip, "score": do_score}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=list(MODES))
    raise SystemExit(MODES[ap.parse_args().mode](None))
