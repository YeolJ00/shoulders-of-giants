"""Contact damping, properly: several drop heights, then across assets.

The standing cd result is 7 clips on one asset at ONE drop height. That is the sample size
at which the CoM readout measured rho = +0.107 and looked broken, and then measured +0.961
on 30 clips -- so rho = +0.893 on 7 points is not yet a result, it is a hint. Two things are
wrong with it and both are fixed here.

  TOO FEW CLIPS.  7 points cannot fit a 174-feature readout. 27 per asset.

  ONE CONDITION.  The old sweep varied cd at a single 22 cm drop. An outcome depends on
  every parameter that shapes it, so one condition cannot identify any one of them -- the
  lesson from the tilt probe, where one ramp angle gave 29% on mu and several gave 16%.
  DROP HEIGHT is the condition we set here, exactly as ramp angle is for tilt: it changes
  the impact speed, hence how much energy the contact has to dissipate, so cd and height
  are separable only when both are varied.

Geometry note: the drop is at the table CENTRE, not near the edge. probe_bounce_settle.bounce
drops near the edge and asks whether the object leaves the table, which is a threshold
readout; the magnitude readout wants the object to stay in frame and keep bouncing, and an
object that rolls off has its later rebounds measured against the floor.

Solver set: this probe's own (pen_frac = 0.005 via probe_bounce_settle.build). At the tilt
rig's 1e-4 the contact is 50x stiffer and the rebound spans 1.03x over a 64x cd sweep --
rendering that would render a probe that no longer responds to its own parameter.

    W scripts/o12_damping.py sim                 # ONLY=tag,tag to restrict
    ... blender per lab, VIEWS=a CAM_PULL=<from _o12_box.json> ...
    W scripts/o12_damping.py clip
    CUDA_VISIBLE_DEVICES=$G C scripts/o4_joint.py score --lab outputs/scene/d_<tag>
    W scripts/o12_damping.py fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

TARGET_W = 0.19                       # same common size as the friction run
CDS = np.geomspace(400.0, 25600.0, 9)
HEIGHTS = (0.12, 0.22, 0.32)          # the condition we set
MASS = 0.40
ASSETS = [("bowl", "rigid/wooden_bowl_01"), ("pot", "rigid/brass_pot_01"),
          ("teapot", "rigid/Threshold_Porcelain_Teapot_White"),
          ("sorter", "rigid/Poppin_File_Sorter_Blue"),
          ("box", "rigid/cardboard_box_01"),
          ("books", "rigid/book_encyclopedia_set_01"),
          ("shark", "rigid/Weisshai_Great_White_Shark")]
OUT = Path("docs/results/o12_damping.json")
BOX = Path("outputs/scene/_o12_box.json")


def lab_of(tag):
    return Path(f"outputs/scene/d_{tag}")


def scale_for(asset):
    from src.data.assets import load_asset
    cat, _, nm = asset.rpartition("/")
    e = load_asset(cat, nm).extents
    return TARGET_W / float(max(e[0], e[1]))


def rebound(z):
    """First rebound height in mm: the rise after the FIRST contact, not the global min.

    Taking the global minimum measured the floor for any object that later rolled off the
    table, and every rebound then came out 0.
    """
    d = np.diff(z)
    lo = next((i for i in range(1, len(d)) if d[i - 1] < 0 <= d[i]), None)
    if lo is None:
        return None
    up = next((i for i in range(lo + 1, len(d)) if d[i] < 0), len(z) - 1)
    return float(1000.0 * (z[up] - z[lo]))


def do_sim(args):
    import warp as wp
    import probe_bounce_settle as PB
    import v1_killtest as V
    wp.init()
    only = os.environ.get("ONLY")
    allmeta = {}
    with wp.ScopedDevice("cuda:0"):
        for tag, asset in ASSETS:
            if only and tag not in only.split(","):
                continue
            sc = scale_for(asset)
            PB.OBJ, PB.SCALE = asset, sc
            lab = lab_of(tag); lab.mkdir(parents=True, exist_ok=True)
            print(f"\n{tag}: {asset}  scale {sc:.3f}", flush=True)
            scenes, meta, spec = {}, {}, []
            for h in HEIGHTS:
                for cd in CDS:
                    s = PB.build(float(cd), V.NF, pos=(0.0, 0.0, PB.GZ + h), mass=MASS)
                    r = V.track(s, stride=PB.SUBSTEPS)   # stride must match the BUILD
                    if r is None:
                        print(f"   h={h:.2f} cd={cd:7.0f} DIVERGED"); continue
                    z = np.array([p["loc"][2] for p in r[0]])
                    rb = rebound(z)
                    if rb is None:
                        print(f"   h={h:.2f} cd={cd:7.0f} no contact in window"); continue
                    key = f"{tag}_h{int(100*h):02d}_cd{cd:07.0f}"
                    scenes[key] = {"tilt_deg": 0.0, "objects": {"subject": r[0]}}
                    meta[key] = r[1]
                    spec.append({"key": key, "probe": "bounce", "param": "cd",
                                 "value": float(cd), "ref": float(h), "tilt": 0.0,
                                 "slide_cm": rb, "asset": tag, "scale": sc,
                                 "height": float(h)})
                    print(f"   h={h:.2f} cd={cd:7.0f} rebound {rb:7.2f} mm", flush=True)
            (lab / "scene_poses.json").write_text(json.dumps(scenes))
            (lab / "scene.json").write_text(json.dumps(
                {"ground_z": PB.GZ, "objects": {"subject": {
                    "asset": asset, "scale": sc, "rot_z": 0.0,
                    "pos": [0.0, 0.0, PB.GZ + 0.30]}}}))
            (lab / "meta.json").write_text(json.dumps(meta))
            (lab / "spec.json").write_text(json.dumps(spec, indent=1))
            allmeta.update({f"{tag}:{k}": v for k, v in meta.items()})
            print(f"  wrote {len(scenes)} scenes to {lab}")
    if allmeta and not only:
        _choose_box(allmeta)
    return 0


def _choose_box(allmeta):
    """ONE box and pull over every asset -- framing must not vary with the comparison."""
    import o2_boundary as O
    from src.render.crop import project_points
    from src.render.views import VIEWS
    from t0_tracking import pulled_camera
    W, H = VIEWS["a"]["width"], VIEWS["a"]["height"]
    best = None
    for pull in (1.00, 1.20, 1.60, 2.00, 2.50):
        cam = pulled_camera(pull, view="a")
        box = O.union_box(cam, allmeta, W, H)
        worst, bad = 1.0, 0
        for v in allmeta.values():
            uv, ok = project_points(cam, np.asarray(v, float).reshape(-1, 3))
            uv = uv[ok]
            c = float(((uv[:, 0] >= box[0]) & (uv[:, 0] <= box[2]) &
                       (uv[:, 1] >= box[1]) & (uv[:, 1] <= box[3])).mean())
            worst = min(worst, c); bad += c < 0.99
        print(f"  pull {pull:.2f}: box {box[2]-box[0]:>3}px worst {100*worst:5.1f}% "
              f"{bad:>3}/{len(allmeta)} out of frame")
        if bad == 0 and best is None:
            best = (pull, box)
    if best is None:
        print("  NO pull contains every clip")
        return
    BOX.write_text(json.dumps({"pull": best[0], "box": list(best[1])}))
    print(f"  chose CAM_PULL={best[0]:.2f}, box {best[1]} -> {BOX}")


def do_clip(args):
    cfg = json.loads(BOX.read_text())
    os.environ["CROP_BOX"] = ",".join(str(v) for v in cfg["box"])
    os.environ["CAM_PULL"] = str(cfg["pull"])
    import o2_boundary
    only = os.environ.get("ONLY")
    for tag, _ in ASSETS:
        if only and tag not in only.split(","):
            continue
        if not (lab_of(tag) / "spec.json").exists():
            continue
        sys.argv = [sys.argv[0], "clip"]
        o2_boundary.LAB = lab_of(tag)
        print(f"{tag}:", flush=True)
        o2_boundary.do_clip(args)
    return 0


def _sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float("nan") if d == 0 else float((ra * rb).sum() / d)


def _rows():
    import glob
    import o4_joint as O
    probs = {}
    for f in glob.glob("outputs/outcome/cache/*.json"):
        d = json.load(open(f))
        probs[str(Path(d["clip"]).resolve())] = np.asarray(d["probs"], np.float64)
    pull = json.loads(BOX.read_text())["pull"] if BOX.exists() else 1.20
    out = {}
    for tag, _ in ASSETS:
        lab = lab_of(tag)
        if not (lab / "spec.json").exists():
            continue
        con = O.containment(lab, pull=pull)
        rr = []
        for r in json.loads((lab / "spec.json").read_text()):
            c = (lab / "clips" / f"{r['key']}.mp4").resolve()
            p = probs.get(str(c))
            if p is None or con.get(r["key"], 0) < O.MIN_CONTAIN:
                continue
            rr.append({**r, "p": p})
        if rr:
            out[tag] = rr
    return out


def do_fit(args):
    import o4_joint as O
    R = _rows()
    if not R:
        sys.exit("no scored clips -- run the score stage first")
    print(f"{len(R)} assets: " + ", ".join(f"{k} ({len(v)})" for k, v in R.items()))
    res = {"per_asset": {}, "holdout": {}}

    print("\nSIMULATOR SIDE -- does the rebound respond to cd at each drop height?")
    print(f"  {'asset':<8}{'height':>8}{'rebound range':>16}{'rho(cd, rebound)':>18}")
    for t in R:
        for h in sorted({r["height"] for r in R[t]}):
            g = sorted([r for r in R[t] if r["height"] == h], key=lambda r: r["value"])
            y = np.array([r["slide_cm"] for r in g]); c = np.array([r["value"] for r in g])
            print(f"  {t:<8}{h:>8.2f}{y.min():>7.1f}-{y.max():<8.1f}"
                  f"{_sp(c, y):>18.3f}")

    print("\nREADOUT -- each asset calibrated on its own clips, leave-one-out")
    print(f"  {'asset':<8}{'n':>4}{'LOO sd':>9}{'factor':>8}{'rho(read,true)':>16}")
    X = {t: np.log(np.array([r["p"] for r in R[t]]) + 1e-8) for t in R}
    Y = {t: np.log10(np.maximum([r["slide_cm"] for r in R[t]], 1e-2)) for t in R}
    for t in R:
        pr = np.empty(len(Y[t]))
        for j in range(len(Y[t])):
            k = np.ones(len(Y[t]), bool); k[j] = False
            pr[j] = O.ridge(X[t][k], Y[t][k])(X[t][j][None])[0]
        sd = float((Y[t] - pr).std())
        res["per_asset"][t] = {"n": len(Y[t]), "loo_sd": sd, "rho": _sp(pr, Y[t])}
        print(f"  {t:<8}{len(Y[t]):>4}{sd:>9.3f}{10**sd:>8.2f}{_sp(pr, Y[t]):>16.3f}")

    print("\nRECOVERING cd -- hold out a whole cd, refit, read it back from its clips")
    print("  one drop height at a time against all heights joined")
    for t in R:
        fwd = {}
        for h in sorted({r["height"] for r in R[t]}):
            g = sorted([r for r in R[t] if r["height"] == h], key=lambda r: r["value"])
            c = np.log10([r["value"] for r in g])
            y = np.log10(np.maximum([r["slide_cm"] for r in g], 1e-2))
            keep = [0]
            for i in range(1, len(y)):                 # monotone prefix, for inversion
                if y[i] < y[keep[-1]]:
                    keep.append(i)
            fwd[h] = (c[keep], y[keep])
        cds = sorted({r["value"] for r in R[t]})
        e1, ej = [], []
        for c0 in cds:
            hold = [i for i, r in enumerate(R[t]) if r["value"] == c0]
            keep = [i for i, r in enumerate(R[t]) if r["value"] != c0]
            if len(hold) < 2 or len(keep) < 10:
                continue
            rd = O.ridge(X[t][keep], Y[t][keep])
            hats = []
            for i in hold:
                h = R[t][i]["height"]
                cc, yy = fwd[h]
                if len(cc) < 3:
                    continue
                o = np.argsort(yy)
                hats.append(float(10 ** np.interp(rd(X[t][i][None])[0], yy[o], cc[o])))
            if not hats:
                continue
            j = float(10 ** np.mean(np.log10(hats)))
            e1 += [abs(np.log10(x / c0)) for x in hats]
            ej.append(abs(np.log10(j / c0)))
        if ej:
            res["holdout"][t] = {"single_pct": float(100 * (10 ** np.median(e1) - 1)),
                                 "joint_pct": float(100 * (10 ** np.median(ej) - 1)),
                                 "n": len(ej)}
            print(f"  {t:<8} one height {100*(10**np.median(e1)-1):>5.0f}%   "
                  f"heights joined {100*(10**np.median(ej)-1):>5.0f}%   (n={len(ej)})")
    print(f"  {'prior':<8} flat over 400-25600 = a 64x range, ~700% median")

    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.set_defaults(f=do_sim)
    s = sub.add_parser("clip"); s.set_defaults(f=do_clip)
    s = sub.add_parser("fit"); s.set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
