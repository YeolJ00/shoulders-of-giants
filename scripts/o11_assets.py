"""Friction across FIVE assets -- five independent fits, and one cost question.

Each asset gets its OWN readout, calibrated on its own sweep against the simulator's own
slide. That is the accurate arrangement (own-object LOO sd 0.057 against 0.150 borrowing
another object's) and it needs no ground truth, so it is the design rather than a
workaround.

The cross-asset question is about COST, not accuracy. Per-asset calibration is ~100 clips
per asset; that is fine for five and impossible for a library. Bowl -> pot transfer had good
scatter (0.150) and a bad scale -- one constant factor of 3.4 -- and removing that single
offset recovered R^2 = +0.61. If the offset is PREDICTABLE from cheap geometry, a new asset
needs a handful of clips instead of a hundred. Two objects cannot distinguish "predictable"
from "random per object"; five can.

Two design constraints, both of which would otherwise wreck the comparison:

  ONE CROP BOX FOR ALL FIVE.  A per-asset union box grows with whatever that asset did, so
  a measured "object offset" could just be a framing difference. Framing is held identical
  across assets so the offset is about the object.

  RAMP ANGLES BELOW THE TOPPLE ANGLE.  The vase topples at 34 deg, so a 36 deg ramp would
  make it fall over rather than slide and the probe would read a different event. 21/24/27
  is under every asset's topple angle. That in turn caps mu: above ~0.6 nothing moves at
  those angles, so the sweep stops there rather than spending clips where the bowl's
  high-mu band already measured 61% error.

Assets are scaled to a common 0.19 m horizontal extent so none of them is reading size.
Mass is identical across all five -- slide distance is mass-independent, which is the whole
reason the collide probe was retired, and holding it fixed keeps that true by construction.

    W scripts/o11_assets.py sim
    ... blender per asset lab ...
    W scripts/o11_assets.py clip
    CUDA_VISIBLE_DEVICES=$G C scripts/o4_joint.py score --lab outputs/scene/a_<tag>
    W scripts/o11_assets.py fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

TARGET_W = 0.19                      # every asset scaled to this horizontal extent
TILTS = (21.0, 24.0, 27.0)
MUS = np.geomspace(0.10, 0.60, 9)
MASS = 0.40
# Nine assets, not five. With four valid assets and six candidate geometric features, an
# exact permutation test says SOME feature reaches |rho| = 1.000 by chance 25% of the time,
# so the footprint result from the first five was not evidence. Six assets puts that near 2%
# and eight near zero. The vase stays in the list but is excluded from the friction analysis
# by its own topple count -- it reads a topple, not a slide.
ASSETS = [("bowl", "rigid/wooden_bowl_01"), ("pot", "rigid/brass_pot_01"),
          ("vase", "rigid/ceramic_vase_01"), ("teapot", "rigid/Threshold_Porcelain_Teapot_White"),
          ("sorter", "rigid/Poppin_File_Sorter_Blue"),
          ("box", "rigid/cardboard_box_01"), ("books", "rigid/book_encyclopedia_set_01"),
          ("lion", "rigid/Schleich_Lion_Action_Figure"),
          ("shark", "rigid/Weisshai_Great_White_Shark"),
          ("dino", "rigid/Great_Dinos_Triceratops_Toy")]
NEW = ("box", "books", "lion", "shark", "dino")
OUT = Path("docs/results/o11_assets.json")
BOX = Path("outputs/scene/_o11_box.json")


def scale_for(asset):
    from src.data.assets import load_asset
    cat, _, nm = asset.rpartition("/")
    e = load_asset(cat, nm).extents
    return TARGET_W / float(max(e[0], e[1]))


def lab_of(tag):
    return Path(f"outputs/scene/a_{tag}")


def do_sim(args):
    import warp as wp
    import v1_killtest as V
    from src.sim.tilt_probe import ramp_gravity_seq
    wp.init()
    only = os.environ.get("ONLY")
    allmeta = {}
    with wp.ScopedDevice("cuda:0"):
        for tag, asset in ASSETS:
            if only and tag not in only.split(","):
                continue
            sc = scale_for(asset)
            V.OBJ, V.SCALE = asset, sc
            lab = lab_of(tag); lab.mkdir(parents=True, exist_ok=True)
            print(f"\n{tag}: {asset}  scale {sc:.3f}", flush=True)
            scenes, meta, spec = {}, {}, []
            for t in TILTS:
                g = ramp_gravity_seq(0.0, t, V.NF, V.SUBSTEPS, settle_frames=6)
                seq = list(np.linspace(0.0, t, V.NF))
                for mu in MUS:
                    r = V.track(V.build(mu=float(mu), mass=MASS,
                                        pos=(V.TILT_X0, 0.0, V.GZ + 0.002), gseq=g),
                                tilt_seq=seq)
                    if r is None:
                        print(f"   t{t:.0f} mu={mu:.3f} DIVERGED"); continue
                    # did it TOPPLE instead of sliding? the probe must read one event
                    # The sim tilts GRAVITY and leaves the plane flat, so the ramp normal
                    # in these coordinates is (0,0,1) -- not a rotated one. Against a
                    # rotated normal every sliding bowl flagged as toppling.
                    tipped = any(
                        np.degrees(np.arccos(np.clip(
                            np.asarray(p["mat_flat"], float)[2, 2], -1, 1))) > 30.0
                        for p in r[0])
                    key = f"{tag}_t{int(t):02d}_mu{mu:05.3f}"
                    scenes[key] = {"tilt_deg": seq, "objects": {"subject": r[0]}}
                    meta[key] = r[1]
                    dx = 100 * (r[0][-1]["flat"][0] - r[0][0]["flat"][0])
                    spec.append({"key": key, "probe": "tilt", "param": "mu",
                                 "value": float(mu), "ref": 0.0, "tilt": float(t),
                                 "slide_cm": dx, "tipped": bool(tipped),
                                 "asset": tag, "scale": sc})
                    print(f"   t{t:.0f} mu={mu:5.3f} slid {dx:7.2f} cm"
                          f"{'  TOPPLED' if tipped else ''}", flush=True)
            (lab / "scene_poses.json").write_text(json.dumps(scenes))
            (lab / "scene.json").write_text(json.dumps(
                {"ground_z": V.GZ, "objects": {"subject": {
                    "asset": asset, "scale": sc, "rot_z": 0.0,
                    "pos": [0.0, 0.0, V.GZ + 0.30]}}}))
            (lab / "meta.json").write_text(json.dumps(meta))
            (lab / "spec.json").write_text(json.dumps(spec, indent=1))
            allmeta.update({f"{tag}:{k}": v for k, v in meta.items()})
            print(f"  wrote {len(scenes)} scenes to {lab}")
    if not only:
        _choose_box(allmeta)
    return 0


def _choose_box(allmeta):
    """ONE box over every asset, at the tightest pull that contains all of them."""
    import o2_boundary as O
    from src.render.crop import project_points
    from src.render.views import VIEWS
    from t0_tracking import pulled_camera
    W, H = VIEWS["a"]["width"], VIEWS["a"]["height"]
    best = None
    for pull in (1.20, 1.60, 2.00, 2.50, 3.00, 3.60):
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
        print("  NO pull contains every clip -- widen or drop the worst asset")
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
    out = {}
    for tag, _ in ASSETS:
        lab = lab_of(tag)
        if not (lab / "spec.json").exists():
            continue
        con = O.containment(lab, pull=json.loads(BOX.read_text())["pull"])
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


def _geom(tag):
    """Cheap geometric features, the candidates for predicting an asset's offset."""
    from src.data.assets import load_asset
    asset = dict(ASSETS)[tag]
    cat, _, nm = asset.rpartition("/")
    m = load_asset(cat, nm)
    V = np.asarray(m.vertices)
    e = np.asarray(m.extents, float)
    sc = scale_for(asset)
    w = float(max(e[0], e[1])) * sc
    h = float(e[2]) * sc
    zc = float(V.mean(0)[2] - V[:, 2].min()) * sc
    return {"width": w, "height": h, "aspect": h / w, "com_z": zc,
            "footprint": float(e[0] * e[1]) * sc * sc,
            "topple_deg": float(np.degrees(np.arctan((w / 2) / max(zc, 1e-9))))}


def do_fit(args):
    import o4_joint as O
    R = _rows()
    print(f"{len(R)} assets scored: " + ", ".join(f"{k} ({len(v)})" for k, v in R.items()))
    valid = [t for t in R if sum(r["tipped"] for r in R[t]) <= len(R[t]) // 3]
    print(f"friction probe valid (topples on <= 1/3 of clips): {valid}")
    print()

    X = {t: np.log(np.array([r["p"] for r in R[t]]) + 1e-8) for t in R}
    Y = {t: np.log10(np.maximum([abs(r["slide_cm"]) for r in R[t]], 1e-2)) for t in R}

    print("PER-ASSET READOUT -- each calibrated on its own sweep, leave-one-out")
    print(f"  {'asset':<8}{'n':>4}{'LOO sd':>9}{'factor':>8}{'rho(read,true)':>16}")
    own = {}
    for t in R:
        pr = np.empty(len(Y[t]))
        for j in range(len(Y[t])):
            k = np.ones(len(Y[t]), bool); k[j] = False
            pr[j] = O.ridge(X[t][k], Y[t][k])(X[t][j][None])[0]
        sd = float((Y[t] - pr).std())
        own[t] = sd
        print(f"  {t:<8}{len(Y[t]):>4}{sd:>9.3f}{10**sd:>8.2f}{_sp(pr, Y[t]):>16.3f}")

    print()
    print("TRANSFER -- readout from one asset applied to another (offset = the scale error)")
    print(f"  {'from -> to':<20}{'bias (dec)':>12}{'factor':>8}{'scatter':>9}"
          f"{'R2 raw':>9}{'R2 -offset':>12}{'rho':>7}")
    off = {}
    for a in valid:
        for b in valid:
            if a == b:
                continue
            pred = O.ridge(X[a], Y[a])(X[b])
            r = Y[b] - pred
            sst = ((Y[b] - Y[b].mean()) ** 2).sum()
            off[(a, b)] = float(r.mean())
            print(f"  {a+' -> '+b:<20}{r.mean():>+12.3f}{10**abs(r.mean()):>8.2f}"
                  f"{r.std():>9.3f}{1-(r**2).sum()/sst:>9.2f}"
                  f"{1-((r-r.mean())**2).sum()/sst:>12.2f}{_sp(pred, Y[b]):>7.3f}")

    print()
    print("IS THE OFFSET PREDICTABLE FROM GEOMETRY?")
    print("  If it is, a new asset needs a handful of clips instead of a full sweep.")
    G = {t: _geom(t) for t in valid}
    # per-asset level: the mean log slide the readout would assign, vs its own truth
    lvl = {t: float(np.mean(Y[t])) for t in valid}
    feats = ["width", "height", "aspect", "com_z", "footprint", "topple_deg"]
    print(f"  {'feature':<12}{'rho with asset level':>22}")
    for f in feats:
        print(f"  {f:<12}{_sp([G[t][f] for t in valid], [lvl[t] for t in valid]):>22.3f}")
    print()
    print(f"  pairwise offsets span {min(abs(v) for v in off.values()):.3f} to "
          f"{max(abs(v) for v in off.values()):.3f} decades "
          f"({10**min(abs(v) for v in off.values()):.2f}x to "
          f"{10**max(abs(v) for v in off.values()):.2f}x)")
    OUT.write_text(json.dumps(
        {"own_sd": own, "valid": valid, "geom": G, "level": lvl,
         "offsets": {f"{a}->{b}": v for (a, b), v in off.items()},
         "n": {t: len(R[t]) for t in R},
         "tipped": {t: int(sum(r["tipped"] for r in R[t])) for t in R}},
        indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.set_defaults(f=do_sim)
    s = sub.add_parser("clip"); s.set_defaults(f=do_clip)
    s = sub.add_parser("fit"); s.set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
