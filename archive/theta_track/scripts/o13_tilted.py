"""RETIRED 2026-10-06 -- kept for the record, not for use.

The friction READOUT this script produced (rendered clip -> V-JEPA SSv2 classifier -> slide,
inverted through the simulator's own slide(mu) curve; reported 2-8% held-out mu) is
retired:
  * it was calibrated against the simulator, so it can only ever say what the simulator
    did -- never what a real material does;
  * the simulator it was calibrated on has since been corrected (2 mm seating drop,
    vertex-mean centre of mass), which moves its slides by 28-63% (scripts/p6_recheck.py),
    so the 2-8% describes a superseded simulator;
  * friction now comes from REAL footage instead: Physics 101 ramp clips, tracked and fitted
    through Newton itself, which reproduces the real slides to 0.4-1.0 mm
    (scripts/p1_ramp_friction.py, p2_ramp_all.py, p4_sim_loop.py, p5_transfer.py).

Friction on a TILTED ramp, all assets rendered together in one grid.

Two changes from o11_assets, both decided from measurements in this project:

  TILTED, NOT TILTING.  The plane is now really inclined (MeshProbeScene(tilt_deg=)) rather
  than level with gravity rotated. A tilting ramp carries a rate dimension that is a 9-11x
  lever on the outcome at fixed mu (slide 374 -> 41 cm at mu = 0.15), splits the simulator
  and render frames -- the source of four wrong-frame bugs, the last of which rendered
  objects floating free of the ramp -- and omits the rotating-frame terms a real rotating
  table has. A fixed tilt has none of that, and gives 57x of slide range over a 4.4x mu
  sweep against 13-22x for the tilting ramp.

  ONE RENDER FOR ALL ASSETS.  Per frame, ~1.43 s is fixed work and ~0.58 s scales with
  pixels, of which only ~0.16 s is sampling -- so halving samples buys 6% and the lever is
  fewer passes. All assets are composed into one frame as cells along the camera's right
  vector (src/render/grid.py) and cut apart afterwards. Measured 1.71x at 4 cells.
  A cell is pinned to an ASSET, never to a theta, so its fixed geometry is absorbed by that
  asset's own calibration.

MU GRID IS PER TILT, centred on that tilt's own threshold mu* = tan^2(tilt)/mu_table. On a
tilted plane the object slides for the whole clip, so a mu far below mu* travels metres and
leaves a 1.58 m table: at 24 deg, mu/mu* = 0.5 slides 696 cm while 1.0 slides 72 cm. The
informative region is around the threshold and that is where the samples go. Tilt is a
condition we set, so a per-tilt grid is known, not a confound.

    W scripts/o13_tilted.py sim
    ... blender on outputs/scene/t_grid, RENDER_SCALE_X=<cells>, CAM_PULL from the box file
    W scripts/o13_tilted.py clip
    CUDA_VISIBLE_DEVICES=$G C scripts/o4_joint.py score --lab outputs/scene/t_<asset>
    W scripts/o13_tilted.py fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

GZ = 0.706
FPS, NF, SUBSTEPS = 24.0, 64, 420
PEN_FRAC = 1.0e-4
TABLE = (0.79, 0.49)
TARGET_W = 0.19
MASS = 0.40
TABLE_MU = 0.45
# FIVE tilts, each covering a ~1.7x window of mu around its own threshold, chosen so the
# windows tile mu in [0.15, 1.2]:  mu* = 0.17 / 0.27 / 0.45 / 0.74 / 1.23.
# The window is narrow and that is the tilted probe's real cost. Below f ~ 0.85 the object
# slides off a 1.58 m table (412 cm at 30 deg, f = 0.80); above f ~ 1.5 the slide saturates
# at the creep floor (24 deg gave 15.0 / 15.2 / 15.4 / 12.2 / 18.7 cm for f = 1.4...3.0,
# i.e. noise) or the object grips and CLIMBS ONTO ITS RIM -- at 30 deg, f = 2.48 its normal
# gap rose 34 -> 139 mm and the 57 cm of "slide" was a tumble, identically at 420 and 1260
# substeps, so physical rather than numerical.
TILTS = (15.0, 19.0, 24.0, 30.0, 37.0)
MU_FACTORS = np.geomspace(0.85, 1.55, 6)      # of mu* at that tilt
START_ALONG = -0.34                            # metres uphill of the pivot
PITCH = 0.42                                   # cell spacing, metres
ASSETS = [("bowl", "rigid/wooden_bowl_01"), ("pot", "rigid/brass_pot_01"),
          ("teapot", "rigid/Threshold_Porcelain_Teapot_White"),
          ("sorter", "rigid/Poppin_File_Sorter_Blue"),
          ("box", "rigid/cardboard_box_01"),
          ("books", "rigid/book_encyclopedia_set_01"),
          ("shark", "rigid/Weisshai_Great_White_Shark")]
GRID = Path("outputs/scene/t_grid")
BOXES = Path("outputs/scene/_t_boxes.json")
OUT = Path("docs/results/o13_tilted.json")


def mu_star(tilt):
    return float(np.tan(np.radians(tilt)) ** 2 / TABLE_MU)


def scale_for(asset):
    from src.data.assets import load_asset
    cat, _, nm = asset.rpartition("/")
    return TARGET_W / float(max(load_asset(cat, nm).extents[:2]))


def cond_key(tilt, f):
    return f"t{int(tilt):02d}_f{f:05.3f}"


def do_sim(args):
    import warp as wp
    from src.sim.mesh_scene import MeshProbeScene
    from src.render.grid import cell_offsets
    wp.init()
    tags = [t for t, _ in ASSETS]
    offs = cell_offsets(len(tags), PITCH, view="a")
    GRID.mkdir(parents=True, exist_ok=True)
    scenes, gobjs, solo = {}, {}, {}
    per = {t: {"spec": [], "meta": {}} for t in tags}
    with wp.ScopedDevice("cuda:0"):
        for (tag, asset), off in zip(ASSETS, offs):
            sc = scale_for(asset)
            gobjs[tag] = {"asset": asset, "scale": sc, "rot_z": 0.0,
                          "pos": [float(off[0]), float(off[1]), GZ + 0.30]}
            print(f"\n{tag}: {asset} scale {sc:.3f}", flush=True)
            for tilt in TILTS:
                ms = mu_star(tilt)
                for f in MU_FACTORS:
                    mu = float(ms * f)
                    s = MeshProbeScene([asset], [[0., 0., 0.]], [[0., 0., 0.]],
                                       masses=[MASS], ground_z=GZ,
                                       dt=1.0 / (FPS * SUBSTEPS), n_steps=NF * SUBSTEPS,
                                       k=2500.0, cd=[3200.0], mu=[mu], mesh_scale=[sc],
                                       faces=900, ground_mu=TABLE_MU, ground_cd=3000.0,
                                       table=TABLE, tilt_deg=tilt)
                    s.calibrate_stiffness(pen_frac=PEN_FRAC)
                    s.seat_on_plane(along=START_ALONG)
                    s.rollout()
                    P, Q = s.positions(SUBSTEPS), s.rotations(SUBSTEPS)
                    if not (np.isfinite(P).all() and np.isfinite(Q).all()
                            and np.abs(P).max() < 50):
                        print(f"   t{tilt:.0f} f{f:.2f} DIVERGED"); continue
                    n, tv = s._gn, s._gt
                    slide = float(np.dot(P[-1, 0] - P[0, 0], tv)) * 100.0
                    # did it leave the table? then the magnitude is truncated, not measured
                    d = P[:, 0] - s._gp
                    off_table = bool((np.abs(d @ tv) > TABLE[0]).any())
                    # LIFTED / TUMBLED: the body climbing off the plane means the observable
                    # is no longer a slide. Both are flagged here and dropped at fit time
                    # rather than silently averaged in.
                    gap = d @ n
                    lifted = bool((gap - gap[0]).max() > 0.020)
                    zang = np.degrees(np.arccos(np.clip(
                        [float(quat_to_mat(q)[:, 2] @ n) for q in Q[:, 0]], -1, 1)))
                    tumbled = bool((zang > 25.0).any())
                    ck = cond_key(tilt, f)
                    V = s.local.numpy().astype(float)
                    idx = np.linspace(0, len(V) - 1, min(48, len(V))).astype(int)
                    seq, pts = [], []
                    for t2 in range(len(P)):
                        R = quat_to_mat(Q[t2, 0])
                        loc = P[t2, 0] - R @ s.coms[0]
                        seq.append({"loc": [float(v) for v in (loc + off)],
                                    "mat": [[float(v) for v in r] for r in R]})
                        pts.append(((R @ V[idx].T).T + P[t2, 0] + off).tolist())
                    scenes.setdefault(ck, {"tilt_deg": tilt, "objects": {}})
                    scenes[ck]["objects"][tag] = seq
                    # ALSO write the asset on its own, with the cell offset removed. The
                    # grid saves render passes only while the cells fit the frame: seven in
                    # a row need the camera ~3 m back, where one cell is 14% of the width,
                    # so matching single-cell resolution would cost 8.6 Mpx against 1.7 Mpx
                    # rendered separately. Per-asset is the fallback and keeps the
                    # appearance the readouts were built on.
                    solo.setdefault(tag, {})[ck] = {
                        "tilt_deg": tilt,
                        "objects": {"subject": [
                            {"loc": [float(q["loc"][i] - off[i]) for i in range(3)],
                             "mat": q["mat"]} for q in seq]}}
                    per[tag]["meta"][ck] = [
                        [[float(v[i] - off[i]) for i in range(3)] for v in fr] for fr in pts]
                    per[tag]["grid_meta"] = per[tag].get("grid_meta", {})
                    per[tag]["grid_meta"][ck] = pts
                    per[tag]["spec"].append(
                        {"key": ck, "probe": "tilt_fixed", "param": "mu", "value": mu,
                         "ref": ms, "tilt": tilt, "factor": float(f), "slide_cm": slide,
                         "off_table": off_table, "lifted": lifted, "tumbled": tumbled,
                         "asset": tag, "scale": sc})
                    bad = ("  OFF TABLE" if off_table else "") + \
                          ("  LIFTED" if lifted else "") + ("  TUMBLED" if tumbled else "")
                    print(f"   t{tilt:.0f} mu={mu:6.3f} (f={f:.2f}) slide {slide:8.2f} cm"
                          f"{bad}", flush=True)
    (GRID / "scene_poses.json").write_text(json.dumps(scenes))
    (GRID / "scene.json").write_text(json.dumps({"ground_z": GZ, "objects": gobjs}))
    for tag in tags:
        d = Path(f"outputs/scene/t_{tag}"); d.mkdir(parents=True, exist_ok=True)
        (d / "spec.json").write_text(json.dumps(per[tag]["spec"], indent=1))
        (d / "meta.json").write_text(json.dumps(per[tag]["meta"]))
        (d / "scene_poses.json").write_text(json.dumps(solo.get(tag, {})))
        a = dict(gobjs[tag]); a["pos"] = [0.0, 0.0, GZ + 0.30]
        (d / "scene.json").write_text(json.dumps(
            {"ground_z": GZ, "objects": {"subject": a}}))
    print(f"\nwrote {len(scenes)} grid scenes x {len(tags)} cells to {GRID}")
    _choose_boxes()
    return 0


def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _choose_boxes():
    """One crop box PER ASSET, over all that asset's conditions, at a shared camera pull."""
    import o2_boundary as O
    from src.render.crop import project_points
    from src.render.views import VIEWS
    from t0_tracking import pulled_camera
    W0, H = VIEWS["a"]["width"], VIEWS["a"]["height"]
    ncell = len(ASSETS)
    for pull in (2.20, 2.80, 3.40, 4.20):
        cam = pulled_camera(pull, view="a")
        W = W0 * ncell                    # RENDER_SCALE_X widens the frame
        boxes, ok_all = {}, True
        for tag, _ in ASSETS:
            meta = json.loads(Path(f"outputs/scene/t_{tag}/meta.json").read_text())
            # ONLY THE CLEAN CONDITIONS SET THE BOX. Computing it over every condition let
            # the off-table clips -- 400 cm of travel, dropped at fit time anyway -- size a
            # box that then could not contain anything: every asset came out at the 448 px
            # cap with 0-44% containment. The box must cover what is actually measured.
            spec = json.loads(Path(f"outputs/scene/t_{tag}/spec.json").read_text())
            clean = {r["key"] for r in spec
                     if not (r["off_table"] or r["lifted"] or r["tumbled"])}
            meta = {k: v for k, v in meta.items() if k in clean}
            if not meta:
                print(f"  {tag}: no clean conditions"); continue
            b = O.union_box(cam, meta, W, H, pad=0.12)
            worst = 1.0
            for v in meta.values():
                uv, o = project_points(cam, np.asarray(v, float).reshape(-1, 3))
                uv = uv[o]
                worst = min(worst, float(((uv[:, 0] >= b[0]) & (uv[:, 0] <= b[2]) &
                                          (uv[:, 1] >= b[1]) & (uv[:, 1] <= b[3])).mean()))
            boxes[tag] = {"box": list(b), "worst": worst}
            if worst < 0.99:
                ok_all = False
        sizes = [b["box"][2] - b["box"][0] for b in boxes.values()]
        print(f"  pull {pull:.2f}: boxes {min(sizes)}-{max(sizes)}px, "
              f"worst containment {100*min(b['worst'] for b in boxes.values()):.1f}%")
        if ok_all:
            BOXES.write_text(json.dumps({"pull": pull, "cells": ncell,
                                         "width": W, "height": H, "boxes": boxes}))
            print(f"  chose pull {pull:.2f} -> {BOXES}")
            return
    print("  no pull contains every asset's trajectory; widen PITCH or shorten the slide")


SOLO = Path("outputs/scene/_t_solo.json")


def do_clip(args):
    """Cut each asset's own frames with ONE box per asset, over its clean conditions.

    The grid path is kept in src/render/grid.py but is not used here: seven cells in a row
    need the camera far enough back that a cell is 14% of the frame width, so matching
    single-cell resolution costs more pixels than rendering separately. Per-asset keeps the
    appearance the readouts were built on.
    """
    import imageio.v2 as imageio
    from src.render.crop import crop_clip
    cfg = json.loads(SOLO.read_text())
    for tag, _ in ASSETS:
        if tag not in cfg:
            continue
        lab = Path(f"outputs/scene/t_{tag}")
        box = tuple(cfg[tag]["box"])
        spec = json.loads((lab / "spec.json").read_text())
        dst = lab / "clips"; dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for r in spec:
            ps = sorted((lab / f"sim_{r['key']}@a").glob("f*.png"))[:NF]
            if len(ps) < NF:
                continue
            frames = np.stack([imageio.imread(q)[..., :3] for q in ps])
            w = imageio.get_writer(str(dst / f"{r['key']}.mp4"), fps=int(FPS),
                                   codec="libx264", quality=9, macro_block_size=1)
            for fr in crop_clip(frames, box):
                w.append_data(fr)
            w.close()
            n += 1
        print(f"  {tag}: {n} clips, box {box} ({box[2]-box[0]}px), "
              f"pull {cfg[tag]['pull']}", flush=True)
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
    cfg = json.loads(SOLO.read_text())
    probs = {}
    for f in glob.glob("outputs/outcome/cache/*.json"):
        d = json.load(open(f))
        probs[str(Path(d["clip"]).resolve())] = np.asarray(d["probs"], np.float64)
    out = {}
    for tag, _ in ASSETS:
        lab = Path(f"outputs/scene/t_{tag}")
        if tag not in cfg or not (lab / "spec.json").exists():
            continue
        con = O.containment(lab, pull=cfg[tag]["pull"])
        rr = []
        for r in json.loads((lab / "spec.json").read_text()):
            # a clip that left the table, lifted off the plane or tumbled is not a slide
            if r["off_table"] or r["lifted"] or r["tumbled"]:
                continue
            c = (lab / "clips" / f"{r['key']}.mp4").resolve()
            p2 = probs.get(str(c))
            if p2 is None or con.get(r["key"], 0) < O.MIN_CONTAIN:
                continue
            rr.append({**r, "p": p2})
        if rr:
            out[tag] = rr
    return out


def do_fit(args):
    import o4_joint as O
    R = _rows()
    if not R:
        sys.exit("no scored clean clips -- run the score stage first")
    print(f"{len(R)} assets: " + ", ".join(f"{k} ({len(v)})" for k, v in R.items()))
    res = {"per_asset": {}, "holdout": {}}
    X = {t: np.log(np.array([r["p"] for r in R[t]]) + 1e-8) for t in R}
    Y = {t: np.log10(np.maximum([abs(r["slide_cm"]) for r in R[t]], 1e-2)) for t in R}

    print()
    print("READOUT -- each asset on its own clean clips, leave-one-out")
    print(f"  {'asset':<8}{'n':>4}{'tilts':>7}{'LOO sd':>9}{'factor':>8}{'rho':>8}")
    for t in R:
        pr = np.empty(len(Y[t]))
        for j in range(len(Y[t])):
            k = np.ones(len(Y[t]), bool); k[j] = False
            pr[j] = O.ridge(X[t][k], Y[t][k])(X[t][j][None])[0]
        sd = float((Y[t] - pr).std())
        nt = len({r["tilt"] for r in R[t]})
        res["per_asset"][t] = {"n": len(Y[t]), "tilts": nt, "loo_sd": sd,
                               "rho": _sp(pr, Y[t])}
        print(f"  {t:<8}{len(Y[t]):>4}{nt:>7}{sd:>9.3f}{10**sd:>8.2f}{_sp(pr, Y[t]):>8.3f}")

    print()
    print("RECOVERING mu -- hold out a whole mu, refit, read it back")
    print("  one tilt at a time against all tilts joined")
    for t in R:
        fwd = {}
        for tl in sorted({r["tilt"] for r in R[t]}):
            g = sorted([r for r in R[t] if r["tilt"] == tl], key=lambda r: r["value"])
            if len(g) < 3:
                continue
            mu = np.log10([r["value"] for r in g])
            y = np.log10(np.maximum([abs(r["slide_cm"]) for r in g], 1e-2))
            keep = [0]
            for i in range(1, len(y)):
                if y[i] < y[keep[-1]]:
                    keep.append(i)
            if len(keep) >= 3:
                fwd[tl] = (mu[keep], y[keep])
        if not fwd:
            continue
        e1, ej = [], []
        for mu0 in sorted({r["value"] for r in R[t]}):
            hold = [i for i, r in enumerate(R[t]) if r["value"] == mu0]
            keep = [i for i, r in enumerate(R[t]) if r["value"] != mu0]
            if not hold or len(keep) < 8:
                continue
            rd = O.ridge(X[t][keep], Y[t][keep])
            hats = []
            for i in hold:
                tl = R[t][i]["tilt"]
                if tl not in fwd:
                    continue
                mm, yy = fwd[tl]
                o = np.argsort(yy)
                hats.append(float(10 ** np.interp(rd(X[t][i][None])[0], yy[o], mm[o])))
            if not hats:
                continue
            e1 += [abs(np.log10(h / mu0)) for h in hats]
            ej.append(abs(np.log10(float(10 ** np.mean(np.log10(hats))) / mu0)))
        if ej:
            res["holdout"][t] = {"single_pct": float(100 * (10 ** np.median(e1) - 1)),
                                 "joint_pct": float(100 * (10 ** np.median(ej) - 1)),
                                 "n": len(ej)}
            print(f"  {t:<8} one tilt {100*(10**np.median(e1)-1):>5.0f}%   "
                  f"tilts joined {100*(10**np.median(ej)-1):>5.0f}%   (n={len(ej)})")
    print(f"  {'prior':<8} flat over the swept mu range, ~{100*(np.sqrt(8)-1):.0f}%")
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.set_defaults(f=do_sim)
    s = sub.add_parser("clip"); s.set_defaults(f=do_clip)
    s = sub.add_parser("boxes"); s.set_defaults(f=lambda a: _choose_boxes())
    s = sub.add_parser("fit"); s.set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
