"""F4 -- fit mu to the motion prior's TRAJECTORY, jointly over tilts, per asset.

The prior (Flow Poke Transformer) states where each object point should be one second
later; the simulator is inverted to reach that. The prior does not have to respond to mu --
it only has to say what the motion should be. Two questions, both answerable without
ground truth:

  1. JOINT FIT.  One mu per asset, fitted to every tilt at once, with the disagreement
     between tilts reported. Fitting each tilt alone and comparing (f2/earlier) asks a
     single-condition question; a material property must be one number across conditions.

  2. DOES THE PRIOR KNOW THE OBJECT?  The earlier target was ~24-32 px whatever the tilt.
     If it is also ~the same for every asset, the fitted mu is just "barely slides here" --
     a prior we could write down for free. Tested against two nulls through each asset's
     own simulator:
       const      one target for every asset and tilt (no information at all)
       tilt-only  the other assets' target at the same tilt (keeps whatever FPT knows
                  about the ramp, removes what it knows about THIS object)
     If mu_FPT == mu_tilt-only for every asset, the object-specific part of the prediction
     contributes nothing.

Design, all fixed in advance:
  * common mu grid for all tilts (the o13 grid is per tilt and the windows do not overlap,
    so a joint fit there would be extrapolation). Sim only -- no clip renders.
  * every rollout is kept: off-table, lifted and tumbled are physical outcomes, and a
    trajectory comparison scores them like any other. (Those flags exist for the slide
    READOUT, which measures a slide.) Only divergence is dropped.
  * ONE camera pull for every asset, so a pixel means the same thing across assets and the
    tilt-only null is not comparing different zooms.
  * 6 deg orbit laid out over 64 frames, camera_static=False, background poke MEASURED from
    frames 0 and 24 in both axes (the earlier run set x and y of the poke to the same value).
  * horizon frame 0 -> 24 (1 s at 24 fps).

    W scripts/f4_fpt_joint.py sim
    blender per asset: LAB=outputs/scene/f4_<tag> VIEWS=a CAM_PULL=2.0 CAM_ORBIT=6 \
        ORBIT_NF=64 FRAMES=0,24
    C scripts/f4_fpt_joint.py score
    W scripts/f4_fpt_joint.py fit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "models/fpt"))

import o13_tilted as T

NFW = 25                       # frames simulated: 0..24
T1 = 24                        # horizon, frames
ORBIT, ORBIT_NF, PULL = 6.0, 64, 2.0
MU = np.geomspace(0.08, 2.5, 18)
N_Q = 16                       # query points per object
OUT = Path("docs/results/f4_fpt_joint.json")

# RAMP MODE (F4_RAMP=1): the plane is a PLANK lying on the level table -- low end on the
# tabletop, high end propped on a block -- instead of a tilted table, which reads as a table
# falling over. Only the plane's position changes (pivot, extent); at a given tilt and mu
# the physics is translation-invariant. Labs go to f6_<tag>.
import os
RAMP = os.environ.get("F4_RAMP") == "1"
PLANK_HALF = (0.50, 0.20)            # 1.0 m x 0.4 m board, fits the 1.13 x 0.71 m tabletop
PLANK_THICK = 0.025
PLANK_START = -0.30                  # 0.20 m below the high end


def plank_pivot(tilt):
    """Plane centre such that the plank's low end rests on the tabletop."""
    return np.array([0.0, 0.0, T.GZ + PLANK_THICK + PLANK_HALF[0] * np.sin(np.radians(tilt))])


def lab(tag):
    return Path(f"outputs/scene/{'f6' if RAMP else 'f4'}_{tag}")


def mkey(tilt, mu):
    return f"t{int(tilt):02d}_m{mu:06.4f}"


def tkey(tilt):
    return f"t{int(tilt):02d}"


def do_sim(args):
    import warp as wp
    from src.sim.mesh_scene import MeshProbeScene
    wp.init()
    only = set(args.assets.split(",")) if args.assets else None
    with wp.ScopedDevice("cuda:0"):
        for tag, asset in T.ASSETS:
            if only and tag not in only:
                continue
            sc = T.scale_for(asset)
            d = lab(tag); d.mkdir(parents=True, exist_ok=True)
            spec, meta, poses = [], {}, {}
            print(f"\n{tag}: {asset} scale {sc:.3f}", flush=True)
            for tilt in T.TILTS:
                for mu in MU:
                    s = MeshProbeScene([asset], [[0., 0., 0.]], [[0., 0., 0.]],
                                       masses=[T.MASS], ground_z=T.GZ,
                                       dt=1.0 / (T.FPS * T.SUBSTEPS),
                                       n_steps=NFW * T.SUBSTEPS,
                                       k=2500.0, cd=[3200.0], mu=[float(mu)],
                                       mesh_scale=[sc], faces=900, ground_mu=T.TABLE_MU,
                                       ground_cd=3000.0,
                                       table=PLANK_HALF if RAMP else T.TABLE, tilt_deg=tilt,
                                       pivot=plank_pivot(tilt) if RAMP else None)
                    s.calibrate_stiffness(pen_frac=T.PEN_FRAC)
                    s.seat_on_plane(along=PLANK_START if RAMP else T.START_ALONG)
                    s.rollout()
                    P, Q = s.positions(T.SUBSTEPS), s.rotations(T.SUBSTEPS)
                    # rule 3: a spatial bound, not isfinite alone
                    if not (np.isfinite(P).all() and np.isfinite(Q).all()
                            and np.abs(P).max() < 50):
                        print(f"   t{tilt:.0f} mu={mu:.3f} DIVERGED"); continue
                    n, tv = s._gn, s._gt
                    dd = P[:, 0] - s._gp
                    gap = dd @ n
                    zang = np.degrees(np.arccos(np.clip(
                        [float(T.quat_to_mat(q)[:, 2] @ n) for q in Q[:, 0]], -1, 1)))
                    V = s.local.numpy().astype(float)
                    idx = np.linspace(0, len(V) - 1, min(48, len(V))).astype(int)
                    pts, seq = [], []
                    for t2 in range(len(P)):
                        R = T.quat_to_mat(Q[t2, 0])
                        pts.append(((R @ V[idx].T).T + P[t2, 0]).tolist())
                        loc = P[t2, 0] - R @ s.coms[0]
                        seq.append({"loc": [float(v) for v in loc],
                                    "mat": [[float(v) for v in r] for r in R]})
                    k = mkey(tilt, mu)
                    meta[k] = pts
                    row = {"key": k, "tilt": tilt, "mu": float(mu),
                           "slide_cm": float(np.dot(P[-1, 0] - P[0, 0], tv)) * 100.0,
                           "off_table": bool((np.abs(dd @ tv) >
                                              (PLANK_HALF if RAMP else T.TABLE)[0]).any()),
                           "lifted": bool((gap - gap[0]).max() > 0.020),
                           "tumbled": bool((zang > 25.0).any())}
                    spec.append(row)
                    # one render per TILT: frame 0 is identical for every mu (the body is
                    # seated at rest at the same place), so keep the least-moving pose set
                    poses[tkey(tilt)] = {"tilt_deg": tilt, "objects": {"subject": seq}}
                    if RAMP:
                        poses[tkey(tilt)]["ramp"] = {"pivot": plank_pivot(tilt).tolist(),
                                                     "half": list(PLANK_HALF),
                                                     "thick": PLANK_THICK}
                    fl = "".join(f"  {x.upper()}" for x in ("off_table", "lifted", "tumbled")
                                 if row[x])
                    print(f"   t{tilt:.0f} mu={mu:6.3f} slide {row['slide_cm']:8.2f} cm{fl}",
                          flush=True)
            (d / "spec.json").write_text(json.dumps(spec, indent=1))
            (d / "meta.json").write_text(json.dumps(meta))
            (d / "scene_poses.json").write_text(json.dumps(poses))
            (d / "scene.json").write_text(json.dumps(
                {"ground_z": T.GZ, "objects": {"subject": {
                    "asset": asset, "scale": sc, "rot_z": 0.0,
                    "pos": [0.0, 0.0, T.GZ + 0.30]}}}))
    return 0


# --------------------------------------------------------------------------- camera

def cam_at(t):
    """The camera the renderer used at frame t -- same formula as CAM_ORBIT/ORBIT_NF."""
    from t0_tracking import PinholeCam
    from src.render.views import VIEWS
    vc = VIEWS["a"]
    eye, tgt = np.asarray(vc["eye"], float), np.asarray(vc["target"], float)
    eye = tgt + (eye - tgt) * PULL
    a = np.radians(ORBIT) * (t / (ORBIT_NF - 1) - 0.5)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return PinholeCam(tgt + R @ (eye - tgt), tgt, vc["fov_deg"], vc["width"], vc["height"])


def wh():
    from src.render.views import VIEWS
    return np.array([VIEWS["a"]["width"], VIEWS["a"]["height"]], float)


def query_idx(pts0):
    """N_Q in-frame vertex indices at frame 0. The same for every mu of one tilt."""
    from src.render.crop import project_points
    uv, ok = project_points(cam_at(0), np.asarray(pts0, float))
    cand = np.flatnonzero(ok)
    return cand[np.linspace(0, len(cand) - 1, min(N_Q, len(cand))).astype(int)]


def sim_flow(pts0, pts1, idx):
    """Image-space displacement 0 -> T1 of the chosen vertices, px, camera motion included
    -- which is what the prior predicts under camera_static=False."""
    from src.render.crop import project_points
    a, _ = project_points(cam_at(0), np.asarray(pts0, float)[idx])
    b, _ = project_points(cam_at(T1), np.asarray(pts1, float)[idx])
    return b - a


# --------------------------------------------------------------------------- score

def bg_shift(f0, f1):
    """Background displacement between two frames, (dx, dy) px, from the top band."""
    import imageio.v2 as imageio
    a = np.asarray(imageio.imread(f0)[..., :3].mean(-1), np.float32)
    b = np.asarray(imageio.imread(f1)[..., :3].mean(-1), np.float32)
    ys, xs = slice(30, 110), slice(80, 460)
    best = (np.inf, 0, 0)
    for dy in range(-12, 13):
        bb = np.roll(b, -dy, axis=0)
        for dx in range(-60, 61):
            e = np.abs(a[ys, xs] - np.roll(bb, -dx, axis=1)[ys, xs]).mean()
            if e < best[0]:
                best = (e, dx, dy)
    return best[1], best[2]


def do_score(args):
    import torch
    from PIL import Image
    import f0_fpt_gate as F
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda:0"
    m = F.load_model(dev)
    W, H = wh()
    only = set(args.assets.split(",")) if args.assets else None
    for tag, _ in T.ASSETS:
        if only and tag not in only:
            continue
        d = lab(tag)
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        res = {}
        for tilt in T.TILTS:
            rd = d / f"sim_{tkey(tilt)}@a"
            f0, f1 = rd / "f0000.png", rd / f"f{T1:04d}.png"
            rows = [r for r in spec if r["tilt"] == tilt]
            if not (f0.exists() and f1.exists() and rows):
                print(f"  {tag} t{tilt:.0f}: missing render"); continue
            dx, dy = bg_shift(f0, f1)
            idx = query_idx(meta[rows[0]["key"]][0])
            from src.render.crop import project_points
            uv, _ = project_points(cam_at(0), np.asarray(meta[rows[0]["key"]][0], float)[idx])
            q = (uv / np.array([W, H])).astype(np.float32)
            im = np.asarray(Image.open(f0).convert("RGB").resize((518, 518), Image.BILINEAR),
                            np.float32) / 127.5 - 1.0
            img = torch.tensor(im.transpose(2, 0, 1)[None], device=dev)
            # background poke at two top-band points, carrying the MEASURED camera flow
            Z = torch.tensor([[[0.15, 0.10], [0.85, 0.10]]], dtype=torch.float32, device=dev)
            pf = torch.tensor([[[dx / W, dy / H]] * 2], dtype=torch.float32, device=dev)
            with torch.no_grad():
                dist = m.predict_parallel(Z, pf, torch.tensor(q[None], device=dev), False,
                                          m.embed_image(img))
            w = dist.mixture_distribution.probs[0].double().cpu().numpy()          # Q,K
            mu_ = dist.component_distribution.mean[0].double().cpu().numpy()       # Q,K,2
            L = dist.component_distribution.scale_tril[0].double().cpu().numpy()   # Q,K,2,2
            # a repeat call must give the same answer (rule 7)
            with torch.no_grad():
                d2 = m.predict_parallel(Z, pf, torch.tensor(q[None], device=dev), False,
                                        m.embed_image(img))
            rep = float(np.abs(d2.component_distribution.mean[0].double().cpu().numpy()
                               - mu_).max())
            mean = (w[..., None] * mu_).sum(1) * np.array([W, H])                  # px
            sw = float(w[np.arange(len(w)), np.linalg.norm(mu_, axis=-1).argmin(1)].mean())
            res[tkey(tilt)] = {"tilt": tilt, "bg": [dx, dy], "idx": idx.tolist(),
                               "w": w.tolist(), "mu": mu_.tolist(), "L": L.tolist(),
                               "mean_px": mean.tolist(), "static_w": sw, "repeat": rep}
            print(f"  {tag:<7} t{tilt:>2.0f}: bg ({dx:+d},{dy:+d}) px   target "
                  f"{np.linalg.norm(mean, axis=-1).mean():6.1f} px   static w {sw:.3f}"
                  f"   repeat {rep:.1e}", flush=True)
        (d / "fpt.json").write_text(json.dumps(res))
    return 0


# --------------------------------------------------------------------------- fit

def mix_logp(f_px, w, mu_, L, W, H):
    """log density of per-point flows under the per-point 2-D mixture (normalised units)."""
    f = f_px / np.array([W, H])
    out = np.empty(len(f))
    for i in range(len(f)):
        lp = []
        for k in range(len(w[i])):
            Li = np.asarray(L[i][k]); r = np.linalg.solve(Li, f[i] - np.asarray(mu_[i][k]))
            lp.append(np.log(w[i][k] + 1e-300) - 0.5 * r @ r
                      - np.log(np.abs(np.diag(Li))).sum() - np.log(2 * np.pi))
        lp = np.array(lp); mx = lp.max()
        out[i] = mx + np.log(np.exp(lp - mx).sum())
    return out


def do_fit(args):
    W, H = wh()
    tags = [t for t, _ in T.ASSETS if (lab(t) / "fpt.json").exists()]
    D = {}
    for tag in tags:
        d = lab(tag)
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        fpt = json.loads((d / "fpt.json").read_text())
        per = {}
        for tk, c in fpt.items():
            rows = sorted([r for r in spec if tkey(r["tilt"]) == tk], key=lambda r: r["mu"])
            idx = np.array(c["idx"])
            fl = np.stack([sim_flow(meta[r["key"]][0], meta[r["key"]][T1], idx) for r in rows])
            lp = np.array([mix_logp(f, c["w"], c["mu"], c["L"], W, H).mean() for f in fl])
            per[tk] = {"tilt": c["tilt"], "mu": np.array([r["mu"] for r in rows]),
                       "flow": fl, "tgt": np.asarray(c["mean_px"]), "logp": lp,
                       "rows": rows, "static_w": c["static_w"]}
        D[tag] = per

    G = np.geomspace(MU[0], MU[-1], 400)
    lg = np.log(G)

    def mag_curve(p):
        """log mean |flow| over points, interpolated onto the fine mu grid."""
        y = np.log(np.linalg.norm(p["flow"], axis=-1).mean(1))
        return np.interp(lg, np.log(p["mu"]), y)

    def vec_curve(p):
        return np.stack([np.interp(lg, np.log(p["mu"]), p["flow"][:, i, j])
                         for i in range(p["flow"].shape[1]) for j in range(2)], -1
                        ).reshape(len(G), p["flow"].shape[1], 2)

    def argmin_mu(J):
        i = int(np.argmin(J))
        return float(G[i]), bool(i in (0, len(G) - 1))

    def tmag(p):
        return float(np.log(np.linalg.norm(p["tgt"], axis=-1).mean()))

    # --- target table: does the prior's prediction depend on the object, the tilt, both?
    tilts = sorted({p["tilt"] for per in D.values() for p in per.values()})
    Y = np.full((len(tags), len(tilts)), np.nan)
    for a, tag in enumerate(tags):
        for b, tl in enumerate(tilts):
            p = D[tag].get(tkey(tl))
            if p is not None:
                Y[a, b] = tmag(p)
    print("\nTARGET -- prior's mean predicted displacement over 1 s, px")
    print(f"  {'asset':<8}" + "".join(f"{int(t):>7}d" for t in tilts))
    for a, tag in enumerate(tags):
        print(f"  {tag:<8}" + "".join(f"{np.exp(v):>8.1f}" if np.isfinite(v) else f"{'-':>8}"
                                     for v in Y[a]))
    ok = np.isfinite(Y).all(0)
    Yc = Y[:, ok]
    gm = Yc.mean()
    va = Yc.shape[1] * ((Yc.mean(1) - gm) ** 2).sum()
    vt = Yc.shape[0] * ((Yc.mean(0) - gm) ** 2).sum()
    vtot = ((Yc - gm) ** 2).sum()
    print(f"  variance of log target: asset {100*va/vtot:.0f}%  tilt {100*vt/vtot:.0f}%  "
          f"interaction {100*(vtot-va-vt)/vtot:.0f}%   (tilts present for all: "
          f"{[int(t) for t in np.array(tilts)[ok]]})")
    c_const = float(np.nanmean(Y))

    out = {"target_px": {tag: {str(int(t)): (float(np.exp(Y[a, b])) if np.isfinite(Y[a, b])
                                             else None) for b, t in enumerate(tilts)}
                         for a, tag in enumerate(tags)},
           "var_share": {"asset": va / vtot, "tilt": vt / vtot}, "assets": {}}

    print("\nFIT -- one mu per asset, all tilts jointly (| = hit the grid edge)")
    print(f"  {'asset':<8}{'per-tilt mu (magnitude)':<42}{'spread':>7}{'JOINT':>8}"
          f"{'LOTO':>13}{'vector':>8}{'logp':>8}{'const':>8}{'tilt-only':>10}")
    for a, tag in enumerate(tags):
        per = D[tag]
        tks = sorted(per, key=lambda k: per[k]["tilt"])
        C = {k: mag_curve(per[k]) for k in tks}
        V = {k: vec_curve(per[k]) for k in tks}
        single = {k: argmin_mu((C[k] - tmag(per[k])) ** 2) for k in tks}
        Jm = lambda ks, tg: np.mean([(C[k] - tg[k]) ** 2 for k in ks], 0)
        tg_fpt = {k: tmag(per[k]) for k in tks}
        mj, ej = argmin_mu(Jm(tks, tg_fpt))
        loto = [argmin_mu(Jm([k for k in tks if k != h], tg_fpt))[0] for h in tks]
        # vector: squared image-space error relative to the target's own size
        Jv = np.mean([((V[k] - per[k]["tgt"]) ** 2).sum(-1).mean(-1)
                      / (np.linalg.norm(per[k]["tgt"], axis=-1) ** 2).mean() for k in tks], 0)
        mv, ev = argmin_mu(Jv)
        Jl = -np.mean([np.interp(lg, np.log(per[k]["mu"]), per[k]["logp"]) for k in tks], 0)
        ml, el = argmin_mu(Jl)
        mc, ec = argmin_mu(Jm(tks, {k: c_const for k in tks}))
        # tilt-only null: other assets' targets at the same tilt, geometric mean
        tg_null = {}
        for k in tks:
            b = tilts.index(per[k]["tilt"])
            v = [Y[o, b] for o in range(len(tags)) if o != a and np.isfinite(Y[o, b])]
            tg_null[k] = float(np.mean(v))
        mn, en = argmin_mu(Jm(tks, tg_null))
        resid = float(np.sqrt(Jm(tks, tg_fpt).min()))
        st = "  ".join(f"{int(per[k]['tilt'])}:{single[k][0]:.2f}{'|' if single[k][1] else ''}"
                       for k in tks)
        sv = np.array([single[k][0] for k in tks])
        print(f"  {tag:<8}{st:<42}{sv.max()/sv.min():>6.2f}x{mj:>7.3f}{'|' if ej else ' '}"
              f"{min(loto):>6.2f}-{max(loto):<5.2f}{mv:>8.3f}{ml:>8.3f}{mc:>8.3f}{mn:>10.3f}")
        out["assets"][tag] = {
            "single": {str(int(per[k]["tilt"])): single[k][0] for k in tks},
            "single_edge": {str(int(per[k]["tilt"])): single[k][1] for k in tks},
            "joint": mj, "joint_edge": ej, "joint_rms_log": resid, "loto": loto,
            "vector": mv, "logp": ml, "const": mc, "tilt_only": mn,
            "static_w": {str(int(per[k]["tilt"])): per[k]["static_w"] for k in tks}}

    A = out["assets"]
    lj = np.log([A[t]["joint"] for t in tags])
    lc = np.log([A[t]["const"] for t in tags])
    ln = np.log([A[t]["tilt_only"] for t in tags])
    print("\nDOES THE PRIOR KNOW THE OBJECT?")
    print(f"  across assets, joint mu spans {np.exp(lj.max()-lj.min()):.2f}x")
    print(f"  |log(mu_FPT / mu_const)|     median {np.exp(np.median(np.abs(lj-lc))):.2f}x")
    print(f"  |log(mu_FPT / mu_tilt-only)| median {np.exp(np.median(np.abs(lj-ln))):.2f}x")
    print(f"  share of across-asset variance in log mu NOT explained by the const null: "
          f"{100*np.var(lj-lc)/max(np.var(lj),1e-12):.0f}%")
    out["summary"] = {"joint_span": float(np.exp(lj.max() - lj.min())),
                      "vs_const": float(np.exp(np.median(np.abs(lj - lc)))),
                      "vs_tilt_only": float(np.exp(np.median(np.abs(lj - ln))))}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, f in (("sim", do_sim), ("score", do_score), ("fit", do_fit)):
        s = sub.add_parser(nm); s.set_defaults(f=f)
        s.add_argument("--assets", default="")
    a = ap.parse_args()
    sys.exit(a.f(a))
