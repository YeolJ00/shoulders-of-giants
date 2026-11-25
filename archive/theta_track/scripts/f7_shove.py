"""F7 -- SHOVE probe for a motion prior: does MYRIAD's deceleration pick out one mu?

Ramps failed for both motion priors (f4/f5/f6): from rest, nothing moves in their
predictions, and a candidate's own first 1/12 s is 0.3-3 px whatever mu is, so a poke
carries no mu. The one physical behaviour MYRIAD showed is DECELERATION of something
already moving (its ball_roll: 42 -> 12 px/step). A shove is that situation:

  level table, the object given v0 along +x, friction stops it after v0^2 / (2 mu_eff g),
  mu_eff = sqrt(mu * mu_table). No ramp, nothing unusual in the scene.

SELF-POKED, as before: each candidate mu pokes MYRIAD with ITS OWN first simulated 1/12 s
(which now carries v0). No external motion enters. The question: is there a mu at which
MYRIAD's continuation agrees with the simulator's, is it the SAME mu at every v0 (a
material property must be), and does it differ between objects (else it is a prior)?

    W scripts/f7_shove.py sim
    blender per asset: LAB=outputs/scene/f7_<tag> VIEWS=a CAM_PULL=2.0 CAM_ORBIT=6 \
        ORBIT_NF=64 FRAMES=0,24      (frame 0 only is used; cam_at(0) as in f4/f5)
    C scripts/f7_shove.py run
    W scripts/f7_shove.py fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import o13_tilted as T
import f4_fpt_joint as F

V0 = (0.8, 1.2, 1.6)                  # m/s
MU = F.MU                              # 0.08 .. 2.5, 18 values
MU_POKE = MU[::3]                      # 6 candidates scored by MYRIAD per condition
TABLE_HALF = (0.565, 0.353)            # the REAL tabletop (wooden_table_02: 1.13 x 0.71 m)
START = -0.30                          # m along x from the table centre
OUT = Path("docs/results/f7_shove.json")


def lab(tag):
    return Path(f"outputs/scene/f7_{tag}")


def vkey(v0):
    return f"v{int(round(v0 * 10)):02d}"


def mkey(v0, mu):
    return f"{vkey(v0)}_m{mu:06.4f}"


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
            for v0 in V0:
                for mu in MU:
                    s = MeshProbeScene([asset], [[0., 0., 0.]], [[float(v0), 0., 0.]],
                                       masses=[T.MASS], ground_z=T.GZ,
                                       dt=1.0 / (T.FPS * T.SUBSTEPS),
                                       n_steps=F.NFW * T.SUBSTEPS, k=2500.0, cd=[3200.0],
                                       mu=[float(mu)], mesh_scale=[sc], faces=900,
                                       ground_mu=T.TABLE_MU, ground_cd=3000.0,
                                       table=TABLE_HALF, tilt_deg=0.0)
                    from src.sim.rigs import apply_tilt_rig
                    apply_tilt_rig(s)      # stiffness + restitution (alpha), src/sim/rigs.py
                    s.seat_on_plane(along=START)
                    s.rollout()
                    P, Q = s.positions(T.SUBSTEPS), s.rotations(T.SUBSTEPS)
                    if not (np.isfinite(P).all() and np.isfinite(Q).all()
                            and np.abs(P).max() < 50):
                        print(f"   v{v0} mu={mu:.3f} DIVERGED"); continue
                    dd = P[:, 0] - s._gp
                    zang = np.degrees(np.arccos(np.clip(
                        [float(T.quat_to_mat(q)[:, 2] @ np.array([0, 0, 1.])) for q in Q[:, 0]],
                        -1, 1)))
                    V = s.local.numpy().astype(float)
                    idx = np.linspace(0, len(V) - 1, min(48, len(V))).astype(int)
                    pts, seq = [], []
                    for t2 in range(len(P)):
                        R = T.quat_to_mat(Q[t2, 0])
                        pts.append(((R @ V[idx].T).T + P[t2, 0]).tolist())
                        loc = P[t2, 0] - R @ s.coms[0]
                        seq.append({"loc": [float(v) for v in loc],
                                    "mat": [[float(v) for v in r] for r in R]})
                    x = P[:, 0, 0] - P[0, 0, 0]
                    sp = np.abs(np.diff(x)) * T.FPS
                    stop = next((i / T.FPS for i in range(1, len(sp)) if sp[i] < 0.02), None)
                    k = mkey(v0, mu)
                    meta[k] = pts
                    row = {"key": k, "v0": v0, "mu": float(mu),
                           "slide_cm": float(x[-1]) * 100.0, "stop_s": stop,
                           "off_table": bool((np.abs(dd[:, 0]) > TABLE_HALF[0]).any()),
                           "tumbled": bool((zang > 25.0).any()),
                           "analytic_cm": 100.0 * v0 ** 2 / (2 * np.sqrt(mu * T.TABLE_MU) * 9.81)}
                    spec.append(row)
                    poses.setdefault("s0", {"tilt_deg": 0.0, "objects": {"subject": seq}})
                    fl = "".join(f"  {x_.upper()}" for x_ in ("off_table", "tumbled") if row[x_])
                    print(f"   v{v0} mu={mu:6.3f} slide {row['slide_cm']:6.1f} cm "
                          f"(v0^2/2mu g {row['analytic_cm']:6.1f})  stop "
                          f"{'-' if stop is None else f'{stop:.2f} s'}{fl}", flush=True)
            (d / "spec.json").write_text(json.dumps(spec, indent=1))
            (d / "meta.json").write_text(json.dumps(meta))
            (d / "scene_poses.json").write_text(json.dumps(poses))
            (d / "scene.json").write_text(json.dumps(
                {"ground_z": T.GZ, "objects": {"subject": {
                    "asset": asset, "scale": sc, "rot_z": 0.0,
                    "pos": [0.0, 0.0, T.GZ + 0.30]}}}))
    return 0


def do_run(args):
    """Unpoked once per asset (frame 0 is the same for every v0 and mu), then self-poked at
    MU_POKE for every v0 -- batched through f5's simulate_batch."""
    import time
    import torch
    import imageio.v2 as imageio
    import f5_myriad as M
    from src.render.crop import project_points
    m = M.load_model()
    W, H = F.wh()
    l, t, s = M.crop_sq(int(W), int(H))
    out, jobs = {}, []
    for tag, _ in T.ASSETS:
        d = lab(tag)
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        k0 = spec[0]["key"]
        idx = F.query_idx(meta[k0][0])
        uv, _ = project_points(F.cam_at(0), np.asarray(meta[k0][0], float)[idx])
        q = (uv - [l, t]) / s
        im = imageio.imread(d / "sim_s0@a" / "f0000.png")[..., :3][t:t + s, l:l + s]
        out[tag] = {"idx": idx.tolist(), "poked": {}}
        jobs.append((tag, None, im, q, None))
        keys = {r["key"] for r in spec}
        for v0 in V0:
            for mu in MU_POKE:
                k = mkey(v0, mu)
                if k not in keys:
                    continue
                u1, _ = project_points(F.cam_at(0), np.asarray(meta[k][2], float)[idx])
                jobs.append((tag, k, im, q, (u1 - [l, t]) / s))
    groups = {}
    for j in jobs:
        groups.setdefault((len(j[3]), j[4] is not None), []).append(j)
    t0 = time.time(); done = 0
    for (nq, poked), js in groups.items():
        for c0 in range(0, len(js), M.CHUNK):
            ch = js[c0:c0 + M.CHUNK]
            P = M.simulate_batch(m, [j[2] for j in ch], [j[3] for j in ch],
                                 [j[4] for j in ch] if poked else None,
                                 seed=M.SEED + c0 + (10000 if poked else 0))
            for j, Pj in zip(ch, P):
                px = (Pj * s + np.array([l, t])).tolist()
                if j[1] is None:
                    out[j[0]]["paths_px"] = px
                else:
                    out[j[0]]["poked"][j[1]] = px
            done += len(ch)
            print(f"  {done}/{len(jobs)} cases  {time.time()-t0:6.0f} s  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    Path("outputs/scene/f7_myriad_paths.json").write_text(json.dumps(out))
    return 0


def do_fit(args):
    import imageio.v2 as imageio
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import f5_myriad as M
    R = json.loads(Path("outputs/scene/f7_myriad_paths.json").read_text())
    kb = len(M.BG)
    tags = [t for t, _ in T.ASSETS if t in R]
    res = {}
    fig, axs = plt.subplots(len(tags), len(V0), figsize=(5.2 * len(V0), 3.6 * len(tags)), dpi=90)
    cmap = plt.get_cmap("viridis")
    print("displacement along the shove over 1 s, px: MYRIAD (self-poked) / simulator")
    print(f"{'asset':<7}{'v0':>5} " + "".join(f"{f'mu={m_:.2f}':>15}" for m_ in MU_POKE)
          + f"{'crossing mu':>13}")
    for a, tag in enumerate(tags):
        d = lab(tag)
        spec = {r["key"]: r for r in json.loads((d / "spec.json").read_text())}
        meta = json.loads((d / "meta.json").read_text())
        idx = np.array(R[tag]["idx"])
        un = np.array(R[tag]["paths_px"])[:, :, kb:]
        res[tag] = {"unpoked_px": float(np.median(np.linalg.norm(
            (un[:, -1] - un[:, 0]).mean(1), axis=-1))), "v0": {}}
        im = imageio.imread(d / "sim_s0@a" / "f0000.png")[..., :3]
        for b, v0 in enumerate(V0):
            ax = axs[a, b]; ax.imshow(im)
            my, sm, mus, cells = [], [], [], []
            for i, mu in enumerate(MU_POKE):
                k = mkey(v0, mu)
                if k not in R[tag]["poked"]:
                    continue
                S = M.sim_paths(meta, k, idx)                         # T,N,2
                dh = (S[-1] - S[0]).mean(0); dh /= max(np.linalg.norm(dh), 1e-9)
                pk = np.array(R[tag]["poked"][k])[:, :, kb:]         # E,T,N,2
                y = float(np.median((pk[:, -1] - pk[:, 0]).mean(1) @ dh))
                x = float(((S[-1] - S[0]).mean(0)) @ dh)
                my.append(y); sm.append(x); mus.append(mu)
                bad = "*" if (spec[k]["off_table"] or spec[k]["tumbled"]) else " "
                cells.append(f"{y:6.1f}/{x:<6.1f}{bad} ")
                col = cmap(i / (len(MU_POKE) - 1))
                c = S.mean(1)
                ax.plot(c[:, 0], c[:, 1], "-", color=col, lw=2.2)
                ax.plot(c[-1, 0], c[-1, 1], "o", color=col, ms=4)
                mc = np.median(pk.mean(2), axis=0)
                ax.plot(mc[:, 0], mc[:, 1], "--", color=col, lw=1.6)
                ax.plot(mc[-1, 0], mc[-1, 1], "x", color=col, ms=6, mew=2)
            # crossing: where log(MYRIAD / sim) changes sign, interpolated in log mu
            g = np.log(np.maximum(my, 1e-3)) - np.log(np.maximum(sm, 1e-3))
            cross = None
            for i in range(len(g) - 1):
                if g[i] * g[i + 1] <= 0 and g[i] != g[i + 1]:
                    f = g[i] / (g[i] - g[i + 1])
                    cross = float(np.exp(np.log(mus[i]) + f * (np.log(mus[i + 1]) - np.log(mus[i]))))
                    break
            res[tag]["v0"][vkey(v0)] = {"mu": mus, "myriad": my, "sim": sm, "crossing": cross}
            print(f"{tag:<7}{v0:>5} " + "".join(cells)
                  + f"{'none' if cross is None else f'{cross:.3f}':>10}")
            c0 = M.sim_paths(meta, mkey(v0, MU_POKE[0]), idx)[0].mean(0)
            ax.set_xlim(c0[0] - 60, c0[0] + 260); ax.set_ylim(c0[1] + 160, c0[1] - 90)
            ax.axis("off"); ax.set_title(f"{tag}  v0 = {v0} m/s", fontsize=10)
    fig.suptitle("SOLID = simulator (object centre, 1 s), one colour per mu (dark = low mu = "
                 "slides far).  DASHED = MYRIAD poked with that mu's own first 1/12 s",
                 fontsize=11)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.985))
    fig.savefig("docs/results/f7_shove.png")
    OUT.write_text(json.dumps(res, indent=1))
    print("  * = off the table or tumbled within 1 s")
    print("\nCROSSINGS -- the same mu at every v0 would be a reading; none is no target")
    for tag in tags:
        cr = [res[tag]["v0"][vkey(v)]["crossing"] for v in V0]
        print(f"  {tag:<7} " + "  ".join("none " if c is None else f"{c:.3f}" for c in cr)
              + f"   unpoked {res[tag]['unpoked_px']:.1f} px")
    print("wrote docs/results/f7_shove.png and", OUT)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, f in (("sim", do_sim), ("run", do_run), ("fit", do_fit)):
        s_ = sub.add_parser(nm); s_.set_defaults(f=f); s_.add_argument("--assets", default="")
    a = ap.parse_args()
    sys.exit(a.f(a))
