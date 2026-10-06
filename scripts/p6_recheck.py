"""P6 -- what do today's two simulator fixes do to the probes already run?

Fixes: seat_on_plane clearance 2 mm -> 0 (a release-time drop gave sliders a head start),
and centre of mass / inertia from the object's build (mass_model "auto") instead of the
vertex mean. Both change the physics every tilted probe measured. Nothing on disk is
overwritten: every condition of a stored lab is re-simulated in memory three ways --

  LEGACY  mass_model="vertex", clearance 2 mm  -> must REPRODUCE the stored numbers
          (otherwise the comparison itself is broken);
  SEAT    seating fix only;
  BOTH    seating + mass-model fix (the current default).

o13 (tilted-ramp friction, 7 assets): slide_cm per condition; o14 (topple, vase and pot):
time-to-topple per condition. Reported: reproduction error, size of each change, and
whether the probe still orders its parameter the same way (Spearman within each tilt).

    W scripts/p6_recheck.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "docs/results/p6_recheck.json"
VARIANTS = {"legacy": ("vertex", 0.002), "seat": ("vertex", 0.0), "both": ("auto", 0.0)}


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def o13_one(asset, scale, tilt, mu, mm, cl):
    import o13_tilted as T
    from src.sim.mesh_scene import MeshProbeScene
    s = MeshProbeScene([asset], [[0., 0., 0.]], [[0., 0., 0.]], masses=[T.MASS], ground_z=T.GZ,
                       dt=1.0 / (T.FPS * T.SUBSTEPS), n_steps=T.NF * T.SUBSTEPS, k=2500.0,
                       cd=[3200.0], mu=[mu], mesh_scale=[scale], faces=900, ground_mu=T.TABLE_MU,
                       ground_cd=3000.0, table=T.TABLE, tilt_deg=tilt, mass_model=mm)
    s.calibrate_stiffness(pen_frac=T.PEN_FRAC)
    s.seat_on_plane(along=T.START_ALONG, clearance=cl)
    s.rollout()
    P = s.positions(T.SUBSTEPS)
    if not np.isfinite(P).all() or np.abs(P).max() > 50:
        return None
    return float(np.dot(P[-1, 0] - P[0, 0], s._gt)) * 100.0


def o14_one(lab, row, mm, cl):
    import o14_com as C
    from src.sim.mesh_scene import MeshProbeScene
    sc = json.loads((lab / "scene.json").read_text())["objects"]["subject"]
    ko = dict(masses=[C.MASS], ground_z=C.GZ, dt=1.0 / (C.FPS * C.SUBSTEPS),
              n_steps=C.NF * C.SUBSTEPS, k=2500.0, cd=[C.CD], mu=[C.MU],
              mesh_scale=[sc["scale"]], faces=900, ground_mu=0.45, ground_cd=3000.0,
              table=(0.79, 0.49), tilt_deg=row["tilt"], mass_model=mm)
    if abs(row["value"]) > 1e-9:
        ko["com_offset"] = [[float(row["value"]), 0.0, 0.0]]
    s = MeshProbeScene([sc["asset"]], [[0., 0., 0.]], [[0., 0., 0.]], **ko)
    s.calibrate_stiffness(pen_frac=1.0e-4)
    s.seat_on_plane(along=0.0, clearance=cl)
    s.rollout()
    Q = s.rotations(C.SUBSTEPS)
    ang = np.array([np.degrees(np.arccos(np.clip(C.q2m(q)[:, 2] @ s._gn, -1, 1))) for q in Q[:, 0]])
    over = np.where(ang > C.TOPPLE_DEG)[0]
    return float(over[0] / C.FPS) if len(over) else float(C.NF / C.FPS)


def main():
    import warp as wp
    import o13_tilted as T
    wp.init()
    res = {"o13": {}, "o14": {}}
    with wp.ScopedDevice("cuda:0"):
        print("o13 tilted-ramp friction: slide cm  (stored vs legacy-resim vs seat vs both)")
        for tag, asset in T.ASSETS:
            spec = json.loads((REPO / f"outputs/scene/t_{tag}/spec.json").read_text())
            sc = T.scale_for(asset)
            rows = []
            for r in spec:
                v = {k: o13_one(asset, sc, r["tilt"], r["value"], mm, cl)
                     for k, (mm, cl) in VARIANTS.items()}
                rows.append({"key": r["key"], "tilt": r["tilt"], "mu": r["value"],
                             "stored": r["slide_cm"], **v})
            ok = [x for x in rows if None not in (x["legacy"], x["seat"], x["both"])]
            rep = np.median([abs(x["legacy"] - x["stored"]) for x in ok])
            dseat = np.median([abs(x["seat"] - x["legacy"]) / max(abs(x["legacy"]), 1) for x in ok])
            dboth = np.median([abs(x["both"] - x["legacy"]) / max(abs(x["legacy"]), 1) for x in ok])
            rho = []
            for tl in sorted({x["tilt"] for x in ok}):
                g = [x for x in ok if x["tilt"] == tl]
                if len(g) >= 4:
                    rho.append((sp([x["mu"] for x in g], [x["legacy"] for x in g]),
                                sp([x["mu"] for x in g], [x["both"] for x in g])))
            res["o13"][tag] = {"rows": rows, "repro_cm": float(rep), "rel_change_seat": float(dseat),
                               "rel_change_both": float(dboth), "rho_legacy_both": rho}
            print(f"  {tag:<7} reproduce |legacy-stored| {rep:6.3f} cm   change: seat {100*dseat:5.1f}%  "
                  f"both {100*dboth:5.1f}%   rho(mu,slide) per tilt legacy->both: "
                  + " ".join(f"{a:+.2f}->{b:+.2f}" for a, b in rho), flush=True)
        print("\no14 topple: time-to-topple s")
        for tag in ("vase", "pot"):
            lab = REPO / f"outputs/scene/c_{tag}"
            spec = json.loads((lab / "spec.json").read_text())
            rows = []
            for r in spec:
                v = {k: o14_one(lab, r, mm, cl) for k, (mm, cl) in VARIANTS.items()}
                rows.append({"key": r["key"], "tilt": r["tilt"], "com": r["value"],
                             "stored": r["t_topple_s"], **v})
            rep = np.median([abs(x["legacy"] - x["stored"]) for x in rows])
            ch = np.median([abs(x["both"] - x["legacy"]) for x in rows])
            rho = []
            for tl in sorted({x["tilt"] for x in rows}):
                g = [x for x in rows if x["tilt"] == tl]
                rho.append((sp([x["com"] for x in g], [x["legacy"] for x in g]),
                            sp([x["com"] for x in g], [x["both"] for x in g])))
            nev = (sum(x["legacy"] >= 1.33 for x in rows), sum(x["both"] >= 1.33 for x in rows))
            res["o14"][tag] = {"rows": rows, "repro_s": float(rep), "abs_change_both_s": float(ch),
                               "rho_legacy_both": rho, "never_toppled_legacy_both": nev}
            print(f"  {tag:<5} reproduce {rep:.3f} s   change (both) median {ch:.3f} s   "
                  f"never-toppled {nev[0]}->{nev[1]} of {len(rows)}   rho(com,t) per tilt: "
                  + " ".join(f"{a:+.2f}->{b:+.2f}" for a, b in rho), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
