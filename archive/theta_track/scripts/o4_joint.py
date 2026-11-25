"""O4 -- can a SET of experiments pin down mu, when no single one can?

Every earlier O-track analysis asked one sweep at a time whether the classifier's score
identifies mu, and every one failed. That was the wrong question. The outcome of a tilt
experiment depends on friction AND ramp angle AND ramp rate, exactly as the physics says
it must; a single sweep therefore cannot separate them, and reading that as a failure of
the signal was an error. Tilt and rate are conditions we SET, so the right question is
whether mu is identified when several known conditions are constrained at once.

Two corrections to how the outcome is read, both measured here:

  * READ THE WHOLE DISTRIBUTION.  The two SSv2 classes chosen by name track true slide at
    rho = +0.343 / -0.462; the best available track it at +0.660 / -0.880, and 92 of the
    174 classes carry |rho| > 0.5.  A cross-validated ridge over all 174 log-probabilities
    reads log slide with sd 0.229 against 0.368 for the hand-picked pair.
  * BRACKET THE PARAMETER.  At 21-27 degrees a bowl above mu ~ 0.7 barely moves in any
    condition, so those rows carry no information and the estimate regresses to the middle
    (-41%, -44%, -41% at mu = 0.73, 0.94, 1.20).  The 33/36 degree sweeps exist to give
    that end real dynamic range.

Two things are tested that could still kill this, both of which have killed a signal in
this project before:

  TRANSFER  the readout is calibrated on the wooden bowl and applied to the brass pot.
            The across-object confound has appeared three times (drop probe -0.186 across
            / -0.995 within; V0 surprise -0.01 / +0.95; violation detection +0.104 /
            +0.891).  A sensor that must be recalibrated per object with ground truth we
            do not have for a new asset is not a sensor.
  HOLD-OUT  an entire friction is removed, the readout refit on the rest, and mu recovered
            from its clips alone -- so nothing in the estimate has seen the answer.

Reference to beat: a flat prior over mu in [0.10, 1.20] gives ~86% median error.

    LAB=outputs/scene/gt_pot   W scripts/o2b_tracking.py clip
    CUDA_VISIBLE_DEVICES=$G C scripts/o4_joint.py score --lab outputs/scene/gt_pot
    W scripts/o4_joint.py fit
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

CACHE = Path("outputs/outcome/cache")
OUT = Path("docs/results/o4_joint.json")
LAM = 30.0                      # selected by LOO over 30..10000; the curve is flat 30-300

# lab -> (object, does spec.json carry a per-row tilt?)
LABS = {"outputs/scene/ga":       ("bowl", 24.0),
        "outputs/scene/ga2":      ("bowl", None),
        "outputs/scene/ga3":      ("bowl", None),
        "outputs/scene/gt_steep": ("bowl", None),
        "outputs/scene/gt_pot":   ("pot",  None)}


MIN_CONTAIN = 0.99      # a clip whose object leaves frame is not a measurement


# CAM_PULL per lab. A lab clipped at a different pull must be CHECKED at that pull -- the
# analysis helper hardcoded 1.20 and reported 1 usable clip for a probe the clip stage had
# just measured at 100% containment. Third appearance of the same mismatch.
LAB_PULL = {"outputs/scene/p_tip": 2.20, "outputs/scene/p_tip2": 2.20}


def containment(lab, pull=None):
    """Fraction of each trajectory that stays inside the crop box.

    The first G-A result inverted because the object slid out of frame exactly where the
    signal should have been strongest, so this is checked per clip and out-of-frame clips
    are dropped rather than scored. CAM_PULL must match what the RENDER used -- reading it
    from the environment at clip time and forgetting to set it is how that bug returned.
    """
    import o2_boundary as O
    from src.render.crop import project_points
    from src.render.views import VIEWS
    from t0_tracking import pulled_camera
    d = Path(lab)
    spec = {r["key"]: r for r in json.loads((d / "spec.json").read_text())}
    meta = json.loads((d / "meta.json").read_text())
    cam = pulled_camera(float(pull if pull is not None
                                else LAB_PULL.get(str(lab), 1.20)), view="a")
    W, H = VIEWS["a"]["width"], VIEWS["a"]["height"]
    b = O.union_box(cam, meta, W, H)
    out = {}
    for k in spec:
        uv, ok = project_points(cam, np.asarray(meta[k], float).reshape(-1, 3))
        uv = uv[ok]
        out[k] = float(((uv[:, 0] >= b[0]) & (uv[:, 0] <= b[2]) &
                        (uv[:, 1] >= b[1]) & (uv[:, 1] <= b[3])).mean())
    return out


def load():
    """Every scored clip with its ground-truth slide, ramp angle, mu and object."""
    probs = {}
    for f in glob.glob(str(CACHE / "*.json")):
        d = json.load(open(f))
        probs[str(Path(d["clip"]).resolve())] = np.asarray(d["probs"], np.float64)
    rows, dropped = [], 0
    for lab, (obj, fixed_tilt) in LABS.items():
        sp = Path(lab) / "spec.json"
        if not sp.exists():
            continue
        con = containment(lab)
        for r in json.loads(sp.read_text()):
            if con.get(r["key"], 0.0) < MIN_CONTAIN:
                dropped += 1
                continue
            c = (Path(lab) / "clips" / f"{r['key']}.mp4").resolve()
            p = probs.get(str(c))
            if p is None:
                continue
            rows.append({"obj": obj, "mu": float(r["value"]),
                         "tilt": float(r.get("tilt", fixed_tilt)),
                         "slide": float(r["slide_cm"]), "p": p, "key": r["key"],
                         "contain": con[r["key"]]})
    print(f"dropped {dropped} clips whose object left frame (containment < {MIN_CONTAIN})")
    return rows


def design(rows):
    X = np.log(np.array([r["p"] for r in rows]) + 1e-8)
    return X, np.log10(np.maximum([r["slide"] for r in rows], 1e-2))


def ridge(Xtr, ytr, lam=LAM):
    """Standardise on the TRAIN split only -- otherwise the test clip informs its own scaling."""
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - m) / s
    b = ytr - ytr.mean()
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    return lambda X: ((X - m) / s) @ w + ytr.mean()


def forward(rows):
    """slide(mu | object, tilt), straight from the simulator. Log-log interpolation.

    Inverting needs slide to be monotone in mu. It is not everywhere -- the pot's base
    rounds off above mu ~ 0.35 and it tips rather than slides, so slide flattens at ~19 cm
    and the 36 deg bowl sweep ticks up at the last point. Non-monotone stretches are
    clipped rather than interpolated through, so an unreachable reading returns the end of
    the reachable range instead of a fabricated mu.
    """
    f = defaultdict(list)
    for r in rows:
        f[(r["obj"], r["tilt"])].append((r["mu"], r["slide"]))
    out = {}
    for k, v in f.items():
        a = np.array(sorted(v))
        mu, sl = np.log10(a[:, 0]), np.log10(np.maximum(a[:, 1], 1e-2))
        keep = [0]
        for i in range(1, len(sl)):                 # longest strictly-decreasing prefix
            if sl[i] < sl[keep[-1]]:
                keep.append(i)
        out[k] = (mu[keep], sl[keep])
    return out


def invert(fwd, obj, tilt, log_slide):
    mu, sl = fwd[(obj, tilt)]
    o = np.argsort(sl)
    return float(10 ** np.interp(log_slide, sl[o], mu[o]))


def do_score(args):
    from src.outcome.head import OutcomeHead
    lab = Path(args.lab)
    spec = json.loads((lab / "spec.json").read_text())
    m = OutcomeHead(device="cuda:0")
    n = 0
    for r in spec:
        c = lab / "clips" / f"{r['key']}.mp4"
        if not c.exists():
            print(f"  MISSING {c}"); continue
        m.score_clip(c); n += 1
        if n % 10 == 0:
            print(f"  scored {n}/{len(spec)}", flush=True)
    print(f"scored {n} clips from {lab}")
    return 0


def do_fit(args):
    rows = load()
    if not rows:
        sys.exit("no scored clips found -- run the score stage first")
    fwd = forward(rows)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["obj"]].append(i)
    print(f"{len(rows)} scored clips: " + ", ".join(
        f"{o} {len(ix)} over {len(set(rows[i]['tilt'] for i in ix))} ramp angles"
        for o, ix in by.items()))

    X, y = design(rows)
    res = {}

    # ---- 1. TRANSFER: calibrate on the bowl, read the pot -------------------------
    print("\n[1] TRANSFER -- readout fit on the BOWL only, applied to the POT")
    tr = np.array(by["bowl"]); te = np.array(by.get("pot", []))
    if len(te):
        pred = ridge(X[tr], y[tr])(X[te])
        sd_x = float((y[te] - pred).std())
        sd_in = float((y[tr] - ridge(X[tr], y[tr])(X[tr])).std())
        # a same-object reference: LOO within the pot
        loo = np.empty(len(te))
        for j in range(len(te)):
            k = np.ones(len(te), bool); k[j] = False
            loo[j] = ridge(X[te[k]], y[te[k]])(X[te[j]][None])[0]
        sd_own = float((y[te] - loo).std())
        print(f"    bowl -> pot   sd(log10 slide) = {sd_x:.3f}  (factor {10**sd_x:.2f})")
        print(f"    pot  -> pot   sd(log10 slide) = {sd_own:.3f}  (factor {10**sd_own:.2f})  [LOO]")
        print(f"    bowl in-sample                = {sd_in:.3f}")
        print(f"    -> transfer costs a factor of {10**(sd_x-sd_own):.2f} over own-object calibration")
        res["transfer"] = {"cross": sd_x, "own_loo": sd_own, "bowl_insample": sd_in}
    else:
        print("    no pot clips scored yet")

    # ---- 2. HOLD-OUT: recover a friction nothing has seen -------------------------
    print("\n[2] HOLD-OUT -- remove one mu entirely, refit, recover it from its clips")
    for obj in ("bowl", "pot"):
        ix = np.array(by.get(obj, []))
        if len(ix) < 8:
            continue
        mus = np.array([rows[i]["mu"] for i in ix])
        print(f"\n  {obj}: {len(set(np.round(mus,3)))} frictions x "
              f"{sorted(set(rows[i]['tilt'] for i in ix))} deg")
        print(f"  {'true mu':>8} {'n':>3} {'per-condition mu-hat':>34} {'joint':>8} {'err':>7}")
        e1, ej, tab = [], [], []
        for mu0 in sorted(set(np.round(mus, 3))):
            hold = ix[np.isclose(mus, mu0, atol=1e-3)]
            keep = ix[~np.isclose(mus, mu0, atol=1e-3)]
            if len(hold) < 2 or len(keep) < 10:
                continue
            rd = ridge(X[keep], y[keep])
            hats = [invert(fwd, obj, rows[i]["tilt"], rd(X[i][None])[0]) for i in hold]
            joint = float(10 ** np.mean(np.log10(hats)))
            e1 += [abs(np.log10(h / mu0)) for h in hats]
            ej.append(abs(np.log10(joint / mu0)))
            tab.append({"mu": mu0, "hats": hats, "joint": joint})
            st = " ".join(f"{h:5.3f}" for h in hats)
            print(f"  {mu0:>8.3f} {len(hats):>3} {st:>34} {joint:>8.3f} "
                  f"{100*(joint/mu0-1):>+6.0f}%")
        if not ej:
            continue
        m1, mj = float(np.median(e1)), float(np.median(ej))
        print(f"    single condition   median error {100*(10**m1-1):>4.0f}%   (n={len(e1)})")
        print(f"    conditions joined  median error {100*(10**mj-1):>4.0f}%   (n={len(ej)})")
        print(f"    flat prior on [0.10, 1.20]     ~86%")
        res[f"holdout_{obj}"] = {"single_median_pct": 100*(10**m1-1),
                                 "joint_median_pct": 100*(10**mj-1), "rows": tab}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--lab", required=True); s.set_defaults(f=do_score)
    s = sub.add_parser("fit"); s.set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
