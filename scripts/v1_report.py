"""V1 verdict and plots. Runs in the WARP env (matplotlib is not in `cosmos`).

    /home/jooyeolyun/anaconda3/envs/warp/bin/python scripts/v1_report.py

The pass/fail criteria were fixed BEFORE the sweep was run and are restated here so the
verdict cannot be chosen after seeing the curves:

  violation control   >= floor + 5 sigma, and above every legal theta
  mass null control   spread <= 2 sigma_floor
  KILL TEST PASS      on at least one axis: spread >= 3 sigma_floor AND
                      |Spearman rho| >= 0.7 against |log theta - log theta_ref|
  KILL TEST FAIL      controls fire, legal sweep flat -> negative result, no V2

theta_ref is the value used in the reference render, not a handbook number. The primary
condition is spread-above-floor plus monotonicity; where the minimum lands is REPORTED,
not required (rule 2 -- invariants, not reference values).
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC = Path("docs/results/v1_surprise.json")
OUT = Path("docs/results")
LABEL = {"cd": "contact damping  cd", "mu": "friction  mu", "mass": "mass  (kg)"}


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    rows = json.loads(SRC.read_text())["rows"]
    by = defaultdict(list)
    for r in rows:
        by[r["probe"]].append(r)

    # ---- the noise floor: identical physics, one nuisance changed --------------
    floor = by.get("floor", [])
    seeds = [r for r in floor if r["param"].startswith("seed")]
    jits = [r for r in floor if r["param"].startswith("jit")]
    s_seed = float(np.std([r["surprise"] for r in seeds], ddof=1)) if len(seeds) > 1 else 0.0
    s_jit = float(np.std([r["surprise"] for r in jits], ddof=1)) if len(jits) > 1 else 0.0
    sigma = max(s_seed, s_jit)
    f_mean = float(np.mean([r["surprise"] for r in floor])) if floor else float("nan")
    print(f"noise floor  ({len(seeds)} render seeds, {len(jits)} camera jitters)")
    print(f"  sigma(render seed)   {s_seed:.6f}")
    print(f"  sigma(camera jitter) {s_jit:.6f}")
    print(f"  sigma used           {sigma:.6f}   mean {f_mean:.5f}")

    # ---- per-axis statistics ---------------------------------------------------
    stats, verdict = {}, {}
    for probe in ("bounce", "settle", "tilt", "null"):
        rs = sorted(by.get(probe, []), key=lambda r: r["value"])
        if len(rs) < 3:
            continue
        v = np.array([r["value"] for r in rs])
        y = np.array([r["surprise"] for r in rs])
        ref = rs[0]["ref"]
        dist = np.abs(np.log(v / ref))
        st = {"param": rs[0]["param"], "values": v.tolist(), "surprise": y.tolist(),
              "motion": [r["motion"] for r in rs], "ref": ref,
              "spread": float(y.max() - y.min()),
              "spread_sigma": float((y.max() - y.min()) / sigma) if sigma else float("inf"),
              "rho_vs_dist": spearman(dist, y),
              "rho_vs_theta": spearman(np.log(v), y),
              # A theta that damps harder also MOVES less (measured: motion 4.05 -> 1.54
              # across the cd sweep), and V0 showed surprise tracks motion within a scene.
              # So a monotone surprise-vs-theta curve may be reading "how much moved"
              # rather than "how implausible it was". The two are told apart by SHAPE: a
              # plausibility signal has a minimum somewhere inside the sweep, a
              # motion-mediated artefact is monotone and tracks this correlation.
              "rho_vs_motion": spearman([r["motion"] for r in rs], y),
              # surprise(real)/surprise(frame-shuffled copy of the same clip): normalised
              # by that clip's own difficulty, so appearance and motion budget cancel and
              # what is left is how much the ORDER of the frames helped.
              "ratio": [r.get("ratio") for r in rs],
              "ratio_spread_sigma": None,
              "rho_ratio_vs_dist": spearman(dist, [r.get("ratio", np.nan) for r in rs]),
              "argmin_ratio": float(v[int(np.argmin([r.get("ratio", np.inf) for r in rs]))]),
              "argmin": float(v[int(np.argmin(y))])}
        stats[probe] = st
        print(f"\n{probe}  ({st['param']})")
        for r in rs:
            print(f"   {r['value']:9.3f}  surprise {r['surprise']:.5f}"
                  f"  ({(r['surprise'] - f_mean) / sigma:+6.1f} sigma)"
                  f"  motion {r['motion']:5.2f}")
        print(f"   spread {st['spread']:.5f} = {st['spread_sigma']:.1f} sigma"
              f"   rho(|log theta - log ref|) = {st['rho_vs_dist']:+.3f}"
              f"   rho(log theta) = {st['rho_vs_theta']:+.3f}")
        print(f"   minimum at {st['param']} = {st['argmin']:.3f}  (reference {ref:.3f})")
        print(f"   rho(surprise, clip motion) = {st['rho_vs_motion']:+.3f}"
              f"   <- if this is large, the curve may be reading motion, not plausibility")
        if st["ratio"][0] is not None:
            print(f"   normalised by each clip's own shuffled copy:"
                  f"  rho = {st['rho_ratio_vs_dist']:+.3f}"
                  f"   minimum at {st['param']} = {st['argmin_ratio']:.3f}")
            print("      " + "  ".join(f"{x:.4f}" for x in st["ratio"]))
        if probe != "null":
            verdict[probe] = bool(st["spread_sigma"] >= 3.0 and abs(st["rho_vs_dist"]) >= 0.7)

    # ---- controls ---------------------------------------------------------------
    viol = sorted(by.get("violation", []), key=lambda r: -r["surprise"])
    legal_max = max((r["surprise"] for p in ("bounce", "settle", "tilt")
                     for r in by.get(p, [])), default=float("nan"))
    print("\nviolation control  (must be >= floor + 5 sigma AND above every legal theta)")
    for r in viol:
        z = (r["surprise"] - f_mean) / sigma if sigma else float("inf")
        print(f"   {r['param']:<10} {r['surprise']:.5f}  ({z:+6.1f} sigma)"
              f"  {'above' if r['surprise'] > legal_max else 'BELOW'} every legal theta")
    v_ok = bool(viol) and all(r["surprise"] >= f_mean + 5 * sigma and r["surprise"] > legal_max
                              for r in viol)
    n_ok = stats.get("null", {}).get("spread_sigma", float("inf")) <= 2.0

    print(f"\ncontrols:  violations {'PASS' if v_ok else 'FAIL'}"
          f"   mass null {'PASS' if n_ok else 'FAIL'}"
          f" (spread {stats.get('null', {}).get('spread_sigma', float('nan')):.1f} sigma)")
    passed = [k for k, ok in verdict.items() if ok]
    print(f"KILL TEST: {'PASS on ' + ', '.join(passed) if passed else 'FAIL -- no axis clears the bar'}")

    _plots(stats, viol, f_mean, sigma, legal_max)
    (OUT / "v1_verdict.json").write_text(json.dumps(
        {"sigma": sigma, "sigma_seed": s_seed, "sigma_jitter": s_jit, "floor_mean": f_mean,
         "stats": stats, "violations": [{"kind": r["param"], "surprise": r["surprise"]}
                                        for r in viol],
         "controls": {"violations_pass": v_ok, "null_pass": bool(n_ok)},
         "verdict": verdict}, indent=1))
    print(f"\nwrote {OUT}/v1_verdict.json")


def _plots(stats, viol, f_mean, sigma, legal_max):
    axes = [p for p in ("bounce", "settle", "tilt", "null") if p in stats]
    fig, axs = plt.subplots(1, len(axes) + 1, figsize=(3.4 * (len(axes) + 1), 3.9))
    for ax, probe in zip(axs, axes):
        st = stats[probe]
        v, y = np.array(st["values"]), np.array(st["surprise"])
        ax.axhspan(f_mean - 3 * sigma, f_mean + 3 * sigma, color="#e9e4d8", zorder=0,
                   label="noise floor  (+-3 sigma)")
        ax.plot(v, y, "o-", color="#b4451f" if probe != "null" else "#7a7a7a", ms=4)
        ax.axvline(st["ref"], color="#2f6f8f", ls="--", lw=1, label="reference theta")
        ax.set_xscale("log")
        ax.set_xlabel(LABEL.get(st["param"], st["param"]))
        ax.set_title(f"{probe}\nspread {st['spread_sigma']:.1f} sigma, "
                     f"rho {st['rho_vs_dist']:+.2f}", fontsize=9)
        if probe == axes[0]:
            ax.set_ylabel("surprise   (mean causal L1)")
            ax.legend(fontsize=7, frameon=False)
    bx = axs[-1]
    names = [r["param"] for r in viol]
    vals = [r["surprise"] for r in viol]
    bx.barh(names, vals, color="#b4451f")
    bx.axvline(legal_max, color="#2f6f8f", ls="--", lw=1)
    bx.text(legal_max, -0.6, " highest legal theta", color="#2f6f8f", fontsize=7)
    bx.axvline(f_mean, color="#666", ls=":", lw=1)
    lo = min(vals + [f_mean, legal_max]); hi = max(vals + [f_mean, legal_max])
    bx.set_xlim(lo - 0.06 * (hi - lo + 1e-9), hi + 0.1 * (hi - lo + 1e-9))
    bx.set_title("gross violations\n(control: these must fire)", fontsize=9)
    fig.suptitle("V1 -- does surprise depend on theta?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "v1_surprise_vs_theta.png", dpi=150)
    print(f"wrote {OUT}/v1_surprise_vs_theta.png")


if __name__ == "__main__":
    main()
