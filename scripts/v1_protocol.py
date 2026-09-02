"""Re-score the V1 clips using the PUBLISHED intuitive-physics protocol.

Source: facebookresearch/jepa-intuitive-physics, evaluation_code/. Three things there
differ from what scripts/v1_killtest.py did, and the third invalidates its control:

  1. MASKING is a temporal PREFIX. src/masks/causal.py builds
     `mask[nctxt_frames:, :, :] = 0`, and evals/.../utils.py::get_time_masks splits
     `patch_idcs[:k]` / `patch_idcs[k:]`. V0 rejected the prefix regime, but tested it
     unfairly -- 16 frames of context predicting 48, a 3:1 horizon -- where the reference
     uses 1:1 on a 16-frame window.

  2. The STATISTIC is over CONTEXT LENGTH, not over time. eval.py accumulates one loss per
     context length into `all_losses_ctxt`, so `loss_real` is [N, n_context_lengths], and
     the readings are `loss.max(1)[0]` and `loss.mean(1)`. v1_killtest averaged over the
     28 time slots of a single context length -- a different axis entirely, and the one
     that demonstrably dilutes a localised violation.

  3. Evaluation is PAIRWISE on matched clips:
     `acc_pairwise_max = (loss_real.max(1)[0] < loss_fake.max(1)[0])`. Never an absolute
     score against a heterogeneous set.

Target normalisation (`F.layer_norm(h, (h.size(-1),))`) and the L1 distance already match.

    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/v1_protocol.py [LAB]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO))

OUT = Path("docs/results") / ("v1_protocol_" + __import__("os").environ.get(
    "SURPRISE_MODEL", "vitl").split("/")[-1] + ".json")
# context lengths in temporal slots; the reference sweeps this axis and reduces over it
CTXT = tuple(range(2, 31, 2))


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def prefix_losses(m, toks, ctxt=CTXT):
    """One L1 per context length -- the reference's `all_losses_ctxt`."""
    out = []
    for n in ctxt:
        if not 0 < n < m.depth:
            continue
        p, g = m.predict(toks, m.slot_tokens(0, n), m.slot_tokens(n, m.depth))
        out.append(float((p - g).abs().mean()))
    return np.asarray(out)


def main():
    lab = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/scene/v1")
    from src.surprise import clips as C
    from src.surprise.vjepa import SurpriseModel

    spec = {r["key"]: r for r in json.loads((lab / "spec.json").read_text())}
    import os
    m = SurpriseModel(model_id=os.environ.get("SURPRISE_MODEL",
                                              "facebook/vjepa2-vitl-fpc64-256"),
                      device="cuda:0")
    print(f"{m.model_id}: {m.grid}x{m.grid} patches, {m.depth} slots, "
          f"{m.n_tokens} tokens", flush=True)
    rows = {}
    for key in sorted(spec):
        clip = lab / "clips" / f"{key}.mp4"
        if not clip.exists():
            continue
        frames, _ = C.prepare(clip, m.frames, m.size)
        L = prefix_losses(m, m.encode(frames))
        rows[key] = {**spec[key], "losses": L.tolist(),
                     "max": float(L.max()), "mean": float(L.mean())}
        print(f"  {key:<22} max {L.max():.5f}  mean {L.mean():.5f}", flush=True)

    # noise floor for THESE statistics: re-renders of one identical rollout with only a
    # Cycles seed or a 3 mm camera shift changed. Without it, "+0.0068" is not a number.
    floor = []
    for d in sorted(Path(str(lab) + "_floor").glob("*")):
        for clip in sorted((d / "clips").glob("*.mp4")):
            frames, _ = C.prepare(clip, m.frames, m.size)
            L = prefix_losses(m, m.encode(frames))
            floor.append({"lab": d.name, "max": float(L.max()), "mean": float(L.mean())})
    fs = {}
    if len(floor) > 1:
        for stat in ("max", "mean"):
            fs[stat] = float(np.std([f[stat] for f in floor], ddof=1))
        print(f"\nnoise floor over {len(floor)} re-renders: "
              + "  ".join(f"sigma({k}) {v:.5f}" for k, v in fs.items()))

    src_key = next(k for k, r in rows.items()
                   if r["probe"] == "bounce" and abs(r["value"] - r["ref"]) / r["ref"] < 1e-6)
    src = rows[src_key]
    print(f"\n=== PAIRWISE violation control (reference protocol), source {src_key}")
    print(f"{'violation':<12}{'max':>10}{'vs src':>10}   {'mean':>10}{'vs src':>10}   verdict")
    n_ok_max = n_ok_mean = n = 0
    for k, r in sorted(rows.items(), key=lambda kv: -kv[1]["max"]):
        if r["probe"] != "violation":
            continue
        n += 1
        dmax, dmean = r["max"] - src["max"], r["mean"] - src["mean"]
        n_ok_max += dmax > 0; n_ok_mean += dmean > 0
        zs = f"{dmax / fs['max']:+.1f} sigma" if fs.get("max") else ""
        print(f"{r['param']:<12}{r['max']:>10.5f}{dmax:>+10.5f}   "
              f"{r['mean']:>10.5f}{dmean:>+10.5f}   "
              f"{'DETECTED' if dmax > 0 else 'missed':<9}{zs}")
    print(f"\npairwise accuracy, max statistic : {n_ok_max}/{n}")
    print(f"pairwise accuracy, mean statistic: {n_ok_mean}/{n}")

    print("\n=== theta sweeps under the reference statistic")
    for probe in ("bounce", "settle", "tilt", "null"):
        rs = sorted([r for r in rows.values() if r["probe"] == probe],
                    key=lambda r: r["value"])
        if len(rs) < 3:
            continue
        v = np.array([r["value"] for r in rs])
        dist = np.abs(np.log(v / rs[0]["ref"]))
        for stat in ("max", "mean"):
            y = np.array([r[stat] for r in rs])
            sp = (y.max() - y.min()) / fs[stat] if fs.get(stat) else float("nan")
            print(f"  {probe:<7} {stat:<5} spread {sp:6.1f} sigma   "
                  f"rho|log theta - log ref| {spearman(dist, y):+.3f}   "
                  f"rho(log theta) {spearman(np.log(v), y):+.3f}"
                  f"   min at {v[int(np.argmin(y))]:.3f} (ref {rs[0]['ref']:.3f})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"ctxt_slots": list(CTXT), "rows": rows,
                               "floor": floor, "floor_sigma": fs}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
