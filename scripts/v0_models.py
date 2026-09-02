"""Which scoring model? Same gates, same clips, side by side.

Rule 5: any change to the scoring model invalidates calibration. So a model swap is not
a config edit, it is a re-run of V0 plus a fresh noise floor. This script is the cheap
half -- the discrimination comparison that decides whether a swap is worth the floor.

V-JEPA **2.1** is deliberately absent, and not by oversight. Meta's 2.1 weights exist
(`vjepa2_1_vitb_dist_vitG_384.pt`, ported to `apiantonio/vjepa2.1-vit-base-384`), but
that checkpoint is a ViT-B *distilled from ViT-G*, and its predictor projects into the
TEACHER's space: `pred_teacher_embed_dim = 1664` against an encoder `hidden_size` of 768.
Its model output has no `target_hidden_state` at all. So the predicted and actual
representations live in different spaces and the prediction error we score is not
computable from that checkpoint alone -- it needs the 1664-wide teacher encoder, and no
published V-JEPA 2 encoder is that wide (ViT-g is 1408, ViT-H 1280, ViT-L 1024). The 2.1
port is usable for features and classification; it cannot produce surprise.

    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/v0_models.py [--paired]

`--paired` is the version to believe. The two-reference-clip comparison is enough to
disqualify a model that gets the sign wrong, but not enough to rank models: it is n = 2.
Paired mode scores EVERY V1 clip under both models and compares the shuffled-vs-real
margin clip by clip, which is a paired test on ~40 samples of exactly the footage we care
about.

Note what is NOT comparable across models: "beats zero-prediction by X%". Each model's
target is layer-normed in its OWN representation space, and a wider representation is
intrinsically harder to predict, so that number ranks nothing. Only within-model
contrasts -- real vs shuffled, scored in the same space -- can be compared between models.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization")

from src.surprise import clips as C                                    # noqa: E402
from src.surprise.vjepa import DEPTH, CAUSAL_START, slot_tokens        # noqa: E402

# Both are crop 256 / patch 16 / tubelet 2 / 64 frames, so the token grid is identical
# (8192) and src/surprise/vjepa.py's constants hold for both without change.
MODELS = ["facebook/vjepa2-vitl-fpc64-256", "facebook/vjepa2-vitg-fpc64-256"]
OUT = Path("docs/results/v0_models.json")


def paired():
    """Shuffled-vs-real margin per clip, under each model. A paired test."""
    from src.surprise.vjepa import SurpriseModel

    paths = sorted(Path("outputs/scene/v1/clips").glob("*.mp4"))
    if not paths:
        sys.exit("no clips under outputs/scene/v1/clips -- run the clip stage first")
    frames = {p.stem: C.prepare(p)[0] for p in paths}
    print(f"{len(frames)} clips, paired across {len(MODELS)} models")
    out = {}
    for mid in MODELS:
        m = SurpriseModel(model_id=mid, device="cuda:0")
        rows = {}
        for k, f in frames.items():
            rng = np.random.default_rng(0)
            real = m.score(f)["causal"]["mean"]
            shuf = m.score(f[rng.permutation(len(f))])["causal"]["mean"]
            rows[k] = 100 * (shuf / real - 1)
            print(f"  {mid.split('-')[1]:5s} {k:<24} margin {rows[k]:+6.2f}%", flush=True)
        out[mid] = rows
        del m
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    a, b = [out[m] for m in MODELS]
    keys = sorted(set(a) & set(b))
    da = np.array([a[k] for k in keys]); db = np.array([b[k] for k in keys])
    wins = int((da > db).sum())
    print(f"\n{MODELS[0]}: mean margin {da.mean():+.2f}%  ({int((da > 0).sum())}/{len(keys)} correct sign)")
    print(f"{MODELS[1]}: mean margin {db.mean():+.2f}%  ({int((db > 0).sum())}/{len(keys)} correct sign)")
    print(f"paired: the first model has the larger margin on {wins}/{len(keys)} clips")
    Path("docs/results/v0_models_paired.json").write_text(json.dumps(
        {"per_clip": out, "summary": {"mean": {MODELS[0]: float(da.mean()),
                                              MODELS[1]: float(db.mean())},
                                      "correct_sign": {MODELS[0]: int((da > 0).sum()),
                                                       MODELS[1]: int((db > 0).sum())},
                                      "n": len(keys), "first_model_wins": wins}}, indent=1))
    print("wrote docs/results/v0_models_paired.json")


def main():
    from src.surprise.vjepa import SurpriseModel

    refs = {n: C.prepare(Path(f"outputs/surprise/ref_kinetics_{n}.mp4"))[0]
            for n in ("archery", "bowling")
            if Path(f"outputs/surprise/ref_kinetics_{n}.mp4").exists()}
    sweep = sorted(Path("outputs/scene/v1/clips").glob("bounce_cd*.mp4"))
    ours = {p.stem: C.prepare(p)[0] for p in sweep}
    print(f"{len(refs)} reference clips, {len(ours)} bounce clips")

    results = {}
    for mid in MODELS:
        print(f"\n=== {mid}")
        m = SurpriseModel(model_id=mid, device="cuda:0")
        r = {"hidden": int(m.model.config.hidden_size)}

        # 1. does it predict at all, and does temporal order help
        for name, f in refs.items():
            toks = m.encode(f)
            mod, zero = [], []
            for s in range(CAUSAL_START, DEPTH):
                p, g = m.predict(toks, slot_tokens(0, s), slot_tokens(s, s + 1))
                mod.append(float((p - g).abs().mean())); zero.append(float(g.abs().mean()))
            real = m.score(f)["causal"]["mean"]
            rng = np.random.default_rng(0)
            shuf = m.score(f[rng.permutation(len(f))])["causal"]["mean"]
            r[name] = {"beats_zero_pct": 100 * (1 - np.mean(mod) / np.mean(zero)),
                       "shuffled_vs_real_pct": 100 * (shuf / real - 1)}
            print(f"  {name:8s} beats zero by {r[name]['beats_zero_pct']:5.1f}%   "
                  f"shuffled vs real {r[name]['shuffled_vs_real_pct']:+6.2f}%", flush=True)

        # 2. the reading that actually matters: theta response on the bounce sweep,
        #    normalised by each clip's own shuffled copy
        if ours:
            vals, ratios = [], []
            for k, f in ours.items():
                rng = np.random.default_rng(0)
                a = m.score(f)["causal"]["mean"]
                b = m.score(f[rng.permutation(len(f))])["causal"]["mean"]
                vals.append(float(k.split("cd")[1])); ratios.append(a / b)
                print(f"  {k:<22} ratio {a / b:.4f}", flush=True)
            r["bounce"] = {"cd": vals, "ratio": ratios,
                           "ratio_spread": float(max(ratios) - min(ratios))}
            print(f"  bounce ratio spread {r['bounce']['ratio_spread']:.5f}")
        results[mid] = r
        del m
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    if "--paired" in sys.argv:
        paired()
    else:
        main()
