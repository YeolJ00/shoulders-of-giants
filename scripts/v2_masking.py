"""Many maskings per clip: is it more sensitive, and does it say WHERE?

Two questions, and the second is the one that matters for fitting:

  1. Does max-over-maskings detect violations a prefix mask misses? The prefix averages
     the whole future into one number, so a brief anomaly is diluted by construction --
     measured, the authored teleport scores 2/6 under a prefix mask despite a per-slot
     spike ~100x the noise floor.

  2. Does the ARGMAX mask point at the moment the clip actually goes wrong? We know the
     answer for authored violations because we authored them: teleport jumps at frame 32
     (slot 16), vanish removes frames 32-40 (slots 16-20), freeze and through begin at
     frame 21 (slot ~10). If the peak lands there, the signal localises in time -- and a
     localised signal is what assigns blame to a PARAMETER, since a peak at impact
     implicates contact damping and a peak during the slide implicates friction.

    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/v2_masking.py [LAB]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

LAB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/scene/v1")

# where each authored violation actually starts, in SLOTS (tubelet 2, so slot = frame/2).
# corrupt() in v1_killtest puts teleport at n//2, vanish at n//2..n//2+8, freeze and
# through at n//3 onward, and rewind everywhere.
TRUTH = {"teleport": 16.0, "vanish": 18.0, "freeze": 10.5, "through": 10.5, "rewind": None}


def main():
    from src.surprise import clips as C
    from src.surprise.masks import families
    from src.surprise.vjepa import SurpriseModel
    import v1_killtest as V
    from v1_protocol import object_patches
    from t0_tracking import pulled_camera

    spec = {r["key"]: r for r in json.loads((LAB / "spec.json").read_text())}
    cam = pulled_camera(V.CAM_PULL)
    boxes = V.shared_boxes(cam, json.loads((LAB / "meta.json").read_text()), spec, 544, 448)
    m = SurpriseModel(device="cuda:0")
    rows = {}
    for key in sorted(spec):
        clip = LAB / "clips" / f"{key}.mp4"
        if not clip.exists():
            continue
        frames, _ = C.prepare(clip, m.clip_frames, m.size)
        toks = m.encode(frames)
        try:
            ps = object_patches(m, LAB, key, boxes[key], cam)
        except Exception:
            ps = None
        rec = {**spec[key]}
        for fam, masks in families(m.depth, m.tps, per_slot=ps).items():
            vals, centres = [], []
            for _, ctx, tgt, centre in masks:
                p, g = m.predict(toks, ctx, tgt)
                vals.append(float((p - g).abs().mean())); centres.append(centre)
            vals = np.asarray(vals)
            j = int(np.argmax(vals))
            rec[f"{fam}_max"] = float(vals.max())
            rec[f"{fam}_mean"] = float(vals.mean())
            rec[f"{fam}_argmax_slot"] = float(centres[j])
            if fam == "block":
                rec["block_profile"] = [float(x) for x in vals[:m.depth - 2]]
        rows[key] = rec
        print(f"  {key:<34} prefix {rec['prefix_max']:.5f}  block {rec['block_max']:.5f}"
              f"  peak@slot {rec['block_argmax_slot']:.0f}", flush=True)
    dst = Path("docs/results") / f"v2_masking_{LAB.name}.json"
    dst.write_text(json.dumps({"model": m.model_id, "rows": rows}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
