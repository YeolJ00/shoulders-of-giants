"""L1 -- real-data audit (PHASE1_AMENDMENT.md section 8): in real manipulation recordings, what
fraction of the frames in which an object MOVES is it free of the hand (sliding, falling,
settling, toppling)? Claim 3 is tested on free dynamics; if hand-object sets have almost none,
claim 3 needs a targeted capture or an explicit hand-object scope.

    $L scripts/l1_real_audit.py oakink2 [--sequences 12]

OakInk2 (HuggingFace kelvin34501/OakInk-v2, anno_preview/*.pkl, not gated): object poses per
mocap frame (120 Hz) and MANO hand parameters. Without the (licensed) MANO model the hand is
represented by its root translation: an object is called hand-free when both wrists are > 25 cm
from its origin -- conservative, since a hand cannot reach 25 cm past its wrist. Moving = speed
above 2 cm/s, measured over 4 frames.
"""
import json
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/real/raw_oakink2"
API = "https://huggingface.co/api/datasets/kelvin34501/OakInk-v2/tree/main/anno_preview"
URL = "https://huggingface.co/datasets/kelvin34501/OakInk-v2/resolve/main/"


def oakink2(n_seq=12, seed=0):
    RAW.mkdir(parents=True, exist_ok=True)
    files = sorted(x["path"] for x in json.loads(urllib.request.urlopen(API, timeout=60).read()))
    pick = [files[i] for i in np.random.default_rng(seed).choice(len(files), n_seq, replace=False)]
    rows = []
    for p in pick:
        dst = RAW / Path(p).name
        if not dst.exists():
            urllib.request.urlretrieve(URL + p, dst)
        d = pickle.load(open(dst, "rb"))
        frames = d["mocap_frame_id_list"]
        wr = np.array([[np.asarray(d["raw_mano"][f][h]).reshape(3) for h in ("rh__tsl", "lh__tsl")] for f in frames])
        for o in d["obj_list"]:
            T = np.array([np.asarray(d["obj_transf"][o][f]) for f in frames])
            pos = T[:, :3, 3]
            v = np.linalg.norm(pos[4:] - pos[:-4], axis=1) * 120.0 / 4.0
            moving = v > 0.02
            far = np.linalg.norm(wr[4:] - pos[4:, None], axis=2).min(1) > 0.25
            rows.append({"seq": Path(p).stem[:40], "object": o, "frames": len(v), "moving": int(moving.sum()),
                         "moving_hand_free": int((moving & far).sum())})
    mv = sum(r["moving"] for r in rows)
    free = sum(r["moving_hand_free"] for r in rows)
    return {"dataset": "OakInk2 (anno_preview sample)", "sequences": n_seq, "objects": len(rows),
            "moving_frames": mv, "moving_hand_free_frames": free, "free_fraction": free / max(mv, 1), "rows": rows}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "oakink2"
    n = int(sys.argv[sys.argv.index("--sequences") + 1]) if "--sequences" in sys.argv else 12
    out = REPO / "docs/results/l1_real_audit.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    if which == "oakink2":
        res["oakink2"] = oakink2(n)
        r = res["oakink2"]
        print(f"OakInk2: {r['sequences']} sequences, {r['objects']} objects; moving frames {r['moving_frames']}, "
              f"hand-free while moving {r['moving_hand_free_frames']} -> free fraction {r['free_fraction']:.3f}")
    out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
