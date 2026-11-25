"""L2 -- baseline B3: the teacher itself with prior-mean parameters (what a conventional simulator
gives when an object's parameters are unknown), on held-out episodes.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l2_teacher_baseline.py --data data/datasets/v0_test --shard 0

Every object becomes a uniform solid at the prior's geometric-mean density (387 kg/m^3) with
mu 0.346 and e 0.45; hinges get the prior means too. The episode is replayed with the gripper
following its recorded pose (as in the gate), and each body's mean vertex error is measured at
1, 8, 30 frames after frame 8 and at the end -- the horizons l2_eval.py reports for the model.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/v0_test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--out", default="docs/results/l2_teacher_baseline.json")
    a = ap.parse_args()
    import warp as wp
    from lsim.data import library
    from lsim.data.episode import KD_TRACK, KP_TRACK, rotate, run_batch
    from lsim.data.params import ObjectParams
    from lsim.data.scene import sample_scene
    wp.init()
    ds = REPO / a.data
    d = np.load(ds / f"shard_{a.shard:05d}.npz")
    meta = json.loads((ds / f"shard_{a.shard:05d}.json").read_text())
    split = json.loads((ds / "assets_split.json").read_text())
    names = split["test"] if a.data.rstrip("/").endswith("_test") else split["train"]
    rng = np.random.default_rng(meta["seed"])
    specs = [sample_scene(rng, names, seed=meta["seed"] * 1000 + k) for k in range(len(meta["worlds"]))]
    worlds = [w for w in range(len(specs)) if meta["worlds"][w]["objects"]][:a.episodes]
    rec_q, rec_qd = d["body_q"], d["body_qd"]
    pert = []
    for w in worlds:
        sp = copy.deepcopy(specs[w])
        assert [o.name for o in sp.objects] == [o["name"] for o in meta["worlds"][w]["spec"]]
        for o in sp.objects:
            o.params = ObjectParams(mu=float(np.sqrt(0.1 * 1.2)), restitution=0.45, density=float(np.sqrt(50.0 * 3000.0)),
                                    build="solid", wall=0.0, regions=[])
        if sp.hinge is not None:
            sp.hinge.damping = float(np.sqrt(0.5 * 20.0)) if sp.hinge.kind == "drawer" else float(np.sqrt(0.005 * 0.5))
            sp.hinge.mu, sp.hinge.restitution = 0.4, 0.275
        pert.append(sp)

    def replay_of(w):
        wd, h, gb = meta["worlds"][w], meta["worlds"][w]["hinge"], meta["worlds"][w]["gripper_body"]
        return {"keyframes": wd["gripper_keyframes"], "pokes": wd["pokes"], "families": wd["families"],
                "families_sampled": wd["families_sampled"], "cap": 500.0,
                "track": np.concatenate([rec_q[:, gb], rec_qd[:, gb]], 1),
                "hinge": (h["mode"], h["q0"]) if h else None, "hinge_used": "hinge" in wd["families_sampled"]}
    res = run_batch(pert, seed=meta["seed"], replay=[replay_of(w) for w in worlds], log=lambda m: None, kp=KP_TRACK, kd=KD_TRACK)
    t0 = 8
    frames = [t0 + 1, t0 + 8, t0 + 30, rec_q.shape[0] - 1]
    errs = []
    for i, w in enumerate(worlds):
        wd = meta["worlds"][w]
        for k, ob in enumerate(wd["objects"]):
            v = library.load(ob["name"])["points_v"].astype(float)
            b_rec, b_sim = wd["object_bodies"][k], res["infos"][i]["objects"][k]
            errs.append([np.linalg.norm(rotate(res["body_q"][f, b_sim, 3:], v) + res["body_q"][f, b_sim, :3]
                                        - rotate(rec_q[f, b_rec, 3:], v) - rec_q[f, b_rec, :3], axis=1).mean() for f in frames])
    E = 1000 * np.array(errs)
    out = {"data": a.data, "shard": a.shard, "episodes": len(worlds), "bodies": len(E), "horizons_frames": [1, 8, 30, frames[-1] - t0],
           "median_mm": np.median(E, 0).round(2).tolist(), "p90_mm": np.percentile(E, 90, 0).round(1).tolist()}
    print(f"B3 teacher with prior-mean parameters ({len(E)} bodies): error at +1/+8/+30/end frames from frame {t0}: "
          f"median {out['median_mm']} mm, p90 {out['p90_mm']} mm", flush=True)
    (REPO / a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
