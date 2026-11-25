"""L1 -- Phase 1 gate on generated shards (PHASE1_AMENDMENT.md section 7).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l1_gate.py --data data/datasets/v0 --shard 0 [--episodes 128]

Identifiability is measured in the information setting the model is trained in: a perturbation
counts only if something the model sees changes while everything it is GIVEN is held fixed.
Replays make the gripper follow its RECORDED pose (stiff PD), because the pose is always given.
  H -- force slot hidden (30% of gripper contacts; real hands): identified if the object's
       motion changes.
  G -- force slot given (70%): identified if the motion OR the gripper's contact force on the
       object changes (with the path held, a heavier object needs a larger push).
Pokes are hidden in both: only the velocity change they cause is defined, so a mass-perturbed
object receives a mass-scaled poke, and a poke cannot identify mass. (An earlier version tested
G by replaying the recorded gripper forces open loop; once an object strays, the force keeps
pushing where the gripper no longer is, and replay noise reached p90 147 mm, max 79 m.)

1 Noise: the replay twice; per-episode thresholds 5 x max(1.5 mm, motion noise) and
  5 x max(0.01 N s, force noise), force measured as the impulse of the difference.
2 Identifiability (rule 1 applied to the dataset): ONE parameter of ONE object -- a uniformly
  random one, since section 3 asks it of every dynamic object -- perturbed by its prior's
  half-width: mu x or / 3.46, restitution +-0.4, mass x or / 4, centre of mass moved 12.5% of the
  object's size, joint damping x or / sqrt(range). Divergence = that object's (or the hinged
  part's) largest position difference over the episode.
3 Coverage: fraction of probed objects identified in G (targets >= 0.6 for mu, e, mass, CoM;
  >= 0.3 for joint damping, over hinged episodes that use the hinge); H is reported as the
  hand-object case.
4 Per family that touched the probed object: identified fraction (> 0.5 = identifying).
5 Free-dynamics fraction, family fractions vs the amendment, per-object interactions.
6 Invariants: finite, speeds bounded, nothing below the floor.
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/results/l1_gate.json"
FAMILY_TARGET = {"push": 0.6, "off_centre": 0.4, "tilt_release": 0.3, "strike": 0.3, "collision": 0.3,
                 "sweep": 0.15, "random": 0.2}


def touching(wd, k):
    fams = {f["family"] for f in wd["families"] if f.get("target") == k or f.get("other") == k}
    o = wd["objects"][k]
    if o["drop_height"] > 0:
        fams.add("drop")
    if any(p["object"] == k for p in wd["pokes"]):
        fams.add("poke")
    if o["on_top_of"] >= 0 or any(x["on_top_of"] == k for x in wd["objects"]):
        fams.add("stack")
    if wd["spec"][k].get("on_ramp", -1) >= 0:
        fams.add("ramp")
    return fams or {"none"}


def interactions(meta, d):
    """Per object of the recording: gripper impulse > 0.02 N s, dropped, poked, in contact with
    another object while either moves > 5 cm/s for >= 3 frames."""
    idx, F, qd = d["contact_idx"].astype(int), d["contact_F"], d["body_qd"]
    rows = []
    for w, wd in enumerate(meta["worlds"]):
        n = len(wd["objects"])
        for k in range(n):
            b = wd["object_bodies"][k]
            g = (idx[:, 1] == w) & (idx[:, 2] == 2 + k) & (idx[:, 3] == 1)
            oo = (idx[:, 1] == w) & (idx[:, 2] == 2 + k) & (idx[:, 3] >= 2) & (idx[:, 3] < 2 + n)
            moving = 0
            for f, j in idx[oo][:, [0, 3]]:
                ob = wd["object_bodies"][j - 2]
                moving += max(np.linalg.norm(qd[f + 1, b, :3]), np.linalg.norm(qd[f + 1, ob, :3])) > 0.05
            rows.append({"grip": bool(np.linalg.norm(F[g], axis=1).sum() / 30.0 > 0.02),
                         "drop": wd["objects"][k]["drop_height"] > 0,
                         "poke": any(p["object"] == k for p in wd["pokes"]), "object_object": bool(moving >= 3)})
    out = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    out["any"] = float(np.mean([any(r.values()) for r in rows]))
    out["objects"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/v0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=128)
    a = ap.parse_args()
    import warp as wp
    from lsim.data import library
    from lsim.data.episode import KD_TRACK, KP_TRACK, rotate, run_batch
    from lsim.data.scene import sample_scene
    wp.init()
    ds = REPO / a.data
    d = np.load(ds / f"shard_{a.shard:05d}.npz")
    meta = json.loads((ds / f"shard_{a.shard:05d}.json").read_text())
    train = json.loads((ds / "assets_split.json").read_text())["train"]
    rng = np.random.default_rng(meta["seed"])
    W, K = len(meta["worlds"]), meta["K"]
    p_ramp = float(meta.get("p_ramp", 0.0))
    specs = [sample_scene(rng, train, seed=meta["seed"] * 1000 + k, p_ramp=p_ramp) for k in range(W)]
    for sp, wd in zip(specs, meta["worlds"]):                     # the regenerated scenes are the recorded ones
        assert [o.name for o in sp.objects] == [o["name"] for o in wd["spec"]]
    worlds = [w for w in range(W) if meta["worlds"][w]["objects"]][:a.episodes]
    prng = np.random.default_rng(0)
    probe = {w: int(prng.integers(len(meta["worlds"][w]["objects"]))) for w in worlds}
    rec_q, rec_qd = d["body_q"], d["body_qd"]
    nf = rec_q.shape[0] - 1
    cidx = d["contact_idx"].astype(int)

    def replay_of(w):
        wd, h, gb = meta["worlds"][w], meta["worlds"][w]["hinge"], meta["worlds"][w]["gripper_body"]
        return {"keyframes": wd["gripper_keyframes"], "pokes": wd["pokes"], "families": wd["families"],
                "families_sampled": wd["families_sampled"], "cap": 500.0,
                "track": np.concatenate([rec_q[:, gb], rec_qd[:, gb]], 1),
                "hinge": (h["mode"], h["q0"]) if h else None, "hinge_used": "hinge" in wd["families_sampled"]}

    def run(ws, sps):
        return run_batch(sps, seed=meta["seed"], replay=[replay_of(w) for w in ws], log=lambda m: None,
                         kp=KP_TRACK, kd=KD_TRACK)

    def grip_force(res, i, slot):
        """Force on `slot` by the gripper per frame; slot -1 = the hinged part."""
        return res["contact_F"][:, i, res["K"] - 1 if slot < 0 else slot, 1]

    def traj(res, i, k):
        return res["body_q"][:, res["infos"][i]["objects"][k], :3]

    def hinge_traj(res, i):
        q = res["body_q"][:, res["infos"][i]["hinge"]["link"]]
        if res["infos"][i]["hinge"]["kind"] == "drawer":
            return q[:, :3]
        return q[:, :3] + np.stack([rotate(qq[3:], np.array([[0.0, -0.2, 0.0]]))[0] for qq in q])   # lid: 0.2 m out

    def div(x, y):
        return float(np.linalg.norm(x - y, axis=1).max())

    def fdiv(x, y):
        return float(np.linalg.norm(x - y, axis=1).sum() / 30.0)          # N s

    r1 = run(worlds, [copy.deepcopy(specs[w]) for w in worlds])
    r2 = run(worlds, [copy.deepcopy(specs[w]) for w in worlds])
    rec = np.array([div(traj(r1, i, probe[w]), rec_q[:, meta["worlds"][w]["object_bodies"][probe[w]], :3])
                    for i, w in enumerate(worlds)])
    noise = np.array([div(traj(r1, i, probe[w]), traj(r2, i, probe[w])) for i, w in enumerate(worlds)])
    fnoise = np.array([fdiv(grip_force(r1, i, 2 + probe[w]), grip_force(r2, i, 2 + probe[w])) for i, w in enumerate(worlds)])
    thr, fthr = 5.0 * np.maximum(1.5e-3, noise), 5.0 * np.maximum(0.01, fnoise)
    report = {"vs_recording_mm": {"median": 1e3 * float(np.median(rec)), "p90": 1e3 * float(np.percentile(rec, 90))},
              "noise_mm": {"median": 1e3 * float(np.median(noise)), "p90": 1e3 * float(np.percentile(noise, 90)),
                           "max": 1e3 * float(noise.max())},
              "force_noise_Ns": {"median": float(np.median(fnoise)), "p90": float(np.percentile(fnoise, 90))},
              "threshold_mm_median": 1e3 * float(np.median(thr)), "force_threshold_Ns_median": float(np.median(fthr))}
    print(f"replay vs recording: median {1e3*np.median(rec):.2f} mm, p90 {1e3*np.percentile(rec, 90):.1f} mm | "
          f"noise median {1e3*np.median(noise):.3f} mm, p90 {1e3*np.percentile(noise, 90):.2f} mm, max {1e3*noise.max():.1f} mm | "
          f"force noise median {np.median(fnoise):.4f} N s, p90 {np.percentile(fnoise, 90):.3f} N s", flush=True)

    prng = np.random.default_rng(1)
    results = {}
    for param in ("mu", "restitution", "com", "mass", "joint_damping"):
        pert, used = [], []
        for i, w in enumerate(worlds):
            sp = copy.deepcopy(specs[w]); o = sp.objects[probe[w]]; lab = meta["worlds"][w]["objects"][probe[w]]
            if param == "mu":
                o.params.mu = float(np.clip(o.params.mu * 3.46 if o.params.mu * 3.46 <= 1.2 else o.params.mu / 3.46, 0.1, 1.2))
            elif param == "restitution":
                o.params.restitution = o.params.restitution + 0.4 if o.params.restitution + 0.4 <= 0.85 else o.params.restitution - 0.4
            elif param == "mass":
                o.mass_scale = 4.0 if lab["mass"] * 4.0 <= 5.0 else 0.25
            elif param == "com":
                size = float(max(library.load(o.name)["extents"]))
                ang = prng.uniform(-np.pi, np.pi)
                o.com_shift = (0.125 * size * np.cos(ang), 0.125 * size * np.sin(ang), 0.0)
            elif param == "joint_damping":
                if sp.hinge is None or "hinge" not in meta["worlds"][w]["families_sampled"]:
                    continue
                f = np.sqrt(40.0) if sp.hinge.kind == "drawer" else 10.0
                hi = 20.0 if sp.hinge.kind == "drawer" else 0.5
                sp.hinge.damping = sp.hinge.damping * f if sp.hinge.damping * f <= hi else sp.hinge.damping / f
            pert.append(sp); used.append(i)
        if not used:
            results[param] = {"episodes": 0}
            continue
        res = run([worlds[i] for i in used], pert)
        dv, fv, fams = [], [], []
        for j, i in enumerate(used):
            w = worlds[i]
            if param == "joint_damping":
                dv.append(div(hinge_traj(res, j), hinge_traj(r1, i))); fams.append({"hinge"})
                fv.append(fdiv(grip_force(res, j, -1), grip_force(r1, i, -1)))
            else:
                dv.append(div(traj(res, j, probe[w]), traj(r1, i, probe[w])))
                fv.append(fdiv(grip_force(res, j, 2 + probe[w]), grip_force(r1, i, 2 + probe[w])))
                fams.append(touching(meta["worlds"][w], probe[w]))
        dv, fv = np.array(dv), np.array(fv)
        okH = dv > thr[used]
        okG = okH | (fv > fthr[used])
        by_fam = defaultdict(list)
        for x, gH, gG, fs in zip(dv, okH, okG, fams):
            for f in fs:
                by_fam[f].append((x, gH, gG))
        table = {f: {"n": len(v), "median_mm": 1e3 * float(np.median([x for x, _, _ in v])),
                     "identified_G": float(np.mean([g for _, _, g in v])), "identified_H": float(np.mean([g for _, g, _ in v])),
                     "identifying_G": bool(np.mean([g for _, _, g in v]) > 0.5)} for f, v in sorted(by_fam.items())}
        results[param] = {"episodes": len(used), "median_mm": 1e3 * float(np.median(dv)),
                          "median_force_Ns": float(np.median(fv)), "coverage": float(okG.mean()),
                          "coverage_H": float(okH.mean()), "by_family": table}
        print(f"COVERAGE {param:14s} G {okG.mean():.2f}  H {okH.mean():.2f}  n {len(used):3d}  median {1e3*np.median(dv):6.1f} mm, "
              f"{np.median(fv):.3f} N s | " + ", ".join(f"{f} {t['identified_G']:.2f}/{t['identified_H']:.2f} (n {t['n']})"
                                                       for f, t in table.items()), flush=True)

    # free-dynamics fraction, family fractions and invariants, on the recording itself
    touched = set(map(tuple, cidx[(cidx[:, 3] == 1) & (cidx[:, 2] >= 2)][:, :3].tolist()))
    total = sum(len(wd["objects"]) for wd in meta["worlds"]) * nf
    free = 1.0 - len(touched) / max(total, 1)
    n_obj = [len(wd["objects"]) for wd in meta["worlds"]]
    fam_frac = {}
    for f, target in FAMILY_TARGET.items():
        elig = [wd for wd, n in zip(meta["worlds"], n_obj) if n >= (2 if f == "collision" else 1)]
        fam_frac[f] = {"fraction": float(np.mean([any(x["family"] == f for x in wd["families"]) for wd in elig])),
                       "target": target}
    hinged = [wd for wd in meta["worlds"] if wd["hinge"] is not None]
    fam_frac["hinge"] = {"fraction": float(np.mean([any(x["family"] == "hinge" for x in wd["families"]) for wd in hinged]))
                         if hinged else 0.0, "target": 0.3}
    inter = interactions(meta, d)
    v = np.linalg.norm(d["body_qd"][:, :, :3], axis=-1)
    inv = {"finite": bool(np.isfinite(rec_q).all()), "max_speed_m_s": float(v.max()),
           "lowest_z_m": float(rec_q[:, :, 2].min()), "above_floor": bool(rec_q[:, :, 2].min() > -0.80),
           "max_xy_m": float(np.abs(rec_q[:, :, :2]).max()), "on_floor_plate": bool(np.abs(rec_q[:, :, :2]).max() < 10.0)}
    print(f"free-dynamics fraction {free:.2f} (target >= 0.5); invariants {inv}", flush=True)
    print("family fractions: " + ", ".join(f"{f} {v['fraction']:.2f}/{v['target']}" for f, v in fam_frac.items()), flush=True)
    print("per-object interactions: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                  for k, v in inter.items()), flush=True)
    out = {"data": a.data, "shard": a.shard, "episodes": len(worlds), "probe": "uniformly random object",
           "replay": report, "params": results, "coverage_is": "G (force slot given): motion or gripper force changes; "
                                                               "coverage_H: motion changes", "free_dynamics_fraction": free, "family_fractions": fam_frac,
           "object_interactions": inter, "invariants": inv,
           "targets": {"coverage_mu_e_mass_com": 0.6, "coverage_joint_damping": 0.3, "free_dynamics": 0.5}}
    OUT.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
