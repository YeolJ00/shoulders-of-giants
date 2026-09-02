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

import os as _os
OUT = Path("docs/results") / ("v1_protocol_"
                              + _os.environ.get("SURPRISE_MODEL", "vitl").split("/")[-1]
                              + (f"_f{_os.environ['CLIP_FRAMES']}"
                                 if _os.environ.get("CLIP_FRAMES") else "") + ".json")
# context lengths in temporal slots; the reference sweeps this axis and reduces over it
CTXT = tuple(range(2, 31, 2))     # trimmed to the clip's depth in prefix_losses


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def object_patches(m, lab, key, box, cam):
    """Which spatial patches the OBJECT occupies, per temporal slot.

    A full-frame target averages the object against a static table: at 16x16 patches a
    bowl covers maybe 20 of 256, so 92% of the reading is scenery that cannot depend on
    theta. It does not add noise, it adds a large constant -- but it shrinks the
    fractional signal we are trying to resolve against a 0.29% floor. The box comes from
    the same projection the crop uses, so it costs nothing and is exact.
    """
    from src.surprise.regions import box_to_spatial_tokens
    pts = np.asarray(json.loads((lab / "meta.json").read_text())[key], float)
    x0, y0, x1, y1 = box
    sc = m.size / float(x1 - x0)          # crop is square and resized to m.size
    per_slot = []
    for sidx in range(m.depth):
        f0 = sidx * m.tubelet
        sel = pts[min(f0, len(pts) - 1): min(f0 + m.tubelet, len(pts))]
        uv = []
        for p in sel:
            u, d = cam.project(p)
            uv.append(u[d > 0])
        uv = np.concatenate([a for a in uv if len(a)]) if any(len(a) for a in uv) else None
        if uv is None:
            per_slot.append(np.arange(m.tps)); continue
        b = (uv[:, 0].min() - x0, uv[:, 1].min() - y0,
             uv[:, 0].max() - x0, uv[:, 1].max() - y0)
        sp = box_to_spatial_tokens([v * sc for v in b], (1.0, 0.0, 0.0),
                                   grid=m.grid, patch=m.patch, dilate=1)
        per_slot.append(sp if len(sp) else np.arange(m.tps))
    return per_slot


def prefix_losses_by_slot(m, toks, ctxt=CTXT):
    """Loss per TARGET SLOT, not averaged over the whole future.

    The reference reduces with `.mean((2,3))` over all target tokens, which suits IntPhys:
    its violations are sustained, so a vanished object stays vanished and shifts every
    term. Ours are not all like that. The authored teleport moves ONE slot out of 30 and
    its per-slot spike measures ~100x the noise floor, yet it scores worst of the five
    (1/6) because averaging over the whole future dilutes it in every term -- while
    `freeze` and `through`, which change the entire remaining trajectory, score 5/6.

    So this returns the (context_length, slot) grid and lets the caller take a max over
    TIME as well as over context length.
    """
    out = []
    for n in ctxt:
        if not 0 < n < m.depth:
            continue
        tgt = m.slot_tokens(n, m.depth)
        p, g = m.predict(toks, m.slot_tokens(0, n), tgt)
        d = (p - g).abs().mean(-1)[0]                      # per target token
        out.append(d.reshape(m.depth - n, m.tps).mean(-1).cpu().numpy())   # per slot
    return out


def prefix_losses(m, toks, ctxt=CTXT, per_slot=None):
    """One L1 per context length -- the reference's `all_losses_ctxt`.

    per_slot restricts the TARGET to the object's own patches at each future slot; the
    context always stays the full prefix, so the model still sees the whole scene.
    """
    out = []
    for n in ctxt:
        if not 0 < n < m.depth:
            continue
        if per_slot is None:
            tgt = m.slot_tokens(n, m.depth)
        else:
            tgt = np.concatenate([s * m.tps + per_slot[s] for s in range(n, m.depth)])
        p, g = m.predict(toks, m.slot_tokens(0, n), tgt)
        out.append(float((p - g).abs().mean()))
    return np.asarray(out)


def main():
    lab = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/scene/v1")
    from src.surprise import clips as C
    from src.surprise.vjepa import SurpriseModel

    spec = {r["key"]: r for r in json.loads((lab / "spec.json").read_text())}
    sys.path.insert(0, str(REPO / "scripts"))
    import v1_killtest as V
    from t0_tracking import pulled_camera
    cam = pulled_camera(V.CAM_PULL)
    boxes = V.shared_boxes(cam, json.loads((lab / "meta.json").read_text()), spec, 544, 448)
    import os
    m = SurpriseModel(model_id=os.environ.get("SURPRISE_MODEL",
                                              "facebook/vjepa2-vitl-fpc64-256"),
                      device="cuda:0")
    nf = int(os.environ.get("CLIP_FRAMES", m.frames))
    if nf != m.frames:
        m.set_clip_frames(nf)
    print(f"{m.model_id}: {m.grid}x{m.grid} patches, {m.clip_frames} frames "
          f"({24.0 * nf / 64:.1f} fps effective), {m.depth} slots, {m.n_tokens} tokens",
          flush=True)
    rows = {}
    for key in sorted(spec):
        clip = lab / "clips" / f"{key}.mp4"
        if not clip.exists():
            continue
        frames, _ = C.prepare(clip, m.clip_frames, m.size)
        toks = m.encode(frames)
        L = prefix_losses(m, toks)
        S = prefix_losses_by_slot(m, toks)
        peak = float(max(a.max() for a in S))              # max over context AND time
        rec = {**spec[key], "losses": L.tolist(),
               "max": float(L.max()), "mean": float(L.mean()), "peak": peak}
        try:
            ps = object_patches(m, lab, key, boxes[key], cam)
            O = prefix_losses(m, toks, per_slot=ps)
            rec.update(obj_max=float(O.max()), obj_mean=float(O.mean()),
                       obj_patches=int(np.median([len(x) for x in ps])))
        except Exception as e:
            print("   object-restricted reading unavailable:", e)
        rows[key] = rec
        print(f"  {key:<22} max {L.max():.5f}  mean {L.mean():.5f} peak {peak:.5f}"
              + (f"   object max {rec['obj_max']:.5f} ({rec['obj_patches']} patches)"
                 if "obj_max" in rec else ""), flush=True)

    # noise floor for THESE statistics: re-renders of one identical rollout with only a
    # Cycles seed or a 3 mm camera shift changed. Without it, "+0.0068" is not a number.
    floor = []
    for d in sorted(Path(str(lab) + "_floor").glob("*")):
        for clip in sorted((d / "clips").glob("*.mp4")):
            frames, _ = C.prepare(clip, m.clip_frames, m.size)
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
