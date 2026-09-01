"""V0 -- validate the surprise harness before it is allowed to measure anything.

Run (cosmos env, on one polled GPU, released on exit):

    G=$(python src/gpu.py | tail -1 | sed 's/.*-> //')
    CUDA_VISIBLE_DEVICES=$G /home/jooyeolyun/anaconda3/envs/cosmos/bin/python \
        scripts/v0_surprise.py

Nothing here needs a new render: ~950 cached mp4s already sit under outputs/judge/.
Six gates, in order, each of which stops the work if it fails -- a harness that
cannot pass them cannot be interpreted.

  1. DETERMINISM   same clip twice must give the identical float (rule 7).
  2. TWO ROUTES    our encode()+predict() must reproduce the library's own
                   VJEPA2Model.forward(context_mask, target_mask) exactly.
                   Rule 2: check against an invariant, not a reference value.
  3. TOKEN ORDER   the masks mean nothing unless token index -> (t, h, w) is what
                   vjepa.py claims. Perturb a known frame window and a known pixel
                   block; assert EXACTLY the predicted ids move at the patch
                   embedding, which is purely local.
  4. PREDICTS      the model must beat predicting zero. Against a layer-normed
                   target, zero scores mean|target|; a predictor that has learned
                   nothing about this footage would tie with it.
  5. ORDERING      a frame-shuffled clip must read as MORE surprising than the real
                   one, on reference footage. Note what is NOT gated: a frozen clip is
                   not reliably LESS surprising -- one reference puts static below real
                   and the other above it. Unnatural stillness is odd in its own right.
  6. SPEED         within one clip, stretch the SAME content over more frames so only
                   its speed changes, and discrimination collapses monotonically. This
                   is measured PER CLIP: pooled across clips it vanishes, which is the
                   same across-content confound the probe table already warns about.
                   Sets the render design target, not a pass/fail bar.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.surprise import clips as C                                    # noqa: E402
from src.surprise.vjepa import (                                        # noqa: E402
    ALL_TOKENS, CAUSAL_START, DEPTH, GRID, N_TOKENS, TOKENS_PER_SLOT,
    SurpriseModel, slot_tokens,
)

DEFAULT_CLIP = "outputs/judge/yesregion/s0_tilt__brass_pot@a.mp4"
OUT = Path("docs/results/v0_surprise.json")
FIG = Path("docs/results/v0_motion_floor.png")

# Reference footage with real motion. The harness has to be shown working on video the
# model was trained for before a null result on our renders means anything.
REF = {
    "archery": "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/"
               "archery/-Qz25rXdMjE_000014_000024.mp4",
    "bowling": "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/"
               "bowling/-WH-lxmGJVY_000005_000015.mp4",
}
REF_DIR = Path("outputs/surprise")

# Clips to report the motion budget of, spanning what the probes actually produce.
OURS = ["docs/videos/bigscene_tilt_a.mp4",
        "outputs/judge/yesregion/s0_tilt__brass_pot@a.mp4",
        "outputs/judge/yesregion/s0_collide__wooden_bowl@a.mp4",
        "outputs/judge/yesregion/s0_drop__brass_pot@a.mp4"]


def reference_clip(name="archery"):
    """Cached; downloaded once. Not committed -- outputs/ is gitignored."""
    path = REF_DIR / f"ref_kinetics_{name}.mp4"
    if not path.exists():
        import urllib.request
        REF_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading reference footage -> {path}", flush=True)
        urllib.request.urlretrieve(REF[name], path)
    return path


def spearman(a, b):
    """Rank correlation. scipy is not installed in the scoring env and is not worth
    adding for one line -- rule 5 keeps that env pinned to what the model needs."""
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, float)))
        return order.astype(float)
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def gate(ok, name, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch

    frames, xf = C.prepare(args.clip)
    print(f"clip {args.clip}\n  -> {frames.shape} uint8, letterbox scale={xf[0]:.4f} "
          f"offset=({xf[1]:.0f},{xf[2]:.0f})", flush=True)

    m = SurpriseModel(device=args.device)
    print(f"model {m.model_id} dtype={m.dtype}\nconfig {json.dumps(m.config())}", flush=True)
    results, passed = {}, []

    # ---------------------------------------------------------------- 1. determinism
    print("\n1. determinism")
    t0 = time.time()
    a = m.score(frames)["causal"]
    dt = time.time() - t0
    b = m.score(frames)["causal"]
    passed.append(gate(a["mean"] == b["mean"] and a["curve"] == b["curve"],
                       "same clip twice", f"mean={a['mean']:.9f} vs {b['mean']:.9f}"))
    print(f"       one clip, full causal curve: {dt:.1f} s")
    results["determinism"] = {"a": a["mean"], "b": b["mean"], "identical": a["mean"] == b["mean"],
                              "seconds_per_clip": dt}

    # ------------------------------------------------------------------ 2. two routes
    print("\n2. two routes to one quantity")
    px = m._pixels(frames)
    ctx_ids, tgt_ids = slot_tokens(0, 8), slot_tokens(8, DEPTH)
    ctx = torch.as_tensor(ctx_ids, device=px.device).unsqueeze(0)
    tgt = torch.as_tensor(tgt_ids, device=px.device).unsqueeze(0)
    with torch.no_grad():
        ref = m.model(pixel_values_videos=px, context_mask=[ctx], target_mask=[tgt])
    ours, _ = m.predict(m.encode(frames), ctx_ids, tgt_ids)
    delta = float((ours - ref.predictor_output.last_hidden_state.float()).abs().max())
    passed.append(gate(delta == 0.0, "encode+predict == VJEPA2Model.forward",
                       f"max|delta| = {delta:.3g}"))
    results["two_routes"] = {"max_abs_delta": delta}

    # ----------------------------------------------------------------- 3. token order
    print("\n3. token index -> (t, h, w)")
    # the embeddings wrapper does the (B,T,C,H,W)->(B,C,T,H,W) permute the conv needs;
    # patch_embeddings alone would be handed 64 "channels"
    emb = m.model.encoder.embeddings
    with torch.no_grad():
        e0 = emb(px)

    f_t = frames.copy(); f_t[32:48] = 0            # frames [32,48) -> slots [16,24)
    with torch.no_grad():
        e_t = emb(m._pixels(f_t))
    moved = (e_t - e0).abs().amax(-1)[0] > 0
    want = torch.zeros_like(moved)
    want[torch.as_tensor(slot_tokens(16, 24), device=moved.device)] = True
    ok_t = bool((moved == want).all())
    passed.append(gate(ok_t, "temporal slots",
                       f"{int(moved.sum())} tokens moved, expected {int(want.sum())}"))

    f_s = frames.copy(); f_s[:, 0:32, 0:32] = 0    # pixels -> patch rows/cols 0..1
    with torch.no_grad():
        e_s = emb(m._pixels(f_s))
    moved_s = (e_s - e0).abs().amax(-1)[0] > 0
    spatial = np.array([r * GRID + c for r in range(2) for c in range(2)], np.int64)
    want_s = torch.zeros_like(moved_s)
    want_s[torch.as_tensor(slot_tokens(0, DEPTH, spatial), device=moved_s.device)] = True
    ok_s = bool((moved_s == want_s).all())
    passed.append(gate(ok_s, "spatial patches",
                       f"{int(moved_s.sum())} tokens moved, expected {int(want_s.sum())}"))

    # the patch embedding is local, so the two checks above are exact. The encoder is
    # not -- attention mixes every token -- so report how concentrated the disturbance
    # stays rather than asserting on it.
    with torch.no_grad():
        h0 = m.model(pixel_values_videos=px, skip_predictor=True).last_hidden_state
        ht = m.model(pixel_values_videos=m._pixels(f_t), skip_predictor=True).last_hidden_state
    d = (ht - h0).abs().mean(-1)[0]
    conc = float(d[want].mean() / d[~want].mean())
    print(f"       encoder disturbance inside/outside the blanked slots: {conc:.1f}x")
    results["token_order"] = {"temporal_ok": ok_t, "spatial_ok": ok_s,
                              "encoder_concentration": conc}

    # ---------------------------------------------------- 4. does it predict at all
    print("\n4. does the predictor beat predicting nothing")
    toks = m.encode(frames)
    model_l1, zero_l1 = [], []
    for s in range(CAUSAL_START, DEPTH):
        p, g = m.predict(toks, slot_tokens(0, s), slot_tokens(s, s + 1))
        model_l1.append(float((p - g).abs().mean()))
        zero_l1.append(float(g.abs().mean()))
    model_l1, zero_l1 = float(np.mean(model_l1)), float(np.mean(zero_l1))
    passed.append(gate(model_l1 < zero_l1, "model < zero-prediction",
                       f"{model_l1:.4f} vs {zero_l1:.4f} "
                       f"({100 * (1 - model_l1 / zero_l1):.1f}% better)"))
    results["beats_zero"] = {"model": model_l1, "zero": zero_l1}

    # ------------------------------------------------------------------- 5. ordering
    print("\n5. does the number respond to the dynamics (reference footage)")
    def variants(f):
        # a fresh generator per call: a shared one advances between calls, so the same
        # clip would get a different shuffle in gate 5 than in gate 6 (rule 7).
        rng = np.random.default_rng(0)
        return {"static": np.repeat(f[:1], len(f), axis=0), "real": f,
                "reversed": f[::-1].copy(), "shuffled": f[rng.permutation(len(f))]}

    order = {}
    for name in REF:
        f, _ = C.prepare(reference_clip(name))
        sc = {k: m.score(v)["causal"]["mean"] for k, v in variants(f).items()}
        order[name] = sc
        print(f"       {name} (motion {C.motion(f):.2f})")
        for k, v in sc.items():
            print(f"         {k:9s} {v:.5f}   ({100 * (v / sc['real'] - 1):+.2f}% vs real)")
    passed.append(gate(all(s_["shuffled"] > s_["real"] for s_ in order.values()),
                       "frame-shuffled reads as more surprising than real"))
    results["ordering_reference"] = order

    # --------------------------------------------------------------- 6. motion floor
    print("\n6. speed dependence, within each clip")
    print("   the SAME content stretched over 64 frames, so only its speed changes")
    sweep = []
    for name in REF:
        raw = C.read_frames(reference_clip(name))
        for n in (64, 48, 40, 32, 24, 16, 12, 8):
            f = C.letterbox(C.resample(raw[:n], C.FRAMES))[0]
            v = {k: m.score(x)["causal"]["mean"] for k, x in variants(f).items()}
            row = {"clip": name, "src_frames": n, "motion": C.motion(f), "real": v["real"],
                   "static_pct": 100 * (v["static"] / v["real"] - 1),
                   "shuffled_pct": 100 * (v["shuffled"] / v["real"] - 1)}
            sweep.append(row)
            print(f"       {name:8s} motion {row['motion']:5.2f}   "
                  f"static {row['static_pct']:+6.2f}%   "
                  f"shuffled {row['shuffled_pct']:+6.2f}%", flush=True)

    rho = {}
    for name in REF:
        rows = [r for r in sweep if r["clip"] == name]
        rho[name] = spearman([r["motion"] for r in rows],
                             [r["shuffled_pct"] for r in rows])
        print(f"       {name:8s} rho(motion, shuffled-vs-real) = {rho[name]:+.3f}")
    pooled = spearman([r["motion"] for r in sweep], [r["shuffled_pct"] for r in sweep])
    print(f"       pooled across both clips        = {pooled:+.3f}   "
          f"<- the across-content confound, in one number")
    passed.append(gate(all(v >= 0.7 for v in rho.values()),
                       "slowing a clip destroys its own discrimination",
                       " ".join(f"{k}={v:+.2f}" for k, v in rho.items())))
    results["motion_sweep"] = sweep
    results["motion_rho"] = {"per_clip": rho, "pooled": pooled}

    print("\n   what our own probes produce:")
    ours = {}
    for p_ in OURS:
        if Path(p_).exists():
            f, _ = C.prepare(p_)
            ours[p_] = C.motion(f)
            flag = "" if ours[p_] >= C.MOTION_TARGET else "   << below the design target"
            print(f"       {ours[p_]:5.2f}  {Path(p_).name}{flag}")
    results["our_clips"] = ours

    # ------------------------------------------------------------------- 7. profile
    print("\n7. per-slot causal curve (the clip given on the command line)")
    curve = m.score(frames)["causal"]["curve"]
    lo, hi = min(curve), max(curve)
    for i, v in enumerate(curve):
        bar = "#" * int(round(28 * (v - lo) / (hi - lo + 1e-12)))
        print(f"       slot {i + CAUSAL_START:2d} (frame {2 * (i + CAUSAL_START):2d}) "
              f"{v:.4f} {bar}")
    results["curve"] = curve

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"clip": args.clip, "model": m.model_id,
                               "config": m.config(), "results": results}, indent=1))
    print(f"\nwrote {OUT}")
    ok = all(passed)
    print(f"\nV0 {'PASS' if ok else 'FAIL'} ({sum(passed)}/{len(passed)} gates)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
