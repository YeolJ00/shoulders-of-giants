"""F5 -- MYRIAD: a motion prior that predicts PATHS, not one displacement.

FPT (f4) answers "where is this point at one fixed later moment" -- one arrow per point, no
time axis -- and on our ramps that arrow contained no downhill component at all (it copied
the camera poke). MYRIAD (CompVis, same repo) predicts point trajectories step by step from
a single image: `predict_simulate` samples each point's next position autoregressively, so
the output is a PATH per point, comparable frame by frame with the simulator's own.

Protocol, fixed before looking:
  * STATIC camera and camera_static=True. MYRIAD's poke format fixes positions up to a time
    for ALL tracks at once, so a moving camera cannot be given as a known background path;
    and a static camera removes the camera-copy confound that sank f4. The conditioning
    frame is f4's frame 0 (the orbit's first pose), used as an ordinary still view, and
    every simulated point is projected with that same fixed camera.
  * Background points are poked with ZERO motion (they are static -- true); the object's
    points are UNPOKED. Poking the object with its own early motion is the leak that gave
    FPT a false 43/43: the model then extrapolates the momentum we handed it.
  * Image centre-cropped to square exactly as MYRIAD's own embedder does (better_resize).
  * 12 steps of 1/12 s (the qual.py time grid, t x 30) = 1 s, the simulator's horizon.
  * Samples: an ensemble per case (the head is a flow-matching sampler, so every path is a
    draw from a distribution). Seeded.
  * POSITIVE CONTROL first: MYRIAD's own falling_ball / ball_roll images, object unpoked.
    If a ball in mid-air does not fall here, nothing downstream is interpretable.

    C scripts/f5_myriad.py control
    C scripts/f5_myriad.py run
    W scripts/f5_myriad.py plot
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "models/myriad"))

CKPT = "/home/jooyeolyun/.cache/torch/hub/checkpoints/myriad_openset.pt"
STEPS = 13                     # positions t = 0..12, 1/12 s apart
DT = 1.0 / 12.0
ENS = 32
SEED = 0
import os
_SUF = "_ramp" if os.environ.get("F4_RAMP") == "1" else ""
OUT = Path(f"docs/results/f5_myriad{_SUF}.json")
BG = [(0.08, 0.06), (0.5, 0.05), (0.92, 0.06), (0.05, 0.5), (0.95, 0.5),
      (0.08, 0.95), (0.5, 0.97), (0.92, 0.95)]


class _Stub:
    """Stand-in for classes the checkpoint pickles alongside the weights (its omegaconf
    training config). Only sd["model"] is used, so these need only to unpickle."""
    def __init__(self, *a, **k): pass
    def __setstate__(self, st): self.__dict__["_state"] = st


class _StubPickle:
    import pickle as _p

    class Unpickler(_p.Unpickler):
        def find_class(self, mod, name):
            if mod.split(".")[0] in ("omegaconf", "hydra"):
                return _Stub
            return super().find_class(mod, name)

    load = staticmethod(_p.load)
    __name__ = "pickle"


def load_model(dev="cuda:0"):
    """MYRIAD open-set. DINOv3's HF config is gated, but the MYRIAD checkpoint carries its
    own fine-tuned DINOv3 weights, so only the ViT-L/16 architecture is needed; a STRICT
    state-dict load verifies every shape."""
    import torch
    import transformers
    from transformers import DINOv3ViTConfig
    cfg = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                          num_attention_heads=16, num_register_tokens=4, patch_size=16,
                          layerscale_value=1e-5, rope_theta=100.0, layer_norm_eps=1e-5,
                          key_bias=False, query_bias=True, value_bias=True, proj_bias=True,
                          mlp_bias=True, use_gated_mlp=False)
    orig = transformers.AutoModel.from_pretrained
    transformers.AutoModel.from_pretrained = lambda *a, **k: transformers.AutoModel.from_config(cfg)
    import myriad.dinov3 as D
    D.AutoModel.from_pretrained = transformers.AutoModel.from_pretrained
    try:
        from myriad.model import MyriadStepByStep_Large
        m = MyriadStepByStep_Large()
    finally:
        transformers.AutoModel.from_pretrained = orig
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, pickle_module=_StubPickle)
    ck = sd["model"]
    # transformers 5 nests DINOv3's blocks one module deeper than the version MYRIAD was
    # saved with ("...model.model.layer.N" -> "...model.model.model.layer.N"). Map each
    # expected key to the checkpoint key it equals once that extra level is removed; the
    # load stays STRICT, so any key or shape that does not correspond still fails.
    want = m.state_dict().keys()
    remap = {}
    for k in want:
        if k not in ck:
            alt = k.replace("image_embedder.model.model.model.", "image_embedder.model.model.", 1)
            if alt in ck:
                remap[alt] = k
    ck = {remap.get(k, k): v for k, v in ck.items()}
    print(f"[myriad] remapped {len(remap)} DINOv3 keys")
    m.load_state_dict(ck, strict=True)
    m.requires_grad_(False)
    return m.eval().to(dev)


def simulate_batch(m, imgs, obj_qs, obj_q1s=None, n_ens=ENS, seed=SEED, dev="cuda:0"):
    """Several cases in ONE predict_simulate call.

    MYRIAD rolls a path out one point-step at a time (~290 sequential rounds for 24 tracks x
    12 steps), and each round is tiny, so a single case leaves the GPU waiting on Python to
    launch kernels (measured: one core at 86%, GPU at 33%). The round count is fixed by the
    sequence, so the lever is doing more work per round: cases x ensemble in one batch.
    Cases in a batch must share the track count and whether the object is poked.

    imgs: list of HxWx3 uint8 squares. obj_qs: list of (N,2) in [0,1]. obj_q1s: None, or a
    list of KNOWN object positions after the image -- (N,2) for step 1 only, or (k,N,2) for
    steps 1..k. Known positions are MYRIAD's prefix in time-major order, so k steps of every
    track are given, background held at its t0 position. All cases in a batch share k.
    Returns (B, E, STEPS, N_bg+N, 2).
    """
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    kb = len(BG)
    given = []
    for i, oq in enumerate(obj_qs):
        q0 = np.concatenate([np.array(BG, np.float32), oq.astype(np.float32)], 0)
        if obj_q1s is None:
            g = [q0, q0[:kb]]                               # t0 for all, t1 = t0 for bg
        else:
            kn = np.asarray(obj_q1s[i], np.float32)
            kn = kn[None] if kn.ndim == 2 else kn           # (k,N,2)
            g = [q0]
            for step in kn:                                 # steps 1..k, bg then object
                g += [q0[:kb], step]
        given.append(np.concatenate(g, 0))
    given = np.stack(given)                                             # B,L,2
    B, n_traj = len(imgs), kb + obj_qs[0].shape[0]
    x = torch.tensor(np.stack([im.transpose(2, 0, 1) for im in imgs]), dtype=torch.float32,
                     device=dev) / 127.5 - 1.0
    ts = torch.tensor((np.arange(STEPS) * DT * 30.0)[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        d = m.embed_image(x.repeat_interleave(n_ens, 0))
        # MYRIAD_FP32=1: no bf16 autocast -- the precision OWM was reproduced at (f10).
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=os.environ.get("MYRIAD_FP32") != "1"):
            out = m.predict_simulate(n_traj=n_traj, ts=ts.repeat(B * n_ens, 1),
                                     given_pos=torch.tensor(given, device=dev)
                                     .repeat_interleave(n_ens, 0),
                                     camera_static=True, d_img=d, verbose=False)
    out = out.float().cpu().numpy()
    return out.reshape(B, n_ens, *out.shape[1:])


def simulate(m, img_sq, obj_q, n_ens=ENS, seed=SEED, dev="cuda:0", obj_q1=None):
    """One case. Returns (E, STEPS, N_bg+N, 2) normalised positions; background first."""
    return simulate_batch(m, [img_sq], [obj_q], None if obj_q1 is None else [obj_q1],
                          n_ens=n_ens, seed=seed, dev=dev)[0]


def do_control(args):
    from PIL import Image
    m = load_model()
    res = {}
    # object query positions from MYRIAD's own qual.py
    cases = {"falling_ball": [(0.31, 0.61), (0.28, 0.60), (0.33, 0.64), (0.34, 0.57), (0.28, 0.68)],
             "ball_roll": [(0.27, 0.72)], "egg_roll": [(0.27, 0.72)]}
    for nm, q in cases.items():
        im = Image.open(REPO / f"models/myriad/qual_examples/{nm}.jpg").convert("RGB")
        w, h = im.size; s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        sq = np.asarray(im.crop((l, t, l + s, t + s)))
        qq = (np.array(q) * [w, h] - [l, t]) / s
        P = simulate(m, sq, qq)
        ob = P[:, :, len(BG):]
        dxy = (ob[:, -1] - ob[:, 0]).mean(1) * s                 # E,2 px
        bgm = np.linalg.norm(P[:, -1, :len(BG)] - P[:, 0, :len(BG)], axis=-1).mean() * s
        res[nm] = {"paths": P.tolist(), "side": s}
        print(f"  {nm:<13} object 1 s: median dx {np.median(dxy[:,0]):+6.1f} dy "
              f"{np.median(dxy[:,1]):+6.1f} px (of {s}); |disp| median "
              f"{np.median(np.linalg.norm(dxy,axis=-1)):5.1f}; moved >5% of frame in "
              f"{(np.linalg.norm(dxy,axis=-1) > 0.05*s).mean()*100:3.0f}% of samples; "
              f"background {bgm:.1f} px")
    Path("docs/results/f5_control.json").write_text(json.dumps(res))
    return 0


def crop_sq(W, H):
    s = min(W, H)
    return (W - s) // 2, (H - s) // 2, s


CHUNK = int(os.environ.get("MYRIAD_CHUNK", "4"))     # cases per call; x ENS sequences


def do_run(args):
    """Collect every (asset, tilt, variant) job, then run them in batches of CHUNK."""
    import time
    import torch
    import imageio.v2 as imageio
    import f4_fpt_joint as F
    from src.render.crop import project_points
    m = load_model()
    W, H = F.wh()
    l, t, s = crop_sq(int(W), int(H))
    out, jobs = {}, []
    for tag, _ in F.T.ASSETS:
        d = F.lab(tag)
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        fpt = json.loads((d / "fpt.json").read_text())
        out[tag] = {}
        for tk, c in fpt.items():
            idx = np.array(c["idx"])
            rows = sorted([r for r in spec if F.tkey(r["tilt"]) == tk], key=lambda r: r["mu"])
            uv, _ = project_points(F.cam_at(0), np.asarray(meta[rows[0]["key"]][0], float)[idx])
            q = (uv - [l, t]) / s
            im = imageio.imread(d / f"sim_{tk}@a" / "f0000.png")[..., :3][t:t + s, l:l + s]
            out[tag][tk] = {"idx": idx.tolist(), "poked": {}}
            jobs.append((tag, tk, None, im, q, None))
            # SELF-POKED: each candidate mu pokes with ITS OWN first simulated step
            # (frames 0 -> 2 = 1/12 s). No external motion enters; the question is whether
            # MYRIAD's continuation agrees with the simulator's for the right mu only.
            ms = F.T.mu_star(float(rows[0]["tilt"]))
            for fac in (0.7, 1.0, 1.5):
                r = min(rows, key=lambda r: abs(np.log(r["mu"] / (ms * fac))))
                u1, _ = project_points(F.cam_at(0), np.asarray(meta[r["key"]][2], float)[idx])
                jobs.append((tag, tk, r["key"], im, q, (u1 - [l, t]) / s))
    groups = {}
    for j in jobs:
        groups.setdefault((len(j[4]), j[5] is not None), []).append(j)
    t0 = time.time(); done = 0
    for (nq, poked), js in groups.items():
        for c0 in range(0, len(js), CHUNK):
            ch = js[c0:c0 + CHUNK]
            P = simulate_batch(m, [j[3] for j in ch], [j[4] for j in ch],
                               [j[5] for j in ch] if poked else None,
                               seed=SEED + c0 + (10000 if poked else 0))
            for j, Pj in zip(ch, P):
                px = (Pj * s + np.array([l, t])).tolist()
                if j[2] is None:
                    out[j[0]][j[1]]["paths_px"] = px
                else:
                    out[j[0]][j[1]]["poked"][j[2]] = px
            done += len(ch)
            print(f"  {done}/{len(jobs)} cases  {time.time()-t0:6.0f} s  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    Path(f"outputs/scene/f5_myriad_paths{_SUF}.json").write_text(json.dumps(out))
    return 0


def sim_paths(meta, key, idx):
    """Simulated positions at the MYRIAD time grid (every 2nd frame of 24 fps), fixed cam."""
    import f4_fpt_joint as F
    from src.render.crop import project_points
    return np.stack([project_points(F.cam_at(0), np.asarray(meta[key][2 * k], float)[idx])[0]
                     for k in range(STEPS)])                    # T,N,2


def do_plot(args):
    import imageio.v2 as imageio
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import f4_fpt_joint as F
    R = json.loads(Path(f"outputs/scene/f5_myriad_paths{_SUF}.json").read_text())
    kb = len(BG)
    tags = [t for t, _ in F.T.ASSETS if t in R]
    tilts = [15, 19, 24, 30, 37]
    fig, axs = plt.subplots(len(tags), 5, figsize=(16, 3.3 * len(tags)), dpi=90)
    summ = {}
    print("displacement ALONG the simulator's slide direction over 1 s, px (median of samples)")
    print(f"{'asset':<7}{'tilt':>5}{'unpoked':>9}   self-poked: MYRIAD / sim at 0.7, 1.0, 1.5 x mu*")
    for a, tag in enumerate(tags):
        d = F.lab(tag)
        spec = {r["key"]: r for r in json.loads((d / "spec.json").read_text())}
        meta = json.loads((d / "meta.json").read_text())
        summ[tag] = {}
        for b, tl in enumerate(tilts):
            tk = F.tkey(tl); ax = axs[a, b]
            c = R[tag][tk]; idx = np.array(c["idx"])
            un = np.array(c["paths_px"])[:, :, kb:]                  # E,T,N,2
            keys = sorted(c["poked"], key=lambda k: spec[k]["mu"])
            Sfast = sim_paths(meta, min((k for k in spec if spec[k]["tilt"] == tl),
                                        key=lambda k: spec[k]["mu"]), idx)
            dh = (Sfast[-1] - Sfast[0]).mean(0); dh /= np.linalg.norm(dh)
            udown = float(np.median((un[:, -1] - un[:, 0]).mean(1) @ dh))
            cells, rec = [], {"unpoked_down": udown, "poked": {}}
            for k in keys:
                pk = np.array(c["poked"][k])[:, :, kb:]
                S = sim_paths(meta, k, idx)
                my = float(np.median((pk[:, -1] - pk[:, 0]).mean(1) @ dh))
                sm = float(((S[-1] - S[0]).mean(0)) @ dh)
                p1 = float(((S[1] - S[0]).mean(0)) @ dh)
                rec["poked"][k] = {"mu": spec[k]["mu"], "myriad": my, "sim": sm, "poke": p1}
                cells.append(f"{my:6.1f}/{sm:<6.1f}(poke {p1:4.1f})")
            summ[tag][tk] = rec
            print(f"{tag:<7}{tl:>5}{udown:>+9.1f}   " + "  ".join(cells))
            # panel: unpoked (red), self-poked at mu* (orange) vs simulator at that mu (blue)
            im = imageio.imread(d / f"sim_{tk}@a" / "f0000.png")[..., :3]
            ax.imshow(im)
            kmid = keys[len(keys) // 2]
            pk = np.array(c["poked"][kmid])[:, :, kb:]
            S = sim_paths(meta, kmid, idx)
            for arr, col, al in ((un, "#e8322b", 0.35), (pk, "#ff9f1a", 0.45)):
                for e in range(min(arr.shape[0], 10)):
                    for j in range(arr.shape[2]):
                        ax.plot(arr[e, :, j, 0], arr[e, :, j, 1], "-", color=col, lw=0.6, alpha=al)
            for j in range(S.shape[1]):
                ax.plot(S[:, j, 0], S[:, j, 1], "-", color="#2b6fd6", lw=1.0, alpha=0.9)
            p0 = un[0, 0]
            ax.scatter(p0[:, 0], p0[:, 1], s=6, color="w", zorder=5)
            cx, cy = p0.mean(0)
            ax.set_xlim(cx - 150, cx + 150); ax.set_ylim(cy + 190, cy - 90); ax.axis("off")
            ax.set_title(f"{tag} {tl}°  (mu={spec[kmid]['mu']:.2f})", fontsize=10)
    fig.suptitle("RED = MYRIAD, object unpoked.  ORANGE = MYRIAD poked with the simulator's own "
                 "first 1/12 s.  BLUE = simulator (same mu). 16 points, 1 s", fontsize=12)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.985))
    fig.savefig(f"docs/results/f5_myriad_paths{_SUF}.png")
    OUT.write_text(json.dumps(summ, indent=1))
    print(f"wrote docs/results/f5_myriad_paths{_SUF}.png and", OUT)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, f in (("control", do_control), ("run", do_run), ("plot", do_plot)):
        sub.add_parser(nm).set_defaults(f=f)
    a = ap.parse_args()
    sys.exit(a.f(a))
