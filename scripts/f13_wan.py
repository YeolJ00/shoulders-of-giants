"""F13 -- Wan 2.2 image-to-video as the motion source: generate, track, invert. PILOT.

A generated video as the target is what this project's core rule excluded ("video models
never generate our targets") -- revisited by decision. The guard is the same battery of
tests that retired every earlier signal, applied to TRACKED points of the generated clip:

  1. does the object move at all, and downhill (not with the camera, not deforming)?
  2. rigidity: do its tracked points keep their mutual distances?
  3. inverting each clip through OUR simulator's forward curve, do the tilts give the SAME
     mu for one object (a material property), and different mu for different objects?

Model: Wan-AI/Wan2.2-I2V-A14B-Diffusers (Physics-IQ Verified 32.2, the best open i2v with
HunyuanVideo 1.5). Conditioning frame = our real-ramp render (f6, plank on a level table,
static camera cam_at(0)), resized 544x448 -> 576x480. Prompt NEUTRAL about the outcome --
it names the scene, not what happens -- so the model, not the prompt, decides whether the
object slides. 81 frames at 16 fps (5 s); comparisons use 0-1 s, the simulator's horizon.
Tracker: CoTracker3 offline (torch.hub cache), 16 object points projected from the sim's
frame-0 vertices -- the same points f4-f9 used -- plus 8 background points.

    C scripts/f13_wan.py gen   --gpu-shard k/N
    C scripts/f13_wan.py track
    W scripts/f13_wan.py fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.append(str(REPO / "models/_deps_wan"))    # ftfy: vendored, not in the cosmos env

MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
W_OUT, H_OUT, NF, FPS = 576, 480, 81, 16
ASSETS = ("box", "sorter")
TILTS = (19, 24, 30)
SEEDS = (0, 1)
NAMES = {"box": "cardboard box", "sorter": "blue plastic file sorter", "bowl": "wooden bowl",
         "pot": "brass pot", "teapot": "white porcelain teapot", "books": "set of books",
         "shark": "toy shark"}
NEG = ("camera motion, camera shake, zoom, pan, blurry, distorted, deformed object, "
       "morphing, extra objects, low quality, cartoon")
# F13_PROMPT=slide: the prompt states THAT the object slides; HOW FAST is left to the
# model. Only the speed can then carry mu -- it must rise with tilt and differ by object.
MODE = os.environ.get("F13_PROMPT", "neutral")
_SUF = "" if MODE == "neutral" else f"_{MODE}"
OUT = Path(f"outputs/scene/f13_wan{_SUF}")
RES = Path(f"docs/results/f13_wan{_SUF}.json")


def prompt(tag):
    if MODE == "slide":
        return (f"A realistic video, static camera. A {NAMES[tag]} slides down an inclined "
                f"wooden plank that rests on a wooden table, propped up on a block, on a "
                f"sunny street.")
    return (f"A realistic video, static camera. A {NAMES[tag]} is on an inclined wooden plank "
            f"that rests on a wooden table, propped up on a block, on a sunny street.")


def cases():
    for tag in ASSETS:
        for tl in TILTS:
            for sd in SEEDS:
                yield tag, tl, sd


def cid(tag, tl, sd):
    return f"{tag}_t{tl:02d}_s{sd}"


def do_gen(args):
    import torch
    from PIL import Image
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    k, n = (int(x) for x in args.shard.split("/"))
    todo = [c for i, c in enumerate(cases()) if i % n == k]
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.enable_model_cpu_offload(gpu_id=0)
    import time
    for tag, tl, sd in todo:
        dst = OUT / f"{cid(tag, tl, sd)}.mp4"
        if dst.exists():
            print("skip", dst.name); continue
        img = Image.open(f"outputs/scene/f6_{tag}/sim_t{tl:02d}@a/f0000.png").convert("RGB")
        img = img.resize((W_OUT, H_OUT), Image.BICUBIC)
        t0 = time.time()
        fr = pipe(image=img, prompt=prompt(tag), negative_prompt=NEG, height=H_OUT,
                  width=W_OUT, num_frames=NF, num_inference_steps=40,
                  guidance_scale=3.5, guidance_scale_2=3.5,       # Wan 2.2 i2v defaults
                  generator=torch.Generator("cpu").manual_seed(sd)).frames[0]
        export_to_video(fr, str(dst), fps=FPS)
        print(f"  {dst.name}  {time.time()-t0:.0f} s", flush=True)
    return 0


def query_points(tag, tl):
    """16 object points (the f4/f6 query vertices) + 8 background points, in Wan pixels."""
    import f4_fpt_joint as F
    from src.render.crop import project_points
    d = Path(f"outputs/scene/f6_{tag}")
    fpt = json.loads((d / "fpt.json").read_text())
    spec = json.loads((d / "spec.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    idx = np.array(fpt[F.tkey(tl)]["idx"])
    k0 = next(r["key"] for r in spec if F.tkey(r["tilt"]) == F.tkey(tl))
    uv, _ = project_points(F.cam_at(0), np.asarray(meta[k0][0], float)[idx])
    W, H = F.wh()
    sc = np.array([W_OUT / W, H_OUT / H])
    bg = np.array([[0.06, 0.06], [0.5, 0.05], [0.94, 0.06], [0.04, 0.35],
                   [0.96, 0.35], [0.06, 0.95], [0.5, 0.97], [0.94, 0.95]]) * [W_OUT, H_OUT]
    return uv * sc, bg, idx, sc


def do_track(args):
    import torch
    import imageio.v2 as imageio
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda:0")
    res = {}
    for tag, tl, sd in cases():
        p = OUT / f"{cid(tag, tl, sd)}.mp4"
        if not p.exists():
            continue
        fr = np.stack([f[..., :3] for f in imageio.get_reader(str(p))])       # T,H,W,3
        obj, bg, _, _ = query_points(tag, tl)
        q = np.concatenate([obj, bg])
        qt = torch.tensor(np.concatenate([np.zeros((len(q), 1)), q], 1)[None],
                          dtype=torch.float32, device="cuda:0")                # t,x,y
        v = torch.tensor(fr, device="cuda:0").permute(0, 3, 1, 2)[None].float()
        with torch.no_grad():
            tr, vis = cot(v, queries=qt)
        res[cid(tag, tl, sd)] = {"tracks": tr[0].cpu().numpy().tolist(),
                                 "vis": vis[0].cpu().numpy().tolist(), "n_obj": len(obj)}
        print(f"  tracked {p.name}: {fr.shape[0]} frames", flush=True)
    (OUT / "tracks.json").write_text(json.dumps(res))
    return 0


def do_fit(args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    import f4_fpt_joint as F
    from src.render.crop import project_points
    Tk = json.loads((OUT / "tracks.json").read_text())
    W, H = F.wh()
    # the same instants in both clocks: generated 16 fps, simulator 24 fps
    t_s = np.array([0.25, 0.5, 0.75, 1.0])
    g_f, s_f = np.round(t_s * FPS).astype(int), np.round(t_s * 24).astype(int)
    res, rows = {}, []
    print(f"{'clip':<16}{'bg px':>7}{'down 1s':>9}{'down 5s':>9}{'side 1s':>9}"
          f"{'rigid':>7}{'  mu fitted (sim)':>18}{'  sim@mu*':>9}")
    for c, d in Tk.items():
        tag, tl, sd = c.split("_")[0], int(c.split("_")[1][1:]), int(c.split("_")[2][1:])
        tr = np.array(d["tracks"]); n = d["n_obj"]
        ob, bgt = tr[:, :n], tr[:, n:]
        cam = float(np.linalg.norm(bgt[-1] - bgt[0], axis=-1).mean())
        obj_q, _, idx, sc = query_points(tag, tl)
        dd = Path(f"outputs/scene/f6_{tag}")
        spec = sorted([r for r in json.loads((dd / "spec.json").read_text())
                       if r["tilt"] == tl], key=lambda r: r["mu"])
        meta = json.loads((dd / "meta.json").read_text())
        def sim_c(r, fs):
            return np.stack([project_points(F.cam_at(0), np.asarray(meta[r["key"]][f], float)
                                            [idx])[0].mean(0) * sc for f in fs])
        S0 = sim_c(spec[0], [0, 24])
        dh = S0[1] - S0[0]; dh /= np.linalg.norm(dh)
        nrm = np.array([-dh[1], dh[0]])
        c0 = ob[0].mean(0)
        down = (ob.mean(1) - c0) @ dh
        side = (ob.mean(1) - c0) @ nrm
        # rigidity: relative change of pairwise distances, frame 0 -> 1 s
        def pd(x):
            return np.linalg.norm(x[:, None] - x[None], axis=-1)[np.triu_indices(n, 1)]
        rig = float(np.median(np.abs(pd(ob[g_f[-1]]) - pd(ob[0])) / np.maximum(pd(ob[0]), 1)))
        g = down[g_f]
        # invert: sim along-downhill at the same instants for every mu; least squares in
        # log(1 + px) so a near-zero generated slide is still comparable
        errs = []
        for r in spec:
            s = (sim_c(r, s_f) - sim_c(r, [0])[0]) @ dh
            errs.append(np.mean((np.log1p(np.maximum(s, 0)) - np.log1p(np.maximum(g, 0))) ** 2))
        i = int(np.argmin(errs))
        ms = F.T.mu_star(float(tl))
        rstar = min(spec, key=lambda r: abs(np.log(r["mu"] / ms)))
        sstar = float(((sim_c(rstar, [24]) - sim_c(rstar, [0])[0]) @ dh)[0])
        res[c] = {"tag": tag, "tilt": tl, "seed": sd, "cam_px": cam, "down_1s": float(down[g_f[-1]]),
                  "down_5s": float(down[-1]), "side_1s": float(side[g_f[-1]]), "rigidity": rig,
                  "mu_fit": float(spec[i]["mu"]), "mu_fit_edge": bool(i in (0, len(spec) - 1)),
                  "sim_at_mustar_1s": sstar}
        print(f"{c:<16}{cam:>7.1f}{down[g_f[-1]]:>9.1f}{down[-1]:>9.1f}{side[g_f[-1]]:>9.1f}"
              f"{100*rig:>6.0f}%{spec[i]['mu']:>12.3f}{'|' if i in (0, len(spec)-1) else ' '}"
              f"{sstar:>12.1f}")
    print("\n  bg px = background drift over 5 s (static camera -> ~0); down/side = object centre"
          " along/across the sim's slide direction, px at 576x480; rigid = median relative change"
          " of point-pair distances by 1 s; '|' = fitted mu at the grid edge")
    print("\nSAME mu ACROSS TILTS? (per asset, per seed)")
    for tag in ASSETS:
        for sd in SEEDS:
            ms = [res[cid(tag, tl, sd)]["mu_fit"] for tl in TILTS if cid(tag, tl, sd) in res]
            if ms:
                print(f"  {tag:<7} seed {sd}: " + "  ".join(f"{m:.3f}" for m in ms)
                      + f"   spread {max(ms)/min(ms):.2f}x")
    RES.write_text(json.dumps(res, indent=1))
    # figure: frame 0 with tracked object points over 5 s (orange) and the sim at mu* (blue)
    cs = list(Tk)
    fig, axs = plt.subplots(len(ASSETS) * len(SEEDS), len(TILTS),
                            figsize=(5 * len(TILTS), 4.3 * len(ASSETS) * len(SEEDS)), dpi=90,
                            squeeze=False)
    for c in cs:
        tag, tl, sd = c.split("_")[0], int(c.split("_")[1][1:]), int(c.split("_")[2][1:])
        ax = axs[ASSETS.index(tag) * len(SEEDS) + SEEDS.index(sd), TILTS.index(tl)]
        rd = imageio.get_reader(str(OUT / f"{c}.mp4")); f0 = next(iter(rd)); rd.close()
        ax.imshow(f0)
        tr = np.array(Tk[c]["tracks"]); n = Tk[c]["n_obj"]
        for j in range(n):
            ax.plot(tr[:, j, 0], tr[:, j, 1], "-", color="#ff9f1a", lw=1.0)
            ax.plot(tr[:g_f[-1] + 1, j, 0], tr[:g_f[-1] + 1, j, 1], "-", color="#e8322b", lw=1.6)
        for j in range(n, tr.shape[1]):
            ax.plot(tr[:, j, 0], tr[:, j, 1], "-", color="#7fd4ff", lw=1.0)
        ax.set_title(f"{c}  (down 1 s: {res[c]['down_1s']:.0f} px, 5 s: {res[c]['down_5s']:.0f})",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle("Wan 2.2 i2v, tracked with CoTracker3. RED = object points 0-1 s, ORANGE = "
                 "to 5 s, CYAN = background points (should stay put)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"docs/results/f13_wan{_SUF}.png")
    print("wrote", RES, f"and docs/results/f13_wan{_SUF}.png")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("gen"); s.set_defaults(f=do_gen); s.add_argument("--shard", default="0/1")
    sub.add_parser("track").set_defaults(f=do_track)
    sub.add_parser("fit").set_defaults(f=do_fit)
    a = ap.parse_args()
    sys.exit(a.f(a))
