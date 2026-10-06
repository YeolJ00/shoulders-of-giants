"""F0 -- does a real-video MOTION PRIOR respond to theta? The cheapest kill test.

The classifier readout measures a displacement the simulator already knows exactly, so it
adds nothing for our own renders and its calibration does not transfer between objects
(1.02x to 54x scale offsets). A motion prior inverts the roles: the simulator supplies the
motion, and the prior supplies what motion is EXPECTED. That is the thing we actually lack.

Flow Poke Transformer (CompVis) outputs `MixtureSameFamily` over flow at query points, given
an image, observed flow at "poke" points, and a camera_static flag. `MixtureSameFamily`
carries torch's `.log_prob`, so our own motion can be SCORED rather than re-measured:

    theta -> sim -> image-space flow -> log p(flow | image, poke)

If this works it also restores end-to-end gradients, because the render is reduced to a
single conditioning frame nothing backprops through, and every remaining link -- log_prob,
the camera projection, and theta -> trajectory through the Warp tape -- is differentiable.

RULE 1 FIRST. A prior trained on real video has no obligation to respond to friction. Before
anything is built on it: sweep mu, score each clip, and check that log p moves. Two ways it
can fail and both are reported:

  FLAT      log p barely changes over a 10x mu sweep -> the prior cannot see friction here,
            and this dies where the judge (rho ~ 0) and surprise (chance) died.
  MONOTONE  log p rises or falls across the whole sweep with no interior maximum -> it is
            reading "how much motion", the confound that has recurred through this project
            (judge rho -0.884 with motion magnitude, surprise +0.98). A usable prior must
            prefer some mu and disprefer both extremes.

Scored on the tilted-ramp clips already on disk, with the exact trajectories from meta.json.
No new renders.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "models/fpt"))

CKPT = "/home/jooyeolyun/.cache/torch/hub/checkpoints/flow_poke_open_set_base.pt"
OUT = Path("docs/results/f0_fpt_gate.json")
P0, P1, P2 = 0, 4, 20          # poke window 0->4 frames, scored continuation 4->20


def load_model(device="cuda:0"):
    import torch
    from flow_poke.model import FlowPokeTransformer_Base
    m = FlowPokeTransformer_Base()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["model"])
    m.requires_grad_(False)
    return m.eval().to(device)


def clip_trajectory(lab, key, box, pull, n_pts=8):
    """Object points in CLIP coordinates, normalised to [0,1]^2, per frame."""
    from src.render.crop import project_points
    from t0_tracking import pulled_camera
    meta = json.loads((Path(lab) / "meta.json").read_text())[key]
    cam = pulled_camera(pull, view="a")
    side = box[2] - box[0]
    out = []
    for fr in meta:
        P = np.asarray(fr, float).reshape(-1, 3)
        uv, ok = project_points(cam, P)
        uv = uv[ok]
        if len(uv) == 0:
            return None
        idx = np.linspace(0, len(uv) - 1, min(n_pts, len(uv))).astype(int)
        q = (uv[idx] - np.array([box[0], box[1]])) / side
        out.append(q)
    return np.stack(out)          # (T, n_pts, 2) in [0,1]


def frame0(mp4, size=518):
    """First frame, square, in [-1,1], as the conditioning image."""
    import imageio.v2 as imageio
    from PIL import Image
    rd = imageio.get_reader(str(mp4))
    fr = next(iter(rd))[..., :3]
    rd.close()
    im = Image.fromarray(fr).resize((size, size), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 127.5 - 1.0
    return a.transpose(2, 0, 1)[None]


def do_run(args):
    import torch
    import o13_tilted as T
    dev = "cuda:0"
    m = load_model(dev)
    cfg = json.loads(Path("outputs/scene/_t_solo.json").read_text())
    R = T._rows()
    res = {}
    for tag in (args.assets.split(",") if args.assets else list(R)):
        if tag not in R or tag not in cfg:
            continue
        box, pull = tuple(cfg[tag]["box"]), cfg[tag]["pull"]
        lab = f"outputs/scene/t_{tag}"
        by = {}
        for r in R[tag]:
            by.setdefault(r["tilt"], []).append(r)
        for tilt, g in sorted(by.items()):
            g = sorted(g, key=lambda r: r["value"])
            if len(g) < 4:
                continue
            rows = []
            for r in g:
                tr = clip_trajectory(lab, r["key"], box, pull)
                if tr is None or len(tr) <= P2:
                    continue
                pk_pos = torch.tensor(tr[P0][None], dtype=torch.float32, device=dev)
                pk_fl = torch.tensor((tr[P1] - tr[P0])[None], dtype=torch.float32, device=dev)
                q_pos = torch.tensor(tr[P1][None], dtype=torch.float32, device=dev)
                obs = torch.tensor((tr[P2] - tr[P1])[None], dtype=torch.float64, device=dev)
                img = torch.tensor(frame0(f"{lab}/clips/{r['key']}.mp4"), device=dev)
                with torch.no_grad():
                    d_img = m.embed_image(img)
                    dist = m.predict_parallel(pk_pos, pk_fl, q_pos, True, d_img)
                    lp = float(dist.log_prob(obs).mean())
                rows.append({"mu": r["value"], "slide_cm": abs(r["slide_cm"]),
                             "logp": lp,
                             "flow_px": float(np.linalg.norm(tr[P2] - tr[P1], axis=-1).mean()
                                              * (box[2] - box[0]))})
            if len(rows) < 4:
                continue
            lp = np.array([x["logp"] for x in rows])
            mu = np.array([x["mu"] for x in rows])
            i = int(np.argmax(lp))
            interior = 0 < i < len(lp) - 1
            res[f"{tag}_t{int(tilt)}"] = {
                "rows": rows, "spread": float(lp.max() - lp.min()),
                "argmax_mu": float(mu[i]), "interior_max": interior,
                "rho_mu": float(np.corrcoef(np.log(mu), lp)[0, 1]),
                "rho_motion": float(np.corrcoef(
                    np.log([x["flow_px"] + 1e-6 for x in rows]), lp)[0, 1])}
            k = res[f"{tag}_t{int(tilt)}"]
            print(f"  {tag:<8} t{int(tilt):<3} n={len(rows)}  "
                  f"log p spread {k['spread']:7.2f}  argmax mu {k['argmax_mu']:.3f}"
                  f"{'  INTERIOR' if interior else '  at an edge'}"
                  f"   rho(logp,mu) {k['rho_mu']:+.2f}  rho(logp,motion) {k['rho_motion']:+.2f}",
                  flush=True)
            print("      mu   : " + " ".join(f"{x['mu']:7.3f}" for x in rows))
            print("      log p: " + " ".join(f"{x['logp']:7.2f}" for x in rows))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    if res:
        sp = [v["spread"] for v in res.values()]
        ni = sum(v["interior_max"] for v in res.values())
        print(f"\nVERDICT over {len(res)} sweeps:")
        print(f"  log p spread: median {np.median(sp):.2f} nats "
              f"(flat would be ~0)")
        print(f"  interior maximum in {ni}/{len(res)} sweeps "
              f"(an edge maximum means it prefers one extreme)")
        rm = [abs(v["rho_motion"]) for v in res.values()]
        print(f"  |rho(log p, motion magnitude)|: median {np.median(rm):.2f} "
              f"(high => reading how much it moved, the old confound)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="")
    ap.set_defaults(f=do_run)
    a = ap.parse_args()
    sys.exit(a.f(a))
