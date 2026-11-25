"""L2 -- real cloth beside the teacher and the model, drawn over the recording's own camera frames.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_render_real_cloth.py <case> --ckpt outputs/lsim/cloth_v3/last.pt --tag v3

Three panels over PhysTwin's camera 0, every frame of the recording: the reconstruction (the mesh fitted to the
tracked points), the teacher with its best material for the recording's training frames
(scripts/l2_real_cloth_teacher.py; from the first frame, the held patch following the recording), and the learned
model (given the first 8 frames, then alone, the held patch following the recording, its material hidden). Each
panel's error: the mean distance of the hand-annotated points to where the panel puts them (PhysTwin's track error;
the reconstruction's own is the reference, it drifts on folding cloth); frames after the split are predictions.

Writes outputs/lsim/render_real/<case>_<tag>.mp4 and a poster frame (.jpg).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lsim.real import cloth as rc
from lsim.real import safe_pickle

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/real/raw_phystwin"
FFMPEG = "/home/jooyeolyun/anaconda3/envs/svg/bin/ffmpeg"
GIVEN = 8
COLORS = {"real": (255, 255, 255), "teacher": (46, 160, 110), "model": (150, 105, 220)}


def project(X, c2w, K):
    Xw = X @ rc.FLIP.T                                                  # back to PhysTwin's frame (z down)
    Xc = (Xw - c2w[:3, 3]) @ c2w[:3, :3]
    uv = (Xc @ K.T)[:, :2] / Xc[:, 2:3]
    return uv, Xc[:, 2]


def draw_mesh(im, X, tris, c2w, K, color, alpha=0.38):
    uv, depth = project(X, c2w, K)
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    order = np.argsort(-depth[tris].mean(1))                            # far triangles first
    for t in tris[order]:
        d.polygon([tuple(uv[i]) for i in t], fill=color + (int(255 * alpha),), outline=color + (150,))
    return Image.alpha_composite(im.convert("RGBA"), over)


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--ckpt", default="outputs/lsim/cloth_v3/last.pt")
    ap.add_argument("--tag", default="v3")
    a = ap.parse_args()
    from lsim.model.cloth_net import build_cloth_model
    from lsim.train.cloth_train import rollout
    ep = rc.RealClothEpisode(rc.OUT / "phystwin" / a.case)
    T = ep.n_frames                                                    # the whole recording (PhysTwin's protocol)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    A = ck["args"]
    m = build_cloth_model(A).to(dev).eval()
    m.load_state_dict(ck["model"])
    with torch.no_grad():
        pred = rollout(m, ep, GIVEN - 1, T - GIVEN, dev=dev, hidden=("density", "stretch", "bend", "mu"))[0]
    Xr = ep.x(np.arange(T))
    Xm = np.concatenate([Xr[:GIVEN], pred], 0)
    tf = np.load(REPO / f"outputs/lsim/real_teacher/{a.case}.npz")
    Xt = tf["x_best"][:T]
    tres = {}
    for t in ("pt_a", "pt_b"):
        f = REPO / f"docs/results/l2_real_cloth_teacher_{t}.json"
        tres.update(json.loads(f.read_text()) if f.exists() else {})
    mat = tres.get(a.case, {}).get("material")
    fr = np.arange(T)
    split = int(ep.meta["split"]["train_end"])
    err = {k: ep.gt_error(X, fr) for k, X in (("real", Xr), ("teacher", Xt), ("model", Xm))}   # the annotated points
    c2w = np.asarray(safe_pickle.load(RAW / "extracted" / a.case / "calibrate.pkl"))[0]
    K = np.asarray(json.loads((RAW / "extracted" / a.case / "metadata.json").read_text())["intrinsics"])[0]
    tris = ep.tris
    labels = {"real": "real (the mesh fitted to the tracked points)",
              "teacher": "teacher, best material" + (f": {1e3 * mat['w']:.0f} g/m2, Y {mat['Y']:.0f} N/m, mu {mat['mu']:.2f}" if mat else ""),
              "model": f"learned model {a.tag}, alone after {GIVEN} frames"}
    out = REPO / "outputs/lsim/render_real"
    frames_dir = out / f"{a.case}_{a.tag}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    f_big, f_small = font(17), font(15)
    for t in range(T):
        base = Image.open(RAW / "extracted_frames" / a.case / f"cam0_{t:03d}.png").convert("RGB")
        panels = []
        for k, X in (("real", Xr), ("teacher", Xt), ("model", Xm)):
            im = draw_mesh(base, X[t], tris, c2w, K, COLORS[k]).convert("RGB")
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, im.width, 26], fill=(20, 20, 22))
            d.text((8, 4), labels[k], font=f_small, fill=COLORS[k] if k != "real" else (235, 235, 235))
            d.rectangle([0, im.height - 28, im.width, im.height], fill=(20, 20, 22))
            e = err[k][t]
            line = f"t = {t / 30:.2f} s" + (f"   off the annotated points by {1e3 * e:.1f} mm" if np.isfinite(e) else "")
            if t >= split:
                line += "   (predicting)" if k != "real" else "   (test frames)"
            if k == "model" and t < GIVEN:
                line = f"t = {t / 30:.2f} s   given"
            d.text((8, im.height - 24), line, font=f_big, fill=(255, 193, 7) if k != "real" else (235, 235, 235))
            panels.append(im)
        w, h = panels[0].size
        sheet = Image.new("RGB", (3 * w + 8, h), (255, 255, 255))
        for i, p in enumerate(panels):
            sheet.paste(p, (i * (w + 4), 0))
        sheet.save(frames_dir / f"c{t:04d}.png")
    mp4 = out / f"{a.case}_{a.tag}.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(frames_dir / "c%04d.png"),
                    "-vf", "scale=1704:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow",
                    "-movflags", "+faststart", str(mp4)], check=True)
    Image.open(frames_dir / f"c{T - 1:04d}.png").save(out / f"{a.case}_{a.tag}_last.jpg", quality=86)
    print(f"[render] {mp4}: {T} frames; error at the last frame: teacher {1e3 * err['teacher'][-1]:.1f} mm, "
          f"model {1e3 * err['model'][-1]:.1f} mm, reconstruction {1e3 * err['real'][-1]:.1f} mm", flush=True)


if __name__ == "__main__":
    main()
