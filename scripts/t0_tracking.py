"""T0 -- can a vision model READ an observable off our clips, rather than judge them?

Two supervision signals have now failed rule 1 on this project. Cosmos 3 scores rho ~ 0
against every parameter and answers one-bit threshold questions wrong 9/9. V-JEPA 2
surprise tracks how much a clip moves (rho +0.98 with motion) and rates an object passing
through a solid table 42 sigma MORE plausible than a legal bounce.

But every number in the probe table -- 0.40 deg slip residual, rho -0.995 on bounce, 7/7
on the balance -- was read by the SIMULATOR, off its own state. So the failure is specific:
these models cannot JUDGE physics in our renders. Whether they can PERCEIVE it is a
different question, and it has never been measured here.

That is what this measures, and it is nearly free: the 57 V1 clips already on disk have
exact ground truth. We know the camera, and meta.json stores the object's world points per
frame, so the true pixel trajectory is computable to the pixel. No annotation, no new
renders, no assumptions.

Two readings, in order of what they would let us build:

  1. TRACKING ERROR   median distance in pixels between the recovered trajectory and the
                      true one. Does the model know where the object is?
  2. THE OBSERVABLE   the quantity a probe actually fits against -- here total displacement
                      across the clip -- recovered from the tracked trajectory and
                      correlated against theta across the sweep. This is the one that
                      matters: a tracker that is 3 px off but recovers the theta ordering
                      is a usable instrument; one that is 1 px off but returns noise on
                      the observable is not.

If this passes, the architecture inverts: the vision model becomes an INSTRUMENT that reads
an observable, the simulator computes the same observable from its own state, and theta is
fit by matching the two -- with real footage supplying the targets later. The model would
then only have to do the thing it is actually trained for. If it fails, we have measured
that video models cannot serve this project at all, which is a result in its own right.

    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/t0_tracking.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# TWO backbones, deliberately. V-JEPA 2 is not dropped by the V1 verdict: what failed
# there was surprise as a PLAUSIBILITY SCORE, which says nothing about whether its encoder
# knows where things are. It is a video model with temporal features, so if anything it is
# the better candidate here; DINOv2 is the per-frame image-model control that tells us
# whether temporal modelling buys anything at all.
BACKBONES = {
    "dinov2-large":  {"id": "facebook/dinov2-large", "kind": "image", "patch": 14, "size": 252},
    "vjepa2-vitl":   {"id": "facebook/vjepa2-vitl-fpc64-256", "kind": "video",
                      "patch": 16, "size": 256},
}
OUT = Path("docs/results/t0_tracking.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


class PinholeCam:
    """The render camera, reimplemented without warp.

    src/render/camera.py imports warp at module scope, and the scoring env has no warp --
    the same two-env split that governs everything else here. The projection is a dozen
    lines, so it is copied rather than the env polluted. Convention matches camera.py
    exactly: rows of R map world into (right, down, forward), v grows downward.
    """

    def __init__(self, eye, target, fov_deg, width, height, up=(0.0, 0.0, 1.0)):
        eye = np.asarray(eye, float); target = np.asarray(target, float)
        fwd = target - eye; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.asarray(up, float)); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])
        self.t = -self.R @ eye
        self.fx = self.fy = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
        self.cx, self.cy = 0.5 * width, 0.5 * height

    def project(self, pts):
        pc = (self.R @ np.asarray(pts, float).T).T + self.t
        z = np.maximum(pc[:, 2], 1e-6)
        return np.stack([self.fx * pc[:, 0] / z + self.cx,
                         self.fy * pc[:, 1] / z + self.cy], 1), pc[:, 2]


def pulled_camera(pull, view="a"):
    from src.render.views import VIEWS
    vc = VIEWS[view]
    eye, tgt = np.asarray(vc["eye"], float), np.asarray(vc["target"], float)
    return PinholeCam(tgt + (eye - tgt) * pull, tgt, vc["fov_deg"],
                      vc["width"], vc["height"])


def truth(lab, key, box, cam, n_frames, size):
    """The object's true centroid per frame, in the coordinates the model is shown."""
    pts = np.asarray(json.loads((lab / "meta.json").read_text())[key], float)[:n_frames]
    out = []
    x0, y0, x1, y1 = box
    sc = size / float(x1 - x0)
    for p in pts:
        uv, depth = cam.project(p)
        uv = uv[depth > 0]
        c = uv.mean(0) if len(uv) else np.array([np.nan, np.nan])
        out.append([(c[0] - x0) * sc, (c[1] - y0) * sc])
    return np.asarray(out)


def _features(cfg, frames, torch):
    """(T, g, g, D) L2-normalised patch features, from either kind of backbone."""
    g = cfg["size"] // cfg["patch"]
    if cfg["kind"] == "image":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(
            cfg["id"], size={"height": cfg["size"], "width": cfg["size"]},
            crop_size={"height": cfg["size"], "width": cfg["size"]})
        model = AutoModel.from_pretrained(cfg["id"], dtype=torch.float32).to("cuda:0").eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), 8):
                b = proc(images=list(frames[i:i + 8]), return_tensors="pt").to(model.device)
                h = model(**b).last_hidden_state[:, 1:]            # drop CLS
                out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu())
        F = torch.cat(out).numpy().reshape(len(frames), g, g, -1)
        times = np.arange(len(frames), dtype=float)
    else:
        # V-JEPA 2: one spatiotemporal token per (2 frames) x (16x16 px), so the track
        # comes out at slot resolution and each point sits at the centre of its tubelet.
        from src.surprise.vjepa import SurpriseModel, DEPTH, GRID
        m = SurpriseModel(model_id=cfg["id"], device="cuda:0")
        toks = m.encode(frames)
        F = torch.nn.functional.normalize(toks, dim=-1)[0].float().cpu().numpy()
        F = F.reshape(DEPTH, GRID, GRID, -1)
        times = np.arange(DEPTH, dtype=float) * 2.0 + 0.5
        del m
    import gc
    gc.collect(); torch.cuda.empty_cache()
    return F, times


def _track(F, patch, start_xy):
    """Match the frame-0 patch forward by cosine similarity -- the simplest correspondence
    method there is. If plain feature matching reads the observable, no tracking stack is
    needed; if it does not, that is the point at which installing one is justified."""
    g = F.shape[1]
    c0 = min(max(int(start_xy[0] // patch), 0), g - 1)
    r0 = min(max(int(start_xy[1] // patch), 0), g - 1)
    tmpl = F[0, r0, c0]
    out = []
    for t in range(len(F)):
        j = int(np.argmax(F[t].reshape(-1, F.shape[-1]) @ tmpl))
        out.append([(j % g) * patch + patch / 2, (j // g) * patch + patch / 2])
    return np.asarray(out, float)


def main():
    import torch
    from src.surprise import clips as C
    import v1_killtest as V

    lab = Path("outputs/scene/v1")
    spec = {r["key"]: r for r in json.loads((lab / "spec.json").read_text())}
    cam = pulled_camera(V.CAM_PULL)
    boxes = V.shared_boxes(cam, json.loads((lab / "meta.json").read_text()), spec, 544, 448)
    keys = [k for k in sorted(spec)
            if spec[k]["probe"] in ("bounce", "tilt")
            and (lab / "clips" / f"{k}.mp4").exists()]
    print(f"{len(keys)} clips, {len(BACKBONES)} backbones", flush=True)

    rows = []
    for name, cfg in BACKBONES.items():
        print(f"\n=== {name}  ({cfg['id']})", flush=True)
        for k in keys:
            raw = C.resample(C.read_frames(lab / "clips" / f"{k}.mp4"), V.NF)
            frames = C.letterbox(raw, cfg["size"])[0]
            gt_full = truth(lab, k, boxes[k], cam, V.NF, cfg["size"])
            F, times = _features(cfg, frames, torch)
            pr = _track(F, cfg["patch"], gt_full[0])
            gt = np.stack([gt_full[int(round(t))] for t in np.clip(times, 0, V.NF - 1)])
            err = np.linalg.norm(pr - gt, axis=1) / cfg["size"] * 100     # % of frame
            rows.append({**spec[k], "backbone": name,
                         "median_err_pct": float(np.median(err)),
                         "p90_err_pct": float(np.percentile(err, 90)),
                         "true_disp": float(np.linalg.norm(gt[-1] - gt[0]) / cfg["size"]),
                         "tracked_disp": float(np.linalg.norm(pr[-1] - pr[0]) / cfg["size"])})
            print(f"  {k:<22} median err {rows[-1]['median_err_pct']:5.1f}% of frame"
                  f"   disp true {rows[-1]['true_disp']:.3f}  tracked "
                  f"{rows[-1]['tracked_disp']:.3f}", flush=True)

    print("\n" + "=" * 68)
    for name in BACKBONES:
        rs = [r for r in rows if r["backbone"] == name]
        print(f"{name}: median tracking error "
              f"{np.median([r['median_err_pct'] for r in rs]):.1f}% of frame width")
        for probe in ("bounce", "tilt"):
            ps = sorted([r for r in rs if r["probe"] == probe], key=lambda r: r["value"])
            if len(ps) < 3:
                continue
            v = [r["value"] for r in ps]
            print(f"   {probe:<7} rho(theta, TRUE disp) "
                  f"{spearman(v, [r['true_disp'] for r in ps]):+.3f}"
                  f"   rho(theta, TRACKED disp) "
                  f"{spearman(v, [r['tracked_disp'] for r in ps]):+.3f}   <- decides it")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
