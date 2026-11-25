"""Watch long episodes (15-30 s, chained actions): each episode a Blender clip (scripts/blender_render.py, exported by
l1_render_cloth / l1_render_rope), an mp4 labelled with what is happening at each instant (which action of how many,
or idle), and a contact sheet of 10 instants.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l1_render_long.py [export|blender|compose|all] <shard without suffix>:<episode> [...]
        [--size 640x480 --samples 32 --out outputs/lsim/render_long]

Cloth or rope by the shard's kind. Output: <out>/<kind>_<dataset>_e<k>.mp4 and <kind>_<dataset>_e<k>_sheet.jpg; the
Blender frames under <out>/<kind>/.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
BLENDER = "/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender"


def timeline(wd, settle, n):
    """Per recorded frame: what the gripper is doing ("action k of n: family", "idle", "held from the start")."""
    plans = [p for p in wd["families"] if p.get("family") != "held" and "skipped" not in p]
    lab = ["idle"] * n
    held0 = any(p.get("family") == "held" for p in wd["families"])
    first = min([p["start"] - settle for p in plans], default=n)
    if held0:
        for f in range(max(0, min(first, n))):
            lab[f] = "held up by a corner" if wd.get("cloth") else "held up"
    for k, p in enumerate(plans):
        a = max(p["start"] - settle, 0)
        b = (p["end"] - settle) if "end" in p else (plans[k + 1]["start"] - settle if k + 1 < len(plans) else n)
        cut = " (cut by the end)" if p.get("cut_by_episode_end") else ""
        for f in range(a, min(b, n)):
            lab[f] = f"action {k + 1} of {len(plans)}: {p['family'].replace('_', ' ')}{cut}"
    return lab


def export(shard, e, out):
    meta = json.loads(Path(shard + ".json").read_text())
    kind = meta.get("kind", "cloth")
    name = f"{kind}_{Path(shard).parent.name}_e{e}"
    if kind == "rope":
        import l1_render_rope as m
        m.OUT = out / kind
        d = m.export(shard, e)
    else:
        import l1_render_cloth as m
        m.OUT = out / kind
        m.export(shard, [e])
        d = m.OUT / f"{Path(shard).name}_e{e}"
    target = out / kind / name
    if d != target:
        if target.exists():
            import shutil
            shutil.rmtree(target)
        d.rename(target)
    wd = meta["worlds"][e]
    clip = json.loads((target / "clip.json").read_text())
    clip["camera"] = frame(target, kind, wd)
    clip["timeline"] = timeline(wd, int(meta["settle_frames"]), clip["frames"])
    clip["n_actions"] = sum(1 for p in wd["families"] if p.get("family") != "held" and "skipped" not in p)
    (target / "clip.json").write_text(json.dumps(clip))
    return target


def frame(d, kind, wd):
    """A camera that holds the whole episode: what the body covers (1st-99th percentile over all frames) and where the
    gripper works near the table, plus the objects, seen from the exporters' angle (l1_render_rope's framing)."""
    p = np.load(d / "poses.npz")
    x = p[f"teacher_{kind}0"].reshape(-1, 3)
    grip = p["teacher"][:, 1, :3]
    low = grip[:, 2] - 0.055 < 0.3
    pts = np.concatenate([x, grip[low]] + [np.asarray(o["pos"])[None] for o in wd["obstacles"]])
    pts = pts[pts[:, 2] > -0.3] if (pts[:, 2] > -0.3).any() else pts          # frame the table, not the floor
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    ctr = 0.5 * (lo + hi)
    ctr[2] = 0.5 * (max(lo[2], -0.05) + min(hi[2], 0.35))
    span = float(np.linalg.norm(np.maximum(hi - lo, [0.3, 0.3, 0.1])))
    az, el = np.radians(-60.0), np.radians(35.0)
    dist = 0.62 * span / np.tan(np.radians(19.0))
    eye = ctr + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return {"eye": eye.tolist(), "target": ctr.tolist(), "fov": 38.0}


def blender(d, size, samples):
    subprocess.run([BLENDER, "--background", "--python", str(REPO / "scripts/blender_render.py"), "--", str(d), str(size[0]),
                    str(size[1]), str(samples)], check=True, stdout=subprocess.DEVNULL)
    print(f"[blender] {d.name}", flush=True)


def compose(d, out):
    r = __import__("l2_render")
    from PIL import Image
    clip = json.loads((d / "clip.json").read_text())
    accent = (31, 92, 70, 255) if clip["kind"] == "cloth" else (191, 110, 31, 255)
    comp = d / "composite"
    comp.mkdir(exist_ok=True)
    story = clip["story"].split(":")[0]                            # "starts <init>" (the families are in the timeline)
    for f in range(clip["frames"]):
        im = Image.open(d / "teacher" / f"f{f:04d}.png").convert("RGB")
        r._label(im, clip["labels"]["teacher"], [(f"t = {f / clip['fps']:5.2f} s   {story}, {clip['n_actions']} actions", r.WHITE),
                                                 (clip["timeline"][f], r.AMBER)], accent)
        im.save(comp / f"c{f:04d}.png")
    r.encode(comp, out / f"{d.name}.mp4", clip["fps"])
    times = np.linspace(0, clip["frames"] - 1, 10).astype(int)     # contact sheet: 10 instants, 5 x 2
    w, h = im.size
    tw, th = w // 2, h // 2
    sheet = Image.new("RGB", (5 * tw + 4 * 3, 2 * th + 3), (255, 255, 255))
    for j, f in enumerate(times):
        sheet.paste(Image.open(comp / f"c{f:04d}.png").resize((tw, th)), ((j % 5) * (tw + 3), (j // 5) * (th + 3)))
    sheet.save(out / f"{d.name}_sheet.jpg", quality=90)
    print(f"[compose] {out / (d.name + '.mp4')}, {out / (d.name + '_sheet.jpg')}", flush=True)


def main():
    args = sys.argv[1:]
    opt = {"--size": "640x480", "--samples": "32", "--out": "outputs/lsim/render_long"}
    for k in list(opt):
        if k in args:
            i = args.index(k)
            opt[k] = args[i + 1]
            args = args[:i] + args[i + 2:]
    out = REPO / opt["--out"]
    out.mkdir(parents=True, exist_ok=True)
    size = tuple(int(v) for v in opt["--size"].split("x"))
    what = args[0] if args[0] in ("export", "blender", "compose", "all") else "all"
    items = [(a.rsplit(":", 1)[0], int(a.rsplit(":", 1)[1])) for a in (args[1:] if what == args[0] else args)]
    for s, e in items:
        kind = json.loads(Path(s + ".json").read_text()).get("kind", "cloth")
        d = out / kind / f"{kind}_{Path(s).parent.name}_e{e}"
        if what in ("export", "all"):
            d = export(s, e, out)
        if what in ("blender", "all"):
            blender(d, size, int(opt["--samples"]))
        if what in ("compose", "all"):
            compose(d, out)


if __name__ == "__main__":
    main()
