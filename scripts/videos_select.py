"""Pick a representative clip set and re-encode it small for an artifact gallery."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

def spec(lab):
    return json.loads((REPO / "outputs/scene" / lab / "spec.json").read_text())

def clip(lab, key):
    for sub in ("clips", "clips_b"):
        p = REPO / "outputs/scene" / lab / sub / f"{key}.mp4"
        if p.exists():
            return p
    return None

def enc(src, dst, w=520):
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", f"scale={w}:-2", "-c:v", "libx264", "-crf", "23",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-an",
                    "-movflags", "+faststart", str(dst)], check=True)

PICK = []

def take(lab, rows, label, sub, group, fmt):
    for r in rows:
        c = clip(lab, r["key"])
        if c is None:
            print("  MISSING", lab, r["key"]); continue
        name = f"{group}_{len(PICK):02d}.mp4"
        enc(c, OUT / name)
        PICK.append({"file": name, "group": group, "label": label(r), "sub": sub(r),
                     "value": r["value"], "obs": r["slide_cm"],
                     "kb": round((OUT / name).stat().st_size / 1024)})
        print(f"  {group:<10} {name}  {PICK[-1]['label']:<22} {PICK[-1]['kb']:>5} KB")

# ---- friction: TILTED ramp, one asset, one tilt, low -> high mu
# Only clips that actually slid: a clip that left the table, lifted off the plane or
# tumbled is a different event and is excluded from the fit as well.
def clean(rows):
    return [r for r in rows
            if not (r.get("off_table") or r.get("lifted") or r.get("tumbled"))]

s0 = clean([r for r in spec("t_sorter") if r["tilt"] == 24.0])
s0 = sorted(s0, key=lambda r: r["value"])
take("t_sorter", [s0[0], s0[len(s0)//3], s0[2*len(s0)//3], s0[-1]],
     lambda r: f"mu = {r['value']:.2f}", lambda r: f"slides {abs(r['slide_cm']):.0f} cm",
     "friction", None)

# ---- damping: cd sweep at one drop height
s = sorted(spec("p_bounce"), key=lambda r: r["value"])
take("p_bounce", [s[0], s[2], s[4], s[-1]], lambda r: f"cd = {r['value']:.0f}",
     lambda r: f"rebounds {r['slide_cm']:.0f} mm", "damping", None)

# ---- centre of mass: offsets at one ramp angle
s = sorted([r for r in spec("p_tip2") if r["tilt"] == 36.0], key=lambda r: r["value"])
take("p_tip2", [s[0], s[4], s[-5], s[-1]],
     lambda r: f"CoM {1000*r['value']:+.0f} mm",
     lambda r: "uphill" if r["value"] < 0 else ("centred" if r["value"] == 0 else "downhill"),
     "com", None)

# ---- mass: the beam flipping as the reference passes the subject
s = sorted([r for r in spec("p_bal") if r["value"] == 0.18], key=lambda r: r["ref"])
sel = [s[0], s[4], s[5], s[-1]]
take("p_bal", sel, lambda r: f"reference {1000*r['ref']:.0f} g",
     lambda r: "subject side down" if r["slide_cm"] > 0 else "reference side down", "mass", None)

# ---- five assets, same probe, matched tilt
for tag in ("bowl", "pot", "teapot", "sorter", "shark"):
    g = clean([r for r in spec(f"t_{tag}") if r["tilt"] == 19.0])
    if not g:
        g = clean(spec(f"t_{tag}"))
    g = sorted(g, key=lambda r: r["value"])
    r = g[len(g)//2]
    take(f"t_{tag}", [r], lambda rr, t=tag: t,
         lambda rr: f"mu {rr['value']:.2f}, slides {abs(rr['slide_cm']):.0f} cm",
         "assets", None)

(OUT / "index.json").write_text(json.dumps(PICK, indent=1))
tot = sum(p["kb"] for p in PICK)
print(f"\n{len(PICK)} clips, {tot/1024:.1f} MB total")
