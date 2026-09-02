"""The clips themselves: docs/V1_CLIPS.html, every frame inlined.

Artifacts and file:// pages cannot fetch, so each clip is embedded as a base64 GIF
(two-pass palette, the scripts/iter_report.py pattern). Kept small and subsampled so the
whole gallery stays under the artifact size cap.

    /home/jooyeolyun/anaconda3/envs/warp/bin/python scripts/v1_clips_page.py
"""
import base64
import json
import subprocess
import tempfile

import numpy as np
from pathlib import Path

LAB = Path("outputs/scene/v1")
OUT = Path("docs/V1_CLIPS.html")
WIDTH, FPS = 190, 10

GROUPS = [
    ("bounce", "cd", "Bounce — 8 cm centre drop, contact damping swept 32×",
     "Low cd rings and bounces; high cd lands dead. The simulator reads this at ρ = −0.995."),
    ("settle", "cd", "Settle — tipped onto its dome and released",
     "Every clip starts at the same 22° tip, so only the decay differs. ρ = +0.973 sim-side."),
    ("tilt", "mu", "Tilt — continuous ramp to 24°, friction swept",
     "Slides 57 cm at μ=0.10 and 1.9 cm at μ=0.35."),
    ("null", "mass", "NULL CONTROL — mass swept 32× on the same ramp",
     "Slip is v²/(2μg): mass cannot change it, and the sim confirms 17.8→19.9 cm. "
     "These clips should be indistinguishable. Surprise moves 9.9σ across them."),
    ("violation", "", "CONTROLS — authored impossibilities on the reference track",
     "Same asset, lighting, framing and crop as the legal clips. Only the motion is "
     "impossible."),
]


def gif(mp4, width=WIDTH, fps=FPS):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as d:
        pal, out = f"{d}/p.png", f"{d}/o.gif"
        vf = f"fps={fps},scale={width}:-1:flags=lanczos"
        subprocess.run([ff, "-y", "-i", str(mp4), "-vf", vf + ",palettegen", pal],
                       check=True, capture_output=True)
        subprocess.run([ff, "-y", "-i", str(mp4), "-i", pal, "-lavfi",
                        vf + " [x]; [x][1:v] paletteuse", out],
                       check=True, capture_output=True)
        return "data:image/gif;base64," + base64.b64encode(Path(out).read_bytes()).decode()


CSS = """
:root{--ink:#191d1b;--ink-2:#4a534f;--ink-3:#6f7873;--rule:#d8d9d2;--bg:#f4f4ef;
 --panel:#fff;--panel-2:#eceee7;--signal:#b4451f;--ref:#2f6f8f;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ink:#e6e8e3;--ink-2:#a8b0ab;--ink-3:#7b837e;--rule:#2f3531;--bg:#141715;
 --panel:#1b1f1d;--panel-2:#232825;--signal:#e0763f;--ref:#6fb0cf}}
:root[data-theme="dark"]{--ink:#e6e8e3;--ink-2:#a8b0ab;--ink-3:#7b837e;--rule:#2f3531;
 --bg:#141715;--panel:#1b1f1d;--panel-2:#232825;--signal:#e0763f;--ref:#6fb0cf}
:root[data-theme="light"]{--ink:#191d1b;--ink-2:#4a534f;--rule:#d8d9d2;--bg:#f4f4ef;
 --panel:#fff;--panel-2:#eceee7;--signal:#b4451f;--ref:#2f6f8f}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;
 margin:0;padding:0 20px 80px}
.wrap{max-width:74rem;margin:0 auto}
header{padding:48px 0 22px;border-bottom:1px solid var(--rule)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
 text-transform:uppercase;color:var(--ink-3)}
h1{font-family:var(--serif);font-size:clamp(28px,4.6vw,42px);margin:.3em 0 .15em;
 font-weight:600;text-wrap:balance}
.lede{font-size:17px;color:var(--ink-2);max-width:46rem;text-wrap:pretty}
h2{font-family:var(--serif);font-size:22px;margin:46px 0 3px;font-weight:600}
h2 .tag{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
 color:#fff;background:var(--signal);padding:3px 8px;border-radius:2px;
 vertical-align:middle;margin-left:10px}
.sub{color:var(--ink-3);font-size:14px;margin:0 0 16px;max-width:52rem;text-wrap:pretty}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(196px,1fr))}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:8px}
.card img{width:100%;display:block;border-radius:2px;background:var(--panel-2)}
.k{font-family:var(--mono);font-size:12px;margin-top:7px;font-variant-numeric:tabular-nums}
.k b{color:var(--signal)}
.m{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);
 font-variant-numeric:tabular-nums}
footer{margin-top:60px;padding-top:16px;border-top:1px solid var(--rule);
 font-family:var(--mono);font-size:11.5px;color:var(--ink-3)}
"""


def main():
    rows = {r["key"]: r for r in
            json.loads(Path("docs/results/v1_surprise.json").read_text())["rows"]}
    P = ["<title>V1 Clips</title>", f"<style>{CSS}</style>", '<div class="wrap">',
         "<header>", '<div class="eyebrow">reify · V-track · what the model was shown</div>',
         "<h1>The clips behind the kill test</h1>",
         '<p class="lede">One object, one parameter at a time, fixed appearance and a crop '
         "box shared across each sweep so framing cannot vary with θ. Every clip is 64 "
         "frames at 24 fps.</p></header>"]

    for probe, param, title, blurb in GROUPS:
        rs = sorted([r for r in rows.values() if r["probe"] == probe],
                    key=lambda r: r["value"])
        if not rs:
            continue
        if probe in ("bounce", "settle", "tilt") and len(rs) > 5:
            idx = np.linspace(0, len(rs) - 1, 5).round().astype(int)
            rs = [rs[i] for i in sorted(set(idx))]   # ends and middle of the sweep
        tag = ' <span class="tag">control</span>' if probe in ("null", "violation") else ""
        P.append(f"<h2>{title}{tag}</h2><p class='sub'>{blurb}</p><div class='grid'>")
        for r in rs:
            clip = Path(r["clip"])
            if not clip.exists():
                continue
            lbl = (r["param"] if probe == "violation"
                   else f"{param} = {r['value']:g}")
            P.append(f"<div class='card'><img src='{gif(clip)}' alt='{lbl}'>"
                     f"<div class='k'><b>{lbl}</b></div>"
                     f"<div class='m'>surprise {r['surprise']:.4f}"
                     f" · motion {r['motion']:.2f}</div></div>")
            print("  " + r["key"], flush=True)
        P.append("</div>")

    P.append("<footer>reify · V1 · generated by scripts/v1_clips_page.py</footer></div>")
    OUT.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
