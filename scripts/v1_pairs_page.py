"""Show each matched pair side by side: docs/V1_PAIRS.html.

The point is to let a person check the premise the whole tier rests on -- that the
"impossible" member actually looks impossible. Tier 1 scores at chance, and before
concluding anything about the model it is worth confirming the clips are what we think
they are. The seating bug was found exactly this way.
"""
import base64, json, subprocess, sys, tempfile
from pathlib import Path

LAB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/scene/v1")
SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/results/v1_protocol_vitl_f64.json")
OUT = Path("docs/V1_PAIRS.html")
W, FPS = 200, 12


def gif(mp4):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as d:
        pal, out = f"{d}/p.png", f"{d}/o.gif"
        vf = f"fps={FPS},scale={W}:-1:flags=lanczos"
        subprocess.run([ff, "-y", "-i", str(mp4), "-vf", vf + ",palettegen", pal],
                       check=True, capture_output=True)
        subprocess.run([ff, "-y", "-i", str(mp4), "-i", pal, "-lavfi",
                        vf + " [x]; [x][1:v] paletteuse", out], check=True, capture_output=True)
        return "data:image/gif;base64," + base64.b64encode(Path(out).read_bytes()).decode()


CSS = """
:root{--ink:#191d1b;--ink-2:#4a534f;--ink-3:#6f7873;--rule:#d8d9d2;--bg:#f4f4ef;
 --panel:#fff;--bad:#b4451f;--good:#2f6f8f;
 --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --serif:"Iowan Old Style",Palatino,Georgia,serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ink:#e6e8e3;--ink-2:#a8b0ab;--ink-3:#7b837e;--rule:#2f3531;--bg:#141715;
 --panel:#1b1f1d;--bad:#e0763f;--good:#6fb0cf}}
:root[data-theme="dark"]{--ink:#e6e8e3;--ink-2:#a8b0ab;--ink-3:#7b837e;--rule:#2f3531;
 --bg:#141715;--panel:#1b1f1d;--bad:#e0763f;--good:#6fb0cf}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);margin:0;
 padding:0 20px 70px;line-height:1.5}
.wrap{max-width:72rem;margin:0 auto}
header{padding:44px 0 20px;border-bottom:1px solid var(--rule)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
 color:var(--ink-3)}
h1{font-family:var(--serif);font-size:clamp(26px,4.4vw,40px);margin:.3em 0 .2em;font-weight:600}
.lede{font-size:17px;color:var(--ink-2);max-width:44rem;text-wrap:pretty}
h2{font-family:var(--serif);font-size:20px;margin:40px 0 12px;font-weight:600}
.pair{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:center;
 background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:10px;
 margin-bottom:12px}
.pair img{width:100%;display:block;border-radius:2px}
.cap{font-family:var(--mono);font-size:11px;color:var(--ink-3);margin-top:6px}
.cap b{color:var(--ink)}
.verdict{font-family:var(--mono);font-size:12px;text-align:right;min-width:8.5rem}
.verdict .d{font-size:20px}
.hit{color:var(--good)} .miss{color:var(--bad)}
p{max-width:44rem;color:var(--ink-2);text-wrap:pretty}
footer{margin-top:50px;padding-top:14px;border-top:1px solid var(--rule);
 font-family:var(--mono);font-size:11px;color:var(--ink-3)}
"""


def main():
    rows = json.loads(SRC.read_text())["rows"]
    viol = [(k, r) for k, r in rows.items()
            if r["probe"] == "violation" and r.get("pair_with") in rows]
    viol.sort(key=lambda kv: (kv[1]["param"], kv[1]["pair_with"]))
    P = ["<title>V1 Matched Pairs</title>", f"<style>{CSS}</style>", '<div class="wrap">',
         "<header>", '<div class="eyebrow">reify · V-track · tier 1</div>',
         "<h1>Does the impossible clip look impossible?</h1>",
         '<p class="lede">Left: a legal rollout. Right: the same rollout with an authored '
         "violation — identical asset, lighting, framing and crop, only the motion is "
         "impossible. Tier 1 scores at chance (63.3%, CI [45.5, 78.1] over 30 pairs) "
         "against 96.7–100% for the published benchmark, so the premise is worth "
         "checking by eye before blaming the model.</p></header>"]
    by_kind = {}
    for k, r in viol:
        by_kind.setdefault(r["param"], []).append((k, r))
    for kind, items in by_kind.items():
        P.append(f"<h2>{kind}</h2>")
        for k, r in items[:3]:
            src = r["pair_with"]
            a, b = LAB / "clips" / f"{src}.mp4", LAB / "clips" / f"{k}.mp4"
            if not (a.exists() and b.exists()):
                continue
            d = 100 * (r["max"] - rows[src]["max"]) / rows[src]["max"]
            cls = "hit" if d > 0 else "miss"
            P.append(
                f'<div class="pair">'
                f'<div><img src="{gif(a)}"><div class="cap">legal · <b>{src}</b></div></div>'
                f'<div><img src="{gif(b)}"><div class="cap">violation · <b>{kind}</b></div></div>'
                f'<div class="verdict {cls}"><div class="d">{d:+.2f}%</div>'
                f'{"detected" if d > 0 else "MISSED"}</div></div>')
            print("  ", k, flush=True)
    P.append("<footer>reify · scripts/v1_pairs_page.py</footer></div>")
    OUT.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
