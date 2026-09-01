"""Build the V1 page: docs/V1_SURPRISE.html (self-contained, nothing fetched).

    /home/jooyeolyun/anaconda3/envs/warp/bin/python scripts/v1_html.py

Reads docs/results/{v1_verdict,v1_surprise,v0_surprise}.json, the PNGs beside them, and
a handful of clips which it inlines as GIFs (Artifacts and file:// pages cannot fetch,
so every byte has to be in the document -- same constraint as scripts/iter_report.py).
"""
import base64
import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path

R = Path("docs/results")
OUT = Path("docs/V1_SURPRISE.html")


def b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def gif(mp4, width=210, fps=12):
    """mp4 -> inlined GIF, two-pass palette so the bowl does not band."""
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
        return b64(out, "image/gif")


CSS = """
:root{
  --ink:#191d1b; --ink-2:#4a534f; --ink-3:#6f7873; --rule:#d8d9d2;
  --bg:#f4f4ef; --panel:#ffffff; --panel-2:#eceee7;
  --signal:#b4451f; --ref:#2f6f8f; --ok:#3f6b45; --warn:#96631a;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ink:#e6e8e3; --ink-2:#a8b0ab; --ink-3:#7b837e; --rule:#2f3531;
  --bg:#141715; --panel:#1b1f1d; --panel-2:#232825;
  --signal:#e0763f; --ref:#6fb0cf; --ok:#7fb98a; --warn:#d3a24a;
}}
:root[data-theme="dark"]{
  --ink:#e6e8e3; --ink-2:#a8b0ab; --ink-3:#7b837e; --rule:#2f3531;
  --bg:#141715; --panel:#1b1f1d; --panel-2:#232825;
  --signal:#e0763f; --ref:#6fb0cf; --ok:#7fb98a; --warn:#d3a24a;
}
:root[data-theme="light"]{
  --ink:#191d1b; --ink-2:#4a534f; --rule:#d8d9d2;
  --bg:#f4f4ef; --panel:#ffffff; --panel-2:#eceee7;
  --signal:#b4451f; --ref:#2f6f8f; --ok:#3f6b45; --warn:#96631a;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;
     margin:0;padding:0 20px 80px}
.wrap{max-width:60rem;margin:0 auto}
header{padding:52px 0 26px;border-bottom:1px solid var(--rule)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--ink-3)}
h1{font-family:var(--serif);font-size:clamp(30px,5vw,46px);line-height:1.05;margin:.28em 0 .12em;
   text-wrap:balance;font-weight:600}
.lede{font-size:18px;color:var(--ink-2);max-width:46rem;text-wrap:pretty}
h2{font-family:var(--serif);font-size:24px;margin:52px 0 4px;font-weight:600}
h2+.sub{color:var(--ink-3);font-size:14px;margin:0 0 18px}
.verdict{display:flex;align-items:baseline;gap:12px;margin:26px 0 6px;flex-wrap:wrap}
.stamp{font-family:var(--mono);font-size:13px;letter-spacing:.1em;text-transform:uppercase;
  padding:5px 11px;border-radius:2px;color:#fff;background:var(--signal)}
.stamp.ok{background:var(--ok)}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));margin:18px 0}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:15px 16px}
.card .k{font-family:var(--mono);font-size:11px;letter-spacing:.11em;text-transform:uppercase;
  color:var(--ink-3)}
.card .v{font-family:var(--mono);font-size:27px;font-variant-numeric:tabular-nums;margin:5px 0 1px}
.card .n{font-size:13px;color:var(--ink-2)}
.bar{height:3px;background:var(--panel-2);margin-top:11px;border-radius:2px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--signal)}
figure{margin:20px 0;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  padding:12px}
figure img{width:100%;display:block;border-radius:2px}
figcaption{font-size:13px;color:var(--ink-2);margin-top:10px;text-wrap:pretty}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:13px;
  font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:7px 11px;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600}
tr.hi td{color:var(--signal)}
.clips{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.clip{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:9px}
.clip img{width:100%;display:block;border-radius:2px}
.clip .cap{font-family:var(--mono);font-size:11px;color:var(--ink-2);margin-top:7px}
p{max-width:46rem;text-wrap:pretty}
.note{border-left:2px solid var(--warn);padding:2px 0 2px 15px;color:var(--ink-2);
  margin:20px 0;max-width:46rem}
code{font-family:var(--mono);font-size:.9em;background:var(--panel-2);padding:1px 5px;
  border-radius:2px}
footer{margin-top:64px;padding-top:18px;border-top:1px solid var(--rule);
  font-family:var(--mono);font-size:11.5px;color:var(--ink-3)}
"""


def card(k, v, n, frac=None):
    bar = f'<div class="bar"><i style="width:{min(100, max(2, 100*frac)):.0f}%"></i></div>' \
        if frac is not None else ""
    return f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>' \
           f'<div class="n">{n}</div>{bar}</div>'


def main():
    V = json.loads((R / "v1_verdict.json").read_text())
    rows = json.loads((R / "v1_surprise.json").read_text())["rows"]
    sigma, fmean = V["sigma"], V["floor_mean"]
    stats, verdict = V["stats"], V["verdict"]
    passed = [k for k, ok in verdict.items() if ok]
    ctrl = V["controls"]

    head = "PASS" if passed else "FAIL"
    lede = (f"Surprise moves with θ on {', '.join(passed)}."
            if passed else
            "Surprise does not move with θ above the render noise floor on any axis "
            "we can currently render.")

    P = [f'<!-- V1 -->', f'<title>V1 Kill Test</title>', f"<style>{CSS}</style>",
         '<div class="wrap">', "<header>",
         '<div class="eyebrow">reify · V-track · V1</div>',
         "<h1>Does predictive-model surprise depend on θ?</h1>",
         f'<p class="lede">{lede} The criteria below were fixed before the sweep ran.</p>',
         '<div class="verdict">',
         f'<span class="stamp {"ok" if passed else ""}">kill test {head}</span>',
         f'<span class="stamp {"ok" if ctrl["violations_pass"] else ""}">'
         f'violations {"fire" if ctrl["violations_pass"] else "do not fire"}</span>',
         f'<span class="stamp {"ok" if ctrl["null_pass"] else ""}">'
         f'mass null {"flat" if ctrl["null_pass"] else "not flat"}</span>',
         "</div></header>"]

    # ---- headline numbers
    P.append("<h2>The axes</h2>")
    P.append('<p class="sub">Spread across the sweep, measured in units of the render '
             'noise floor σ. Anything under 3σ is not a reading.</p>')
    P.append('<div class="grid">')
    for probe in ("bounce", "settle", "tilt", "null"):
        st = stats.get(probe)
        if not st:
            continue
        note = "null control — must stay flat" if probe == "null" else \
            f"ρ = {st['rho_vs_dist']:+.2f} vs |log θ − log θref|"
        P.append(card(f"{probe} · {st['param']}", f"{st['spread_sigma']:.1f}σ",
                      note, min(st["spread_sigma"] / 12.0, 1.0)))
    P.append(card("noise floor σ", f"{sigma:.4f}",
                  f"seed {V['sigma_seed']:.4f} · jitter {V['sigma_jitter']:.4f}"))
    P.append("</div>")

    for fig, cap in [("v1_surprise_vs_theta.png",
                      "Surprise against each swept parameter, with the noise floor drawn "
                      "on as a band and the reference θ marked. The right panel is the "
                      "violation control."),
                     ("v0_speed_dependence.png",
                      "V0 calibration. Within one clip, slowing it destroys the harness's "
                      "ability to tell it from a frame-shuffled version — but pooled "
                      "across clips that correlation vanishes, which is the across-content "
                      "confound this project already knows about, reproduced in the new "
                      "signal.")]:
        if (R / fig).exists():
            P.append(f'<figure><img src="{b64(R / fig, "image/png")}" alt="">'
                     f"<figcaption>{cap}</figcaption></figure>")

    # ---- per-axis tables
    P.append("<h2>Every clip</h2>")
    P.append('<p class="sub">σ is distance from the floor mean. Motion is mean '
             '|frame difference| — carried here because surprise tracks it.</p>')
    P.append('<div class="scroll"><table><tr><th>clip</th><th>θ</th>'
             "<th>surprise</th><th>σ from floor</th><th>motion</th></tr>")
    order = {"bounce": 0, "settle": 1, "tilt": 2, "null": 3, "violation": 4, "floor": 5}
    for r in sorted(rows, key=lambda r: (order.get(r["probe"], 9), r["value"])):
        z = (r["surprise"] - fmean) / sigma if sigma else float("nan")
        hi = ' class="hi"' if r["probe"] == "violation" else ""
        val = r["param"] if r["probe"] in ("violation", "floor") else f"{r['value']:.3f}"
        P.append(f"<tr{hi}><td>{r['probe']} · {val}</td><td>{val}</td>"
                 f"<td>{r['surprise']:.5f}</td><td>{z:+.1f}</td>"
                 f"<td>{r['motion']:.2f}</td></tr>")
    P.append("</table></div>")

    # ---- clips
    picks = []
    for probe, want in (("bounce", "min"), ("bounce", "max"), ("violation", None)):
        rs = [r for r in rows if r["probe"] == probe]
        if not rs:
            continue
        if want == "min":
            picks.append(("lowest cd (bounciest)", min(rs, key=lambda r: r["value"])))
        elif want == "max":
            picks.append(("highest cd (deadest)", max(rs, key=lambda r: r["value"])))
        else:
            for r in rs[:3]:
                picks.append((f"violation: {r['param']}", r))
    if picks:
        P.append("<h2>What the model was shown</h2>")
        P.append('<div class="clips">')
        for cap, r in picks:
            if Path(r["clip"]).exists():
                P.append(f'<div class="clip"><img src="{gif(r["clip"])}" alt="">'
                         f'<div class="cap">{cap}<br>surprise {r["surprise"]:.5f}</div></div>')
        P.append("</div>")

    # ---- judge baseline
    jf = R / "v1_judge.json"
    if jf.exists():
        J = json.loads(jf.read_text())
        P.append("<h2>The judge, on the same clips</h2>")
        P.append('<p class="sub">Identical data, identical framing, one-bit threshold '
                 "questions the simulator already knows the answer to.</p>")
        P.append('<div class="scroll"><table><tr><th>clip</th><th>question</th>'
                 "<th>p(yes)</th><th>margin</th></tr>")
        for r in J:
            P.append(f"<tr><td>{r['key']}</td><td>{r['question'][:52]}</td>"
                     f"<td>{r['p_yes']:.3f}</td><td>{r['margin']:+.2f}</td></tr>")
        P.append("</table></div>")

    P.append("<h2>How to read this</h2>")
    P.append(
        "<p>The simulator makes every motion here; the model only reacts to it. That is "
        "the whole architecture — hallucination cannot enter θ because no external "
        "model ever manufactures a trajectory.</p>"
        "<div class='note'><b>The caveat that matters.</b> A θ that damps harder also "
        "<i>moves less</i>, and V0 showed surprise tracks motion within a scene. So a "
        "monotone surprise-vs-θ curve may be reading how much moved rather than how "
        "implausible it was. The two are told apart by shape: a plausibility signal has a "
        "minimum inside the sweep; a motion-mediated artefact is monotone. "
        "<code>rho_vs_motion</code> in the verdict JSON is the number to watch.</div>"
        "<p>The noise floor is not a guess. It is the spread of surprise over "
        "re-renders of one identical rollout, with only a Cycles sampling seed or a "
        "3&nbsp;mm camera shift changed — nuisances no physics depends on.</p>")

    P.append(f"<footer>reify · V1 kill test · {date.today().isoformat()} · "
             f"{len(rows)} clips · facebook/vjepa2-vitl-fpc64-256 · "
             "generated by scripts/v1_html.py</footer></div>")
    OUT.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
