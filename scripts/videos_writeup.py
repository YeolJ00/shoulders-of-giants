import json, sys
from pathlib import Path
S = Path(sys.argv[1])
P = json.loads((S / "vid" / "index.json").read_text())
STYLE = Path("/home/nas5/jooyeolyun/repos/simulation-assestization/docs/V_TRACK.html"
             ).read_text().split("<style>")[1].split("</style>")[0]

SEC = [
 ("friction", "Friction", "a ramp held at a fixed angle",
  "The ramp is <em>already</em> tilted and the object is released on it. Each tilt is swept "
  "around its own slip threshold, where the distance travelled is most sensitive to "
  "friction. Same object, same ramp, same camera &mdash; only &mu; changes.",
  "Read from pixels on 7 assets: <b>&mu; recovered to 2&ndash;8%</b> with a whole friction "
  "held out, against 16&ndash;23% when the ramp was <em>tilting</em> instead of tilted."),
 ("damping", "Contact damping", "drop it and watch the bounce",
  "The bowl falls onto the table. Damping is how much of that energy the contact keeps. "
  "It changes nothing about the fall and everything about what happens after.",
  "Read from pixels on 7 assets, 27 clips each: <b>cd recovered to a 16% median</b> "
  "against ~700% for a flat prior, and joining drop heights beats a single height on 5 of "
  "7 assets."),
 ("com", "Centre of mass", "tilt it until it topples",
  "Same vase, same mass, same friction &mdash; only where the weight sits inside it. "
  "Uphill and it survives further; downhill and it goes over early.",
  "<b>These clips predate a fix</b> and are being re-measured: the pose was left in the "
  "simulator's level frame while the renderer tilted the table, so the vase rides off the "
  "ramp. The simulator-side reading is unaffected."),
 ("mass", "Mass", "weigh it against a known reference",
  "Mass cannot be read from how something slides &mdash; a shoved body travels "
  "v&sup2;/2&mu;g whatever it weighs. So it is weighed. Watch which way the beam goes as "
  "the reference block grows past the subject.",
  "Read from pixels: the beam direction is called correctly on <b>34 of 36</b> clips, and "
  "the flip point recovers mass to <b>9%</b>."),
 ("assets", "Five assets, one probe", "the same tilt, matched conditions",
  "Every asset scaled to the same width, given the same mass, filmed through one shared "
  "crop box. Each gets its own readout calibrated against the simulator's own slide.",
  "Per-asset readouts work on every asset, leave-one-out factor "
  "<b>1.11&times;&ndash;1.18&times;</b> on the tilted ramp. How many tilts an asset tolerates "
  "before its own base friction torques it over is asset-dependent &mdash; 5 for the sorter, "
  "2 for the bowl &mdash; and the simulator says which before anything is rendered."),
]

def card(p):
    return (f'<figure><video src="vid/{p["file"]}" muted loop playsinline preload="metadata" '
            f'controls></video>'
            f'<figcaption><b>{p["label"]}</b><span>{p["sub"]}</span></figcaption></figure>')

body = ""
for key, title, sub, blurb, kick in SEC:
    vids = "".join(card(p) for p in P if p["group"] == key)
    body += (f'<h2>{title}</h2><h3>{sub}</h3><p>{blurb}</p>'
             f'<div class="grid">{vids}</div><div class="kick">{kick}</div>')

EXTRA = """
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));margin:18px 0}
figure{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
 overflow:hidden}
figure video{width:100%;display:block;background:#000;aspect-ratio:544/448;object-fit:cover}
figcaption{padding:9px 11px;font-family:var(--mono);font-size:12px;line-height:1.5}
figcaption b{display:block;font-size:13.5px}
figcaption span{color:var(--ink-3)}
.kick{font-size:14.5px;color:var(--ink);background:var(--sunk);border-radius:2px;
 padding:12px 16px;margin:6px 0 0;max-width:40rem}
.hint{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);margin:6px 0 0}
"""

H = f"""<title>Four Parameters on Video</title>
<style>{STYLE}{EXTRA}</style>
<div class="wrap">
<header>
<div class="eyebrow">reify &middot; &theta; &middot; the clips themselves</div>
<h1>What each parameter does, as filmed</h1>
<p class="lede">These are the actual rendered clips the classifier was scored on &mdash; not
illustrations of them. In every row one parameter moves and everything else is held: same
asset, same lighting, same camera, one shared crop box.</p>
<p class="hint">{len(P)} clips &middot; press play, or scrub &middot; Cycles, 544&times;448, 64 frames at 24 fps</p>
</header>
{body}
<h2>Why it is filmed this way</h2>
<div class="note"><b>One crop box per probe, never one per clip.</b> A box computed per clip
grows with whatever the object did, so the measurement would read the framing &mdash; which
varies with the parameter. Twice in this project a per-clip or per-lab box inverted a result
outright. Every clip in a row above shares one box, and each one is checked for containment:
an object that leaves frame is dropped rather than scored.</div>
<div class="note"><b>The ramp is really tilted now, not faked with gravity.</b> The contact
plane used to be a horizontal height test, so an incline could only be made by rotating
gravity and leaving the ground level &mdash; which split the world into a level simulator
frame and a tilted render frame, and produced four wrong-frame bugs, the last of which
rendered objects floating 11.5 cm off the ramp. A real tilted plane keeps one frame for both,
drops the rotating-frame approximation, and removes the ramp <em>rate</em> &mdash; which at
fixed friction was a 9&ndash;11&times; lever on the outcome all by itself.</div>
<footer>reify &middot; generated by scripts/videos_writeup.py &middot; clips re-encoded from
outputs/scene/*/clips at 520 px</footer></div>"""
(S / "videos.html").write_text(H)
print(f"wrote {S}/videos.html ({len(H)} bytes), {len(P)} videos")
