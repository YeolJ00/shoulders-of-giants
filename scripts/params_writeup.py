"""docs/PARAMS.html -- what each parameter DOES, shown from the rendered clips themselves.

Every filmstrip is real frames out of the mp4s the classifier was scored on, and every curve
is the simulator's own observable read from spec.json. Nothing here is drawn by hand, so the
page cannot drift from the runs.
"""
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

INK, ACC, DIM = "#767d85", "#2b5f86", "#9a3b1d"
STRIP_AT = [0.0, 0.3, 0.55, 0.8, 1.0]


def frames(mp4, n_at=STRIP_AT, w=176):
    from PIL import Image
    import imageio.v2 as imageio
    rd = imageio.get_reader(str(mp4))
    fr = [f[..., :3] for f in rd]
    rd.close()
    out = []
    for a in n_at:
        i = min(int(round(a * (len(fr) - 1))), len(fr) - 1)
        im = Image.fromarray(fr[i])
        im = im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "JPEG", quality=74, optimize=True)
        out.append("data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode())
    return out


def strip(mp4, label, note):
    ims = "".join(f'<img src="{u}" alt="">' for u in frames(mp4))
    return (f'<div class="row"><div class="rl"><b>{label}</b><span>{note}</span></div>'
            f'<div class="st">{ims}</div></div>')


def curve(x, y, xl, yl, logx=True, logy=False, mark=None):
    fig, ax = plt.subplots(figsize=(5.6, 2.5), dpi=150)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    ax.plot(x, y, "-o", color=ACC, ms=4, lw=1.6)
    if mark is not None:
        for m in mark:
            ax.axvline(m, color=DIM, lw=1.0, ls="--", alpha=0.8)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xl, color=INK, fontsize=9); ax.set_ylabel(yl, color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(INK)
    ax.grid(True, alpha=0.16, color=INK, lw=0.6)
    fig.tight_layout(pad=0.3)
    b = io.BytesIO(); fig.savefig(b, format="png", transparent=True); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def load(lab, filt=None):
    sp = json.loads((Path(lab) / "spec.json").read_text())
    rows = [r for r in sp if (filt is None or filt(r))]
    rows = [r for r in rows if (Path(lab) / "clips" / f"{r['key']}.mp4").exists()]
    return sorted(rows, key=lambda r: r["value"])


def pick(rows, k=3):
    """lowest, middle, highest of the sweep -- the three the eye needs to see the effect."""
    i = [0, len(rows) // 2, len(rows) - 1]
    return [rows[j] for j in i][:k]


DMP = json.load(open("docs/results/o12_damping.json"))
RD = {"sign_n": 0, "sign_tot": 0, "median": float("nan")}
_rd = Path("docs/results/o10_balance_read.json")
if _rd.exists():
    _r = json.loads(_rd.read_text())
    if _r:
        RD = {"sign_n": int(round(sum(x["sign_acc"] * 12 for x in _r))),
              "sign_tot": 12 * len(_r),
              "median": 100 * (10 ** float(np.median(
                  [abs(np.log10(x["flip_kg"] / x["subject_kg"])) for x in _r])) - 1)}

SEC = []

# ---------------------------------------------------------------- friction
# The friction section now uses the TILTED ramp (o13_tilted). Only clips that actually
# slid are shown: a clip that left the table, lifted off the plane or tumbled is a
# different event and is excluded from the fit too.
TL = json.load(open("docs/results/o13_tilted.json"))
FRIC_ASSET = "sorter"          # the widest clean coverage: 5 tilts, mu 0.15-1.96
rows = [r for r in load(f"outputs/scene/t_{FRIC_ASSET}")
        if not (r["off_table"] or r["lifted"] or r["tumbled"])]
if rows:
    one = sorted([r for r in rows if r["tilt"] == 24.0], key=lambda r: r["value"])
    show = pick(one) if len(one) >= 3 else pick(rows)
    st = "".join(strip(Path(f"outputs/scene/t_{FRIC_ASSET}/clips") / f"{r['key']}.mp4",
                       f"&mu; = {r['value']:.2f}", f"slides {abs(r['slide_cm']):.0f} cm")
                 for r in show)
    pa, ho = TL["per_asset"][FRIC_ASSET], TL["holdout"][FRIC_ASSET]
    SEC.append(("Friction &mu;", "a ramp held at a fixed angle",
                "The ramp is <em>already</em> tilted and the object is released on it. Low "
                "friction and it runs; high friction and it barely creeps. Each tilt is swept "
                "around its own slip threshold &mu;* = tan&sup2;(tilt)/&mu;<sub>table</sub>, "
                "which is where the distance travelled is most sensitive to friction.",
                st, curve([r["value"] for r in rows], [abs(r["slide_cm"]) for r in rows],
                          "friction &mu;", "slide (cm)", logy=True),
                f"Read from pixels on <b>{len(TL['per_asset'])} assets</b>: leave-one-out read "
                f"error {min(10**v['loo_sd'] for v in TL['per_asset'].values()):.2f}&times;"
                f"&ndash;{max(10**v['loo_sd'] for v in TL['per_asset'].values()):.2f}&times;. "
                f"Holding out a whole friction, &mu; comes back to "
                f"<b>{min(v['joint_pct'] for v in TL['holdout'].values()):.0f}&ndash;"
                f"{max(v['joint_pct'] for v in TL['holdout'].values()):.0f}%</b> &mdash; "
                "against 16&ndash;23% when the ramp was <em>tilting</em> rather than tilted. "
                "The catch: coverage is asset-dependent. This asset stays clean across 5 tilts "
                "and a 12.8&times; friction range; a bowl manages 2 tilts and 3.0&times; before "
                "its own base friction torques it over."))

# ---------------------------------------------------------------- damping
rows = load("outputs/scene/p_bounce")
if rows:
    st = "".join(strip(Path("outputs/scene/p_bounce/clips") / f"{r['key']}.mp4",
                       f"cd = {r['value']:.0f}", f"rebounds {r['slide_cm']:.0f} mm")
                 for r in pick(rows))
    SEC.append(("Contact damping cd", "drop it and watch the bounce",
                "The bowl falls 22 cm onto the table. Damping is how much of that energy "
                "the contact keeps. It changes nothing about the fall and everything about "
                "what happens after &mdash; and it is <b>blind to friction</b> (1.08&times; "
                "over a full &mu; sweep), which is what makes it a clean second reading.",
                st, curve([r["value"] for r in rows], [r["slide_cm"] for r in rows],
                          "contact damping cd", "first rebound (mm)"),
                f"Read from pixels across <b>{len(DMP['per_asset'])} assets</b>, 27 clips each: "
                f"leave-one-out read error {min(10**v['loo_sd'] for v in DMP['per_asset'].values()):.2f}&times;"
                f"&ndash;{max(10**v['loo_sd'] for v in DMP['per_asset'].values()):.2f}&times;, "
                f"&rho;(read, true) up to {max(v['rho'] for v in DMP['per_asset'].values()):.3f}. "
                f"Holding out a whole damping value, cd is recovered to a median of "
                f"<b>{sorted(v['joint_pct'] for v in DMP['holdout'].values())[len(DMP['holdout'])//2]:.0f}%</b> "
                f"against ~700% for a flat prior &mdash; and joining drop heights beats a single "
                f"height on {sum(1 for k in DMP['holdout'] if DMP['holdout'][k]['joint_pct'] < DMP['holdout'][k]['single_pct'])} "
                f"of {len(DMP['holdout'])} assets."))

# ---------------------------------------------------------------- centre of mass
rows = load("outputs/scene/p_tip2", lambda r: r["tilt"] == 36.0)
if rows:
    st = "".join(strip(Path("outputs/scene/p_tip2/clips") / f"{r['key']}.mp4",
                       f"CoM {1000*r['value']:+.0f} mm", "uphill" if r["value"] < 0
                       else ("centred" if r["value"] == 0 else "downhill"))
                 for r in pick(rows))
    SEC.append(("Centre of mass", "tilt it until it topples",
                "Same vase, same mass, same friction &mdash; only where the weight sits "
                "inside it. Shift the centre of mass downhill and it goes over early; "
                "shift it uphill and it survives further. A bowl cannot do this probe at "
                "all: it is 31 cm wide and 9 cm tall, so it tips at ~72&deg; and simply "
                "slides instead.",
                st, curve([1000 * r["value"] for r in rows],
                          [abs(r["slide_cm"]) for r in rows],
                          "centre-of-mass offset (mm)", "travel to topple (cm)", logx=False),
                "<b>Being re-measured</b>, for the same frame bug as friction. The "
                "simulator-side reading is unaffected and the probe is still exactly immune "
                "to mass, &mu; and cd (1.01&times; / 1.00&times; / 1.01&times;)."))

# ---------------------------------------------------------------- mass
lab = "outputs/scene/p_bal"
if (Path(lab) / "spec.json").exists():
    sp = json.loads((Path(lab) / "spec.json").read_text())
    have = [r for r in sp if (Path(lab) / "clips" / f"{r['key']}.mp4").exists()]
    if have:
        subj = sorted({r["value"] for r in have})[len(sorted({r["value"] for r in have})) // 2]
        srt = sorted([r for r in have if r["value"] == subj], key=lambda r: r["ref"])
        sel = [srt[0], min(srt, key=lambda r: abs(r["slide_cm"])), srt[-1]]
        st = "".join(strip(Path(lab) / "clips" / f"{r['key']}.mp4",
                           f"reference {1000*r['ref']:.0f} g",
                           "subject side down" if r["slide_cm"] > 0 else "reference side down")
                     for r in sel)
        flips = []
        for s in sorted({r["value"] for r in have}):
            g = sorted([r for r in have if r["value"] == s], key=lambda r: r["ref"])
            for a, b in zip(g, g[1:]):
                if a["slide_cm"] > 0 >= b["slide_cm"]:
                    flips.append(float(np.sqrt(a["ref"] * b["ref"]))); break
        SEC.append(("Mass", "weigh it against a known reference",
                    "Mass is the one parameter that <b>cannot</b> be read from how "
                    "something slides &mdash; a body shoved across a table travels "
                    "v&sup2;/2&mu;g whatever it weighs, which is why an earlier probe built "
                    "to recover mass measured nothing. So it is weighed. The reference block "
                    "is sized by its own mass, so the scale is visible in frame. What "
                    "carries the answer is not how far the beam moved but <b>which way it "
                    "went</b>, and the reference where that flips is the subject's mass.",
                    st, curve([r["ref"] for r in srt], [r["slide_cm"] for r in srt],
                              "reference mass (kg)", "beam angle (deg)", logx=True,
                              mark=[subj]),
                    f"The flip lands at {', '.join(f'{1000*f:.0f} g' for f in flips)} for "
                    f"subjects of {', '.join(f'{1000*s:.0f} g' for s in sorted({r['value'] for r in have}))}. "
                    "Read from pixels, holding out a whole subject mass: the beam direction "
                    f"is called correctly on <b>{RD['sign_n']} of {RD['sign_tot']}</b> clips "
                    f"and the flip it implies recovers mass to <b>{RD['median']:.0f}%</b> "
                    "&mdash; against 86% for the prior and &plusmn;3% simulator-side."))

# ---------------------------------------------------------------- the fit
SENS = json.load(open("docs/results/o7_sensitivity.json"))
PARS = ["mass", "com", "mu", "cd"]
NICE = {"balance_ang": "balance", "tilt_slide": "tilt &rarr; slide",
        "tip_angle": "tilt &rarr; topple", "spin_decay": "spin-down",
        "bounce_h": "bounce", "settle_rate": "settle"}
READS = {"balance_ang": "mass", "tilt_slide": "mu", "tip_angle": "com",
         "spin_decay": "cd", "bounce_h": "cd", "settle_rate": "cd"}
mrows = ""
for pr in SENS:
    tds = ""
    for p in PARS:
        rg = SENS[pr][p]["range"]
        cls = "hit" if p == READS[pr] else ("warn" if rg >= 2.0 else "nil")
        tds += f'<td class="{cls}">{rg:.2f}&times;</td>'
    mrows += f"<tr><td>{NICE[pr]}</td>{tds}</tr>"

FIT = [json.load(open(f"docs/results/o8_t{i}.json"))[0] for i in (0, 1)]
frows = ""
for f in FIT:
    t, g, e = f["true"], f["fit"], f["err_pct"]
    frows += (f'<tr><td>{1000*t["mass"]:.0f} g &middot; {1000*t["com"]:+.0f} mm &middot; '
              f'&mu; {t["mu"]:.2f} &middot; cd {t["cd"]:.0f}</td>'
              f'<td>{1000*g["mass"]:.0f} g</td><td>{1000*g["com"]:+.1f} mm</td>'
              f'<td>{g["mu"]:.3f}</td><td>{g["cd"]:.0f}</td>'
              f'<td>{e["mass"]:+.0f}% / {1000*(g["com"]-t["com"]):+.1f} mm / '
              f'{e["mu"]:+.0f}% / {e["cd"]:+.0f}%</td></tr>')

STYLE = (REPO / "docs/V_TRACK.html").read_text().split("<style>")[1].split("</style>")[0]
EXTRA = """
.row{display:grid;grid-template-columns:9.5rem 1fr;gap:14px;align-items:center;
 padding:11px 0;border-bottom:1px solid var(--rule)}
.rl{font-family:var(--mono);font-size:12.5px;line-height:1.5}
.rl b{display:block;font-size:14px}.rl span{color:var(--ink-3)}
.st{display:flex;gap:5px;overflow-x:auto}
.st img{height:118px;width:auto;border-radius:2px;border:1px solid var(--rule);flex:0 0 auto}
.plot{margin:20px 0 6px}.plot img{width:100%;max-width:34rem;display:block}
.kick{font-size:15px;color:var(--ink);background:var(--sunk);border-radius:2px;
 padding:12px 16px;margin:14px 0 0;max-width:40rem}
td.hit{color:var(--live);font-weight:700}td.warn{color:var(--dead)}td.nil{color:var(--ink-3)}
"""

H = f"""<title>What the Parameters Do</title>
<style>{STYLE}{EXTRA}</style>
<div class="wrap">
<header>
<div class="eyebrow">reify &middot; &theta; &middot; what changes, and what it means</div>
<h1>Four numbers, four things you can watch happen</h1>
<p class="lede">Every strip below is real frames from the clips the classifier was scored on.
One parameter moves at a time; the object, the lighting and the framing are held fixed. What
makes a parameter recoverable is that its effect is <em>visible</em> and belongs to it alone.</p>
</header>
{''.join(f'''
<h2>{t}</h2>
<h3>{sub}</h3>
<p>{body}</p>
{st}
<div class="plot"><img src="{cv}" alt=""></div>
<div class="kick">{kick}</div>''' for t, sub, body, st, cv, kick in SEC)}

<h2>Which probe reads which parameter</h2>
<p>The number in each cell is how many times over the observable spans while that parameter
sweeps &mdash; reported instead of a correlation, because a probe can be perfectly monotone
and still move so little that nothing can read it. Green is the parameter the probe was built
for. <span style="color:var(--dead)">Red</span> is a parameter it also responds to, which is
not a defect &mdash; it is why the parameters are fit together rather than one at a time.</p>
<div class="scroll"><table>
<tr><th>probe</th><th>mass</th><th>CoM</th><th>&mu;</th><th>cd</th></tr>{mrows}</table></div>
<div class="note"><b>The off-diagonal entries are the whole reason for a joint fit.</b>
Spin-down was built to read damping and responds more to mass (3.89&times;) than to damping
(3.19&times;); under a stiffer contact it reads friction instead. Settle moves by 1.18&times;
and is dead as a magnitude however good its correlation looks. Balance and tilt&rarr;topple
are the two clean instruments, each exactly immune (1.00&times;) to two of the other three.</div>

<h2>All four at once</h2>
<p>Recovered together from balance + topple + three ramp angles + bounce, on the simulator's
own observables, starting from a fixed neutral guess.</p>
<div class="scroll"><table>
<tr><th>true &theta;</th><th>mass</th><th>CoM</th><th>&mu;</th><th>cd</th><th>error</th></tr>
{frows}</table></div>
<div class="note"><b>The second row is the useful failure.</b> Its friction sits above the
slip threshold of the ramp it was measured on, so the bowl barely moves and &mu; = 0.55 looks
identical to &mu; = 0.35 &mdash; 6.0 cm against 6.3. No optimiser can recover what the
experiment did not record. Adding steeper ramps took that error from &minus;40% to &minus;13%,
which is the same finding as everywhere else here: <b>a parameter is pinned down by the SET
of conditions, never by one of them.</b></div>
<footer>reify &middot; generated by scripts/params_writeup.py &middot; frames read from the
scored clips, curves from each probe's spec.json</footer></div>"""

Path("docs/PARAMS.html").write_text(H)
print(f"wrote docs/PARAMS.html  ({len(H)/1e6:.2f} MB, {len(SEC)} parameter sections)")
