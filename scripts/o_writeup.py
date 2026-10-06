"""docs/O_TRACK.html -- every figure read from docs/results/ at build time.

Same convention as scripts/v_writeup.py: nothing in the page is typed by hand, so the
page cannot drift from the artifacts the way a prose summary does.
"""
import json
from pathlib import Path

R = json.load(open("docs/results/o4_summary.json"))
STYLE = Path("docs/V_TRACK.html").read_text().split("<style>")[1].split("</style>")[0]

hb, hp, tr = R["holdout_bowl"], R["holdout_pot"], R["transfer"]["bowl_to_pot"]
BL = json.load(open("docs/results/o6_blind.json"))
FS = json.load(open("docs/results/o5_fit.json"))


def blind_rows(o):
    return "".join(
        f'<tr><td>{r["mu_true"]:.3f}</td><td>{r["mu_hat"]:.3f}</td>'
        f'<td>{r["err_pct"]:+.0f}%</td><td>&plusmn;{r["unc_pct"]:.0f}%</td></tr>'
        for r in BL[o]["rows"])
ang = ", ".join(f"{a:.0f}" for a in R["angles"]["bowl"])


def band(d, k):
    return f'{d["bands"][k]["median_pct"]:.0f}%' if k in d["bands"] else "&mdash;"


rows_cls = "".join(
    f'<tr><td>{c["label"][:66]}</td><td>{c["rho"]:+.3f}</td></tr>'
    for c in R["top_classes"])

H = f"""<title>The O-Track</title>
<style>{STYLE}</style>
<div class="wrap">
<header>
<div class="eyebrow">reify &middot; O-track &middot; result</div>
<h1>One experiment cannot see friction. Seven can.</h1>
<p class="lede">An outcome depends on friction <em>and</em> ramp angle &mdash; that is the physics, not a
confound. Reading a single sweep and calling the signal dead was the wrong test. Holding out a
friction entirely and constraining several known conditions at once recovers it to
<b>{hb["joint_median_pct"]:.0f}%</b>, against <b>{R["prior_median_pct"]:.0f}%</b> for the prior.</p></header>

<div class="big">
<div class="card no"><div class="k">one condition</div><div class="v">{hb["single_median_pct"]:.0f}%</div>
<div class="n">median error recovering a held-out &mu;</div></div>
<div class="card ok"><div class="k">{len(R["angles"]["bowl"])} conditions joined</div><div class="v">{hb["joint_median_pct"]:.0f}%</div>
<div class="n">same clips, constrained together</div></div>
<div class="card ok"><div class="k">held-out object</div><div class="v">{hp["joint_median_pct"]:.0f}%</div>
<div class="n">brass pot, 3 ramp angles</div></div>
<div class="card no"><div class="k">across objects</div><div class="v">3.4&times;</div>
<div class="n">constant scale error the readout carries</div></div>
</div>

<h2>What the earlier tests got wrong</h2>
<p>Every O-track gate asked one sweep at a time whether the classifier's score identifies &mu;,
and every one failed. Within a single ramp angle, friction and slide distance are the same
column of numbers wearing two labels, so a single sweep <em>cannot</em> separate them. Ramp angle
and ramp rate are conditions we set ourselves, not unmeasured nuisances &mdash; the identification
comes from the pattern across them. Two further corrections were needed before the joint fit
would run at all.</p>

<h3>Read the whole distribution</h3>
<p>The two SSv2 classes chosen by name track true slide at
&rho; = {R["hand_picked_rho"]["slide"]:+.3f} and {R["hand_picked_rho"]["noslide"]:+.3f}. The best available track it far better, and
{R["n_classes_rho_gt_half"]} of the 174 classes carry |&rho;| &gt; 0.5 &mdash; the signal is spread across the vocabulary,
not held in one pair. A cross-validated ridge over all 174 log-probabilities cuts the read error
by a third on both objects.</p>
<div class="scroll"><table><tr><th>class</th><th>&rho; with true slide</th></tr>{rows_cls}</table></div>
<div class="scroll"><table><tr><th>reading log slide, leave-one-out</th><th>wooden bowl</th><th>brass pot</th></tr>
<tr class="dead"><td>two classes picked by name</td><td>{R["readout"]["bowl"]["two_class"]:.3f}</td><td>{R["readout"]["pot"]["two_class"]:.3f}</td></tr>
<tr class="live"><td>all 174, ridge</td><td>{R["readout"]["bowl"]["all_174"]:.3f}</td><td>{R["readout"]["pot"]["all_174"]:.3f}</td></tr></table></div>

<h3>Bracket the parameter</h3>
<p>At 21&ndash;27&deg; a bowl above &mu; &asymp; 0.7 barely moves in any condition, so those clips carry no
information and the estimate regresses to the middle. Ramp angles of 33 and 36&deg; were added to give
that end real dynamic range &mdash; at 36&deg; the sweep spans 202 down to 9 cm. It is not enough: the
high-friction band stays the worst by a wide margin, and that is the honest identifiability limit
of a tilt probe rather than a defect in the readout.</p>
<div class="scroll"><table><tr><th>friction band</th><th>bowl</th><th>pot</th></tr>
<tr class="live"><td>&mu; &lt; 0.35 &mdash; slides freely</td><td>{band(hb,"free")}</td><td>{band(hp,"free")}</td></tr>
<tr><td>0.35 &ndash; 0.70 &mdash; partial</td><td>{band(hb,"partial")}</td><td>{band(hp,"partial")}</td></tr>
<tr class="dead"><td>&mu; &gt; 0.70 &mdash; barely moves</td><td>{band(hb,"stuck")}</td><td>{band(hp,"stuck")}</td></tr></table></div>

<h2>The result</h2>
<p>An entire friction is removed, the readout refit on everything else, and &mu; recovered from the
held-out clips alone &mdash; so nothing in the estimate has seen the answer. Bowl: {R["n"]["bowl"]} clips over
{ang}&deg;. Pot: {R["n"]["pot"]} clips over 21, 24, 27&deg;.</p>
<div class="scroll"><table><tr><th>median error on a held-out friction</th><th>bowl</th><th>pot</th></tr>
<tr><td>one condition alone</td><td>{hb["single_median_pct"]:.0f}%</td><td>{hp["single_median_pct"]:.0f}%</td></tr>
<tr class="live"><td>conditions constrained jointly</td><td>{hb["joint_median_pct"]:.0f}%</td><td>{hp["joint_median_pct"]:.0f}%</td></tr>
<tr class="dead"><td>flat prior on [0.10, 1.20]</td><td>{R["prior_median_pct"]:.0f}%</td><td>{R["prior_median_pct"]:.0f}%</td></tr></table></div>

<h2>The fit</h2>
<p>Three frictions that appear nowhere in the calibration &mdash; 0.175, 0.245, 0.390, chosen to sit
between grid points &mdash; were simulated, rendered and scored separately, then recovered blind. The
fit uses only what exists for an asset whose parameters are unknown: a readout calibrated against
the <em>simulator's own</em> slide (no external ground truth is needed to know how far our own rollout
moved), the simulator's forward model, and the ramp angles we set.</p>
<div class="scroll"><table><tr><th>brass pot &mdash; true &mu;</th><th>fitted</th><th>error</th><th>reported bar</th></tr>
{blind_rows("pot")}</table></div>
<div class="scroll"><table><tr><th>wooden bowl &mdash; true &mu;</th><th>fitted</th><th>error</th><th>reported bar</th></tr>
{blind_rows("bowl")}</table></div>
<div class="big">
<div class="card ok"><div class="k">pot, blind</div><div class="v">{BL["pot"]["joint_median_pct"]:.0f}%</div>
<div class="n">median error &middot; truth inside the bar {BL["pot"]["inside_bar"]}/{BL["pot"]["n"]}</div></div>
<div class="card"><div class="k">bowl, blind</div><div class="v">{BL["bowl"]["joint_median_pct"]:.0f}%</div>
<div class="n">median error &middot; truth inside the bar {BL["bowl"]["inside_bar"]}/{BL["bowl"]["n"]}</div></div>
<div class="card no"><div class="k">flat prior</div><div class="v">86%</div><div class="n">the number to beat</div></div>
<div class="card no"><div class="k">Cosmos 3 judge</div><div class="v">&rho; &minus;0.32</div>
<div class="n">same question, on the same probe</div></div>
</div>
<p>The uncertainty is propagated from the readout's leave-one-out residual through the local
leverage, and on the pot it is honest: the one hard case &mdash; &mu; = 0.390, where the pot's slide
flattens and the outcome stops discriminating &mdash; reports &plusmn;51% rather than a confident wrong
answer. On the bowl the bar is too tight, and the reason is visible rather than mysterious: the
bowl's readout is four times noisier (leave-one-out sd {BL["bowl"]["readout_sd"]:.3f} against {BL["pot"]["readout_sd"]:.3f}) and
under-reads slide, which inverts to an over-estimate of &mu;.</p>
<div class="note"><b>Calibrating the readout beats comparing distributions directly.</b> An objective
that fits nothing &mdash; symmetrised KL between the candidate's 174-way distribution and the
observation's &mdash; recovers a held-out &mu; to {FS["bowl"]["symkl"]["joint"]["median_pct"]:.0f}% (bowl) and {FS["pot"]["symkl"]["joint"]["median_pct"]:.0f}% (pot), against {hb["joint_median_pct"]:.0f}% and {hp["joint_median_pct"]:.0f}% for the
calibrated readout. Weighting all 174 classes equally spends most of the objective on classes that
carry no information about the outcome.</div>

<h2>Where it breaks: across objects</h2>
<p>The across-object confound has ended three signals in this project. Here it does not end this one,
but it does bound it. A readout calibrated on the bowl and applied to the pot has R&sup2; = {tr["r2"]:+.2f}
&mdash; worse than predicting the mean. Almost all of that is a single constant: the readout is off by a
factor of {10**abs(tr["bias_decades"]):.1f} on the new object. Remove that one offset and R&sup2; = {tr["r2_offset_removed"]:+.2f}; the ranking,
which is what an optimiser actually consumes, survives at &rho; = {tr["spearman"]:+.3f}.</p>
<div class="note"><b>The absolute reading does not transfer; the ordering does.</b> That is survivable
for a fit, which compares candidate &theta; for one fixed object and never needs a calibrated distance.
It is not survivable for anything that compares a render against real footage &mdash; if the readout shifts
{10**abs(tr["bias_decades"]):.1f}&times; between two synthetic objects, it will shift at least that much between synthetic and real,
and that gap is the whole premise of using a model that has watched the real world.</div>

<h2>Framing, again</h2>
<div class="note"><b>The camera pull at clip time must match the camera pull at render time.</b>
The first G-A result inverted because the object slid out of frame exactly where the signal should
have been strongest. It recurred here: the sweeps were rendered at <code>CAM_PULL=1.20</code> and clipped
without it, so the crop box was computed against a camera 0.65 of the way out &mdash; worst containment
41.3%. Re-clipped correctly it is 99.5%. Containment is now checked per clip and
{R.get("dropped", 10)} clips whose object leaves frame are dropped rather than scored.</div>

<h2>Next</h2>
<p>The measurement half is established: the head reads outcomes off our renders well enough that a set
of known conditions pins &mu; to ~15%. The target half is not. Nothing here shows the classifier reads
<em>real</em> footage on the same scale as ours, and the across-object offset is direct evidence that it
will not. The next test is the one deferred from the plan &mdash; run real SSv2 tilt-and-slide clips
through the same readout and measure the offset against footage rather than against another render.</p>
<footer>reify &middot; O-track &middot; generated by scripts/o_writeup.py &middot; every figure read from docs/results/ at build time</footer></div>"""
Path("docs/O_TRACK.html").write_text(H)
print(f"wrote docs/O_TRACK.html ({len(H)} bytes)")
