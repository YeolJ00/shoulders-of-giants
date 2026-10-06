"""Solver settings per probe rig, in ONE place (they used to be hard-coded per script).

Settings are validated jointly, not individually (CLAUDE.md), and a probe must run at the
rig it was validated on -- so a probe imports its rig rather than restating numbers.

TILT RIG -- tilted-plane sliding and toppling (o14, p4, f7). Stiff contact (pen_frac 1e-4)
because sliding accuracy is what it measures: fitted to real Physics 101 footage, Newton
reproduces real slides to 0.4-1.0 mm at this setting (scripts/p4_sim_loop.py).
RESTITUTION, changed 2026-10-06: raw cd could not make this rig bounce realistically --
every cd from 50 to 30000 gave e = 0.77-1.00, so impacts were near-elastic and toppling
bodies bounced like superballs (scripts/p7_cd_restitution.py). Damping is now a Hunt-
Crossley alpha, cd = 1.5 alpha k / HC per body (MeshProbeScene.set_restitution_alpha),
which gives the same e on both rigs and for any mass (scripts/p8_alpha.py).
alpha = 0.4 s/m -> e ~ 0.54 at a 2 m/s impact, ~0.70 at 1 m/s: a mid-range DEFAULT for
wood / plastic / ceramic, an ASSUMPTION until real restitution is measured (Physics 101's
30 fps drops could not resolve it).

BOUNCE RIG -- drop / bounce / settle (probe_bounce_settle). Softer contact, coarser step;
raw cd is still its knob, so its validated results stand. Note: there too, e depends on
cd / mass, not cd -- a single cd gives bodies of different mass different restitution.
"""
TILT_RIG = {"pen_frac": 1.0e-4, "fps": 24.0, "substeps": 420, "alpha": 0.4}
BOUNCE_RIG = {"pen_frac": 5.0e-3, "fps": 24.0, "substeps": 60}


def apply_tilt_rig(scene):
    """Stiffness and restitution of the tilt rig on a constructed MeshProbeScene."""
    scene.set_restitution_alpha(TILT_RIG["alpha"])
    scene.calibrate_stiffness(pen_frac=TILT_RIG["pen_frac"])
    return scene
