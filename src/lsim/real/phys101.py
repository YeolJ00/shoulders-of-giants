"""Physics 101 real ramp slides: the first real-world check for teachers and models.

Raw videos: data/phys101/phys101_v1.0.zip (git-ignored, 4.5 GB, phys101.csail.mit.edu).
Tracked slides (CoTracker3 on the side view, metres along a 20 deg board) were extracted by the
theta track (archive/theta_track/scripts/p1_ramp_friction.py, p2_ramp_all.py) and are kept as
small derived files in data/real/:
  phys101_ramp_tracks_cardboard.json   93 trials, 9 cardboard cubes (p1 format, used here)
  phys101_ramp_tracks_all.json         all 15 materials (p2 format)
"""
import json
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ZIP = REPO / "data/phys101/phys101_v1.0.zip"
TRACKS = {"cardboard": REPO / "data/real/phys101_ramp_tracks_cardboard.json",
          "all": REPO / "data/real/phys101_ramp_tracks_all.json"}


def objects():
    """{object: (mass_g, volume_cm3)} as measured by the dataset authors."""
    z = zipfile.ZipFile(ZIP)

    def table(name):
        return {l.split()[0]: float(l.split()[1]) for l in z.read(name).decode().split("\n") if l.strip()}

    mass, vol = table("phys101/objects/mass"), table("phys101/objects/vol")
    return {k: (mass[k], vol[k]) for k in mass if k in vol}


def ramp_tracks(kind="cardboard"):
    return json.loads(TRACKS[kind].read_text())


def ramp_segment(trial):
    """(t, s): the on-ramp part of one tracked trial, s in metres along the ramp."""
    t = np.arange(len(trial["s"])) / trial["fps"]
    seg = slice(max(trial["start"] - 1, 0), trial["peak"] + 1)
    return t[seg], np.asarray(trial["s"])[seg]


def align_err(real_t, real_s, sim_s, sim_fps=30.0):
    """Release-aligned fit error: min over release time t0 (61 values in [t_1 - 0.25, t_1 + 0.05])
    and offset s0 (closed form) of RMS(real - (s0 + sim(t - t0))). Returns (rms, t0, s0).
    Only release time and offset are aligned, never the acceleration."""
    tg = np.arange(len(sim_s)) / sim_fps
    best = None
    for t0 in np.linspace(real_t[0] - 0.25, real_t[0] + 0.05, 61):
        tau = real_t - t0
        if tau.min() < 0:
            continue
        sim = np.interp(tau, tg, sim_s)
        s0 = float(np.mean(real_s - sim))
        e = float(np.sqrt(np.mean((real_s - s0 - sim) ** 2)))
        if best is None or e < best[0]:
            best = (e, float(t0), s0)
    return best
