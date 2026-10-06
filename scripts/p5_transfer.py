"""P5 -- from real measurements to OUR assets: friction and density by material.

Two parts.

A. TEST OUR FRICTION RULE ON REAL DATA. The simulator combines a contact pair as
   mu_pair = sqrt(mu_A * mu_B). Every real Physics 101 slide gives TWO pair frictions: the
   object on the ramp board (p1/p2) and, after the ramp, the same object decelerating on the
   TABLE before it reaches the target. If the rule holds,
       mu_pair(i, table) / mu_pair(i, board) = sqrt(mu_table / mu_board)
   is the SAME constant for every material i. Table segment: from 2 frames after peak speed
   (past the ramp-table junction) until speed falls below 30% of peak (before the stop at the
   target); v = v0 - a t, mu_pair(table) = a / g.

B. TRANSFER. Pair frictions fix each material's own mu only up to ONE common scale (mu_i * c,
   mu_surface / c gives the same pairs), so one anchor is unavoidable. Stated once:
       ANCHOR: our scene's wooden table (ground_mu = 0.45 in every probe) behaves like the
       Physics 101 ramp board, i.e. mu_pair(material, our table) = measured mu_pair(material,
       board). Then body mu = mu_pair^2 / 0.45.
   Density: Physics 101's EFFECTIVE density (mass / enclosing volume -- its cardboard
   volumes are exact enclosing cubes) times each asset's ENCLOSED volume at its authored
   (real) size, so mass = rho_eff * V uses one volume definition throughout.
   Every asset gets a match grade: exact / close proxy / friction-only / not covered.

    W scripts/p5_transfer.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
G = 9.81
TABLE_MU_SIM = 0.45
OUT = REPO / "docs/results/p5_transfer.json"

# asset -> (Physics 101 material, match grade, note)
MAP = {
    "cardboard_box_01": ("cardboard", "exact", ""),
    "Threshold_Porcelain_Teapot_White": ("porcelain", "exact", "hollow, mass assumed"),
    "ceramic_vase_01": ("porcelain", "close", "ceramic ~ porcelain; hollow, mass assumed"),
    "wooden_bowl_01": ("w_block", "close", "turned solid-wood walls (wall volume, open bowl)"),
    "Weisshai_Great_White_Shark": ("p_toy", "close", "plastic toy"),
    "Schleich_Lion_Action_Figure": ("p_toy", "close", "plastic toy"),
    "Great_Dinos_Triceratops_Toy": ("p_toy", "close", "plastic toy"),
    "Pony_C_Clamp_1440": ("m_pole", "close", "solid metal"),
    "Poppin_File_Sorter_Blue": ("p_block", "friction-only", "thin plastic shell; p_block is solid"),
    "brass_pot_01": ("m_pole", "friction-only", "no hollow-metal class for density"),
    "book_encyclopedia_set_01": (None, "not covered", "no paper/book class"),
    "baseball_01": (None, "not covered", "no leather/ball class; rolls anyway"),
    "food_apple_01": (None, "not covered", "no organic class"),
}


def table_mu(r):
    s = np.array(r["s"]); fps = r["fps"]
    v = np.gradient(s) * fps
    k = r["peak"]
    seg = [i for i in range(k + 2, len(v)) if v[i] > 0.3 * v[k]]
    if len(seg) < 4:
        return None
    t = np.array(seg) / fps
    a = -np.polyfit(t, v[seg], 1)[0]
    return float(a / G) if a > 0 else None


def part_a():
    mat = lambda c: c.split("/")[0].rsplit("_", 1)[0]
    rows = []
    for r in json.loads((REPO / "docs/results/p1_ramp_friction.json").read_text()):
        if "/20_" in r["clip"] and r.get("fit"):
            rows.append((mat(r["clip"]), r["fit"]["mu"], table_mu(r)))
    for r in json.loads((REPO / "docs/results/p2_ramp_all.json").read_text()):
        if "/20_" in r["clip"] and r.get("outcome") == "slides" and not r.get("impossible"):
            rows.append((mat(r["clip"]), r["mu"], table_mu(r)))
    res = {}
    print("A. friction rule on real data: mu_pair(table) / mu_pair(board) per material")
    print(f"  {'material':<11}{'n':>4}{'board':>8}{'table':>8}{'ratio':>8}")
    for m in sorted({x[0] for x in rows}):
        b = np.array([x[1] for x in rows if x[0] == m and x[2] is not None])
        tb = np.array([x[2] for x in rows if x[0] == m and x[2] is not None])
        if len(b) < 3:
            continue
        ratio = float(np.median(tb / b))
        res[m] = {"n": int(len(b)), "mu_board": float(np.median(b)),
                  "mu_table": float(np.median(tb)), "ratio": ratio}
        print(f"  {m:<11}{len(b):>4}{np.median(b):>8.3f}{np.median(tb):>8.3f}{ratio:>8.2f}")
    r = np.array([v["ratio"] for v in res.values()])
    print(f"  ratio across materials: median {np.median(r):.2f}, spread (max/min) {r.max()/r.min():.2f}x, "
          f"CV {r.std()/r.mean():.2f}   (rule predicts one constant)")
    return res


def enclosed_volume(asset):
    from src.data.assets import load_asset
    import f14_llm_outcomes as F
    cat, _, nm = ("rigid/" + asset).rpartition("/")
    m = load_asset(cat, nm)
    V = np.asarray(m.vertices)
    pitch = float(np.ptp(V, 0).max()) / 80
    import trimesh
    pts = np.asarray(trimesh.sample.sample_surface(m, 400000, seed=0)[0])   # seeded: rule 7
    lo = V.min(0) - pitch
    ijk = np.floor((pts - lo) / pitch).astype(int)
    shell = np.zeros(ijk.max(0) + 2, bool); shell[tuple(ijk.T)] = True
    solid = F._fill(shell)
    return float(solid.sum() * pitch ** 3), np.ptp(V, 0)


# HOW EACH ASSET IS BUILT decides how mass transfers. Measured on the nine cardboard cubes:
# mass / volume varies 35% (0.135 -> 0.046 g/cm3 as they grow) while mass / AREA is constant,
# 0.106 g/cm2, CV 0.03 -- a hollow object's mass scales with its surface, so a small box's
# "effective density" grossly over-weighs a big one (4.1 kg for our 39x52x34 cm box).
#   areal   measured areal density x mesh surface area      (cardboard: measured)
#   solid   measured volumetric density x enclosed volume  (toys, clamp, solid-wood bowl walls)
#   shell   NOT MEASURED: area x assumed wall thickness x handbook material density, a range
CARDBOARD_AREAL = None    # filled from the cubes at run time
BUILD = {"cardboard_box_01": "areal",
         "Threshold_Porcelain_Teapot_White": ("shell", 0.3, 0.5, 2.3, 2.4),
         "ceramic_vase_01": ("shell", 0.3, 0.6, 2.0, 2.4),
         "brass_pot_01": ("shell", 0.08, 0.15, 8.4, 8.7),
         "Poppin_File_Sorter_Blue": ("shell", 0.15, 0.25, 0.9, 1.2)}


def cardboard_areal():
    import zipfile
    z = zipfile.ZipFile(REPO / "data/phys101/phys101_v1.0.zip")
    P = lambda f: {l.split()[0]: float(l.split()[1]) for l in z.read(f).decode().split("\n")
                   if l.strip()}
    mass, vol = P("phys101/objects/mass"), P("phys101/objects/vol")
    ra = np.array([mass[k] / (6 * vol[k] ** (2 / 3)) for k in mass if k.startswith("cardboard")])
    return float(np.median(ra)), float(ra.std() / ra.mean())


def part_b(A):
    D = json.loads((REPO / "docs/results/p101_density.json").read_text())
    S = json.loads((REPO / "docs/results/p2_ramp_summary.json").read_text())
    cb = [r["fit"]["mu"] for r in json.loads((REPO / "docs/results/p1_ramp_friction.json").read_text())
          if "/20_" in r["clip"] and r.get("fit")]
    mu_board = {m: v["mu_median"] for m, v in S.items() if v.get("mu_median")}
    mu_board["cardboard"] = float(np.median(cb))
    print(f"\nB. transfer (ANCHOR: our table ~ the ramp board; body mu = mu_pair^2 / {TABLE_MU_SIM})")
    print(f"  {'asset':<34}{'material':<11}{'grade':<14}{'mu_pair':>8}{'body mu':>8}"
          f"  {'build':<7}mass")
    out = {}
    for asset, (m, grade, note) in MAP.items():
        vol, ext = enclosed_volume(asset)
        row = {"material": m, "grade": grade, "note": note, "V_enclosed_cm3": vol * 1e6,
               "extent_cm": (ext * 100).round(1).tolist()}
        if m and m in mu_board:
            row["mu_pair"] = mu_board[m]; row["mu_body"] = mu_board[m] ** 2 / TABLE_MU_SIM
        from src.data.assets import load_asset
        area_cm2 = float(load_asset("rigid", asset).area * 1e4)
        b = BUILD.get(asset, "solid")
        row["area_cm2"] = area_cm2
        if b == "areal":
            ar, cv = cardboard_areal()
            row.update({"build": "areal (measured)", "areal_g_cm2": ar, "areal_cv": cv,
                        "mass_g": ar * area_cm2})
        elif isinstance(b, tuple):
            _, t0, t1, r0, r1 = b
            row.update({"build": f"shell (ASSUMED wall {t0*10:.1f}-{t1*10:.1f} mm, "
                                 f"rho {r0}-{r1} g/cm3; mesh area may double-count open "
                                 f"containers' inner+outer surfaces)",
                        "mass_g_range": [area_cm2 * t0 * r0, area_cm2 * t1 * r1]})
        elif m and grade in ("exact", "close") and m in D:
            row.update({"build": "solid (measured density)", "rho_eff": D[m]["median"],
                        "mass_g": D[m]["median"] * vol * 1e6,
                        "mass_g_range": [D[m]["min"] * vol * 1e6, D[m]["max"] * vol * 1e6]})
        out[asset] = row
        f = lambda k, w, p: f"{row[k]:>{w}.{p}f}" if k in row else f"{'-':>{w}}"
        mtxt = (f"{row['mass_g']:>7.0f} g" if "mass_g" in row else
                f"{row['mass_g_range'][0]:>5.0f}-{row['mass_g_range'][1]:.0f} g ASSUMED"
                if "mass_g_range" in row else "        -")
        print(f"  {asset:<34}{str(m):<11}{grade:<14}{f('mu_pair', 8, 3)}{f('mu_body', 8, 3)}"
              f"  {row.get('build', '-').split(' (')[0]:<7}{mtxt}")
    return out


# Assets with NO Physics 101 class. A specification is as good as a measurement; a published
# bulk density is a reference, not ours; anything else is stated as not measured.
REFERENCE = {
    "baseball_01": {"mass_g": [142, 149], "mass_src": "rulebook specification (5-5.25 oz)",
                    "cor": [0.514, 0.578], "cor_src": "rulebook specification 0.546 +- 0.032, "
                    "against ash wood at 58 mph -- far faster than a table drop"},
    "food_apple_01": {"rho": [0.75, 0.85], "rho_src": "published bulk density (apples float); "
                      "reference, not measured here"},
    "book_encyclopedia_set_01": {"rho": [0.6, 0.9], "rho_src": "published density of bound "
                                 "books/paper; reference, not measured here"},
}


def cd_for_restitution(e, mass_kg):
    """contact damping cd giving restitution e on the BOUNCE-RIG settings (pen_frac 5e-3,
    60 substeps), from the measured curve e(cd / mass) (scripts/p7_cd_restitution.py: the
    0.1 and 0.4 kg curves coincide in cd/m). The tilt-rig settings cannot represent e < 0.77
    at all, so no cd is returned for them."""
    D = json.loads((REPO / "docs/results/p7_cd_restitution.json").read_text())
    xs, es = [], []
    for k, v in D.items():
        if k.startswith("bounce rig"):
            m = float(k.rsplit(", ", 1)[1].split(" ")[0])     # label itself contains ", "
            xs += [c / m for c in v["cd"]]; es += v["e"]
    o = np.argsort(xs); xs = np.log(np.array(xs)[o]); es = np.array(es)[o]
    # e falls monotonically with cd/m: interpolate log(cd/m) as a function of e
    oo = np.argsort(es)
    return float(mass_kg * np.exp(np.interp(e, es[oo], xs[oo])))


def part_c(B):
    print("\nC. assets with no Physics 101 class")
    for asset, ref in REFERENCE.items():
        row = B[asset]
        if "mass_g" in ref:
            row["mass_g_range"] = ref["mass_g"]; row["mass_src"] = ref["mass_src"]
        if "rho" in ref:
            v = row["V_enclosed_cm3"]
            row["mass_g_range"] = [ref["rho"][0] * v, ref["rho"][1] * v]
            row["mass_src"] = ref["rho_src"]
        if "cor" in ref:
            mkg = np.mean(row["mass_g_range"]) / 1000
            row["cor"] = ref["cor"]; row["cor_src"] = ref["cor_src"]
            row["cd_bounce_rig"] = [cd_for_restitution(e, mkg) for e in ref["cor"][::-1]]
        row["friction"] = "not measured"
        print(f"  {asset:<26} mass {row['mass_g_range'][0]:.0f}-{row['mass_g_range'][1]:.0f} g  "
              f"({row['mass_src']})" + (f"\n  {'':<26} COR {ref['cor'][0]}-{ref['cor'][1]} -> cd "
              f"{row['cd_bounce_rig'][0]:.0f}-{row['cd_bounce_rig'][1]:.0f} on the bounce rig"
              if "cor" in ref else ""))
    return B


if __name__ == "__main__":
    A = part_a()
    B = part_c(part_b(A))
    OUT.write_text(json.dumps({"rule_test": A, "assets": B,
                               "anchor": "our table ~ Physics 101 ramp board; body mu = "
                                         "mu_pair^2 / 0.45"}, indent=1))
    print("wrote", OUT)
