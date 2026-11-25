"""L1 -- SimReady colliders (scripts/l1_simready_extract.py) -> GSO-format scene objects (lsim.data.simready).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES="" $L -I scripts/l1_simready_convert.py [--workers 16] [--limit N] [--only name,name]

CPU only. Reads data/assets/simready/raw/, writes data/assets/simready/{proc,parts}/, manifest.json, joints.json.
Resumable per asset (delete proc/<name>.row.json to redo one).

Per asset:
  collision  ALLSIDES: its lod-1 scan when that is closed, else the authored CoACD hulls. Others: the authored
             colliders; each collider that is not closed (open shells, the toaster and cabinet panels) is replaced by
             our CoACD parts of it. The closed pieces together are the collision mesh (overlapping closed shells:
             a particle meets the outer surface first, which is all the penalty contact needs).
  rest pose  the convex hull's face groups; a pose is stable when the centre of mass projects inside its face with
             a tipping angle >= 5 deg. The authored pose is kept when its bottom face is stable; otherwise the stable
             face with the lowest centre of mass. CoM: the authored composite when every part has one, else the
             uniform solid's.
  checks     closed (every collision piece watertight after decimation), size (largest extent 3-60 cm, smallest
             >= 8 mm: a table object), stable (a stable face was found).
"""
import json
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data/assets/simready"
RAW = ROOT / "raw"
MAX_FACES = 2000
TIP_DEG = 5.0


def pieces(d, key):
    import trimesh
    v, f, nv, nf = d[f"{key}_v"], d[f"{key}_f"], d[f"{key}_nv"], d[f"{key}_nf"]
    out, o, fo = [], 0, 0
    for a, b in zip(nv, nf):
        m = trimesh.Trimesh(np.asarray(v[o:o + a], np.float64), np.asarray(f[fo:fo + b]), process=True)
        o, fo = o + a, fo + b
        m.merge_vertices(merge_tex=True, merge_norm=True)
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        out.append(m)
    return out


def closed(m):
    return bool(len(m.faces) >= 4 and m.is_watertight and m.is_winding_consistent)


def orient(m):
    import trimesh
    trimesh.repair.fix_normals(m)
    if closed(m) and m.volume < 0:
        m.invert()
    return m


FALLBACKS = []
_COACD = """
import sys, numpy as np, coacd
d = np.load(sys.argv[1]); coacd.set_log_level("error")
parts = coacd.run_coacd(coacd.Mesh(d["v"].astype(np.float64), d["f"].astype(np.int64)), threshold=float(d["t"]),
                        max_convex_hull=int(d["k"]), seed=0, preprocess_resolution=30,
                        mcts_iterations=60, mcts_max_depth=2)
np.savez(sys.argv[2], n=len(parts), **{f"v{i}": v for i, (v, f) in enumerate(parts)}, **{f"f{i}": f for i, (v, f) in enumerate(parts)})
"""


def coacd_parts(m, threshold=0.05, max_parts=16):
    """Our CoACD parts of a mesh, in a child process: CoACD can abort (an assertion in its triangle intersection test
    killed a pool worker) and took 15-30 min on open NVIDIA meshes at its default search; here a lighter search
    (resolution 30, 60 MCTS iterations, depth 2) on <= 8,000 faces, at most 4 min. On failure or timeout: the convex
    hull of each connected piece (counted in the manifest as coacd_fallbacks)."""
    import subprocess
    import tempfile
    import trimesh
    from lsim.data.assets import decimate
    src = decimate(m, 8000)
    with tempfile.TemporaryDirectory() as td:
        a, b = f"{td}/in.npz", f"{td}/out.npz"
        np.savez(a, v=np.asarray(src.vertices), f=np.asarray(src.faces), t=threshold, k=max_parts)
        try:
            subprocess.run([sys.executable, "-I", "-c", _COACD, a, b], timeout=240, check=True,
                           capture_output=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
            d = np.load(b)
            return [orient(trimesh.Trimesh(d[f"v{i}"], d[f"f{i}"], process=True)) for i in range(int(d["n"]))]
        except Exception:
            FALLBACKS.append(1)
            comps = src.split(only_watertight=False) or [src]
            return [orient(trimesh.Trimesh(c.convex_hull.vertices, c.convex_hull.faces)) for c in comps if len(c.vertices) >= 4]


def decimate_closed(pcs, max_faces=MAX_FACES):
    """Decimate the pieces to ~max_faces in total, each piece kept closed (a piece that would open keeps more faces)."""
    import trimesh
    from lsim.data.assets import decimate
    tot = sum(len(p.faces) for p in pcs)
    out = []
    for p in pcs:
        budget = max(12, int(max_faces * len(p.faces) / max(tot, 1)))
        q = p
        for b in (budget, 2 * budget, 4 * budget):
            if len(p.faces) <= b:
                q = p
                break
            q = orient(decimate(p, b))
            if closed(q):
                break
        if not closed(q):
            q = p if closed(p) else trimesh.Trimesh(p.convex_hull.vertices, p.convex_hull.faces)
        out.append(q)
    return out


def _rot_to(n, target=np.array([0.0, 0.0, -1.0])):
    n = n / np.linalg.norm(n)
    v, c = np.cross(n, target), float(n @ target)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def rest_pose(mesh, com, tip_deg=TIP_DEG):
    """(R, info): the rotation of the authored frame that puts the object on its chosen stable face."""
    from scipy.spatial import ConvexHull
    hull = mesh.convex_hull
    in_facet = np.zeros(len(hull.faces), bool)
    groups = []
    for g in hull.facets:
        groups.append(np.asarray(g)); in_facet[g] = True
    groups += [np.array([i]) for i in np.where(~in_facet)[0]]
    cand = []
    for g in groups:
        n = hull.face_normals[g[0]]
        P = hull.vertices[np.unique(hull.faces[g])]
        h = float(n @ P[0] - n @ com)                     # CoM height above this face (> 0)
        if h <= 0:
            continue
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        q = np.stack([P @ e1, P @ e2], 1)
        c = np.array([com @ e1, com @ e2])
        try:
            ch = ConvexHull(q)
            margin = float(-(ch.equations[:, :2] @ c + ch.equations[:, 2]).max())   # >0 inside
        except Exception:
            margin = -1.0
        area = float(hull.area_faces[g].sum())
        cand.append({"n": n, "h": h, "tip": float(np.degrees(np.arctan2(margin, h))), "area": area})
    down = max(cand, key=lambda c: -c["n"][2])
    if -down["n"][2] > np.cos(np.radians(3.0)) and down["tip"] >= tip_deg:
        best, kept = down, True
    else:
        stable = [c for c in cand if c["tip"] >= tip_deg]
        best = min(stable, key=lambda c: c["h"]) if stable else max(cand, key=lambda c: c["tip"])
        kept = False
    R = _rot_to(best["n"])
    return R, {"authored_pose_kept": kept, "tip_deg": round(best["tip"], 2), "com_height": round(best["h"], 4),
               "stable": best["tip"] >= tip_deg, "authored_bottom_tip_deg": round(down["tip"], 2)}


def _qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def body_physics(b, cols, col_pieces):
    """A body's authored mass, CoM (authored frame), inertia (authored frame, about its CoM), and how they were got."""
    ents = b.get("mass_api", [])
    own = [e for e in ents if e["path"] == b["path"]]
    sub = [e for e in ents if e["path"] != b["path"]]
    pick = lambda key: next((e for e in own + sub if e.get(key) is not None), None)
    mass = next((e["mass"] for e in own if e.get("mass")), None)
    if mass is None and any(e.get("mass") for e in sub):
        mass = float(sum(e["mass"] or 0.0 for e in sub))
    src = "authored mass" if mass else None
    if not mass:                                              # density x closed volume
        vol, rho = 0.0, None
        for cm, p in zip(cols, col_pieces):
            if cm["body"] == b["path"] and closed(p):
                vol += abs(p.volume)
                rho = rho or (cm.get("material") or {}).get("density")
        rho = rho or next((e["density"] for e in ents if e.get("density")), None) or (b.get("material") or {}).get("density")
        if rho and vol > 0:
            mass, src = float(rho * vol), "density x volume"
    e = pick("com_local")
    com = None
    if e is not None:
        W = np.array(e["world"])
        com = W[:3, :3] @ np.array(e["com_local"]) + W[:3, 3]
    ei = pick("inertia_diag")
    I = None
    if ei is not None and any(ei["inertia_diag"]):
        W = np.array(ei["world"])
        Rw = W[:3, :3] / np.cbrt(abs(np.linalg.det(W[:3, :3])) or 1.0)
        Rp = _qmat(ei["principal_axes"]) if ei.get("principal_axes") and any(ei["principal_axes"]) else np.eye(3)
        Rt = Rw @ Rp
        I = Rt @ np.diag(ei["inertia_diag"]) @ Rt.T
    return mass, com, I, src


def materials(cols, bodies):
    mats = [c["material"] for c in cols if c.get("material")] + [b["material"] for b in bodies if b.get("material")]
    uniq = {m["path"]: m for m in mats}
    if not uniq:
        return None, []
    key = lambda k: [m[k] for m in mats if m.get(k) is not None and not (k == "density" and m[k] <= 0)]
    avg = {k: (float(np.mean(key(k))) if key(k) else None) for k in ("static_friction", "dynamic_friction", "restitution", "density")}
    return avg, list(uniq.values())


def scale_evidence(meta, row, res, masses):
    """How large the authored geometry is against the masses the package itself records (no rescaling is applied).
    projectsim's physics.json keeps, for hand-set masses, the geometry-derived mass it replaced ("set by hand to a
    realistic value (was X kg)"): at one density, (geometry mass / realistic mass)^(1/3) is the factor by which the
    geometry is larger than the object it depicts. Otherwise: the implied density mass / closed volume against the
    authored material density (an upper bound on the factor; hollow parts lower the implied density too)."""
    import re
    out = {}
    pj = REPO / "data/external/simready" / meta["root_usd"]
    pj = pj.parent / "physics.json"
    if pj.exists():                                           # parsed as data only
        txt = pj.read_text()
        pairs = [(float(a), float(b)) for a, b in re.findall(
            r'"mass_kg":\s*([0-9.]+),\s*"mass_source":\s*"set by hand to a realistic value \(was ([0-9.]+) kg\)"', txt)]
        if pairs:
            now, was = sum(a for a, _ in pairs), sum(b for _, b in pairs)
            out["hand_set_mass_kg"], out["geometry_mass_kg"] = round(now, 3), round(was, 3)
            out["linear_factor_from_package_masses"] = round((was / now) ** (1 / 3), 3)
    V = float(res.get("volume", 0.0))
    rho = (row["physics"]["friction_restitution"] or {}).get("density")
    m = sum(x for x in masses if x) if any(masses) else None
    if V > 0 and rho and m:
        out["implied_density"] = round(m / V, 1)
        out["authored_density"] = rho
        out["linear_factor_upper_bound_from_density"] = round(max(1.0, (rho * V / m)) ** (1 / 3), 3)
    return out


def work(name):
    import trimesh
    from lsim.data import simready
    from lsim.data.assets import process_mesh
    t0 = time.time()
    meta = json.loads((RAW / f"{name}.json").read_text())
    out = ROOT / "proc" / f"{name}.npz"
    rowf = ROOT / "proc" / f"{name}.row.json"
    if out.exists() and rowf.exists():                       # resume
        return json.loads(rowf.read_text())
    row = {"name": name, "library": meta["library"], "package": meta["package"], "revision": meta["revision"],
           "source_url": meta["source_url"], "root_usd": meta["root_usd"], "package_id": meta.get("package_id"),
           "license": meta.get("package_license") or ("CC-BY-4.0" if "projectsim" in meta["library"] else None),
           "category": simready.category(meta["library"], meta["package"])}
    try:
        d = np.load(RAW / f"{name}.npz")
        cols = meta["colliders"]
        col_p = [orient(p) for p in pieces(d, "col")]
        notes = []
        use, source = None, None
        if meta["library"].startswith("ALLSIDES") and len(d["vis_nv"]):
            vis = trimesh.util.concatenate(pieces(d, "vis"))
            vis.merge_vertices(); orient(vis)
            if closed(vis):
                use, source = [vis], "lod-1 scan (closed)"
        n_open = sum(not closed(p) for p in col_p)
        FALLBACKS.clear()
        _fx = {}

        def fixed_of(k):                                     # collider k as closed pieces (our CoACD if it is open)
            if k not in _fx:
                p = col_p[k]
                _fx[k] = [p] if closed(p) else ([q for q in coacd_parts(p) if closed(q)] if len(p.faces) >= 4 else [])
            return _fx[k]
        if use is None:
            use = []
            for k, p in enumerate(col_p):
                use += fixed_of(k)
            source = "authored colliders" + (f" ({n_open} open -> our CoACD)" if n_open else "")
        if not use:
            raise ValueError("no collision geometry")
        # authored physics, per body and composite
        bodies = meta["bodies"]
        bp = [body_physics(b, cols, col_p) for b in bodies]
        masses = [m for m, _, _, _ in bp]
        comb = trimesh.util.concatenate(use)
        com_auth = None
        if bp and all(m and c is not None for m, c, _, _ in bp):
            M = sum(masses)
            com_auth = sum(m * c for m, c, _, _ in bp) / M
        com_geo = comb.center_mass if comb.is_watertight else comb.centroid
        com_rest = com_auth if com_auth is not None else com_geo
        R, pose = rest_pose(comb, com_rest)
        if np.ptp(comb.vertices, axis=0).max() > simready.TABLE_SIZE[1] and not pose["authored_pose_kept"]:
            # furniture, doors, wall cabinets: stand as authored (floor- or wall-mounted), never laid on a side
            R = np.eye(3)
            pose = dict(pose, authored_pose_kept=True, tip_deg=pose["authored_bottom_tip_deg"],
                        stable=pose["authored_bottom_tip_deg"] >= TIP_DEG, note="large: authored pose kept")
        T = np.eye(4); T[:3, :3] = R
        pcs = decimate_closed([p.copy().apply_transform(T) for p in use])
        hi = trimesh.util.concatenate([p.copy().apply_transform(T) for p in use])
        b0 = hi.bounds[0].copy()
        res = process_mesh(hi, name)                          # moves hi to the object frame: origin at its solid CoM
        off = hi.bounds[0] - b0                               # the translation process_mesh applied
        Tf = np.eye(4); Tf[:3, 3] = off
        rest_T = Tf @ T
        col = trimesh.util.concatenate([p.copy().apply_translation(off) for p in pcs])
        col_closed = all(closed(p) for p in pcs)
        res["collide_v"], res["collide_f"] = col.vertices.astype(np.float32), col.faces.astype(np.int32)
        res["collide_pieces"] = np.array([len(p.faces) for p in pcs], np.int32)
        ext = np.ptp(col.vertices, axis=0)
        # authored physics in the object frame
        tf = lambda x: (rest_T[:3, :3] @ x + rest_T[:3, 3]) if x is not None else None
        Iobj = None
        if bp and all(m and c is not None and I is not None for m, c, I, _ in bp) and com_auth is not None:
            Iobj = np.zeros((3, 3))
            for m, c, I, _ in bp:
                r = R @ (c - com_auth)
                Iobj += R @ I @ R.T + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        mat, mats = materials(cols, bodies)
        lib = meta["library"]
        prov_mass = ("measured (weighed object)" if lib.startswith("ALLSIDES") else "authored, class default")
        if any(s == "density x volume" for *_, s in bp):
            prov_mass += "; some parts density x volume"
        res.update({"rest_T": rest_T.astype(np.float64),
                    "authored_mass": np.float64(sum(m for m in masses if m) if any(masses) else np.nan),
                    "authored_com": np.asarray(tf(com_auth) if com_auth is not None else [np.nan] * 3, np.float64),
                    "authored_inertia": np.asarray(Iobj if Iobj is not None else np.full((3, 3), np.nan), np.float64),
                    "mu_static": np.float64(np.nan if not mat or mat["static_friction"] is None else mat["static_friction"]),
                    "mu_dynamic": np.float64(np.nan if not mat or mat["dynamic_friction"] is None else mat["dynamic_friction"]),
                    "restitution": np.float64(np.nan if not mat or mat["restitution"] is None else mat["restitution"]),
                    "density": np.float64(np.nan if not mat or mat["density"] is None else mat["density"])})
        np.savez_compressed(out, **res)
        lo, hi_ = simready.TABLE_SIZE
        checks = {"closed": bool(col_closed), "size": bool(lo <= ext.max() <= hi_ and ext.min() >= simready.MIN_THICK),
                  "stable": bool(pose["stable"])}
        jt = [j for j in meta["joints"] if j["type"] in ("revolute", "prismatic")]
        row.update({"ok": True, "checks": checks, "extents": [round(float(x), 4) for x in ext],
                    "collision_source": source, "collision_pieces": len(pcs), "collision_faces": int(len(col.faces)),
                    "open_authored_colliders": int(n_open), "coacd_fallbacks": len(FALLBACKS), "authored_colliders": len(col_p), "pose": pose,
                    "bodies": len(bodies), "joints": {k: sum(j["type"] == k for j in meta["joints"])
                                                      for k in sorted({j["type"] for j in meta["joints"]})},
                    "jointed": bool(jt), "articulation_roots": meta["articulation_roots"],
                    "physics": {"mass_kg": None if not any(masses) else round(float(sum(m for m in masses if m)), 5),
                                "mass_provenance": prov_mass if any(masses) else None,
                                "com_object_frame": None if com_auth is None else [round(float(x), 5) for x in tf(com_auth)],
                                "inertia_object_frame": None if Iobj is None else np.round(Iobj, 7).tolist(),
                                "friction_restitution": mat, "friction_provenance": "authored, material class default",
                                "materials": mats},
                    "semantics": meta["prim_metadata"].get("semantics"),
                    "seconds": round(time.time() - t0, 1)})
        row["scale_evidence"] = scale_evidence(meta, row, res, masses)
        if meta.get("rescaled"):
            row["rescaled"] = meta["rescaled"]
            row["scale_evidence"] = {"note": "this copy is rescaled; evidence on " + meta["rescaled"]["from"]}
        if jt:
            parts = {}
            prow = []
            for b, (m, c, I, s), in zip(bodies, bp):
                bi = len(prow)
                fixed = [q for k, cm in enumerate(cols) if cm["body"] == b["path"] for q in fixed_of(k)]
                if fixed:
                    pm = trimesh.util.concatenate(decimate_closed([q.copy().apply_transform(rest_T) for q in fixed], 1000))
                    parts[f"p{bi}_v"], parts[f"p{bi}_f"] = pm.vertices.astype(np.float32), pm.faces.astype(np.int32)
                W = rest_T @ np.array(b["world"])
                prow.append({"index": bi, "path": b["path"], "pose_object_frame": np.round(W, 6).tolist(),
                             "mass_kg": m, "mass_source": s, "com_object_frame": None if c is None else tf(c).round(6).tolist(),
                             "inertia_object_frame": None if I is None else (R @ I @ R.T).round(8).tolist(),
                             "material": b.get("material") or next((cm["material"] for cm in cols if cm["body"] == b["path"] and cm.get("material")), None),
                             "mesh": f"p{bi}" if fixed else None, "kinematic": b["kinematic"]})
            np.savez_compressed(ROOT / "parts" / f"{name}.npz", **parts)
            pidx = {p["path"]: p["index"] for p in prow}
            jrow = []
            for j in meta["joints"]:
                jj = dict(j)
                jj["part0"], jj["part1"] = pidx.get(j["body0"]), pidx.get(j["body1"])
                if jj["part0"] is not None:
                    W0 = np.array(prow[jj["part0"]]["pose_object_frame"])
                    anchor = W0[:3, :3] @ np.array(j["local_pos0"]) + W0[:3, 3]
                    jj["anchor_object_frame"] = anchor.round(6).tolist()
                    if j.get("axis") in ("X", "Y", "Z"):
                        e = np.eye(3)["XYZ".index(j["axis"])]
                        R0 = W0[:3, :3] / np.cbrt(abs(np.linalg.det(W0[:3, :3])) or 1.0)
                        jj["axis_object_frame"] = (R0 @ _qmat(j["local_rot0"] or [1, 0, 0, 0]) @ e).round(6).tolist()
                jrow.append(jj)
            row["joint_manifest"] = {"parts": prow, "joints": jrow}
        rowf.write_text(json.dumps(row))
    except Exception as exc:
        row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "trace": traceback.format_exc(limit=3)[-600:]})
    return row


# Assets whose authored geometry is larger than the object it depicts, by the package's own recorded masses (see
# scale_evidence): a rescaled copy "<name>_rescaled" is converted beside the authored one, which stays as authored.
RESCALE = {"projectsim__meat_slicer": 1 / 1.90, "projectsim__forklift": 1 / 2.03}


def derive_rescaled():
    """raw/<name>_rescaled.{npz,json}: geometry, body and joint positions, CoM and prismatic limits times s; masses kept
    (they are the package's realistic values); inertias times s^2 (the same mass over a body s times the size)."""
    for name, k in RESCALE.items():
        if not (RAW / f"{name}.npz").exists():
            continue
        d = dict(np.load(RAW / f"{name}.npz"))
        for key in ("col_v", "vis_v"):
            d[key] = (d[key] * k).astype(np.float32)
        np.savez_compressed(RAW / f"{name}_rescaled.npz", **d)
        m = json.loads((RAW / f"{name}.json").read_text())
        for b in m["bodies"]:
            b["world"] = [row[:3] + [row[3] * k] for row in b["world"][:3]] + [b["world"][3]]
            for e in b.get("mass_api", []):
                e["world"] = [row[:3] + [row[3] * k] for row in e["world"][:3]] + [e["world"][3]]
                if e.get("com_local"):
                    e["com_local"] = [c * k for c in e["com_local"]]
                if e.get("inertia_diag"):
                    e["inertia_diag"] = [i * k * k for i in e["inertia_diag"]]
        for j in m["joints"]:
            j["local_pos0"] = [x * k for x in j["local_pos0"]]
            j["local_pos1"] = [x * k for x in j["local_pos1"]]
            for lim in ("lower_m", "upper_m"):
                if j.get(lim) is not None:
                    j[lim] *= k
        m["name"] = f"{name}_rescaled"
        m["rescaled"] = {"from": name, "linear_factor": k,
                         "basis": "rescaled by the package's recorded geometry masses (physics.json: hand-set realistic "
                                  "mass vs the geometry-derived mass it replaced); authored version kept as " + name}
        (RAW / f"{name}_rescaled.json").write_text(json.dumps(m, indent=1))


def main():
    derive_rescaled()
    names = sorted(p.stem for p in RAW.glob("*.npz"))
    if "--only" in sys.argv:
        names = sys.argv[sys.argv.index("--only") + 1].split(",")
    if "--limit" in sys.argv:
        names = names[:int(sys.argv[sys.argv.index("--limit") + 1])]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    (ROOT / "proc").mkdir(parents=True, exist_ok=True)
    (ROOT / "parts").mkdir(parents=True, exist_ok=True)
    from lsim.data import simready
    rows = []
    with get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for i, r in enumerate(pool.imap_unordered(work, names)):
            rows.append(r)
            if not r["ok"] or i % 25 == 0:
                print(f"  {i + 1}/{len(names)} {r['name']}: {'ok' if r['ok'] else r['error']}", flush=True)
    rows.sort(key=lambda r: r["name"])
    ok = [r for r in rows if r["ok"]]
    sp = simready.stratified_split([r for r in ok if "rescaled" not in r])    # authored assets only: copies must
    for r in ok:                                                               # neither shift the draw nor cross it
        if "rescaled" in r:
            sp[r["name"]] = sp.get(r["rescaled"]["from"])
    jm = {}
    for r in rows:
        r["split"] = sp.get(r["name"])
        if "joint_manifest" in r:
            jm[r["name"]] = dict(r.pop("joint_manifest"), category=r["category"], library=r["library"],
                                 revision=r["revision"], source_url=r["source_url"])
    (ROOT / "manifest.json").write_text(json.dumps(rows, indent=1))
    (ROOT / "joints.json").write_text(json.dumps(jm, indent=1))
    allp = [r for r in ok if all(r["checks"].values())]
    print(f"done: {len(ok)}/{len(rows)} converted; closed {sum(r['checks']['closed'] for r in ok)}, "
          f"size {sum(r['checks']['size'] for r in ok)}, stable {sum(r['checks']['stable'] for r in ok)}, all {len(allp)}; "
          f"jointed {len(jm)}", flush=True)


if __name__ == "__main__":
    main()
