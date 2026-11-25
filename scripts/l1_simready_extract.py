"""L1 -- read downloaded SimReady USD packages into plain arrays (no USD past this step).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    CUDA_VISIBLE_DEVICES="" $L -I scripts/l1_simready_extract.py [--src data/external/simready] [--limit N]

The packages are untrusted third-party data (data/external/simready/<owner>__<repo>/, fetched at pinned
revisions, see its download_manifest.json). This script only READS them with pxr (`python -I`, run from
outside data/external); nothing in lsim imports pxr or any downloaded file. Per asset it writes
data/assets/simready/raw/<name>.npz (collider triangles per body, in metres, Z-up, in the asset's authored
frame) and <name>.json (bodies with their authored mass / CoM / inertia / physics material, joints, the
variant selections used, the root file). The geometry work (closing, CoACD, resting pose) is in
scripts/l1_simready_convert.py.

Variant selections (each library's plain-UsdPhysics description, no solver-specific layer):
  nvidia      /RootNode PhysX = Disabled (MuJoCo / Newton stay Disabled): colliders are the visual-resolution meshes
  ALLSIDES    /Asset lod = 1, PhysX = Disabled: the authored CoACD hulls; the lod-1 scan is kept as `visual`
  ForgeCAD    /Model physics = physics, animation = static: the authored convex hulls
  projectsim  none: the authored convex-decomposition colliders
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/external/simready"
DST = REPO / "data/assets/simready/raw"

LIBS = {  # repo dir -> (short prefix, how to find each asset's root file, variant selections)
    "nvidia__simready-assets": ("nv", lambda d: _declared(d) or sorted((d / "simready_usd").glob("sm_*.usd")),
                                [("/RootNode", "PhysX", "Disabled"), ("/RootNode", "MuJoCo", "Disabled"),
                                 ("/RootNode", "Newton", "Disabled")]),
    "ALLSIDES__simready-scanned-objects": ("allsides", lambda d: [d / "simready_usd" / f"{d.name}.usd"],
                                           [("/Asset", "lod", "1"), ("/Asset", "PhysX", "Disabled")]),
    "ForgeCAD__simready-assets": ("forgecad", lambda d: sorted((d / "simready_usd").glob(f"{d.name}.usda")),
                                  [("/Model", "physics", "physics"), ("/Model", "animation", "static")]),
    "projectsim__simready-articulated-objects-sample": ("projectsim", lambda d: sorted(d.glob(f"{d.name}.usd")),
                                                        []),
}


def _declared(d):
    """The package's declared root file(s) (.metadata/com.nvidia.simready.root_usds.json), when present."""
    f = d / ".metadata" / "com.nvidia.simready.root_usds.json"
    if not f.exists():
        return []
    out = []
    for e in json.loads(f.read_text()).get("entries", []):
        p = (d / e).resolve()
        if p.is_file() and str(p).startswith(str(d.resolve()) + "/"):
            out.append(p)
    return out


def asset_dirs(lib_dir):
    if lib_dir.name.startswith("projectsim"):
        return sorted(p for p in (lib_dir / "assets").iterdir() if p.is_dir())
    return sorted(p for p in lib_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _f(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return [float(x) for x in v]


def _quat(q):
    return None if q is None else [float(q.GetReal())] + [float(x) for x in q.GetImaginary()]   # w, x, y, z


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    try:
        return [_jsonable(x) for x in v]
    except TypeError:
        return str(v)


def implicit_mesh(p):
    """A UsdGeom implicit collider (Cube / Sphere / Cylinder / Capsule) as a closed triangle mesh in its prim frame
    (stage units)."""
    import trimesh
    from pxr import UsdGeom
    t = str(p.GetTypeName())
    if t == "Cube":
        m = trimesh.creation.box(extents=[UsdGeom.Cube(p).GetSizeAttr().Get()] * 3)
        return np.asarray(m.vertices), np.asarray(m.faces)
    if t == "Sphere":
        m = trimesh.creation.icosphere(subdivisions=2, radius=UsdGeom.Sphere(p).GetRadiusAttr().Get())
        return np.asarray(m.vertices), np.asarray(m.faces)
    g = UsdGeom.Cylinder(p) if t == "Cylinder" else UsdGeom.Capsule(p)
    r, h, ax = g.GetRadiusAttr().Get(), g.GetHeightAttr().Get(), str(g.GetAxisAttr().Get())
    m = trimesh.creation.cylinder(radius=r, height=h, sections=24) if t == "Cylinder" else \
        trimesh.creation.capsule(height=h, radius=r, count=[16, 16])
    v = np.asarray(m.vertices)
    if t == "Capsule":
        v = v - [0, 0, h / 2]            # trimesh's capsule starts at z = 0; USD's is centred
    R = {"Z": np.eye(3), "X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]]), "Y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])}[ax]
    return v @ R.T, np.asarray(m.faces)


def extract(root, sel):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.Open(str(root), Usd.Stage.LoadAll)
    stage.SetEditTarget(stage.GetSessionLayer())
    used = []
    for path, vs, v in sel:
        p = stage.GetPrimAtPath(path)
        if p and p.GetVariantSets().HasVariantSet(vs) and v in p.GetVariantSet(vs).GetVariantNames():
            p.GetVariantSet(vs).SetVariantSelection(v)
            used.append([path, vs, v])
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    kgpu = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    up = str(UsdGeom.GetStageUpAxis(stage))
    to_z = np.eye(4) if up == "Z" else np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))

    def world(p):            # 4x4, column-vector convention, metres, Z-up
        m = np.array(xc.GetLocalToWorldTransform(p), np.float64).T
        m[:3, 3] *= mpu
        return to_z @ m

    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    bpaths = [str(b.GetPath()) for b in bodies]

    def owner(p):
        s = str(p.GetPath())
        best = None
        for b in bpaths:
            if (s == b or s.startswith(b + "/")) and (best is None or len(b) > len(best)):
                best = b
        return best

    def pmat(p):
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose="physics")
        if not mat or not mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            return None
        m = UsdPhysics.MaterialAPI(mat.GetPrim())
        return {"path": str(mat.GetPath()), "static_friction": _f(m.GetStaticFrictionAttr().Get()),
                "dynamic_friction": _f(m.GetDynamicFrictionAttr().Get()),
                "restitution": _f(m.GetRestitutionAttr().Get()), "density": _f(m.GetDensityAttr().Get())}

    col_v, col_f, col_meta = [], [], []
    vis_v, vis_f = [], []
    skipped = []
    for p in prims:
        is_col = p.HasAPI(UsdPhysics.CollisionAPI)
        prim_kind = None
        if p.IsA(UsdGeom.Mesh):
            g = UsdGeom.Mesh(p)
            pts = g.GetPointsAttr().Get()
            cnt, idx = g.GetFaceVertexCountsAttr().Get(), g.GetFaceVertexIndicesAttr().Get()
            if pts is None or cnt is None or len(pts) == 0:
                continue
            cnt, idx = np.asarray(cnt), np.asarray(idx)
            starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            tris = [np.stack([idx[s + np.zeros(n - 2, int)], idx[s + 1 + np.arange(n - 2)], idx[s + 2 + np.arange(n - 2)]], 1)
                    for s, n in zip(starts, cnt) if n >= 3]
            if not tris:
                continue
            f = np.concatenate(tris).astype(np.int64)
            if str(g.GetOrientationAttr().Get()) == "leftHanded":
                f = f[:, ::-1]
            pts = np.asarray(pts, np.float64)
        elif is_col and p.GetTypeName() in ("Cube", "Sphere", "Cylinder", "Capsule"):
            prim_kind = str(p.GetTypeName())
            pts, f = implicit_mesh(p)
        elif is_col:
            skipped.append([str(p.GetPath()), str(p.GetTypeName())])
            continue
        else:
            continue
        M = world(p)
        # points are in stage units; M's linear part carries the authored rotation and scale
        v = pts @ (M[:3, :3] * mpu).T + M[:3, 3]
        if np.linalg.det(M[:3, :3]) < 0:
            f = f[:, ::-1]
        if is_col:
            ca = UsdPhysics.CollisionAPI(p)
            if ca.GetCollisionEnabledAttr().Get() is False:
                skipped.append([str(p.GetPath()), "collisionEnabled=0"])
                continue
            approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None
            col_meta.append({"path": str(p.GetPath()), "body": owner(p), "primitive": prim_kind,
                             "approximation": str(approx) if approx else ("exact" if prim_kind else "none"),
                             "material": pmat(p), "n_v": len(v), "n_f": len(f)})
            col_v.append(v.astype(np.float32)); col_f.append(f.astype(np.int32))
        elif UsdGeom.Imageable(p).ComputePurpose() in ("default", "render"):
            vis_v.append(v.astype(np.float32)); vis_f.append(f.astype(np.int32))

    rb = []
    for b in bodies:
        r = {"path": str(b.GetPath()), "world": world(b).tolist(), "material": pmat(b)}
        ra = UsdPhysics.RigidBodyAPI(b)
        r["kinematic"] = bool(ra.GetKinematicEnabledAttr().Get())
        r["enabled"] = ra.GetRigidBodyEnabledAttr().Get() is not False
        rb.append(r)
    def massprops(pr):
        m = UsdPhysics.MassAPI(pr)
        A = lambda a: _f(a.Get()) if a.HasAuthoredValue() else None
        mass, dens = A(m.GetMassAttr()), A(m.GetDensityAttr())
        com, I = A(m.GetCenterOfMassAttr()), A(m.GetDiagonalInertiaAttr())
        pa = m.GetPrincipalAxesAttr()
        return {"mass": None if mass is None else mass * kgpu, "density": dens,
                "com_local": None if com is None else [c * mpu for c in com],      # that prim's frame, metres
                "inertia_diag": None if I is None else [i * kgpu * mpu ** 2 for i in I],
                "principal_axes": _quat(pa.Get()) if pa.HasAuthoredValue() else None}
    # MassAPI sits on the body or on its collider prims (nvidia: on the mesh under the body)
    by_body = {}
    for pr in prims:
        if pr.HasAPI(UsdPhysics.MassAPI):
            by_body.setdefault(owner(pr), []).append(dict(path=str(pr.GetPath()), world=world(pr).tolist(), **massprops(pr)))
    for r in rb:
        r["mass_api"] = by_body.get(r["path"], [])

    joints = []
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        t = str(p.GetTypeName())
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        tgt = lambda ts: None if not ts else (owner(stage.GetPrimAtPath(ts[0])) or str(ts[0]))
        d = {"path": str(p.GetPath()), "type": t.replace("Physics", "").replace("Joint", "").lower() or "generic",
             "body0": tgt(b0), "body1": tgt(b1), "body0_target": [str(x) for x in b0], "body1_target": [str(x) for x in b1],
             "local_pos0": [x * mpu for x in (_f(j.GetLocalPos0Attr().Get()) or [0, 0, 0])],
             "local_pos1": [x * mpu for x in (_f(j.GetLocalPos1Attr().Get()) or [0, 0, 0])],
             "local_rot0": _quat(j.GetLocalRot0Attr().Get()), "local_rot1": _quat(j.GetLocalRot1Attr().Get()),
             "enabled": j.GetJointEnabledAttr().Get() is not False,
             "exclude_from_articulation": bool(j.GetExcludeFromArticulationAttr().Get())}
        if t == "PhysicsRevoluteJoint":
            rj = UsdPhysics.RevoluteJoint(p)
            d.update({"axis": str(rj.GetAxisAttr().Get()), "lower_deg": _f(rj.GetLowerLimitAttr().Get()),
                      "upper_deg": _f(rj.GetUpperLimitAttr().Get())})
        elif t == "PhysicsPrismaticJoint":
            pj = UsdPhysics.PrismaticJoint(p)
            lo, hi = _f(pj.GetLowerLimitAttr().Get()), _f(pj.GetUpperLimitAttr().Get())
            d.update({"axis": str(pj.GetAxisAttr().Get()), "lower_m": None if lo is None else lo * mpu,
                      "upper_m": None if hi is None else hi * mpu})
        drives = {}
        for kind in ("linear", "angular", "rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
            if p.HasAPI(UsdPhysics.DriveAPI, kind):
                dr = UsdPhysics.DriveAPI(p, kind)
                drives[kind] = {"type": str(dr.GetTypeAttr().Get()), "stiffness": _f(dr.GetStiffnessAttr().Get()),
                                "damping": _f(dr.GetDampingAttr().Get()), "max_force": _f(dr.GetMaxForceAttr().Get()),
                                "target_position": _f(dr.GetTargetPositionAttr().Get()),
                                "target_velocity": _f(dr.GetTargetVelocityAttr().Get())}
        d["drives"] = drives
        jf = p.GetAttribute("physxJoint:jointFriction")
        if jf and jf.HasAuthoredValue():
            d["joint_friction"] = _f(jf.Get())
        joints.append(d)
    arts = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    dp = stage.GetDefaultPrim()
    info = {}
    for pr in [dp] if dp else []:
        cd = pr.GetCustomData() or {}
        for k, v in cd.items():
            try:
                json.dumps(v)
                info[k] = v
            except TypeError:
                info[k] = str(v)
        ai = pr.GetAssetInfo() or {}
        info.update({f"assetInfo:{k}": str(v) for k, v in ai.items()})
    info["layer"] = _jsonable(dict(stage.GetRootLayer().customLayerData))
    sem = {}
    for pr in prims:
        for a in pr.GetAttributes():
            if a.GetName().startswith("semantics:") and a.HasAuthoredValue():
                sem.setdefault(a.GetName(), set()).update(str(x) for x in (a.Get() or []))
    info["semantics"] = {k: sorted(v) for k, v in sem.items()}
    return {"meters_per_unit": mpu, "kilograms_per_unit": kgpu, "up_axis": up, "variants": used,
            "default_prim": str(dp.GetPath()) if dp else None, "bodies": rb, "joints": joints,
            "articulation_roots": arts, "colliders": col_meta, "skipped_colliders": skipped,
            "prim_metadata": info}, col_v, col_f, vis_v, vis_f


def pack(name, out, col_v, col_f, vis_v, vis_f):
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0, 3), np.float32)
    np.savez_compressed(DST / f"{name}.npz", col_v=cat(col_v), col_f=cat(col_f) if col_f else np.zeros((0, 3), np.int32),
                        col_nv=np.array([len(v) for v in col_v], np.int64), col_nf=np.array([len(f) for f in col_f], np.int64),
                        vis_v=cat(vis_v), vis_f=cat(vis_f) if vis_f else np.zeros((0, 3), np.int32),
                        vis_nv=np.array([len(v) for v in vis_v], np.int64), vis_nf=np.array([len(f) for f in vis_f], np.int64))
    (DST / f"{name}.json").write_text(json.dumps(out, indent=1))


def main():
    src = Path(sys.argv[sys.argv.index("--src") + 1]) if "--src" in sys.argv else SRC
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    DST.mkdir(parents=True, exist_ok=True)
    revs = json.loads((src / "download_manifest.json").read_text())["revisions"] if (src / "download_manifest.json").exists() else {}
    rows = []
    for lib, (prefix, roots, sel) in LIBS.items():
        if not (src / lib).exists():
            continue
        repo = lib.replace("__", "/", 1)
        for d in asset_dirs(src / lib)[:limit]:
            name = f"{prefix}__{d.name}"
            if only and name not in only:
                continue
            r = {"name": name, "library": repo, "package": d.name, "revision": revs.get(repo)}
            try:
                rs = roots(d)
                if not rs:
                    raise FileNotFoundError("no root USD")
                out, cv, cf, vv, vf = extract(rs[0], sel)
                out.update({k: r[k] for k in ("name", "library", "package", "revision")})
                out["root_usd"] = str(rs[0].relative_to(src))
                rel = str(rs[0].relative_to(src / lib))
                out["source_url"] = f"https://huggingface.co/datasets/{repo}/resolve/{r['revision'] or 'main'}/{rel}"
                pk = d / "com.nvidia.simready.packaging.json"
                if pk.exists():
                    pj = json.loads(pk.read_text())
                    out["package_id"], out["package_license"] = pj.get("package_id"), pj.get("license")
                pack(name, out, cv, cf, vv, vf)
                r.update({"ok": True, "bodies": len(out["bodies"]), "joints": len(out["joints"]), "colliders": len(cv),
                          "visual_meshes": len(vv), "skipped_colliders": len(out["skipped_colliders"])})
            except Exception as exc:
                r.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            rows.append(r)
            if not r["ok"] or len(rows) % 25 == 0:
                print(len(rows), name, "ok" if r["ok"] else r["error"], flush=True)
    if only and (DST / "index.json").exists():                # merge into the existing index
        old = {r["name"]: r for r in json.loads((DST / "index.json").read_text())}
        old.update({r["name"]: r for r in rows})
        rows = sorted(old.values(), key=lambda r: r["name"])
    (DST / "index.json").write_text(json.dumps(rows, indent=0))
    print(f"done: {sum(r['ok'] for r in rows)}/{len(rows)} extracted", flush=True)


if __name__ == "__main__":
    main()
