"""Asset pipeline: a scanned mesh -> learner mesh, teacher collision mesh, geometry labels.

Per asset, written to data/assets/proc/<source>/<name>.npz:
  learner   isotropic remesh at 1.5 cm edges (coarser for large objects so it stays <= 512
            vertices): rest vertices in the object frame (origin at the uniform-solid centre of
            mass), faces, vertex normals, vertex area weights (each vertex's share of surface
            area). Artist and scan vertex density follow modelling detail, not mass (theta track).
  collide   the scan decimated to <= 2,000 faces, same frame: what the teacher collides with.
  labels    watertight, volume, extents; centre of mass and inertia (unit mass) for a uniform
            solid and for a thin shell, about the frame origin.
Density regions (PHASE1_AMENDMENT sections 2 and 4) are sampled per episode, not per asset.
"""
import numpy as np
import trimesh

TARGET_EDGE = 0.015
MAX_VERTS = 512
MAX_COLLIDE_FACES = 2000


def load_mesh(path):
    """Geometry only: texture seams split scan vertices, which leaves the mesh open (not
    watertight) and stops the remesher collapsing across the seams. Drop visuals, merge by
    position."""
    t = trimesh.load(path, force="mesh", process=False)
    m = trimesh.Trimesh(np.asarray(t.vertices, np.float64), np.asarray(t.faces), process=True)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    return m


def _ms(mesh):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, np.float64),
                               face_matrix=np.asarray(mesh.faces, np.int32)))
    return ms


def _tm(ms):
    m = ms.current_mesh()
    return trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)


def remesh(mesh, edge=TARGET_EDGE, max_verts=MAX_VERTS):
    """Isotropic remesh (Botsch-Kobbelt, via MeshLab) at `edge`; for a large object the edge is
    first set from its area so the result stays <= max_verts (an equilateral mesh of edge h has
    ~2 / (sqrt(3) h^2) vertices per unit area). The scan is decimated first, for speed."""
    import pymeshlab
    edge = max(edge, float(np.sqrt(2.0 * mesh.area / (np.sqrt(3.0) * 0.9 * max_verts))))
    base = decimate(mesh, 20000)
    for _ in range(6):
        ms = _ms(base)
        # featuredeg 180: preserve no creases (scan noise makes "creases" everywhere, and the
        # remesher then keeps ~8 mm edges whatever the target)
        ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(edge), iterations=10,
                                                checksurfdist=False, featuredeg=180.0)
        out = _tm(ms)
        if len(out.vertices) <= max_verts:
            return out, edge
        edge *= float(np.sqrt(len(out.vertices) / max_verts)) * 1.05
    return out, edge


def surface_points(mesh, spacing=TARGET_EDGE, n_min=48, n_max=600, k=8, seed=0):
    """Learner vertices for a RIGID object: points sampled evenly on the scan's own surface at
    ~spacing, with face normals and k-nearest-neighbour edges. Isotropic remeshing at 1.5 cm
    collapsed thin parts (a figure kept 4 vertices; 124 of 780 tabletop assets lost > 5 mm at
    their base), and a rigid body needs no true connectivity (the physics layer keeps it rigid)."""
    from scipy.spatial import cKDTree
    n = int(np.clip(mesh.area / (0.866 * spacing ** 2), n_min, n_max))
    pts, fi = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    if len(pts) < n_min:                                     # even sampling can under-deliver
        pts, fi = trimesh.sample.sample_surface(mesh, n, seed=seed)
    nrm = mesh.face_normals[fi]
    _, nb = cKDTree(pts).query(pts, k=min(k + 1, len(pts)))
    e = np.array([(i, j) for i, row in enumerate(nb) for j in row[1:] if i < j] +
                 [(j, i) for i, row in enumerate(nb) for j in row[1:] if j < i])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return pts.astype(np.float32), nrm.astype(np.float32), e.astype(np.int32), np.full(len(pts), mesh.area / len(pts), np.float32)


def decimate(mesh, max_faces=MAX_COLLIDE_FACES):
    if len(mesh.faces) <= max_faces:
        return mesh.copy()
    ms = _ms(mesh)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces, preservenormal=True,
                                                preservetopology=True)
    return _tm(ms)


def convex_parts(mesh, threshold=0.05, max_parts=12, seed=0):
    """Approximate convex decomposition (CoACD, Wei et al. 2022) for solvers that collide convex
    pieces (MuJoCo-Warp's own detection convex-hulls every mesh: a bowl becomes a lid). Returns
    a list of (vertices, faces); a single hull when the mesh is already nearly convex."""
    hull = mesh.convex_hull
    if mesh.is_watertight and mesh.volume > 0.93 * hull.volume:
        return [(np.asarray(hull.vertices, np.float32), np.asarray(hull.faces, np.int32))]
    import coacd
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)),
                            threshold=threshold, max_convex_hull=max_parts, seed=seed)
    return [(np.asarray(v, np.float32), np.asarray(f, np.int32)) for v, f in parts]


def shell_props(mesh, n=40000, seed=0):
    """Centre of mass and inertia (about that CoM) of a thin uniform shell of unit mass."""
    p, _ = trimesh.sample.sample_surface_even(mesh, n, seed=seed) if hasattr(trimesh.sample, "sample_surface_even") \
        else trimesh.sample.sample_surface(mesh, n, seed=seed)
    c = p.mean(0)
    r = p - c
    I = (np.eye(3) * (r * r).sum(1).mean() - (r[:, :, None] * r[:, None, :]).mean(0))
    return c, I


def process(path, name):
    return process_mesh(load_mesh(path), name)


def process_mesh(raw, name):
    """process() on an already loaded, cleaned mesh (SimReady colliders arrive as arrays, lsim.data.simready).
    The mesh is moved in place to the object frame (origin at its uniform-solid centre of mass)."""
    if raw.is_watertight:
        com_solid = raw.center_mass
        I_solid = raw.moment_inertia / raw.mass               # unit mass, about its own CoM
    else:
        com_solid, I_solid = shell_props(raw)                  # no interior: fall back to the shell
    raw.apply_translation(-com_solid)                          # object frame: origin at the solid CoM
    com_shell, I_shell = shell_props(raw)
    learn, edge = remesh(raw)
    pts, nrm, edges, parea = surface_points(raw)
    col = decimate(raw)
    parts = convex_parts(decimate(raw, 8000))
    area = np.zeros(len(learn.vertices))
    np.add.at(area, learn.faces.flatten(), np.repeat(learn.area_faces / 3.0, 3))
    return {
        "name": name,
        "learner_v": learn.vertices.astype(np.float32), "learner_f": learn.faces.astype(np.int32),
        "learner_n": learn.vertex_normals.astype(np.float32), "learner_area": area.astype(np.float32),
        "learner_edge": np.float32(edge),
        "points_v": pts, "points_n": nrm, "points_edges": edges, "points_area": parea,
        "collide_v": col.vertices.astype(np.float32), "collide_f": col.faces.astype(np.int32),
        "convex_n": np.int32(len(parts)),
        "convex_v": np.concatenate([v for v, _ in parts]).astype(np.float32),
        "convex_f": np.concatenate([f + sum(len(pv) for pv, _ in parts[:i]) for i, (_, f) in enumerate(parts)]).astype(np.int32),
        "convex_vcount": np.array([len(v) for v, _ in parts], np.int32),
        "convex_fcount": np.array([len(f) for _, f in parts], np.int32),
        "watertight": np.bool_(raw.is_watertight), "volume": np.float32(raw.volume if raw.is_watertight else 0.0),
        "area": np.float32(raw.area), "extents": raw.extents.astype(np.float32),
        "inertia_solid_unit": np.asarray(I_solid, np.float32),
        "com_shell": np.asarray(com_shell, np.float32), "inertia_shell_unit": np.asarray(I_shell, np.float32),
    }
