"""Render one clip's panels (written by scripts/l2_render.py export): a PNG per frame per source.

    B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
    CUDA_VISIBLE_DEVICES=<gpu> $B --background --python scripts/blender_render.py -- <clip dir> [width height samples [frames]]

The teacher's scene: a table top 1.2 x 1.0 m at z = 0 (a scanned wooden table scaled to it), the floor
at -0.75 m, the scanned objects with their own textures, the gripper and the hinged furniture as the
boxes the model sees. Lit by a studio HDRI (data/render/, copied from the theta track's scene
assets); the camera sees a plain backdrop. Model panels add the teacher's objects as pale ghosts that
cast no shadow: the distance between an object and its ghost is the error. Cloth bodies (kind "cloth":
triangles, UVs, and vertex positions per frame under poses[<source>_cloth<i>]) are drawn as a 2 mm
fabric with a 2 cm check, so stretch and shear show. Rope bodies (kind "rope": nodes per frame under
poses[<source>_rope<i>]) are drawn as a tube of their radius in one fixed rope colour. Soft bodies (kind "soft":
surface triangles over all tet vertices, positions per frame under poses[<source>_soft<i>]) are smooth-shaded in one
fixed look per category (SOFT_LOOK). Resumable: frames already on disk are skipped.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import mathutils
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HDRI = REPO / "data/render/studio_small_03_2k.hdr"
TABLE = REPO / "data/render/wooden_table_02/wooden_table_02_2k.gltf"
TABLE_TOP = (1.2, 1.0, 0.75)                         # the teacher's table: x, y extents; floor 0.75 m below
FLAT = {"gripper": ((0.06, 0.06, 0.07), 0.5, 0.35), "furniture": ((0.82, 0.80, 0.76), 0.0, 0.5),
        "board": ((0.012, 0.012, 0.014), 0.0, 0.9), "cardboard": ((0.62, 0.45, 0.28), 0.0, 0.8),
        "floor": ((0.36, 0.33, 0.30), 0.0, 0.85)}


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def flat_material(name):
    col, metal, rough = FLAT[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*col, 1.0)
    p.inputs["Metallic"].default_value = metal
    p.inputs["Roughness"].default_value = rough
    return m


def ghost_material():
    m = bpy.data.materials.new("ghost")
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.55, 0.85, 1.0, 1.0)
    p.inputs["Emission Color"].default_value = (0.55, 0.85, 1.0, 1.0)
    p.inputs["Emission Strength"].default_value = 0.35
    p.inputs["Alpha"].default_value = 0.3
    p.inputs["Roughness"].default_value = 0.5
    return m


def scan_material(texture):
    m = bpy.data.materials.new(Path(texture).parent.parent.name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(texture))
    nt.links.new(tex.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.6
    return m


def cloth_material(color):
    """Woven-looking fabric: two tones in a 2 cm check over the cloth's own UVs (metres), with sheen."""
    m = bpy.data.materials.new("cloth")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "uv"
    chk = nt.nodes.new("ShaderNodeTexChecker")
    chk.inputs["Scale"].default_value = 50.0                        # UVs in metres: 50 squares per metre
    c = mathutils.Vector(color)
    chk.inputs["Color1"].default_value = (*c, 1.0)
    chk.inputs["Color2"].default_value = (*(0.8 * c), 1.0)
    nt.links.new(uv.outputs["UV"], chk.inputs["Vector"])
    nt.links.new(chk.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.85
    p.inputs["Sheen Weight"].default_value = 0.6
    return m


def cloth_object(b):
    me = bpy.data.meshes.new(b["label"])
    me.from_pydata([tuple(v) for v in b["rest"]], [], [tuple(t) for t in b["tris"]])
    uvl = me.uv_layers.new(name="uv")
    uvs = np.asarray(b["uv"], dtype=np.float32)
    idx = np.array([lp.vertex_index for lp in me.loops])
    uvl.data.foreach_set("uv", uvs[idx].ravel())
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(b["label"], me)
    bpy.context.collection.objects.link(ob)
    sub = ob.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 1
    sol = ob.modifiers.new("thick", "SOLIDIFY")
    sol.thickness, sol.offset = 0.002, 0.0
    ob.data.materials.append(cloth_material(b.get("color", (0.80, 0.42, 0.30))))
    return ob


def rope_material():
    """One fixed rope look for every rope (rule 4): a matte deep-blue cord with a little sheen (it reads against the
    wooden table)."""
    m = bpy.data.materials.new("rope")
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.05, 0.16, 0.45, 1.0)
    p.inputs["Roughness"].default_value = 0.75
    p.inputs["Sheen Weight"].default_value = 0.4
    return m


def rope_object(b):
    """A rope (kind "rope": nodes "rest", "radius" in m) as a tube: a poly curve through its nodes, beveled by the
    radius; node positions per frame under poses[<source>_rope<i>]."""
    cu = bpy.data.curves.new(b["label"], "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = float(b["radius"])
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    pts = np.asarray(b["rest"], np.float32)
    sp.points.add(len(pts) - 1)
    sp.points.foreach_set("co", np.c_[pts, np.ones(len(pts), np.float32)].ravel())
    ob = bpy.data.objects.new(b["label"], cu)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(rope_material())
    return ob


def set_rope(ob, verts):
    v = np.asarray(verts, np.float32)
    ob.data.splines[0].points.foreach_set("co", np.c_[v, np.ones(len(v), np.float32)].ravel())
    ob.data.update_tag()


SOFT_LOOK = {                                       # one fixed look per soft-body category (rule 4): colour, roughness,
    "sponge": ((0.85, 0.72, 0.18), 0.95, 0.0),      # subsurface weight
    "memory_foam": ((0.80, 0.78, 0.70), 0.9, 0.0),
    "closed_foam": ((0.20, 0.45, 0.75), 0.7, 0.0),
    "plush": ((0.45, 0.28, 0.16), 1.0, 0.0),
    "rubber": ((0.78, 0.16, 0.12), 0.35, 0.0),
    "gel": ((0.30, 0.70, 0.45), 0.15, 0.3),
    "bread": ((0.66, 0.40, 0.16), 0.9, 0.0),
    "wide": ((0.55, 0.50, 0.62), 0.6, 0.0),
}


def soft_object(b):
    """A soft body (kind "soft": its surface triangles "tris" over all its tet vertices "rest"; positions per frame
    under poses[<source>_soft<i>]), smooth-shaded in its category's fixed look (SOFT_LOOK)."""
    me = bpy.data.meshes.new(b["label"])
    me.from_pydata([tuple(v) for v in b["rest"]], [], [tuple(t) for t in b["tris"]])
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(b["label"], me)
    bpy.context.collection.objects.link(ob)
    sub = ob.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 1
    col, rough, sss = SOFT_LOOK.get(b.get("look", "wide"), SOFT_LOOK["wide"])
    m = bpy.data.materials.new("soft_" + b.get("look", "wide"))
    m.use_nodes = True
    pb = m.node_tree.nodes["Principled BSDF"]
    pb.inputs["Base Color"].default_value = (*col, 1.0)
    pb.inputs["Roughness"].default_value = rough
    if sss > 0:
        pb.inputs["Subsurface Weight"].default_value = sss
    if b.get("look") == "plush":                    # a little sheen in its own colour (white sheen washed it out)
        pb.inputs["Sheen Weight"].default_value = 0.3
        pb.inputs["Sheen Tint"].default_value = (*col, 1.0)
    ob.data.materials.append(m)
    return ob


def set_cloth(ob, verts):
    ob.data.vertices.foreach_set("co", np.asarray(verts, dtype=np.float32).ravel())
    ob.data.update()


def boxes_object(name, boxes, material):
    """One mesh holding every (centre, half extents) box, in the body's own frame."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    for c, h in boxes:
        r = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=[2 * x for x in h], verts=r["verts"])
        bmesh.ops.translate(bm, vec=c, verts=r["verts"])
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(material)
    return ob


def scan_object(b):
    """The scan in the simulator's own axes (forward Y, up Z keeps the OBJ's coordinates), moved
    into the body frame."""
    bpy.ops.wm.obj_import(filepath=b["obj"], forward_axis="Y", up_axis="Z")
    ob = bpy.context.selected_objects[0]
    ob.data.transform(mathutils.Matrix.Translation([-c for c in b["com"]]))
    ob.data.materials.clear()
    ob.data.materials.append(scan_material(b["texture"]))
    for poly in ob.data.polygons:
        poly.use_smooth = True
    return ob


def usd_object(b):
    """A SimReady obstacle (kind "usd"): its flattened USD with its own materials, under an empty in the body frame
    (rest_T: authored frame -> the converted object frame, lsim.data.simready), so set_pose moves it as one."""
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=b["usd"], import_guide=False, import_proxy=False, import_render=True,
                          import_cameras=False, import_lights=False, scale=1.0)
    new = [o for o in bpy.data.objects if o not in before]
    bpy.context.view_layer.update()
    root = bpy.data.objects.new("usd_" + Path(b["usd"]).stem, None)
    bpy.context.collection.objects.link(root)
    T = mathutils.Matrix(b["rest_T"])
    for o in new:
        if o.parent is None:
            mw = o.matrix_world.copy()
            o.parent = root
            o.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            o.matrix_basis = T @ mw
    return root


def set_pose(ob, q):
    ob.location = (float(q[0]), float(q[1]), float(q[2]))
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (float(q[6]), float(q[3]), float(q[4]), float(q[5]))


def world(backdrop):
    """HDRI light; the camera sees a plain backdrop."""
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(HDRI))
    light = nt.nodes.new("ShaderNodeBackground")
    plain = nt.nodes.new("ShaderNodeBackground")
    plain.inputs["Color"].default_value = (*backdrop, 1.0)
    plain.inputs["Strength"].default_value = 1.8                         # against the -0.9 exposure
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], light.inputs["Color"])
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(light.outputs["Background"], mix.inputs[1])
    nt.links.new(plain.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def table():
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(TABLE))
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ mathutils.Vector(c) for o in new for c in o.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    s = [TABLE_TOP[i] / (hi[i] - lo[i]) for i in range(3)]
    fit = mathutils.Matrix.Translation((0, 0, -s[2] * hi[2])) @ mathutils.Matrix.Diagonal((*s, 1.0))   # top at z = 0
    mw = {o: o.matrix_world.copy() for o in new}
    for o in new:
        o.parent = None
        o.matrix_world = fit @ mw[o]
    me = bpy.data.meshes.new("floor")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=6.0)
    bm.to_mesh(me)
    bm.free()
    fl = bpy.data.objects.new("floor", me)
    bpy.context.collection.objects.link(fl)
    fl.location.z = -TABLE_TOP[2]
    fl.data.materials.append(flat_material("floor"))


def camera(c, w, h):
    cd = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.collection.objects.link(cam)
    cam.location = mathutils.Vector(c["eye"])
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (mathutils.Vector(c["target"]) - cam.location).to_track_quat("-Z", "Y")
    cd.sensor_fit = "HORIZONTAL"
    cd.angle = math.radians(c["fov"])
    cd.clip_start = 0.01
    bpy.context.scene.camera = cam


def settings(w, h, samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.use_persistent_data = True
    sc.view_settings.exposure = -0.9                 # the studio HDRI alone washes the textures out
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
    sc.cycles.denoiser = "OPTIX"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    d = Path(argv[0])
    w, h, samples = (int(argv[1]), int(argv[2]), int(argv[3])) if len(argv) > 3 else (640, 480, 48)
    frames = int(argv[4]) if len(argv) > 4 else None                       # a test render: the first few
    clip = json.loads((d / "clip.json").read_text())
    poses = np.load(d / "poses.npz")
    clear()
    world((0.93, 0.93, 0.93) if clip["kind"] == "phys101" else (0.86, 0.87, 0.89))
    table()
    mats = {k: flat_material(k) for k in ("gripper", "furniture", "board", "cardboard")}
    ghost_mat = ghost_material()
    bodies = []
    for i, b in enumerate(clip["bodies"]):
        if b["kind"] == "cloth":
            ob = cloth_object(b)
        elif b["kind"] == "rope":
            ob = rope_object(b)
        elif b["kind"] == "soft":
            ob = soft_object(b)
        else:
            ob = scan_object(b) if b["kind"] == "mesh" else usd_object(b) if b["kind"] == "usd" else \
                boxes_object(b["label"], b["boxes"], mats[b["material"]])
            if b.get("bevel"):                       # rounded edges (the soft teacher's pads: a box core + margin)
                bv = ob.modifiers.new("bevel", "BEVEL")
                bv.width, bv.segments = float(b["bevel"]), 3
        bodies.append(ob)
    ghosts = {}
    for i in clip["ghost"]:
        g = bpy.data.objects.new(f"ghost_{i}", bodies[i].data)
        bpy.context.collection.objects.link(g)
        for slot in g.material_slots:
            slot.link = "OBJECT"
            slot.material = ghost_mat
        g.visible_shadow = False
        ghosts[i] = g
    for k, s in enumerate(clip["static"]):
        ob = boxes_object(f"static_{k}", s["boxes"], mats[s["material"]])
        if "pos" in s:
            set_pose(ob, [*s["pos"], *s["quat"]])
    camera(clip["camera"], w, h)
    settings(w, h, samples)
    for src in clip["sources"]:
        out = d / src
        out.mkdir(exist_ok=True)
        q = poses[src]
        for g in ghosts.values():
            g.hide_render = src == "teacher" or "teacher" not in poses
        for f in range(frames or clip["frames"]):
            path = out / f"f{f:04d}.png"
            if path.exists():
                continue
            for i, ob in enumerate(bodies):
                if clip["bodies"][i]["kind"] == "cloth":
                    set_cloth(ob, poses[f"{src}_cloth{i}"][f])
                elif clip["bodies"][i]["kind"] == "rope":
                    set_rope(ob, poses[f"{src}_rope{i}"][f])
                elif clip["bodies"][i]["kind"] == "soft":
                    set_cloth(ob, poses[f"{src}_soft{i}"][f])
                else:
                    set_pose(ob, q[f, i])
            if "teacher" in poses:
                for i, g in ghosts.items():
                    set_pose(g, poses["teacher"][f, i])
            bpy.context.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
        print(f"[blender] {d.name} {src}: {clip['frames']} frames", flush=True)


main()
