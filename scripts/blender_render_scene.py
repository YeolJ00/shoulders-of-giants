"""Render a MULTI-OBJECT probe scene: several objects animated in one clip.

blender_render_sim.py animates one subject and parks the rest, which costs one render per
object. This renders every participating object in a single clip so the frames can be
cropped per object afterwards -- render cost per SCENE rather than per object.

Tilt. The simulator produces an incline by rotating GRAVITY and leaving the ground flat.
For the clip to show a tilted table rather than objects sliding on a level one, the whole
world is rotated back by the same angle: the table is rotated about Y through a pivot on
its surface, and the object poses were already rotated to match when they were exported.
+x tilts DOWN, which is the direction the simulator slides them.

Reads  LAB/scene_poses.json
  {key: {"tilt_deg": float, "objects": {name: [{"loc": [...], "mat": [[...]]}, ...]}}}
Writes LAB/sim_<key>/f####.png

Run: LAB=<dir> <blender> --background --python scripts/blender_render_scene.py
"""
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")

import bpy
import mathutils

sys.path.insert(0, str(REPO_ROOT))
from src.render.views import VIEWS  # noqa: E402

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
SCENE = REPO / "outputs" / "scene"
LAB = Path(os.environ["LAB"])
CITY = REPO / "assets" / "scenes" / "city"
CAM = {"eye": [0.72, -0.80, 1.12], "target": [-0.02, 0.02, 0.80], "fov_deg": 46,
       "width": 544, "height": 448}
# The judge sees 544x448; a human watching the clip should not have to. RENDER_SCALE
# multiplies both dimensions for viewing-quality output without moving the camera.
_RS = float(os.environ.get("RENDER_SCALE", "1"))
GROUND_Z = 0.706
PIVOT = mathutils.Vector((-0.05, -0.06, GROUND_Z))   # matches DROP_XY in the sim


def clear():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for b in list(c):
            c.remove(b)


def world_hdri(path):
    w = bpy.data.worlds.new("w"); bpy.context.scene.world = w; w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(path)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def import_gltf(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        return None
    for o in bpy.data.objects:
        o.select_set(False)
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def import_any(asset_dir):
    """Import a staged asset regardless of format.

    Studio assets ship a top-level .gltf; Google-scanned objects ship meshes/model.obj with
    an .mtl beside it. The old code globbed only for *.gltf and silently `continue`d, so six
    of fourteen objects were absent from the render with no error -- the scene just looked
    sparse. Anything unloadable now raises.
    """
    gl = list(asset_dir.glob("*.gltf")) + list(asset_dir.glob("*.glb"))
    if gl:
        return import_gltf(gl[0])
    ob = list(asset_dir.glob("meshes/*.obj")) + list(asset_dir.glob("*.obj"))
    if not ob:
        raise SystemExit(f"no importable mesh in {asset_dir}")
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(ob[0]))
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        raise SystemExit(f"obj import produced no mesh: {ob[0]}")
    for o in bpy.data.objects:
        o.select_set(False)
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    ob_joined = bpy.context.view_layer.objects.active

    # TEXTURE FIX-UP. Google-scanned MTLs say `map_Kd texture.png`, but the file actually
    # lives at materials/textures/texture.png. Blender looks beside the .obj, finds nothing,
    # and renders the object MAGENTA -- which is not just ugly, it is a different material
    # than the one we are asking the judge about (rule 2: fixed target-material appearance).
    tex = asset_dir / "materials" / "textures" / "texture.png"
    if tex.exists():
        img = bpy.data.images.load(str(tex), check_existing=True)
        for slot in ob_joined.material_slots:
            mat = slot.material
            if mat is None:
                continue
            mat.use_nodes = True
            nt = mat.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
                out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"),
                           nt.nodes.new("ShaderNodeOutputMaterial"))
                nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
            texnode = nt.nodes.new("ShaderNodeTexImage"); texnode.image = img
            nt.links.new(texnode.outputs["Color"], bsdf.inputs["Base Color"])
    return ob_joined


def main():
    poses = json.loads((LAB / "scene_poses.json").read_text())
    # A LAB may define its own object set (a bigger scene, different assets). Prefer it;
    # fall back to the shared staging config. Objects named in scene_poses.json but absent
    # here get parked out of frame, which is why a mismatch renders an empty table.
    _lab_cfg = LAB / "scene.json"
    scene_cfg = json.loads((_lab_cfg if _lab_cfg.exists() else SCENE / "scene.json").read_text())
    _missing = set(next(iter(poses.values()))["objects"]) - set(scene_cfg["objects"])
    if _missing:
        raise SystemExit(f"poses name objects absent from scene config: {sorted(_missing)}")
    clear()
    world_hdri(str(CITY / "pretville_street_2k.hdr"))
    # LOOK F. The floor is a SHADOW CATCHER, not a surface. As an opaque plane it was lit
    # by the HDRI to mid-grey and it occluded the environment's own ground, so the street
    # vanished below the horizon and the table floated above a grey slab -- in every render
    # this project has produced. Invisible to camera, still receiving contact shadow, the
    # HDRI ground shows through and the table stays grounded. This is scene context the
    # judge uses to read contact and orientation, not decoration.
    bpy.ops.mesh.primitive_plane_add(size=40)
    bpy.context.active_object.is_shadow_catcher = True

    table = import_gltf(REPO / "assets/scenes/wooden_table_02/wooden_table_02_2k.gltf")
    # A WIDER TABLE for multi-object scenes. 14 staged objects need 0.70 m of depth and
    # the stock table gives 0.706 with no margin, so they overhang the edge and topple off
    # for reasons that have nothing to do with their material. Scaling X/Y only keeps the
    # surface at GROUND_Z, which is what the simulator assumes.
    tsx, tsy = float(os.environ.get("TABLE_SX", "1")), float(os.environ.get("TABLE_SY", "1"))
    if tsx != 1.0 or tsy != 1.0:
        table.scale = (table.scale[0] * tsx, table.scale[1] * tsy, table.scale[2])
        bpy.context.view_layer.update()
    bpy.context.view_layer.update()
    ztop = max((table.matrix_world @ v.co).z for v in table.data.vertices)
    table.location.z += GROUND_Z - ztop
    table_home = table.location.copy()

    # A REAL RAMP, when a scene carries a "ramp" entry: a wooden plank lying on the LEVEL
    # table, low end on the tabletop, high end propped on a block. A table on tilted legs
    # reads as a table falling over -- not a scene a model trained on real video has seen.
    # The plank's TOP FACE IS THE SIMULATOR'S PLANE: the scene gives the sim's own pivot,
    # half-extents and thickness, and the tilt is the scene's tilt_deg. The table then
    # never rotates. (A rotated table also inherited PIVOT above, 5-6 cm off the sim's.)
    _wood = bpy.data.materials.new("ramp_wood"); _wood.use_nodes = True
    nt = _wood.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    # grain runs ALONG the plank (world x), so the bands vary across it (y)
    wv = nt.nodes.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"; wv.bands_direction = "Y"
    wv.inputs["Scale"].default_value = 9.0; wv.inputs["Distortion"].default_value = 1.5
    wv.inputs["Detail"].default_value = 4.0; wv.inputs["Detail Scale"].default_value = 2.0
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (0.16, 0.07, 0.03, 1.0)
    cr.color_ramp.elements[1].color = (0.36, 0.19, 0.09, 1.0)
    nt.links.new(tc.outputs["Object"], wv.inputs["Vector"])
    nt.links.new(wv.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55

    def _box(name, verts):
        faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        me = bpy.data.meshes.new(name); me.from_pydata([tuple(v) for v in verts], [], faces)
        me.update()
        ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
        ob.data.materials.append(_wood)
        return ob

    def build_ramp(tilt_deg, rp):
        th = math.radians(tilt_deg)
        gt = mathutils.Vector((math.cos(th), 0.0, -math.sin(th)))    # downhill
        gn = mathutils.Vector((math.sin(th), 0.0, math.cos(th)))     # plane normal
        gb = mathutils.Vector((0.0, 1.0, 0.0))
        piv = mathutils.Vector(rp["pivot"]); hw, hd = rp["half"]; tk = rp["thick"]
        top = [piv + gt * (sa * hw) + gb * (sb * hd)
               for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        plank = _box("plank", top + [v - gn * tk for v in top])
        # block under the high end, standing on the tabletop, up to the plank's underside
        hi = piv - gt * (hw - 0.05) - gn * tk
        x0, x1 = hi.x - 0.05, hi.x + 0.05
        z1 = hi.z - 0.05 * math.tan(th)     # underside height at the block's downhill face
        bl = [mathutils.Vector((x, y, z)) for (x, y, z) in
              ((x0, -hd, GROUND_Z), (x1, -hd, GROUND_Z), (x1, hd, GROUND_Z), (x0, hd, GROUND_Z),
               (x0, -hd, z1), (x1, -hd, z1), (x1, hd, z1), (x0, hd, z1))]
        block = _box("block", bl)
        return [plank, block]

    objs, home = {}, {}
    for name, o in scene_cfg["objects"].items():
        b = import_any(REPO / "assets" / o["asset"])
        if b is None:
            raise SystemExit(f"failed to import {o['asset']}")
        b.name = name; b.scale = (o["scale"],) * 3
        b.rotation_euler = (b.rotation_euler[0], b.rotation_euler[1], o["rot_z"])
        b.location = tuple(o["pos"]); objs[name] = b; home[name] = tuple(o["pos"])

    cams = {}
    for vn, vc in VIEWS.items():
        cd_ = bpy.data.cameras.new(f"c{vn}")
        co = bpy.data.objects.new(f"c{vn}", cd_)
        bpy.context.collection.objects.link(co)
        # CAM_PULL backs the camera away from its target along the same sight line, for
        # scenes wider than the one these views were framed for. Direction and target are
        # untouched, so the composition is preserved -- only the field of coverage grows.
        _eye = mathutils.Vector(vc["eye"]); _tgt = mathutils.Vector(vc["target"])
        _pull = float(os.environ.get("CAM_PULL", "1"))
        co.location = _tgt + (_eye - _tgt) * _pull
        # CAM_JITTER moves the eye by a few millimetres without changing the target.
        # Used to measure the surprise harness's noise floor: a nuisance change no
        # physics depends on must move the score less than theta does.
        _j = os.environ.get("CAM_JITTER")
        if _j:
            co.location = co.location + mathutils.Vector([float(x) for x in _j.split(",")])
        lk = _tgt - co.location
        co.rotation_euler = lk.to_track_quat("-Z", "Y").to_euler()
        cd_.sensor_fit = "HORIZONTAL"; cd_.angle = math.radians(vc["fov_deg"])
        cams[vn] = co
    only = os.environ.get("VIEWS")
    if only:
        cams = {k: v for k, v in cams.items() if k in only.split(",")}
    bpy.context.scene.camera = cams[list(cams)[0]]

    sc = bpy.context.scene
    sc.render.engine = os.environ.get("ENGINE", "CYCLES")
    if sc.render.engine == "CYCLES":
        sc.cycles.samples = int(os.environ.get("SAMPLES", 48))
        # MOTION BLUR. Off by default in Cycles and never set here, so every clip this
        # project has rendered is perfectly sharp -- an object mid-fall looks like an object
        # suspended at rest. Measured edge sharpness: a real photo of a ball in mid-air is
        # 4.23 against 18.95 for a real static table (4.5x), while our moving and static
        # renders sit at 5.51 and 6.20 (1.1x). Blur is the strongest single-frame cue that
        # something is moving, and we were not producing it. MOTION_BLUR=<shutter> enables
        # it; the shutter is in frames, so 0.5 is a half-frame exposure.
        _mb = float(os.environ.get("MOTION_BLUR", "0"))
        sc.render.use_motion_blur = _mb > 0
        if _mb > 0:
            sc.cycles.motion_blur_position = "CENTER"
            sc.render.motion_blur_shutter = _mb
        # the other half of the noise floor: identical physics, different Monte-Carlo
        # sampling. Cycles is deterministic at a fixed seed, so without this the
        # "render noise" floor would come out as exactly zero and mean nothing.
        sc.cycles.seed = int(os.environ.get("SEED", 0))
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"; prefs.get_devices()
            for d in prefs.devices:
                d.use = (d.type == "CUDA")
            sc.cycles.device = "GPU"
        except Exception as e:
            print("GPU off:", e)
    # RENDER_SCALE_X scales the WIDTH only, for a grid of probe cells laid out along the
    # camera's right vector (src/render/grid.py). The cells sit in a horizontal band, so
    # scaling both axes -- which RENDER_SCALE does -- would spend most of the extra pixels
    # on empty table. With sensor_fit=HORIZONTAL and a fixed `angle`, more resolution_x is
    # more pixels across the SAME horizontal field of view, so each cell grows in pixels
    # without the lens widening and without per-cell perspective changing.
    _RSX = float(os.environ.get("RENDER_SCALE_X", "1"))
    sc.render.resolution_x = int(CAM["width"] * _RS * _RSX)
    sc.render.resolution_y = int(CAM["height"] * _RS)
    sc.render.image_settings.file_format = "PNG"

    # SHARD="k/N" makes this process take only every Nth scene, so several workers can
    # split one LAB without racing on the same output directory. Needed because a frame is
    # GPU-light (nvidia-smi pmon attributes only 3-17% sm to each blender process at
    # 544x448 / 48 samples) and CPU-heavy -- depsgraph sync, BVH refit, OIDN denoise and
    # PNG write dominate, so throughput scales with processes, not with cards.
    _shard = os.environ.get("SHARD")
    _sk, _sn = (int(x) for x in _shard.split("/")) if _shard else (0, 1)
    for _idx, (key, info) in enumerate(poses.items()):
      if _idx % _sn != _sk:
        continue
      acting = info["objects"]
      n_frames = min(len(v) for v in acting.values())
      for vn, camobj in cams.items():
        bpy.context.scene.camera = camobj
        outdir = LAB / f"sim_{key}@{vn}"
        outdir.mkdir(parents=True, exist_ok=True)
        _want = len(os.environ["FRAMES"].split(",")) if os.environ.get("FRAMES") else n_frames
        if len(list(outdir.glob("f*.png"))) == _want:
            print(f"  skip {key}@{vn}: already complete"); continue

        # tilt the table about Y through a pivot on its surface; +x goes down.
        # tilt_deg may be a scalar (fixed incline) or a per-frame list (ramp).
        tilt = info.get("tilt_deg", 0.0)
        tilt_seq = tilt if isinstance(tilt, list) else [float(tilt)] * n_frames
        table.rotation_mode = "XYZ"
        _ramp_obs = []
        _ramp = "ramp" in info
        if _ramp:
            if isinstance(tilt, list):
                raise SystemExit("a ramp needs a fixed tilt; this scene tilts over time")
            _ramp_obs = build_ramp(float(tilt), info["ramp"])

        # anything not participating is moved out of frame rather than left floating
        for n, o in objs.items():
            if n not in acting:
                o.location = (home[n][0], home[n][1] + 3.0, home[n][2])
            else:
                o.rotation_mode = "QUATERNION"

        # CAM_ORBIT=<deg> swings the camera around its target over the clip, keeping the
        # target fixed. Unlike CAM_JITTER (a constant offset used to measure a noise floor)
        # this is a per-frame path, so it produces PARALLAX: with a locked camera an object
        # in mid-air and one resting on the table are visually ambiguous, and with a moving
        # one they separate. It also makes camera_static=False honest for a motion model --
        # the camera really does move, and the induced background flow is computable from
        # this same path.
        _orbit = float(os.environ.get("CAM_ORBIT", "0"))
        # ORBIT_NF: the clip length the orbit is laid out over. A short render (FRAMES=)
        # then follows the SAME camera path as the full clip rather than a compressed one.
        _onf = int(os.environ.get("ORBIT_NF", n_frames))
        # FRAMES="0,24": render only these frames -- for a conditioning frame plus the one
        # the background shift is measured against, when the rest would never be read.
        _only = os.environ.get("FRAMES")
        _only = {int(x) for x in _only.split(",")} if _only else None
        _cam_home = {vn: cams[vn].location.copy() for vn in cams}
        for t in range(n_frames):
            if _only is not None and t not in _only:
                continue
            if _orbit:
                _tgt = mathutils.Vector(VIEWS[vn]["target"])
                _a = math.radians(_orbit) * (t / max(_onf - 1, 1) - 0.5)
                _rel = _cam_home[vn] - _tgt
                _rot = mathutils.Matrix.Rotation(_a, 4, "Z")
                camobj.location = _tgt + (_rot @ _rel)
                camobj.rotation_euler = (_tgt - camobj.location).to_track_quat(
                    "-Z", "Y").to_euler()
            th = math.radians(float(tilt_seq[min(t, len(tilt_seq) - 1)]))
            R = mathutils.Matrix.Rotation(th, 4, "Y")
            if not _ramp:
                table.rotation_euler = (0.0, th, 0.0)
                table.location = PIVOT + (R @ (table_home - PIVOT))
            for n, seq in acting.items():
                o = objs.get(n)
                if o is None:
                    continue
                p = seq[t]
                M = mathutils.Matrix([[p["mat"][r][c] for c in range(3)] for r in range(3)])
                o.rotation_quaternion = M.to_quaternion()
                o.location = mathutils.Vector(p["loc"])
            bpy.context.view_layer.update()
            sc.render.filepath = str(outdir / f"f{t:04d}.png")
            bpy.ops.render.render(write_still=True)

        # restore for the next key
        if _orbit:
            camobj.location = _cam_home[vn]
            camobj.rotation_euler = (mathutils.Vector(VIEWS[vn]["target"])
                                     - camobj.location).to_track_quat("-Z", "Y").to_euler()
        table.rotation_euler = (0.0, 0.0, 0.0); table.location = table_home
        for _o in _ramp_obs:
            bpy.data.objects.remove(_o, do_unlink=True)
        for n, o in objs.items():
            o.rotation_mode = "XYZ"
            o.rotation_euler = (o.rotation_euler[0], o.rotation_euler[1],
                                scene_cfg["objects"][n]["rot_z"])
            o.location = home[n]
        print(f"  rendered {key}@{vn}: {n_frames} frames, {len(acting)} objects")


main()
