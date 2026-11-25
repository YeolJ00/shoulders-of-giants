"""Gallery contact sheet of converted SimReady assets, textured, each in its chosen resting pose.

    B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
    CUDA_VISIBLE_DEVICES="" /home/jooyeolyun/anaconda3/envs/lsim/bin/python -I scripts/l1_render_simready.py --prep [--n 48]
    CUDA_VISIBLE_DEVICES=6 $B --background --python scripts/l1_render_simready.py -- [--n 48] [--px 512] [--samples 64] [--test name,name]
    /home/jooyeolyun/anaconda3/envs/lsim/bin/python -I scripts/l1_render_simready.py --sheet

The prep pass (pxr) picks n assets (default 48) from data/assets/simready/manifest.json, spread over libraries and
categories: table objects that pass all checks, plus up to 6 jointed furniture/appliances. For each it writes a wrapper
USD (the package's root file by reference, the variant selections the converter used, the converter's rest
transform is applied in Blender) and a flattened, de-instanced copy of it; the Blender pass renders each with its own textures under the studio HDRI: table objects on the teacher's wooden table,
larger ones on the floor. The second pass tiles the PNGs into outputs/lsim/render_simready/gallery.png with labels.
The packages are third-party (data/external/simready): Blender's USD importer reads them; nothing is executed.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAN = REPO / "data/assets/simready/manifest.json"
RAW = REPO / "data/assets/simready/raw"
SRC = REPO / "data/external/simready"
OUT = REPO / "outputs/lsim/render_simready"
HDRI = REPO / "data/render/studio_small_03_2k.hdr"
TABLE = REPO / "data/render/wooden_table_02/wooden_table_02_2k.gltf"
TABLE_TOP = (1.2, 1.0, 0.75)


def pick(rows, n=48, n_big=6, seed=0):
    import random
    rnd = random.Random(seed)
    ok = [r for r in rows if r.get("ok")]
    table = [r for r in ok if all(r["checks"].values())]
    big = [r for r in ok if r["jointed"] and r["checks"]["closed"] and r["checks"]["stable"] and not r["checks"]["size"]]
    rnd.shuffle(big)
    big = big[:n_big]
    groups = {}
    for r in table:
        groups.setdefault((r["library"], r["category"]), []).append(r)
    for g in groups.values():
        rnd.shuffle(g)
    out, keys = [], sorted(groups)
    while len(out) < n - len(big) and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n - len(big):
                out.append(groups[k].pop())
    return sorted(out, key=lambda r: (r["category"], r["name"])) + big


def wrapper(r, path):
    meta = json.loads((RAW / f"{r['name']}.json").read_text())
    var = "".join(f'        string {vs} = "{v}"\n' for p, vs, v in meta["variants"] if p == meta["default_prim"])
    root = (SRC / meta["root_usd"]).resolve()
    path.write_text(f'''#usda 1.0
(
    defaultPrim = "W"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "W"
{{
    def Xform "A" (
        prepend references = @{root}@
        variants = {{
{var}        }}
    )
    {{
    }}
}}
''')
    return meta


def flat_usd(r, out_dir):
    """lsim python (pxr): the asset's wrapper and a flattened, de-instanced copy of it that Blender imports right; the
    path of the copy. Blender's USD importer misplaces the parts of instanced prototypes (an NVIDIA stool's base came
    out 0.4 m above its wheels) and of prims authored with !resetXformStack! (a ForgeCAD bottle cap 10 cm above the
    bottle, a stapler's arm beside it): de-instance, and give each reset prim the plain local transform that puts it
    where pxr does. Cached by name."""
    from pxr import Gf, Usd, UsdGeom
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = out_dir / f"{r['name']}.flat.usda"
    if flat.exists():
        return flat
    wrap = out_dir / f"{r['name']}.usda"
    wrapper(r, wrap)
    st = Usd.Stage.Open(str(wrap))
    st.SetEditTarget(st.GetSessionLayer())
    for _ in range(8):
        inst = [p for p in st.Traverse() if p.IsInstanceable()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    resets = [p for p in st.Traverse() if p.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(p).GetResetXformStack()]
    world = {p.GetPath(): xc.GetLocalToWorldTransform(p) for p in resets}
    for p in resets:
        local = world[p.GetPath()] * xc.GetParentToWorldTransform(p).GetInverse()     # USD row vectors: child * parent
        x = UsdGeom.Xformable(p)
        x.ClearXformOpOrder()
        x.SetResetXformStack(False)
        x.AddTransformOp(opSuffix="baked").Set(Gf.Matrix4d(local))
    st.Flatten().Export(str(flat))
    return flat


def prep(n=48):
    """lsim python: pick the assets and write each one's flattened copy (flat_usd)."""
    sel = pick(json.loads(MAN.read_text()), n)
    for r in sel:
        flat_usd(r, OUT / "wrappers")
        print("prep", r["name"], flush=True)
    (OUT / "selection.json").write_text(json.dumps([{k: r[k] for k in ("name", "library", "category", "extents", "jointed", "checks")}
                                                    for r in sel], indent=1))


def blender_main():
    import bpy
    import mathutils
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opt = lambda k, d: int(argv[argv.index(k) + 1]) if k in argv else d
    n, px, samples = opt("--n", 48), opt("--px", 512), opt("--samples", 64)
    if "--test" in argv:                                  # render the named assets only (before the manifest exists)
        names = argv[argv.index("--test") + 1].split(",")
        sel = [{"name": nm, "library": "", "category": "", "extents": [0.1], "jointed": False,
                "checks": {"size": True}} for nm in names]
        global OUT
        OUT = OUT / "test"
    else:
        sel = json.loads((OUT / "selection.json").read_text())
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    (OUT / "wrappers").mkdir(parents=True, exist_ok=True)
    for r in sel:
        tile = OUT / "tiles" / f"{r['name']}.png"
        if tile.exists():
            continue
        wrap = OUT / "wrappers" / f"{r['name']}.flat.usda"
        bpy.ops.wm.read_factory_settings(use_empty=True)
        world()
        on_table = r["checks"]["size"]
        ground(on_table)
        before = set(bpy.data.objects)
        bpy.ops.wm.usd_import(filepath=str(wrap), import_guide=False, import_proxy=False, import_render=True,
                              import_cameras=False, import_lights=False, scale=1.0)
        newall = [o for o in bpy.data.objects if o not in before]
        # the rest transform is applied here, not in the wrapper: rigid parts authored with resetXformStack (ForgeCAD
        # caps, stapler arms, lamp heads) ignore a transform above them in USD
        import numpy as np
        T = mathutils.Matrix(np.load(REPO / "data/assets/simready/proc" / f"{r['name']}.npz")["rest_T"].tolist())
        bpy.context.view_layer.update()
        for o in newall:
            if o.parent is None:
                o.matrix_world = T @ o.matrix_world
        new = [o for o in newall if o.type == "MESH" and not o.hide_render]
        bpy.context.view_layer.update()
        pts = [o.matrix_world @ v.co for o in new for v in o.data.vertices]     # true extent (a rotated bound box is larger)
        if not pts:
            print(f"[render] {r['name']}: nothing imported", flush=True)
            continue
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        dz = -lo[2] + (0.0 if on_table else -TABLE_TOP[2])      # rest on the table top (z = 0) or the floor
        for o in newall:
            if o.parent is None:
                o.location.z += dz
        c = mathutils.Vector([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (hi[2] - lo[2]) / 2 + (0.0 if on_table else -TABLE_TOP[2])])
        rad = 0.5 * math.dist(lo, hi)
        cam(c, rad)
        settings(px, samples)
        bpy.context.scene.render.filepath = str(tile)
        bpy.ops.render.render(write_still=True)
        print(f"[render] {r['name']} ok", flush=True)


def world():
    import bpy
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(HDRI))
    light, plain = nt.nodes.new("ShaderNodeBackground"), nt.nodes.new("ShaderNodeBackground")
    plain.inputs["Color"].default_value = (0.93, 0.93, 0.92, 1.0)
    plain.inputs["Strength"].default_value = 1.8
    path, mix, out = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], light.inputs["Color"])
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(light.outputs["Background"], mix.inputs[1])
    nt.links.new(plain.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def ground(with_table):
    import bmesh
    import bpy
    import mathutils
    if with_table:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(TABLE))
        new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        bpy.context.view_layer.update()
        pts = [o.matrix_world @ v.co for o in new for v in o.data.vertices]     # true extent (a rotated bound box is larger)
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        s = [TABLE_TOP[i] / (hi[i] - lo[i]) for i in range(3)]
        fit = mathutils.Matrix.Translation((0, 0, -s[2] * hi[2])) @ mathutils.Matrix.Diagonal((*s, 1.0))
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
    m = bpy.data.materials.new("floor")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.36, 0.33, 0.30, 1.0)
    b.inputs["Roughness"].default_value = 0.85
    fl.data.materials.append(m)


def cam(c, rad, fov=35.0):
    import bpy
    import mathutils
    d = rad / math.sin(math.radians(fov / 2)) * 1.08
    eye = c + mathutils.Vector((math.cos(math.radians(-60)) * 0.80, math.sin(math.radians(-60)) * 0.80, 0.60)).normalized() * d
    cd = bpy.data.cameras.new("cam")
    ob = bpy.data.objects.new("cam", cd)
    bpy.context.collection.objects.link(ob)
    ob.location = eye
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (c - eye).to_track_quat("-Z", "Y")
    cd.angle = math.radians(fov)
    cd.clip_start = 0.005
    bpy.context.scene.camera = ob


def settings(px, samples):
    import bpy
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x = sc.render.resolution_y = px
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.exposure = -0.9
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


def sheet(cols=8, px=320):
    from PIL import Image, ImageDraw, ImageFont
    sel = json.loads((OUT / "selection.json").read_text())
    sel = [s for s in sel if (OUT / "tiles" / f"{s['name']}.png").exists()]
    rows = (len(sel) + cols - 1) // cols
    lab = 34
    im = Image.new("RGB", (cols * px, rows * (px + lab)), "white")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, s in enumerate(sel):
        x, y = (i % cols) * px, (i // cols) * (px + lab)
        im.paste(Image.open(OUT / "tiles" / f"{s['name']}.png").convert("RGB").resize((px, px)), (x, y))
        ext = max(s["extents"]) * 100
        dr.text((x + 4, y + px + 2), s["name"].split("__", 1)[1][:44], fill="black", font=font)
        dr.text((x + 4, y + px + 17), f"{s['library'].split('/')[0]} | {s['category']} | {ext:.0f} cm"
                + (" | jointed" if s["jointed"] else ""), fill=(90, 90, 90), font=font)
    im.save(OUT / "gallery.png")
    print("wrote", OUT / "gallery.png", len(sel), "tiles")


if __name__ == "__main__":
    if "--sheet" in sys.argv:
        sheet()
    elif "--prep" in sys.argv:
        prep(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 48)
    else:
        blender_main()
