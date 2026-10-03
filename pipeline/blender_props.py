"""Headless Blender prop shots (3D coins, jars ...) for the money videos.

Runs inside GitHub Actions (see .github/workflows/blender-props.yml) with
no desktop needed:

    blender -b -noaudio -P pipeline/blender_props.py -- --shot jars --out out/props/jars

It renders a transparent PNG sequence (RGBA, with a shadow-catcher floor so
the shadows land on whatever footage is placed underneath). The workflow then
turns the frames into a transparent .webm and a preview .mp4.

Only use it where a 3D prop shot is worth it; character acting stays with
the Gemini clips.

Shots so far:
  jars   Three glass jars (red, green, blue) and gold coins that drop into
         them with real physics. --counts 5 3 2 gives the 50/30/20 split
         (spend / save / share) as coin counts.
Add a new shot by writing build_<name>(args, scene) and listing it in SHOTS.
"""
import argparse
import math
import os
import random
import sys

import bmesh
import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--shot", default="jars")
    p.add_argument("--out", default="out/props/jars")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--res", type=int, nargs=2, default=[720, 1280])
    p.add_argument("--counts", type=int, nargs=3, default=[5, 3, 2])
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--denoise", type=int, default=1)
    p.add_argument("--first", type=int, default=0, help="render from this frame")
    p.add_argument("--last", type=int, default=0, help="render up to this frame")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--still", type=int, default=0,
                   help="render only this one frame (quick look)")
    return p.parse_args(argv)


# ---------- small helpers -------------------------------------------------

def set_input(node, names, value):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return


def make_material(name, color, metallic=0.0, rough=0.5, transmission=0.0, ior=1.45):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    set_input(b, ["Base Color"], (color[0], color[1], color[2], 1.0))
    set_input(b, ["Metallic"], metallic)
    set_input(b, ["Roughness"], rough)
    set_input(b, ["Transmission Weight", "Transmission"], transmission)
    set_input(b, ["IOR"], ior)
    return m


def link(scene, ob):
    scene.collection.objects.link(ob)
    return ob


def activate(ob):
    bpy.context.view_layer.objects.active = ob
    for o in bpy.context.view_layer.objects:
        o.select_set(o == ob)


def smooth(ob):
    for poly in ob.data.polygons:
        poly.use_smooth = True


# ---------- scene setup ---------------------------------------------------

def setup_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    # needs the official Blender build (the apt one has no denoiser: --denoise 0)
    scene.cycles.use_denoising = bool(args.denoise)
    if args.denoise:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.03
    scene.cycles.sample_clamp_indirect = 3.0
    scene.cycles.glossy_bounces = 3
    scene.cycles.transparent_max_bounces = 6
    scene.cycles.max_bounces = 5
    scene.render.film_transparent = True
    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.frame_start = 1
    scene.frame_end = int(args.seconds * args.fps)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.92, 0.9, 0.95, 1)
    bg.inputs["Strength"].default_value = 0.7
    scene.world = world

    # warm key light (window side) + soft fill
    for name, loc, power, size in (("key", (-2.2, -2.4, 3.4), 450, 3.0),
                                   ("fill", (2.6, -2.0, 1.8), 120, 3.5)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = power
        ld.size = size
        ld.color = (1.0, 0.92, 0.8) if name == "key" else (0.85, 0.9, 1.0)
        lo = link(scene, bpy.data.objects.new(name, ld))
        lo.location = loc
        con = lo.constraints.new("TRACK_TO")
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"
    return scene


def add_camera(scene, target=(0, 0, 0.2), start=(0, -2.9, 2.9), end=(0, -2.6, 2.7)):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 24
    cam = link(scene, bpy.data.objects.new("cam", cam_data))
    tgt = link(scene, bpy.data.objects.new("target", None))
    tgt.location = target
    con = cam.constraints.new("TRACK_TO")
    con.target = tgt
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    cam.location = start
    cam.keyframe_insert("location", frame=scene.frame_start)
    cam.location = end   # slow push-in so the shot feels alive
    cam.keyframe_insert("location", frame=scene.frame_end)
    scene.camera = cam


def add_floor(scene):
    """Invisible floor: catches shadows (and any coin that misses a jar)."""
    bpy.ops.mesh.primitive_cube_add(size=1)
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.scale = (14, 14, 0.2)
    floor.location = (0, 0, -0.1)
    floor.is_shadow_catcher = True
    activate(floor)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    floor.rigid_body.collision_shape = "BOX"
    floor.rigid_body.friction = 0.8


# ---------- shot: jars ----------------------------------------------------

JAR_R, JAR_H = 0.32, 0.45
COIN_R, COIN_T = 0.085, 0.018


def make_jar(scene, name, x, tint):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=40,
                          radius1=JAR_R, radius2=JAR_R, depth=JAR_H)
    bm.faces.ensure_lookup_table()
    top = [f for f in bm.faces if f.normal.z > 0.9]
    bmesh.ops.delete(bm, geom=top, context="FACES")      # open top
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    jar = link(scene, bpy.data.objects.new(name, me))
    jar.location = (x, 0, JAR_H / 2)
    smooth(jar)
    mod = jar.modifiers.new("wall", "SOLIDIFY")
    mod.thickness = 0.035
    mod.offset = 1.0
    activate(jar)
    bpy.ops.object.modifier_apply(modifier="wall")
    jar.data.materials.append(make_material(name + "_glass", tint, rough=0.04,
                                            transmission=1.0, ior=1.06))
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    jar.rigid_body.collision_shape = "MESH"
    jar.rigid_body.friction = 0.6
    # glass would cast a solid black shadow in Cycles; use a soft blob instead
    jar.visible_shadow = False
    add_blob_shadow(scene, name + "_shadow", x)
    return jar


def add_blob_shadow(scene, name, x, strength=0.5):
    """Soft round contact shadow under a jar (the floor only catches shadows)."""
    bpy.ops.mesh.primitive_plane_add(size=2)
    pl = bpy.context.active_object
    pl.name = name
    pl.location = (x + 0.03, 0.03, 0.003)
    pl.scale = (0.5, 0.5, 1)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    set_input(b, ["Base Color"], (0.05, 0.03, 0.02, 1.0))
    set_input(b, ["Roughness"], 1.0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    gr = nt.nodes.new("ShaderNodeTexGradient")
    gr.gradient_type = "SPHERICAL"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = strength
    nt.links.new(tc.outputs["Object"], gr.inputs["Vector"])
    nt.links.new(gr.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], b.inputs["Alpha"])
    pl.data.materials.append(m)
    pl.visible_shadow = False


def build_jars(args, scene):
    rnd = random.Random(args.seed)
    xs = (-0.82, 0.0, 0.82)   # spend, save, share
    # light tints so the gold coins stay visible through the glass
    tints = ((1.0, 0.62, 0.58), (0.62, 0.95, 0.68), (0.62, 0.78, 1.0))
    names = ("spend", "save", "share")
    for n, x, t in zip(names, xs, tints):
        make_jar(scene, "jar_" + n, x, t)

    # one gold coin mesh, shared by every coin
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=32,
                          radius1=COIN_R, radius2=COIN_R, depth=COIN_T)
    coin_mesh = bpy.data.meshes.new("coin")
    bm.to_mesh(coin_mesh)
    bm.free()
    gold = make_material("gold", (1.0, 0.72, 0.18), metallic=1.0, rough=0.28)
    coin_mesh.materials.append(gold)

    # drop order: interleave the jars so the counts build up together
    queue = []
    left = list(args.counts)
    while any(left):
        for j in range(3):
            if left[j]:
                queue.append(j)
                left[j] -= 1
    total = len(queue)
    first, last = 14, scene.frame_end - 60      # leave ~2.5 s for coins to settle
    gap = max(3, (last - first) // max(total, 1))

    for i, j in enumerate(queue):
        coin = link(scene, bpy.data.objects.new("coin%02d" % i, coin_mesh))
        # every held coin gets its own height so none overlap (an overlap
        # flings the coin out of the jar the moment it is released)
        coin.location = (xs[j] + rnd.uniform(-0.08, 0.08),
                         rnd.uniform(-0.08, 0.08), 1.2 + 0.07 * i)
        coin.rotation_euler = (rnd.uniform(-0.5, 0.5), rnd.uniform(-0.5, 0.5),
                               rnd.uniform(0, math.pi))
        smooth(coin)
        activate(coin)
        bpy.ops.rigidbody.object_add(type="ACTIVE")
        rb = coin.rigid_body
        rb.collision_shape = "CONVEX_HULL"
        rb.mass = 0.02
        rb.friction = 0.6
        rb.restitution = 0.25
        rb.use_margin = True
        rb.collision_margin = 0.001
        drop = first + i * gap
        # held in the air (invisible) until its drop frame
        rb.kinematic = True
        rb.keyframe_insert("kinematic", frame=1)
        coin.hide_render = True
        coin.keyframe_insert("hide_render", frame=1)
        rb.kinematic = False
        rb.keyframe_insert("kinematic", frame=drop)
        coin.hide_render = False
        coin.keyframe_insert("hide_render", frame=drop)

    rbw = scene.rigidbody_world
    rbw.time_scale = 1.0
    rbw.substeps_per_frame = 12
    rbw.solver_iterations = 30
    rbw.point_cache.frame_start = 1
    rbw.point_cache.frame_end = scene.frame_end


SHOTS = {"jars": build_jars}


def main():
    args = parse_args()
    scene = setup_scene(args)
    add_camera(scene)
    add_floor(scene)
    SHOTS[args.shot](args, scene)

    # step through the frames once, in order, so the physics is simulated
    for f in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(f)

    os.makedirs(args.out, exist_ok=True)
    probe = args.still or scene.frame_start
    scene.frame_set(probe)
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in [o for o in scene.objects if o.name.startswith("coin")][:4]:
        ev = ob.evaluated_get(dg)
        print("PROBE frame", probe, ob.name, "hide_render", ob.hide_render,
              "pos", tuple(round(v, 2) for v in ev.matrix_world.translation))
    if args.still:
        scene.frame_set(args.still)
        scene.render.filepath = os.path.join(args.out, "still.png")
        bpy.ops.render.render(write_still=True)
    else:
        if args.first:
            scene.frame_start = args.first
        if args.last:
            scene.frame_end = args.last
        scene.frame_set(scene.frame_start)
        scene.render.filepath = os.path.join(args.out, "f_")
        bpy.ops.render.render(animation=True)
    print("DONE", args.shot, args.out)


main()
