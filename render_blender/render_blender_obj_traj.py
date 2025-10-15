# render_blender_obj_traj_fix.py
# Usage:
# blender -b --python render_blender_obj_traj_fix.py -- /path/to/object.obj /path/to/output_dir 81

import bpy
import sys
import os
from mathutils import Vector
import traceback

def enable_obj_addon():
    """
    尝试启用 io_scene_obj addon（Wavefront OBJ importer）。
    返回 True 如果成功启用或已启用，False 如果不可用。
    """
    addon_name = "io_scene_obj"
    try:
        # 如果已经启用，直接返回 True
        if addon_name in bpy.context.preferences.addons.keys():
            print(f"Addon {addon_name} already enabled.")
            return True

        # 尝试使用 wm.addon_enable 启用
        bpy.ops.wm.addon_enable(module=addon_name)
        # 若无异常，检查是否生效
        if addon_name in bpy.context.preferences.addons.keys():
            print(f"Successfully enabled addon: {addon_name}")
            return True
        else:
            print(f"Called addon_enable but addon not listed in preferences.addons.")
            return False
    except Exception as e:
        print(f"Exception while enabling addon {addon_name}: {e}")
        # 打印可用的 addons 目录和现有 addon 列表供诊断
        try:
            print("bpy.utils.script_paths():", bpy.utils.script_paths())
        except Exception:
            pass
        try:
            print("Installed addon modules (keys):", list(bpy.context.preferences.addons.keys()))
        except Exception:
            pass
        return False

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # remove meshes, materials data-blocks (to keep memory clean)
    for block in list(bpy.data.meshes):
        try:
            bpy.data.meshes.remove(block, do_unlink=True)
        except Exception:
            pass
    for block in list(bpy.data.materials):
        try:
            bpy.data.materials.remove(block, do_unlink=True)
        except Exception:
            pass

def import_obj(obj_path):
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    # Ensure OBJ importer addon enabled
    ok = enable_obj_addon()
    if not ok:
        # 如果没法启用，抛出带诊断信息的异常
        msg = (
            "OBJ importer addon 'io_scene_obj' could not be enabled or found.\n"
            "Possible causes:\n"
            " - Blender's scripts/addons directory is missing from this installation.\n"
            " - This Blender build was stripped of addons.\n"
            "Diagnostics: print bpy.utils.script_paths() and bpy.context.preferences.addons.keys().\n"
            "Workarounds:\n"
            " 1) Run with a Blender that has addons, or install/enable io_scene_obj in your Blender.\n"
            " 2) Use a pure-Python OBJ loader (not implemented here) to create mesh data.\n"
            " 3) Provide the output of the diagnostic commands to the maintainer for help.\n"
        )
        raise RuntimeError(msg)

    # 现在尝试导入
    prev_selected = set(bpy.context.selected_objects)
    bpy.ops.import_scene.obj(filepath=obj_path)
    imported = [o for o in bpy.context.selected_objects if o not in prev_selected]
    if not imported:
        # 有些情况下导入会把对象加入场景但选择状态不同，尝试抓取新添加的 mesh objects
        imported = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not imported:
        raise RuntimeError("OBJ import produced no objects (after enabling addon).")
    return imported

# --- 其余函数大体与原脚本相同（渲染/摄像机/光照等） ---
def compute_bbox_center_and_size(objects):
    bb_min = Vector((1e9,1e9,1e9))
    bb_max = Vector((-1e9,-1e9,-1e9))
    for o in objects:
        if not hasattr(o, "bound_box"):
            continue
        for corner in o.bound_box:
            world_corner = o.matrix_world @ Vector(corner)
            bb_min.x = min(bb_min.x, world_corner.x)
            bb_min.y = min(bb_min.y, world_corner.y)
            bb_min.z = min(bb_min.z, world_corner.z)
            bb_max.x = max(bb_max.x, world_corner.x)
            bb_max.y = max(bb_max.y, world_corner.y)
            bb_max.z = max(bb_max.z, world_corner.z)
    center = (bb_min + bb_max) / 2.0
    size = bb_max - bb_min
    max_dim = max(size.x, size.y, size.z)
    return center, size, max_dim

def setup_camera_at(center, cam_offset_vec):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = center + cam_offset_vec
    empty = bpy.data.objects.new("TrackTarget", None)
    empty.location = center
    bpy.context.collection.objects.link(empty)
    cons = cam.constraints.new(type='TRACK_TO')
    cons.target = empty
    cons.track_axis = 'TRACK_NEGATIVE_Z'
    cons.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam
    return cam, empty

def create_basic_light(center, distance):
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector((distance, distance, distance))
    light.rotation_euler = (0.785, 0, 0.785)
    light_data2 = bpy.data.lights.new(name="Area", type='AREA')
    light2 = bpy.data.objects.new(name="Area", object_data=light_data2)
    bpy.context.collection.objects.link(light2)
    light2.location = center + Vector((-distance, -distance, distance))
    return light, light2

def setup_render(output_dir, resolution=1024, engine='CYCLES', samples=32):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    try:
        scene.render.engine = engine
        if engine == 'CYCLES':
            scene.cycles.samples = samples
    except Exception as e:
        print("Falling back to EEVEE or default engine:", e)
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.image_settings.file_format = 'PNG'
    scene.use_nodes = False
    try:
        scene.view_settings.view_transform = 'Standard'
    except:
        pass

def animate_camera_linear(cam, start_loc, end_loc, start_frame, end_frame):
    cam.location = start_loc
    cam.keyframe_insert(data_path="location", frame=start_frame)
    cam.location = end_loc
    cam.keyframe_insert(data_path="location", frame=end_frame)

def render_frames(output_dir, start_frame, end_frame):
    scene = bpy.context.scene
    os.makedirs(output_dir, exist_ok=True)
    for f in range(start_frame, end_frame + 1):
        scene.frame_set(f)
        filename = f"frame_{f:03d}.png"
        scene.render.filepath = os.path.join(output_dir, filename)
        print(f"Rendering frame {f} -> {scene.render.filepath}")
        bpy.ops.render.render(write_still=True)

def main():
    argv = sys.argv
    if "--" in argv:
        idx = argv.index("--")
        script_args = argv[idx+1:]
    else:
        script_args = []

    if len(script_args) < 2:
        print("Usage: blender -b --python render_blender_obj_traj_fix.py -- /path/to/object.obj /path/to/output_dir [num_frames]")
        return

    obj_path = script_args[0]
    output_dir = script_args[1]
    num_frames = int(script_args[2]) if len(script_args) >= 3 else 81

    print("OBJ:", obj_path)
    print("OUTPUT:", output_dir)
    print("FRAMES:", num_frames)

    try:
        clear_scene()
        imported_objects = import_obj(obj_path)

        center, size, max_dim = compute_bbox_center_and_size(imported_objects)
        print("Object center:", center, "size:", size, "max_dim:", max_dim)

        dist = max(1.0, max_dim * 2.5)
        height = max_dim * 0.3
        move_extent = max(1.5, max_dim * 3.0)
        start_x = center.x - move_extent/2.0
        end_x   = center.x + move_extent/2.0
        start_loc = Vector((start_x, center.y - dist, center.z + height))
        end_loc   = Vector((end_x,   center.y - dist, center.z + height))

        print("Camera start loc:", start_loc, "end loc:", end_loc)

        cam, empty = setup_camera_at(center, cam_offset_vec=start_loc - Vector((0,0,0)))
        create_basic_light(center, dist)
        setup_render(output_dir, resolution=1024, engine='CYCLES', samples=64)
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = num_frames
        animate_camera_linear(cam, start_loc, end_loc, 1, num_frames)
        render_frames(output_dir, 1, num_frames)
        print("Rendering complete.")
    except Exception as e:
        print("ERROR during rendering:")
        traceback.print_exc()
        # 在失败时打印有助诊断的信息
        try:
            print("Installed addons:", list(bpy.context.preferences.addons.keys()))
        except Exception:
            pass
        try:
            print("Script paths:", bpy.utils.script_paths())
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
