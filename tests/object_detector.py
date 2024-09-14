import bpy
from src.constants import BASE_PATH
from src.utils import get_active_lights,get_globally_visible_object_names,hide_all_except_one_globally,unhide_globally_objects
from src.object_detector_utils import create_new_world_material_with_sky,create_bbox_and_cameras,render_from_cameras_to_memory



object_color_to_string_dict = {}
object_part_color_to_string_dict = {}

IN_FILE_PATH = BASE_PATH / "input" / "object_detector_test_scene.blend"


# Ensure file exists before proceeding
if not IN_FILE_PATH.exists():
    print(f"Error: The file at {IN_FILE_PATH} does not exist.")
    exit(1)
    
bpy.ops.wm.open_mainfile(filepath=str(IN_FILE_PATH))

#hide all objects except the object
globally_visible_object_names = get_globally_visible_object_names(scene_name="Scene",view_layer_name = "ViewLayer")
print(globally_visible_object_names)
hide_all_except_one_globally(globally_visible_object_names,'object')



obj = bpy.data.objects['object']



original_world = bpy.data.scenes[0].world

active_lights = get_active_lights(bpy.data.scenes[0])
#deactivate_lights_in_render(active_lights)

create_new_world_material_with_sky(bpy.data.scenes[0])

bbox_and_cameras = create_bbox_and_cameras(obj)

rendered_images = render_from_cameras_to_memory(scene_name = bpy.data.scenes[0].name,camera_names = bbox_and_cameras[1])


unhide_globally_objects(globally_visible_object_names)

#show rendered images using imshow from opencv
#visualize_images(rendered_images)






#bpy.data.scenes[0].world = original_world
#activate_lights_in_render(active_lights)
