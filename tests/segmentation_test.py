import pathlib
import json

import bpy

from seg_utils import convert_all_curve_particle_instances_to_mesh
from seg_utils import set_sub_segmentation,set_object_segmentation,set_particle_system_segmentation

from seg_materials import create_segmentation_material
from constants import MESH, CURVE,META,CURVES,OBJECT_SEGMENTATION,SUB_SEGMENTATION,GEOMETRY,SEMANTIC_NAME,SUB_SEGMENTATION_PREFIX
from render_settings import set_segmentation_render_settings


base_path = pathlib.Path(__file__).parent.parent

object_color_to_string_dict = {}
object_part_color_to_string_dict = {}

IN_FILE_PATH = base_path / "input" / "blender-3.5-splash.blend"
OUT_FILE_PATH = base_path / "output" / "segmented_scene.blend"
object_color_json_file = base_path / "output" / "object_color_to_string_dict.json"
object_parts_color_json_file = base_path / "output" / "object_parts_color_to_string_dict.json"

# Ensure file exists before proceeding
if not IN_FILE_PATH.exists():
    print(f"Error: The file at {IN_FILE_PATH} does not exist.")
    exit(1)
    
bpy.ops.wm.open_mainfile(filepath=str(IN_FILE_PATH))

convert_all_curve_particle_instances_to_mesh()

objects_for_segmentation = [obj for obj in bpy.data.objects if obj.type in [MESH, CURVE, CURVES,META]]

object_segmentation_material = create_segmentation_material( name = OBJECT_SEGMENTATION,attribute_name = OBJECT_SEGMENTATION)

sub_segmentation_material = create_segmentation_material( name = SUB_SEGMENTATION,attribute_name = SUB_SEGMENTATION,attribute_type=GEOMETRY)


for obj in objects_for_segmentation:
        
    color = set_object_segmentation(obj = obj,object_segmentation_material = object_segmentation_material)
    object_color_to_string_dict[str(tuple(color))] = obj.name

    sub_segmentation_dict = set_sub_segmentation(obj = obj,prefix_list=SUB_SEGMENTATION_PREFIX,sub_segmentation_material=sub_segmentation_material)
    object_part_color_to_string_dict.update(sub_segmentation_dict)

    set_particle_system_segmentation(obj = obj,segmentation_custom_property=SEMANTIC_NAME)

current_scene = bpy.context.scene
set_segmentation_render_settings(scene_name=current_scene.name)

with open(object_color_json_file, 'w') as f:
    json.dump(object_color_to_string_dict, f)

with open(object_parts_color_json_file, 'w') as f:
    json.dump(object_part_color_to_string_dict, f)


output_file_path = pathlib.Path(OUT_FILE_PATH)
bpy.ops.wm.save_as_mainfile(filepath=str(output_file_path), copy=True, compress=True)

    
