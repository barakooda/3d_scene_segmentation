import bpy
from src.constants import BASE_PATH
from src.utils import get_active_lights,get_globally_visible_object_names,unhide_globally_objects,activate_lights_in_render,save_dict_to_json,get_objects_name_to_object_attribute
from src.object_classifier_utils import create_new_world_material_with_sky,classify_objects

from src.ai_image_classifier import AIImageClassifier

from src.constants import SEMANTIC_NAME



object_color_to_string_dict = {}
object_part_color_to_string_dict = {}

IN_FILE_PATH = BASE_PATH / "input" / "object_classifier_test_scene.blend"


# Ensure file exists before proceeding
if not IN_FILE_PATH.exists():
    print(f"Error: The file at {IN_FILE_PATH} does not exist.")
    exit(1)

classifier = AIImageClassifier(model_accuracy='s', max_labels=10000, seed=13)

bpy.ops.wm.open_mainfile(filepath=str(IN_FILE_PATH))

#hide all objects except the object
globally_visible_object_names = get_globally_visible_object_names(scene_name="Scene",view_layer_name = "ViewLayer")
print(globally_visible_object_names)



original_world = bpy.data.scenes[0].world
active_lights = get_active_lights(bpy.data.scenes[0])
create_new_world_material_with_sky(bpy.data.scenes[0])

classify_objects(classifier, globally_visible_object_names)


object_name_to_semantic_name_dict = get_objects_name_to_object_attribute(globally_visible_object_names,SEMANTIC_NAME)

file_path = BASE_PATH / "output" / "object_name_to_semantic_name.json"
save_dict_to_json(file_path=str(file_path),dictionary = object_name_to_semantic_name_dict)


bpy.data.scenes[0].world = original_world
unhide_globally_objects(globally_visible_object_names)
activate_lights_in_render(active_lights)
