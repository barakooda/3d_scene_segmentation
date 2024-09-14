import bpy
from src.constants import BB_MIN, BB_MAX, CENTER

from src.logger_config import setup_logger

# Set up logging
logger = setup_logger(__name__)

def get_context_window():
    for window in bpy.context.window_manager.windows:
        if window:
            return window
    return None


def clean_all_material_nodes(material_nodes: bpy.types.bpy_prop_collection):
    for node in material_nodes:
        material_nodes.remove(node)

def create_shader_node(
    material: bpy.types.Material,
    node_name: str = "attribute_node_name01",
    node_type: str = "ShaderNodeAttribute",
    node_location: tuple = (0, 0),
    node_size: tuple = (140, 140),
) -> bpy.types.ShaderNode:
    shader_node = material.node_tree.nodes.new(type=node_type)
    shader_node.name = node_name
    shader_node.label = node_name
    shader_node.location = node_location
    shader_node.width = node_size[0]
    shader_node.height = node_size[1]

    return shader_node


def remove_prefix_from_string(group_name:str ,prefix_list:list) -> str:
    if prefix_list is None:
        return group_name
    
    for prefix in prefix_list:
        if group_name.startswith(prefix):
            group_name = group_name[len(prefix):]
            break
    return group_name

def add_material_to_object(obj: bpy.types.Object, material: bpy.types.Material):

    logger.info(f"Adding material {material.name} to object {obj.name}")

    if obj.material_slots:
        for slot in obj.material_slots:
            if slot.material is None:
                obj.data.materials.append(material)
                continue
           
            if slot.is_property_readonly('material'):
                obj.data.materials.clear()
                obj.data.materials.append(material)
                
            else:
                
                slot.material = material

    else:
        obj.data.materials.append(material)


def filter_by_startwith_names(objects: list = None, prefixes: list = None) -> list:
    #Filter a list of objects by checking if their 'name' attribute starts with any of the specified prefixes.

    print(prefixes)
    if objects is None or prefixes is None:
        return objects
    
    filtered_objects = []   
    for obj in objects:
        if obj.name.startswith(tuple(prefixes)):
            filtered_objects.append(obj)
    
    print (f"Filtered objects: {[obj.name for obj in filtered_objects]}")

    return filtered_objects


def is_point_in_bounding_box(polygon_data: dict, vertex_group_data: dict):
    bb_min = vertex_group_data[BB_MIN]
    bb_max = vertex_group_data[BB_MAX]
    polygon_center = polygon_data[CENTER]

    out_of_bbox = (
        polygon_center[0] < bb_min[0]
        or polygon_center[1] < bb_min[1]
        or polygon_center[2] < bb_min[2]
        or polygon_center[0] > bb_max[0]
        or polygon_center[1] > bb_max[1]
        or polygon_center[2] > bb_max[2]
    )
    return out_of_bbox



def apply_rotation_to_vertices(obj: bpy.types.Object) -> None:
    # Ensure the object is a mesh
    if obj.type != 'MESH':
        return
    
    # Get the mesh data from the object
    mesh = obj.data
    
    # Extract the rotation component from the object's transformation matrix
    # Decompose the world matrix into its components
    loc, rot, scale = obj.matrix_world.decompose()
    
    # Create a rotation matrix from the rotation component
    rotation_matrix = rot.to_matrix().to_4x4()
    
    # Apply the rotation matrix to each vertex
    for vertex in mesh.vertices:
        vertex.co = rotation_matrix @ vertex.co
    
    # Reset only the rotation of the object's transformation matrix
    obj.rotation_euler = (0, 0, 0)
    
    mesh.update()
    bpy.context.view_layer.update()



def get_active_lights(scene: bpy.types.Scene) -> list:
    active_lights = []  # List to store active lights

    # Loop through all objects in the scene to find active lights
    for obj in scene.objects:
        if obj.type == 'LIGHT' and not obj.hide_render:
            active_lights.append(obj)  # Store the light object in the list

    return active_lights

def deactivate_lights_in_render(lights_list):
    for light in lights_list:
        light.hide_render = True  # Deactivate individual lights in render

def activate_lights_in_render(lights_list):
    for light in lights_list:
        light.hide_render = False  # Activate individual lights in render


def get_globally_visible_object_names(scene_name : str = "Scene",view_layer_name : str = "ViewLayer")->list[str]:
    # Return a list of names of objects that are not globally hidden
    
    scene = bpy.data.scenes.get(scene_name)
    view_layer = scene.view_layers.get(view_layer_name)

    return [obj.name for obj in bpy.data.objects if not obj.hide_render and obj in view_layer.objects.values()]

def hide_all_except_one_globally(visible_object_names:list[str], chosen_object_name:str)->None:
    # Loop through the list of object names and globally hide all except the chosen one
    for obj_name in visible_object_names:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            if obj.name == chosen_object_name:
                obj.hide_render = False
                obj.hide_viewport = False
            else:
                obj.hide_render = True 
                obj.hide_viewport = True 

def unhide_globally_objects(visible_object_names :list[str])->None:
    # Loop through the list of objects and make sure they are not hidden
    for obj_name in visible_object_names:
        
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.hide_render = False # Unhide the object
            obj.hide_viewport = False
