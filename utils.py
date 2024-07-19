import bpy
from constants import BB_MIN, BB_MAX, CENTER

from logger_config import setup_logger

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