from collections import defaultdict
import bpy
import bmesh
from utils import add_material_to_object,filter_by_startwith_names,is_point_in_bounding_box
import hashlib
from constants import SEMANTIC_NAME,UNDEFINED_OBJECT,MAX_UINT16,OBJECT_SEGMENTATION,ATTRIBUTE_NAME
from constants import INDICES, BB_MIN, BB_MAX,SUB_SEGMENTATION_ATTRIBUTE_NAME, SUB_SEGMENTATION
from seg_materials import create_hair_particle_system_material


from logger_config import setup_logger

# Set up logging
logger = setup_logger(__name__)




def convert_all_curve_particle_instances_to_mesh() -> None:
    objects = get_all_particle_systems_instance_objects()

    if objects is {}:

        logger.info(f"there are no use of particle system instances")
        return

    for obj in objects:
        if obj.type == "CURVE":
            obj.hide_viewport = False
            convert_curve_to_mesh(obj.name)


def convert_curve_to_mesh(obj_name: str) -> None:
    """
    Converts a Blender curve object to a mesh object if it is not already a mesh.

    Args:
        obj_name (str): The name of the object to convert.
    """
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "CURVE":
        print(f"Object not found or not a curve: {obj_name}")
        return

    try:
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        
        # Select and set the curve object as active
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.update()

        # Ensure there's a VIEW_3D area to perform the conversion
        view_3d_area = next((area for area in bpy.context.screen.areas if area.type == "VIEW_3D"), None)
        if not view_3d_area:
            raise ValueError("No VIEW_3D area found")

        # Temporarily override context to VIEW_3D for conversion
        with bpy.context.temp_override(area=view_3d_area):
            bpy.ops.object.convert(target='MESH')
        
    except Exception as e:
        print(f"Could not convert {obj.name} to mesh: {e}")

    finally:
        # Ensure all objects are deselected after operation
        bpy.ops.object.select_all(action='DESELECT')


def get_all_particle_systems_instance_objects() -> set:
    instance_objects = set()

    for particle_system in bpy.data.particles:
        if particle_system.instance_collection and particle_system.instance_collection.objects:
            instance_objects.update(particle_system.instance_collection.objects)
            particle_system.instance_collection.hide_viewport = False

    return instance_objects


def string_to_rgba_color(name: str,
                       color_bit_depth: int = 16,
                       channels_num:int=3,
                       max_uint_size:int = MAX_UINT16,
                       normalize: bool = True) -> list:
        
        split_size_in_bytes = color_bit_depth // 8

        bytes_array = get_bytes_array_from_string(input_string=name,split_size_in_bytes=split_size_in_bytes,channels_num=channels_num)

        splited_bytes_array = split_bytes_array(bytes_array = bytes_array, split_size = split_size_in_bytes)
        
        color = get_color_from_split_bytes_list(splited_bytes_array,alpha = max_uint_size) 
        if normalize:
            color = [element / max_uint_size for element in color]

        return color

def get_bytes_array_from_string(input_string: str,split_size_in_bytes:int = 2,channels_num:int = 3 ) -> bytes:
    num_of_bytes = split_size_in_bytes * channels_num  # Total bytes for RGB components

    # Convert the input string to bytes
    input_bytes = input_string.encode('utf-8')

    # Create a Blake2b hash object with the specified digest size
    hasher = hashlib.blake2b(digest_size=num_of_bytes)
    hasher.update(input_bytes)  # Update the hash with the input bytes

    # Get the hash digest as bytes
    bytes_array = hasher.digest()
    return bytes_array


def split_bytes_array(bytes_array: bytes, split_size: int = 2) -> list:

    return [bytes_array[index : index + split_size] for index in range(0, len(bytes_array), split_size)]



def get_color_from_split_bytes_list(key_list: list, alpha: int = MAX_UINT16) -> list:

    red = int.from_bytes(key_list[0], "little", signed=False)
    green = int.from_bytes(key_list[1], "little", signed=False)
    blue = int.from_bytes(key_list[2], "little", signed=False)
    return [red, green, blue, alpha]



def set_object_segmentation(obj:bpy.types.Object = None,semantic_name:str = UNDEFINED_OBJECT,object_segmentation:str =  OBJECT_SEGMENTATION, object_segmentation_material:bpy.types.Material = None) ->list:
    

    if not obj:
        print("object is None")
        return [0,0,0] #default color
    
    semantic_name = obj.get(SEMANTIC_NAME)
    if not semantic_name:
        obj[SEMANTIC_NAME] = obj.name
        semantic_name = obj[SEMANTIC_NAME]
        

    
    color = string_to_rgba_color(semantic_name)
    obj[object_segmentation] = color

    
    if object_segmentation_material:
        add_material_to_object(obj, object_segmentation_material)
    
    return color


def is_contain_vertex_groups(obj: bpy.types.Object, vertex_group_filter: str = None) -> bool:
    # Check if there are vertex groups
    if not obj.vertex_groups:
        return False

    # If a filter is provided, check for specific vertex group
    if vertex_group_filter:
        # Return True if any vertex group's name contains the filter
        return any(vertex_group_filter in group.name for group in obj.vertex_groups)
    else:
        # If no filter is provided, return True since there are vertex groups
        return len(obj.vertex_groups) > 0

def set_sub_segmentation(obj:bpy.types.Object = None,prefix_list:list = None,sub_segmentation:str =  SUB_SEGMENTATION, sub_segmentation_material:bpy.types.Material = None) -> dict:
    string_per_color_dict = {}
    
    if is_contain_vertex_groups(obj):
        color_per_string_dict = create_colors_by_groups_names(obj,prefix_list=prefix_list)
        
        if not color_per_string_dict:
            return {}
        
        add_color_attribute_to_face_corner_of_vertex_group(obj, color_per_string_dict, SUB_SEGMENTATION_ATTRIBUTE_NAME)
        if (sub_segmentation_material):
            pass
            add_material_to_object(obj, sub_segmentation_material)

        string_per_color_dict = {str(tuple(value)): key for key, value in color_per_string_dict.items()}

    return string_per_color_dict

def create_colors_by_groups_names(
    obj: bpy.types.Object,
    color_bit_depth: int = 16,
    prefix_list: list = None) -> dict:


    colors_by_groups_names_segmentation_dic = {}
    groups = []
    
    if prefix_list:
        groups = filter_by_startwith_names(obj.vertex_groups, prefixes = prefix_list)
    
    else:
        print("no prefix list")
        return None

    for group in groups:

        group_name = group.name
        colors_by_groups_names_segmentation_dic[group_name] = string_to_rgba_color(group_name, color_bit_depth=color_bit_depth)
    
    return colors_by_groups_names_segmentation_dic



def map_polygons_to_vertex_groups(data_per_polygon, data_per_vertex_group) -> dict:
    polygons_indices_per_group_name = defaultdict(list)

    for group_index, vertex_group_data in data_per_vertex_group.items():

        for polygon_index, polygon_data in enumerate(data_per_polygon):

            out_of_bbox = is_point_in_bounding_box(polygon_data, vertex_group_data)

            if out_of_bbox:
                continue

            if polygon_data["indices"].issubset(vertex_group_data["indices"]):
                polygons_indices_per_group_name[group_index].append(polygon_index)

    return polygons_indices_per_group_name


def get_data_per_polygon(obj: bpy.types.Object = None) -> list[dict]:
    # [{ 'indices':{}},'center':list }]

    data_per_polygon = []

    for polygon in obj.data.polygons:
        center = list(polygon.center)
        indices = set(polygon.vertices)

        data_per_polygon.append({"indices": indices, "center": center})

    return data_per_polygon

def get_vertices_data_per_group(obj: bpy.types.Object, group_filter: dict = None) -> dict:
    """
    Collects vertices data for specified vertex groups within a Blender object.

    Parameters:
    obj (bpy.types.Object): The Blender object with vertex groups.
    group_filter (dict): Optional dictionary to filter vertex groups by name.

    Returns:
    dict: A dictionary with vertex group indices as keys and dictionaries with vertices indices,
          and bounding box minimum and maximum as values.
    """
    # Initialize the dictionary to store vertices data.
    vertices_data_per_group = {}

    # Determine vertex groups to process.
    if group_filter:
        vertex_groups = (vg for vg in obj.vertex_groups if vg.name in group_filter)
    else:
        vertex_groups = obj.vertex_groups

    # Prepare a set of indices for the vertex groups to be processed.
    vertex_group_indices = {vg.index for vg in vertex_groups}

    # Process each vertex in the object.
    for vertex in obj.data.vertices:
        # Find indices of vertex groups that this vertex is part of and are also in our list.
        relevant_groups = vertex_group_indices.intersection(vg.group for vg in vertex.groups)

        # Vertex position
        position = vertex.co[:]

        # Update group data with vertex index and update bounding box.
        for group_index in relevant_groups:
            group_data = vertices_data_per_group.setdefault(group_index, {
                INDICES: set(),
                BB_MIN: [float('inf')] * 3,
                BB_MAX: [-float('inf')] * 3
            })

            # Add vertex index.
            group_data[INDICES].add(vertex.index)

            # Update bounding box.
            for i in range(3):
                group_data[BB_MIN][i] = min(group_data[BB_MIN][i], position[i])
                group_data[BB_MAX][i] = max(group_data[BB_MAX][i], position[i])

    return vertices_data_per_group


def get_polygon_indices_for_vertex_group(obj, segmentation_dic):

    data_per_vertex_group = get_vertices_data_per_group(obj=obj, group_filter=segmentation_dic)


    data_per_polygon = get_data_per_polygon(obj)


    polygons_indices_per_group_name = map_polygons_to_vertex_groups(data_per_polygon, data_per_vertex_group)

    return polygons_indices_per_group_name


def set_faces_corners_color_by_group_name(obj, group_colors, attr_name, group_poly_indices):
    """
    Sets colors for mesh vertex corners based on vertex group names.

    Parameters:
    obj (bpy.types.Object): The mesh object.
    group_colors (dict): A dictionary mapping group names to color tuples.
    attr_name (str): The name for the new color attribute.
    group_poly_indices (dict): A dictionary mapping vertex group indices to polygon indices.
    """
    # Access the object's vertex groups and mesh data
    vertex_groups = obj.vertex_groups
    mesh = obj.data

    # Ensure a color attribute exists, else create one
    if attr_name not in mesh.attributes:
        mesh.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='CORNER')

    # Access the created or existing color attribute
    color_attribute = mesh.attributes[attr_name]

    # Set colors for specified polygons based on their group index
    for group_index, polygons in group_poly_indices.items():
        group_name = vertex_groups[group_index].name
        color = group_colors.get(group_name, (1.0, 1.0, 1.0, 1.0))  # Default to white if group name is missing

        # Apply color to each vertex corner of each polygon
        for poly_index in polygons:
            for loop_index in mesh.polygons[poly_index].loop_indices:
                color_attribute.data[loop_index].color = color


def add_color_attribute_to_face_corner_of_vertex_group(
    obj: bpy.types.Object = None,
    segmentation_dic: dict = None,
    attribute_name: str = ATTRIBUTE_NAME,
) -> None:
    
    if segmentation_dic:
        polygons_indices_per_group_name = get_polygon_indices_for_vertex_group(obj, segmentation_dic)
        set_faces_corners_color_by_group_name(obj, segmentation_dic, attribute_name, polygons_indices_per_group_name)





def set_particle_system_segmentation(obj :bpy.types.Object = None,segmentation_custom_property:str = None)->None:
    
    if (segmentation_custom_property):
        particle_systems = [particle_system for particle_system in obj.particle_systems if segmentation_custom_property in particle_system.settings]
    
    if not particle_systems:
        return

    for particle_system in particle_systems:
        color = string_to_rgba_color(particle_system.settings[segmentation_custom_property]) 
        material = create_hair_particle_system_material(particle_system=particle_system,color= color)

        obj.data.materials.append(material)
        particle_system.settings.material = len(obj.data.materials)
