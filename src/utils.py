import json
import bpy
from src.constants import BB_MIN, BB_MAX, CENTER,SEMANTIC_NAME
import re
import numpy as np
import cv2
from pathlib import Path
import uuid

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

def save_dict_to_json(file_path:str,dictionary:dict)->None: 
    with open(str(file_path), 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)


def get_objects_name_to_object_attribute(globally_visible_object_names:list,attribute_name:str)->dict:
    object_name_to_semantic_name_dict = {}
    for obj_name in globally_visible_object_names:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            if obj.get(attribute_name):
                object_name_to_semantic_name_dict[obj_name] = obj[attribute_name]

    print(object_name_to_semantic_name_dict)
    return object_name_to_semantic_name_dict


def get_string_without_trailing_digits(string:str)->str:
    # Use regex to find the first occurrence of a digit
    match = re.search(r'\d', string)
    
    # If a digit is found, return the string up to that point
    if match:
        return string[:match.start()]
    
    # If no digits are found, return the full string
    return string

# render viewport image function
def render_viewport_image(file_path:str,scene_name:str = "Scene",view_layer_name:str = "ViewLayer",camera_name:str = "Camera")->None:
    
    scene = bpy.data.scenes.get(scene_name)
    

    # Set the camera object by name
    camera = bpy.data.objects.get(camera_name)



    # Check if the camera exists in the scene
    if camera:
        # Set the camera as the active camera for the scene
        bpy.data.scenes['Scene'].camera = camera
    else:
        print("Camera not found")

    # Render the viewport using the active camera
    bpy.data.scenes['Scene'].render.filepath = file_path
    
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':  # Ensure it’s the 3D Viewport
                with bpy.context.temp_override(window=window, area=area):
                    bpy.ops.render.opengl(write_still=True, view_context=True)
                break
    
    
def create_stitched_image(rendered_images, camera_names, base_path, output_size=(1280, 720), image_size=(224, 224)):
    """
    Creates a 2x3 stitched RGBA image from 6 rendered images and places it on the top-right corner of an empty image.
    
    Args:
    - rendered_images: List or dictionary of 6 images to be stitched together.
    - camera_names: List of camera names corresponding to the images.
    - base_path: The base folder where the output image will be saved.
    - output_size: The size of the final output image (default is 1280x720).
    - image_size: The size to which each image will be resized (default is 224x224).
    
    Returns:
    - None. Saves the final image in the output folder with the name of the first camera.
    """
    
    # Create a blank RGBA image initialized to transparent (all zeros, including alpha)
    empty_image = np.zeros((*output_size[::-1], 4), dtype=np.uint8)  # (height, width, 4 channels)

    if len(rendered_images) == 6:
        # Resize all images to the given size and add an alpha channel
        resized_images = []
        for i in range(6):
            # Resize the image
            img = cv2.resize(rendered_images[camera_names[i]], image_size)
            
            # Convert the image to RGBA (add an alpha channel)
            if img.shape[2] == 3:  # If the image is RGB, add alpha
                img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            else:
                img_rgba = img
            
            # Set the alpha channel to 255 (fully opaque)
            img_rgba[:, :, 3] = 255
            resized_images.append(img_rgba)

        # Stitch images in a 2x3 grid
        row1 = cv2.hconcat(resized_images[:2])  # First 2 images (top row)
        row2 = cv2.hconcat(resized_images[2:4])  # Next 2 images (middle row)
        row3 = cv2.hconcat(resized_images[4:])   # Last 2 images (bottom row)

        # Combine the 3 rows to make a 2x3 grid
        stitched_images = cv2.vconcat([row1, row2, row3])

        # Define the top-left corner position for placing the stitched images on the empty image
        x_offset = output_size[0] - stitched_images.shape[1]  # Align it to the right
        y_offset = 0  # Start at the top (y=0)

        # Place the stitched images on the top-right corner of the empty RGBA image
        empty_image[y_offset:y_offset+stitched_images.shape[0], x_offset:x_offset+stitched_images.shape[1]] = stitched_images

        #convert to bgr
        empty_image = cv2.cvtColor(empty_image, cv2.COLOR_RGBA2BGRA)

        # Save the final RGBA image to PNG file using the name from camera_names[0]

        file_path = Path(base_path) / "output" / f"{camera_names[0]}.png"
        file_path = get_unique_file_path(file_path)
        #do not overwrite existing files change name instead
        cv2.imwrite(str(file_path), empty_image)


def get_unique_file_path(file_path):
    """
    Generates a unique file path by appending a random UUID string to the file name.
    
    Args:
    - file_path: The initial path of the file.
    
    Returns:
    - A unique file path with a random UUID appended before the file extension.
    """
    unique_file_path = file_path.with_name(f"{file_path.stem}_{uuid.uuid4().hex}{file_path.suffix}")
    return unique_file_path