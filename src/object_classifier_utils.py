import bpy
import cv2
import math
from mathutils import Vector, Matrix
import numpy as np
import os
import time
from PIL import Image

import nltk
import torch
import open_clip
from ai_models.data.seed_words import seed_words
from src.constants import AI_MODEL,BASE_PATH
from src.utils import hide_all_except_one_globally
from src.utils import get_string_without_trailing_digits
from src.constants import SEMANTIC_NAME
import sys


def delete_camera_and_bouding_box()-> None:
    for obj in bpy.data.objects:
        if obj.name.startswith(("detection_camera")) or obj.name.endswith(("_bbox_temp")):
            bpy.data.objects.remove(obj)


def create_new_world_material_with_sky(scene: bpy.types.Scene)-> bpy.types.World:
    # Create a new world
    new_world = bpy.data.worlds.new("object_detector_world")

    # Enable nodes for the world
    new_world.use_nodes = True

    # Get the node tree and clear existing nodes
    node_tree = new_world.node_tree
    nodes = node_tree.nodes
    nodes.clear()

    # Create a new Sky Texture node
    #sky_node = nodes.new(type="ShaderNodeTexSky")
    #sky_node.location = (-300, 0)  # Position the node for better visibility

    # Create a new Background node
    background_node = nodes.new(type="ShaderNodeBackground")
    background_node.inputs['Strength'].default_value = 0.5  # Set the strength to 0.5
    background_node.location = (0, 0)  # Position the node

    # Create a new World Output node
    world_output_node = nodes.new(type="ShaderNodeOutputWorld")
    world_output_node.location = (300, 0)  # Position the node

    # Connect the Sky Texture node to the Background node
    #node_tree.links.new(sky_node.outputs['Color'], background_node.inputs['Color'])

    # Connect the Background node to the World Output node
    node_tree.links.new(background_node.outputs['Background'], world_output_node.inputs['Surface'])

    # Assign the new world material to the scene
    scene.world = new_world

    # Toggle off transparency in the Film section
    scene.render.film_transparent = True

    return new_world



def create_bbox_and_cameras(obj: bpy.types.Object)-> list[bpy.types.Object, list[str]]:
    
    delete_camera_and_bouding_box()
    
    #apply_rotation_to_vertices(obj)
    
    
    """
    Creates a mesh object that accurately matches the bounding box of the given object
    based on its eight corners transformed into world space, and creates cameras for each face.
    """
    if not obj.bound_box:
        return None

    mat_world = obj.matrix_world
    world_corners = [mat_world @ Vector(corner) for corner in obj.bound_box]
    
    mesh = bpy.data.meshes.new(name=obj.name + "_bbox_temp")
    bbox_obj = bpy.data.objects.new(name=obj.name + "_bbox_temp", object_data=mesh)
    
    collection = obj.users_collection[0] if obj.users_collection else bpy.context.collection
    collection.objects.link(bbox_obj)
    
    # Define vertices and faces ensuring the normals will be calculated outward
    faces = [
        (0, 1, 2, 3),  # Left face
        (7, 6, 5, 4),  # Right face
        (0, 4, 5, 1),  # Front face
        (1, 5, 6, 2),  # Top face
        (2, 6, 7, 3),  # Back face
        (3, 7, 4, 0)   # Bottom face
    ]
    
    mesh.from_pydata(world_corners, [], faces)
    
    mesh.update()
    
    
    

    # Create cameras facing each face
    cameras = []
    for face_idx, face in enumerate(mesh.polygons):
        face_center = sum((world_corners[v] for v in face.vertices), Vector()) / len(face.vertices)
        
        face_normal = face.normal.normalized()
            

        # Calculate tangent
        if face_idx == 0:
            vert_0_world = world_corners[face.vertices[2]]
            vert_1_world = world_corners[face.vertices[1]]
        
        if face_idx == 1:
            vert_0_world = world_corners[face.vertices[2]]
            vert_1_world = world_corners[face.vertices[1]]
        
        if face_idx == 2:
            vert_0_world = world_corners[face.vertices[0]]
            vert_1_world = world_corners[face.vertices[1]]
        
        if face_idx == 3:
            vert_0_world = world_corners[face.vertices[1]]
            vert_1_world = world_corners[face.vertices[0]]
        
        if face_idx == 4:
            vert_0_world = world_corners[face.vertices[1]]
            vert_1_world = world_corners[face.vertices[0]]
        
        if face_idx == 5:
            vert_0_world = world_corners[face.vertices[1]]
            vert_1_world = world_corners[face.vertices[0]]
        
            
        tangent = (vert_1_world - vert_0_world).normalized()
        
        # Calculate bitangent (cross product of normal and tangent)
        bitangent = face_normal.cross(tangent).normalized()
        
        
        rotation_matrix = Matrix((
        tangent,    # X-axis (right direction, tangent)
        bitangent,  # Y-axis (up direction, bitangent)
        face_normal # Z-axis (forward direction, normal)
        )).transposed()  # Transpose to convert from row to column matrix

        
        
        camera_data = bpy.data.cameras.new(name=f"detection_camera_{face_idx}")
        camera_obj = bpy.data.objects.new(name=f"detection_camera_{face_idx}", object_data=camera_data)
        collection.objects.link(camera_obj)
        
        
        camera_obj.matrix_world = rotation_matrix.to_4x4()
        
        # Position camera at the center of the face and back it up along the normal
        max_dimension = max(bbox_obj.dimensions)
        camera_fov = max(camera_obj.data.angle_x,camera_obj.data.angle_y)
        distance =  (max_dimension*0.5) / math.tan(camera_fov*0.5) #- max_dimension * 0.5
        
        camera_obj.location = face_center +  face_normal * distance # Move camera back along the normal
        
        # Setup camera rotation
        #rot_quat = face_normal.to_track_quat('Z', 'Y')
        
        #camera_obj.rotation_euler = rot_quat.to_euler()

        
        cameras.append(camera_obj.name)
        
        #bbox_obj.hide_viewport = True
        bbox_obj.hide_render = True
        
        

    return [bbox_obj, cameras]



def render_from_cameras_to_memory(scene_name: str, camera_names: list) -> dict:
    """
    Renders images from a list of cameras in a specified scene and returns them as NumPy arrays using OpenCV.
    """

    # Ensure the tmp directory exists
    tmp_dir = BASE_PATH / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # Get the scene by name
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' not found.")


    #set render resolution
    scene.render.resolution_x = 224
    scene.render.resolution_y = 224
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 100
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 2048
    scene.cycles.use_denoising = False
    scene.render.use_persistent_data = True

    

    rendered_images = {}

    # Loop over camera names
    for camera_name in camera_names:
        # Get the camera object by its name
        camera = bpy.data.objects.get(camera_name)
        if camera is None or camera.type != 'CAMERA':
            print(f"Camera '{camera_name}' not found or is not a valid camera object.")
            continue

        # Set the camera for the scene
        scene.camera = camera
        bpy.context.view_layer.update()

        # Render the scene
        bpy.ops.render.render()

        # Get the rendered image (Render Result)
        tmp_file = tmp_dir / f"{camera_name}.png"
        bpy.data.images["Render Result"].save_render(filepath=str(tmp_file), scene=scene)

        # Load the image using OpenCV
        image = cv2.imread(str(tmp_file))
        if image is None:
            print(f"Error loading image from file '{tmp_file}'. Skipping.")
            continue

        # Convert BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store the NumPy array in the dictionary
        rendered_images[camera_name] = image_rgb

    return rendered_images

def visualize_images(rendered_images)-> None:
    for camera_name, image in rendered_images.items():
        if image is not None:
            print(f"Camera: {camera_name}, Image shape: {image.shape}, Image dtype: {image.dtype}")

            if image.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
            # Convert the RGBA image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                image_rgb = image  # If the image is already RGB, no conversion is needed

        # Display the image using OpenCV
            cv2.imshow(camera_name, image_rgb)

    cv2.waitKey(0)  # Wait for a key press to close the windows
    cv2.destroyAllWindows()



# Generate only noun labels using WordNet
def generate_noun_labels(word_list, max_labels=10000):
    noun_labels = set()

    for word in word_list:
        synsets = nltk.corpus.wordnet.synsets(word, pos=nltk.corpus.wordnet.NOUN)

        for syn in synsets:
            noun_labels.update(lemma.name() for lemma in syn.lemmas())
            noun_labels.update(lemma.name() for lemma in syn.hypernyms())
            noun_labels.update(lemma.name() for lemma in syn.hyponyms())

        if len(noun_labels) >= max_labels:
            break

    return list(noun_labels)[:max_labels]



# Tokenize all labels ahead of time to avoid recomputation in each batch
def tokenize_labels(labels, device):
    return torch.cat([open_clip.tokenize(label) for label in labels]).to(device)


def save_labels_to_file(labels, file_path):
    with open(file_path, 'w') as file:
        for label in labels:
            file.write(f"{label}\n")



# Process the labels in batches and find the best matching label
def process_in_batches(image_features, text_inputs, labels, model, batch_size=500):
    max_prob = -1
    best_label = None

    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i+batch_size]
        batch_text_inputs = text_inputs[i:i+batch_size]

        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = model.encode_text(batch_text_inputs)

            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            batch_max_prob = probs.max()
            if batch_max_prob > max_prob:
                max_prob = batch_max_prob
                best_label = batch_labels[probs.argmax()]

        del text_features
        torch.cuda.empty_cache()

    return best_label, max_prob


def load_and_preprocess_image(image_path, preprocess, device):
    """Loads and preprocesses a single image."""
    image = preprocess(Image.open(image_path).resize((224, 224))).unsqueeze(0).to(device)
    return image


def encode_image(image, model, device):
    """Encodes a single image using the model."""
    with torch.no_grad(), torch.amp.autocast(device):
        return model.encode_image(image)


def process_image_file(image_file, input_folder_path, preprocess, model, device, text_inputs, labels):
    """Processes a single image file and returns the best label."""
    image_path = os.path.join(input_folder_path, image_file)
    try:
        image = load_and_preprocess_image(image_path, preprocess, device)
        start_time = time.time()

        image_features = encode_image(image, model, device)
        print(f"Image encoded in {time.time() - start_time:.2f} seconds")

        best_label, best_prob = process_in_batches(image_features, text_inputs, labels, model)
        best_label = best_label.split('.')[0]

        print(f"Image: {image_file} - Predicted label: {best_label} with probability {best_prob:.4f}")
        return best_label

    except Exception as e:
        print(f"Error processing image {image_file}: {e}")
        return None


def update_final_labels(final_labels, best_label):
    """Updates the final label counts."""
    if best_label:
        if best_label in final_labels:
            final_labels[best_label] += 1
        else:
            final_labels[best_label] = 1


def get_best_label(final_labels) ->str:
    """Finds the label with the maximum count."""
    if final_labels:
        max_label = max(final_labels, key=final_labels.get)
        if final_labels[max_label] > 1:
            print(f"Max label: {max_label} with count: {final_labels[max_label]}")
            return max_label
        else:
            print("Could not recognize the object and fit it to any label.")
            return "unknown"
    else:
        print("No labels detected.")
        return None

def set_seed(seed,device = None, cuda_deterministic=True):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def generate_label_from_images(model_accuracy : str = 's',max_labels:int=10000) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed=13,device=device)
    model_name = AI_MODEL[model_accuracy][0]
    pretrained_weights = AI_MODEL[model_accuracy][1]

    # Load model
    start_time = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_weights)
    model = model.to(device).half().eval()
    print(f"Model {model_name} loaded in {time.time() - start_time:.2f} seconds")

    # Generate noun labels
    labels = generate_noun_labels(seed_words, max_labels=max_labels)
    
    text_inputs = tokenize_labels(labels, device)

    # Folder path for images
    input_folder_path = BASE_PATH / "tmp"
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize final labels dictionary
    final_labels = {}

    # Process each image
    for image_file in image_files:
        best_label = process_image_file(image_file, input_folder_path, preprocess, model, device, text_inputs, labels)
        update_final_labels(final_labels, best_label)

    # Find and return max label
    return get_best_label(final_labels)


def is_noun(word) -> bool:
    tokens = nltk.word_tokenize(word)
    tagged = nltk.pos_tag(tokens)
    # 'NN', 'NNS', 'NNP', and 'NNPS' are noun tags
    return tagged[0][1] in ('NN', 'NNS', 'NNP', 'NNPS')



def classify_objects(classifier, globally_visible_object_names) -> None:
    
    base_objects_names_set = {}

    for name in globally_visible_object_names:
        
        
        base_name = get_string_without_trailing_digits(name)

        obj = bpy.data.objects.get(name)

        if obj.type == 'MESH' or obj.type == 'CURVE':
            
            #this to check if the object is already classified (trying to avoid reclassification for duplicated objects)
            if base_name in base_objects_names_set:
                obj[SEMANTIC_NAME] = base_objects_names_set.get(base_name)
                
                print(f"The object {obj} is classified as: {label}")
                continue

            hide_all_except_one_globally(globally_visible_object_names,obj.name)
        
            bbox_and_cameras = create_bbox_and_cameras(obj)
        
            rendered_images = render_from_cameras_to_memory(scene_name = bpy.data.scenes[0].name,camera_names = bbox_and_cameras[1])
        
            classifier.classify_images(rendered_images.values())
        
            label = classifier.get_best_label()
        
            obj[SEMANTIC_NAME] = label
            base_objects_names_set[base_name] = label

            print(f"The object {obj} is classified as: {label}")
            delete_camera_and_bouding_box()
            