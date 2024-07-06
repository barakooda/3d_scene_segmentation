import bpy
from utils import clean_all_material_nodes, create_shader_node
from logger_config import setup_logger

# Set up logging
logger = setup_logger(__name__)

def create_segmentation_material(
    name: str = "", 
    attribute_name: str = "color", 
    attribute_type: str = "OBJECT"  # "GEOMETRY", "OBJECT", "INSTANCER"
) -> bpy.types.Material:
    """
    Create a segmentation material with the given name, attribute name, and attribute type.
    
    :param name: Name of the material.
    :param attribute_name: Name of the attribute to use.
    :param attribute_type: Type of the attribute (e.g., "OBJECT", "GEOMETRY", "INSTANCER").
    :return: The created material.
    """
    logger.info("Creating segmentation material: %s", name)
    
    material = bpy.data.materials.new("MAT_" + name)
    material.use_nodes = True
    clean_all_material_nodes(material.node_tree.nodes)

    face_corner_attribute_node = create_shader_node(
        material=material,
        node_name="face_corner_attribute",
        node_type="ShaderNodeAttribute",
        node_location=(550, 450),
        node_size=(280, 140),
    )
    face_corner_attribute_node.attribute_type = attribute_type
    face_corner_attribute_node.attribute_name = attribute_name

    emission_node = material.node_tree.nodes.new(type="ShaderNodeEmission")
    emission_node.location = (950, 500)

    shader_mix_node = material.node_tree.nodes.new(type="ShaderNodeMixShader")
    shader_mix_node.location = (1500, 550)

    refraction_bsdf_node = material.node_tree.nodes.new(type="ShaderNodeBsdfRefraction")
    refraction_bsdf_node.location = (950, 250)

    refraction_attribute_node = material.node_tree.nodes.new(type="ShaderNodeAttribute")
    refraction_attribute_node.attribute_name = "ior"
    refraction_attribute_node.name = "ior_attribute"
    refraction_attribute_node.label = "ior_attribute"
    refraction_attribute_node.location = (430, 148)

    is_transparent_node = material.node_tree.nodes.new(type="ShaderNodeAttribute")
    is_transparent_node.attribute_name = "is_transparent"
    is_transparent_node.name = "is_transparent_attribute"
    is_transparent_node.label = "is_transparent_attribute"
    is_transparent_node.location = (450, 700)

    output_node = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (1800, 600)

    # Link nodes
    links = material.node_tree.links
    links.new(face_corner_attribute_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], shader_mix_node.inputs[1])
    links.new(refraction_attribute_node.outputs["Fac"], refraction_bsdf_node.inputs["IOR"])
    links.new(refraction_bsdf_node.outputs["BSDF"], shader_mix_node.inputs[2])
    links.new(is_transparent_node.outputs["Fac"], shader_mix_node.inputs["Fac"])
    links.new(shader_mix_node.outputs["Shader"], output_node.inputs["Surface"])

    return material

def create_hair_particle_system_material(
    particle_system: bpy.types.ParticleSystem, 
    color: list
) -> bpy.types.Material:
    """
    Create a hair particle system material for the given particle system and color.
    
    :param particle_system: The particle system to create the material for.
    :param color: The color to use for the emission node.
    :return: The created material.
    """
    if not particle_system:
        logger.error("Particle system is None")
        return None

    logger.info("Creating hair particle system material for: %s", particle_system.name)
    
    material = bpy.data.materials.new("MAT_" + particle_system.name)
    material.use_nodes = True
    clean_all_material_nodes(material.node_tree.nodes)

    emission_node = material.node_tree.nodes.new(type="ShaderNodeEmission")
    emission_node.location = (950, 500)
    emission_node.inputs[0].default_value = color

    output_node = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (1800, 600)

    # Link nodes
    links = material.node_tree.links
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    return material
