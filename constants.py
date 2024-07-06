# constants.py

"""
Constants used for the 3D scene and segmentation settings.
"""

# Object Types
MESH: str = "MESH"
CURVE: str = "CURVE"
CURVES: str = "CURVES"
GEOMETRY: str = "GEOMETRY"
META: str = "META"


# Segmentation Types
SEMANTIC_NAME: str = "semantic_name"
OBJECT_SEGMENTATION: str = "object_segmentation"
SUB_SEGMENTATION: str = "sub_segmentation"
UNDEFINED_OBJECT: str = "undefined_object"

# Attribute Names
ATTRIBUTE_NAME: str = "attribute_name"
SUB_SEGMENTATION_ATTRIBUTE_NAME: str = "sub_segmentation"
SUB_SEGMENTATION_PREFIX: list = ["seg", "sub", "DEF"]

# Maximum Values
MAX_UINT16: int = 2**16 - 1

# Bounding Box and Center
INDICES: str = "indices"
BB_MIN: str = "bb_min"
BB_MAX: str = "bb_max"
CENTER: str = "center"