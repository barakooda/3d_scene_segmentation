import bpy
from src.logger_config import setup_logger

# Set up logging
logger = setup_logger(__name__)

def set_segmentation_render_settings(scene_name: str = "Scene"):
    logger.info("Setting segmentation render settings for scene: %s", scene_name)

    try:
        scene = bpy.data.scenes[scene_name]
    except KeyError:
        logger.error("Scene '%s' not found in bpy.data.scenes", scene_name)
        return

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.preview_samples = 1
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    scene.render.dither_intensity = 0

    set_cameras_for_segmentation()

def set_cameras_for_segmentation():
    logger.info("Setting cameras for segmentation")

    for camera in bpy.data.cameras:
        camera.dof.use_dof = False

if __name__ == "__main__":
    set_segmentation_render_settings()
