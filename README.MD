# 3D Scene Segmentation

This repository contains a suite of Python scripts designed for segmenting 3D scenes using Blender's Python API. The scripts automate the process of applying semantic and sub-segmentation to 3D objects in Blender, facilitating the generation of segmented 3D models for various applications such as machine learning and visualization.

## Table of Contents
- [Features](#features)
- [Future Features](#future-features)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Configure Development Environment](#configure-development-environment)
    - [Run the Scripts](#run-the-scripts)
- [Scripts Overview](#scripts-overview)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features
- **Semantic Segmentation**: Assigns unique colors to different objects in a 3D scene based on their type or other attributes.
- **Sub-Segmentation**: Applies detailed segmentation within individual objects based on predefined vertex group prefixes.
- **Color Consistency**: Utilizes a hashing mechanism to ensure consistent color assignment across different sessions.
- **High Color Depth**: Supports 8-bit to 32-bit per channel color depth for precise color differentiation.
- **Automated Material Application**: Automatically applies materials and shaders to segmented parts.

## Future Features
Planned enhancements to extend functionality and improve user experience include:
- **Instance Segmentation**: Implementation of instance segmentation to distinguish individual instances of the same object type within a scene.
- **Automatic Object Detection and Classification**: Integration of AI-based detectors to automatically identify and classify objects in 3D scenes.
- **Support for AOV in Rendering**: Adding Arbitrary Output Variable (AOV) support to enhance rendering capabilities, allowing users to output custom render passes.
- **Automation for Cryptomatte**: Developing automated processes for Cryptomatte in Blender for cases where the current system is limited, enhancing the ability to handle complex masking and object isolation.
- **Refraction Support in Materials**: Adding support for realistic refraction in transparent materials such as glass, enhancing the visual realism of rendered scenes.

## Installation
### Prerequisites
1. **Download and Install Blender Portable**:
    - Download Blender 4.2 Portable from the official Blender website. Extract it to `C:\\blender` on Windows.

2. **Clone the Repository**:
    - Clone this repository to your local machine using Git:
        ```bash
        git clone https://github.com/yourUsername/3d_scene_segmentation.git
        ```
    - Navigate to the directory where you cloned the repo.

3. **Setup Startup Script**:
    - Create a `startup.py` file in `C:\\blender\\4.2\\scripts\\startup`. Inside the `startup.py` file, add the following lines:
        ```python
        import sys
        sys.path.append('C:\\\\path\\\\to\\\\your\\\\cloned\\\\repo')
        ```

### Configure Development Environment
1. **Setting Up Visual Studio Code**:
    - Install Visual Studio Code.
    - Install the Python extension for Visual Studio Code.
    - Install the `fake-bpy-module` for autocomplete features:
        ```bash
        pip install fake-bpy-module-latest
        ```
    - Configure Visual Studio Code to use the Python interpreter that comes with Blender, usually located in your Blender directory.

### Run the Scripts
- Open Blender, and in a Python console or via the scripting tab, you can run the scripts from the repository by importing them or using Blender’s text editor to open and run the `.py` files.

## Scripts Overview
- **constants.py**: Defines constants used across the segmentation processes.
- **logger_config.py**: Configures logging for tracking the script operations.
- **render_settings.py**: Sets up rendering settings optimized for segmentation.
- **utils.py**: Provides utility functions for object and material management.
- **segmentation_test.py**: Main script that orchestrates the loading, processing, and saving of segmented 3D models.
- **seg_materials.py**: Manages creation and application of materials for segmentation.
- **seg_utils.py**: Contains utility functions specifically for segmentation tasks.

## Contributing
Contributions to improve the functionality or efficiency of the segmentation processes are welcome. To contribute:
1. Fork the repository.
2. Create a new branch for each significant change.
3. Commit your changes.
4. Push to your fork and submit a pull request.

## License
This project is open source and available under the MIT License.

## Acknowledgements
Thanks to all contributors who have helped in refining these scripts and the community for the continuous feedback and support.
