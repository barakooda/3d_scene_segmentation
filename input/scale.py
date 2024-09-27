import os
import cv2

# Define the folder with textures
base_folder = r"D:\temp\scene_segmentation\input\textures"
max_size = 512

# Function to resize image proportionally using OpenCV
def resize_image(image_path, max_size):
    # Read the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If the image is not successfully loaded, skip it
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Determine the scaling factor
    scaling_factor = min(max_size / width, max_size / height)

    # Resize only if necessary
    if scaling_factor < 1:
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        # Save the resized image, overwrite the original
        cv2.imwrite(image_path, resized_img)
        print(f"Resized: {image_path} to {new_size}")
    else:
        print(f"Skipped resizing for: {image_path} as it's already smaller")

# Recursively find and resize all image files
for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.exr')):
            image_path = os.path.join(root, file)
            resize_image(image_path, max_size)

print("Finished resizing all images.")
