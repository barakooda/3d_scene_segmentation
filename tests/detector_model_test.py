import os
from src.ai_image_classifier import AIImageClassifier
from src.constants import BASE_PATH
from object_classifier_utils import generate_label_from_images
import cv2


# Main function for processing images and generating labels
if __name__ == "__main__":
    
    # Create an instance of the classifier
    classifier = AIImageClassifier(model_accuracy='s', max_labels=10000, seed=13)

    # List of OpenCV images (NumPy arrays)
    image_arrays = []

    input_folder_path = BASE_PATH / "tmp"
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        image_path = os.path.join(input_folder_path, filename)
        image = cv2.imread(str(image_path))
        if image is not None:
            image_arrays.append(image)
        else:
            print(f"Warning: Failed to read image '{filename}'. Skipping.")
    

    # Classify the images
    label = classifier.classify_images(image_arrays)
    print(f"The images are classified as: {label}")


