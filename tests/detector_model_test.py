from src.object_detector_utils import generate_label_from_images


# Main function for processing images and generating labels
if __name__ == "__main__":

    print ( generate_label_from_images(model_accuracy = 'm',max_labels=10000) )

