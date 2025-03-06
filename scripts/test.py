import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Function to display an image with matplotlib
def display_image(image, title="Detection Results"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# Function to run inference and visualize results
def test_model(model_path, test_images_path, results_path):
    # Load the YOLO model
    print("Loading the model...")
    model = YOLO(model_path)

    # Create results directory if it doesn't exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Loop through test images
    for image_name in os.listdir(test_images_path):
        image_path = os.path.join(test_images_path, image_name)

        # Ensure the file is an image
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"Processing image: {image_name}")

        # Read the image
        image = cv2.imread(image_path)

        # Run inference on the image
        results = model.predict(source=image, save=False, conf=0.25)

        # Get results and visualize detections
        annotated_image = results[0].plot()  # Annotated image with detections

        # Save the result to the results folder
        save_path = os.path.join(results_path, image_name)
        cv2.imwrite(save_path, annotated_image)

        # Display the result
        display_image(annotated_image, title=f"Results for {image_name}")

if __name__ == "__main__":
    # Paths
    MODEL_PATH = "models/yolo/yolo11n.pt"  # Replace with your model path
    TEST_IMAGES_PATH = "datasets/coco/images/test2017"      # Replace with the path to your test images
    RESULTS_PATH = "results/model_test"        # Replace with the path to save results

    # Run the test
    test_model(MODEL_PATH, TEST_IMAGES_PATH, RESULTS_PATH)
