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


# Function to run inference on a video and visualize results
def test_model_on_videos(model_path, videos_path, results_path):
    # Load the YOLO model
    print("Loading the model...")
    model = YOLO(model_path)

    # Create results directory if it doesn't exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Loop through all video files in the specified directory
    for video_name in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video_name)

        # Ensure the file is a video
        if not video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        print(f"Processing video: {video_name}")

        # Open the video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video {video_name}")
            continue

        # Get video properties
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Set up output path for annotated video
        output_video_path = os.path.join(results_path, f"annotated_{video_name}")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Loop through video frames
        while True:
            ret, frame = video.read()
            if not ret:
                break  # Exit the loop if there are no frames left to read

            # Run inference on the frame
            results = model.predict(source=frame, save=False, conf=0.25)

            # Get results and visualize detections
            annotated_frame = results[0].plot()  # Annotated frame with detections

            # Write the frame to the output video
            out.write(annotated_frame)

            # Optionally, display the frame in real-time (press 'q' to stop)
            cv2.imshow(f"Video Inference - {video_name}", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources for the current video
        video.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Annotated video saved to: {output_video_path}")

    print("All videos processed.")

if __name__ == "__main__":
    # Paths
    MODEL_PATH = "models/yolo/yolo11n.pt"  # Replace with your model path
    TEST_IMAGES_PATH = "datasets/coco/images/test2017"      # Replace with the path to your test images
    TEST_VIDEOS_PATH = "datasets/automated-driving/videos/test"      # Replace with the path to your test images
    RESULTS_PATH = "results/model_test"        # Replace with the path to save results

    # Run the test
    #test_model(MODEL_PATH, TEST_IMAGES_PATH, RESULTS_PATH)
    test_model_on_videos(MODEL_PATH, TEST_VIDEOS_PATH, RESULTS_PATH)
