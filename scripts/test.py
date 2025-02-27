import cv2
from ultralytics import YOLO

def detect_realtime(source=0, weights="yolov8s.pt"):
    # Load the YOLOv8 model
    model = YOLO(weights)

    # Open video capture (webcam or video file)
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model.predict(source=frame, imgsz=640, conf=0.5)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_realtime(source=0)  # Use webcam as input
