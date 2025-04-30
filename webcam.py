# Import necessary libraries
import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('D:/FOR RENDO (ONLY)/runs/detect/train15/weights/best.pt')
# Make sure the path matches where your best.pt model is saved

# Start capturing video from the default webcam (device 0)
cap = cv2.VideoCapture(0)

# Run an infinite loop to read frames from the webcam
while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break  # Exit loop if frame not read properly

    # Perform object detection on the current frame
    results = model.predict(source=frame, conf=0.7)  # Set confidence threshold to 70%

    # Annotate the frame with detection results (bounding boxes, labels)
    annotated_frame = results[0].plot()

    # Display the annotated frame in a window
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # Wait for 1ms and break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
