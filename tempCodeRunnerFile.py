import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO('D:/FOR RENDO (ONLY)/runs/detect/train15/weights/best.pt')
  # Updated to your actual model path

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Set confidence threshold during prediction
    results = model.predict(source=frame,conf=0.7)  # <-- added conf=0.6 here

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()