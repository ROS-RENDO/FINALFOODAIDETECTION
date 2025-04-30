🍎 Fruit Detection with YOLOv8 and OpenCV
📚 Introduction
This project is about detecting fruits using a trained YOLOv8 model.
The goal is to show real-time detection using a webcam with OpenCV.

🔧 Tools Used
Roboflow — to prepare and export the fruit dataset.

YOLOv8 — to train the object detection model.

OpenCV — to read images, webcam video, and draw detection results.


🔗 Project Flow
Dataset Preparation

Collect and label fruit images using Roboflow.

Export dataset in YOLOv8 format.

Training the Model

Use Ultralytics YOLOv8 to train the model.

Save the trained model weights:
D:\FOR RENDO (ONLY)\runs\detect\train15\weights\best.pt

Webcam Detection

Load the trained model inside a Python script (app.py).

Capture frames from the webcam using OpenCV.

Pass each frame into YOLOv8 to predict fruits.

Draw bounding boxes and labels on the live video feed.

⚡ Simple Concept (How it Detects)
Webcam frame ➔ Model predicts fruits ➔ OpenCV draws boxes ➔ Display live to user

🖼 Example (Visualization)
Before Detection: Webcam frame with no boxes.

After Detection: Boxes and labels (like "Apple", "Banana") are shown in real-time.
