# Import necessary libraries
from roboflow import Roboflow
from ultralytics import YOLO

# Ensure this script runs only when executed directly (not when imported)
if __name__ == "__main__":

    # Initialize Roboflow with your API key
    rf = Roboflow(api_key="K1epz6Y0VhIcXlo3HLLj")

    # Access the specific project and version in your Roboflow workspace
    project = rf.workspace("fruit-detection-byxkd").project("freshness-fruits-and-vegetables-nojcz")
    version = project.version(1)

    # Download the dataset in YOLOv8 format
    dataset = version.download("yolov8")

    # Load a pre-trained YOLOv8 nano model (yolov8n.pt)
    # Other options: yolov8s.pt (small), yolov8m.pt (medium), etc.
    model = YOLO("yolov8n.pt")  # Automatically downloads if not available locally

    # Train the YOLO model using the downloaded dataset
    model.train(
        data=dataset.location + "/data.yaml",  # Path to data configuration file
        epochs=30,                              # Number of training epochs
        imgsz=640,                              # Input image size for training
        device="cuda",                          # Use GPU ("cuda") or CPU ("cpu")
    )

    # After training, the best weights are saved at 'runs/train/exp/weights/best.pt'
    print("Training complete!")  # Notify that training has finished
