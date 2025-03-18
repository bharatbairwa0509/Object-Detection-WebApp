import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained model
model_path = "/content/drive/MyDrive/Object_Detection_WebApp/src/runs/detect/train11/weights/best.pt"
model = YOLO(model_path)

def run_model_inference(image_path):
    """
    Runs YOLO model inference on an input image.
    Args:
        image_path (str): Path to the image file.
    Returns:
        result_img (numpy array): Image with bounding boxes.
    """
    results = model(image_path)  # Run inference

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found at {image_path}")

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]} {confidence}"

            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img
