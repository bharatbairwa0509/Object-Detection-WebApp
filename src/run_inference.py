import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load the trained YOLO model
model_path = "/content/drive/MyDrive/Object_Detection_WebApp/src/runs/detect/train11/weights/best.pt"
model = YOLO(model_path)

# Define the test image path
img_path = "/content/drive/MyDrive/Object_Detection_WebApp/test_images/sample.jpg"

# Ensure the test image exists
try:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found at {img_path}")
except Exception as e:
    print(e)
    exit()

# Run inference on the image
results = model(img_path, save=True, conf=0.3)  # Set confidence threshold

# Get the saved detection result path
output_dir = "/content/drive/MyDrive/Object_Detection_WebApp/src/runs/detect/"
predict_folders = sorted([f for f in os.listdir(output_dir) if f.startswith("predict")])
if not predict_folders:
    raise FileNotFoundError("❌ No prediction folder found! Run YOLO inference first.")

latest_predict_folder = predict_folders[-1]
detected_img_path = f"{output_dir}{latest_predict_folder}/sample.jpg"

# Load and display the detected image
detected_img = cv2.imread(detected_img_path)
detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(detected_img)
plt.axis("off")
plt.title("YOLO Object Detection Results")
plt.show()

print(f"✅ Detection results saved in: {detected_img_path}")
