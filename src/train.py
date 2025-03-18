from ultralytics import YOLO
import torch

# Load the trained model from the correct folder
model = YOLO("/content/drive/MyDrive/Object_Detection_WebApp/src/runs/detect/train11/weights/best.pt")

# Define save path
save_path = "/content/drive/MyDrive/Object_Detection_WebApp/models/best_model.pt"

# Save the PyTorch model
torch.save(model.model.state_dict(), save_path)

print(f"✅ Model saved successfully at {save_path}")
