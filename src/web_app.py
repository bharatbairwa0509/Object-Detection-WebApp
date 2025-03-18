import gradio as gr
import cv2
import numpy as np
from model_inference import run_model_inference

def process_image(image):
    """Process uploaded image, run inference, and return result."""
    image_path = "temp.jpg"
    cv2.imwrite(image_path, image)
    result_img = run_model_inference(image_path)
    return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# Create Gradio Interface
gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs="image",
    title="🔍 Object Detection Web App",
    description="Upload an image and detect objects using YOLO.",
).launch(share=True)
