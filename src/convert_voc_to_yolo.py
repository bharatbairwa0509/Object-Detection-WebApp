import os
import xml.etree.ElementTree as ET

# Define dataset paths in Google Drive
BASE_DIR = "/content/drive/MyDrive/Object_Detection_WebApp/data/BCCD"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")
IMAGES_DIR = os.path.join(BASE_DIR, "JPEGImages")
OUTPUT_DIR = os.path.join(BASE_DIR, "labels")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names for BCCD dataset
classes = ["RBC", "WBC", "Platelets"]

# Verify if required directories exist
if not os.path.exists(ANNOTATIONS_DIR):
    raise FileNotFoundError(f"❌ Annotations directory not found: {ANNOTATIONS_DIR}")
if not os.path.exists(IMAGES_DIR):
    raise FileNotFoundError(f"❌ Images directory not found: {IMAGES_DIR}")

def convert_voc_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_filename = root.find("filename").text

    # Ensure image extension
    if not image_filename.endswith(".jpg"):
        image_filename += ".jpg"
    
    image_path = os.path.join(IMAGES_DIR, image_filename)
    if not os.path.exists(image_path):
        print(f"⚠️ Warning: Image {image_filename} not found! Skipping...")
        return

    # Get image size
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            continue

        class_id = classes.index(class_name)

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format (normalized)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        bbox_width = (xmax - xmin) / img_width
        bbox_height = (ymax - ymin) / img_height

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Write to YOLO annotation file
    txt_filename = os.path.splitext(image_filename)[0] + ".txt"
    txt_filepath = os.path.join(OUTPUT_DIR, txt_filename)
    with open(txt_filepath, "w") as f:
        f.write("\n".join(yolo_annotations))

# Process all XML files
num_converted = 0
for xml_file in os.listdir(ANNOTATIONS_DIR):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(ANNOTATIONS_DIR, xml_file))
        num_converted += 1

print(f"✅ VOC to YOLO conversion completed successfully! Converted {num_converted} files.")
