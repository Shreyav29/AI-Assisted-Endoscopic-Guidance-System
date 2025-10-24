import os
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import time
import matplotlib.pyplot as plt
#import torch
from ultralytics import YOLO
import onnx
import torch.onnx
import onnxruntime


# Helper function to extract unique labels
def extract_unique_labels(labels_folder,combine_right_left_flag = False):
    """
    Extract all unique labels from the XML files in the dataset.
    Args:
        labels_folder (str): Path to the folder containing XML label files.
    Returns:
        set: A set of unique labels found in the dataset.
    """
    unique_labels = set()
    for label_file in os.listdir(labels_folder):
        if label_file.endswith(".xml"):
            label_path = os.path.join(labels_folder, label_file)
            try:
                tree = ET.parse(label_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    label = obj.find('name').text

                    #Combining Right and Left Labels 
                    if combine_right_left_flag == True : 
                        if label == "Right Arytenoid" or label == "Left Arytenoid": 
                            label = "Arytenoid"
                        if label == "Right Vocal Cord" or label == "Left Vocal Cord": 
                            label = "Vocal Cord"
                    unique_labels.add(label)
            except Exception as e:
                print(f"Error reading XML file {label_path}: {e}")
    return unique_labels

# Preprocessing Functions
def parse_xml(file_path, combine_right_left_flag = False ):
    tree = ET.parse(file_path)
    root = tree.getroot()
    bboxes, labels = [], []
    for obj in root.findall('object'):
        label = obj.find('name').text

        #Combining Right and Left Labels 
        if combine_right_left_flag == True : 
            if label == "Right Arytenoid" or label == "Left Arytenoid": 
                label = "Arytenoid"
            if label == "Right Vocal Cord" or label == "Left Vocal Cord": 
                label = "Vocal Cords"
        
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
        labels.append(label)
    return bboxes, labels

def preprocess_image(image_path, target_size=(416, 416)):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    top, left = (target_size[0] - nh) // 2, (target_size[1] - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized_image
    return canvas, scale, top, left

def convert_to_yolo_format(class_mapping,bboxes, labels, image_size, scale, top, left):
    yolo_bboxes = []
    for bbox, label in zip(bboxes, labels):
        if label not in class_mapping:
            print(f"Skipping unknown class: {label}")
            #print(label_path)
            continue  # Skip unknown classes
        class_id = class_mapping[label]
        xmin, ymin, xmax, ymax = bbox
        x_center = ((xmin + xmax) / 2 * scale + left) / image_size[1]
        y_center = ((ymin + ymax) / 2 * scale + top) / image_size[0]
        width = ((xmax - xmin) * scale) / image_size[1]
        height = ((ymax - ymin) * scale) / image_size[0]
        yolo_bboxes.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return yolo_bboxes

def save_yolo_data(output_dataset_path, split, image_name, image, yolo_bboxes):
    split_path = os.path.join(output_dataset_path, split)
    image_path = os.path.join(split_path, "images", image_name)
    label_path = os.path.join(split_path, "labels", os.path.splitext(image_name)[0] + ".txt")
    cv2.imwrite(image_path, image)
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_bboxes))



def validate_label(label_path, image_size):
    """
    Validate a YOLO label file to ensure it follows the correct format.
    Args:
        label_path (str): Path to the label file.
        image_size (tuple): Dimensions of the corresponding image (height, width).
    Returns:
        bool: True if the label file is valid, False otherwise.
    """
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False  # Each line must have 5 values: class_id, x_center, y_center, width, height
                
                class_id, x_center, y_center, width, height = map(float, parts)
                if class_id < 0 or x_center < 0 or y_center < 0 or width < 0 or height < 0:
                    return False  # Negative values are invalid
                
                if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                    return False  # Values must be normalized (0 to 1)
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return False

    return True

def validate_dataset(images_folder, labels_folder):
    """
    Validate each image and its corresponding label file in the dataset.
    Args:
        images_folder (str): Path to the images folder.
        labels_folder (str): Path to the labels folder.
    Returns:
        list: List of problematic files (both images and labels).
    """
    problematic_files = []

    # Get all image files
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.png'))]

    for image_name in image_files:
        image_path = os.path.join(images_folder, image_name)
        label_path = os.path.join(labels_folder, os.path.splitext(image_name)[0] + ".txt")

        # Check if the image can be opened
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Invalid or corrupted image file: {image_path}")
                problematic_files.append(image_path)
                continue

            image_size = image.shape[:2]  # (height, width)
        except Exception as e:
            print(f"Error reading image file {image_path}: {e}")
            problematic_files.append(image_path)
            continue

        # Check if the label file exists
        if not os.path.exists(label_path):
            print(f"Missing label file for image: {image_path}")
            problematic_files.append(image_path)
            continue

        # Validate the label file
        if not validate_label(label_path, image_size):
            print(f"Invalid label file: {label_path}")
            problematic_files.append(label_path)

    return problematic_files



###################################Tuning_model.py######################################


# Helper to read YOLO-format labels
def read_yolo_labels(label_path):
    """Read YOLO-format labels from a .txt file."""
    bboxes = []
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid lines
            class_id, x_center, y_center, width, height = map(float, parts)
            bboxes.append((int(class_id), x_center, y_center, width, height))
    return bboxes


# Helper to draw bounding boxes
def draw_bboxes(image, bboxes, class_names, color=(0, 255, 0), label_prefix="GT"):
    """Draw bounding boxes on an image."""
    if not isinstance(class_names, list):
        class_names = list(class_names)  # Ensure class_names is a list

    h, w, _ = image.shape
    for class_id, x_center, y_center, width, height in bboxes:
        xmin = int((x_center - width / 2) * w)
        ymin = int((y_center - height / 2) * h)
        xmax = int((x_center + width / 2) * w)
        ymax = int((y_center + height / 2) * h)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{label_prefix}: {class_names[class_id]}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image



def mobile_optimized_model_conversion(experiment_folder, pt_model_path,dummy_input):
    # Step 1: Load the trained YOLOv8 model
    model = YOLO(pt_model_path)  # Replace with your trained model file
    model.model.eval()  # Set model to evaluation mode

    # Step 2: Define dummy input with 416x416 resolution
    print("dummy_input:", dummy_input)


    # Step 3: Export YOLOv8 to ONNX format
    #onnx_path = pt_model_path.split('.')[0] +'.onnx'
    onnx_path =  os.path.join(experiment_folder, "trained_model.onnx")
    model.export(format="onnx", dynamic=True, opset=12)
    print(f"✅ YOLOv8 model successfully converted to ONNX: {onnx_path}")

    # Step 4: Convert ONNX to TorchScript


    # Load the ONNX model to verify it was correctly exported
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Step 5: Load the ONNX model using ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # Step 6: Define a PyTorch model wrapper to convert ONNX back to TorchScript
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, session):
            super().__init__()
            self.session = session
            self.input_name = session.get_inputs()[0].name

        def forward(self, x):
            ort_inputs = {self.input_name: x.cpu().numpy()}
            ort_outs = self.session.run(None, ort_inputs)
            return torch.tensor(ort_outs[0])  # Convert ONNX output back to PyTorch Tensor

    # Step 7: Initialize the ONNXWrapper and convert it to TorchScript
    onnx_wrapper = ONNXWrapper(ort_session)

    # Step 8: Convert to TorchScript using tracing
    traced_model = torch.jit.trace(onnx_wrapper, dummy_input)

    # Step 9: Save the TorchScript model
    #torchscript_path = pt_model_path.split('.')[0] +'_torchscript.pt'
    torchscript_path = os.path.join(experiment_folder, "trained_model_torchscript.pt")
    traced_model.save(torchscript_path)
    print(f"✅ TorchScript model saved successfully as {torchscript_path}")

    # Step 10: Optimize for mobile deployment
    from torch.utils.mobile_optimizer import optimize_for_mobile

    optimized_model = optimize_for_mobile(traced_model)
    #mobile_optimized_pt_model_path= pt_model_path.split('.')[0] +'_mobile_optimized.pt'
    mobile_optimized_pt_model_path= os.path.join(experiment_folder, "trained_model_mobile_optimized.pt")
    optimized_model.save(mobile_optimized_pt_model_path)
    print("✅ Optimized TorchScript model saved for mobile deployment.")