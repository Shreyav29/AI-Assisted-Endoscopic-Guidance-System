# 🧠 AI-Assisted Endoscopic Guidance System

This project was developed in collaboration with a clinical institution to build an **AI-powered endoscopic guidance system** that can automatically **detect, label, and orient anatomical structures** in real time during endoscopic procedures.  
The goal was to assist clinicians with **real-time visual feedback**—labeling key throat structures and identifying the **camera’s directional orientation** (“True North”) to improve procedural accuracy and efficiency.

---

## 🚀 Overview

The system combines two complementary AI components:

1. **Endoscopic Object Detection Model** – a YOLOv8-based detection model to identify and label anatomical structures.
2. **True North Classification Model** – a CNN-based classifier that determines the directional orientation of the endoscope.

These models together provide **spatial awareness** and **semantic understanding** of the endoscopic view, enabling clinicians to navigate efficiently during live procedures.

---

## 🧩 Problem Statement

Manual labeling of anatomical structures during live endoscopy is difficult, inconsistent, and heavily dependent on clinician experience.  
We aimed to design a real-time model capable of:
- Detecting and naming relevant throat and airway landmarks (epiglottis, glottis, arytenoids, vocal cords, trachea, etc.)
- Estimating the **directional alignment** of the endoscope (north/south/east/west/center)
- Operating efficiently on **mobile hardware** for real-time inference without cloud dependency

---

## 📊 Data Preparation

- **Dataset:** Thousands of endoscopic images collected under clinical supervision  
- **Annotation Format:** Expert-labeled XML annotations converted to YOLO format  
- **Preprocessing Pipeline:**
  - Parsed XML annotations  
  - Resized and letterboxed images to **416×416**  
  - Merged left/right anatomical labels for better accuracy  
  - Validated annotation integrity  




#AI-Assisted Endoscopic Guidance System

##Project Overview
Our team partnered with a clinical institution to build a computer-vision system capable of
automatically identifying and labeling anatomical structures during endoscopic
procedures. The system’s goal was to assist clinicians by displaying real-time, labeled
bounding boxes around key throat and airway structures while also determining the
directional orientation of the endoscope.
Two complementary AI components were developed:
1. Endoscopic Object Detection Model – a YOLOv8-based detection
system(classifier) for identifying anatomical structures.
2. True North Classification Model – a lightweight classifier that determines the
relative viewing orientation (“True North”) of the camera feed.

##Problem Statement
Manual labeling of internal structures during live endoscopic procedures is difficult,
inconsistent, and depends heavily on clinician experience.
We aimed to design a model capable of:
• Detecting and naming relevant anatomical landmarks such as the epiglottis, glottis,
arytenoids, vocal cords, and trachea in real-time.
• Understanding the directional alignment of the endoscope (north/south/east/west
center) to guide scope positioning.

##Data Preparation and Annotation
• Thousands of endoscopic images were collected internally under clinical
supervision.
• Images were annotated using XML format through expert manual labeling.
• Labels included both left/right variants of structures (e.g., Left Arytenoid / Right
Arytenoid), which were merged during preprocessing for better accuracy.
• A custom data preprocessing pipeline was built to handle:

o Parsing XML annotations
o Letterboxing and resizing to 416×416
o Conversion to YOLO format (class_id x_center y_center width height)
o Validation of label integrity and consistency

##Model Development
Endoscopic YOLO Detection Model
• Architecture: YOLOv8s (small variant, CSPDarknet-inspired, anchor-free)
• Framework: PyTorch (Ultralytics YOLOv8 library)
• Input: 416×416 RGB frames from endoscopic video
• Training: Fine-tuned on proprietary dataset for 100 epochs (Used GPU)
• Evaluation Metrics: Precision, Recall, F1-Score, and mean Average Precision
(mAP@0.5)
True North Classification Model
• Architecture: CNN classifier trained to predict one of 9 spatial zones (north, south,
east, west, and center combinations)
• Input: Single frame divided into 9 regions; class represents the quadrant containing
the known north point.
• Accuracy: ~92% on out-of-sample holdout set

##Results and Performance
Endoscopic YOLO Detection Model
• Achieved mean Average Precision (mAP@0.5) = 0.78 on holdout data.
• Demonstrated >90% precision across most anatomical structures.
• Primary confusions occur between left and right vocal cords or arytenoids, which
are visually symmetric.

• The model’s precision-recall trade-off curve indicated that at 80% recall, the
system maintained ~85% precision, suggesting balanced sensitivity and reliability
for clinical use.
True North Classifier
• Overall Accuracy: 92%
• High recall for dominant orientations (north-center, south-center)
• Lower performance on rare corner orientations (e.g., north-east, south-west) due to
class imbalance. We plan on retraining with larger dataset in the next phase of
training.

Deployment and Optimization
After training, the YOLOv8 model was exported and optimized for on-device deployment:
1. Exported to ONNX format using the Ultralytics API
2. Validated with ONNX Runtime for correctness
3. Converted to TorchScript and optimized with PyTorch’s optimize_for_mobile utility
4. Packaged for mobile devices to enable real-time inference directly on endoscopic
camera systems without cloud latency
The resulting system runs locally on mobile hardware, labeling structures at near real-time
frame rates with minimal compute overhead.

Conclusion
The final integrated system successfully
• Identifies key throat structures,
• Labels them in real-time, and
• Assists clinicians with camera orientation feedback.
The combination of YOLOv8 object detection and the True North classifier provides both

spatial awareness and semantic understanding, demonstrating a foundation for AI-
assisted endoscopy.
