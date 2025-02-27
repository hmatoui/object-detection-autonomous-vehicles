# Object Detection for Autonomous Vehicles

This project develops an object detection system optimized for autonomous vehicles, using the YOLO architecture for real-time detection. It is trained and tested on the COCO Dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)

## Overview

Object detection is critical for the safe operation of autonomous vehicles. This system uses YOLOv8 to detect objects in real-time from video streams, focusing on optimizing accuracy and speed for autonomous driving applications.

## Dataset

We use the [COCO Dataset](https://cocodataset.org/) for training and validation. Ensure the download the dataset and place it in the `data/coco/` folder.

## Installation

1. Clone the repository:
   
   ```bash
   git clone https://github.com/hmatoui-username/object-detection-autonomous-vehicles.git
   cd object-detection-autonomous-vehicles
   ```

2. Install dependencies
   
   ```bash
   pip install -r requirements.txt
   ```

3. Set Up YOLOv8
   
   1. **Install Ultralytics Library**:
      YOLOv8 is available in the `ultralytics` package. Install it via pip:
      
      ```bash
      pip install ultralytics
      ```
   
   2. **Import the YOLOv8 Module**:
      In the Python scripts, use `from ultralytics import YOLO` to access YOLOv8's functionalities.
   
   3. **Organize the Dataset**:
      
      The dataset should be in the YOLO format:
      
      ```bash
      data/coco/
      ├── train/images       # Training images
      ├── train/labels       # Corresponding YOLO labels
      ├── val/images         # Validation images
      ├── val/labels         # Corresponding YOLO labels
      
      ```
   
   4. **Prepare Dataset Configuration File:**
      Create a dataset configuration file (`data/coco.yaml`): 
      
      ```bash
      path: ../data/coco  # Dataset root directory
      train: train/images  # Training images directory
      val: val/images      # Validation images directory
      
      nc: 80  # Number of classes
      names: ../data/coco.names  # Class names file
      
      ```

## Usage

### Training

Train YOLO on the COCO dataset:

```bash
python scripts/train.py --data data/coco/ --epochs 50
```

### Detection

Run real-time object detection on video streams:

```bash
python scripts/detect.py --source 0  # Webcam as input
```

### Deployment

Run the app locally:

```bash
cd app
python app.py
```

Access the app at `http://localhost:5000`


