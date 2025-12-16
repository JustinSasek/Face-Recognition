# Face Recognition

## Summary

A face recognition system built with PyTorch that detects whether a specific person's face is present in a video frame. The project captures a personal image dataset using webcam input, trains a convolutional neural network (CNN) to distinguish between target and non-target faces, and provides real-time inference capabilities.

## Repository Overview

### Core Modules (`face-classification/`)

- **`createDataset.py`** – Dataset creation utility that captures images from webcam with customizable hotkey controls. Generates labeled images and metadata files for model training.

- **`faceDataloader.py`** – PyTorch dataset and dataloader implementation. Handles image preprocessing, normalization, and automatic train/validation/test splitting (65%/20%/15%).

- **`model.py`** – CNN architecture with 7 convolutional layers, pooling stages, and sigmoid activation for binary classification. Input size: 224×224 RGB images.

- **`main.py`** – Training loop framework (in development). Loads data and initializes the model for training with L1 loss.

- **`demo.py`** – Real-time inference script. Loads a trained model and performs live face detection on webcam feed.

- **`create.py`** – Entry point to launch the interactive dataset creation process.

### Root Level

- **`mlps_mnist.py`** – Standalone MNIST digit classification example using a multi-layer perceptron (unrelated to main project).

## Setup

### Prerequisites

- Python 3.7+
- Webcam access

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Face-Recognition
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision opencv-python keyboard pandas numpy
   ```

## Instructions

### 1. Create Your Dataset

Run the dataset creation tool to capture your face and background images:

```bash
cd face-classification
python create.py
```

The script will:
- Prompt for an image label (identity ID) – enter `1` for your face, `0` for background
- Display a countdown and begin capturing frames at a configurable rate
- Press **spacebar** to stop capturing for the current label
- Enter an empty label to finish the collection process

Images are saved to `eclair-faces/img/` with metadata in `eclair-faces/id.txt` and `eclair-faces/size.txt`.

### 2. Train the Model

Prepare and launch training (training loop is partially implemented):

```bash
python main.py
```

This loads your dataset and trains the CNN to classify whether the target face is present. The model automatically splits your data for training, validation, and testing.

### 3. Run Live Inference

After training, use the demo to perform real-time face detection:

```bash
python demo.py
```

The script displays `True` or `False` for each frame, indicating whether the target face is detected.

### Notes

- Ensure the correct webcam is selected (modify `WEBCAM_NUM` in relevant files if using multiple cameras)
- The model expects 224×224 RGB images (automatic preprocessing applied)
- Save your trained model as `model.pt` before running inference
