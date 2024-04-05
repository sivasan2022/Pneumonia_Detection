# Pneumonia Detection using Neural Network

## About the Project

This project aims to develop a neural network model to detect pneumonia from chest X-ray images.

## Dataset

The dataset consists of chest X-ray images categorized into two classes:

- Normal
- Pneumonia

The dataset is not included in this repository due to its size. You can obtain the dataset from [source link](#) and place it in the `data` directory.

## Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/pneumonia-detection.git
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place the dataset in the `data` directory.
2. Run the `pneumonia_detection.py` script to preprocess the data, train the model, and evaluate the model.

## Model Architecture

The neural network model consists of:

- Input layer: Conv2D with ReLU activation
- MaxPooling layer
- Conv2D with ReLU activation
- MaxPooling layer
- Flatten layer
- Dense layer with ReLU activation
- Dropout layer
- Output layer with Sigmoid activation

## Training

The model is trained using the Adam optimizer and binary cross-entropy loss function. Data augmentation techniques such as shear range, zoom range, and horizontal flip are applied to improve the model's generalization capabilities.

## Evaluation

The model's performance is evaluated using accuracy and loss metrics on the test dataset.


