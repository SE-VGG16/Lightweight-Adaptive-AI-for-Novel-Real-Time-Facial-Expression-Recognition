# Lightweight-Adaptive-AI-for-Novel-Real-Time-Facial-Expression-Recognition
Lightweight Adaptive AI for Novel Real-Time Facial Expression Recognition
# Adaptive Multi-Scale Fusion Transformer (AMFT) for Facial Expression Recognition

This repository contains the PyTorch implementation of the **Adaptive Multi-Scale Fusion Transformer (AMFT)** for facial expression recognition (FER) as described in the paper:

> **Lightweight Adaptive AI for Novel Real-Time Facial Expression Recognition**  
> Zarnigor Tagmatova, Alpamis Kutlimuratov, Sabina Umirzakova, Sanjar Mirzakhalilov, Komil Tashev, Azizjon Meliboev, Akmalbek Abdusalomov, and Young Im Cho

AMFT integrates several state-of-the-art techniques to achieve high accuracy and efficiency:
- **Multi-scale Feature Extraction:** Captures fine-grained details (edges, textures) as well as global facial structures using a modified ResNet-50 backbone.
- **Adaptive Fusion Module:** Employs both spatial and channel attention to dynamically emphasize the most relevant facial features.
- **Transformer Encoder:** Enhances feature representation with global self-attention to capture long-range dependencies.
- **Optional LSTM Module:** Allows temporal modeling for video-based FER by capturing the dynamics of facial expressions over time.

## Repository Structure

- **model.py:**  
  Contains the implementation of the AMFT model including the ResNet-50 backbone, adaptive fusion (spatial and channel attention), transformer encoder, and optional LSTM for temporal processing.

- **preprocessing.py:**  
  Implements a dataset class for facial expression recognition. It handles image loading, resizing, normalization, and augmentation based on CSV annotations.

- **train.py:**  
  Provides the training loop. This script loads the training dataset, initializes the model and optimizer, and saves model checkpoints after each epoch.

- **test.py:**  
  Loads a saved checkpoint and evaluates the model on the test dataset by calculating loss and overall accuracy.

- **evaluation.py:**  
  Computes additional evaluation metrics such as confusion matrix and classification report. It also visualizes the confusion matrix using matplotlib and seaborn.

## Features

- **ResNet-50 Backbone:** Uses a pre-trained ResNet-50 for robust initial feature extraction.
- **Adaptive Attention:** Integrates spatial and channel attention modules to focus on critical facial regions.
- **Transformer-based Enhancement:** Incorporates a transformer encoder to capture global dependencies.
- **Temporal Dynamics (Optional):** Supports LSTM modules for modeling sequences in video-based FER.
- **Modular Design:** Easy-to-read, modular code structure for straightforward customization and extension.

## Requirements

- Python 3.6+
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [Pillow](https://python-pillow.org/)

Install the required packages using pip:

```bash
pip install torch torchvision pandas scikit-learn matplotlib seaborn pillow
