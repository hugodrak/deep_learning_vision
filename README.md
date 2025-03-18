
# Deep-Learning-Vision-Pipeline

## Overview
An end-to-end image classification pipeline using PyTorch. Includes data loading, model architectures (ResNet, custom CNN), training loops, and evaluation scripts.

## Features
- **Data Preprocessing**: Automated data augmentation and normalization.
- **Multiple Models**: ResNet variant and a custom CNN.
- **Evaluation Metrics**: Accuracy, F1-score, confusion matrix.

## Setup & Installation
```bash
conda create -n vision-env python=3.8
conda activate vision-env
pip install -r requirements.txt
```

## Usage
1. Place your training/validation images in `data/train` and `data/val`.
2. Run `python src/train.py` to begin training.
3. Evaluate using `python src/evaluate.py`.

## Future Enhancements
- Transfer learning with larger datasets.
- Deployment with TorchServe or Flask-based API.
