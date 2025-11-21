# Food-11 Image Classification with ResNet

This project implements a deep learning solution for classifying food images into 11 categories using the Food-11 dataset. It features a custom **LightResNet-18** architecture designed for efficiency and high accuracy, along with a standard VGG16 implementation for comparison.

## Project Overview

*   **Task**: Multi-class image classification (11 classes).
*   **Model**: Custom LightResNet-18 (Residual Network with reduced channel depth).
*   **Accuracy**: Achieves ~80% accuracy on the validation set.
*   **Framework**: PyTorch.

## Requirements

*   Python 3.8+
*   PyTorch
*   Torchvision
*   NumPy
*   Pandas
*   OpenCV (opencv-python)
*   Matplotlib
*   Seaborn
*   Scikit-learn
*   TensorBoard

Install dependencies via pip:
```bash
pip install torch torchvision numpy pandas opencv-python matplotlib seaborn scikit-learn tensorboard
```

## File Structure

Recommended file structure for the project:

```
project_root/
├── dataset/
│   ├── training/       # Training images (Label_Id.jpg)
│   ├── validation/     # Validation images
│   └── evaluation/     # Test images
├── runs/               # TensorBoard logs
├── main.py             # Main training and prediction script
├── Lab2_Guide.ipynb    # Jupyter Notebook with step-by-step guide
├── README.md           # Project documentation
├── requirements.txt    # List of dependencies
├── custom_resnet_best.pth  # Saved model weights (generated after training)
├── ans_ours.csv        # Prediction results for Custom ResNet
└── ans_vgg.csv         # Prediction results for VGG16
```

## Key Features

1.  **Robust Data Loading**: Handles image paths with Chinese characters using `cv2.imdecode`.
2.  **Data Augmentation**: Implements a rich set of augmentations (Flip, Rotation, ColorJitter, Crop, Affine) to improve generalization.
3.  **LightResNet Architecture**: A custom ResNet-18 variant with halved channel counts (start with 32) to reduce parameter count and memory usage while maintaining residual learning benefits.
4.  **Regularization**: Uses Dropout (p=0.5) and Weight Decay to prevent overfitting.
5.  **Advanced Training**: Utilizes `CosineAnnealingLR` for smooth learning rate decay.

## Usage

1.  **Configure Paths**: Open `main.py` and update the `TRAIN_DIR`, `VAL_DIR`, and `TEST_DIR` variables to point to your dataset location.
2.  **Run Training**:
    ```bash
    python main.py
    ```
    This will:
    *   Train the Custom ResNet model for 50 epochs.
    *   Save the best model weights to `custom_resnet_best.pth`.
    *   Generate predictions for the evaluation set in `ans_ours.csv`.

3.  **Monitor Training**:
    ```bash
    tensorboard --logdir=runs
    ```

## Results

*   **Custom ResNet**: ~80% Validation Accuracy.
*   **VGG16 (Transfer Learning)**: ~85%+ Validation Accuracy (optional).