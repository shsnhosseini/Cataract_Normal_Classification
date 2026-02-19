## Cataract Classification

A deep learning project for binary classification of cataract and normal eye images using Convolutional Neural Networks (CNN) and Transfer Learning with Xception.

### Project Overview

This project implements two approaches to classify eye images as either **cataract-affected** or **normal**:

1. **Custom CNN Model**: A custom-built convolutional neural network trained from scratch
2. **Transfer Learning Model**: Utilizing a pre-trained Xception model fine-tuned for cataract classification

The notebook demonstrates the complete pipeline from data preprocessing to model training, evaluation, and inference.

### Quick Start

To use the **pre-trained model** for predictions:

1. Run the section **"Downloading and loading the saved model"** - loads the trained model architecture and weights
2. Run the section **"Prediction of images in folder test_images"** - performs predictions on test images

### Dataset

The project uses a binary classification dataset consisting of:
- **Cataract samples**: Images of eyes affected by cataracts
- **Normal samples**: Images of healthy eyes

The data is automatically downloaded and processed through the notebook. The dataset is split into:
- **Training set**: ~80% of data
- **Validation set**: ~10% of data  
- **Test set**: ~10% of data (294 samples) used for estimating generalization performance

#### Data Preprocessing

- Image size standardized to **250×250 pixels**
- **Batch size**: 32 samples per batch
- **Data augmentation** applied to training data:
  - Horizontal and vertical flips
  - Random rotations (±10%)
  - Random zoom (±10%)
- **Normalization**: Pixel values rescaled to [0, 1]

### Models

#### Model 1: Custom CNN

A custom convolutional neural network built with:
- Input normalization layer
- Multiple Conv2D blocks with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Global Average Pooling
- Dense layers with Binary Cross-Entropy loss
- **Optimizer**: Nadam
- **Activation**: Sigmoid (binary classification)

#### Model 2: Transfer Learning (Xception)

Uses a pre-trained Xception model from ImageNet with:
- **Base model**: Xception (pre-trained on ImageNet)
- **Frozen weights** during initial training to leverage learned features
- **Custom top layers**: Added dense layers for binary classification
- **Input preprocessing**: Rescaling to [-1, 1] range (Xception requirement)
- **Fine-tuning capability**: Base model can be unfrozen for advanced training

### Files

| File | Description |
|------|-------------|
| `cataract_classification.ipynb` | Main Jupyter notebook with complete pipeline |
| `model.json` | CNN model architecture in JSON format |
| `weights.best.hdf5` | Trained weights for the CNN model |
| `tl_model.json` | Transfer Learning model architecture in JSON format |
| `tl_weights.best.hdf5` | Trained weights for the Transfer Learning model (This file has been removed from the repository due to its large size) |
| `result.csv` | Predictions output file with image names and predicted labels |
| `test_images/` | Folder containing test images for prediction |

### Notebook Sections

1. **Setup** - Install required packages and import libraries
2. **Preprocessing** - Download and organize dataset, split into train/val/test
3. **Data Loading & Augmentation** - Load images using `image_dataset_from_directory`, apply augmentation
4. **Model: CNN** - Build, compile, and visualize custom CNN model
5. **Training** - Train the CNN model with checkpointing and early stopping
6. **Downloading and Loading Saved Model** - Load pre-trained CNN model
7. **Estimating Generalization Performance** - Evaluate on test set
8. **Prediction of Images** - Make predictions on new test images
9. **Transfer Learning** - Build and train Xception-based model
10. **Model Saving** - Save model architecture and weights

### Requirements

```
tensorflow >= 2.0
keras
opencv-python
numpy
pandas
matplotlib
scikit-learn
seaborn
scipy
split-folders
gdown
visualkeras
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Additional notebook-specific packages:
```bash
!pip install split-folders[full]
!pip install -U gdown
!pip install visualkeras
```

### Training Details

#### CNN Model Training
- **Epochs**: 700 (with early stopping, patience=100)
- **Batch size**: 32
- **Loss function**: Binary cross-entropy
- **Metrics**: Accuracy
- **Callbacks**: 
  - ModelCheckpoint: Saves best weights based on validation accuracy
  - EarlyStopping: Stops training if no improvement for 100 epochs

#### Transfer Learning Model Training
- **Epochs**: 300 (with early stopping)
- **Frozen base model**: Initial training with frozen Xception weights
- **Same callbacks and hyperparameters** as CNN model

### Model Evaluation

The model is evaluated on the test set (10% of data) which provides an unbiased estimate of generalization performance. Output includes:
- Test loss
- Test accuracy

### Prediction Output

Predictions are saved to `result.csv` with:
- **name**: Image filename
- **label**: Predicted class ('cataract' or 'normal')

Predictions use a threshold of 0.5 on the sigmoid output for binary classification.

### Usage Example

```python
# Load the trained model
from tensorflow.keras.models import model_from_json

# Load architecture
with open('model.json', 'r') as json_file:
    loaded_model = model_from_json(json_file.read())

# Load weights
loaded_model.load_weights('weights.best.hdf5')

# Compile
loaded_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

# Make predictions
predictions = loaded_model.predict(test_images)
```

### Key Features

✅ Data augmentation for improved generalization  
✅ Model checkpointing to save best weights  
✅ Early stopping to prevent overfitting  
✅ Visualization of training history (loss and accuracy curves)  
✅ Two different approaches: Custom CNN and Transfer Learning  
✅ Batch processing for efficient inference  
✅ Automated prediction pipeline with CSV output  

### Hardware Requirements

- GPU recommended (NVIDIA GPU with CUDA support)
- Minimum 4GB RAM
- The notebook checks for GPU availability and can run on CPU (slower)


### Notes

- The notebook is designed to run on Google Colab with Google Drive integration for data storage
- Paths in the notebook reference `/content/drive/MyDrive/` which are Google Colab paths
- For local execution, update dataset paths accordingly
- The pre-trained models (`weights.best.hdf5` and `tl_weights.best.hdf5`) contain the best checkpoints from training
