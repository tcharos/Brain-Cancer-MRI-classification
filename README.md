# Brain Cancer MRI Classification

A deep learning project for classifying brain cancer types from MRI images using various CNN architectures including custom models and pre-trained networks (VGG16, ResNet-50, InceptionV3).

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements multiple deep learning architectures to classify brain MRI images into three categories:
- **Glioma** (brain_glioma)
- **Meningioma** (brain_menin)
- **Pituitary Tumor** (brain_tumor)

The project explores both custom CNN architectures and transfer learning approaches with data augmentation to achieve optimal classification performance.

## 📊 Dataset

**Source**: [Brain Cancer - MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset consists of MRI images organized into three classes:
- Images are preprocessed to 224x224 pixels (299x299 for InceptionV3)
- RGB color format
- Split into Training (70%), Validation (15%), and Testing (15%) sets
- Random seed: 123 for reproducibility

### Data Preprocessing

- Automatic train/validation/test splitting
- Image resizing and normalization
- Support for both Google Colab and local environments
- Optional Google Drive integration

### Download the Dataset

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Download the dataset (you'll need a Kaggle account)
3. Extract the files to your project directory
4. The dataset should be organized with folders for each tumor type

## 🧠 Models

### Custom CNN Architectures

The project includes multiple custom CNN designs with varying depths and complexity:
- **CNN3**: 3 convolutional blocks
- **CNN4**: 4 convolutional blocks
- **CNN5**: 5 convolutional blocks
- **CNN6**: 6 convolutional blocks (best performing custom model)

### Transfer Learning Models

Pre-trained models fine-tuned for brain tumor classification:
- **VGG16**: Classic deep architecture with 16 layers
- **ResNet-50**: Residual network with skip connections
- **InceptionV3**: Multi-scale feature extraction

All transfer learning models use:
- ImageNet pre-trained weights
- Frozen base layers during initial training
- Custom classification head
- Batch normalization and dropout for regularization

## 🛠️ Requirements

```python
tensorflow>=2.x
keras
numpy
scikit-learn
matplotlib
seaborn
Pillow
```

### Hardware
- GPU recommended (tested on Google Colab with A100)
- Minimum 8GB RAM
- CUDA-compatible GPU for training acceleration

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-cancer-mri-classification.git
cd brain-cancer-mri-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Google Colab users:
```python
# The notebook automatically installs scikit-learn
!pip install -q scikit-learn
```

## 🚀 Usage

### Running in Google Colab

1. Upload the notebook to Google Colab
2. Enable GPU acceleration (Runtime → Change runtime type → GPU)
3. Mount Google Drive (optional, for data storage):
```python
USE_DRIVE = True  # Set to False for local storage
```

### Running Locally

1. Prepare your dataset in the following structure:
```
dataset/
├── brain_glioma/
├── brain_menin/
└── brain_tumor/
```

2. Run the notebook cells sequentially or execute specific sections:
```python
# Load and preprocess data
split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Build and train a model
model = build_custom_cnn_6()
history, model_path = compile_and_fit(model, train_ds, val_ds, model_name="CNN6", epochs=30)

# Evaluate
evaluate_model(model, test_ds, class_names)
```

### Training a Model

```python
# Example: Train VGG16 with data augmentation
vgg16_model = build_vgg16()
history, model_path = compile_and_fit(
    vgg16_model, 
    aug_train_ds, 
    val_ds,
    model_name="VGG16_aug",
    epochs=20,
    lr=1e-3
)
```

### Evaluation

```python
# Load best model and evaluate
best_model = keras.models.load_model(model_path)
evaluate_model(best_model, test_ds, class_names, model_name="VGG16")
```

## 📁 Project Structure

```
brain-cancer-mri-classification/
├── brain_cancer_mri_classification.ipynb  # Main notebook
├── README.md                              # This file
├── requirements.txt                       # Dependencies
├── models/                                # Saved models
│   └── summaries/                        # Model architecture summaries
├── data/                                  # Dataset (not included)
│   ├── Training/
│   ├── Validation/
│   └── Testing/
└── results/                               # Training histories and plots
```

## 📈 Results

The project provides comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision scores
- **Recall**: Class-wise recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

### Visualization Tools

- Class distribution plots
- Training/validation curves (loss and accuracy)
- Confusion matrices
- Sample predictions with true/predicted labels

## ✨ Features

### Data Augmentation

Two configuration options:
- **Configuration A**: Geometric transformations (flip, rotation, zoom)
- **Configuration B**: Brightness and contrast adjustments

```python
data_augment = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05)
])
```

### Model Management

- Automatic model checkpointing (saves best model based on validation accuracy)
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Model summary export to text files

### Monitoring & Visualization

- Real-time training progress
- Comprehensive EDA (Exploratory Data Analysis)
- Dataset statistics and class distribution
- Sample image display from each class

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Steps to Contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar on Kaggle
- TensorFlow and Keras teams for the deep learning framework
- Pre-trained model architectures from ImageNet

### Dataset Citation

If you use this dataset in your research, please cite:
```
Nickparvar, M. (2021). Brain Tumor MRI Dataset. 
Available at: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
```

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Always consult with medical professionals for clinical diagnosis.
