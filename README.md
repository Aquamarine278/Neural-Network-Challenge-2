# Neural-Network-Challenge-2

## Overview
This project builds a neural network model to predict employee attrition and department classification using deep learning. The dataset is preprocessed and trained using TensorFlow and scikit-learn.

## Features
- **Data Preprocessing**: Standardization and one-hot encoding.
- **Neural Network Model**: A shared architecture with two branches for multi-output prediction.
- **Training & Evaluation**: Uses categorical and binary cross-entropy loss with Adam optimizer.

## Dataset
The dataset contains employee-related attributes such as:
- **Independent Variables (Features)**: Education, Age, DistanceFromHome, JobSatisfaction, etc.
- **Dependent Variables (Targets)**: Attrition (Yes/No), Department (HR, R&D, Sales).

## Technologies Used
- Python
- TensorFlow/Keras
- Scikit-learn
- Pandas & NumPy

## Model Architecture
- **Input Layer**: Accepts numerical features.
- **Shared Layers**: Two dense layers (64 & 128 neurons) with ReLU activation.
- **Branch for Department**:
  - Hidden Layer (32 neurons, ReLU)
  - Output Layer (3 neurons, Softmax activation)
- **Branch for Attrition**:
  - Hidden Layer (32 neurons, ReLU)
  - Output Layer (2 neurons, Sigmoid activation)

## Training
The model is compiled with:
- **Loss Function**:
  - Categorical Cross-Entropy (Department)
  - Binary Cross-Entropy (Attrition)
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Epochs**: 100
- **Batch Size**: 32

## Performance Evaluation
- The model is evaluated using test data.
- Accuracy for both classification tasks is printed.
- Balanced accuracy could be considered for better evaluation due to class imbalances.

## Improvements
- **Addressing Class Imbalance**: Oversampling or undersampling.
- **Model Enhancement**: Increase neurons or add more layers.
- **Feature Engineering**: Exploring additional relevant features.

## How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn pandas numpy
   ```
2. Run the script to preprocess data, train, and evaluate the model:
   ```bash
   python script.py
   ```


