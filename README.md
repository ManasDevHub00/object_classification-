# Object Classification Using Machine Learning

## Project Overview
A machine learning-based object classification system that identifies everyday objects using image processing, feature extraction, and webcam-based real-time prediction.

## Features
- Custom image dataset.
- Image preprocessing using OpenCV.
- Feature extraction using HOG, HSV Histogram, and Grayscale features.
- Data augmentation using rotation, brightness adjustment, and flipping.
- 70:30 stratified train-test split.
- Model comparison using multiple ML algorithms.
- Real-time webcam object prediction.
- Unknown object detection using a 40% confidence threshold.

## Dataset
10 object categories:
- Charger
- Cup
- Eraser
- Highlighter
- Marker
- Pen
- Pencil
- Remote
- Scissors
- Spoon

## Machine Learning Algorithms
1. Support Vector Machine (SVM)
2. Random Forest
3. Logistic Regression

The best model is automatically selected based on Macro F1-score.

## Evaluation
- Accuracy
- Macro F1-score
- Classification Report
- Confusion Matrix
- Model Accuracy Comparison Graph

## Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image
- Matplotlib
- Joblib

## Installation

```bash
pip install opencv-python numpy scikit-learn scikit-image matplotlib joblib
```

## Run Project

```bash
python main.py
```

The program trains and evaluates the models, saves the best model as `best_model.pkl`, generates evaluation graphs, and starts webcam prediction automatically.

**Press Q to exit the webcam.**

## Output Files
- `best_model.pkl`
- `confusion_matrix.png`
- `accuracy_comparison.png`

## Author
**Manas Sharma**

GitHub: [ManasDevHub00](https://github.com/ManasDevHub00)