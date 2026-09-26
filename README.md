# 🥔 Potato Disease Detection

A machine learning web application that detects potato leaf diseases from uploaded images using TensorFlow and Flask.

## 📌 Project Overview

This project uses a deep learning image classification model to identify the condition of potato leaves.

The model classifies images into three categories:

- Early Blight
- Late Blight
- Healthy Potato Leaf

A Flask web application provides an interface where users can upload a potato leaf image and receive the predicted class, confidence score, disease description, and general advice.

## 🚀 Features

- Potato leaf image classification
- Image upload through a Flask web application
- TensorFlow-based deep learning model
- Prediction confidence score
- Low-confidence prediction handling
- Disease description
- General disease-management advice
- Simple and user-friendly interface


# 📂 Project Structure
potato-disease-detection/
│
├── potato_disease_model/
│   ├── assets/
│   ├── variables/
│   ├── fingerprint.pb
│   └── saved_model.pb
│
├── static/
│   ├── uploads/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── .gitignore

## 🧠 Model

The model was trained using TensorFlow and transfer learning.

### Model Input


Image Size: 224 × 224
Channels: 3 (RGB)
Input Shape: (1, 224, 224, 3)

# 🧪 Model Training

The model was trained separately using a TensorFlow training notebook.

Training Process

1. Collected and organized potato leaf images into three classes.
2. Created training and validation datasets.
3. Resized images to 224 × 224.
4. Converted images into TensorFlow tensors.
5. Used a pretrained CNN model for feature extraction.
6. Added a classification layer for the three classes.
7. Trained the model using the training dataset.
8. Evaluated the model using validation data.
9. Saved the trained model in TensorFlow SavedModel format.
10. Used the saved model in the Flask application for prediction.
