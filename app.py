from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)

# -----------------------------------------
# Upload folder
# -----------------------------------------

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------------------
# Load trained model
# -----------------------------------------

model = tf.saved_model.load("potato_disease_model")

predict_fn = model.signatures["serving_default"]


# -----------------------------------------
# Class names
# -----------------------------------------

class_names = [
    "Early Blight",
    "Late Blight",
    "Healthy Potato Leaf"
]


# -----------------------------------------
# Disease information
# -----------------------------------------

disease_info = {

    "Early Blight": {
        "description":
            "Early blight is a fungal disease that commonly affects potato leaves.",

        "advice":
            "Remove severely affected leaves and maintain good field hygiene."
    },

    "Late Blight": {
        "description":
            "Late blight is a disease that can cause dark lesions on potato leaves.",

        "advice":
            "Remove infected plant material and follow appropriate disease-management practices."
    },

    "Healthy Potato Leaf": {
        "description":
            "The model detected a healthy potato leaf.",

        "advice":
            "Continue regular monitoring and good crop-management practices."
    }
}


# -----------------------------------------
# Home page
# -----------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------------
# Prediction
# -----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check image
    if "image" not in request.files:

        return "No image uploaded"


    file = request.files["image"]


    # Check file name
    if file.filename == "":

        return "No file selected"


    # -------------------------------------
    # Save uploaded image
    # -------------------------------------

    filename = str(uuid.uuid4()) + ".jpg"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # -------------------------------------
    # Open image
    # -------------------------------------

    image = Image.open(file).convert("RGB")


    # Save original image
    image.save(filepath)


    # -------------------------------------
    # Preprocess image
    # -------------------------------------

    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)


    # -------------------------------------
    # Prediction
    # -------------------------------------

    prediction = predict_fn(
        images=tf.constant(image)
    )


    # Get model output
    output = prediction["output_0"].numpy()


    # Get predicted class
    predicted_class = np.argmax(output[0])


    # Get confidence
    confidence = np.max(output[0])

    confidence_percent = float(confidence) * 100


    # -------------------------------------
    # Low confidence check
    # -------------------------------------

    if confidence_percent < 70:

        return render_template(

            "index.html",

            prediction="Prediction Uncertain",

            confidence=round(
                confidence_percent,
                2
            ),

            image_path=filepath,

            description=
                "The model is not sufficiently confident about this image.",

            advice=
                "Please upload a clear potato leaf image with good lighting."
        )


    # -------------------------------------
    # Normal prediction
    # -------------------------------------

    disease = class_names[predicted_class]


    return render_template(

        "index.html",

        prediction=disease,

        confidence=round(
            confidence_percent,
            2
        ),

        image_path=filepath,

        description=
            disease_info[disease]["description"],

        advice=
            disease_info[disease]["advice"]
    )


# -----------------------------------------
# Run Flask
# -----------------------------------------

if __name__ == "__main__":

    app.run(debug=True)