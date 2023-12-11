from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np


def get_labels():
    with open("labels.txt", "r") as file:
        labels = file.read().splitlines()
    return labels


def preprocessing_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def predict_image(image):
    model = load_model("model.h5", compile=False)
    labels = get_labels()

    predict = model.predict(image)
    best_index = np.argmax(predict)
    class_name = labels[best_index]

    return class_name


app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['jpg', 'png', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "uploads"


def allowed_file(filename):
    return "." in filename and filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200


@app.route("/ml/vision", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            # Check if the request contains an image file
            if "image" not in request.files:
                return jsonify({
                    "status": {
                        "code": 400,
                        "message": "No image file provided"
                    },
                    "data": None
                }), 400

            img = request.files["image"]

            # Check if the file is an allowed image type
            if img and allowed_file(img.filename):
                # Save input image
                filename = secure_filename(img.filename)
                img.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                image_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], filename)

                # Preprocess the input image
                preprocessed_img = preprocessing_image(image_path)

                # Predict the class name
                class_name = predict_image(preprocessed_img)

                return jsonify({
                    "status": {
                        "code": 200,
                        "message": "Prediction successful"
                    },
                    "data": {
                        "title": class_name
                    }
                }), 200

            else:
                return jsonify({
                    "status": {
                        "code": 400,
                        "message": "Invalid or unsupported file type"
                    },
                    "data": None
                }), 400

        except Exception as e:
            return jsonify({
                "status": {
                    "code": 500,
                    "message": f"Internal Server Error: {str(e)}"
                },
                "data": None
            }), 500

    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None
        }), 405


if __name__ == "__main__":
    app.run()
