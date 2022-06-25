from keras.models import load_model
from PIL import Image
import numpy as np
import flask
from flask_cors import CORS
import io
import cv2

# Inicjalizacja aplikacji oraz modelu Keras
app = flask.Flask(__name__)
CORS(app)
model = None

# Skojarzenie wartości numerycznych z literami alfabetu
alfabet = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
           13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
           25: 'Z'}


def load_model_from_disk():
    global model
    model = load_model('model.h5')


def prepare_image(image, target):
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target)

    img_copy = cv2.GaussianBlur(image, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))

    return img_final


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, target=(400, 400))
            preds = model.predict(image)
            img_pred = alfabet[np.argmax(preds)]

            data["prediction"] = img_pred
            data["success"] = True

    return flask.jsonify(data)

if __name__ == "__main__":
    load_model_from_disk()
    app.run()
