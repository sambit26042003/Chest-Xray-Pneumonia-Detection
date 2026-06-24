import os
import json
import numpy as np
import tensorflow as tf

MODEL_PATH = "Chest_Xray_Pneumonia_EfficientNetB0 (1).keras"

print("Loading model from:")
print(os.path.abspath(MODEL_PATH))

model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open("labels.json", "r") as f:
    class_names = json.load(f)


def predict_image(image_array):
    """
    Predict NORMAL or PNEUMONIA
    """

    prediction = model.predict(image_array, verbose=0)[0][0]

    print("Raw prediction:", prediction)

    if prediction >= 0.5:
        label = class_names[1]
        confidence = prediction
    else:
        label = class_names[0]
        confidence = 1 - prediction

    return label, float(confidence)