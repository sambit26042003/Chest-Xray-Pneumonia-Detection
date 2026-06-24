from utils import preprocess_image
from predict import predict_image

image_path = "images/test.jpeg"

image, image_array = preprocess_image(image_path)

label, confidence = predict_image(image_array)

print("Prediction :", label)
print("Confidence :", round(confidence * 100, 2), "%")