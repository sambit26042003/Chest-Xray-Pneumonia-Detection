import tensorflow as tf
import matplotlib.pyplot as plt

from utils import preprocess_image
from gradcam import generate_gradcam

# Load model
model = tf.keras.models.load_model("Chest_Xray_Pneumonia_EfficientNetB0.keras")

# Load image
image, image_array = preprocess_image("images/test.jpeg")

# Generate heatmap
heatmap = generate_gradcam(model, image_array)

# Display
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.colorbar()
plt.show()