import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
import matplotlib.pyplot as plt


def generate_gradcam(model, image_array, last_conv_layer_name="top_conv"):
    """
    Generate Grad-CAM heatmap.
    """

    # Get EfficientNet base model
    base_model = model.get_layer("efficientnetb0")

    # Last convolutional layer
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.
    """

    heatmap = cv2.resize(
        heatmap,
        (original_image.width, original_image.height)
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cm.jet(heatmap)[:, :, :3]

    heatmap = np.uint8(255 * heatmap)

    original = np.array(original_image)

    superimposed = cv2.addWeighted(
        original,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return Image.fromarray(superimposed)