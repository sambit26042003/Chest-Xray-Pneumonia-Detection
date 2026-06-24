import tensorflow as tf

model = tf.keras.models.load_model("Chest_Xray_Pneumonia_EfficientNetB0.keras")

model.summary(expand_nested=True)