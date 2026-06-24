import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model(
    "best_model.keras"
)

# Get the EfficientNetB0 base model
base_model = model.get_layer("efficientnetb0")

print("=" * 60)
print("EfficientNetB0 Layers")
print("=" * 60)

for layer in base_model.layers:
    print(layer.name)