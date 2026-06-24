import tensorflow as tf

model = tf.keras.models.load_model("Chest_Xray_Pneumonia_EfficientNetB0.keras")

print("="*80)
print("TOP MODEL")
print("="*80)
model.summary(expand_nested=False)

print("\n")
print("="*80)
print("NESTED MODEL")
print("="*80)

base = model.get_layer("efficientnetb0")
base.summary()