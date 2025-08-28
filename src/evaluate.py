import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model("models/cnn_model.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/interim/test",
    image_size=(128, 128),
    batch_size=32
)

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc*100:.2f}%")
