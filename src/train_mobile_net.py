import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 32
EPOCHS_HEAD = 10        # train classifier head
EPOCHS_FINE_TUNE = 20   # fine-tune deeper layers
IMG_SIZE = (244, 244)

# -----------------------------
# Dataset
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/interim/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/interim/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/interim/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Normalization + Augmentation
normalization_layer = layers.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

num_classes = len(train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    train_ds
    .map(lambda x, y: (normalization_layer(x), y))
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
val_ds = (
    val_ds
    .map(lambda x, y: (normalization_layer(x), y))
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Base Model (MobileNetV2)
# -----------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False   # freeze backbone

# -----------------------------
# Build Model
# -----------------------------
model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax",
                 kernel_regularizer=regularizers.l2(1e-4))
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Compile for head training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=["accuracy"]
)

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("models/mobilenetv2_best.keras", monitor="val_accuracy",
                             save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

# -----------------------------
# Phase 1: Train Head
# -----------------------------
print("\n🔹 Training classifier head...")
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# -----------------------------
# Phase 2: Fine-tune deeper layers
# -----------------------------
print("\n🔹 Fine-tuning backbone...")
base_model.trainable = True
for layer in base_model.layers[:-40]:   # freeze all but last 40 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # smaller LR for fine-tune
    loss=loss_fn,
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# -----------------------------
# Final Evaluation
# -----------------------------
print("\n🔹 Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Save final model
model.save("models/mobilenetv2_final.keras")
