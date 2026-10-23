"""
Train a Plant Disease classifier using a folder-based image dataset.

Expected dataset layout:
dataset/
  Tomato___Early_blight/
  Tomato___Late_blight/
  Tomato___healthy/
  ...

The script uses transfer learning with MobileNetV2.
"""
from pathlib import Path
import tensorflow as tf

DATASET = Path("dataset")
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 8

if not DATASET.exists():
    raise SystemExit("Create a 'dataset' folder and put one subfolder per class inside it.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH
)

class_names = train_ds.class_names
Path("model").mkdir(exist_ok=True)
Path("model/class_names.txt").write_text("\n".join(class_names))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("model/model.keras")
print("Saved model/model.keras")
print("Classes:", class_names)
