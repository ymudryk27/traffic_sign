import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "data"
IMG_SIZE = 64
BATCH = 64

val_dir = os.path.join(DATA_DIR, "val")

val_gen = ImageDataGenerator(rescale=1.0/255)
val_it = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)

print("Classes:", val_it.class_indices)
print("Num validation images:", val_it.samples)

model = keras.models.load_model("model/traffic_model.h5")
loss, acc = model.evaluate(val_it, verbose=1)
print("Validation accuracy =", acc)
print("Validation loss =", loss)
