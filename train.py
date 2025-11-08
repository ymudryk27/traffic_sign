import os, json, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(img_size, num_classes):
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling="avg"
    )
    base.trainable = False  # transfer learning

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, base


def main(data_dir, img_size, batch, epochs):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_gen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1
    )
    val_gen = ImageDataGenerator(rescale=1.0/255)

    train_it = train_gen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch,
        class_mode="categorical",
        shuffle=True
    )
    val_it = val_gen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch,
        class_mode="categorical",
        shuffle=False
    )

    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)

    # Save index -> class_id mapping based on generator's class_indices
    class_indices = train_it.class_indices  # e.g. {"0": 0, "1": 1, ...}
    index_to_class = {str(v): k for k, v in class_indices.items()}
    with open(os.path.join("model", "labels.json"), "w", encoding="utf-8") as f:
        json.dump(index_to_class, f, ensure_ascii=False, indent=2)

    model, base = build_model(img_size, len(class_indices))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy", mode="max")
    ]

    model.fit(train_it, validation_data=val_it, epochs=epochs, callbacks=callbacks)

    # Optional fine-tuning
    base.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_it, validation_data=val_it, epochs=max(2, epochs // 3), callbacks=callbacks)

    # Save final model in modern Keras format
    os.makedirs("model", exist_ok=True)
    model.save(os.path.join("model", "traffic_model.keras"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    main(args.data_dir, args.img_size, args.batch, args.epochs)