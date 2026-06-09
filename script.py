from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 32
SEED = 42


def first_existing_dir(root: Path, names: Iterable[str]) -> Optional[Path]:
    """Return the first matching child directory, ignoring case."""
    if not root.exists():
        return None

    children = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
    for name in names:
        found = children.get(name.lower())
        if found:
            return found
    return None


def detect_dataset_dirs(data_dir: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Find Train, Validation, and Test folders if the dataset uses splits."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    train_dir = first_existing_dir(data_dir, ["Train", "Training", "train"])
    validation_dir = first_existing_dir(data_dir, ["Validation", "Valid", "Val", "validation"])
    test_dir = first_existing_dir(data_dir, ["Test", "Testing", "test"])

    if train_dir:
        return train_dir, validation_dir, test_dir

    return data_dir, None, None


def detect_class_names(train_dir: Path) -> list[str]:
    """Return class names with fake as 0 and real as 1."""
    fake_dir = first_existing_dir(train_dir, ["Fake", "fake", "Deepfake", "deepfake"])
    real_dir = first_existing_dir(train_dir, ["Real", "real"])

    if not fake_dir or not real_dir:
        raise FileNotFoundError(f"Expected Fake and Real class folders inside: {train_dir}")

    return [fake_dir.name, real_dir.name]


def load_image_dataset(directory: Path, class_names: list[str], shuffle: bool):
    """Load one image dataset split."""
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        shuffle=shuffle,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    ).prefetch(tf.data.AUTOTUNE)


def build_datasets(data_dir: Path, validation_split: float = 0.2):
    """Load training and validation datasets."""
    train_dir, validation_dir, _ = detect_dataset_dirs(data_dir)
    class_names = detect_class_names(train_dir)

    if validation_dir:
        train_ds = load_image_dataset(train_dir, class_names, shuffle=True)
        val_ds = load_image_dataset(validation_dir, class_names, shuffle=False)
        return train_ds, val_ds, class_names

    # Fallback: auto split
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        validation_split=validation_split,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        validation_split=validation_split,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model() -> keras.Model:
    """Create a transfer-learning classifier using MobileNetV2."""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="deepfake_detector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train(data_dir: Path, model_path: Path, epochs: int, validation_split: float) -> None:
    train_ds, val_ds, class_names = build_datasets(data_dir, validation_split)
    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=model_path, monitor="val_auc", mode="max", save_best_only=True),
    ]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Class mapping: {class_names[0]}=0, {class_names[1]}=1")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.save(model_path)
    print(f"Model saved to: {model_path}")


def evaluate(data_dir: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _, _, test_dir = detect_dataset_dirs(data_dir)
    if not test_dir:
        raise FileNotFoundError(f"No Test folder found inside: {data_dir}")

    train_dir, _, _ = detect_dataset_dirs(data_dir)
    class_names = detect_class_names(train_dir)
    test_ds = load_image_dataset(test_dir, class_names, shuffle=False)

    model = keras.models.load_model(model_path)
    results = model.evaluate(test_ds, verbose=1, return_dict=True)

    print("Test results:")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.4f}")


def predict(model_path: Path, image_path: Path, threshold: float = 0.5) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = keras.models.load_model(model_path)
    image = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = keras.utils.img_to_array(image)
    image_batch = tf.expand_dims(image_array, axis=0)

    real_probability = float(model.predict(image_batch, verbose=0)[0][0])
    label = "REAL" if real_probability >= threshold else "FAKE"
    confidence = real_probability if label == "REAL" else 1.0 - real_probability

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Real probability: {real_probability:.4f}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake Image Detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", required=True, type=Path)
    train_parser.add_argument("--model", default=Path("artifacts/deepfake_model.keras"), type=Path)
    train_parser.add_argument("--epochs", default=12, type=int)
    train_parser.add_argument("--validation-split", default=0.2, type=float)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--data", required=True, type=Path)
    evaluate_parser.add_argument("--model", default=Path("artifacts/deepfake_model.keras"), type=Path)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", default=Path("artifacts/deepfake_model.keras"), type=Path)
    predict_parser.add_argument("--image", required=True, type=Path)
    predict_parser.add_argument("--threshold", default=0.5, type=float)

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        train(args.data, args.model, args.epochs, args.validation_split)
    elif args.command == "evaluate":
        evaluate(args.data, args.model)
    elif args.command == "predict":
        predict(args.model, args.image, args.threshold)


if __name__ == "__main__":
    # For GitHub Actions, we will pass arguments via command line
    # Example: python script.py train --data /path/to/dataset
    main()