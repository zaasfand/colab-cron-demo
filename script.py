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
        raise FileNotFoundError(
            f"Expected Fake and Real class folders inside: {train_dir}"
        )

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
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

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
    """Train the model and save it to disk."""
    train_ds, val_ds, class_names = build_datasets(data_dir, validation_split)
    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Class mapping: {class_names[0]}=0, {class_names[1]}=1")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.save(model_path)
    print(f"Model saved to: {model_path}")


def evaluate(data_dir: Path, model_path: Path) -> None:
    """Evaluate the saved model on the Test folder."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    train_dir, _, test_dir = detect_dataset_dirs(data_dir)
    if not test_dir:
        raise FileNotFoundError(f"No Test folder found inside: {data_dir}")

    class_names = detect_class_names(train_dir)
    test_ds = load_image_dataset(test_dir, class_names, shuffle=False)
    model = keras.models.load_model(model_path)
    results = model.evaluate(test_ds, verbose=1, return_dict=True)

    print("Test results:")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.4f}")


def predict(model_path: Path, image_path: Path, threshold: float) -> None:
    """Predict whether one image is real or fake."""
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
    print(f"Fake probability: {1.0 - real_probability:.4f}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run a deepfake image detector.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the deepfake classifier.")
    train_parser.add_argument("--data", required=True, type=Path, help="Dataset folder.")
    train_parser.add_argument("--model", default=Path("artifacts/deepfake_model.keras"), type=Path, help="Where to save the trained model.")
    train_parser.add_argument("--epochs", default=12, type=int, help="Number of training epochs.")
    train_parser.add_argument("--validation-split", default=0.2, type=float, help="Validation split between 0 and 1.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the saved model on the Test split.")
    evaluate_parser.add_argument("--data", required=True, type=Path, help="Dataset folder.")
    evaluate_parser.add_argument("--model", default=Path("artifacts/deepfake_model.keras"), type=Path, help="Trained model path.")

    predict_parser = subparsers.add_parser("predict", help="Predict real/fake for one image.")
    predict_parser.add_argument("--model", default=Path("artifacts/deepfake_model.keras"), type=Path, help="Trained model path.")
    predict_parser.add_argument("--image", required=True, type=Path, help="Image to classify.")
    predict_parser.add_argument("--threshold", default=0.5, type=float, help="Real/fake decision threshold.")

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
    # Comment out the call to main() to prevent argparse from trying to parse
    # the kernel's command-line arguments, which are not valid commands for this script.
    # You can uncomment this line and pass specific arguments to main() if you want to run
    # a command directly, e.g.:
    # main(argv=["train", "--data", "path/to/your/dataset", "--epochs", "5"])
    pass

!nvidia-smi

import torch

print(torch.cuda.is_available())

import os
from pathlib import Path

# The dataset was successfully uploaded as 'Dataset for MLDL.rar' and unRARed
# into the '/content/Dataset for MLDL' directory by previous cells.
# We will now set the DATASET_PATH to this extracted directory.
DATASET_PATH = Path("/content/Dataset for MLDL")

if not DATASET_PATH.is_dir():
    # This block will only execute if the expected directory does not exist.
    # This usually means the previous unRAR extraction failed or the runtime was reset.
    print(f"Error: Expected dataset directory '{DATASET_PATH}' not found.")
    print("Please ensure you have run the cells to upload and extract the RAR file (cells zoL6-S83SYtS and FNPScSMJdHpb).")
    print("If you have a different dataset or location, please update DATASET_PATH accordingly.")
    raise FileNotFoundError(f"Dataset directory '{DATASET_PATH}' not found. Please re-run the extraction steps.")
else:
    print(f"Dataset directory found at: {DATASET_PATH}")
    print("Proceeding with the existing extracted dataset. No new upload/extraction needed.")

# You can now use DATASET_PATH in subsequent calls, e.g., train(DATASET_PATH, ...)

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files

uploaded = files.upload()

!apt-get install unrar -y
!unrar x "Dataset for MLDL.rar"

!ls

import os
print(os.listdir())

!find . -type f | head -50

!nvidia-smi

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception, MobileNetV2

# Define the input shape for consistency
INPUT_SHAPE = (299, 299, 3)

def build_hybrid_attention_model(input_shape=INPUT_SHAPE):
    """Builds a hybrid model combining Xception and MobileNetV2 with self-attention."""

    # 1. Input Layer
    inputs = keras.Input(shape=input_shape)

    # 2. Base Model 1: Xception
    # Preprocess input for Xception. It expects inputs scaled to [-1, 1]
    xception_preprocess = keras.applications.xception.preprocess_input(inputs)
    xception_base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    xception_base.trainable = False
    x1 = xception_base(xception_preprocess, training=False)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # 3. Base Model 2: MobileNetV2
    # Preprocess input for MobileNetV2. It also expects inputs scaled to [-1, 1]
    mobilenet_preprocess = keras.applications.mobilenet_v2.preprocess_input(inputs)
    mobilenet_base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    mobilenet_base.trainable = False
    x2 = mobilenet_base(mobilenet_preprocess, training=False)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # 4. Combine Features
    combined_features = layers.concatenate([x1, x2])

    # 5. Self-Attention Mechanism (simplified dot-product attention)
    # The idea is to allow the model to dynamically weight the importance of different feature parts
    # Here, we use a simple attention block on the concatenated features
    query = layers.Dense(combined_features.shape[-1], activation='relu', name='attention_query')(combined_features)
    key = layers.Dense(combined_features.shape[-1], activation='relu', name='attention_key')(combined_features)
    value = layers.Dense(combined_features.shape[-1], activation='relu', name='attention_value')(combined_features)

    # Apply scaled dot-product attention
    attention_output = layers.Attention(name='self_attention')([query, key, value])

    # Add a residual connection around the attention block
    attended_features = layers.Add(name='attention_residual')([combined_features, attention_output])

    # 6. Classification Head
    x = layers.Dropout(0.5)(attended_features)
    outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)

    # 7. Create and Compile the Model
    model = keras.Model(inputs, outputs, name='hybrid_attention_deepfake_detector')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

# Instantiate the hybrid attention model
base_model = build_hybrid_attention_model()

from google.colab import files

uploaded = files.upload()

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import os # Import os for file system operations

img_path = None

# Check if 'uploaded' is defined and not empty (from previous upload)
if 'uploaded' in globals() and uploaded:
    img_path = list(uploaded.keys())[0]
    print(f"Using uploaded file: {img_path}")
else:
    print("Warning: 'uploaded' variable not found or empty. Attempting to locate image in /content.")
    # Try to find the most recently uploaded image in /content
    image_files = [f for f in os.listdir('/content') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        # Sort by modification time to get the latest file
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join('/content', x)), reverse=True)
        img_path = os.path.join('/content', image_files[0])
        print(f"Using image found: {img_path}")
    else:
        print("No image file found automatically in /content.")
        # Prompt user for manual path if no file is found
        manual_path = input("Please provide the full path to your image file (e.g., /content/my_image.jpg): ")
        if os.path.exists(manual_path):
            img_path = manual_path
            print(f"Using manually provided path: {img_path}")
        else:
            raise FileNotFoundError(f"Image file not found at '{manual_path}'. Please check the path and try again, or re-run the upload cell (iD-P3g7iiLcb).")

# Ensure img_path is set before proceeding
if img_path is None:
    raise FileNotFoundError("No image path could be determined. Please upload a file or provide a path.")

img = image.load_img(img_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prediction = base_model.predict(img_array)

print(prediction)

print(prediction)

import tensorflow as tf

# The prediction from base_model is a feature map, not a single probability.
# We need to add a classification head to convert these features into a single score.
prediction_features = base_model.predict(img_array)

# Define a simple classification head to process the features
# This head will take the output shape of prediction_features (excluding batch dimension)
input_features = tf.keras.Input(shape=prediction_features.shape[1:]) # e.g., (10, 10, 2048)
x = tf.keras.layers.GlobalAveragePooling2D()(input_features)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
classifier_head_model = tf.keras.Model(input_features, outputs)

# Now, use this classification head to get the final score from the features
score = classifier_head_model.predict(prediction_features, verbose=0)[0][0]

if score > 0.5:
    print(f"Fake ({score:.2%} confidence)")
else:
    print(f"Real ({(1-score):.2%} confidence)")