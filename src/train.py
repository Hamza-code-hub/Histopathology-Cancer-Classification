"""
Model training entry point.

Example:

python -m src.train \
    --data dataset \
    --model resnet50 \
    --epochs 10
"""

import argparse
from pathlib import Path
import tensorflow as tf

from .data_loader import load_datasets
from .models import get_model
from .config import MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train histopathology classifier"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="dataset",
        help="Dataset directory",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnetb3"],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    return parser.parse_args()


def train(
    data_dir,
    model_name,
    epochs,
    learning_rate,
):
    train_ds, val_ds, class_names = load_datasets(
        data_dir
    )

    model = get_model(
        model_name,
        num_classes=len(class_names),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        MODEL_DIR /
        f"{model_name}_best.keras"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    final_path = (
        MODEL_DIR /
        f"{model_name}_final.keras"
    )

    model.save(final_path)

    print("\nTraining complete")
    print("Classes:", class_names)
    print("Saved:", final_path)

    return model, history, class_names


if __name__ == "__main__":
    args = parse_args()

    train(
        data_dir=args.data,
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
