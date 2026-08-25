"""
Model evaluation utilities.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(
    model_path,
    dataset,
    class_names,
    output_dir="assets",
):
    """
    Evaluate model and generate confusion matrix.
    """

    model = tf.keras.models.load_model(
        model_path
    )

    y_true = []
    y_pred = []

    for images, labels in dataset:
        probabilities = model.predict(
            images,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        targets = np.argmax(
            labels.numpy(),
            axis=1,
        )

        y_true.extend(targets)
        y_pred.extend(predictions)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )

    plt.tight_layout()

    output_path = (
        output_dir /
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved to: "
        f"{output_path}"
    )

    return matrix
