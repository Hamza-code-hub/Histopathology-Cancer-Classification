"""
Training-history visualization utilities.
"""

from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_history(
    history,
    output_dir="assets",
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = [
        ("accuracy", "val_accuracy"),
        ("loss", "val_loss"),
    ]

    for train_metric, val_metric in metrics:

        if train_metric not in history.history:
            continue

        plt.figure(figsize=(8, 5))

        plt.plot(
            history.history[train_metric],
            label=f"Training {train_metric}",
        )

        if val_metric in history.history:
            plt.plot(
                history.history[val_metric],
                label=f"Validation {train_metric}",
            )

        plt.xlabel("Epoch")
        plt.ylabel(train_metric.title())
        plt.title(
            f"Training and Validation "
            f"{train_metric.title()}"
        )

        plt.legend()
        plt.tight_layout()

        output_path = (
            output_dir /
            f"{train_metric}_curve.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
