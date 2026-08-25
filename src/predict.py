"""
Single-image inference utility.

Example:

python -m src.predict \
    --model models/resnet50_best.keras \
    --image sample.jpg
"""

import argparse

import numpy as np
import tensorflow as tf

from .config import (
    IMAGE_SIZE,
    CLASS_NAMES,
)


def load_image(
    image_path,
    image_size=IMAGE_SIZE,
):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=image_size,
    )

    image = tf.keras.utils.img_to_array(
        image
    )

    return np.expand_dims(
        image,
        axis=0,
    )


def predict_image(
    model_path,
    image_path,
    class_names=CLASS_NAMES,
):
    model = tf.keras.models.load_model(
        model_path
    )

    image = load_image(
        image_path
    )

    probabilities = model.predict(
        image,
        verbose=0,
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = (
        class_names[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            name: float(prob)
            for name, prob
            in zip(
                class_names,
                probabilities,
            )
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = predict_image(
        args.model,
        args.image,
    )

    print(
        "\nPrediction:",
        result["class"],
    )

    print(
        "Confidence:",
        f'{result["confidence"]:.2%}',
    )

    print("\nAll probabilities:")

    for class_name, probability \
            in result["probabilities"].items():

        print(
            f"{class_name}: "
            f"{probability:.2%}"
        )
