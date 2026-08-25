"""
Image preprocessing and augmentation utilities.
"""

import tensorflow as tf


def get_data_augmentation():
    """
    Return augmentation pipeline for training images.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def normalize_image(image):
    """
    Convert image to float32 and normalize to [0, 1].
    """

    image = tf.cast(image, tf.float32)
    return image / 255.0
