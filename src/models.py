"""
Deep learning model definitions.

Supported backbones:
- ResNet50
- EfficientNetB3
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

try:
    from .config import IMAGE_SIZE, NUM_CLASSES
    from .preprocessing import get_data_augmentation
except ImportError:
    from config import IMAGE_SIZE, NUM_CLASSES
    from preprocessing import get_data_augmentation


def build_resnet50(
    input_shape=(*IMAGE_SIZE, 3),
    num_classes=NUM_CLASSES,
    dropout_rate=0.30,
    train_backbone=False,
):
    """
    Build transfer-learning classifier with ResNet50.
    """

    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    base_model.trainable = train_backbone

    inputs = layers.Input(shape=input_shape)

    x = get_data_augmentation()(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    return Model(
        inputs,
        outputs,
        name="ResNet50_Histopathology",
    )


def build_efficientnetb3(
    input_shape=(*IMAGE_SIZE, 3),
    num_classes=NUM_CLASSES,
    dropout_rate=0.30,
    train_backbone=False,
):
    """
    Build transfer-learning classifier with EfficientNetB3.
    """

    base_model = tf.keras.applications.EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    base_model.trainable = train_backbone

    inputs = layers.Input(shape=input_shape)

    x = get_data_augmentation()(inputs)

    # EfficientNet preprocessing is already included
    # in recent TensorFlow/Keras implementations.
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    return Model(
        inputs,
        outputs,
        name="EfficientNetB3_Histopathology",
    )


def get_model(model_name, **kwargs):
    """
    Model factory.
    """

    name = model_name.lower()

    if name in {"resnet50", "resnet"}:
        return build_resnet50(**kwargs)

    if name in {
        "efficientnetb3",
        "efficientnet",
        "effb3",
    }:
        return build_efficientnetb3(**kwargs)

    raise ValueError(
        "Unknown model. Use 'resnet50' or 'efficientnetb3'."
    )
