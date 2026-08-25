import tensorflow as tf

from src.models import (
    build_resnet50,
    build_efficientnetb3,
)


def test_resnet50_output_shape():
    model = build_resnet50(
        input_shape=(224, 224, 3),
        num_classes=5,
    )

    assert model.output_shape == (
        None,
        5,
    )


def test_efficientnetb3_output_shape():
    model = build_efficientnetb3(
        input_shape=(224, 224, 3),
        num_classes=5,
    )

    assert model.output_shape == (
        None,
        5,
    )
