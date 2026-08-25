import tensorflow as tf

from src.preprocessing import normalize_image


def test_normalize_image():
    image = tf.constant(
        [[[255.0, 0.0, 127.5]]]
    )

    result = normalize_image(image)

    assert float(
        tf.reduce_max(result)
    ) <= 1.0

    assert float(
        tf.reduce_min(result)
    ) >= 0.0
