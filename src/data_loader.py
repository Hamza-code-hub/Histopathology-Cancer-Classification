"""
Dataset loading utilities.
"""

from pathlib import Path
import tensorflow as tf

try:
    from .config import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED
except ImportError:
    from config import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED


AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(
    data_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=RANDOM_SEED,
):
    """
    Load training and validation datasets from directory structure.

    Expected structure:

    dataset/
    ├── colon_adenocarcinoma/
    ├── colon_benign/
    ├── lung_adenocarcinoma/
    ├── lung_benign/
    └── lung_squamous_cell_carcinoma/
    """

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names
