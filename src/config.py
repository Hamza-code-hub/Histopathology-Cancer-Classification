from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
RANDOM_SEED = 42

CLASS_NAMES = [
    "colon_adenocarcinoma",
    "colon_benign",
    "lung_adenocarcinoma",
    "lung_benign",
    "lung_squamous_cell_carcinoma",
]
