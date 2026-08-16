from pathlib import Path
import torch
import os

# =========================================================
# PROJECT PATHS
# =========================================================

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DATA_DIR = PROJECT_ROOT / "data"
# DATASET_DIR = DATA_DIR / "Birds_25"

# SOURCE_TRAIN_DIR = DATASET_DIR / "train"
# VAL_DIR = DATASET_DIR / "valid"

# MODEL_DIR = PROJECT_ROOT / "models"
# RESULTS_DIR = PROJECT_ROOT / "results"

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# Dataset path
#
# Local Mac default:
#   project/data/Birds_25
#
# Kaggle:
#   supplied using BIRD_DATASET_DIR environment variable
# ---------------------------------------------------------

DATASET_DIR = Path(
    os.getenv(
        "BIRD_DATASET_DIR",
        PROJECT_ROOT / "data" / "Birds_25",
    )
)

SOURCE_TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "valid"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# =========================================================
# DATASET SETTINGS
# =========================================================

NUM_CLASSES = 25

# Original dataset:
# train = 80% of complete dataset
# valid = 20% of complete dataset
#
# We split the original training set:
#
# 80% of original train -> final training
# 20% of original train -> final testing
#
# Overall:
# train = 64%
# test  = 16%
# val   = 20%

TEST_SPLIT_FROM_TRAIN = 0.20

RANDOM_SEED = 42


# =========================================================
# IMAGE SETTINGS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32


# =========================================================
# DATALOADER SETTINGS
# =========================================================

# Keep this conservative on macOS initially.
# NUM_WORKERS = 0
NUM_WORKERS = int(
    os.getenv("NUM_WORKERS", "0")
)


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")

else:
    DEVICE = torch.device("cpu")


# =========================================================
# TRAINING SETTINGS
# =========================================================

RESNET_EPOCHS = 3

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


if __name__ == "__main__":
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Training directory: {SOURCE_TRAIN_DIR}")
    print(f"Validation directory: {VAL_DIR}")
    print(f"Using device: {DEVICE}")