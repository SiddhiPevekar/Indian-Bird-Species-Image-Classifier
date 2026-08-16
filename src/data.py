from collections import Counter

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.config import (
    SOURCE_TRAIN_DIR,
    VAL_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    TEST_SPLIT_FROM_TRAIN,
)

from src.preprocessing import (
    train_transform,
    eval_transform,
)


# =========================================================
# CREATE DATASETS
# =========================================================

def create_datasets():
    """
    Creates training, validation and test datasets.

    Original dataset:
        train/ = 80%
        valid/ = 20%

    We split original train/ into:
        80% training
        20% testing

    Final overall split:
        train = 64%
        test  = 16%
        val   = 20%
    """

    # -----------------------------------------------------
    # Load original training directory WITHOUT transforms.
    #
    # We mainly need this object to obtain:
    #   - image paths
    #   - class labels
    #   - class names
    # -----------------------------------------------------

    base_train_dataset = ImageFolder(
        root=SOURCE_TRAIN_DIR
    )

    # targets contains label IDs like:
    #
    # [0, 0, 0, ..., 1, 1, 1, ..., 24]
    #
    targets = base_train_dataset.targets

    indices = list(range(len(base_train_dataset)))

    # -----------------------------------------------------
    # Stratified train/test split
    #
    # This keeps every bird species equally represented.
    # -----------------------------------------------------

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SPLIT_FROM_TRAIN,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=targets,
    )

    # -----------------------------------------------------
    # Important:
    #
    # Training and testing point to the SAME image files,
    # but use DIFFERENT transforms.
    #
    # Training -> augmentation
    # Testing  -> deterministic preprocessing
    # -----------------------------------------------------

    full_train_dataset = ImageFolder(
        root=SOURCE_TRAIN_DIR,
        transform=train_transform,
    )

    full_test_dataset = ImageFolder(
        root=SOURCE_TRAIN_DIR,
        transform=eval_transform,
    )

    # Select only the appropriate image indices.
    train_dataset = Subset(
        full_train_dataset,
        train_indices,
    )

    test_dataset = Subset(
        full_test_dataset,
        test_indices,
    )

    # -----------------------------------------------------
    # Validation data already exists separately.
    # -----------------------------------------------------

    val_dataset = ImageFolder(
        root=VAL_DIR,
        transform=eval_transform,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        base_train_dataset.classes,
        base_train_dataset.class_to_idx,
    )


# =========================================================
# CREATE DATALOADERS
# =========================================================

def create_dataloaders():
    """
    Creates PyTorch DataLoaders for training,
    validation and testing.
    """

    (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names,
        class_to_idx,
    ) = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names,
        class_to_idx,
    )


# =========================================================
# DATASET VERIFICATION
# =========================================================

def verify_dataset():
    """
    Prints information about our dataset to verify
    that our split and labels are correct.
    """

    (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names,
        class_to_idx,
    ) = create_datasets()

    print("\n========== DATASET SUMMARY ==========\n")

    print(f"Number of classes: {len(class_names)}")

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Testing images:    {len(test_dataset)}")

    total = (
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
    )

    print(f"Total images:      {total}")

    print("\n========== CLASS MAPPING ==========\n")

    for class_name, label in class_to_idx.items():
        print(f"{label:2d} -> {class_name}")

    # -----------------------------------------------------
    # Verify class distribution in train/test.
    # -----------------------------------------------------

    train_targets = [
        train_dataset.dataset.targets[i]
        for i in train_dataset.indices
    ]

    test_targets = [
        test_dataset.dataset.targets[i]
        for i in test_dataset.indices
    ]

    val_targets = val_dataset.targets

    train_counts = Counter(train_targets)
    test_counts = Counter(test_targets)
    val_counts = Counter(val_targets)

    print("\n========== SAMPLES PER CLASS ==========\n")

    print("Class                        Train   Val   Test")

    for index, class_name in enumerate(class_names):

        print(
            f"{class_name:28s} "
            f"{train_counts[index]:5d} "
            f"{val_counts[index]:5d} "
            f"{test_counts[index]:5d}"
        )


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    verify_dataset()