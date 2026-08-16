import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import (
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR,
    RESNET_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from src.data import create_dataloaders
from src.models import build_resnet50


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
):
    """
    Trains the model for one complete epoch.

    Returns:
        average training loss
        training accuracy
    """

    # Enable training behaviour.
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress_bar:

        # Move images and labels to MPS / CUDA / CPU.
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients from previous batch.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Calculate classification loss.
        loss = criterion(
            outputs,
            labels,
        )

        # Backpropagation.
        loss.backward()

        # Update trainable parameters.
        optimizer.step()

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        current_accuracy = (
            correct_predictions
            / total_samples
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{current_accuracy:.4f}",
        )

    epoch_loss = (
        running_loss
        / total_samples
    )

    epoch_accuracy = (
        correct_predictions
        / total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# =========================================================
# VALIDATE MODEL
# =========================================================

def validate(
    model,
    dataloader,
    criterion,
):
    """
    Evaluates the model on the validation dataset.

    No gradient updates happen here.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    # Disable gradient computation.
    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass only.
            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    epoch_loss = (
        running_loss
        / total_samples
    )

    epoch_accuracy = (
        correct_predictions
        / total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# =========================================================
# PLOT TRAINING HISTORY
# =========================================================

def plot_training_history(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
):
    """
    Saves training and validation curves.
    """

    output_directory = (
        RESULTS_DIR
        / "training_curves"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(train_losses) + 1,
    )

    # -----------------------------------------------------
    # LOSS CURVE
    # -----------------------------------------------------

    plt.figure()

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss",
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet50 Loss")
    plt.legend()

    plt.savefig(
        output_directory
        / "resnet50_loss.png",
        bbox_inches="tight",
    )

    plt.close()

    # -----------------------------------------------------
    # ACCURACY CURVE
    # -----------------------------------------------------

    plt.figure()

    plt.plot(
        epochs,
        train_accuracies,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        val_accuracies,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet50 Accuracy")
    plt.legend()

    plt.savefig(
        output_directory
        / "resnet50_accuracy.png",
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# TRAIN RESNET50
# =========================================================

def train_resnet50():

    print("\n========== RESNET50 TRAINING ==========\n")

    print(f"Device: {DEVICE}")
    print(f"Epochs: {RESNET_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    (
        train_loader,
        val_loader,
        _,
        class_names,
        _,
    ) = create_dataloaders()

    print(
        f"Training images: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Classes: "
        f"{len(class_names)}"
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    model = build_resnet50(
        freeze_backbone=True
    )

    model = model.to(DEVICE)

    # -----------------------------------------------------
    # Loss function
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Optimizer
    #
    # Only parameters with requires_grad=True are supplied.
    #
    # Because the backbone is frozen, this means AdamW
    # updates only our new classifier.
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
            parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "resnet50_best.pth"
    )

    # -----------------------------------------------------
    # History
    # -----------------------------------------------------

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_validation_accuracy = 0.0

    total_start_time = time.time()

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for epoch in range(
        RESNET_EPOCHS
    ):

        print(
            f"\nEpoch "
            f"{epoch + 1}/{RESNET_EPOCHS}"
        )

        print("-" * 40)

        epoch_start_time = time.time()

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        (
            val_loss,
            val_accuracy,
        ) = validate(
            model,
            val_loader,
            criterion,
        )

        # -------------------------------------------------
        # Save history
        # -------------------------------------------------

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )

        epoch_time = (
            time.time()
            - epoch_start_time
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Epoch Time: "
            f"{epoch_time / 60:.2f} minutes"
        )

        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------

        if (
            val_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                val_accuracy
            )

            torch.save(
                model.state_dict(),
                model_path,
            )

            print(
                "Saved new best model:"
            )

            print(
                model_path
            )

    # =====================================================
    # TRAINING FINISHED
    # =====================================================

    total_time = (
        time.time()
        - total_start_time
    )

    print(
        "\n========== TRAINING COMPLETE ==========\n"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Total Training Time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Best Model: "
        f"{model_path}"
    )

    # -----------------------------------------------------
    # Save graphs
    # -----------------------------------------------------

    plot_training_history(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_resnet50()