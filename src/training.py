import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import (
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from src.data import create_dataloaders
from src.models import build_model


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
):

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

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{correct_predictions / total_samples:.4f}",
        )

    epoch_loss = (
        running_loss / total_samples
    )

    epoch_accuracy = (
        correct_predictions / total_samples
    )

    return epoch_loss, epoch_accuracy


# =========================================================
# VALIDATION
# =========================================================

def validate(
    model,
    dataloader,
    criterion,
):

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in tqdm(
            dataloader,
            desc="Validation",
            leave=False,
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    epoch_loss = (
        running_loss / total_samples
    )

    epoch_accuracy = (
        correct_predictions / total_samples
    )

    return epoch_loss, epoch_accuracy


# =========================================================
# SAVE CURVES
# =========================================================

def save_training_curves(
    model_name,
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
):

    output_dir = (
        RESULTS_DIR / "training_curves"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(train_losses) + 1,
    )

    # Loss
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
    plt.title(f"{model_name} Loss")
    plt.legend()

    plt.savefig(
        output_dir / f"{model_name}_loss.png",
        bbox_inches="tight",
    )

    plt.close()

    # Accuracy
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
    plt.title(f"{model_name} Accuracy")
    plt.legend()

    plt.savefig(
        output_dir / f"{model_name}_accuracy.png",
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# GENERIC MODEL TRAINER
# =========================================================

def train_model(
    model_name,
    epochs,
):

    print(
        f"\n========== {model_name.upper()} TRAINING ==========\n"
    )

    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {LEARNING_RATE}")

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    (
        train_loader,
        val_loader,
        _,
        class_names,
        _,
    ) = create_dataloaders(
        model_name
    )

    print(
        f"Training images: {len(train_loader.dataset)}"
    )

    print(
        f"Validation images: {len(val_loader.dataset)}"
    )

    print(
        f"Classes: {len(class_names)}"
    )

    # -----------------------------------------------------
    # Build selected model
    # -----------------------------------------------------

    model = build_model(
        model_name=model_name,
        freeze_backbone=True,
        pretrained=True,
    )

    model = model.to(DEVICE)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Only train unfrozen parameters
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Checkpoint path
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"{model_name}_best.pth"
    )

    best_validation_accuracy = 0.0

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    total_start_time = time.time()

    # =====================================================
    # TRAINING
    # =====================================================

    for epoch in range(epochs):

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        print("-" * 40)

        epoch_start = time.time()

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
        )

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
            time.time() - epoch_start
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        print(
            f"Epoch Time: {epoch_time / 60:.2f} minutes"
        )

        # -------------------------------------------------
        # Save best checkpoint
        # -------------------------------------------------

        if val_accuracy > best_validation_accuracy:

            best_validation_accuracy = (
                val_accuracy
            )

            torch.save(
                model.state_dict(),
                model_path,
            )

            print(
                f"Saved best checkpoint: {model_path}"
            )

    total_time = (
        time.time() - total_start_time
    )

    print(
        "\n========== TRAINING COMPLETE ==========\n"
    )

    print(
        f"Model: {model_name}"
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
        f"Checkpoint: {model_path}"
    )

    save_training_curves(
        model_name,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
    )


# =========================================================
# COMMAND LINE ARGUMENTS
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train Indian bird classifier"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "resnet50",
            "efficientnet_v2_b0",
            "mobilenet_v2",
            "vit_b32",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = parse_arguments()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
    )