from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tqdm import tqdm

from src.config import (
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR,
    NUM_CLASSES,
)

from src.data import create_dataloaders
from src.models import build_resnet50


# =========================================================
# LOAD TRAINED RESNET50
# =========================================================

def load_resnet50():
    """
    Rebuild the ResNet50 architecture and load
    our best trained checkpoint.
    """

    model = build_resnet50(
        freeze_backbone=True
    )

    model_path = (
        MODEL_DIR
        / "resnet50_best.pth"
    )

    # Load only the saved model weights.
    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    # Disable dropout and put BatchNorm layers
    # into inference mode.
    model.eval()

    return model


# =========================================================
# GET MODEL PREDICTIONS
# =========================================================

def get_predictions(
    model,
    dataloader,
):
    """
    Runs inference over the complete test dataset.

    Returns:
        true labels
        predicted labels
        class probabilities
    """

    all_labels = []
    all_predictions = []
    all_probabilities = []

    model.eval()

    with torch.no_grad():

        for images, labels in tqdm(
            dataloader,
            desc="Testing ResNet50",
        ):

            images = images.to(DEVICE)

            # Raw output scores / logits.
            outputs = model(images)

            # Convert logits into probabilities.
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            # Highest probability class.
            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities,
):
    """
    Calculates the metrics used in the original project.
    """

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    top3_accuracy = top_k_accuracy_score(
        labels,
        probabilities,
        k=3,
        labels=list(range(NUM_CLASSES)),
    )

    # Dataset is class-balanced.
    # Macro averaging gives equal importance to each species.

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        labels,
        predictions,
    )

    return {
        "Model": "ResNet50",
        "Accuracy": accuracy,
        "Top-3 Accuracy": top3_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
    }


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    labels,
    predictions,
    class_names,
):
    """
    Generates and saves the ResNet50 confusion matrix.
    """

    output_directory = (
        RESULTS_DIR
        / "confusion_matrices"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    # Larger figure because we have 25 species.
    fig, ax = plt.subplots(
        figsize=(16, 16)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
    )

    ax.set_title(
        "ResNet50 Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        output_directory
        / "resnet50_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved to:\n"
        f"{output_path}"
    )


# =========================================================
# SAVE METRICS
# =========================================================

def save_metrics(metrics):
    """
    Stores model metrics in CSV format.

    Later EfficientNet, MobileNet, ViT and ensembles
    will be added to the same comparison table.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "metrics.csv"
    )

    dataframe = pd.DataFrame(
        [metrics]
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nMetrics saved to:\n"
        f"{output_path}"
    )


# =========================================================
# EVALUATE RESNET50
# =========================================================

def evaluate_resnet50():

    print(
        "\n========== RESNET50 TEST EVALUATION ==========\n"
    )

    print(
        f"Device: {DEVICE}"
    )

    (
        _,
        _,
        test_loader,
        class_names,
        _,
    ) = create_dataloaders()

    print(
        f"Test images: "
        f"{len(test_loader.dataset)}"
    )

    # ---------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------

    model = load_resnet50()

    print(
        "Loaded best ResNet50 checkpoint."
    )

    # ---------------------------------------------
    # Predictions
    # ---------------------------------------------

    (
        labels,
        predictions,
        probabilities,
    ) = get_predictions(
        model,
        test_loader,
    )

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities,
    )

    print(
        "\n========== TEST RESULTS ==========\n"
    )

    for metric_name, value in metrics.items():

        if metric_name == "Model":
            continue

        print(
            f"{metric_name:16s}: "
            f"{value:.4f}"
        )

    # ---------------------------------------------
    # Save results
    # ---------------------------------------------

    save_metrics(
        metrics
    )

    save_confusion_matrix(
        labels,
        predictions,
        class_names,
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate_resnet50()