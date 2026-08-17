import argparse

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
from src.models import build_model


# =========================================================
# DISPLAY NAMES
# =========================================================

MODEL_DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "efficientnet_v2_b0": "EfficientNetV2-B0",
    "mobilenet_v2": "MobileNetV2",
    "vit_b32": "ViT-B/32",
}


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_name):
    """
    Builds the selected architecture and loads its
    trained checkpoint.

    pretrained=False is intentional because our checkpoint
    already contains the complete model state.
    """

    model = build_model(
        model_name=model_name,
        freeze_backbone=True,
        pretrained=False,
    )

    checkpoint_path = (
        MODEL_DIR
        / f"{model_name}_best.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# =========================================================
# PREDICTIONS
# =========================================================

def get_predictions(
    model,
    dataloader,
    model_name,
):
    """
    Runs inference across the complete test set.

    Returns:
        true labels
        predicted labels
        probability vectors
    """

    all_labels = []
    all_predictions = []
    all_probabilities = []

    model.eval()

    with torch.no_grad():

        for images, labels in tqdm(
            dataloader,
            desc=f"Testing {model_name}",
        ):

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_labels.append(
                labels.numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    labels = np.concatenate(
        all_labels
    )

    predictions = np.concatenate(
        all_predictions
    )

    probabilities = np.concatenate(
        all_probabilities
    )

    return (
        labels,
        predictions,
        probabilities,
    )


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    model_name,
    labels,
    predictions,
    probabilities,
):

    return {
        "Model": MODEL_DISPLAY_NAMES[model_name],

        "Accuracy": accuracy_score(
            labels,
            predictions,
        ),

        "Top-3 Accuracy": top_k_accuracy_score(
            labels,
            probabilities,
            k=3,
            labels=list(range(NUM_CLASSES)),
        ),

        "Precision": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "Recall": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "F1 Score": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            labels,
            predictions,
        ),
    }


# =========================================================
# SAVE METRICS
# =========================================================

def save_metrics(metrics):
    """
    Adds or updates this model inside metrics.csv.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "metrics.csv"
    )

    new_row = pd.DataFrame(
        [metrics]
    )

    if output_path.exists():

        existing = pd.read_csv(
            output_path
        )

        # Remove old result for same model
        existing = existing[
            existing["Model"]
            != metrics["Model"]
        ]

        dataframe = pd.concat(
            [
                existing,
                new_row,
            ],
            ignore_index=True,
        )

    else:

        dataframe = new_row

    dataframe.to_csv(
        output_path,
        index=False,
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    model_name,
    labels,
    predictions,
    class_names,
):

    output_dir = (
        RESULTS_DIR
        / "confusion_matrices"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

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
        f"{MODEL_DISPLAY_NAMES[model_name]} Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# SAVE PROBABILITIES
# =========================================================

def save_predictions(
    model_name,
    labels,
    predictions,
    probabilities,
):
    """
    Saves probability outputs.

    These are important because Step 9 will combine
    probabilities from all four models to create our
    linear and geometric ensembles.
    """

    output_dir = (
        RESULTS_DIR
        / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{model_name}_predictions.npz"
    )

    np.savez_compressed(
        output_path,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )

    print(
        f"Predictions saved: {output_path}"
    )


# =========================================================
# EVALUATE
# =========================================================

def evaluate_model(
    model_name,
):

    print(
        f"\n========== "
        f"{MODEL_DISPLAY_NAMES[model_name]} "
        f"TEST EVALUATION ==========\n"
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
    ) = create_dataloaders(
        model_name
    )

    print(
        f"Test images: "
        f"{len(test_loader.dataset)}"
    )

    model = load_model(
        model_name
    )

    print(
        "Checkpoint loaded."
    )

    (
        labels,
        predictions,
        probabilities,
    ) = get_predictions(
        model,
        test_loader,
        model_name,
    )

    metrics = calculate_metrics(
        model_name,
        labels,
        predictions,
        probabilities,
    )

    print(
        "\n========== RESULTS ==========\n"
    )

    for name, value in metrics.items():

        if name == "Model":
            continue

        print(
            f"{name:16s}: {value:.4f}"
        )

    save_metrics(
        metrics
    )

    save_confusion_matrix(
        model_name,
        labels,
        predictions,
        class_names,
    )

    save_predictions(
        model_name,
        labels,
        predictions,
        probabilities,
    )

    print(
        "\nEvaluation complete."
    )


# =========================================================
# ARGUMENTS
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Evaluate bird classification model"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "efficientnet_v2_b0",
            "mobilenet_v2",
            "vit_b32",
        ],
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = parse_arguments()

    evaluate_model(
        args.model
    )