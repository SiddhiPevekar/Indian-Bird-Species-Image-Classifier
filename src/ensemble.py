import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from torchvision.datasets import ImageFolder

from src.config import (
    RESULTS_DIR,
    SOURCE_TRAIN_DIR,
    NUM_CLASSES,
)


# =========================================================
# MODEL PREDICTION FILES
# =========================================================

MODEL_NAMES = [
    "resnet50",
    "efficientnet_v2_b0",
    "mobilenet_v2",
    "vit_b32",
]


# =========================================================
# LOAD PREDICTIONS
# =========================================================

def load_all_predictions():
    """
    Load probability outputs from all four trained models.

    Also verifies that every model was evaluated on
    exactly the same test samples in the same order.
    """

    prediction_dir = (
        RESULTS_DIR / "predictions"
    )

    probability_arrays = []

    reference_labels = None

    for model_name in MODEL_NAMES:

        path = (
            prediction_dir
            / f"{model_name}_predictions.npz"
        )

        data = np.load(path)

        labels = data["labels"]
        probabilities = data["probabilities"]

        print(
            f"{model_name:22s} "
            f"{probabilities.shape}"
        )

        # First model becomes our reference.
        if reference_labels is None:

            reference_labels = labels

        else:

            # Very important:
            # Ensemble predictions are valid only if every
            # probability row corresponds to the same image.
            if not np.array_equal(
                reference_labels,
                labels,
            ):

                raise ValueError(
                    f"Label ordering mismatch for {model_name}"
                )

        probability_arrays.append(
            probabilities
        )

    # Shape:
    #
    # 4 models × 6000 images × 25 classes

    stacked_probabilities = np.stack(
        probability_arrays,
        axis=0,
    )

    return (
        reference_labels,
        stacked_probabilities,
    )


# =========================================================
# LINEAR MEAN ENSEMBLE
# =========================================================

def linear_mean_ensemble(
    probabilities,
):
    """
    Arithmetic mean of the four model probability vectors.
    """

    return np.mean(
        probabilities,
        axis=0,
    )


# =========================================================
# GEOMETRIC MEAN ENSEMBLE
# =========================================================

def geometric_mean_ensemble(
    probabilities,
):
    """
    Geometric mean of probabilities.

    Log-space computation is used for numerical stability.
    """

    epsilon = 1e-12

    log_probabilities = np.log(
        probabilities + epsilon
    )

    mean_log_probabilities = np.mean(
        log_probabilities,
        axis=0,
    )

    geometric_probabilities = np.exp(
        mean_log_probabilities
    )

    # Normalize each prediction vector so probabilities
    # sum to 1.

    geometric_probabilities /= (
        geometric_probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    return geometric_probabilities


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    model_name,
    labels,
    probabilities,
):

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "Model": model_name,

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
# SAVE CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    name,
    labels,
    probabilities,
    class_names,
):

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    output_dir = (
        RESULTS_DIR
        / "confusion_matrices"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        f"{name} Confusion Matrix"
    )

    plt.tight_layout()

    filename = (
        name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    output_path = (
        output_dir
        / f"{filename}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved confusion matrix: {output_path}"
    )


# =========================================================
# SAVE ENSEMBLE PROBABILITIES
# =========================================================

def save_ensemble_predictions(
    name,
    labels,
    probabilities,
):

    output_dir = (
        RESULTS_DIR
        / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    output_path = (
        output_dir
        / f"{filename}_predictions.npz"
    )

    np.savez_compressed(
        output_path,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )


# =========================================================
# UPDATE METRICS CSV
# =========================================================

def update_metrics_csv(
    metrics_list,
):

    output_path = (
        RESULTS_DIR / "metrics.csv"
    )

    new_results = pd.DataFrame(
        metrics_list
    )

    if output_path.exists():

        existing = pd.read_csv(
            output_path
        )

        ensemble_names = (
            new_results["Model"].tolist()
        )

        # Remove old ensemble rows if this script
        # is executed multiple times.
        existing = existing[
            ~existing["Model"].isin(
                ensemble_names
            )
        ]

        results = pd.concat(
            [
                existing,
                new_results,
            ],
            ignore_index=True,
        )

    else:

        results = new_results

    results.to_csv(
        output_path,
        index=False,
    )

    return results


# =========================================================
# MAIN ENSEMBLE PIPELINE
# =========================================================

def run_ensembles():

    print(
        "\n========== ENSEMBLE EVALUATION ==========\n"
    )

    labels, probabilities = (
        load_all_predictions()
    )

    print(
        "\nAll model labels match."
    )

    print(
        f"Combined tensor shape: "
        f"{probabilities.shape}"
    )

    # -----------------------------------------------------
    # Build ensembles
    # -----------------------------------------------------

    linear_probabilities = (
        linear_mean_ensemble(
            probabilities
        )
    )

    geometric_probabilities = (
        geometric_mean_ensemble(
            probabilities
        )
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    linear_metrics = calculate_metrics(
        "Linear Mean Ensemble",
        labels,
        linear_probabilities,
    )

    geometric_metrics = calculate_metrics(
        "Geometric Mean Ensemble",
        labels,
        geometric_probabilities,
    )

    metrics_list = [
        linear_metrics,
        geometric_metrics,
    ]

    print(
        "\n========== ENSEMBLE RESULTS ==========\n"
    )

    for metrics in metrics_list:

        print(
            metrics["Model"]
        )

        print(
            f"Accuracy        : "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"Top-3 Accuracy  : "
            f"{metrics['Top-3 Accuracy']:.4f}"
        )

        print(
            f"Precision       : "
            f"{metrics['Precision']:.4f}"
        )

        print(
            f"Recall          : "
            f"{metrics['Recall']:.4f}"
        )

        print(
            f"F1 Score        : "
            f"{metrics['F1 Score']:.4f}"
        )

        print(
            f"MCC             : "
            f"{metrics['MCC']:.4f}"
        )

        print()

    # -----------------------------------------------------
    # Class names
    # -----------------------------------------------------

    dataset = ImageFolder(
        SOURCE_TRAIN_DIR
    )

    class_names = dataset.classes

    # -----------------------------------------------------
    # Confusion matrices
    # -----------------------------------------------------

    save_confusion_matrix(
        "Linear Mean Ensemble",
        labels,
        linear_probabilities,
        class_names,
    )

    save_confusion_matrix(
        "Geometric Mean Ensemble",
        labels,
        geometric_probabilities,
        class_names,
    )

    # -----------------------------------------------------
    # Save ensemble probability vectors
    # -----------------------------------------------------

    save_ensemble_predictions(
        "Linear Mean Ensemble",
        labels,
        linear_probabilities,
    )

    save_ensemble_predictions(
        "Geometric Mean Ensemble",
        labels,
        geometric_probabilities,
    )

    # -----------------------------------------------------
    # Update comparison CSV
    # -----------------------------------------------------

    final_results = update_metrics_csv(
        metrics_list
    )

    print(
        "\n========== COMPLETE MODEL COMPARISON ==========\n"
    )

    print(
        final_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    run_ensembles()