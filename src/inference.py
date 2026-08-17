import numpy as np
import torch
from PIL import Image

from src.config import (
    DEVICE,
    MODEL_DIR,
    SOURCE_TRAIN_DIR,
)

from src.models import build_model
from src.preprocessing import get_transforms

from torchvision.datasets import ImageFolder


# =========================================================
# MODEL NAMES
# =========================================================

MODEL_NAMES = [
    "resnet50",
    "efficientnet_v2_b0",
    "mobilenet_v2",
    "vit_b32",
]


DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "efficientnet_v2_b0": "EfficientNetV2-B0",
    "mobilenet_v2": "MobileNetV2",
    "vit_b32": "ViT-B/32",
}


# =========================================================
# CLASS NAMES
# =========================================================

def get_class_names():

    dataset = ImageFolder(
        SOURCE_TRAIN_DIR
    )

    return dataset.classes


# =========================================================
# LOAD ONE MODEL
# =========================================================

def load_trained_model(model_name):

    model = build_model(
        model_name=model_name,
        freeze_backbone=True,
        pretrained=False,
    )

    checkpoint_path = (
        MODEL_DIR
        / f"{model_name}_best.pth"
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
# LOAD ALL MODELS
# =========================================================

def load_all_models():

    models = {}

    for model_name in MODEL_NAMES:

        print(
            f"Loading {DISPLAY_NAMES[model_name]}..."
        )

        models[model_name] = (
            load_trained_model(
                model_name
            )
        )

    return models


# =========================================================
# PREPROCESS SINGLE IMAGE
# =========================================================

def preprocess_image(
    image,
    model_name,
):

    # Every architecture gets the same preprocessing
    # that was used during its evaluation.

    _, eval_transform = (
        get_transforms(
            model_name
        )
    )

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = eval_transform(
        image
    )

    # Add batch dimension:
    #
    # [3, H, W]
    # becomes
    # [1, 3, H, W]

    tensor = tensor.unsqueeze(0)

    return tensor


# =========================================================
# PREDICT WITH ONE MODEL
# =========================================================

def predict_model(
    model,
    image,
    model_name,
):

    tensor = preprocess_image(
        image,
        model_name,
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    probabilities = (
        probabilities
        .cpu()
        .numpy()[0]
    )

    return probabilities


# =========================================================
# ENSEMBLES
# =========================================================

def linear_ensemble(
    probability_arrays,
):

    stacked = np.stack(
        probability_arrays,
        axis=0,
    )

    return np.mean(
        stacked,
        axis=0,
    )


def geometric_ensemble(
    probability_arrays,
):

    stacked = np.stack(
        probability_arrays,
        axis=0,
    )

    epsilon = 1e-12

    log_probabilities = np.log(
        stacked + epsilon
    )

    ensemble = np.exp(
        np.mean(
            log_probabilities,
            axis=0,
        )
    )

    ensemble /= ensemble.sum()

    return ensemble


# =========================================================
# GET TOP-K
# =========================================================

def get_top_predictions(
    probabilities,
    class_names,
    k=3,
):

    top_indices = np.argsort(
        probabilities
    )[::-1][:k]

    results = []

    for index in top_indices:

        results.append(
            {
                "class": class_names[index],
                "confidence": float(
                    probabilities[index]
                ),
            }
        )

    return results


# =========================================================
# COMPLETE PREDICTION PIPELINE
# =========================================================

def predict_image(
    image,
    models,
):

    class_names = (
        get_class_names()
    )

    model_probabilities = {}

    # -----------------------------------------------------
    # Individual models
    # -----------------------------------------------------

    for model_name in MODEL_NAMES:

        probabilities = predict_model(
            models[model_name],
            image,
            model_name,
        )

        model_probabilities[
            model_name
        ] = probabilities

    probability_list = [
        model_probabilities[name]
        for name in MODEL_NAMES
    ]

    # -----------------------------------------------------
    # Ensembles
    # -----------------------------------------------------

    linear_probabilities = (
        linear_ensemble(
            probability_list
        )
    )

    geometric_probabilities = (
        geometric_ensemble(
            probability_list
        )
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    results = {}

    for model_name in MODEL_NAMES:

        results[
            DISPLAY_NAMES[model_name]
        ] = get_top_predictions(
            model_probabilities[
                model_name
            ],
            class_names,
            k=3,
        )

    results[
        "Linear Mean Ensemble"
    ] = get_top_predictions(
        linear_probabilities,
        class_names,
        k=3,
    )

    results[
        "Geometric Mean Ensemble"
    ] = get_top_predictions(
        geometric_probabilities,
        class_names,
        k=3,
    )

    return results


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        f"Device: {DEVICE}"
    )

    models = load_all_models()

    print(
        "\nAll four models loaded successfully."
    )