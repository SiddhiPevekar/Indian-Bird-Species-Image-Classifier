import torch.nn as nn
import timm

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    mobilenet_v2,
    MobileNet_V2_Weights,
    vit_b_32,
    ViT_B_32_Weights,
)

from src.config import NUM_CLASSES


# =========================================================
# FREEZE HELPER
# =========================================================

def freeze_model(model):
    """
    Freeze all parameters in a pretrained model.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False


# =========================================================
# RESNET50
# =========================================================

def build_resnet50(
    num_classes=NUM_CLASSES,
    freeze_backbone=True,
    pretrained=True,
):

    weights = (
        ResNet50_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet50(weights=weights)

    if freeze_backbone:
        freeze_model(model)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


# =========================================================
# MOBILENET V2
# =========================================================

def build_mobilenet_v2(
    num_classes=NUM_CLASSES,
    freeze_backbone=True,
    pretrained=True,
):

    weights = (
        MobileNet_V2_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v2(weights=weights)

    if freeze_backbone:
        freeze_model(model)

    # Original MobileNet classifier:
    #
    # Dropout
    # Linear(1280 -> 1000)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


# =========================================================
# VISION TRANSFORMER ViT-B/32
# =========================================================

def build_vit_b32(
    num_classes=NUM_CLASSES,
    freeze_backbone=True,
    pretrained=True,
):

    weights = (
        ViT_B_32_Weights.DEFAULT
        if pretrained
        else None
    )

    model = vit_b_32(weights=weights)

    if freeze_backbone:
        freeze_model(model)

    in_features = (
        model.heads.head.in_features
    )

    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


# =========================================================
# EFFICIENTNET V2 B0
# =========================================================

def build_efficientnet_v2_b0(
    num_classes=NUM_CLASSES,
    freeze_backbone=True,
    pretrained=True,
):

    model = timm.create_model(
        "tf_efficientnetv2_b0.in1k",
        pretrained=pretrained,
    )

    if freeze_backbone:
        freeze_model(model)

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


# =========================================================
# GENERIC MODEL BUILDER
# =========================================================

def build_model(
    model_name,
    num_classes=NUM_CLASSES,
    freeze_backbone=True,
    pretrained=True,
):

    model_name = model_name.lower()

    if model_name == "resnet50":

        return build_resnet50(
            num_classes,
            freeze_backbone,
            pretrained,
        )

    elif model_name == "mobilenet_v2":

        return build_mobilenet_v2(
            num_classes,
            freeze_backbone,
            pretrained,
        )

    elif model_name == "efficientnet_v2_b0":

        return build_efficientnet_v2_b0(
            num_classes,
            freeze_backbone,
            pretrained,
        )

    elif model_name == "vit_b32":

        return build_vit_b32(
            num_classes,
            freeze_backbone,
            pretrained,
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# =========================================================
# PARAMETER COUNTER
# =========================================================

def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    model_names = [
        "resnet50",
        "efficientnet_v2_b0",
        "mobilenet_v2",
        "vit_b32",
    ]

    # pretrained=False avoids downloading every model
    # just for this architecture test.

    for model_name in model_names:

        model = build_model(
            model_name,
            pretrained=False,
        )

        total, trainable = (
            count_parameters(model)
        )

        print(
            f"\n{model_name}"
        )

        print(
            f"Total parameters: "
            f"{total:,}"
        )

        print(
            f"Trainable parameters: "
            f"{trainable:,}"
        )