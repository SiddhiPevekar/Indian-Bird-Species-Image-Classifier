from torchvision import transforms
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V2_Weights,
    ViT_B_32_Weights,
)

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def get_transforms(model_name):
    """
    Returns:
        training transform
        validation/test transform
    """

    model_name = model_name.lower()

    # =====================================================
    # RESNET50
    # =====================================================

    if model_name == "resnet50":

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # =====================================================
    # MOBILENET V2
    # =====================================================

    elif model_name == "mobilenet_v2":

        weights = MobileNet_V2_Weights.DEFAULT
        eval_transform = weights.transforms()

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # =====================================================
    # VIT-B/32
    # =====================================================

    elif model_name == "vit_b32":

        weights = ViT_B_32_Weights.DEFAULT
        eval_transform = weights.transforms()

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # =====================================================
    # EFFICIENTNET V2 B0
    # =====================================================

    elif model_name == "efficientnet_v2_b0":

        # We only need model configuration here,
        # therefore pretrained=False avoids downloading weights.
        temp_model = timm.create_model(
            "tf_efficientnetv2_b0.in1k",
            pretrained=False,
        )

        data_config = resolve_data_config(
            temp_model.pretrained_cfg,
            model=temp_model,
        )

        # Deterministic transform for validation/test.
        eval_transform = create_transform(
            **data_config,
            is_training=False,
        )

        # Training version adds augmentation.
        train_transform = create_transform(
            **data_config,
            is_training=True,
            hflip=0.5,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return train_transform, eval_transform