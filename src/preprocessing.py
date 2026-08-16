from torchvision import transforms

from src.config import IMAGE_SIZE


# =========================================================
# IMAGENET NORMALIZATION
# =========================================================

# Our networks use ImageNet pretrained weights.
# Therefore, we normalize images using ImageNet statistics.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# TRAINING TRANSFORMS
# =========================================================

train_transform = transforms.Compose(
    [
        # Resize every image to the size expected by our models.
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Data augmentation.
        # Randomly flips approximately half the training images.
        transforms.RandomHorizontalFlip(p=0.5),

        # Convert PIL image into PyTorch Tensor.
        transforms.ToTensor(),

        # Normalize using ImageNet statistics.
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# =========================================================
# VALIDATION / TEST TRANSFORMS
# =========================================================

eval_transform = transforms.Compose(
    [
        # No random augmentation during evaluation.
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)