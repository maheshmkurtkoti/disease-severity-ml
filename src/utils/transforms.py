"""
X-ray image transforms and augmentation pipeline.
Uses albumentations — significantly better than torchvision for medical images.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Training augmentations — conservative for medical images.
    CLAHE is especially useful for X-rays to enhance contrast.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation/test transforms — no augmentation, just resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """Transforms for inference/demo — same as val."""
    return get_val_transforms(image_size)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy array, ensuring RGB."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)
