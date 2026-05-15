"""PyTorch Dataset classes and transforms."""
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224, aug_config: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Get data augmentation transforms.

    Args:
        split: "train" or "val"/"test".
        image_size: Image size for resizing.
        aug_config: Augmentation configuration dict.

    Returns:
        torchvision transforms.Compose instance.
    """
    if aug_config is None:
        aug_config = {}

    if split == "train":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=aug_config.get("horizontal_flip_p", 0.5)),
            T.RandomVerticalFlip(p=aug_config.get("vertical_flip_p", 0.5)),
            T.RandomRotation(degrees=aug_config.get("rotation_degrees", 25.0)),
            T.ColorJitter(
                brightness=aug_config.get("color_jitter_brightness", 0.15),
                contrast=aug_config.get("color_jitter_contrast", 0.15),
                saturation=aug_config.get("color_jitter_saturation", 0.1),
                hue=aug_config.get("color_jitter_hue", 0.02),
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # val / test
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SkinLesionDataset(Dataset):
    """Binary skin lesion classification dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            df: DataFrame with 'image_path' and 'target' columns.
            transform: Optional image transforms.
        """
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        target = float(row["target"])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.float32),
            "image_path": str(image_path),
        }
