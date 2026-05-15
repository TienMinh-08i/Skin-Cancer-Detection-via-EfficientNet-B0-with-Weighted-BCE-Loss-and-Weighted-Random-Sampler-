"""Class imbalance handling methods."""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, WeightedRandomSampler as TorchWeightedRandomSampler

try:
    from imblearn.over_sampling import SMOTE as IMBLEARN_SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

import pandas as pd


def create_weighted_sampler(df: pd.DataFrame) -> TorchWeightedRandomSampler:
    """
    Create WeightedRandomSampler for class imbalance.

    Args:
        df: DataFrame with 'target' column.

    Returns:
        PyTorch WeightedRandomSampler.
    """
    targets = df["target"].values.astype(int)
    class_counts = np.bincount(targets, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]

    sampler = TorchWeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def apply_smote(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling.

    Args:
        X: Feature array (flattened image features).
        y: Target array.
        k_neighbors: Number of neighbors for SMOTE.

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    if not HAS_IMBLEARN:
        raise ImportError("imblearn not installed. Install with: pip install imbalanced-learn")

    smote = IMBLEARN_SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Mixup augmentation to a batch.

    Args:
        images: Batch of images (B, C, H, W).
        targets: Batch of targets (B,).
        alpha: Beta distribution parameter.

    Returns:
        Tuple of (mixed_images, mixed_targets).
    """
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return mixed_images, mixed_targets


def cutmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutMix augmentation to a batch.

    Args:
        images: Batch of images (B, C, H, W).
        targets: Batch of targets (B,).
        alpha: Beta distribution parameter.

    Returns:
        Tuple of (mixed_images, mixed_targets).
    """
    batch_size, channels, height, width = images.size()
    index = torch.randperm(batch_size)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Sample random box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)

    cx = np.random.randint(0, width)
    cy = np.random.randint(0, height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    # Apply cutmix
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda for actual mixed area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (height * width))

    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return images, mixed_targets


class MixupCollator:
    """Collate function that applies Mixup."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])

        images, targets = mixup_batch(images, targets, self.alpha)

        return {
            "image": images,
            "target": targets,
            "image_path": [item["image_path"] for item in batch],
        }


class CutMixCollator:
    """Collate function that applies CutMix."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])

        images, targets = cutmix_batch(images, targets, self.alpha)

        return {
            "image": images,
            "target": targets,
            "image_path": [item["image_path"] for item in batch],
        }
