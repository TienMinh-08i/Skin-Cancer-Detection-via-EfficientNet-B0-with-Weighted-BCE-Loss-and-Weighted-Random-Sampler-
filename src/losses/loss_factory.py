"""Loss functions for class imbalance."""
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.

    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor for positive class.
            gamma: Focusing parameter.
            reduction: "mean", "sum", or "none".
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Binary logits (B,).
            targets: Binary targets (B,).

        Returns:
            Loss value.
        """
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Compute pt (probability of correct class)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Compute alpha_t
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha, dtype=torch.float32),
            torch.full_like(targets, 1.0 - self.alpha, dtype=torch.float32),
        )

        # Focal loss
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss using effective number of samples.

    Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples"
    """

    def __init__(
        self,
        class_counts: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            class_counts: [num_negative, num_positive].
            beta: Re-weighting parameter (0 < beta < 1).
            gamma: Focal loss gamma parameter.
            reduction: "mean", "sum", or "none".
        """
        super().__init__()
        if len(class_counts) != 2:
            raise ValueError("Binary classification requires exactly 2 class counts.")

        counts = np.array(class_counts, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
        weights = weights / weights.sum() * 2.0

        self.register_buffer(
            "class_weights",
            torch.tensor(weights, dtype=torch.float32),
        )
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Binary logits (B,).
            targets: Binary targets (B,).

        Returns:
            Loss value.
        """
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Compute pt
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Get class weights
        target_long = targets.long()
        alpha_t = self.class_weights[target_long]

        # Class-balanced focal loss
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_loss(
    loss_type: str,
    class_counts: List[int],
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    cb_beta: float = 0.9999,
    device: str = "cuda",
) -> nn.Module:
    """
    Build loss function based on config.

    Args:
        loss_type: "bce", "weighted_bce", "focal", or "cb_focal".
        class_counts: [num_negative, num_positive].
        focal_alpha: Alpha for focal loss.
        focal_gamma: Gamma for focal loss.
        cb_beta: Beta for class-balanced focal loss.
        device: Device for pos_weight tensor.

    Returns:
        Loss function module.
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    elif loss_type == "weighted_bce":
        num_neg, num_pos = class_counts[0], class_counts[1]
        pos_weight = num_neg / max(num_pos, 1)
        print(f"Using weighted BCE with pos_weight={pos_weight:.4f}")
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    elif loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    elif loss_type == "cb_focal":
        return ClassBalancedFocalLoss(
            class_counts=class_counts,
            beta=cb_beta,
            gamma=focal_gamma,
        ).to(device)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
