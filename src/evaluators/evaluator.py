"""Evaluation utilities."""
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from ..utils.metrics import compute_binary_metrics


def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate model on one epoch.

    Args:
        model: Neural network model.
        data_loader: DataLoader for evaluation.
        criterion: Loss function.
        device: Computation device.
        use_amp: Whether to use AMP.

    Returns:
        Tuple of (avg_loss, metrics_dict, y_true, y_prob, image_paths).
    """
    model.eval()

    losses = []
    all_targets = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Forward pass
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    logits = model(images).view(-1)
                    loss = criterion(logits, targets)
            else:
                logits = model(images).view(-1)
                loss = criterion(logits, targets)

            # Collect outputs
            probs = torch.sigmoid(logits).cpu().numpy()
            losses.append(loss.item() * images.size(0))
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())
            all_paths.extend(batch["image_path"])

    # Compute metrics
    avg_loss = float(np.sum(losses) / len(data_loader.dataset)) if len(data_loader.dataset) > 0 else 0.0
    metrics = compute_binary_metrics(
        np.array(all_targets),
        np.array(all_probs),
        threshold=0.5,
    )

    return avg_loss, metrics, np.array(all_targets), np.array(all_probs), all_paths
