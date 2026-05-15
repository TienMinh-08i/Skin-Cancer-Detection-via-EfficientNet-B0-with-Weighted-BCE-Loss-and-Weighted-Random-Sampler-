"""Model factory for creating neural network architectures."""
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def build_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = 1,
    cache_dir: Optional[Path] = None,
) -> nn.Module:
    """
    Build a classification model.

    Args:
        model_name: Model identifier (e.g., "efficientnet_b0", "convnext_tiny").
        pretrained: Whether to use pretrained ImageNet weights.
        num_classes: Number of output classes (1 for binary).
        cache_dir: Optional directory for cached pretrained models (for offline usage).

    Returns:
        PyTorch model.
    """
    if not HAS_TIMM:
        raise ImportError("timm not installed. Install with: pip install timm")

    # Set cache directories for offline usage (HPC)
    if cache_dir is not None:
        cache_dir = Path(cache_dir).resolve()
        os.environ["TORCH_HOME"] = str(cache_dir)
        os.environ["TIMM_HOME"] = str(cache_dir)

    # Map common names to timm names
    model_map = {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b3": "efficientnet_b3",
        "convnext_tiny": "convnext_tiny",
        "vit_tiny": "vit_tiny_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model_name={model_name}. Available: {list(model_map.keys())}")

    timm_name = model_map[model_name]
    cache_info = f", cache={cache_dir}" if cache_dir else ""
    print(f"Building model: {model_name} (timm: {timm_name}), pretrained={pretrained}{cache_info}")

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
