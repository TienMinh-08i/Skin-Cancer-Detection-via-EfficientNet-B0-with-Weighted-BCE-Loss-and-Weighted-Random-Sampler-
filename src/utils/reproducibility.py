"""Reproducibility utilities for seeding and deterministic behavior."""
import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, cudnn_deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
        cudnn_deterministic: If True, set cuDNN to deterministic mode.
                            May reduce performance but ensures reproducibility.
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
