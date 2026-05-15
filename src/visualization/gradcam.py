"""Grad-CAM visualization utilities."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False


def denormalize_image(
    img_tensor: torch.Tensor,
    mean: List[float] = None,
    std: List[float] = None,
) -> np.ndarray:
    """
    Denormalize image tensor to [0, 1] range.

    Args:
        img_tensor: Image tensor (C, H, W).
        mean: ImageNet mean values.
        std: ImageNet std values.

    Returns:
        NumPy array (H, W, C) in [0, 1] range.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    img = img_tensor.detach().cpu().clone()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Find the best convolutional layer for Grad-CAM.

    Args:
        model: Neural network model.

    Returns:
        Target layer for Grad-CAM.
    """
    # Try common layer names for different architectures
    candidates = ["conv_head", "head", "classifier"]
    for name in candidates:
        if hasattr(model, name):
            return getattr(model, name)

    # Try to find the last Conv2d layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is not None:
        return last_conv

    raise ValueError("Could not find a suitable layer for Grad-CAM")


def generate_gradcam(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    transform=None,
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap for an image.

    Args:
        model: Neural network model.
        image: Input image (PIL or numpy array).
        device: Computation device.
        transform: Image transform function.

    Returns:
        Grad-CAM heatmap or None if failed.
    """
    if not HAS_GRAD_CAM:
        print("pytorch_grad_cam not installed")
        return None

    try:
        # Prepare image
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if transform is None:
            # Default transform
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        input_tensor = transform(image).unsqueeze(0).to(device)

        # Get target layer
        target_layer = get_target_layer(model)

        # Compute Grad-CAM
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]

        return grayscale_cam

    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None


def save_gradcam_visualization(
    image_path: str,
    gradcam_heatmap: np.ndarray,
    pred_prob: float,
    true_label: int,
    save_path: Path,
    mean: List[float] = None,
    std: List[float] = None,
) -> None:
    """
    Save Grad-CAM visualization.

    Args:
        image_path: Path to original image.
        gradcam_heatmap: Grad-CAM heatmap.
        pred_prob: Predicted probability.
        true_label: Ground truth label.
        save_path: Path to save visualization.
        mean: ImageNet mean values.
        std: ImageNet std values.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # Load original image
    original_img = Image.open(image_path).convert("RGB")
    original_img = original_img.resize((224, 224))
    original_arr = np.array(original_img) / 255.0

    # Apply Grad-CAM overlay
    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image
        visualization = show_cam_on_image(original_arr, gradcam_heatmap, use_rgb=True)
    except Exception as e:
        print(f"Error overlaying Grad-CAM: {e}")
        visualization = original_arr

    # Save figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(original_arr)
    axes[0].axis("off")
    axes[0].set_title(f"Original\nTrue: {['Benign', 'Malignant'][true_label]}", fontsize=10)

    axes[1].imshow(visualization)
    axes[1].axis("off")
    axes[1].set_title(f"Grad-CAM\nPred: {pred_prob:.3f}", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
