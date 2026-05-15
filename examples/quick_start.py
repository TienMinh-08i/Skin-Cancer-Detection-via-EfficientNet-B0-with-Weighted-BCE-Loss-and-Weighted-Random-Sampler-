"""Quick start example script."""
# This example demonstrates how to use the skin cancer classification library

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import Config
from src.datasets import load_metadata, stratified_k_fold_split, SkinLesionDataset, get_transforms
from src.models import build_model, count_parameters
from src.losses import build_loss
from src.utils import seed_everything
from src.utils.metrics import compute_binary_metrics

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def example_basic_usage():
    """Basic usage example."""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)

    # 1. Setup reproducibility
    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Load config
    config = Config()
    config.model.model_name = "efficientnet_b0"
    config.training.batch_size = 32
    config.training.num_epochs = 5
    config.dataset.cv_folds = 2

    print(f"Config loaded:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")

    # 3. Build model
    model = build_model(config.model.model_name, pretrained=True).to(device)
    params = count_parameters(model)
    print(f"\nModel: {config.model.model_name}")
    print(f"Trainable parameters: {params:,}")

    # 4. Create dummy data for demonstration
    print("\n" + "="*80)
    print("Creating dummy dataset for demonstration...")
    print("="*80)

    # Create dummy dataframe
    n_samples = 100
    data = {
        "image_path": [f"/path/to/image_{i}.jpg" for i in range(n_samples)],
        "target": np.random.randint(0, 2, n_samples),
    }
    dummy_df = pd.DataFrame(data)

    # 5. Cross-validation split
    from src.datasets import stratified_k_fold_split
    folds = stratified_k_fold_split(dummy_df, n_splits=2, random_state=42)

    for fold_idx, (train_df, val_df) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_df)} samples (class dist: {train_df['target'].value_counts().to_dict()})")
        print(f"  Val: {len(val_df)} samples (class dist: {val_df['target'].value_counts().to_dict()})")

    # 6. Metrics computation example
    print("\n" + "="*80)
    print("Computing metrics example...")
    print("="*80)

    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.15, 0.25, 0.95])

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    print(f"\nMetrics at threshold=0.5:")
    for key, value in metrics.items():
        if not str(value).startswith("nan"):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def example_loss_functions():
    """Example of using different loss functions."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Loss Functions")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy class counts
    class_counts = [90, 10]  # 90 benign, 10 malignant (imbalanced)

    loss_types = ["bce", "weighted_bce", "focal", "cb_focal"]

    print(f"Class counts: {class_counts}")
    print(f"Imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1\n")

    for loss_type in loss_types:
        print(f"Building {loss_type}...")
        criterion = build_loss(
            loss_type,
            class_counts=class_counts,
            device=str(device),
        )
        print(f"  ✓ {loss_type} created")


def example_augmentations():
    """Example of data augmentations."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Data Augmentations")
    print("="*80)

    from src.configs import AugmentationConfig

    aug_config = AugmentationConfig()
    aug_dict = aug_config.__dict__

    print("Available augmentations:")
    for key, value in aug_dict.items():
        print(f"  {key}: {value}")

    # Get transforms
    train_transform = get_transforms("train", image_size=224, aug_config=aug_dict)
    val_transform = get_transforms("val", image_size=224)

    print(f"\nTrain transform (with augmentation): {len(train_transform.transforms)} operations")
    print(f"Val transform (no augmentation): {len(val_transform.transforms)} operations")


def example_threshold_optimization():
    """Example of threshold optimization."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Threshold Optimization")
    print("="*80)

    from src.visualization import optimize_thresholds, generate_threshold_report
    from pathlib import Path
    import tempfile

    # Dummy predictions
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0] * 10)
    y_prob = np.random.rand(len(y_true))

    # Optimize
    opt_thresholds = optimize_thresholds(
        y_true, y_prob,
        methods=["youden", "f1", "clinical"],
        fn_cost_ratio=1.0,
    )

    print("\nOptimized thresholds:")
    for method, (threshold, score) in opt_thresholds.items():
        print(f"  {method}: threshold={threshold:.3f}, score={score:.4f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SKIN CANCER CLASSIFICATION - QUICK START EXAMPLES")
    print("="*80)

    example_basic_usage()
    example_loss_functions()
    example_augmentations()
    example_threshold_optimization()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Prepare your ISIC 2019 dataset")
    print("2. Create a config file: python scripts/generate_config.py")
    print("3. Edit the config with your dataset paths")
    print("4. Run training: python scripts/main_train.py --config configs/your_config.yaml")
    print("5. Evaluate: python scripts/main_eval.py --checkpoint ... --test-csv ... --config ...")
    print("="*80 + "\n")
