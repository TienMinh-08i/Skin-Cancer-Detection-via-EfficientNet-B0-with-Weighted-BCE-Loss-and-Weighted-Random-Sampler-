"""Main evaluation script for test set and Grad-CAM generation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configs import Config
from src.datasets import SkinLesionDataset, get_transforms
from src.models import build_model
from src.utils import setup_logger
from src.utils.metrics import compute_binary_metrics
from src.visualization import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    optimize_thresholds,
    generate_threshold_report,
    save_error_analysis,
    generate_gradcam,
    save_gradcam_visualization,
    denormalize_image,
)


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_df: pd.DataFrame,
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Evaluate a saved checkpoint on test set."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Recreate model
    model = build_model(
        config.model.model_name,
        pretrained=False,
        cache_dir=config.models_cache_dir,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create test loader
    test_transform = get_transforms("val", config.dataset.image_size)
    test_ds = SkinLesionDataset(test_df, transform=test_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        shuffle=False,
    )

    # Evaluate
    all_targets = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_targets.extend(batch["target"].numpy().tolist())
            all_probs.extend(probs.tolist())
            all_paths.extend(batch["image_path"])

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)

    # Compute metrics at default threshold
    metrics = compute_binary_metrics(y_true, y_prob, threshold=config.evaluation.threshold)

    print("\nTest Metrics (threshold={:.2f}):".format(config.evaluation.threshold))
    for key in ["roc_auc", "auprc", "f1", "sensitivity", "specificity", "accuracy"]:
        print(f"  {key}: {metrics.get(key, np.nan):.4f}")

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "image_path": all_paths,
        "target": y_true,
        "probability": y_prob,
        "prediction": (y_prob >= config.evaluation.threshold).astype(int),
    })
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Plot curves
    plot_roc_curve(y_true, y_prob, "Test ROC Curve", output_dir / "roc_curve.png")
    plot_pr_curve(y_true, y_prob, "Test PR Curve", output_dir / "pr_curve.png")
    y_pred = (y_prob >= config.evaluation.threshold).astype(int)
    plot_confusion_matrix(y_true, y_pred, "Test Confusion Matrix", output_dir / "confusion_matrix.png")

    # Threshold optimization
    if config.threshold_optimization.enabled:
        opt_thresholds = optimize_thresholds(
            y_true, y_prob,
            methods=config.threshold_optimization.methods,
            fn_cost_ratio=config.threshold_optimization.clinical_fn_cost_ratio,
        )
        print("\nOptimal Thresholds:")
        for method, (threshold, score) in opt_thresholds.items():
            print(f"  {method}: {threshold:.3f} (score={score:.4f})")

        threshold_df = generate_threshold_report(
            y_true, y_prob,
            np.linspace(0, 1, 101),
            output_dir / "threshold_report.csv",
        )

    # Error analysis
    if config.evaluation.compute_error_analysis:
        save_error_analysis(
            y_true, y_prob, all_paths,
            output_dir / "error_analysis",
            threshold=config.evaluation.threshold,
            n_fp=config.evaluation.error_analysis_n_samples,
            n_fn=config.evaluation.error_analysis_n_samples,
        )

    # Grad-CAM
    if config.evaluation.compute_gradcam:
        gradcam_dir = output_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)

        # Select images to visualize
        # Prioritize malignant (class 1) as they're clinically important
        malignant_idx = np.where(y_true == 1)[0]
        benign_idx = np.where(y_true == 0)[0]

        n_total = min(config.evaluation.error_analysis_n_samples * 2, len(malignant_idx) + len(benign_idx))
        n_mal = min(len(malignant_idx), n_total // 2)
        n_ben = min(len(benign_idx), n_total - n_mal)

        if len(malignant_idx) > 0:
            mal_sample_idx = np.random.choice(malignant_idx, size=min(n_mal, len(malignant_idx)), replace=False)
        else:
            mal_sample_idx = []

        if len(benign_idx) > 0:
            ben_sample_idx = np.random.choice(benign_idx, size=min(n_ben, len(benign_idx)), replace=False)
        else:
            ben_sample_idx = []

        sample_idx = np.concatenate([mal_sample_idx, ben_sample_idx])

        print(f"\nGenerating Grad-CAM visualizations for {len(sample_idx)} samples...")

        test_transform = get_transforms("val", config.dataset.image_size)

        for i, idx in enumerate(sample_idx):
            image_path = all_paths[idx]
            pred_prob = y_prob[idx]
            true_label = y_true[idx]

            try:
                # Generate Grad-CAM
                image = Image.open(image_path).convert("RGB")
                input_tensor = test_transform(image).unsqueeze(0).to(device)

                heatmap = generate_gradcam(model, image, device, test_transform)
                if heatmap is not None:
                    # Save visualization
                    save_path = gradcam_dir / f"gradcam_{i:03d}_true{true_label}_prob{pred_prob:.3f}.png"
                    save_gradcam_visualization(
                        image_path, heatmap,
                        pred_prob, true_label,
                        save_path,
                    )
            except Exception as e:
                print(f"Error generating Grad-CAM for {image_path}: {e}")

        print(f"Grad-CAM visualizations saved to {gradcam_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate skin cancer classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = Config.from_yaml(Path(args.config))
    config.device = args.device

    # Load test data
    test_df = pd.read_csv(args.test_csv)
    print(f"Loaded test set: {len(test_df)} samples")
    print(f"Class distribution: {test_df['target'].value_counts().to_dict()}")

    # Evaluate
    metrics = evaluate_checkpoint(
        Path(args.checkpoint),
        test_df,
        config,
        device,
        output_dir,
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
