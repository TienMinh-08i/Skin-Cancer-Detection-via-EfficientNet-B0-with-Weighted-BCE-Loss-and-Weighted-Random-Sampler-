"""Main training script for cross-validation experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configs import Config
from src.datasets import (
    SkinLesionDataset,
    get_transforms,
    load_metadata,
    stratified_k_fold_split,
    create_weighted_sampler,
    MixupCollator,
    CutMixCollator,
)
from src.models import build_model, count_parameters
from src.losses import build_loss
from src.trainers import Trainer
from src.evaluators import evaluate_epoch
from src.utils import seed_everything, setup_logger
from src.utils.metrics import compute_binary_metrics
from src.visualization import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_training_curves,
    optimize_thresholds,
    generate_threshold_report,
    save_error_analysis,
)


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
) -> tuple:
    """Create data loaders for training/validation/testing."""
    train_transform = get_transforms("train", config.dataset.image_size, config.augmentation.__dict__)
    val_transform = get_transforms("val", config.dataset.image_size)

    train_ds = SkinLesionDataset(train_df, transform=train_transform)
    val_ds = SkinLesionDataset(val_df, transform=val_transform)
    test_ds = SkinLesionDataset(test_df, transform=val_transform)

    # Imbalance methods
    sampler = None
    collate_fn = None

    if config.imbalance.weighted_sampler_enabled:
        sampler = create_weighted_sampler(train_df)
        print("Using WeightedRandomSampler")

    if config.imbalance.mixup_enabled:
        collate_fn = MixupCollator(alpha=config.imbalance.mixup_alpha)
        print("Using Mixup augmentation")

    if config.imbalance.cutmix_enabled:
        collate_fn = CutMixCollator(alpha=config.imbalance.cutmix_alpha)
        print("Using CutMix augmentation")

    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def train_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
    device: torch.device,
    results_dir: Path,
) -> dict:
    """Train model for one fold."""
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx + 1}/{config.dataset.cv_folds}")
    print(f"{'='*80}")

    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_df, val_df, test_df, config)

    # Build model
    model = build_model(
        config.model.model_name,
        pretrained=config.model.pretrained,
        cache_dir=config.models_cache_dir,
    ).to(device)
    print(f"Model: {config.model.model_name}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Build loss
    targets = train_df["target"].values.astype(int)
    class_counts = np.bincount(targets, minlength=2)
    print(f"Class counts: {class_counts}")

    criterion = build_loss(
        config.loss.loss_type,
        class_counts=list(class_counts),
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        cb_beta=config.loss.cb_focal_beta,
        device=str(device),
    ).to(device)

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    # Build scheduler
    scheduler = None
    if config.scheduler.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
        )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_dir=fold_dir / "checkpoints",
        log_dir=fold_dir / "logs",
        experiment_name=f"fold_{fold_idx}",
    )

    train_result = trainer.train()

    # Load best model and evaluate on test set
    trainer.load_best_checkpoint()
    test_loss, test_metrics, y_true, y_prob, image_paths = evaluate_epoch(
        trainer.model,
        test_loader,
        criterion,
        device,
        use_amp=config.training.use_amp,
    )

    # Save test predictions
    pred_df = pd.DataFrame({
        "image_path": image_paths,
        "target": y_true.astype(int),
        "probability": y_prob.astype(float),
        "prediction": (y_prob >= config.evaluation.threshold).astype(int),
    })
    pred_path = fold_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save visualizations
    fig_dir = fold_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_roc_curve(
        y_true, y_prob,
        f"Fold {fold_idx + 1} - Test ROC",
        fig_dir / "roc_curve.png",
    )
    plot_pr_curve(
        y_true, y_prob,
        f"Fold {fold_idx + 1} - Test PR Curve",
        fig_dir / "pr_curve.png",
    )
    y_pred = (y_prob >= config.evaluation.threshold).astype(int)
    plot_confusion_matrix(
        y_true, y_pred,
        f"Fold {fold_idx + 1} - Confusion Matrix",
        fig_dir / "confusion_matrix.png",
    )
    plot_training_curves(
        train_result["log_df"],
        f"Fold {fold_idx + 1} - Training Curves",
        fig_dir / "training_curves.png",
    )

    # Threshold optimization
    if config.threshold_optimization.enabled:
        opt_thresholds = optimize_thresholds(
            y_true, y_prob,
            methods=config.threshold_optimization.methods,
            fn_cost_ratio=config.threshold_optimization.clinical_fn_cost_ratio,
        )
        threshold_df = generate_threshold_report(
            y_true, y_prob,
            np.linspace(0, 1, 101),
            fig_dir / "threshold_report.csv",
        )

    # Error analysis
    if config.evaluation.compute_error_analysis:
        save_error_analysis(
            y_true, y_prob, image_paths,
            fold_dir / "error_analysis",
            threshold=config.evaluation.threshold,
            n_fp=config.evaluation.error_analysis_n_samples,
            n_fn=config.evaluation.error_analysis_n_samples,
        )

    # Compile fold results
    fold_result = {
        "fold": fold_idx,
        "test_loss": test_loss,
        "best_epoch": trainer.best_epoch,
        "checkpoint_path": str(fold_dir / "checkpoints" / f"fold_{fold_idx}_best.pt"),
    }
    fold_result.update(test_metrics)

    print(f"Fold {fold_idx + 1} Results:")
    for key in ["roc_auc", "auprc", "f1", "sensitivity", "specificity"]:
        print(f"  {key}: {test_metrics.get(key, np.nan):.4f}")

    return fold_result


def main():
    parser = argparse.ArgumentParser(description="Train skin cancer classifier with cross-validation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    print(f"Loading config from {config_path}")
    config = Config.from_yaml(config_path)
    config.device = args.device

    # Setup reproducibility
    seed_everything(config.seed, cudnn_deterministic=False)
    device = torch.device(config.device)
    print(f"Device: {device}")

    # Setup output directories
    config.outputs.root_dir = Path(config.outputs.root_dir)
    for attr in ["checkpoints_dir", "logs_dir", "figures_dir", "results_dir"]:
        path = getattr(config.outputs, attr)
        if not path.is_absolute():
            path = config.outputs.root_dir / path
        setattr(config.outputs, attr, path)
        getattr(config.outputs, attr).mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from {config.dataset.csv_path}")
    df = load_metadata(
        csv_path=config.dataset.csv_path,
        image_dir=config.dataset.image_dir,
        target_col=config.dataset.target_col,
        image_id_cols=config.dataset.image_id_cols,
    )

    # Subsample if needed
    if config.dataset.max_samples is not None and len(df) > config.dataset.max_samples:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df,
            train_size=config.dataset.max_samples,
            stratify=df["target"],
            random_state=config.seed,
        )
        df = df.reset_index(drop=True)
        print(f"Subsampled to {len(df)} samples")

    # Create folds
    print(f"\nCreating {config.dataset.cv_folds}-fold stratified cross-validation")
    folds = stratified_k_fold_split(
        df,
        n_splits=config.dataset.cv_folds,
        random_state=config.seed,
    )

    # Train each fold
    fold_results = []
    results_dir = config.outputs.results_dir / "cv_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_df, val_df) in enumerate(folds):
        # Split val into val+test
        from sklearn.model_selection import train_test_split
        val_test_df = val_df.copy()
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=0.5,
            stratify=val_test_df["target"],
            random_state=config.seed + fold_idx,
        )
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        try:
            fold_result = train_fold(
                fold_idx, train_df, val_df, test_df,
                config, device, results_dir,
            )
            fold_results.append(fold_result)
        except Exception as e:
            print(f"Error training fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(results_dir / "cv_results_summary.csv", index=False)

        print(f"\n{'='*80}")
        print("Cross-Validation Results")
        print(f"{'='*80}")
        print(results_df.to_string())

        # Compute means and stds
        metrics_cols = ["roc_auc", "auprc", "f1", "sensitivity", "specificity", "accuracy"]
        summary = {}
        for col in metrics_cols:
            if col in results_df.columns:
                mean = results_df[col].mean()
                std = results_df[col].std()
                summary[col] = f"{mean:.4f} ± {std:.4f}"

        print(f"\nSummary Metrics (mean ± std):")
        for key, val in summary.items():
            print(f"  {key}: {val}")

        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(results_dir / "cv_summary_metrics.csv", index=False)

        # Save config
        config.to_yaml(results_dir / "config_used.yaml")

        print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
