"""Training utilities."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..evaluators import evaluate_epoch
from ..utils.logger import setup_logger


class Trainer:
    """Training orchestrator."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        config: Any,
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
        experiment_name: str,
    ):
        """
        Args:
            model: Neural network.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            test_loader: Test DataLoader.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            config: Config object.
            device: Computation device.
            checkpoint_dir: Directory to save checkpoints.
            log_dir: Directory to save logs.
            experiment_name: Name of experiment.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            experiment_name,
            log_file=self.log_dir / f"{experiment_name}.log",
        )

        # Use old GradScaler API for PyTorch 1.13 compatibility
        self.scaler = GradScaler(enabled=config.training.use_amp and device.type == "cuda")
        self.best_score = -float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.log_rows: List[Dict[str, Any]] = []

    def train_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Training loop for one epoch.

        Returns:
            Tuple of (avg_loss, metrics_dict).
        """
        self.model.train()

        losses = []
        all_targets = []
        all_probs = []

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass
            if self.config.training.use_amp and self.device.type == "cuda":
                with autocast():
                    logits = self.model(images).view(-1)
                    loss = self.criterion(logits, targets)
            else:
                logits = self.model(images).view(-1)
                loss = self.criterion(logits, targets)

            # Backward pass
            if self.config.training.gradient_accumulation_steps > 1:
                loss = loss / self.config.training.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Collect metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                losses.append(loss.item() * images.size(0))
                all_targets.extend(targets.cpu().numpy().tolist())
                all_probs.extend(probs.tolist())

        # Compute epoch metrics
        avg_loss = float(np.sum(losses) / len(self.train_loader.dataset)) if len(self.train_loader.dataset) > 0 else 0.0

        from ..utils.metrics import compute_binary_metrics
        metrics = compute_binary_metrics(np.array(all_targets), np.array(all_probs), threshold=0.5)

        return avg_loss, metrics

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validation loop.

        Returns:
            Tuple of (avg_loss, metrics_dict).
        """
        avg_loss, metrics, _, _, _ = evaluate_epoch(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
            use_amp=self.config.training.use_amp,
        )
        return avg_loss, metrics

    def should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        metric_name = self.config.training.early_stopping_metric
        score = val_metrics.get(metric_name, -float("inf"))

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = len(self.log_rows) + 1
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.config.training.early_stopping_patience

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save checkpoint."""
        ckpt_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "val_metrics": val_metrics,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, ckpt_path)

    def load_best_checkpoint(self) -> Dict[str, Any]:
        """Load best checkpoint."""
        ckpt_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training results dictionary.
        """
        self.logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        self.logger.info(f"Early stopping metric: {self.config.training.early_stopping_metric}")

        for epoch in range(1, self.config.training.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_metrics = self.train_one_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Check early stopping
            should_stop = self.should_stop_early(val_metrics)
            if should_stop or epoch == self.config.training.num_epochs:
                self.save_checkpoint(epoch, val_metrics, is_best=True)

            epoch_time = time.time() - epoch_start

            # Log
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_roc_auc": train_metrics.get("roc_auc", np.nan),
                "train_auprc": train_metrics.get("auprc", np.nan),
                "train_f1": train_metrics.get("f1", np.nan),
                "val_roc_auc": val_metrics.get("roc_auc", np.nan),
                "val_auprc": val_metrics.get("auprc", np.nan),
                "val_sensitivity": val_metrics.get("sensitivity", np.nan),
                "val_specificity": val_metrics.get("specificity", np.nan),
                "val_f1": val_metrics.get("f1", np.nan),
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
            }
            self.log_rows.append(row)

            self.logger.info(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_metrics.get('roc_auc', np.nan):.4f} | "
                f"val_{self.config.training.early_stopping_metric}={val_metrics.get(self.config.training.early_stopping_metric, np.nan):.4f} | "
                f"best_epoch={self.best_epoch}"
            )

            if should_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save training log
        log_df = pd.DataFrame(self.log_rows)
        log_path = self.log_dir / f"{self.experiment_name}_training_log.csv"
        log_df.to_csv(log_path, index=False)
        self.logger.info(f"Training log saved to {log_path}")

        return {
            "log_df": log_df,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "log_path": str(log_path),
        }
