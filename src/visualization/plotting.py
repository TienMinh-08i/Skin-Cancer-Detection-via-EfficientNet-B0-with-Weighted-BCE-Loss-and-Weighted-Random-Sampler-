"""Plotting utilities for paper-ready figures."""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """
    Plot ROC curve.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        title: Plot title.
        save_path: Path to save figure.
        dpi: DPI for saving.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        print(f"Skipping ROC curve for {title} - only one class")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        title: Plot title.
        save_path: Path to save figure.
        dpi: DPI for saving.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        print(f"Skipping PR curve for {title} - only one class")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        title: Plot title.
        save_path: Path to save figure.
        dpi: DPI for saving.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im)

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign", "Malignant"], fontsize=11)
    plt.yticks(tick_marks, ["Benign", "Malignant"], fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    log_df: pd.DataFrame,
    title: str,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """
    Plot training curves.

    Args:
        log_df: Training log DataFrame.
        title: Plot title.
        save_path: Path to save figure.
        dpi: DPI for saving.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    axes[0, 0].plot(log_df["epoch"], log_df["train_loss"], label="train", marker="o")
    axes[0, 0].plot(log_df["epoch"], log_df["val_loss"], label="val", marker="s")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # ROC-AUC
    if "val_roc_auc" in log_df.columns:
        axes[0, 1].plot(log_df["epoch"], log_df["val_roc_auc"], label="val ROC-AUC", marker="o")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("ROC-AUC")
        axes[0, 1].set_title("Validation ROC-AUC")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    # AUPRC
    if "val_auprc" in log_df.columns:
        axes[1, 0].plot(log_df["epoch"], log_df["val_auprc"], label="val AUPRC", marker="o", color="green")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUPRC")
        axes[1, 0].set_title("Validation AUPRC")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    # F1
    if "val_f1" in log_df.columns:
        axes[1, 1].plot(log_df["epoch"], log_df["val_f1"], label="val F1", marker="s", color="red")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].set_title("Validation F1 Score")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_combined_roc(
    results: List[Dict],
    save_path: Path,
    dpi: int = 300,
) -> None:
    """
    Plot combined ROC curves for multiple experiments.

    Args:
        results: List of result dictionaries.
        save_path: Path to save figure.
        dpi: DPI for saving.
    """
    plt.figure(figsize=(8, 7))
    plotted = 0

    for result in results:
        y_true = result.get("y_true")
        y_prob = result.get("y_prob")
        name = result.get("name", "")

        if y_true is None or y_prob is None:
            continue

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_value:.3f})")
        plotted += 1

    if plotted == 0:
        print("No valid ROC curves to plot")
        plt.close()
        return

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Test ROC Curves - Model Comparison", fontsize=12)
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
