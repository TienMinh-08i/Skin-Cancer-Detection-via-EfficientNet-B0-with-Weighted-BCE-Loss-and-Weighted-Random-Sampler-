"""Metrics computation utilities."""
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Dictionary containing all computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    # Probability-based metrics
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = np.nan
        metrics["auprc"] = np.nan

    # Prediction-based metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["sensitivity"] = metrics["recall"]  # Alias
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    return metrics


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[float, Dict[str, float]]:
    """
    Compute metrics for multiple thresholds.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        thresholds: Array of threshold values.

    Returns:
        Dictionary mapping threshold -> metrics dict.
    """
    result = {}
    for threshold in thresholds:
        result[float(threshold)] = compute_binary_metrics(y_true, y_prob, threshold)
    return result


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
    fn_cost: float = 1.0,
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        method: Method for threshold selection.
                - "youden": Youden index (sensitivity + specificity - 1)
                - "f1": Maximize F1 score
                - "clinical": Minimize clinical cost (FN cost weighted)
        fn_cost: Cost weight for false negatives (for clinical method).

    Returns:
        Tuple of (optimal_threshold, optimal_score).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if method == "youden":
        specificity = 1 - fpr
        youden_scores = tpr + specificity - 1
        best_idx = np.argmax(youden_scores)
        return float(thresholds[best_idx]), float(youden_scores[best_idx])

    elif method == "f1":
        best_f1 = -1
        best_threshold = 0.5
        for threshold in np.linspace(0, 1, 101):
            metrics = compute_binary_metrics(y_true, y_prob, threshold)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
        return float(best_threshold), float(best_f1)

    elif method == "clinical":
        # Cost = FN_count * fn_cost + FP_count
        best_cost = float("inf")
        best_threshold = 0.5
        for threshold in np.linspace(0, 1, 101):
            y_pred = (y_prob >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            cost = fn * fn_cost + fp
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
        return float(best_threshold), float(best_cost)

    else:
        raise ValueError(f"Unknown method: {method}")
