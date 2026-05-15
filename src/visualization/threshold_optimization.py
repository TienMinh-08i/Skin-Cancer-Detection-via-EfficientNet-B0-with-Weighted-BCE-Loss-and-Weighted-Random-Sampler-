"""Threshold optimization utilities."""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..utils.metrics import compute_binary_metrics, find_optimal_threshold


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    methods: List[str] = None,
    fn_cost_ratio: float = 1.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Optimize classification thresholds using multiple methods.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        methods: List of methods (e.g., ["youden", "f1", "clinical"]).
        fn_cost_ratio: Cost ratio for false negatives in clinical method.

    Returns:
        Dictionary mapping method -> (optimal_threshold, score).
    """
    if methods is None:
        methods = ["youden", "f1", "clinical"]

    results = {}
    for method in methods:
        try:
            if method == "clinical":
                threshold, score = find_optimal_threshold(y_true, y_prob, method=method, fn_cost=fn_cost_ratio)
            else:
                threshold, score = find_optimal_threshold(y_true, y_prob, method=method)
            results[method] = (float(threshold), float(score))
        except Exception as e:
            print(f"Error optimizing threshold with method {method}: {e}")

    return results


def generate_threshold_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
    save_path: Path,
) -> pd.DataFrame:
    """
    Generate a comprehensive threshold report.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        thresholds: Thresholds to evaluate.
        save_path: Path to save CSV report.

    Returns:
        DataFrame with threshold metrics.
    """
    rows = []

    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold)
        y_pred = (y_prob >= threshold).astype(int)

        row = {
            "threshold": threshold,
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "auprc": metrics["auprc"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Threshold report saved to {save_path}")

    return df
