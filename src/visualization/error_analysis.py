"""Error analysis utilities."""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def analyze_errors(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    image_paths: List[str],
    threshold: float = 0.5,
    n_fp: int = 10,
    n_fn: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Identify and analyze misclassifications.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        image_paths: List of image file paths.
        threshold: Classification threshold.
        n_fp: Number of false positives to analyze.
        n_fn: Number of false negatives to analyze.

    Returns:
        Dictionary with "false_positives" and "false_negatives" DataFrames.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # False positives: pred=1, true=0
    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_indices = np.where(fp_mask)[0]
    fp_indices = fp_indices[np.argsort(-y_prob[fp_indices])[:n_fp]]

    # False negatives: pred=0, true=1
    fn_mask = (y_pred == 0) & (y_true == 1)
    fn_indices = np.where(fn_mask)[0]
    fn_indices = fn_indices[np.argsort(y_prob[fn_indices])[:n_fn]]

    fp_df = pd.DataFrame({
        "image_path": [image_paths[i] for i in fp_indices],
        "true_label": y_true[fp_indices].astype(int),
        "pred_label": y_pred[fp_indices].astype(int),
        "confidence": y_prob[fp_indices],
        "error_type": "FP",
    })

    fn_df = pd.DataFrame({
        "image_path": [image_paths[i] for i in fn_indices],
        "true_label": y_true[fn_indices].astype(int),
        "pred_label": y_pred[fn_indices].astype(int),
        "confidence": y_prob[fn_indices],
        "error_type": "FN",
    })

    return {
        "false_positives": fp_df,
        "false_negatives": fn_df,
    }


def save_error_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    image_paths: List[str],
    save_dir: Path,
    threshold: float = 0.5,
    n_fp: int = 10,
    n_fn: int = 10,
) -> None:
    """
    Save error analysis with visualizations.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        image_paths: List of image file paths.
        save_dir: Directory to save analysis.
        threshold: Classification threshold.
        n_fp: Number of false positives.
        n_fn: Number of false negatives.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Analyze
    errors = analyze_errors(y_true, y_prob, image_paths, threshold, n_fp, n_fn)

    # Save CSVs
    errors["false_positives"].to_csv(save_dir / "false_positives.csv", index=False)
    errors["false_negatives"].to_csv(save_dir / "false_negatives.csv", index=False)

    print(f"Error analysis saved to {save_dir}")
