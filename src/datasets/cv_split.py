"""Dataset preparation and cross-validation utilities."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_metadata(
    csv_path: Path,
    image_dir: Path,
    target_col: str = "target",
    image_id_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Load metadata CSV and resolve image paths.

    Args:
        csv_path: Path to metadata CSV.
        image_dir: Directory containing images.
        target_col: Name of target column.
        image_id_cols: List of column names to try for image IDs.

    Returns:
        DataFrame with resolved image paths and binary targets.
    """
    if image_id_cols is None:
        image_id_cols = ["image_name", "image_id", "filename"]

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded metadata: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Infer and clean target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")

    # Convert target to binary
    df["target"] = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    df = df[df["target"].isin([0, 1])]

    print(f"After cleaning target: {len(df)} samples")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")

    # Resolve image paths
    image_files = sorted(image_dir.glob("**/*.[jJ][pP][gG]"))
    image_files.extend(sorted(image_dir.glob("**/*.[pP][nN][gG]")))
    image_index = {p.stem: str(p) for p in image_files}
    image_index.update({p.name: str(p) for p in image_files})

    print(f"Indexed {len(image_files)} image files")

    # Try to match image IDs
    df["image_path"] = None
    for id_col in image_id_cols:
        if id_col not in df.columns:
            continue
        mask = df["image_path"].isna()
        df.loc[mask, "image_path"] = (
            df.loc[mask, id_col]
            .astype(str)
            .str.strip()
            .apply(lambda x: image_index.get(x) or image_index.get(Path(x).stem))
        )

    # Drop unresolved
    before = len(df)
    df = df.dropna(subset=["image_path"])
    print(f"Resolved {len(df)} image paths (dropped {before - len(df)})")

    return df.reset_index(drop=True)


def stratified_k_fold_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create stratified k-fold splits.

    Args:
        df: DataFrame with 'target' column.
        n_splits: Number of folds.
        random_state: Random seed.

    Returns:
        List of (train_df, val_df) tuples for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["target"])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"  Train: {len(train_df)} (class dist: {train_df['target'].value_counts().to_dict()})")
        print(f"  Val:   {len(val_df)} (class dist: {val_df['target'].value_counts().to_dict()})")

        folds.append((train_df, val_df))

    return folds


def stratified_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits for a single fold (non-CV baseline).

    Args:
        df: DataFrame with 'target' column.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        random_state: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["target"],
        random_state=random_state,
    )

    # Second split: train vs val (within train+val)
    relative_val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["target"],
        random_state=random_state,
    )

    print("Single-fold stratified split:")
    print(f"  Train: {len(train_df)} (class dist: {train_df['target'].value_counts().to_dict()})")
    print(f"  Val:   {len(val_df)} (class dist: {val_df['target'].value_counts().to_dict()})")
    print(f"  Test:  {len(test_df)} (class dist: {test_df['target'].value_counts().to_dict()})")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
