from pathlib import Path
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Data.load_data import (
    get_ticker_processed_base_dir,
    get_ticker_processed_split_dir,
    get_ticker_processed_stats_dir,
)
from Data.retrieve_data import normalize_ticker
from Features.feature_constants import SCALE_FEATURE_COLUMNS
from Features.feature_scaling import normalize_continuous_features, save_normalization_stats

# Default dataset name for processed artifacts
DATASETS_DIRNAME = "datasets"
DEFAULT_DATASET_NAME = "15min"


def _datasets_root(processed_dir: Path) -> Path:
    return processed_dir / DATASETS_DIRNAME


def _infer_dataset_name(processed_dir: Path) -> str | None:
    datasets_root = _datasets_root(processed_dir)
    if not datasets_root.exists():
        return None
    candidates = [p for p in datasets_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.name


def _resolve_dataset_dir(processed_dir: Path, dataset_name: str) -> Path:
    return _datasets_root(processed_dir) / dataset_name


def load_processed_frames(
    processed_dir: Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    x_filename: str = "X.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the cleaned, unnormalized feature matrix and labels produced by feature_matrix.
    """
    processed_dir = Path(processed_dir)
    dataset_dir = _resolve_dataset_dir(processed_dir, dataset_name)
    X = pd.read_parquet(dataset_dir / x_filename)
    y = pd.read_parquet(dataset_dir / "y.parquet")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    return X, y


def chronological_split_indices(
    n: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Chronologically split indices into train/val/test.
    Fractions are relative; test takes the remainder.
    """
    if n == 0:
        raise ValueError("No rows to split")
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0,1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    all_idx = np.arange(n)
    splits = {
        "train": all_idx[:train_end],
        "val": all_idx[train_end:val_end],
        "test": all_idx[val_end:],
    }
    return splits


def fit_scaler_on_train(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    scale_cols: set[str] = SCALE_FEATURE_COLUMNS,
) -> dict:
    """
    Fit scaler on train indices only and return stats.
    """
    if len(train_idx) == 0:
        raise ValueError("Train split is empty.")
    train_X = X.iloc[train_idx]
    _, stats = normalize_continuous_features(train_X, scale_cols)
    return stats


def save_split_indices(
    split_root: Path,
    dataset_name: str,
    splits: Dict[str, np.ndarray],
    x_stem: str
) -> None:
    """
    Persist split indices.
    """
    split_dir = split_root / dataset_name / x_stem
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "train_idx.npy", splits["train"])
    np.save(split_dir / "val_idx.npy", splits["val"])
    np.save(split_dir / "test_idx.npy", splits["test"])


def main(
    processed_dir: Path | None = None,
    dataset_name: str | None = None,
    ticker: str = "$SPY",
    label_mode: str | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    x_filename: str = "X.parquet",
) -> None:
    """
    End-to-end: load processed features (unnormalized), split chronologically,
    fit scaler on train, and save split indices + scaler stats.
    """
    clean_ticker = normalize_ticker(ticker)
    if processed_dir is None:
        processed_dir = get_ticker_processed_base_dir(clean_ticker)
    if dataset_name is None:
        inferred = _infer_dataset_name(processed_dir)
        dataset_name = inferred or DEFAULT_DATASET_NAME

    X, y = load_processed_frames(processed_dir, dataset_name,x_filename=x_filename)
    splits = chronological_split_indices(len(X), train_frac, val_frac)
    stats = fit_scaler_on_train(X, splits["train"], SCALE_FEATURE_COLUMNS)

    split_root = get_ticker_processed_split_dir(clean_ticker)
    stats_dir = get_ticker_processed_stats_dir(clean_ticker)
    x_stem = Path(x_filename).stem
    save_split_indices(split_root, dataset_name, splits, x_stem)
    save_normalization_stats(
        stats_dir,
        stats,
        filename=f"norm_stats_{dataset_name}_{x_stem}_train.json",
    )
    print(
        f"Saved split indices under {split_root / dataset_name / x_stem} "
        f"and scaler stats to {stats_dir} "
        f"with train/val/test = {train_frac}/{val_frac}/{1 - train_frac - val_frac}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split processed features into train/val/test chronologically."
    )
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ticker", type=str, default="$SPY")
    parser.add_argument("--label_mode", type=str, default="leg", choices=["leg", "swing"])
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--x-file", type=str, default="X.parquet")
    args = parser.parse_args()

    main(
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        dataset_name=args.dataset,
        ticker=args.ticker,
        label_mode=args.label_mode,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        x_filename=args.x_file,
    )
