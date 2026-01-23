"""
Training pipeline skeleton.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.load_data import get_ticker_processed_split_dir
from Data.retrieve_data import normalize_ticker
from Features.feature_matrix_agent import AgentFeatureConfig, build_agent_feature_matrix


@dataclass(frozen=True)
class PipelineConfig:
    ticker: str = "$SPY"
    dataset_name: str = "15min"
    model_name: str = "ga_xgboost"
    x_filename: str = "X_15min_tree.parquet"
    drop_na: bool = False
    include_state_placeholders: bool = False


def build_agent_training_matrix(cfg: PipelineConfig | None = None) -> pd.DataFrame:
    config = cfg or PipelineConfig()
    agent_cfg = AgentFeatureConfig(
        ticker=config.ticker,
        dataset_name=config.dataset_name,
        model_name=config.model_name,
        drop_na=config.drop_na,
        include_state_placeholders=config.include_state_placeholders,
    )
    return build_agent_feature_matrix(config=agent_cfg)


def load_tree_split_indices(
    *,
    ticker: str,
    dataset_name: str,
    x_filename: str,
) -> dict[str, pd.Index]:
    clean = normalize_ticker(ticker)
    split_root = get_ticker_processed_split_dir(clean)
    x_stem = Path(x_filename).stem
    split_dirs = [
        split_root / dataset_name / x_stem,
        split_root / dataset_name,
    ]
    for split_dir in split_dirs:
        train_path = split_dir / "train_idx.npy"
        val_path = split_dir / "val_idx.npy"
        test_path = split_dir / "test_idx.npy"
        missing = [p.name for p in (train_path, val_path, test_path) if not p.exists()]
        if not missing:
            return {
                "train": pd.Index(np.load(train_path)),
                "val": pd.Index(np.load(val_path)),
                "test": pd.Index(np.load(test_path)),
            }
    raise FileNotFoundError(
        f"Missing split files under {split_root / dataset_name} (x_stem={x_stem})."
    )


def split_agent_matrix(
    df: pd.DataFrame,
    splits: dict[str, pd.Index],
    *,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_idx = splits["train"].sort_values()
    val_idx = splits["val"].sort_values()
    test_idx = splits["test"].sort_values()

    train_idx = train_idx.intersection(df.index)
    val_idx = val_idx.intersection(df.index)
    test_idx = test_idx.intersection(df.index)

    def _contiguous_slices(idx: pd.Index, name: str) -> list[slice]:
        if idx.empty:
            return []
        arr = idx.to_numpy()
        slices = []
        start = 0
        for i in range(1, arr.size):
            if arr[i] != arr[i - 1] + 1:
                slices.append(slice(arr[start], arr[i - 1] + 1))
                start = i
        slices.append(slice(arr[start], arr[-1] + 1))
        if verbose:
            print(f"{name} split: {arr.size} rows -> {len(slices)} contiguous chunks")
        return slices

    def _concat_by_slices(slices: list[slice]) -> pd.DataFrame:
        parts = [df.iloc[s] for s in slices if s.start is not None]
        if not parts:
            return df.iloc[0:0].copy()
        out = pd.concat(parts, axis=0)
        return out.reset_index(drop=True)

    train_df = _concat_by_slices(_contiguous_slices(train_idx, "train"))
    val_df = _concat_by_slices(_contiguous_slices(val_idx, "val"))
    test_df = _concat_by_slices(_contiguous_slices(test_idx, "test"))
    return train_df, val_df, test_df


def filter_splits_for_non_nan(
    df: pd.DataFrame,
    splits: dict[str, pd.Index],
    feature_cols: list[str],
) -> dict[str, pd.Index]:
    if not feature_cols:
        return splits
    keep_idx = df.index[~df[feature_cols].isna().any(axis=1)]
    filtered = {}
    for name, idx in splits.items():
        filtered[name] = idx.intersection(keep_idx)
    return filtered


# TODO: Fetch/load data for the target ticker (raw OHLCV + any required metadata).
# TODO: Build the feature matrix from raw data (feature_engineering + scaling).
# TODO: Add labels (pivot/ATR labels, state machine labels, etc.).
# TODO: Split data into train/val/test (time-based split; avoid leakage).
# TODO: Train the model using each label type (iterate label configs + log metrics).
# DONE: Create a new DataFrame that combines model probabilities, state machine
#       outputs, and general market features for RL inputs.
# DONE: Feed the combined features into the RL agent and train the policy.
