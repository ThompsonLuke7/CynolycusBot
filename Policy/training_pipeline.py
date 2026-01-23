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
    df: pd.DataFrame, splits: dict[str, pd.Index]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_idx = splits["train"].sort_values()
    val_idx = splits["val"].sort_values()
    test_idx = splits["test"].sort_values()
    if max(train_idx.max(), val_idx.max(), test_idx.max()) >= len(df):
        raise ValueError("Split indices exceed agent matrix length.")
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


# TODO: Fetch/load data for the target ticker (raw OHLCV + any required metadata).
# TODO: Build the feature matrix from raw data (feature_engineering + scaling).
# TODO: Add labels (pivot/ATR labels, state machine labels, etc.).
# TODO: Split data into train/val/test (time-based split; avoid leakage).
# TODO: Train the model using each label type (iterate label configs + log metrics).
# DONE: Create a new DataFrame that combines model probabilities, state machine
#       outputs, and general market features for RL inputs.
# DONE: Feed the combined features into the RL agent and train the policy.
