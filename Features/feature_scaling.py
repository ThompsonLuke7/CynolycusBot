import json
from pathlib import Path

import numpy as np
import pandas as pd

from Features.feature_sets.feature_constants import SCALE_FEATURE_COLUMNS


def _base_feature_name(name: str) -> str:
    return name.split("__", 1)[0]


def normalize_continuous_features(
    feature_df: pd.DataFrame, scale_cols: set[str] | None = None
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Standardize ONLY the vetted continuous magnitude features.
    Any column not explicitly listed in scale_cols is left untouched.
    """
    target_cols = set(scale_cols) if scale_cols is not None else SCALE_FEATURE_COLUMNS
    norm_df = feature_df.copy()
    stats: dict[str, dict[str, float]] = {}

    for col in norm_df.columns:
        base_col = _base_feature_name(col)
        if col not in target_cols and base_col not in target_cols:
            continue
        if not pd.api.types.is_numeric_dtype(norm_df[col]):
            continue
        mean = float(norm_df[col].mean())
        std = float(norm_df[col].std(ddof=0))
        if std == 0.0 or np.isnan(std):
            continue
        norm_df[col] = (norm_df[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}

    return norm_df, stats


def apply_scaler_from_stats(
    feature_df: pd.DataFrame, stats: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """
    Apply precomputed mean/std stats to a feature frame without refitting.
    """
    norm_df = feature_df.copy()
    for col, vals in stats.items():
        if col not in norm_df.columns:
            continue
        std = vals.get("std")
        mean = vals.get("mean")
        if std is None or std == 0.0 or np.isnan(std) or mean is None:
            continue
        norm_df[col] = (norm_df[col] - mean) / std
    return norm_df


def save_normalization_stats(
    output_dir: Path,
    stats: dict[str, dict[str, float]],
    filename: str = "norm_stats_spy_daily.json",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, "w") as f:
        json.dump(stats, f, indent=2)
