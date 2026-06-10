"""Label alignment for social attention features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.config.momentum_config import LABELS_COMBINED
from signals.social_attention.config import (
    ATTENTION_FEATURES_PATH,
    NARRATIVE_FEATURES_PATH,
    SOCIAL_LABELS_PATH,
    TRAINING_MATRIX_PATH,
)
from signals.social_attention.io import read_table, write_table


def _prepare_momentum_labels(labels_path: Path | str = LABELS_COMBINED) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    if isinstance(labels.index, pd.MultiIndex):
        labels = labels.reset_index()
    elif "timestamp" not in labels.columns:
        labels = labels.reset_index()
    if "level_0" in labels.columns and "timestamp" not in labels.columns:
        labels = labels.rename(columns={"level_0": "timestamp"})
    if "level_1" in labels.columns and "ticker" not in labels.columns:
        labels = labels.rename(columns={"level_1": "ticker"})
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True, errors="coerce")
    labels["ticker"] = labels["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    keep = [c for c in ["timestamp", "ticker", "expansion_target", "expansion_score", "fwd_max_alpha", "fwd_close_return"] if c in labels.columns]
    return labels[keep].dropna(subset=["timestamp", "ticker"]).sort_values(["ticker", "timestamp"])


def build_social_labels(
    *,
    attention_path: Path | str = ATTENTION_FEATURES_PATH,
    narrative_path: Path | str = NARRATIVE_FEATURES_PATH,
    momentum_labels_path: Path | str = LABELS_COMBINED,
    labels_output_path: Path | str = SOCIAL_LABELS_PATH,
    matrix_output_path: Path | str = TRAINING_MATRIX_PATH,
) -> pd.DataFrame:
    attention = read_table(attention_path)
    narrative = read_table(narrative_path)
    if attention.empty:
        out = pd.DataFrame()
        write_table(out, labels_output_path)
        return write_table(out, matrix_output_path)
    features = attention.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True, errors="coerce")
    features["ticker"] = features["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    if not narrative.empty:
        narrative = narrative.copy()
        narrative["timestamp"] = pd.to_datetime(narrative["timestamp"], utc=True, errors="coerce")
        narrative["ticker"] = narrative["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
        features = features.merge(narrative, on=["ticker", "timestamp"], how="left", suffixes=("", "_narrative"))

    labels = _prepare_momentum_labels(momentum_labels_path)
    aligned_parts = []
    for ticker, left in features.sort_values(["ticker", "timestamp"]).groupby("ticker", sort=True):
        right = labels.loc[labels["ticker"].eq(ticker)].sort_values("timestamp")
        if right.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("timestamp"),
            right.rename(columns={"timestamp": "label_timestamp"}).sort_values("label_timestamp"),
            left_on="timestamp",
            right_on="label_timestamp",
            direction="forward",
        )
        merged["ticker"] = ticker
        aligned_parts.append(merged)
    out = pd.concat(aligned_parts, ignore_index=True) if aligned_parts else pd.DataFrame()
    if not out.empty:
        out["social_spike_success"] = out["expansion_target"]
    write_table(out, labels_output_path)
    train = out.loc[out.get("social_spike_success", pd.Series(dtype=float)).notna()].copy() if not out.empty else out
    return write_table(train, matrix_output_path)

