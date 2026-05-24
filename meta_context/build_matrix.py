"""Build a meta-model matrix from specialist signal parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from meta_context.config import META_FEATURE_COLUMNS, META_TRAINING_MATRIX_PATH, ensure_dirs


def _read(path: Path | str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p) if p.suffix.lower() != ".csv" else pd.read_csv(p)


def build_meta_training_matrix(
    frames: Iterable[pd.DataFrame],
    *,
    output_path: Path | str = META_TRAINING_MATRIX_PATH,
) -> pd.DataFrame:
    ensure_dirs()
    merged: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        cur = frame.copy()
        cur["timestamp"] = pd.to_datetime(cur["timestamp"], utc=True, errors="coerce")
        cur["ticker"] = cur["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
        keys = ["timestamp", "ticker"]
        merged = cur if merged is None else merged.merge(cur, on=keys, how="outer")
    out = merged if merged is not None else pd.DataFrame(columns=["timestamp", "ticker", *META_FEATURE_COLUMNS])
    for col in META_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    out.to_parquet(output_path, index=False)
    return out


def build_from_paths(
    *,
    ta_path: Path | str | None = None,
    news_path: Path | str | None = None,
    social_path: Path | str | None = None,
    events_path: Path | str | None = None,
    theme_path: Path | str | None = None,
    leader_path: Path | str | None = None,
    output_path: Path | str = META_TRAINING_MATRIX_PATH,
) -> pd.DataFrame:
    return build_meta_training_matrix(
        [_read(p) for p in (ta_path, news_path, social_path, events_path, theme_path, leader_path)],
        output_path=output_path,
    )

