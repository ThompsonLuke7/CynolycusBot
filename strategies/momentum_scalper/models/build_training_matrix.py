"""Merge features, labels, and scanner states into a training matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategies.momentum_scalper.configs.settings import FEATURES_PATH, LABELS_PATH, SCANNER_SNAPSHOTS_DIR, TRAINING_MATRIX_PATH, ensure_data_dirs
from strategies.momentum_scalper.utils.io import normalize_timestamp_column, write_parquet


def build_training_matrix(
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
    scanner_dir: Path = SCANNER_SNAPSHOTS_DIR,
    output: Path = TRAINING_MATRIX_PATH,
) -> pd.DataFrame:
    ensure_data_dirs()
    features = normalize_timestamp_column(pd.read_parquet(features_path)) if features_path.exists() else pd.DataFrame()
    labels = normalize_timestamp_column(pd.read_parquet(labels_path)) if labels_path.exists() else pd.DataFrame()
    snapshots = []
    for path in sorted(scanner_dir.glob("*.parquet")):
        snapshots.append(normalize_timestamp_column(pd.read_parquet(path)))
    scanner = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()
    matrix = features
    if not labels.empty:
        matrix = matrix.merge(labels, on=["timestamp", "ticker"], how="inner")
    if not scanner.empty:
        keep = [c for c in ["timestamp", "ticker", "scanner_rank", "news_flag"] if c in scanner.columns]
        matrix = matrix.merge(scanner[keep].drop_duplicates(["timestamp", "ticker"]), on=["timestamp", "ticker"], how="left", suffixes=("", "_scanner"))
    write_parquet(matrix, output)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scalper training matrix")
    parser.add_argument("--output", type=Path, default=TRAINING_MATRIX_PATH)
    args = parser.parse_args()
    df = build_training_matrix(output=args.output)
    print(f"wrote {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
