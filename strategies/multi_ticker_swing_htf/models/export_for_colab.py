"""
Bundle the HTF swing training matrix into a Colab-ready archive.

Usage:
    python -m multi_ticker_swing_htf.models.export_for_colab

Produces in multi_ticker_swing_htf/data/training_export/:
    htf_training_matrix_4h.parquet
    feature_manifest.json
    label_manifest.json
    htf_swing_train_colab.py
    htf_swing_colab_bundle.tgz

Target is the continuous swing-quality score `htf_swing_score` (regression); the
trainer ranks setups out-of-fold so the OOF predictions are backtestable.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tarfile
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H
from strategies.multi_ticker_swing_htf.config import PIVOT_LABEL_CONFIG, TRAINING_MATRIX

logger = logging.getLogger(__name__)

EXPORT_DIR = TRAINING_MATRIX.parent.parent / "training_export"
TARGET_COLUMN = "htf_swing_score"
DIAGNOSTIC_COLUMNS = [
    "htf_top_swing_target",
    "target",
    "fwd_best_high_return",
    "fwd_worst_low_return",
    "fwd_close_return",
    "long_persistence",
    "short_persistence",
]
WALK_FORWARD = {
    "train_years": 2,        # 4h Alpaca history starts ~2020-07 (~5.9y) -> ~6 folds
    "embargo_days": 21,      # > forward label window (forward_max_bars ~15 trading days)
    "test_months": 6,
    "min_train_rows": 30000,
}


def export_training_bundle(
    *,
    matrix_path: Path = TRAINING_MATRIX,
    out_dir: Path = EXPORT_DIR,
    target_column: str = TARGET_COLUMN,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"HTF training matrix not found at {matrix_path}. "
            "Run `python -m multi_ticker_swing_htf.main --build-features --build-labels --build-matrix` first."
        )

    df = pd.read_parquet(matrix_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} missing from {matrix_path}")

    # Regression target is only defined for actual long/short swing setups
    # (NaN on flat bars) — keep just those rows.
    n_all = len(df)
    df = df[df[target_column].notna()].copy()
    logger.info("kept %d/%d rows with non-null %s", len(df), n_all, target_column)

    target_path = out_dir / "htf_training_matrix_4h.parquet"
    df.to_parquet(target_path)

    feature_columns = [c for c in FEATURE_COLUMNS_4H if c in df.columns]
    feature_manifest = {
        "feature_columns": feature_columns,
        "label_columns": [c for c in DIAGNOSTIC_COLUMNS if c in df.columns],
        "target_column": target_column,
        "target_kind": "regression",
        "n_rows": int(len(df)),
        "n_tickers": int(df.index.get_level_values("ticker").nunique()) if "ticker" in df.index.names else None,
        "date_min": str(df.index.get_level_values(0).min()),
        "date_max": str(df.index.get_level_values(0).max()),
        "pivot_label_cfg": PIVOT_LABEL_CONFIG,
        "primary_eval_metrics": [
            "spearman_to_target",
            "top_score_bucket_avg_fwd_best_high_return",
            "top_score_bucket_avg_fwd_close_return",
        ],
        "walk_forward": WALK_FORWARD,
    }
    feature_manifest_path = out_dir / "feature_manifest.json"
    feature_manifest_path.write_text(json.dumps(feature_manifest, default=str, indent=2))

    label_manifest = {
        "pivot_label_cfg": PIVOT_LABEL_CONFIG,
        "target_column": target_column,
        "target_kind": "regression",
        "diagnostic_columns": [c for c in DIAGNOSTIC_COLUMNS if c in df.columns],
    }
    label_manifest_path = out_dir / "label_manifest.json"
    label_manifest_path.write_text(json.dumps(label_manifest, default=str, indent=2))

    trainer_src = Path(__file__).parent / "colab" / "htf_swing_train_colab.py"
    trainer_dst = out_dir / trainer_src.name if trainer_src.exists() else None
    if trainer_dst is not None:
        shutil.copy2(trainer_src, trainer_dst)

    bundle_path = out_dir / "htf_swing_colab_bundle.tgz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(target_path, arcname=target_path.name)
        tar.add(feature_manifest_path, arcname=feature_manifest_path.name)
        tar.add(label_manifest_path, arcname=label_manifest_path.name)
        if trainer_dst is not None:
            tar.add(trainer_dst, arcname=trainer_dst.name)

    logger.info("bundle: %s", bundle_path)
    logger.info("  rows: %d  features: %d  target: %s", len(df), len(feature_columns), target_column)
    return bundle_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=TRAINING_MATRIX)
    parser.add_argument("--out", type=Path, default=EXPORT_DIR)
    args = parser.parse_args()
    export_training_bundle(matrix_path=args.matrix, out_dir=args.out)
