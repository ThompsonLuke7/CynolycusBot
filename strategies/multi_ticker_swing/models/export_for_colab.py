"""
Bundle the 30m multi-ticker swing training matrix into a Colab archive.

Usage:
    python -m multi_ticker_swing.models.export_for_colab
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tarfile
from pathlib import Path

import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import (
    FEATURE_COLUMNS,
    NEUTRAL_WEIGHT_FACTOR,
    OOF_N_FOLDS,
    TRAINING_MATRIX,
    TRAIN_FRAC,
    VAL_FRAC,
    XGBOOST_CONFIG,
)

logger = logging.getLogger(__name__)

EXPORT_DIR = TRAINING_MATRIX.parent.parent / "training_export"


def export_training_bundle(
    *,
    matrix_path: Path = TRAINING_MATRIX,
    out_dir: Path = EXPORT_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Training matrix not found at {matrix_path}. "
            "Run `python -m multi_ticker_swing.main --stage all ...` first."
        )

    df = pd.read_parquet(matrix_path)
    feature_columns = [c for c in FEATURE_COLUMNS if c in df.columns]
    target_path = out_dir / "training_matrix_30m.parquet"
    df.to_parquet(target_path, index=False)

    manifest = {
        "feature_columns": feature_columns,
        "target_column": "target",
        "sample_weight_column": "sample_weight" if "sample_weight" in df.columns else None,
        "timestamp_column": "timestamp",
        "ticker_column": "ticker" if "ticker" in df.columns else None,
        "n_rows": int(len(df)),
        "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
        "date_min": str(pd.to_datetime(df["timestamp"], utc=True).min()) if "timestamp" in df.columns else None,
        "date_max": str(pd.to_datetime(df["timestamp"], utc=True).max()) if "timestamp" in df.columns else None,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "neutral_weight_factor": NEUTRAL_WEIGHT_FACTOR,
        "oof_n_folds": OOF_N_FOLDS,
        "xgboost_config": XGBOOST_CONFIG,
        "output_artifacts": [
            "swing_xgb_model.json",
            "eval_metrics.json",
            "feature_importance.csv",
            "selected_features.txt",
            "p_swing_probs.parquet",
            "meta.json",
        ],
    }
    manifest_path = out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, default=str, indent=2))

    trainer_src = Path(__file__).parent / "colab" / "swing_train_colab.py"
    trainer_dst = out_dir / trainer_src.name
    shutil.copy2(trainer_src, trainer_dst)

    bundle_path = out_dir / "swing_colab_bundle.tgz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(target_path, arcname=target_path.name)
        tar.add(manifest_path, arcname=manifest_path.name)
        tar.add(trainer_dst, arcname=trainer_dst.name)

    logger.info("bundle: %s", bundle_path)
    logger.info("  rows: %d  tickers: %s  features: %d", len(df), manifest["n_tickers"], len(feature_columns))
    return bundle_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=TRAINING_MATRIX)
    parser.add_argument("--out", type=Path, default=EXPORT_DIR)
    args = parser.parse_args()
    export_training_bundle(matrix_path=args.matrix, out_dir=args.out)
