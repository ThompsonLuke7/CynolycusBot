"""
Bundle features + labels + feature manifest into a Colab-ready archive.

Usage:
    python -m momentum_expansion.models.export_for_colab

Produces in momentum_expansion/data/training_export/:
    training_matrix_4h.parquet
    feature_manifest.json
    label_manifest.json
    momentum_colab_bundle.tgz   (parquets + json bundled)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tarfile
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.config.momentum_config import (
    LABEL_CONFIG,
    LABELS_COMBINED,
    MOMENTUM_CANDIDATE_FILTER_CONFIG,
    TRAINING_EXPORT_DIR,
    TRAINING_MATRIX,
    TRAINING_MATRIX_CONFIG,
    WALK_FORWARD_CONFIG,
)
from strategies.momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H

logger = logging.getLogger(__name__)


def export_training_bundle(
    *,
    matrix_path: Path = TRAINING_MATRIX,
    out_dir: Path = TRAINING_EXPORT_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Training matrix not found at {matrix_path}. "
            "Run the feature + label pipeline first."
        )

    df = pd.read_parquet(matrix_path)
    target_path = out_dir / "training_matrix_4h.parquet"
    df.to_parquet(target_path)

    feature_manifest = {
        "feature_columns": [c for c in FEATURE_COLUMNS_4H if c in df.columns],
        "label_columns":   [c for c in df.columns if c.startswith("fwd_") or c.startswith("expansion_") or c == "trend_persistence"],
        "target_column":   TRAINING_MATRIX_CONFIG["target_column"],
        "target_kind":     TRAINING_MATRIX_CONFIG["target_kind"],
        # Competition harness config (colab_competition.py): regression on the
        # continuous expansion score + classifier/ranker on a binary strong-setup flag.
        "regression_target_column": TRAINING_MATRIX_CONFIG["target_column"],
        "relevance_column": "is_strong_setup",
        "strong_setup_source_column": "fwd_max_return",
        "strong_setup_threshold": 0.20,
        "train_frac": 0.6,
        "val_frac": 0.2,
        "rank_group": "timestamp",
        "top_k": 20,
        "n_rows":          int(len(df)),
        "n_tickers":       int(df.index.get_level_values("ticker").nunique()) if "ticker" in df.index.names else None,
        "date_min":        str(df.index.get_level_values(0).min()),
        "date_max":        str(df.index.get_level_values(0).max()),
        "label_cfg":       LABEL_CONFIG,
        "training_matrix_cfg": TRAINING_MATRIX_CONFIG,
        "momentum_candidate_filter": MOMENTUM_CANDIDATE_FILTER_CONFIG,
        "primary_eval_metrics": [
            "spearman_to_target",
            "top_score_bucket_avg_fwd_max_return",
            "top_score_bucket_pct_gt_20",
            "top5_per_4h_bar_avg_fwd_max_return",
            "top5_per_4h_bar_pct_gt_20",
            "drawdown",
        ],
        "walk_forward":    WALK_FORWARD_CONFIG,
    }
    feature_manifest_path = out_dir / "feature_manifest.json"
    with open(feature_manifest_path, "w") as f:
        json.dump(feature_manifest, f, default=str, indent=2)

    label_manifest = {k: LABEL_CONFIG[k] for k in LABEL_CONFIG}
    label_manifest["momentum_candidate_filter"] = MOMENTUM_CANDIDATE_FILTER_CONFIG
    label_manifest["training_matrix_cfg"] = TRAINING_MATRIX_CONFIG
    label_manifest["target_column"] = TRAINING_MATRIX_CONFIG["target_column"]
    label_manifest["target_kind"] = TRAINING_MATRIX_CONFIG["target_kind"]
    label_manifest_path = out_dir / "label_manifest.json"
    with open(label_manifest_path, "w") as f:
        json.dump(label_manifest, f, default=str, indent=2)

    bundle_path = out_dir / "momentum_colab_bundle.tgz"
    notebook_src = Path(__file__).parent / "colab" / "momentum_train_colab.py"
    if notebook_src.exists():
        notebook_path = out_dir / notebook_src.name
        shutil.copy2(notebook_src, notebook_path)
    else:
        notebook_path = None

    # Shared competition harness the trainer imports (`from colab_competition import ...`).
    repo_root = Path(__file__).resolve().parents[3]
    harness_src = repo_root / "strategies" / "model_training" / "colab_competition.py"
    harness_dst = out_dir / harness_src.name
    shutil.copy2(harness_src, harness_dst)

    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(target_path, arcname=target_path.name)
        tar.add(feature_manifest_path, arcname=feature_manifest_path.name)
        tar.add(label_manifest_path, arcname=label_manifest_path.name)
        tar.add(harness_dst, arcname=harness_dst.name)
        if notebook_path is not None:
            tar.add(notebook_path, arcname=notebook_path.name)

    logger.info("Bundle written to %s", bundle_path)
    logger.info("  matrix:   %s (%d rows)", target_path.name, len(df))
    logger.info("  features: %d cols", len(feature_manifest["feature_columns"]))
    return bundle_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=TRAINING_MATRIX)
    parser.add_argument("--out", type=Path, default=TRAINING_EXPORT_DIR)
    args = parser.parse_args()
    export_training_bundle(matrix_path=args.matrix, out_dir=args.out)
