"""Colab export for the WS-E ablation arms (deliverable 4).

Bundles a training matrix that includes the market-regime/sector-context
columns (``REGIME_FEATURE_COLUMNS_4H``) alongside the existing baseline
features, plus an arm-aware feature manifest, so the five ablation arms
(baseline / +risk / +liquidity / +sector / +all — ``feature_blocks.ARMS``)
can actually be trained off-repo on Colab (plan defect D6: no local
training). Mirrors the existing
``strategies/momentum_expansion/models/export_for_colab.py`` pattern.

Handles plan defect D3 explicitly (rather than fighting it): the production
``training_matrix_4h.parquet`` deliberately excludes the regime columns
(``build_training_matrix``'s ``dropna(subset=feature_cols, how="any")`` would
otherwise silently delete most of the pre-2021 training set — see the module
docstring in feature_matrix_4h.py). So this module does NOT read
``FEATURE_COLUMNS_4H`` from the production matrix and add regime columns in
place; it builds a SEPARATE augmented matrix (own path, own directory, never
overwrites ``training_matrix_4h.parquet``) that still keeps the two column
sets clearly delineated in the manifest, so a Colab trainer can select
per-arm feature lists and get the SAME pre-2021 rows for the baseline arm as
production training, while +risk/+liquidity/+sector/+all arms naturally see
NaN rows drop out of only their own (smaller, more recent) subset once the
Colab trainer's own dropna runs on the arm's feature list.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.export_colab_ablation
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
    MODULE_ROOT,
    TRAINING_MATRIX,
    TRAINING_MATRIX_CONFIG,
    WALK_FORWARD_CONFIG,
)
from strategies.momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H

from . import screen as S
from .feature_blocks import ARM_ADDED_COLUMNS, ARMS

logger = logging.getLogger(__name__)

ABLATION_EXPORT_DIR = MODULE_ROOT / "data" / "training_export_ablation"
ABLATION_TRAINING_MATRIX_PATH = ABLATION_EXPORT_DIR / "training_matrix_4h_with_regime.parquet"

# All labels the Colab trainer might diagnose against — trend_persistence is
# INCLUDED here only because it already ships in the production matrix and a
# Colab trainer may legitimately want it as a raw diagnostic column; it must
# never be used as a training TARGET (see screen.py's BANNED_LABEL and the
# task's explicit instruction). The manifest's own target_column below is
# expansion_survival_score, matching TRAINING_MATRIX_CONFIG, not this list.
DIAGNOSTIC_LABEL_COLUMNS = [
    "fwd_max_return", "fwd_max_alpha", "fwd_atr_adj_return", "fwd_max_drawdown",
    "fwd_close_return", "trend_persistence", "expansion_score", "expansion_target",
    "expansion_survival_score",
]


def build_ablation_training_matrix(
    *, matrix_path: Path = TRAINING_MATRIX, out_path: Path = ABLATION_TRAINING_MATRIX_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """Read the production training matrix READ-ONLY, join the WS-A regime/
    sector columns onto it, and write the augmented result to a SEPARATE path
    (never touches ``matrix_path``)."""
    if out_path.exists() and not force:
        logger.info("Ablation training matrix already exists at %s. Use force=True to rebuild.", out_path)
        return pd.read_parquet(out_path)

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Production training matrix not found at {matrix_path}. Run the existing "
            "momentum_expansion feature+label pipeline first; this export reads it read-only."
        )
    logger.info("Loading production training matrix (read-only) from %s ...", matrix_path)
    df = pd.read_parquet(matrix_path).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info("  %d rows x %d cols, %s .. %s", len(df), df.shape[1], df["timestamp"].min(), df["timestamp"].max())

    logger.info("Joining market-regime / sector-context features (read-only WS-A tables) ...")
    augmented = S.join_regime_and_sector_features(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(out_path)
    logger.info("Wrote augmented ablation training matrix -> %s (%d rows x %d cols)",
                out_path, len(augmented), augmented.shape[1])
    return augmented


def export_ablation_bundle(
    *, matrix_path: Path = TRAINING_MATRIX, out_dir: Path = ABLATION_EXPORT_DIR, force: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "training_matrix_4h_with_regime.parquet"
    df = build_ablation_training_matrix(matrix_path=matrix_path, out_path=target_path, force=force)

    # Per-arm feature lists, filtered to columns actually present (mirrors
    # export_for_colab.py's `[c for c in FEATURE_COLUMNS_4H if c in df.columns]`
    # pattern -- the production matrix already dropped 2 highly-correlated
    # xsec_* columns via _drop_correlated_features, so baseline is not
    # literally FEATURE_COLUMNS_4H verbatim).
    arms_present = {name: [c for c in cols if c in df.columns] for name, cols in ARMS.items()}
    arms_added_present = {name: [c for c in cols if c in df.columns] for name, cols in ARM_ADDED_COLUMNS.items()}

    manifest = {
        "arms": arms_present,
        "arms_added_columns": arms_added_present,
        "baseline_feature_columns": [c for c in FEATURE_COLUMNS_4H if c in df.columns],
        "target_column": TRAINING_MATRIX_CONFIG["target_column"],
        "target_kind": TRAINING_MATRIX_CONFIG["target_kind"],
        "regression_target_column": TRAINING_MATRIX_CONFIG["target_column"],
        "relevance_column": "is_strong_setup",
        "strong_setup_source_column": "fwd_max_return",
        "strong_setup_threshold": 0.20,
        "banned_target_column": "trend_persistence",
        "banned_target_reason": "forward-looking / leaky label -- never a training target (see LIVING_SUMMARY meta-breadth retraction)",
        "diagnostic_label_columns": [c for c in DIAGNOSTIC_LABEL_COLUMNS if c in df.columns],
        "train_frac": 0.6,
        "val_frac": 0.2,
        "rank_group": "timestamp",
        "top_k": 20,
        "n_rows": int(len(df)),
        "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
        "date_min": str(df["timestamp"].min()),
        "date_max": str(df["timestamp"].max()),
        "label_cfg": LABEL_CONFIG,
        "training_matrix_cfg": TRAINING_MATRIX_CONFIG,
        "walk_forward": WALK_FORWARD_CONFIG,
        "primary_eval_metrics": [
            "spearman_to_target", "rank_ic", "ndcg_at_10",
            "top10_per_4h_bar_avg_fwd_max_alpha", "top10_per_4h_bar_win_rate",
            "turnover", "sharpe", "max_drawdown",
        ],
    }
    manifest_path = out_dir / "ablation_feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, default=str, indent=2)

    repo_root = Path(__file__).resolve().parents[3]
    harness_src = repo_root / "strategies" / "model_training" / "colab_competition.py"
    harness_dst = out_dir / harness_src.name
    shutil.copy2(harness_src, harness_dst)

    notebook_src = Path(__file__).parent / "colab" / "ablation_train_colab.py"
    notebook_dst = out_dir / notebook_src.name if notebook_src.exists() else None
    if notebook_dst is not None:
        shutil.copy2(notebook_src, notebook_dst)
    else:
        logger.warning("No ablation_train_colab.py found at %s -- bundle will omit the trainer script", notebook_src)

    bundle_path = out_dir / "ablation_colab_bundle.tgz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(target_path, arcname=target_path.name)
        tar.add(manifest_path, arcname=manifest_path.name)
        tar.add(harness_dst, arcname=harness_dst.name)
        if notebook_dst is not None:
            tar.add(notebook_dst, arcname=notebook_dst.name)

    logger.info("Ablation Colab bundle written to %s", bundle_path)
    logger.info("  matrix: %s (%d rows)", target_path.name, len(df))
    for name, cols in arms_present.items():
        logger.info("  arm %-12s %d features", name, len(cols))
    return bundle_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, default=TRAINING_MATRIX)
    ap.add_argument("--out", type=Path, default=ABLATION_EXPORT_DIR)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    export_ablation_bundle(matrix_path=args.matrix, out_dir=args.out, force=args.force)


if __name__ == "__main__":
    main()
