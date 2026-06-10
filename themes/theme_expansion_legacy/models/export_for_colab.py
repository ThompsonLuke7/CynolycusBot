"""
Bundle theme-rotation signal labels into a Colab-ready training archive.

Usage:
    python -m theme_expansion.models.export_for_colab

Produces in theme_expansion/outputs/training_export/:
    theme_training_matrix.parquet
    feature_manifest.json
    theme_train_colab.py
    theme_colab_bundle.tgz
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

from theme_expansion.config import OUTPUT_DIR, THEME_SIGNAL_LABELS_PATH

EXPORT_DIR = OUTPUT_DIR / "training_export"
TARGET_COLUMN = "fwd_theme_excess_benchmark_20d"
ID_COLUMNS = ["date", "theme", "category", "benchmark", "signal_market_playbook"]
BASE_FEATURE_COLUMNS = [
    "theme_regime_rank",
    "theme_heat_rank",
    "theme_return_5d",
    "theme_return_10d",
    "theme_return_20d",
    "theme_vs_spy_5d",
    "theme_vs_spy_10d",
    "theme_vs_spy_20d",
    "theme_above_20d_pct",
    "theme_above_50d_pct",
    "theme_breadth",
    "theme_rvol",
    "theme_persistence_score",
]
CATEGORICAL_COLUMNS = ["category", "benchmark", "signal_market_playbook"]


def signal_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "signal_pair_trade_side",
        "signal_market_playbook",
    }
    return [
        c
        for c in df.columns
        if c.startswith("signal_")
        and c not in blocked
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def export_training_bundle(
    *,
    labels_path: Path = THEME_SIGNAL_LABELS_PATH,
    out_dir: Path = EXPORT_DIR,
    target_column: str = TARGET_COLUMN,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not labels_path.exists():
        raise FileNotFoundError(f"Theme signal labels not found: {labels_path}")

    labels = pd.read_parquet(labels_path)
    if target_column not in labels.columns:
        raise ValueError(f"Target column not found: {target_column}")

    numeric_features = [c for c in BASE_FEATURE_COLUMNS if c in labels.columns]
    numeric_features.extend(c for c in signal_feature_columns(labels) if c not in numeric_features)
    keep = [c for c in ID_COLUMNS if c in labels.columns] + numeric_features + [target_column]
    matrix = labels[keep].copy()
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix = matrix[matrix[target_column].notna()].sort_values(["date", "theme"])

    matrix_path = out_dir / "theme_training_matrix.parquet"
    matrix.to_parquet(matrix_path, index=False)

    manifest = {
        "target_column": target_column,
        "target_kind": "regression",
        "feature_columns": numeric_features,
        "categorical_columns": [c for c in CATEGORICAL_COLUMNS if c in matrix.columns],
        "id_columns": [c for c in ID_COLUMNS if c in matrix.columns],
        "n_rows": int(len(matrix)),
        "n_themes": int(matrix["theme"].nunique()) if "theme" in matrix.columns else None,
        "date_min": str(matrix["date"].min().date()),
        "date_max": str(matrix["date"].max().date()),
        "source_labels": str(labels_path),
        "walk_forward": {
            "train_years": 3,  # Alpaca daily history starts ~2020-07; 3y keeps ~5 OOF folds
            "embargo_days": 30,  # > 20d label horizon (fwd_theme_excess_benchmark_20d)
            "test_months": 6,
            "min_train_rows": 30000,
            "min_test_rows": 1000,
        },
    }
    manifest_path = out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    trainer_src = Path(__file__).parent / "colab" / "theme_train_colab.py"
    trainer_dst = out_dir / "theme_train_colab.py"
    if trainer_src.exists():
        shutil.copy2(trainer_src, trainer_dst)

    bundle_path = out_dir / "theme_colab_bundle.tgz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(matrix_path, arcname=matrix_path.name)
        tar.add(manifest_path, arcname=manifest_path.name)
        if trainer_dst.exists():
            tar.add(trainer_dst, arcname=trainer_dst.name)

    print(f"bundle: {bundle_path}")
    print(f"rows: {manifest['n_rows']} themes: {manifest['n_themes']}")
    print(f"features: {len(numeric_features)} target: {target_column}")
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=THEME_SIGNAL_LABELS_PATH)
    parser.add_argument("--out", type=Path, default=EXPORT_DIR)
    parser.add_argument("--target", default=TARGET_COLUMN)
    args = parser.parse_args()
    export_training_bundle(labels_path=args.labels, out_dir=args.out, target_column=args.target)


if __name__ == "__main__":
    main()
