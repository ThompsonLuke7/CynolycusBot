"""Final meta XGBoost model scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signals.meta_context.config import META_FEATURE_COLUMNS, META_MODEL_PATH, META_TRAINING_MATRIX_PATH, ensure_dirs


def train_meta_model(matrix_path: Path | str = META_TRAINING_MATRIX_PATH, *, target_col: str = "target") -> object:
    """Train a final XGBoost meta model when an explicit labeled matrix exists."""
    ensure_dirs()
    df = pd.read_parquet(matrix_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    from xgboost import XGBClassifier

    features = [c for c in META_FEATURE_COLUMNS if c in df.columns]
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_depth=3,
        learning_rate=0.04,
        n_estimators=300,
        n_jobs=4,
        random_state=42,
    )
    clf.fit(df[features].fillna(0.0).astype("float32"), df[target_col].astype(int))
    clf.save_model(str(META_MODEL_PATH))
    return clf


def main() -> int:
    parser = argparse.ArgumentParser(description="Train final meta XGBoost from specialist scores.")
    parser.add_argument("--matrix", default=str(META_TRAINING_MATRIX_PATH))
    parser.add_argument("--target-col", default="target")
    args = parser.parse_args()
    train_meta_model(args.matrix, target_col=args.target_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

