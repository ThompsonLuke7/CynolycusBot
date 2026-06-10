"""Train the social-attention LightGBM specialist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from signals.social_attention.config import (
    FEATURE_IMPORTANCE_PATH,
    LIGHTGBM_CONFIG,
    METRICS_PATH,
    MODEL_MANIFEST_PATH,
    MODEL_PATH,
    TRAINING_MATRIX_PATH,
    ensure_dirs,
)
from signals.social_attention.io import read_table, write_json

EXCLUDE_COLUMNS = {
    "ticker",
    "timestamp",
    "label_timestamp",
    "expansion_target",
    "expansion_score",
    "fwd_max_alpha",
    "fwd_close_return",
    "social_spike_success",
    "top_narrative_cluster_id",
    "narrative_cluster_id",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = ordered.dropna(subset=["timestamp", "social_spike_success"]).sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    if n < 30:
        raise ValueError(f"Need at least 30 labeled rows to train, got {n}.")
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    return ordered.iloc[:i1].copy(), ordered.iloc[i1:i2].copy(), ordered.iloc[i2:].copy()


def _arrays(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).values
    y = df["social_spike_success"].astype(int).values
    return x, y


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(len(y_true))}
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, prob))
        out["log_loss"] = float(log_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6)))
    out["positive_rate"] = float(np.mean(y_true)) if len(y_true) else None
    out["prob_mean"] = float(np.mean(prob)) if len(prob) else None
    return out


def train_lightgbm(
    *,
    matrix_path: Path | str = TRAINING_MATRIX_PATH,
    model_path: Path | str = MODEL_PATH,
    force: bool = False,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required for social_attention training. Install requirements.txt.") from exc

    ensure_dirs()
    model_path = Path(model_path)
    if model_path.exists() and not force:
        raise FileExistsError(f"Model already exists: {model_path}. Pass --force to retrain.")
    matrix = read_table(matrix_path)
    if matrix.empty:
        raise ValueError(f"Training matrix is empty: {matrix_path}")
    train_df, val_df, test_df = _split_time(matrix)
    cols = feature_columns(train_df)
    if not cols:
        raise ValueError("No numeric feature columns found.")
    x_train, y_train = _arrays(train_df, cols)
    x_val, y_val = _arrays(val_df, cols)
    x_test, y_test = _arrays(test_df, cols)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training target has only one class.")

    clf = LGBMClassifier(**LIGHTGBM_CONFIG)
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(model_path))

    test_prob = clf.predict_proba(x_test)[:, 1]
    metrics = {
        "model_kind": "lightgbm",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(cols)),
        "test": _metrics(y_test, test_prob),
    }
    pd.DataFrame({"feature": cols, "importance": clf.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    write_json(metrics, METRICS_PATH)
    write_json(
        {
            "model_kind": "lightgbm",
            "model_path": str(model_path),
            "matrix_path": str(matrix_path),
            "feature_columns": cols,
            "target_column": "social_spike_success",
            "metrics": metrics,
        },
        MODEL_MANIFEST_PATH,
    )
    return metrics


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train social_attention LightGBM model.")
    parser.add_argument("--matrix", default=str(TRAINING_MATRIX_PATH))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_lightgbm(matrix_path=args.matrix, force=args.force), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

