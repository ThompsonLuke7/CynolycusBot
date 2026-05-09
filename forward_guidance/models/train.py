"""Train forward-guidance outperformance models with date-based validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from forward_guidance.config import (
    EVAL_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURES_PATH,
    LIGHTGBM_CONFIG,
    LGB_MODEL_PATH,
    MODEL_META_PATH,
    OOF_N_FOLDS,
    PRIMARY_PROBABILITY,
    TRAINING_MATRIX,
    TRAIN_FRAC,
    VAL_FRAC,
    XGBOOST_CONFIG,
    XGB_MODEL_PATH,
    ensure_data_dirs,
)
from forward_guidance.features.build_matrix import feature_columns
from forward_guidance.utils.io import json_safe, write_json

logger = logging.getLogger(__name__)


def load_training_matrix(path: Path | str = TRAINING_MATRIX) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training matrix not found: {p}")
    return pd.read_parquet(p)


def _time_col(df: pd.DataFrame) -> str:
    for col in ("signal_timestamp", "reaction_date", "earnings_date", "date", "timestamp"):
        if col in df.columns:
            return col
    raise ValueError("No date-like column found for time split.")


def split_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts_col = _time_col(df)
    ordered = df.copy()
    ordered[ts_col] = pd.to_datetime(ordered[ts_col], utc=True, errors="coerce")
    ordered = ordered.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    n = len(ordered)
    if n < 10:
        raise ValueError(f"Need at least 10 labeled events for split, got {n}.")
    i1 = int(n * TRAIN_FRAC)
    i2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    return ordered.iloc[:i1].copy(), ordered.iloc[i1:i2].copy(), ordered.iloc[i2:].copy()


def walk_forward_splits(df: pd.DataFrame, n_folds: int = OOF_N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    ts_col = _time_col(df)
    ordered = df.copy()
    ordered[ts_col] = pd.to_datetime(ordered[ts_col], utc=True, errors="coerce")
    ordered = ordered.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    fold_size = max(len(ordered) // n_folds, 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(1, n_folds):
        train_end = fold * fold_size
        val_end = min((fold + 1) * fold_size, len(ordered))
        if train_end < 5 or val_end <= train_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    return splits


def _arrays(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).values
    y = df["target"].astype(int).values
    return X, y


def probability_metrics(df: pd.DataFrame, probability_col: str = PRIMARY_PROBABILITY) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if df.empty or probability_col not in df.columns:
        return out
    scored = df.dropna(subset=[probability_col]).copy()
    if scored.empty:
        return out
    if "target" in scored.columns and scored["target"].nunique() > 1:
        out["auc"] = float(roc_auc_score(scored["target"].astype(int), scored[probability_col]))
        out["log_loss"] = float(log_loss(scored["target"].astype(int), scored[probability_col].clip(1e-6, 1 - 1e-6)))
    ret_col = "fwd_60d_excess_ret_vs_sector"
    if ret_col in scored.columns:
        try:
            scored["prob_decile"] = pd.qcut(scored[probability_col], 10, labels=False, duplicates="drop") + 1
            deciles = scored.groupby("prob_decile")[ret_col].agg(["count", "mean", "median"]).reset_index()
            out["decile_forward_returns"] = deciles.to_dict(orient="records")
        except ValueError:
            pass
        top = scored.loc[scored[probability_col] >= scored[probability_col].quantile(0.9)]
        if not top.empty:
            returns = top[ret_col].dropna()
            out["top_bucket_count"] = int(len(top))
            out["top_bucket_mean_return"] = float(returns.mean()) if not returns.empty else None
            out["top_bucket_hit_rate"] = float((returns > 0).mean()) if not returns.empty else None
            out["top_bucket_expectancy"] = float(returns.mean()) if not returns.empty else None
            std = float(returns.std()) if len(returns) > 1 else float("nan")
            out["top_bucket_sharpe_like"] = float(returns.mean() / std) if std and std == std else None
    return out


def _save_feature_importance(names: list[str], values: np.ndarray) -> None:
    df = pd.DataFrame({"feature": names, "importance": values})
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("importance", ascending=False).to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def train_xgboost(matrix: pd.DataFrame, *, force: bool = False) -> dict[str, Any]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost training.") from exc

    if XGB_MODEL_PATH.exists() and not force:
        raise FileExistsError(f"Model already exists: {XGB_MODEL_PATH}. Pass --force to retrain.")

    train_df, val_df, test_df = split_train_val_test(matrix)
    cols = feature_columns(matrix)
    X_train, y_train = _arrays(train_df, cols)
    X_val, y_val = _arrays(val_df, cols)
    X_test, y_test = _arrays(test_df, cols)

    clf = XGBClassifier(**XGBOOST_CONFIG)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(XGB_MODEL_PATH))
    _save_feature_importance(cols, clf.feature_importances_)

    test_scored = test_df.copy()
    test_scored[PRIMARY_PROBABILITY] = clf.predict_proba(X_test)[:, 1]
    metrics = probability_metrics(test_scored)
    metrics.update({"model_kind": "xgboost", "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df)})
    write_json(metrics, EVAL_METRICS_PATH)
    write_json(
        {
            "model_kind": "xgboost",
            "model_path": str(XGB_MODEL_PATH),
            "feature_columns": cols,
            "probability_col": PRIMARY_PROBABILITY,
            "source_matrix": str(TRAINING_MATRIX),
            "metrics": metrics,
        },
        MODEL_META_PATH,
    )
    logger.info("Saved XGBoost model to %s", XGB_MODEL_PATH)
    return metrics


def train_lightgbm(matrix: pd.DataFrame, *, force: bool = False) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required for LightGBM training.") from exc

    if LGB_MODEL_PATH.exists() and not force:
        raise FileExistsError(f"Model already exists: {LGB_MODEL_PATH}. Pass --force to retrain.")

    train_df, val_df, test_df = split_train_val_test(matrix)
    cols = feature_columns(matrix)
    X_train, y_train = _arrays(train_df, cols)
    X_val, y_val = _arrays(val_df, cols)
    X_test, y_test = _arrays(test_df, cols)

    clf = LGBMClassifier(**LIGHTGBM_CONFIG)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    LGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(LGB_MODEL_PATH))
    _save_feature_importance(cols, clf.feature_importances_)

    test_scored = test_df.copy()
    test_scored[PRIMARY_PROBABILITY] = clf.predict_proba(X_test)[:, 1]
    metrics = probability_metrics(test_scored)
    metrics.update({"model_kind": "lightgbm", "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df)})
    write_json(metrics, EVAL_METRICS_PATH)
    write_json(
        {
            "model_kind": "lightgbm",
            "model_path": str(LGB_MODEL_PATH),
            "feature_columns": cols,
            "probability_col": PRIMARY_PROBABILITY,
            "source_matrix": str(TRAINING_MATRIX),
            "metrics": metrics,
        },
        MODEL_META_PATH,
    )
    logger.info("Saved LightGBM model to %s", LGB_MODEL_PATH)
    return metrics


def train_models(*, matrix_path: Path | str = TRAINING_MATRIX, model_kind: str = "xgboost", force: bool = False) -> dict[str, Any]:
    ensure_data_dirs()
    matrix = load_training_matrix(matrix_path)
    if matrix.empty:
        raise ValueError("Training matrix is empty.")
    if "target" not in matrix.columns:
        raise ValueError("Training matrix must include target column.")
    matrix = matrix.loc[matrix["target"].notna()].copy()
    if model_kind == "xgboost":
        return train_xgboost(matrix, force=force)
    if model_kind == "lightgbm":
        return train_lightgbm(matrix, force=force)
    if model_kind == "both":
        return {"xgboost": train_xgboost(matrix, force=force), "lightgbm": train_lightgbm(matrix, force=force)}
    raise ValueError(f"Unknown model_kind: {model_kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train forward-guidance outperformance models.")
    parser.add_argument("--matrix", default=str(TRAINING_MATRIX))
    parser.add_argument("--model-kind", choices=["xgboost", "lightgbm", "both"], default="xgboost")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()), format="%(asctime)s %(levelname)s %(message)s")
    metrics = train_models(matrix_path=args.matrix, model_kind=args.model_kind, force=args.force)
    print(json.dumps(json_safe(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
