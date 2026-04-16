"""
XGBoost multi-class model trainer for the multi-ticker swing pipeline.

Target classes:
  0 = short winner  (tb_label = -1)
  1 = neutral / chop (tb_label = 0)
  2 = long winner   (tb_label = +1)

Usage:
  python multi_ticker_swing/models/train.py [--matrix PATH] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("xgboost is required: pip install xgboost") from e

from multi_ticker_swing.config.pipeline_config import (
    EVAL_METRICS_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    MODELS_DIR,
    TRAINING_MATRIX,
    TRAIN_FRAC,
    VAL_FRAC,
    XGBOOST_CONFIG,
)
from multi_ticker_swing.data.load_data import load_training_matrix


# ---------------------------------------------------------------------------
# Time-based split
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    ts_col: str,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict time-based split: train | val | test by timestamp.
    No data leakage between splits.
    """
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    logger.info(
        "Split: train=%d  val=%d  test=%d  "
        "(%.1f%% / %.1f%% / %.1f%%)",
        len(train), len(val), len(test),
        100 * len(train) / n, 100 * len(val) / n, 100 * len(test) / n,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, split: str) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        log_loss,
    )

    acc = accuracy_score(y_true, y_pred)
    ll  = log_loss(y_true, proba, labels=[0, 1, 2])

    report = classification_report(
        y_true, y_pred,
        target_names=["short", "neutral", "long"],
        output_dict=True,
        zero_division=0,
    )

    # Win rate: among predicted long (class 2) how often is target=2?
    long_mask  = y_pred == 2
    short_mask = y_pred == 0
    long_wr  = float((y_true[long_mask] == 2).mean())  if long_mask.any()  else float("nan")
    short_wr = float((y_true[short_mask] == 0).mean()) if short_mask.any() else float("nan")

    metrics = {
        f"{split}_accuracy":     round(acc, 4),
        f"{split}_log_loss":     round(ll, 4),
        f"{split}_long_wr":      round(long_wr, 4) if not np.isnan(long_wr) else None,
        f"{split}_short_wr":     round(short_wr, 4) if not np.isnan(short_wr) else None,
        f"{split}_long_precision":  round(report.get("long", {}).get("precision", 0), 4),
        f"{split}_short_precision": round(report.get("short", {}).get("precision", 0), 4),
        f"{split}_long_recall":     round(report.get("long", {}).get("recall", 0), 4),
        f"{split}_short_recall":    round(report.get("short", {}).get("recall", 0), 4),
        f"{split}_long_n":      int(long_mask.sum()),
        f"{split}_short_n":     int(short_mask.sum()),
    }
    logger.info("[%s]  acc=%.4f  log_loss=%.4f  long_wr=%.4f (n=%d)  short_wr=%.4f (n=%d)",
                split, acc, ll, long_wr or 0, metrics[f"{split}_long_n"],
                short_wr or 0, metrics[f"{split}_short_n"])
    return metrics


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    matrix_path: Path = TRAINING_MATRIX,
    model_path: Path = MODEL_PATH,
    feature_importance_path: Path = FEATURE_IMPORTANCE_PATH,
    eval_metrics_path: Path = EVAL_METRICS_PATH,
    feature_columns: list[str] = FEATURE_COLUMNS,
    xgb_config: dict = XGBOOST_CONFIG,
    force: bool = False,
) -> XGBClassifier:
    """
    Load training matrix, time-split, train XGBoost multiclass model,
    evaluate on val + test, save artefacts.
    """
    if model_path.exists() and not force:
        logger.info("Model already exists at %s — skipping. Use --force to retrain.", model_path)
        clf = XGBClassifier()
        clf.load_model(str(model_path))
        return clf

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    logger.info("Loading training matrix from %s", matrix_path)
    df = load_training_matrix(matrix_path)

    ts_col = _find_ts_col(df)
    train_df, val_df, test_df = time_split(df, ts_col)

    avail_feats = [c for c in feature_columns if c in df.columns]
    logger.info("Using %d features", len(avail_feats))

    X_train = train_df[avail_feats].values.astype(np.float32)
    y_train = train_df["target"].values.astype(int)
    X_val   = val_df[avail_feats].values.astype(np.float32)
    y_val   = val_df["target"].values.astype(int)
    X_test  = test_df[avail_feats].values.astype(np.float32)
    y_test  = test_df["target"].values.astype(int)

    # ------------------------------------------------------------------
    # Class weights to handle class imbalance (neutral dominates)
    # ------------------------------------------------------------------
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = {cls: len(y_train) / (3 * cnt) for cls, cnt in zip(unique, counts)}
    sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])

    # ------------------------------------------------------------------
    # Build XGBClassifier
    # ------------------------------------------------------------------
    cfg = {k: v for k, v in xgb_config.items() if k not in ("early_stopping_rounds",)}
    early = xgb_config.get("early_stopping_rounds", 40)

    clf = XGBClassifier(
        **cfg,
        use_label_encoder=False,
        verbosity=1,
    )

    logger.info("Training XGBoost (%d estimators, max_depth=%d)...",
                cfg.get("n_estimators", 400), cfg.get("max_depth", 5))

    clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early,
        verbose=50,
    )

    best_n = clf.best_iteration + 1 if hasattr(clf, "best_iteration") else cfg.get("n_estimators")
    logger.info("Best iteration: %d", best_n)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    all_metrics: dict = {"best_iteration": best_n, "features_used": len(avail_feats)}

    for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba   = clf.predict_proba(X_s)
        preds   = clf.predict(X_s)
        metrics = compute_metrics(y_s, preds, proba, split_name)
        all_metrics.update(metrics)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    importance = clf.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": avail_feats, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("Top 20 features:\n%s", fi_df.head(20).to_string(index=False))

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    clf.save_model(str(model_path))
    fi_df.to_csv(feature_importance_path, index=False)
    with open(eval_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Model saved → %s", model_path)
    logger.info("Feature importance → %s", feature_importance_path)
    logger.info("Eval metrics → %s", eval_metrics_path)
    return clf


def _find_ts_col(df: pd.DataFrame) -> str:
    for name in ("timestamp", "index", "t", "time", "date"):
        if name in df.columns:
            return name
    return df.columns[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-ticker swing XGBoost model.")
    p.add_argument("--matrix", default=str(TRAINING_MATRIX), help="Path to training_matrix.parquet")
    p.add_argument("--force",  action="store_true", help="Retrain even if model exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(matrix_path=Path(args.matrix), force=args.force)
