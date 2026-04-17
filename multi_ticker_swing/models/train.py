"""
XGBoost multi-class trainer for the 30m multi-ticker swing pipeline.

Target classes:
  0 = short  |  1 = neutral  |  2 = long

Training flow:
  1. Build training matrix (--stage matrix first)
  2. OOF evaluation on the 70% train split (5 sequential folds)
     → catches overfit before committing to a full fit
  3. Full fit on all training data with val as eval_set
  4. Hold-out test evaluation

Sample weights:
  - Soft label weights from the label stage (core=1.0, neighbor=0.75, conflict=0.0)
  - Neutral bars downweighted by NEUTRAL_WEIGHT_FACTOR to balance classes

Usage:
  python -m multi_ticker_swing.models.train [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("xgboost required: pip install xgboost") from e

from sklearn.metrics import accuracy_score, classification_report, log_loss

from multi_ticker_swing.config.pipeline_config import (
    EVAL_METRICS_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NEUTRAL_WEIGHT_FACTOR,
    OOF_N_FOLDS,
    TRAINING_MATRIX,
    TRAIN_FRAC,
    VAL_FRAC,
    XGBOOST_CONFIG,
)
from multi_ticker_swing.data.load_data import load_training_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_ts_col(df: pd.DataFrame) -> str:
    for name in ("timestamp", "index", "t", "time", "date"):
        if name in df.columns:
            return name
    return df.columns[0]


def _time_split(df: pd.DataFrame, ts_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict time-based 70/15/15 split."""
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    t1 = int(n * TRAIN_FRAC)
    t2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    train, val, test = df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()
    logger.info("Split  train=%d  val=%d  test=%d  (%.0f/%.0f/%.0f%%)",
                len(train), len(val), len(test),
                100*len(train)/n, 100*len(val)/n, 100*len(test)/n)
    return train, val, test


def _build_sample_weights(
    y: np.ndarray,
    soft_w: np.ndarray,
    neutral_factor: float = NEUTRAL_WEIGHT_FACTOR,
) -> np.ndarray:
    """
    Final sample weight = soft_label_weight × class_factor.

    Neutral bars (class 1) are downweighted by neutral_factor to prevent
    the model from predicting neutral for everything (~56% of bars).
    Conflict bars (soft_w == 0) stay at 0 and are excluded from training.
    """
    class_factor = np.where(y == 1, neutral_factor, 1.0)
    return (soft_w * class_factor).astype(np.float32)


def _compute_metrics(y_true: np.ndarray, proba: np.ndarray, split: str) -> dict:
    preds = np.argmax(proba, axis=1)
    acc   = accuracy_score(y_true, preds)
    ll    = log_loss(y_true, proba, labels=[0, 1, 2])

    report = classification_report(
        y_true, preds,
        target_names=["short", "neutral", "long"],
        output_dict=True, zero_division=0,
    )

    long_mask  = preds == 2
    short_mask = preds == 0
    long_wr  = float((y_true[long_mask]  == 2).mean()) if long_mask.any()  else float("nan")
    short_wr = float((y_true[short_mask] == 0).mean()) if short_mask.any() else float("nan")

    logger.info(
        "[%s]  acc=%.4f  log_loss=%.4f  "
        "long  prec=%.3f rec=%.3f wr=%.3f (n=%d)  "
        "short prec=%.3f rec=%.3f wr=%.3f (n=%d)",
        split, acc, ll,
        report.get("long",  {}).get("precision", 0),
        report.get("long",  {}).get("recall",    0),
        long_wr  if not np.isnan(long_wr)  else 0, int(long_mask.sum()),
        report.get("short", {}).get("precision", 0),
        report.get("short", {}).get("recall",    0),
        short_wr if not np.isnan(short_wr) else 0, int(short_mask.sum()),
    )

    return {
        f"{split}_accuracy":        round(acc, 4),
        f"{split}_log_loss":        round(ll, 4),
        f"{split}_long_precision":  round(report.get("long",  {}).get("precision", 0), 4),
        f"{split}_long_recall":     round(report.get("long",  {}).get("recall",    0), 4),
        f"{split}_long_wr":         round(long_wr,  4) if not np.isnan(long_wr)  else None,
        f"{split}_long_n":          int(long_mask.sum()),
        f"{split}_short_precision": round(report.get("short", {}).get("precision", 0), 4),
        f"{split}_short_recall":    round(report.get("short", {}).get("recall",    0), 4),
        f"{split}_short_wr":        round(short_wr, 4) if not np.isnan(short_wr) else None,
        f"{split}_short_n":         int(short_mask.sum()),
    }


def _make_clf(xgb_config: dict) -> XGBClassifier:
    cfg   = {k: v for k, v in xgb_config.items() if k != "early_stopping_rounds"}
    return XGBClassifier(**cfg, verbosity=1)


# ---------------------------------------------------------------------------
# Out-of-fold evaluation
# ---------------------------------------------------------------------------

def run_oof(
    X: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
    xgb_config: dict,
    n_folds: int = OOF_N_FOLDS,
) -> dict:
    """
    Sequential time-series OOF: splits X/y into n_folds sequential chunks.
    Each fold trains on all prior folds, predicts on the current fold.
    Returns aggregate OOF metrics.
    """
    logger.info("=== OOF evaluation (%d sequential folds) ===", n_folds)
    n = len(X)
    fold_size = n // n_folds

    oof_proba = np.zeros((n, 3), dtype=np.float32)
    early = xgb_config.get("early_stopping_rounds", 60)

    for fold in range(1, n_folds + 1):
        val_start = (fold - 1) * fold_size
        val_end   = fold * fold_size if fold < n_folds else n
        train_end = val_start

        if train_end < 100:
            logger.info("Fold %d: not enough training data — skipping", fold)
            continue

        X_tr, y_tr, sw_tr = X[:train_end],       y[:train_end],       sw[:train_end]
        X_vl, y_vl        = X[val_start:val_end], y[val_start:val_end]

        # eval_set from last 15% of training fold for early stopping
        es_start = int(train_end * 0.85)
        X_es, y_es = X_tr[es_start:], y_tr[es_start:]

        clf = _make_clf(xgb_config)
        clf.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_es, y_es)],
            early_stopping_rounds=early,
            verbose=False,
        )

        oof_proba[val_start:val_end] = clf.predict_proba(X_vl)
        best = clf.best_iteration + 1 if hasattr(clf, "best_iteration") else "?"
        logger.info("Fold %d/%d  train=%d  val=%d  best_iter=%s",
                    fold, n_folds, train_end, val_end - val_start, best)

    # Only score folds 2..n (fold 1 has no training data)
    scored_start = fold_size
    oof_metrics = _compute_metrics(
        y[scored_start:], oof_proba[scored_start:], "oof"
    )
    logger.info("=== OOF complete ===")
    return oof_metrics


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def train(
    matrix_path: Path = TRAINING_MATRIX,
    model_path: Path = MODEL_PATH,
    feature_importance_path: Path = FEATURE_IMPORTANCE_PATH,
    eval_metrics_path: Path = EVAL_METRICS_PATH,
    feature_columns: list[str] = FEATURE_COLUMNS,
    xgb_config: dict = XGBOOST_CONFIG,
    force: bool = False,
    skip_oof: bool = False,
) -> XGBClassifier:
    if model_path.exists() and not force:
        logger.info("Model exists at %s — use --force to retrain.", model_path)
        clf = XGBClassifier()
        clf.load_model(str(model_path))
        return clf

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading matrix from %s", matrix_path)
    df = load_training_matrix(matrix_path)

    ts_col = _find_ts_col(df)
    train_df, val_df, test_df = _time_split(df, ts_col)

    avail = [c for c in feature_columns if c in df.columns]
    missing = set(feature_columns) - set(avail)
    if missing:
        logger.warning("Features missing from matrix: %s", sorted(missing))
    logger.info("Using %d / %d features", len(avail), len(feature_columns))

    def _arrays(split_df: pd.DataFrame):
        X  = split_df[avail].values.astype(np.float32)
        y  = split_df["target"].values.astype(int)
        sw = _build_sample_weights(
            y,
            split_df["sample_weight"].values.astype(np.float32)
            if "sample_weight" in split_df.columns
            else np.ones(len(y), np.float32),
        )
        return X, y, sw

    X_train, y_train, sw_train = _arrays(train_df)
    X_val,   y_val,   _        = _arrays(val_df)
    X_test,  y_test,  _        = _arrays(test_df)

    # Label distribution
    for split_name, y_s in [("train", y_train), ("val", y_val), ("test", y_test)]:
        cts = np.bincount(y_s, minlength=3)
        logger.info("[%s] class dist  short=%d  neutral=%d  long=%d  (%.1f%% directional)",
                    split_name, cts[0], cts[1], cts[2],
                    100*(cts[0]+cts[2])/max(1,len(y_s)))

    # ------------------------------------------------------------------
    # 1. OOF on training set
    # ------------------------------------------------------------------
    all_metrics: dict = {"features_used": len(avail)}

    if not skip_oof:
        oof_metrics = run_oof(X_train, y_train, sw_train, xgb_config)
        all_metrics.update(oof_metrics)
    else:
        logger.info("OOF skipped (--skip-oof)")

    # ------------------------------------------------------------------
    # 2. Full fit on training set, early-stop on val
    # ------------------------------------------------------------------
    logger.info("=== Full fit ===")
    early = xgb_config.get("early_stopping_rounds", 60)
    clf = _make_clf(xgb_config)

    clf.fit(
        X_train, y_train,
        sample_weight=sw_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early,
        verbose=50,
    )

    best_n = clf.best_iteration + 1 if hasattr(clf, "best_iteration") else xgb_config.get("n_estimators")
    logger.info("Best iteration: %d", best_n)
    all_metrics["best_iteration"] = best_n

    # ------------------------------------------------------------------
    # 3. Evaluate val + test
    # ------------------------------------------------------------------
    for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = clf.predict_proba(X_s)
        all_metrics.update(_compute_metrics(y_s, proba, split_name))

    # ------------------------------------------------------------------
    # 4. Feature importance
    # ------------------------------------------------------------------
    fi_df = (
        pd.DataFrame({"feature": avail, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("Top 20 features:\n%s", fi_df.head(20).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    clf.save_model(str(model_path))
    fi_df.to_csv(feature_importance_path, index=False)
    with open(eval_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Model   → %s", model_path)
    logger.info("FI      → %s", feature_importance_path)
    logger.info("Metrics → %s", eval_metrics_path)
    return clf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 30m swing XGBoost model.")
    p.add_argument("--matrix",   default=str(TRAINING_MATRIX))
    p.add_argument("--force",    action="store_true")
    p.add_argument("--skip-oof", action="store_true",
                   help="Skip OOF evaluation and go straight to full fit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        matrix_path=Path(args.matrix),
        force=args.force,
        skip_oof=args.skip_oof,
    )
