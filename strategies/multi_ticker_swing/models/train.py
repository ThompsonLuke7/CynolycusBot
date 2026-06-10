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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    force=True)   # force=True reconfigures even if Colab/Jupyter already set up the root logger
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("xgboost required: pip install xgboost") from e

from sklearn.metrics import accuracy_score, classification_report, log_loss

from strategies.multi_ticker_swing.config.pipeline_config import (
    EVAL_METRICS_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NEUTRAL_WEIGHT_FACTOR,
    OOF_N_FOLDS,
    RAW_30M_DIR,
    TRAINING_MATRIX,
    TRAIN_FRAC,
    VAL_FRAC,
    XGBOOST_CONFIG,
)
from strategies.multi_ticker_swing.data.load_data import load_training_matrix


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


def _make_clf(xgb_config: dict, early_stopping_rounds: int | None = None) -> XGBClassifier:
    # XGBoost ≥2.0 requires early_stopping_rounds in the constructor, not fit()
    cfg = {k: v for k, v in xgb_config.items() if k != "early_stopping_rounds"}
    if early_stopping_rounds is not None:
        cfg["early_stopping_rounds"] = early_stopping_rounds
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

        clf = _make_clf(xgb_config, early_stopping_rounds=early)
        clf.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_es, y_es)],
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
    return oof_metrics, oof_proba


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
    oof_proba: np.ndarray | None = None

    if not skip_oof:
        oof_metrics, oof_proba = run_oof(X_train, y_train, sw_train, xgb_config)
        all_metrics.update(oof_metrics)
    else:
        logger.info("OOF skipped (--skip-oof)")

    # ------------------------------------------------------------------
    # 2. Full fit on training set, early-stop on val
    # ------------------------------------------------------------------
    logger.info("=== Full fit ===")
    early = xgb_config.get("early_stopping_rounds", 60)
    clf = _make_clf(xgb_config, early_stopping_rounds=early)

    clf.fit(
        X_train, y_train,
        sample_weight=sw_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    best_n = clf.best_iteration + 1 if hasattr(clf, "best_iteration") else xgb_config.get("n_estimators")
    logger.info("Best iteration: %d", best_n)
    all_metrics["best_iteration"] = best_n

    # ------------------------------------------------------------------
    # 3. Evaluate val + test — capture probas for artifact saving
    # ------------------------------------------------------------------
    val_proba  = clf.predict_proba(X_val)
    test_proba = clf.predict_proba(X_test)
    all_metrics.update(_compute_metrics(y_val,  val_proba,  "val"))
    all_metrics.update(_compute_metrics(y_test, test_proba, "test"))

    # Full-dataset probas (train + val + test) for downstream use
    X_all   = np.concatenate([X_train, X_val, X_test], axis=0)
    full_proba = clf.predict_proba(X_all)

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
    # 5. Save model + metrics + features
    # ------------------------------------------------------------------
    clf.save_model(str(model_path))
    fi_df.to_csv(feature_importance_path, index=False)
    with open(eval_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # selected_features.txt
    (MODELS_DIR / "selected_features.txt").write_text("\n".join(avail))

    logger.info("Model   → %s", model_path)
    logger.info("FI      → %s", feature_importance_path)
    logger.info("Metrics → %s", eval_metrics_path)

    # ------------------------------------------------------------------
    # 5b. Save probability artifacts (matching ga_xgboost layout)
    #
    #   p_long_oof_train.npy / p_short_oof_train.npy / p_neutral_oof_train.npy
    #   p_long_test.npy      / p_short_test.npy      / p_neutral_test.npy
    #   p_long_full.npy      / p_short_full.npy      / p_neutral_full.npy
    #   p_swing_probs.parquet  — (timestamp, ticker) indexed, all splits side-by-side
    #   meta.json              — full training provenance
    #   p_swing_oos_manifest.json / p_swing_full_manifest.json
    # ------------------------------------------------------------------
    import datetime as _dt

    run_id = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_swing_30m_train"
    created_at = _dt.datetime.utcnow().isoformat() + "Z"

    # .npy files — shape (n_rows, 3): columns are [short, neutral, long]
    # OOF: only train split rows; NaN for fold-1 rows (no training data)
    if oof_proba is not None:
        np.save(str(MODELS_DIR / "p_long_oof_train.npy"),    oof_proba[:, 2])
        np.save(str(MODELS_DIR / "p_short_oof_train.npy"),   oof_proba[:, 0])
        np.save(str(MODELS_DIR / "p_neutral_oof_train.npy"), oof_proba[:, 1])

    np.save(str(MODELS_DIR / "p_long_test.npy"),    test_proba[:, 2])
    np.save(str(MODELS_DIR / "p_short_test.npy"),   test_proba[:, 0])
    np.save(str(MODELS_DIR / "p_neutral_test.npy"), test_proba[:, 1])

    np.save(str(MODELS_DIR / "p_long_full.npy"),    full_proba[:, 2])
    np.save(str(MODELS_DIR / "p_short_full.npy"),   full_proba[:, 0])
    np.save(str(MODELS_DIR / "p_neutral_full.npy"), full_proba[:, 1])

    # Proba parquet — indexed by (timestamp, ticker), all splits in one file
    # Rows outside a split get NaN for that split's columns.
    def _proba_frame(split_df: pd.DataFrame, proba: np.ndarray | None,
                     suffix: str) -> pd.DataFrame:
        out = split_df[["timestamp", "ticker"]].copy() if "ticker" in split_df.columns \
              else split_df[["timestamp"]].copy()
        if proba is not None:
            out[f"p_long_{suffix}"]    = proba[:, 2]
            out[f"p_short_{suffix}"]   = proba[:, 0]
            out[f"p_neutral_{suffix}"] = proba[:, 1]
        else:
            for col in (f"p_long_{suffix}", f"p_short_{suffix}", f"p_neutral_{suffix}"):
                out[col] = np.nan
        return out

    oof_frame  = _proba_frame(train_df, oof_proba,  "oof_train")
    test_frame = _proba_frame(test_df,  test_proba, "test")
    full_all   = pd.concat([train_df, val_df, test_df], axis=0)[
        ["timestamp"] + (["ticker"] if "ticker" in train_df.columns else [])
    ].copy()
    full_all["p_long_full"]    = full_proba[:, 2]
    full_all["p_short_full"]   = full_proba[:, 0]
    full_all["p_neutral_full"] = full_proba[:, 1]

    proba_df = (
        oof_frame
        .merge(test_frame, on=["timestamp", "ticker"] if "ticker" in oof_frame.columns
               else ["timestamp"], how="outer")
        .merge(full_all,   on=["timestamp", "ticker"] if "ticker" in oof_frame.columns
               else ["timestamp"], how="outer")
    )
    idx_cols = ["timestamp", "ticker"] if "ticker" in proba_df.columns else ["timestamp"]
    proba_df = proba_df.set_index(idx_cols).sort_index()
    proba_path = MODELS_DIR / "p_swing_probs.parquet"
    proba_df.to_parquet(proba_path)
    logger.info("Proba parquet → %s  (%d rows)", proba_path, len(proba_df))

    # meta.json
    from strategies.multi_ticker_swing.config.pipeline_config import (
        NEUTRAL_WEIGHT_FACTOR, OOF_N_FOLDS, TRAIN_FRAC, VAL_FRAC,
        SWING_LABEL_30M_CONFIG,
    )
    n_tickers = int(train_df["ticker"].nunique()) if "ticker" in train_df.columns else 1
    meta = {
        "run_id":              run_id,
        "created_at_utc":      created_at,
        "dataset_name":        "30m_multi_ticker",
        "n_tickers":           n_tickers,
        "label_mode":          "soft_swing_zone",
        "classes":             ["short", "neutral", "long"],
        "selected_features":   len(avail),
        "feature_names":       avail,
        "neutral_weight_factor": NEUTRAL_WEIGHT_FACTOR,
        "oof_n_folds":         OOF_N_FOLDS,
        "train_frac":          TRAIN_FRAC,
        "val_frac":            VAL_FRAC,
        "best_iteration":      best_n,
        "swing_label_config":  SWING_LABEL_30M_CONFIG,
        "xgb_params": {k: v for k, v in XGBOOST_CONFIG.items()},
        "eval_metrics":        all_metrics,
        "artifact_paths": {
            "model":              str(model_path),
            "feature_importance": str(feature_importance_path),
            "eval_metrics":       str(eval_metrics_path),
            "proba_parquet":      str(proba_path),
            "selected_features":  str(MODELS_DIR / "selected_features.txt"),
        },
    }
    meta_path = MODELS_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Meta    → %s", meta_path)

    # Manifests
    oos_manifest = {
        "run_id": run_id, "created_at_utc": created_at,
        "dataset_name": "30m_multi_ticker", "label_mode": "soft_swing_zone",
        "artifact_kind": "swing_30m_multiclass_oos_probabilities",
        "classes": ["short", "neutral", "long"],
        "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
        "oof_train_npy": {
            "p_long":    str(MODELS_DIR / "p_long_oof_train.npy"),
            "p_short":   str(MODELS_DIR / "p_short_oof_train.npy"),
            "p_neutral": str(MODELS_DIR / "p_neutral_oof_train.npy"),
        },
        "test_npy": {
            "p_long":    str(MODELS_DIR / "p_long_test.npy"),
            "p_short":   str(MODELS_DIR / "p_short_test.npy"),
            "p_neutral": str(MODELS_DIR / "p_neutral_test.npy"),
        },
        "oos_parquet_path": str(proba_path),
    }
    full_manifest = {
        "run_id": run_id, "created_at_utc": created_at,
        "dataset_name": "30m_multi_ticker", "label_mode": "soft_swing_zone",
        "artifact_kind": "swing_30m_multiclass_full_probabilities",
        "classes": ["short", "neutral", "long"],
        "row_count": len(full_proba),
        "full_npy": {
            "p_long":    str(MODELS_DIR / "p_long_full.npy"),
            "p_short":   str(MODELS_DIR / "p_short_full.npy"),
            "p_neutral": str(MODELS_DIR / "p_neutral_full.npy"),
        },
        "full_parquet_path": str(proba_path),
    }
    with open(MODELS_DIR / "p_swing_oos_manifest.json", "w") as f:
        json.dump(oos_manifest, f, indent=2)
    with open(MODELS_DIR / "p_swing_full_manifest.json", "w") as f:
        json.dump(full_manifest, f, indent=2)
    logger.info("Manifests saved → %s", MODELS_DIR)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    _plot_results(
        y_train=y_train,
        y_test=y_test,
        test_proba=test_proba,
        test_df=test_df,
        fi_df=fi_df,
        oof_metrics=all_metrics,
        plots_dir=MODELS_DIR / "plots",
    )

    return clf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_proba: np.ndarray,
    test_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    oof_metrics: dict,
    plots_dir: Path,
) -> None:
    """
    Generate and save four diagnostic plots after training:
      1. Probability distributions by true class (separation quality)
      2. OOF vs val vs test metric summary bar chart
      3. Test-set swing setup signals over time (p_long / p_short time series)
      4. Feature importance (top 30)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plots_dir.mkdir(parents=True, exist_ok=True)
    DARK_BG  = "#0d1117"
    GRID_COL = "#21262d"
    TEXT_COL = "#c9d1d9"
    GREEN    = "#26a641"
    RED      = "#f85149"
    BLUE     = "#58a6ff"
    AMBER    = "#e3b341"

    def _style(fig, axes):
        fig.patch.set_facecolor(DARK_BG)
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors=TEXT_COL)
            ax.xaxis.label.set_color(TEXT_COL)
            ax.yaxis.label.set_color(TEXT_COL)
            ax.title.set_color(TEXT_COL)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COL)
            ax.grid(color=GRID_COL, linewidth=0.5)

    # ---- 1. Probability separation by true class -------------------------
    # Top row: raw P(short) / P(neutral) / P(long) by true class
    # Bottom row: directional-conditional P(long | directional) — neutrals excluded.
    #   P(long | dir) = P(long) / (P(long) + P(short))
    #   This normalises out the neutral mass, matching how the live system uses the signal.
    dir_sum  = test_proba[:, 0] + test_proba[:, 2]
    dir_safe = np.where(dir_sum > 0, dir_sum, 1.0)
    p_long_dir  = test_proba[:, 2] / dir_safe   # P(long | directional)
    p_short_dir = test_proba[:, 0] / dir_safe   # P(short | directional)

    # Only directional bars for the conditional plot
    dir_mask   = y_test != 1
    long_mask  = y_test == 2
    short_mask = y_test == 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: raw probabilities (all true classes)
    class_names = ["short", "neutral", "long"]
    true_colors = [RED, AMBER, GREEN]
    for col_idx, cls_name in enumerate(class_names):
        ax = axes[0, col_idx]
        for true_cls, tc_name, tc_color in zip([0, 1, 2], class_names, true_colors):
            mask = y_test == true_cls
            ax.hist(test_proba[mask, col_idx], bins=50, range=(0, 1),
                    alpha=0.55, color=tc_color,
                    label=f"true={tc_name} (n={mask.sum():,})", density=True)
        ax.set_title(f"raw P({cls_name}) — all bars")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)

    # Row 1: directional-conditional (neutrals excluded — mirrors live signal)
    # Left: P(long|dir) for true=long vs true=short only
    for ax, vals, title, note in [
        (axes[1, 0], p_short_dir, "P(short | directional)", "= P(short)/(P(long)+P(short))"),
        (axes[1, 1], p_long_dir,  "P(long  | directional)", "= P(long) /(P(long)+P(short))"),
    ]:
        ax.hist(vals[short_mask], bins=50, range=(0, 1), alpha=0.6, color=RED,
                label=f"true=short (n={short_mask.sum():,})", density=True)
        ax.hist(vals[long_mask],  bins=50, range=(0, 1), alpha=0.6, color=GREEN,
                label=f"true=long  (n={long_mask.sum():,})",  density=True)
        ax.axvline(0.5, color=BLUE, lw=1.0, ls="--", label="0.50")
        ax.set_title(f"{title}\n{note}")
        ax.set_xlabel("directional-conditional probability")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)

    # Right panel of row 1: precision curve vs directional threshold
    thresholds = np.linspace(0.5, 0.95, 50)
    long_prec_curve  = []
    short_prec_curve = []
    long_n_curve     = []
    short_n_curve    = []
    for thr in thresholds:
        pred_long  = p_long_dir  >= thr
        pred_short = p_short_dir >= thr
        lp = (y_test[pred_long]  == 2).mean() if pred_long.any()  else float("nan")
        sp = (y_test[pred_short] == 0).mean() if pred_short.any() else float("nan")
        long_prec_curve.append(lp)
        short_prec_curve.append(sp)
        long_n_curve.append(pred_long.sum())
        short_n_curve.append(pred_short.sum())

    ax_prec = axes[1, 2]
    ax_n    = ax_prec.twinx()
    ax_prec.plot(thresholds, long_prec_curve,  color=GREEN, lw=1.8, label="long precision")
    ax_prec.plot(thresholds, short_prec_curve, color=RED,   lw=1.8, label="short precision")
    ax_n.plot(thresholds, long_n_curve,  color=GREEN, lw=0.8, ls="--", alpha=0.5)
    ax_n.plot(thresholds, short_n_curve, color=RED,   lw=0.8, ls="--", alpha=0.5)
    ax_prec.axhline(0.5, color=BLUE, lw=0.8, ls=":")
    ax_prec.set_title("Precision vs directional threshold\n(dashed = signal count, right axis)")
    ax_prec.set_xlabel("P(long|dir) or P(short|dir) threshold")
    ax_prec.set_ylabel("precision", color=TEXT_COL)
    ax_n.set_ylabel("# signals", color="#888888")
    ax_n.tick_params(colors="#888888")
    ax_prec.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    ax_prec.set_ylim(0, 1)

    _style(fig, axes.flatten())
    fig.suptitle(
        "Test set — raw probabilities (top) vs directional-conditional signal (bottom)\n"
        "Bottom row excludes neutral bars — mirrors how the live system uses P(long) and P(short)",
        color=TEXT_COL, fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(plots_dir / "prob_separation.png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)

    # ---- 2. Metric summary (OOF / val / test) ----------------------------
    splits   = ["oof", "val", "test"]
    metrics  = ["accuracy", "long_precision", "long_wr", "short_precision", "short_wr"]
    m_labels = ["Accuracy", "Long prec", "Long WR", "Short prec", "Short WR"]
    m_colors = [BLUE, GREEN, GREEN, RED, RED]
    x        = np.arange(len(splits))
    width    = 0.15

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (m, label, color) in enumerate(zip(metrics, m_labels, m_colors)):
        vals = []
        for sp in splits:
            key = f"{sp}_{m}"
            v = oof_metrics.get(key)
            vals.append(float(v) if v is not None else 0.0)
        offset = (i - len(metrics) / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.8)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=TEXT_COL)

    ax.set_xticks(x)
    ax.set_xticklabels(["OOF (train)", "Val", "Test"], color=TEXT_COL)
    ax.set_ylim(0, 1.05)
    ax.set_title("Key metrics across splits", color=TEXT_COL)
    ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, edgecolor=GRID_COL)
    _style(fig, [ax])
    plt.tight_layout()
    fig.savefig(plots_dir / "metric_summary.png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)

    # ---- 3. Test-set candle + proba overlay (3 random tickers) ----------
    # Reuses _plot_candles from the shared plotting library.
    # Top panel:    30m candlestick bars (last 60 bars of each ticker's test window)
    # Middle panel: P(long) green line + P(short) red line with threshold
    # Bottom panel: true label colour strip (green=long, red=short, grey=neutral)
    try:
        from Data.plots.plots import _plot_candles, _extract_ohlc
    except ImportError:
        _plot_candles = None

    # Attach probas + true labels to test_df rows so we can filter by ticker
    _tdf = test_df.copy()
    _tdf["_p_long"]  = test_proba[:, 2]
    _tdf["_p_short"] = test_proba[:, 0]
    _tdf["_y"]       = y_test

    # Prefer diverse, liquid, high-beta tickers for representative plots.
    # Falls back to random picks for any slots not covered by the preferred list.
    PREFERRED_PLOT_TICKERS = [
        "NVDA", "LLY", "AMD", "AAPL", "META",      # mega-cap tech / pharma
        "SMCI", "CRWD", "MSTR", "SNOW", "NFLX",     # high-beta growth
        "AMGN", "AVGO", "TSM", "MSFT",              # large-cap diversifiers
    ]
    N_PLOT_TICKERS = 4

    if "ticker" in _tdf.columns:
        tickers_in_test = set(_tdf["ticker"].unique().tolist())
        # Take as many preferred tickers as are available, then pad with random picks
        sample_tickers = [t for t in PREFERRED_PLOT_TICKERS if t in tickers_in_test][:N_PLOT_TICKERS]
        if len(sample_tickers) < N_PLOT_TICKERS:
            remaining = list(tickers_in_test - set(sample_tickers))
            rng = np.random.default_rng(42)
            extra = rng.choice(remaining,
                               size=min(N_PLOT_TICKERS - len(sample_tickers), len(remaining)),
                               replace=False).tolist()
            sample_tickers += extra
    else:
        sample_tickers = []

    PLOT_BARS = 80   # last N 30m bars of each ticker's test window

    for ticker in sample_tickers:
        t_rows = _tdf[_tdf["ticker"] == ticker].copy()
        if "timestamp" in t_rows.columns:
            t_rows = t_rows.sort_values("timestamp")
            ts_series = pd.to_datetime(t_rows["timestamp"])
        else:
            ts_series = None

        # Load raw OHLCV and slice to the same window
        raw_path = RAW_30M_DIR / f"{ticker}.parquet"
        if not raw_path.exists() or _plot_candles is None:
            continue

        raw = pd.read_parquet(raw_path)
        raw.columns = [c.lower() for c in raw.columns]
        if "timestamp" in raw.columns and not isinstance(raw.index, pd.DatetimeIndex):
            raw = raw.set_index("timestamp")
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")

        # Align: keep only rows whose timestamp appears in the test slice
        if ts_series is not None:
            ts_utc = ts_series.dt.tz_localize("UTC") if ts_series.dt.tz is None else ts_series.dt.tz_convert("UTC")
            raw = raw.loc[raw.index.isin(ts_utc)].copy()
            t_rows = t_rows.set_index(ts_series.values)

        raw = raw.tail(PLOT_BARS)
        t_rows = t_rows.tail(PLOT_BARS)

        if len(raw) < 10:
            continue

        n    = len(raw)
        pos  = np.arange(n)

        # Tick labels every ~26 bars ≈ 1 trading day
        tick_step = max(1, n // 8)
        tick_pos  = pos[::tick_step]
        tick_lbl  = [raw.index[i].astimezone(
                        ZoneInfo("America/New_York")
                     ).strftime("%m/%d %H:%M") for i in range(0, n, tick_step)]


        fig = plt.figure(figsize=(20, 10))
        gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 0.4], hspace=0.08)
        ax_price = fig.add_subplot(gs[0])
        ax_prob  = fig.add_subplot(gs[1], sharex=ax_price)
        ax_label = fig.add_subplot(gs[2], sharex=ax_price)
        fig.patch.set_facecolor(DARK_BG)

        # -- candles --
        o = raw["open"].to_numpy(float)
        h = raw["high"].to_numpy(float)
        lo_arr = raw["low"].to_numpy(float)
        c_arr  = raw["close"].to_numpy(float)
        _plot_candles(ax_price, pos, o, h, lo_arr, c_arr,
                      wick_color="#555555", up_color=GREEN, down_color=RED)
        ax_price.set_ylabel("Price", color=TEXT_COL)
        ax_price.set_title(f"{ticker}  |  test set (last {PLOT_BARS} bars)  |  candle + model probabilities",
                           color=TEXT_COL, fontsize=10)

        # -- probabilities: directional-conditional (same as live system) --
        # P(long|dir) = P(long) / (P(long)+P(short)) — neutralises the neutral mass
        p_long_raw  = t_rows["_p_long"].to_numpy(float)[-n:]
        p_short_raw = t_rows["_p_short"].to_numpy(float)[-n:]
        dsum = np.where(p_long_raw + p_short_raw > 0, p_long_raw + p_short_raw, 1.0)
        p_long  = p_long_raw  / dsum
        p_short = p_short_raw / dsum
        ax_prob.fill_between(pos, 0.5, p_long,  where=p_long  > 0.5, alpha=0.30, color=GREEN)
        ax_prob.fill_between(pos, 0.5, p_short, where=p_short > 0.5, alpha=0.30, color=RED)
        ax_prob.plot(pos, p_long,  color=GREEN, lw=1.2, label="P(long | directional)")
        ax_prob.plot(pos, p_short, color=RED,   lw=1.2, label="P(short | directional)")
        ax_prob.axhline(0.50, color=BLUE, lw=0.8, ls="--", label="0.50 threshold")
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel("P(class | directional)", color=TEXT_COL)
        ax_prob.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL,
                       edgecolor=GRID_COL, loc="upper left")

        # -- true label strip --
        y_strip = t_rows["_y"].to_numpy(int)[-n:]
        strip_colors = np.where(y_strip == 2, GREEN, np.where(y_strip == 0, RED, "#333333"))
        for xi, sc in zip(pos, strip_colors):
            ax_label.bar(xi, 1, color=sc, width=1.0, align="center")
        ax_label.set_yticks([])
        ax_label.set_ylabel("true", color=TEXT_COL, fontsize=7)

        # -- shared x ticks --
        ax_label.set_xticks(tick_pos)
        ax_label.set_xticklabels(tick_lbl, rotation=40, ha="right", fontsize=7, color=TEXT_COL)
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_prob.get_xticklabels(),  visible=False)

        for ax in (ax_price, ax_prob, ax_label):
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors=TEXT_COL)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COL)
            ax.grid(color=GRID_COL, linewidth=0.4, alpha=0.6)

        fig.savefig(plots_dir / f"test_candle_proba_{ticker}.png", dpi=130,
                    bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        logger.info("Candle+proba plot → %s", plots_dir / f"test_candle_proba_{ticker}.png")

    # ---- 4. Feature importance top 30 ------------------------------------
    top30 = fi_df.head(30).iloc[::-1]   # reverse so most important is at top
    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(top30["feature"], top30["importance"], color=BLUE, alpha=0.85)
    ax.set_xlabel("importance (gain)")
    ax.set_title("Feature importance — top 30")
    ax.tick_params(axis="y", labelsize=8)
    _style(fig, [ax])
    plt.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)

    logger.info("Plots saved → %s", plots_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 30m swing XGBoost model.")
    p.add_argument("--matrix",     default=str(TRAINING_MATRIX))
    p.add_argument("--force",      action="store_true")
    p.add_argument("--skip-oof",   action="store_true",
                   help="Skip OOF evaluation and go straight to full fit")
    p.add_argument("--plots-only", action="store_true",
                   help="Skip training — load saved model + matrix and regenerate plots only")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.plots_only:
        # Load saved model and regenerate plots without retraining
        logger.info("--plots-only: loading saved model from %s", MODEL_PATH)
        clf = XGBClassifier()
        clf.load_model(str(MODEL_PATH))

        df = load_training_matrix(Path(args.matrix))
        ts_col = _find_ts_col(df)
        train_df, val_df, test_df = _time_split(df, ts_col)
        avail = [c for c in FEATURE_COLUMNS if c in df.columns]

        X_test = test_df[avail].values.astype(np.float32)
        y_test = test_df["target"].values.astype(int)
        test_proba = clf.predict_proba(X_test)

        fi_df = (
            pd.DataFrame({"feature": avail, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        with open(EVAL_METRICS_PATH) as f:
            saved_metrics = json.load(f)

        _plot_results(
            y_train=train_df["target"].values.astype(int),
            y_test=y_test,
            test_proba=test_proba,
            test_df=test_df,
            fi_df=fi_df,
            oof_metrics=saved_metrics,
            plots_dir=MODELS_DIR / "plots",
        )
        logger.info("Plots regenerated → %s", MODELS_DIR / "plots")
    else:
        train(
            matrix_path=Path(args.matrix),
            force=args.force,
            skip_oof=args.skip_oof,
        )
