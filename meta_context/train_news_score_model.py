"""Train the news catalyst specialist model (catalyst classifier for model 2).

Expects the feature matrix produced by ``meta_context/build_catalyst_training_matrix.py``.
Trains an LightGBM binary classifier on the train split, evaluates on the val
and test splits, and writes the booster to disk.

Usage:
    .venv/bin/python -m meta_context.build_catalyst_training_matrix
    .venv/bin/python -m meta_context.train_news_score_model

This module does NOT auto-train when imported. The training run only happens
inside ``main()`` and requires explicit invocation. Per project rules, the
caller (Claude) cannot run ML training without an explicit user request.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MATRIX = Path("meta_context/data/processed/catalyst_training_matrix.parquet")
DEFAULT_MODEL = Path("meta_context/models/news_catalyst_lgbm.txt")
DEFAULT_METRICS = Path("meta_context/data/processed/news_catalyst_eval.json")


def split_xy(df: pd.DataFrame, split: str, *, target: str = "expansion_label") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta) for a given split label, picking the requested target column.

    Supported target values:
        - expansion_label / target_expansion_10pct : legacy binary, max_forward_return >= 10%
        - target_expansion_5pct                    : lenient binary, max_forward_return >= 5%
        - target_crash_5pct                        : crash binary, forward_10d_return <= -5%
        - target_fwd_10d_reg                       : regression on forward_10d_return
    """
    sub = df[df["split"] == split].copy()
    drop_cols = [
        "record_id",
        "ticker",
        "timestamp",
        "catalyst_family",
        "catalyst_subtype",
        "source_quality",  # tag column — we use sq_* one-hots as features instead
        "expansion_label",
        "target_expansion_10pct",
        "target_expansion_5pct",
        "target_crash_5pct",
        "target_fwd_10d_reg",
        "target_trajectory",
        "target_trajectory_code",
        "max_forward_return",
        "max_drawdown",
        "forward_5d_return",
        "forward_10d_return",
        "split",
    ]
    X = sub.drop(columns=[c for c in drop_cols if c in sub.columns])
    # Default expansion_label maps to the 10pct binary (legacy alias).
    if target == "expansion_label" and target not in sub.columns:
        target = "target_expansion_10pct"
    if target not in sub.columns:
        raise KeyError(f"target {target!r} not in matrix; available: {[c for c in sub.columns if c.startswith('target_') or c == 'expansion_label']}")
    y = sub[target]
    if target != "target_fwd_10d_reg":
        y = y.astype(int)
    meta = sub[["record_id", "ticker", "timestamp", "catalyst_family", "catalyst_subtype", "max_forward_return", "forward_10d_return", "max_drawdown"]].copy()
    return X, y, meta


TRAJECTORY_NAMES = {0: "flat", 1: "bull_steady", 2: "bull_volatile", 3: "v_bounce", 4: "crash_stayed"}


def evaluate(
    booster,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    name: str,
    is_regression: bool = False,
    is_multiclass: bool = False,
) -> dict:
    """Compute headline metrics on a held-out split."""
    # Multiclass branch — returns per-class metrics
    if is_multiclass:
        from sklearn.metrics import roc_auc_score, log_loss
        proba = booster.predict_proba(X) if hasattr(booster, "predict_proba") else booster.predict(X)
        classes = sorted(set(y))
        per_class = {}
        for c in classes:
            y_bin = (y == c).astype(int)
            if y_bin.nunique() < 2:
                per_class[TRAJECTORY_NAMES.get(int(c), str(c))] = {"auc": None, "n_positive": int(y_bin.sum())}
                continue
            auc = float(roc_auc_score(y_bin, proba[:, int(c)]))
            df = pd.DataFrame({"y": y_bin.values, "p": proba[:, int(c)],
                               "fwd10": meta["forward_10d_return"].fillna(0).values,
                               "max_fwd": meta["max_forward_return"].fillna(0).values,
                               "dd": meta["max_drawdown"].fillna(0).values})
            df = df.sort_values("p", ascending=False)
            n_top = max(1, int(len(df) * 0.10))
            top_winrate = float(df.head(n_top)["y"].mean())
            baseline = float(y_bin.mean())
            lift = top_winrate / baseline if baseline > 0 else float("nan")
            per_class[TRAJECTORY_NAMES.get(int(c), str(c))] = {
                "auc": auc,
                "baseline_rate": baseline,
                "top_decile_winrate": top_winrate,
                "top_decile_lift": lift,
                "top_decile_mean_fwd_10d": float(df.head(n_top)["fwd10"].mean()),
                "top_decile_mean_max_fwd": float(df.head(n_top)["max_fwd"].mean()),
                "top_decile_mean_drawdown": float(df.head(n_top)["dd"].mean()),
                "n_positive": int(y_bin.sum()),
            }
        return {
            "split": name,
            "n": int(len(y)),
            "objective": "multiclass",
            "n_classes": len(classes),
            "log_loss": float(log_loss(y, proba, labels=list(range(proba.shape[1])))),
            "per_class": per_class,
        }

    # XGBClassifier.predict() returns CLASS LABELS (0/1) — must use predict_proba
    # for ranking metrics. LightGBM Booster.predict() returns probabilities.
    if hasattr(booster, "predict_proba") and not is_regression:
        pred = booster.predict_proba(X)
        pred = pred[:, 1] if pred.ndim > 1 and pred.shape[1] > 1 else pred[:, 0] if pred.ndim > 1 else pred
    else:
        pred = booster.predict(X)
        if pred.ndim > 1:
            pred = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]

    if is_regression:
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        r2 = float(r2_score(y, pred))
        # Lift on top/bottom decile of predicted return
        df = pd.DataFrame({"y": y.values, "p": pred}).sort_values("p", ascending=False)
        n_top = max(1, int(len(df) * 0.10))
        top_decile_actual = float(df.head(n_top)["y"].mean())
        bottom_decile_actual = float(df.tail(n_top)["y"].mean())
        return {
            "split": name,
            "n": int(len(y)),
            "objective": "regression",
            "rmse": rmse,
            "r2": r2,
            "top_decile_actual_mean_return": top_decile_actual,
            "bottom_decile_actual_mean_return": bottom_decile_actual,
            "spread": top_decile_actual - bottom_decile_actual,
        }

    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
    auc = float(roc_auc_score(y, pred)) if len(set(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, pred)) if len(set(y)) > 1 else float("nan")
    ll = float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6)))

    df = pd.DataFrame({"y": y.values, "p": pred,
                       "fwd_max": meta["max_forward_return"].values,
                       "fwd_10d": meta["forward_10d_return"].fillna(0).values,
                       "dd": meta["max_drawdown"].fillna(0).values})
    df = df.sort_values("p", ascending=False)
    n_top = max(1, int(len(df) * 0.10))
    top_decile_winrate = float(df.head(n_top)["y"].mean())
    baseline = float(y.mean())
    lift = top_decile_winrate / baseline if baseline > 0 else float("nan")
    top_decile_max_fwd = float(df.head(n_top)["fwd_max"].mean())
    top_decile_fwd_10d = float(df.head(n_top)["fwd_10d"].mean())
    top_decile_dd = float(df.head(n_top)["dd"].mean())

    return {
        "split": name,
        "n": int(len(y)),
        "baseline_winrate": baseline,
        "auc": auc,
        "avg_precision": ap,
        "log_loss": ll,
        "top_decile_winrate": top_decile_winrate,
        "top_decile_lift": lift,
        "top_decile_mean_max_fwd_return": top_decile_max_fwd,
        "top_decile_mean_fwd_10d_return": top_decile_fwd_10d,
        "top_decile_mean_drawdown": top_decile_dd,
    }


def _train_lightgbm(X_tr, y_tr, X_val, y_val, *, params, num_boost_round, early_stopping):
    import lightgbm as lgb
    train_ds = lgb.Dataset(X_tr, label=y_tr)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=int(num_boost_round),
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(int(early_stopping)), lgb.log_evaluation(50)],
    )
    return booster, "lightgbm"


def _train_xgboost(X_tr, y_tr, X_val, y_val, *, params, num_boost_round, early_stopping, device, is_regression=False, num_class=None):
    import xgboost as xgb
    common = dict(
        tree_method="hist",
        device=device,
        n_estimators=int(num_boost_round),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params.get("bagging_fraction", 0.85)),
        colsample_bytree=float(params.get("feature_fraction", 0.85)),
        early_stopping_rounds=int(early_stopping),
        random_state=42,
        verbosity=1,
    )
    if is_regression:
        booster = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    elif num_class and num_class > 2:
        booster = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=int(num_class),
            **common,
        )
    else:
        booster = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    booster.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return booster, "xgboost"


def train(
    matrix_path: Path = DEFAULT_MATRIX,
    model_path: Path = DEFAULT_MODEL,
    metrics_path: Path = DEFAULT_METRICS,
    *,
    target: str = "target_expansion_10pct",
    engine: str = "lightgbm",
    device: str = "cpu",
    num_boost_round: int = 600,
    learning_rate: float = 0.03,
    max_depth: int = 6,
    num_leaves: int = 63,
    early_stopping: int = 40,
) -> dict:
    if not matrix_path.exists():
        raise SystemExit(
            f"Feature matrix not found at {matrix_path}. Build it first:\n"
            f"  .venv/bin/python -m meta_context.build_catalyst_training_matrix"
        )

    is_regression = target == "target_fwd_10d_reg"
    is_multiclass = target == "target_trajectory_code"
    print(f"loading feature matrix from {matrix_path}")
    df = pd.read_parquet(matrix_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    print(f"target column: {target} (is_regression={is_regression})")
    print("split sizes:", df["split"].value_counts().to_dict())
    target_alias = target if target in df.columns else "expansion_label"
    if not is_regression:
        print(f"label rate per split: {df.groupby('split')[target_alias].mean().to_dict()}")

    X_tr, y_tr, _ = split_xy(df, "train", target=target)
    X_val, y_val, meta_val = split_xy(df, "val", target=target)
    X_te, y_te, meta_te = split_xy(df, "test", target=target)

    print(f"feature count: {X_tr.shape[1]}")
    print(f"train rows: {len(X_tr):,}  val rows: {len(X_val):,}  test rows: {len(X_te):,}")
    print(f"engine: {engine}  device: {device}")

    if is_regression:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 4,
            "verbosity": -1,
            "seed": 42,
        }
    else:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 4,
            "verbosity": -1,
            "seed": 42,
        }

    if engine == "lightgbm":
        if device == "gpu":
            params["device"] = "gpu"
            print("(note: lightgbm GPU requires the OpenCL-built wheel; falls back to CPU if not present)")
        booster, kind = _train_lightgbm(
            X_tr, y_tr, X_val, y_val,
            params=params,
            num_boost_round=num_boost_round,
            early_stopping=early_stopping,
        )
    elif engine == "xgboost":
        booster, kind = _train_xgboost(
            X_tr, y_tr, X_val, y_val,
            params=params,
            num_boost_round=num_boost_round,
            early_stopping=early_stopping,
            device=device,
            is_regression=is_regression,
            num_class=(int(y_tr.nunique()) if is_multiclass else None),
        )
    else:
        raise SystemExit(f"unknown engine: {engine}. Choose 'lightgbm' or 'xgboost'.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "lightgbm":
        booster.save_model(str(model_path))
    else:
        # XGBoost classifier has its own format
        xgb_path = model_path.with_suffix(".xgb.json")
        booster.save_model(str(xgb_path))
        model_path = xgb_path
    print(f"\nsaved booster -> {model_path}")

    train_meta_df = split_xy(df, "train", target=target)[2]
    metrics = {
        "train": evaluate(booster, X_tr, y_tr, train_meta_df, name="train", is_regression=is_regression, is_multiclass=is_multiclass),
        "val": evaluate(booster, X_val, y_val, meta_val, name="val", is_regression=is_regression, is_multiclass=is_multiclass),
        "test": evaluate(booster, X_te, y_te, meta_te, name="test", is_regression=is_regression, is_multiclass=is_multiclass),
    }
    metrics["engine"] = kind
    metrics["device"] = device
    metrics["target"] = target
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"saved metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # Persist feature importance for sanity-check
    if kind == "lightgbm":
        gains = booster.feature_importance(importance_type="gain")
    else:
        gains = booster.feature_importances_
    importance = pd.DataFrame({"feature": X_tr.columns, "gain": gains}).sort_values("gain", ascending=False)
    importance_path = metrics_path.with_name("news_catalyst_feature_importance.csv")
    importance.to_csv(importance_path, index=False)
    print(f"saved feature importance -> {importance_path}")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the news catalyst specialist model.")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--engine", choices=["lightgbm", "xgboost"], default="lightgbm",
                        help="GBM engine. lightgbm = CPU default; xgboost supports CUDA via --device cuda.")
    parser.add_argument("--device", default="cpu",
                        help="cpu / gpu (lightgbm) / cuda (xgboost). xgboost + cuda is the simplest GPU path.")
    parser.add_argument("--target",
                        default="target_expansion_10pct",
                        choices=["target_expansion_10pct", "target_expansion_5pct", "target_crash_5pct",
                                 "target_fwd_10d_reg", "target_trajectory_code"],
                        help="Which target column to predict. trajectory_code is 5-class softprob.")
    parser.add_argument("--num-boost-round", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--early-stopping", type=int, default=40)
    args = parser.parse_args()

    train(
        matrix_path=Path(args.matrix),
        model_path=Path(args.model),
        metrics_path=Path(args.metrics),
        target=args.target,
        engine=args.engine,
        device=args.device,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        early_stopping=args.early_stopping,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
