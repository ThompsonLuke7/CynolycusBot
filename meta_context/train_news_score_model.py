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


def split_xy(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta) for a given split label."""
    sub = df[df["split"] == split].copy()
    drop_cols = [
        "record_id",
        "ticker",
        "timestamp",
        "catalyst_family",
        "catalyst_subtype",
        "expansion_label",
        "max_forward_return",
        "max_drawdown",
        "forward_5d_return",
        "forward_10d_return",
        "split",
    ]
    X = sub.drop(columns=[c for c in drop_cols if c in sub.columns])
    y = sub["expansion_label"].astype(int)
    meta = sub[["record_id", "ticker", "timestamp", "catalyst_family", "catalyst_subtype", "max_forward_return"]].copy()
    return X, y, meta


def evaluate(
    booster,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    name: str,
) -> dict:
    """Compute headline metrics on a held-out split."""
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
    except ImportError as exc:
        raise SystemExit("sklearn is required for evaluation") from exc

    pred = booster.predict(X)
    if pred.ndim > 1:
        pred = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]
    auc = float(roc_auc_score(y, pred)) if len(set(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, pred)) if len(set(y)) > 1 else float("nan")
    ll = float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6)))

    # Top-decile lift — what fraction of winners are in the top-10% predicted
    df = pd.DataFrame({"y": y.values, "p": pred, "fwd_max": meta["max_forward_return"].values})
    df = df.sort_values("p", ascending=False)
    n_top = max(1, int(len(df) * 0.10))
    top_decile_winrate = float(df.head(n_top)["y"].mean())
    baseline = float(y.mean())
    lift = top_decile_winrate / baseline if baseline > 0 else float("nan")
    top_decile_fwd_ret = float(df.head(n_top)["fwd_max"].mean())

    return {
        "split": name,
        "n": int(len(y)),
        "baseline_winrate": baseline,
        "auc": auc,
        "avg_precision": ap,
        "log_loss": ll,
        "top_decile_winrate": top_decile_winrate,
        "top_decile_lift": lift,
        "top_decile_mean_max_fwd_return": top_decile_fwd_ret,
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


def _train_xgboost(X_tr, y_tr, X_val, y_val, *, params, num_boost_round, early_stopping, device):
    import xgboost as xgb
    booster = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,  # 'cuda' or 'cpu'
        n_estimators=int(num_boost_round),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params.get("bagging_fraction", 0.85)),
        colsample_bytree=float(params.get("feature_fraction", 0.85)),
        early_stopping_rounds=int(early_stopping),
        random_state=42,
        verbosity=1,
    )
    booster.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return booster, "xgboost"


def train(
    matrix_path: Path = DEFAULT_MATRIX,
    model_path: Path = DEFAULT_MODEL,
    metrics_path: Path = DEFAULT_METRICS,
    *,
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

    print(f"loading feature matrix from {matrix_path}")
    df = pd.read_parquet(matrix_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    print("split sizes:", df["split"].value_counts().to_dict())
    print("split label rates:", df.groupby("split")["expansion_label"].mean().to_dict())

    X_tr, y_tr, _ = split_xy(df, "train")
    X_val, y_val, meta_val = split_xy(df, "val")
    X_te, y_te, meta_te = split_xy(df, "test")

    print(f"feature count: {X_tr.shape[1]}")
    print(f"train rows: {len(X_tr):,}  val rows: {len(X_val):,}  test rows: {len(X_te):,}")
    print(f"engine: {engine}  device: {device}")

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

    metrics = {
        "train": evaluate(booster, X_tr, y_tr, df[df["split"] == "train"], name="train"),
        "val": evaluate(booster, X_val, y_val, meta_val, name="val"),
        "test": evaluate(booster, X_te, y_te, meta_te, name="test"),
    }
    metrics["engine"] = kind
    metrics["device"] = device
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
