"""CPU XGBoost training for breakout quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from momentum_scalper.configs.settings import MODEL_ARTIFACTS_DIR, TRAINING_MATRIX_PATH, ensure_data_dirs


TARGET = "hit_2R_before_minus_1R_within_15m"
DROP_COLUMNS = {"timestamp", "ticker", TARGET, "catalyst_type"}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def train_xgb(matrix_path: Path = TRAINING_MATRIX_PATH, artifacts_dir: Path = MODEL_ARTIFACTS_DIR) -> dict:
    import xgboost as xgb

    ensure_data_dirs()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(matrix_path)
    if df.empty or TARGET not in df.columns:
        raise RuntimeError(f"Training matrix must contain rows and target column {TARGET}.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    cols = _feature_columns(df)
    split = max(int(len(df) * 0.8), 1)
    train, valid = df.iloc[:split], df.iloc[split:]
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
    )
    model.fit(train[cols].fillna(0.0), train[TARGET].astype(int))
    metrics = {"rows": int(len(df)), "features": cols}
    if not valid.empty and valid[TARGET].nunique() > 1:
        proba = model.predict_proba(valid[cols].fillna(0.0))[:, 1]
        metrics.update(
            {
                "AUC": float(roc_auc_score(valid[TARGET].astype(int), proba)),
                "precision_at_topK": float(valid.iloc[np.argsort(-proba)[: max(1, min(100, len(valid)))]][TARGET].mean()),
                "EV": float((valid[TARGET].astype(int) * proba - (1 - valid[TARGET].astype(int)) * (1 - proba)).mean()),
                "PnL": float(valid.get("MFE", pd.Series(0, index=valid.index)).where(valid[TARGET].astype(bool), valid.get("MAE", 0)).mean()),
                "average_precision": float(average_precision_score(valid[TARGET].astype(int), proba)),
            }
        )
    model.save_model(artifacts_dir / "xgb_breakout_quality.json")
    (artifacts_dir / "feature_manifest.json").write_text(json.dumps({"features": cols, "target": TARGET}, indent=2), encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost breakout model")
    parser.add_argument("--matrix", type=Path, default=TRAINING_MATRIX_PATH)
    args = parser.parse_args()
    metrics = train_xgb(args.matrix)
    print(json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
