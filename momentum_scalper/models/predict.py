"""Breakout quality inference."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from momentum_scalper.configs.settings import MODEL_ARTIFACTS_DIR
from momentum_scalper.rankers.rule_ranker import score_setups


def predict_breakout_quality(features: pd.DataFrame, artifacts_dir: Path = MODEL_ARTIFACTS_DIR) -> pd.DataFrame:
    out = features.copy()
    model_path = artifacts_dir / "xgb_breakout_quality.json"
    manifest_path = artifacts_dir / "feature_manifest.json"
    if not model_path.exists() or not manifest_path.exists():
        ranked = score_setups(out)
        max_score = ranked["score"].max() if not ranked.empty else 1.0
        ranked["breakout_quality_probability"] = ranked["score"] / max(max_score, 1.0)
        return ranked

    import xgboost as xgb

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cols = manifest["features"]
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    out["breakout_quality_probability"] = model.predict_proba(out[cols].fillna(0.0))[:, 1]
    return out.sort_values(["timestamp", "breakout_quality_probability"], ascending=[True, False]).reset_index(drop=True)
