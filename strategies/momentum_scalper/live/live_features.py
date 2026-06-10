"""Realtime scanner -> features -> prediction pipeline."""
from __future__ import annotations

import pandas as pd

from strategies.momentum_scalper.features.build_features import build_features_for_snapshot
from strategies.momentum_scalper.models.predict import predict_breakout_quality


def build_live_rankings(snapshot: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    features = build_features_for_snapshot(snapshot, bars)
    if features.empty:
        return features
    return predict_breakout_quality(features)
