"""Retired legacy ML feature path.

The v1 strategy uses causal scanner and pattern state instead of this path.
"""
from __future__ import annotations

import pandas as pd

from strategies.momentum_scalper.features.build_features import build_features_for_snapshot
from strategies.momentum_scalper.models.predict import predict_breakout_quality


def build_live_rankings(snapshot: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    raise RuntimeError(
        "Legacy live ML rankings are retired. Use MomentumShadowRunner with the v1 causal scanner."
    )
