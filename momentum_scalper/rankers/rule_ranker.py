"""Deterministic baseline momentum setup ranker."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_scalper.configs.settings import FEATURES_PATH


def _numeric_col(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _clip_score(series: pd.Series, scale: float, cap: float = 1.0) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") / scale).clip(lower=0.0, upper=cap).fillna(0.0)


def score_setups(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features.assign(score=pd.Series(dtype=float))
    out = features.copy()
    out["gap_score"] = _clip_score(_numeric_col(out, "gap_pct"), 30.0)
    out["rvol_score"] = _clip_score(_numeric_col(out, "rvol"), 20.0)
    out["float_score"] = (1.0 - _clip_score(_numeric_col(out, "float", np.nan), 100_000_000.0)).fillna(0.5)
    catalysts = out["catalyst_type"] if "catalyst_type" in out.columns else pd.Series("none", index=out.index)
    out["news_score"] = np.where(catalysts.eq("news"), 1.0, 0.0)
    out["breakout_volume_score"] = _clip_score(_numeric_col(out, "volume_spike_ratio"), 5.0)
    out["score"] = out[["gap_score", "rvol_score", "float_score", "news_score", "breakout_volume_score"]].sum(axis=1)
    out["rank"] = out.groupby("timestamp")["score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["timestamp", "rank"]).reset_index(drop=True)


def top_ranked_setups(features: pd.DataFrame, top_n: int = 10, min_score: float = 0.0) -> pd.DataFrame:
    ranked = score_setups(features)
    if ranked.empty:
        return ranked
    return ranked[(ranked["rank"] <= top_n) & (ranked["score"] >= min_score)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank scalper feature rows")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    df = pd.read_parquet(args.features)
    print(top_ranked_setups(df, args.top_n).tail(50).to_string(index=False))


if __name__ == "__main__":
    main()
