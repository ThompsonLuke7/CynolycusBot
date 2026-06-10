"""
Shared coarse filter for actionable momentum-continuation candidates.

The model can only learn the live task if training rows look like live ranking
rows. This helper keeps that gate identical in both places.
"""
from __future__ import annotations

import pandas as pd

from strategies.momentum_expansion.config.momentum_config import MOMENTUM_CANDIDATE_FILTER_CONFIG


def momentum_candidate_mask(
    df: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.Series:
    cfg = {**MOMENTUM_CANDIDATE_FILTER_CONFIG, **(cfg or {})}
    mask = pd.Series(True, index=df.index)

    if cfg.get("exclude_low_price", True) and "low_price_flag" in df.columns:
        mask &= pd.to_numeric(df["low_price_flag"], errors="coerce").fillna(1.0) <= 0.0

    if "dollar_vol_pctile_252" in df.columns:
        mask &= (
            pd.to_numeric(df["dollar_vol_pctile_252"], errors="coerce")
            >= float(cfg["min_dollar_vol_pctile_252"])
        )

    near_high_parts = []
    if "dist_to_52w_high_atr" in df.columns:
        near_high_parts.append(
            pd.to_numeric(df["dist_to_52w_high_atr"], errors="coerce")
            <= float(cfg["max_dist_to_52w_high_atr"])
        )
    if "xsec_near_high_rank" in df.columns:
        near_high_parts.append(
            pd.to_numeric(df["xsec_near_high_rank"], errors="coerce")
            >= float(cfg["min_xsec_near_high_rank"])
        )
    if near_high_parts:
        near_high = near_high_parts[0]
        for part in near_high_parts[1:]:
            near_high |= part
        mask &= near_high.fillna(False)

    rs_parts = []
    if "rs_spy_20" in df.columns:
        rs_parts.append(
            pd.to_numeric(df["rs_spy_20"], errors="coerce")
            >= float(cfg["min_rs_spy_20"])
        )
    if "xsec_ret_20_rank" in df.columns:
        rs_parts.append(
            pd.to_numeric(df["xsec_ret_20_rank"], errors="coerce")
            >= float(cfg["min_xsec_ret_20_rank"])
        )
    if rs_parts:
        relative_strength = rs_parts[0]
        for part in rs_parts[1:]:
            relative_strength |= part
        mask &= relative_strength.fillna(False)

    if "range_pos_20" in df.columns:
        mask &= (
            pd.to_numeric(df["range_pos_20"], errors="coerce")
            >= float(cfg["min_range_pos_20"])
        )

    return mask.fillna(False).astype(bool)


def filter_momentum_candidates(
    df: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    cfg = {**MOMENTUM_CANDIDATE_FILTER_CONFIG, **(cfg or {})}
    if not cfg.get("enabled", True) or df.empty:
        return df
    return df.loc[momentum_candidate_mask(df, cfg=cfg)].copy()
