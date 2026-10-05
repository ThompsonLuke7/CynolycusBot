"""Label arms for the 4H feature study (plan Stage 0).

Both deployed 4H models learn an absolute forward-MFE threshold (momentum:
``fwd_max_return >= 20%``; HTF: ``fwd_best_high_return >= 15%``). Forward MFE
grows with volatility, so the models largely learned "pick volatile names".
These arms re-express the target so volatility alone cannot score well:

  L0   the current MFE label, as is (baseline)
  L1   MFE percentile-ranked WITHIN same-bar volatility quintiles
  L2   executable forward return minus the mean of same-bar volatility-decile
       peers (a vol-matched excess return)
  R    executable forward return, raw (the tradable outcome; reference only)

Volatility buckets use a decision-time measure (lagged daily ATR%), never a
forward one. Buckets are formed per bar, so they are cross-sectional and do not
look across time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_bucket(panel: pd.DataFrame, *, vol_col: str, n: int, group_col: str = "timestamp") -> pd.Series:
    """Per-bar volatility bucket 0..n-1 from the within-bar percentile rank."""
    pct = panel.groupby(group_col)[vol_col].rank(pct=True, method="first")
    return np.minimum((pct * n).apply(np.ceil) - 1, n - 1).astype("Int64")


def build_label_arms(
    panel: pd.DataFrame,
    *,
    mfe_col: str,
    ret_col: str,
    vol_col: str = "daily_atr_pct",
    group_col: str = "timestamp",
) -> pd.DataFrame:
    """Return L0/L1/L2/R columns aligned to ``panel``'s index."""
    out = pd.DataFrame(index=panel.index)
    out["L0"] = panel[mfe_col]
    q5 = vol_bucket(panel, vol_col=vol_col, n=5, group_col=group_col)
    q10 = vol_bucket(panel, vol_col=vol_col, n=10, group_col=group_col)
    out["L1"] = panel[mfe_col].groupby([panel[group_col], q5]).rank(pct=True)
    peer_mean = panel[ret_col].groupby([panel[group_col], q10]).transform("mean")
    out["L2"] = panel[ret_col] - peer_mean
    out["R"] = panel[ret_col]
    # Rows with no decision-time vol cannot be bucketed; don't let them into L1/L2.
    no_vol = panel[vol_col].isna()
    out.loc[no_vol, ["L1", "L2"]] = np.nan
    return out
