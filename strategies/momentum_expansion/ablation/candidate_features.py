"""Candidate daily feature blocks for the 4H feature study (plan Stage 1).

Research-only: nothing here is wired into the live feature pipeline. A block
that survives the screen must be ported into ``feature_matrix_4h`` (same
formulas, same lag) before any model trains on it, or live and research drift.

TIME CONVENTION (same as feature_matrix_4h's DAILY HTF block)
-------------------------------------------------------------
Every column is computed on the daily bar series and then ``.shift(1)``-ed, so
the value stamped on session D uses bars through session D-1 only. The screen
maps it onto a 4H decision bar by that bar's New York session date. A 4H bar on
D therefore never sees D's own daily high/low/close, which matches production.

Unadjusted cache caveat: ``Data/shared/bars/1d`` is raw (see
``core.corporate_actions``). A split inside a lookback window corrupts every
return/level feature spanning it, so ``daily_candidate_features`` NaNs each
column for as long as a non-organic corporate-action flag sits inside that
column's own lookback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.corporate_actions import suspect_sessions
from strategies.momentum_expansion.features.feature_matrix_4h import _atr, _bars_since_true, _ema

# Longest bar lookback each column reads, used for the corporate-action mask.
_LOOKBACK: dict[str, int] = {}


def _reg(name: str, lookback: int) -> str:
    _LOOKBACK[name] = lookback
    return name


def _streak(mask: pd.Series, cap: int = 10) -> pd.Series:
    """Consecutive True count ending at each bar (0 when False), capped."""
    m = mask.fillna(False).astype(int)
    grp = (m == 0).cumsum()
    return m.groupby(grp).cumsum().clip(upper=cap).astype(float)


def daily_candidate_features(d: pd.DataFrame) -> pd.DataFrame:
    """Candidate blocks 1-4 for one ticker's daily bars (oldest first).

    ``d`` needs open/high/low/close/volume. Returns a frame on ``d``'s index,
    already lagged one session (see module docstring).
    """
    o, h, lo, c, v = (d[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    atr = _atr(d, 14).replace(0, np.nan)
    f: dict[str, pd.Series] = {}

    # ---- Block 1: daily MA structure (ATR units) ----------------------------
    emas = {n: _ema(c, n) for n in (20, 50, 100, 200)}
    for n, e in emas.items():
        f[_reg(f"d_ema{n}_dist_atr", n)] = ((c - e) / atr).clip(-50, 50)
    for n in (50, 100):
        f[_reg(f"d_ema{n}_slope10_atr", n + 10)] = ((emas[n] - emas[n].shift(10)) / atr).clip(-50, 50)
        f[_reg(f"d_days_above_ema{n}", 252)] = _bars_since_true(c < emas[n], cap=252)
    f[_reg("d_ema100_touches_20", 120)] = (
        ((lo <= emas[100]) & (c > emas[100])).astype(float).rolling(20).sum()
    )
    f[_reg("d_stack_full", 200)] = (
        (c > emas[20]).astype(float) + (emas[20] > emas[50]) + (emas[50] > emas[100]) + (emas[100] > emas[200])
    )

    # ---- Block 2: candle shape ----------------------------------------------
    rng = (h - lo).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - lo
    clv = ((c - lo) - (h - c)) / rng
    f[_reg("d_body_range", 1)] = body / rng
    f[_reg("d_upper_wick", 1)] = upper / rng
    f[_reg("d_lower_wick", 1)] = lower / rng
    f[_reg("d_clv", 1)] = clv
    f[_reg("d_clv_5", 5)] = clv.rolling(5).mean()
    f[_reg("d_range_atr", 15)] = (rng / atr).clip(0, 20)
    f[_reg("d_hammer", 1)] = ((lower / rng >= 0.5) & (body / rng <= 0.35) & (upper / rng <= 0.15)).astype(float)
    f[_reg("d_bull_engulf", 2)] = (
        (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))
    ).astype(float)
    f[_reg("d_inside_bar", 2)] = ((h <= h.shift(1)) & (lo >= lo.shift(1))).astype(float)
    f[_reg("d_nr7", 7)] = (rng <= rng.rolling(7).min()).astype(float)
    f[_reg("d_up_streak", 10)] = _streak(c > c.shift(1))
    f[_reg("d_down_streak", 10)] = _streak(c < c.shift(1))

    # ---- Block 3: leader / base context -------------------------------------
    for n in (63, 126, 252):
        f[_reg(f"d_ret_{n}", n)] = c / c.shift(n) - 1.0
    f[_reg("d_ret_252_21", 252)] = c.shift(21) / c.shift(252) - 1.0  # 12-1 momentum
    hi126 = h.rolling(126, min_periods=63).max()
    pull = ((hi126 - c) / atr).clip(0, 50)
    f[_reg("d_pullback_126h_atr", 126)] = pull
    f[_reg("d_base_len_126", 126)] = _bars_since_true(h >= hi126, cap=126)
    f[_reg("d_run_x_pullback", 126)] = f["d_ret_126"] / (1.0 + pull)

    # ---- Block 4: volume structure (ratios only; absolute volume is not
    # era-consistent across the IEX->SIP switch) ------------------------------
    up = c > c.shift(1)
    dn = c < c.shift(1)
    upv = v.where(up, 0.0).rolling(20).sum()
    dnv = v.where(dn, 0.0).rolling(20).sum()
    f[_reg("d_updown_vol_20", 21)] = np.log((upv + 1.0) / (dnv + 1.0)).clip(-5, 5)
    f[_reg("d_vol_dryup_10_50", 50)] = v.rolling(10).mean() / v.rolling(50).mean().replace(0, np.nan)
    f[_reg("d_cmf_20", 20)] = (clv.fillna(0) * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f[_reg("d_obv_slope_20", 21)] = (obv - obv.shift(20)) / v.rolling(20).sum().replace(0, np.nan)

    out = pd.DataFrame(f, index=d.index)
    out = _mask_corporate_actions(out, d)
    return out.shift(1)


def _mask_corporate_actions(feats: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """NaN each column while a non-organic flag lies inside its own lookback.

    A flag at row p (its open gaps vs p-1's close) corrupts any value at row t
    whose window reaches back to p-1, i.e. t in [p, p + lookback].
    """
    flags = suspect_sessions(d.reset_index(drop=True))
    if flags.empty:
        return feats
    flags = flags[~flags["organic"].astype(bool)]
    if flags.empty:
        return feats
    n = len(feats)
    out = feats.copy()
    positions = sorted(int(p) for p in flags["idx"])
    for col in out.columns:
        lb = _LOOKBACK[col]
        bad = np.zeros(n, dtype=bool)
        for p in positions:
            bad[p:min(n, p + lb + 1)] = True
        out.loc[bad, col] = np.nan
    return out


CANDIDATE_BLOCKS: dict[str, list[str]] = {
    "daily_ma": ["d_ema20_dist_atr", "d_ema50_dist_atr", "d_ema100_dist_atr", "d_ema200_dist_atr",
                 "d_ema50_slope10_atr", "d_ema100_slope10_atr", "d_days_above_ema50",
                 "d_days_above_ema100", "d_ema100_touches_20", "d_stack_full"],
    "candle": ["d_body_range", "d_upper_wick", "d_lower_wick", "d_clv", "d_clv_5", "d_range_atr",
               "d_hammer", "d_bull_engulf", "d_inside_bar", "d_nr7", "d_up_streak", "d_down_streak"],
    "leader": ["d_ret_63", "d_ret_126", "d_ret_252", "d_ret_252_21", "d_pullback_126h_atr",
               "d_base_len_126", "d_run_x_pullback"],
    "volume": ["d_updown_vol_20", "d_vol_dryup_10_50", "d_cmf_20", "d_obv_slope_20"],
}
