"""
1H entry-trigger rules for momentum_expansion.

Each rule is a deterministic predicate over the most recent 1H bars of a
given ticker. They fire only when:
  - the name is in the active top-N list (passed in by the caller)
  - the bar is RTH (config-toggleable)

Rules:
  break_body_prev_high  — 1H bar trades through prior 1H high and closes green.
  pullback_continuation  — pullback of [pullback_min_atr, pullback_max_atr]
                           within an established uptrend, then close back
                           above ema_fast.
  flag_breakout          — close > rolling-N high after a tight consolidation.
  volume_confirmation    — bar volume > volume_confirm_mult × 20-bar avg AND
                           close > previous bar close.
  ema_reclaim            — recent dip below ema_slow followed by reclaim
                           (close > ema_slow) within ema_reclaim_lookback bars.

All rules return a TriggerResult; an empty list means no trigger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from momentum_expansion.config.momentum_config import ENTRY_RULES_CONFIG


@dataclass
class TriggerResult:
    rule:        str
    bar_ts:      pd.Timestamp
    close:       float
    suggested_stop_atr: float    # initial stop distance in ATR units
    note:        str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr_1h(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, lo, c_p = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - lo), (h - c_p).abs(), (lo - c_p).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _is_rth(ts: pd.Timestamp) -> bool:
    ny = ts.tz_convert("America/New_York") if ts.tzinfo else ts.tz_localize("UTC").tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60


# ---------------------------------------------------------------------------
# Rules (each returns 0 or 1 TriggerResult)
# ---------------------------------------------------------------------------

def _pullback_continuation(df: pd.DataFrame, cfg: dict) -> TriggerResult | None:
    if len(df) < 50:
        return None
    c = df["close"]
    ema_fast = _ema(c, cfg["ema_fast"])
    ema_slow = _ema(c, cfg["ema_slow"])
    atr = _atr_1h(df)

    # Trend filter: ema_fast > ema_slow over last 5 bars
    if not (ema_fast.iloc[-5:] > ema_slow.iloc[-5:]).all():
        return None

    # Rolling 10-bar high before this bar
    look_high = df["high"].iloc[-12:-2].max()
    pullback = look_high - df["low"].iloc[-3:].min()
    if not np.isfinite(atr.iloc[-1]) or atr.iloc[-1] <= 0:
        return None
    pullback_atr = pullback / atr.iloc[-1]
    if not (cfg["pullback_min_atr"] <= pullback_atr <= cfg["pullback_max_atr"]):
        return None

    # Confirmation: latest bar closes back above ema_fast
    if c.iloc[-1] <= ema_fast.iloc[-1]:
        return None

    return TriggerResult(
        rule="pullback_continuation",
        bar_ts=c.index[-1],
        close=float(c.iloc[-1]),
        suggested_stop_atr=1.0,
        note=f"pullback={pullback_atr:.2f}ATR",
    )


def _break_body_prev_high(df: pd.DataFrame, cfg: dict) -> TriggerResult | None:
    if len(df) < 25:
        return None
    row = df.iloc[-1]
    prev_high = float(df["high"].iloc[-2])
    if not np.isfinite(prev_high):
        return None
    if row["high"] < prev_high:
        return None
    if row["close"] <= row["open"]:
        return None
    return TriggerResult(
        rule="break_body_prev_high",
        bar_ts=df.index[-1],
        close=float(row["close"]),
        suggested_stop_atr=1.1,
        note=f"broke prev 1H high={prev_high:.2f} with green body",
    )


def _flag_breakout(df: pd.DataFrame, cfg: dict) -> TriggerResult | None:
    if len(df) < 30:
        return None
    c = df["close"]
    h = df["high"]
    atr = _atr_1h(df)
    n_consol = int(cfg["flag_consolidation_bars"])

    consol_window = df.iloc[-(n_consol + 1):-1]
    if consol_window.empty:
        return None
    flag_high = consol_window["high"].max()
    flag_low = consol_window["low"].min()
    flag_range = flag_high - flag_low

    # Tight flag: range < 1.5 ATR
    if not np.isfinite(atr.iloc[-1]) or atr.iloc[-1] <= 0:
        return None
    if flag_range > 1.5 * atr.iloc[-1]:
        return None

    # Breakout: close > flag_high + breakout_atr * ATR
    if c.iloc[-1] <= flag_high + cfg["flag_breakout_atr"] * atr.iloc[-1]:
        return None

    return TriggerResult(
        rule="flag_breakout",
        bar_ts=c.index[-1],
        close=float(c.iloc[-1]),
        suggested_stop_atr=1.2,
        note=f"flag_h={flag_high:.2f} range={flag_range/atr.iloc[-1]:.2f}ATR",
    )


def _volume_confirmation(df: pd.DataFrame, cfg: dict) -> TriggerResult | None:
    if len(df) < 25:
        return None
    c = df["close"]
    v = df["volume"]
    avg = v.rolling(20).mean().iloc[-1]
    if not np.isfinite(avg) or avg <= 0:
        return None
    if v.iloc[-1] < cfg["volume_confirm_mult"] * avg:
        return None
    if c.iloc[-1] <= c.iloc[-2]:
        return None
    return TriggerResult(
        rule="volume_confirmation",
        bar_ts=c.index[-1],
        close=float(c.iloc[-1]),
        suggested_stop_atr=1.0,
        note=f"vol={v.iloc[-1] / avg:.2f}× avg",
    )


def _ema_reclaim(df: pd.DataFrame, cfg: dict) -> TriggerResult | None:
    if len(df) < 30:
        return None
    c = df["close"]
    ema_slow = _ema(c, cfg["ema_slow"])
    look = int(cfg["ema_reclaim_lookback"])

    # In the last `look` bars, at least one close was below ema_slow,
    # but the latest bar is above.
    recent = c.iloc[-(look + 1):-1]
    recent_slow = ema_slow.iloc[-(look + 1):-1]
    if not (recent < recent_slow).any():
        return None
    if c.iloc[-1] <= ema_slow.iloc[-1]:
        return None
    return TriggerResult(
        rule="ema_reclaim",
        bar_ts=c.index[-1],
        close=float(c.iloc[-1]),
        suggested_stop_atr=1.3,
        note="reclaim above ema_slow",
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

ALL_RULES = (
    _break_body_prev_high,
    _pullback_continuation,
    _flag_breakout,
    _volume_confirmation,
    _ema_reclaim,
)


def evaluate_entry(
    *,
    df_1h: pd.DataFrame,
    cfg: dict | None = None,
) -> list[TriggerResult]:
    """Run every rule against the latest 1H bar; return all that fire."""
    if df_1h is None or df_1h.empty:
        return []
    cfg = {**ENTRY_RULES_CONFIG, **(cfg or {})}
    df = df_1h.copy()
    df.columns = [c.lower() for c in df.columns]
    if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        return []
    bar_ts = df.index[-1]
    if cfg.get("rth_only", True) and not _is_rth(bar_ts):
        return []
    results: list[TriggerResult] = []
    enabled = cfg.get("enabled_rules")
    enabled_set = {str(x) for x in enabled} if enabled else None
    for rule in ALL_RULES:
        rule_name = rule.__name__.lstrip("_")
        if enabled_set is not None and rule_name not in enabled_set:
            continue
        res = rule(df, cfg)
        if res is not None:
            results.append(res)
    return results
