from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    forward_bars: int = 60
    overlap_cooldown_bars: int = 15


def build_event_labels(
    bars: pd.DataFrame,
    events: Iterable[dict],
    *,
    config: LabelConfig = LabelConfig(),
) -> pd.DataFrame:
    """Label confirmed events from future bars; never used by signal generation.

    If target and invalidation are both touched in one 1-minute bar, the label is
    conservative: invalidation wins because intrabar ordering is unknowable.
    """
    frame = _normalize_bars(bars)
    rows: list[dict] = []
    last_index: dict[tuple[str, str], int] = {}
    grouped = {symbol: group.reset_index(drop=True) for symbol, group in frame.groupby("symbol", sort=False)}
    for event in sorted(events, key=lambda item: pd.Timestamp(item["timestamp"])):
        symbol = str(event.get("ticker") or event.get("symbol")).upper()
        direction = str(event.get("direction", "long")).lower()
        group = grouped.get(symbol)
        if group is None or group.empty:
            continue
        ts = pd.Timestamp(event["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        candidates = group.index[group["timestamp"] >= ts]
        if len(candidates) == 0:
            continue
        start = int(candidates[0])
        key = (symbol, str(event.get("setup_type", "unknown")))
        if start - last_index.get(key, -10_000) < config.overlap_cooldown_bars:
            continue
        last_index[key] = start
        entry = float(event.get("entry_price") or group.loc[start, "open"])
        invalidation = float(event["invalidation"])
        targets = [float(x) for x in event.get("targets", []) if x is not None]
        target1 = targets[0] if targets else None
        target2 = targets[1] if len(targets) > 1 else None
        window = group.iloc[start : start + config.forward_bars]
        sign = 1.0 if direction == "long" else -1.0
        mfe = 0.0
        mae = 0.0
        hit1 = hit2 = failed = False
        time_to_target: int | None = None
        breakout_hold = 0
        pivot = event.get("pivot")
        for offset, row in enumerate(window.itertuples(index=False)):
            favorable = (row.high - entry) if sign > 0 else (entry - row.low)
            adverse = (entry - row.low) if sign > 0 else (row.high - entry)
            mfe = max(mfe, favorable)
            mae = max(mae, adverse)
            invalid_hit = row.low <= invalidation if sign > 0 else row.high >= invalidation
            t1_hit = target1 is not None and (row.high >= target1 if sign > 0 else row.low <= target1)
            if invalid_hit:
                failed = True
                break
            if t1_hit and not hit1:
                hit1 = True
                time_to_target = offset
            if target2 is not None and (row.high >= target2 if sign > 0 else row.low <= target2):
                hit2 = True
            if pivot is not None:
                held = row.close >= float(pivot) if sign > 0 else row.close <= float(pivot)
                breakout_hold = breakout_hold + 1 if held else 0
        risk = abs(entry - invalidation)
        realized_r = (mfe / risk if hit1 and risk > 0 else -mae / risk if failed and risk > 0 else sign * (float(window.iloc[-1]["close"]) - entry) / risk if risk > 0 and not window.empty else 0.0)
        rows.append({
            "ticker": symbol, "timestamp": ts, "setup_type": event.get("setup_type"), "direction": direction,
            "target_before_invalidation": bool(hit1 and not failed), "target_1_hit": hit1,
            "target_2_hit": hit2, "vwap_reached_after_reversal": bool(event.get("setup_type") == "v_shaped_capitulation_reversal" and hit1),
            "breakout_held_n_bars": breakout_hold, "breakout_failed": bool(event.get("setup_type") == "breakout_continuation" and failed),
            "reversal_failed": bool(event.get("setup_type") == "v_shaped_capitulation_reversal" and failed),
            "max_favorable_excursion": mfe, "max_adverse_excursion": mae,
            "realized_r": realized_r, "time_to_target": time_to_target,
        })
    return pd.DataFrame(rows)


def _normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"replay bars missing columns: {sorted(missing)}")
    out = frame.copy()
    if "symbol" not in out:
        if "ticker" not in out:
            raise KeyError("replay bars require symbol or ticker")
        out["symbol"] = out["ticker"]
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["timestamp", "symbol"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    return out.reset_index(drop=True)
