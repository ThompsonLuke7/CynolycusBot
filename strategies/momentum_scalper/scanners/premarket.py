"""Causal premarket runner scanner for the standalone momentum scalper."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd

from strategies.momentum_scalper.configs.v1 import ScannerConfigV1
from strategies.momentum_scalper.utils.io import add_session_columns, normalize_timestamp_column


SNAPSHOT_COLUMNS = (
    "timestamp", "ticker", "eligible", "rejection_reasons", "scanner_rank", "score",
    "last_price", "prior_close", "gap_pct", "premarket_volume", "rvol",
    "rvol_history_sessions", "float", "float_available_at", "catalyst_available_at",
    "catalyst_age_minutes", "bid", "ask", "spread_pct", "quote_age_seconds",
    "daily_resistance", "daily_resistance_distance_pct", "halted",
)


def _utc(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("decision timestamp must be timezone-aware")
    return ts.tz_convert("UTC")


def _available_rows(frame: pd.DataFrame | None, decision_at: pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "ticker" not in out.columns:
        return pd.DataFrame()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    availability = "available_at" if "available_at" in out.columns else "timestamp"
    if availability not in out.columns:
        return pd.DataFrame()
    out[availability] = pd.to_datetime(out[availability], utc=True, errors="coerce")
    out = out.dropna(subset=[availability])
    out = out[out[availability] <= decision_at]
    out["_available_at"] = out[availability]
    return out


def _latest_by_ticker(frame: pd.DataFrame | None, decision_at: pd.Timestamp) -> pd.DataFrame:
    out = _available_rows(frame, decision_at)
    if out.empty:
        return out
    return out.sort_values(["ticker", "_available_at"]).drop_duplicates("ticker", keep="last")


def _quote_available_rows(quotes: pd.DataFrame | None) -> pd.DataFrame:
    """Make receipt time part of quote availability; event time alone is unsafe."""

    if quotes is None or quotes.empty:
        return pd.DataFrame()
    out = quotes.copy()
    if "timestamp" not in out.columns:
        return pd.DataFrame()
    event_at = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    received_at = pd.to_datetime(out.get("received_at", event_at), utc=True, errors="coerce")
    explicit = pd.to_datetime(out.get("available_at", received_at), utc=True, errors="coerce")
    out["available_at"] = pd.concat([event_at, received_at, explicit], axis=1).max(axis=1)
    return out


def _prior_close_and_resistance(bars: pd.DataFrame, day: str, ticker: str) -> tuple[float, float]:
    prior = bars[(bars["ticker"] == ticker) & (bars["date"] < day)]
    if prior.empty:
        return np.nan, np.nan
    rth = prior[prior["is_rth"]]
    close_source = rth if not rth.empty else prior
    prior_close = float(close_source.sort_values("timestamp").iloc[-1]["close"])
    daily_high = prior.groupby("date", sort=True)["high"].max().tail(20)
    resistance = float(daily_high.max()) if not daily_high.empty else np.nan
    return prior_close, resistance


def _rvol_history(
    bars: pd.DataFrame,
    *,
    day: str,
    ticker: str,
    cutoff_minute: int,
    lookback: int,
) -> pd.Series:
    prior = bars[
        (bars["ticker"] == ticker)
        & (bars["date"] < day)
        & bars["is_premarket"]
        & (bars["minute_of_day"] <= cutoff_minute)
    ]
    if prior.empty:
        return pd.Series(dtype=float)
    return prior.groupby("date", sort=True)["volume"].sum().tail(lookback)


def _active_halts(halts: pd.DataFrame | None, decision_at: pd.Timestamp) -> set[str]:
    if halts is None or halts.empty or "ticker" not in halts.columns or "halt_timestamp" not in halts.columns:
        return set()
    out = halts.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["halt_timestamp"] = pd.to_datetime(out["halt_timestamp"], utc=True, errors="coerce")
    out["resume_timestamp"] = pd.to_datetime(out.get("resume_timestamp"), utc=True, errors="coerce")
    availability = out.get("available_at", out["halt_timestamp"])
    out["_available_at"] = pd.to_datetime(availability, utc=True, errors="coerce")
    out = out[(out["halt_timestamp"] <= decision_at) & (out["_available_at"] <= decision_at)]
    active = out[out["resume_timestamp"].isna() | (out["resume_timestamp"] > decision_at)]
    return set(active["ticker"])


def _material_catalyst(news: pd.DataFrame | None, decision_at: pd.Timestamp) -> pd.DataFrame:
    out = _available_rows(news, decision_at)
    if out.empty:
        return out
    if "material" not in out.columns:
        return pd.DataFrame()
    return out[out["material"].fillna(False).astype(bool)]


def reconstruct_premarket_snapshot(
    bars: pd.DataFrame,
    *,
    decision_at: datetime | pd.Timestamp,
    metadata: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
    quotes: pd.DataFrame | None = None,
    halts: pd.DataFrame | None = None,
    config: ScannerConfigV1 = ScannerConfigV1(),
) -> pd.DataFrame:
    """Return every current premarket candidate with causal pass/fail evidence.

    ``bars`` must include prior sessions as well as the current session.  The
    function never reads a row whose timestamp or availability is after
    ``decision_at``.
    """

    decision = _utc(decision_at)
    ordered = add_session_columns(normalize_timestamp_column(bars))
    if ordered.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    ordered = ordered[ordered["timestamp"] <= decision].copy()
    if ordered.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    local = decision.tz_convert("America/New_York")
    day = local.strftime("%Y-%m-%d")
    cutoff_minute = local.hour * 60 + local.minute
    current = ordered[(ordered["date"] == day) & (ordered["is_premarket"] | ordered["is_rth"])]
    if current.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    latest_meta = _latest_by_ticker(metadata, decision)
    latest_quotes = _latest_by_ticker(_quote_available_rows(quotes), decision)
    catalysts = _material_catalyst(news, decision)
    latest_catalyst = _latest_by_ticker(catalysts, decision)
    halted = _active_halts(halts, decision)
    meta_map = latest_meta.set_index("ticker") if not latest_meta.empty else pd.DataFrame()
    quote_map = latest_quotes.set_index("ticker") if not latest_quotes.empty else pd.DataFrame()
    catalyst_map = latest_catalyst.set_index("ticker") if not latest_catalyst.empty else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for ticker, history in current.groupby("ticker", sort=True):
        history = history.sort_values("timestamp")
        premarket_history = history[history["is_premarket"]]
        if premarket_history.empty:
            continue
        last = history.iloc[-1]
        last_price = float(last["close"])
        prior_close, resistance = _prior_close_and_resistance(ordered, day, ticker)
        gap_pct = ((last_price / prior_close - 1.0) * 100.0) if prior_close > 0 else np.nan
        premarket_volume = float(pd.to_numeric(premarket_history["volume"], errors="coerce").fillna(0.0).sum())
        rvol_history = _rvol_history(
            ordered, day=day, ticker=ticker, cutoff_minute=cutoff_minute,
            lookback=config.rvol_lookback_sessions,
        )
        rvol_base = float(rvol_history.median()) if not rvol_history.empty else np.nan
        rvol = premarket_volume / rvol_base if rvol_base > 0 else np.nan
        resistance_distance = (
            float("inf") if resistance > 0 and last_price >= resistance
            else ((resistance / last_price - 1.0) * 100.0) if resistance > 0 and last_price > 0
            else np.nan
        )
        reasons: list[str] = []

        float_value = np.nan
        float_available_at = pd.NaT
        if not meta_map.empty and ticker in meta_map.index:
            meta = meta_map.loc[ticker]
            float_value = float(pd.to_numeric(meta.get("float"), errors="coerce"))
            float_available_at = meta.get("_available_at", pd.NaT)
        if not np.isfinite(float_value):
            reasons.append("missing_point_in_time_float")
        elif float_value > config.max_float:
            reasons.append("float_above_limit")

        catalyst_available_at = pd.NaT
        if not catalyst_map.empty and ticker in catalyst_map.index:
            catalyst_available_at = catalyst_map.loc[ticker].get("_available_at", pd.NaT)
        if config.require_material_catalyst and pd.isna(catalyst_available_at):
            reasons.append("missing_material_catalyst")

        bid = ask = spread_pct = quote_age = np.nan
        quote_feed = ""
        if not quote_map.empty and ticker in quote_map.index:
            quote = quote_map.loc[ticker]
            bid = float(pd.to_numeric(quote.get("bid"), errors="coerce"))
            ask = float(pd.to_numeric(quote.get("ask"), errors="coerce"))
            quote_at = pd.Timestamp(quote.get("_available_at"))
            quote_age = max((decision - quote_at).total_seconds(), 0.0)
            quote_feed = str(quote.get("feed") or "").upper().strip()
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid * 100.0 if bid > 0 and ask >= bid and mid > 0 else np.nan
        if config.require_quote and not (np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid):
            reasons.append("missing_two_sided_quote")
        elif np.isfinite(quote_age) and quote_age > config.max_quote_age_seconds:
            reasons.append("stale_quote")
        elif np.isfinite(spread_pct) and spread_pct > config.max_spread_pct:
            reasons.append("spread_above_limit")
        if config.required_quote_feed and quote_feed != config.required_quote_feed.upper():
            reasons.append("unexpected_quote_feed")

        if not np.isfinite(prior_close):
            reasons.append("missing_prior_close")
        elif gap_pct < config.min_gap_pct:
            reasons.append("gap_below_limit")
        if premarket_volume < config.min_premarket_volume:
            reasons.append("premarket_volume_below_limit")
        if len(rvol_history) < config.min_rvol_history_sessions:
            reasons.append("insufficient_rvol_history")
        elif not np.isfinite(rvol) or rvol < config.min_rvol:
            reasons.append("rvol_below_limit")
        if not config.min_price <= last_price <= config.max_price:
            reasons.append("price_outside_range")
        if np.isnan(resistance_distance):
            reasons.append("missing_daily_resistance")
        elif resistance_distance < config.min_daily_resistance_distance_pct:
            reasons.append("near_daily_resistance")
        if ticker in halted:
            reasons.append("active_halt")

        score = (
            min(max(gap_pct, 0.0) / 30.0, 1.0)
            + min(max(rvol, 0.0) / 20.0, 1.0)
            + min(premarket_volume / 2_000_000.0, 1.0)
            + (1.0 - min(float_value / config.max_float, 1.0) if np.isfinite(float_value) else 0.0)
        )
        rows.append({
            "timestamp": decision, "ticker": ticker, "eligible": not reasons,
            "rejection_reasons": tuple(reasons), "scanner_rank": np.nan, "score": score,
            "last_price": last_price, "prior_close": prior_close, "gap_pct": gap_pct,
            "premarket_volume": premarket_volume, "rvol": rvol,
            "rvol_history_sessions": int(len(rvol_history)), "float": float_value,
            "float_available_at": float_available_at, "catalyst_available_at": catalyst_available_at,
            "catalyst_age_minutes": ((decision - catalyst_available_at).total_seconds() / 60.0) if pd.notna(catalyst_available_at) else np.nan,
            "bid": bid, "ask": ask, "spread_pct": spread_pct, "quote_age_seconds": quote_age,
            "daily_resistance": resistance, "daily_resistance_distance_pct": resistance_distance,
            "halted": ticker in halted,
        })
    out = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    if out.empty:
        return out
    eligible = out["eligible"]
    out.loc[eligible, "scanner_rank"] = out.loc[eligible, "score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["eligible", "scanner_rank", "ticker"], ascending=[False, True, True], na_position="last").reset_index(drop=True)


__all__ = ["SNAPSHOT_COLUMNS", "reconstruct_premarket_snapshot"]
