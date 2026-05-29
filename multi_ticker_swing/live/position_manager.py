"""
SwingPositionManager — tracks open option positions and fires exits based on
underlying price movements (matching the backtest exactly).

Exit conditions (checked on each 5m bar):
  1. Hard ATR stop:     underlying crosses sl_price
  2. Trailing stop:     trail arms at arm_pct underlying move, exits at 25% giveback
  3. ATR MFE no-prog:   at bar np_n_bars, if MFE < np_mfe_atr × ATR → exit (tier 2/3)

All closes are submitted via AlpacaOptionsClient.submit_option_order().
"""
from __future__ import annotations

import logging
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from API.Alpaca_API.options.options_api import AlpacaOptionsClient
from multi_ticker_swing.live.universe import TickerConfig

logger = logging.getLogger(__name__)

# Legacy defaults — individual positions now use config.arm_pct / config.giveback_pct
_ARM_PCT_DEFAULT      = 0.025
_GIVEBACK_PCT_DEFAULT = 0.25
_CLOSE_FAILURE_RETRY_SECS = 60
_CLOSE_ORDER_ATTEMPTS = 5
_ORDER_VERIFY_TIMEOUT_SECS = 2.5
_ORDER_VERIFY_POLL_SECS = 0.5
_OPTION_TICK = 0.01
_PENDING_CLOSE_REFRESH_SECS = 30.0
_LIQUIDATION_CLOSE_REASONS = {
    "sl",
    "no_progress",
    "deferred_trail_failed",
    "deferred_trail_timeout",
    "expiration_itm_cutoff",
}
_ET = ZoneInfo("America/New_York")
_DEFER_TRAIL_AFTER_HOUR = 15
_DEFER_TRAIL_AFTER_MINUTE = 55
_DEFER_RECOVERY_BARS = 3
_DEFER_RECOVERY_PCT = 0.0025
_DEFERRED_TRAIL_STATE_PATH = Path("Data/inference/multi_ticker_swing/deferred_trails.json")
_WORTHLESS_CLOSE_STATE_PATH = Path("Data/inference/multi_ticker_swing/worthless_close_abandoned.json")
_EXPIRING_ITM_CLOSE_HOUR = 15
_EXPIRING_ITM_CLOSE_MINUTE = 45
_ASSIGNED_EQUITY_MIN_SHARES = 100
_OPTION_VALUE_EXIT_ENABLED = True
_OPTION_VALUE_QUOTE_MODE = "bid"
_OPTION_PROFIT_TRAIL_ARM_PCT = 1.00
_OPTION_PROFIT_TRAIL_GIVEBACK_PCT = 0.25
_OPTION_TAKE_PROFIT_PCT = 3.00

EventSink = Callable[[str, dict], None]


@dataclass(frozen=True)
class ParsedOptionSymbol:
    root: str
    expiration: date
    call_put: str
    strike: float


@dataclass
class SwingPosition:
    ticker: str
    direction: int           # 1=long, -1=short
    entry_price: float
    entry_time: datetime
    atr_at_entry: float
    option_symbol: str
    qty: int
    config: TickerConfig
    option_entry_price: float | None = None
    option_entry_meta: dict[str, Any] | None = None

    # Derived at entry
    sl_price: float | None = None

    # Mutable tracking state
    best_price: float = field(init=False)
    last_price: float = field(init=False)
    trail_armed: bool = False
    bar_count_5m: int = 0
    option_last_price: float | None = None
    option_best_price: float | None = None
    option_trail_armed: bool = False
    deferred_trail_active: bool = False
    deferred_trail_trigger_time: datetime | None = None
    deferred_trail_trigger_price: float | None = None
    deferred_trail_trigger_pnl_pct: float | None = None
    deferred_trail_bars: int = 0
    deferred_trail_last_price: float | None = None

    def __post_init__(self):
        self.best_price = self.entry_price
        self.last_price = self.entry_price
        option_entry = _as_float(self.option_entry_price)
        if math.isfinite(option_entry) and option_entry > 0.0:
            self.option_entry_price = float(option_entry)
            self.option_last_price = float(option_entry)
            self.option_best_price = float(option_entry)
        else:
            self.option_entry_price = None
            self.option_last_price = None
            self.option_best_price = None
        if self.config.sl_atr > 0 and not _isnan(self.atr_at_entry):
            self.sl_price = (
                self.entry_price - self.direction * self.config.sl_atr * self.atr_at_entry
            )

    def update(self, bar: dict, *, allow_trail_exit: bool = True) -> str | None:
        """
        Process one 5m underlying bar. Returns exit_reason string if exit fires, else None.
        bar: {open, high, low, close, ...}
        """
        self.bar_count_5m += 1
        bar_h = float(bar["high"])
        bar_l = float(bar["low"])
        bar_c = float(bar["close"])
        self.last_price = bar_c

        # 1. Hard ATR stop
        if self.sl_price is not None:
            if self.direction == 1 and bar_l <= self.sl_price:
                return "sl"
            if self.direction == -1 and bar_h >= self.sl_price:
                return "sl"

        # 2. Track MFE (best_price = max favorable price from entry)
        if self.direction == 1:
            self.best_price = max(self.best_price, bar_h)
        else:
            self.best_price = min(self.best_price, bar_l)

        # 3. Trailing stop (arm/giveback thresholds are per-tier via TickerConfig)
        arm_pct      = self.config.arm_pct      if self.config else _ARM_PCT_DEFAULT
        giveback_pct = self.config.giveback_pct if self.config else _GIVEBACK_PCT_DEFAULT
        move_pct = self.direction * (self.best_price - self.entry_price) / self.entry_price
        if move_pct >= arm_pct:
            self.trail_armed = True
        if self.trail_armed:
            peak_profit  = self.direction * (self.best_price - self.entry_price)
            cur_profit   = self.direction * (bar_c        - self.entry_price)
            floor_profit = peak_profit * (1.0 - giveback_pct)
            if allow_trail_exit and cur_profit <= floor_profit:
                return "trail"

        # 4. ATR MFE no-progress (checked once at bar N)
        np_n = self.config.np_n_bars
        np_t = self.config.np_mfe_atr
        if np_n is not None and np_t is not None and self.bar_count_5m == np_n:
            if not self.trail_armed and not _isnan(self.atr_at_entry) and self.atr_at_entry > 0:
                mfe_atr = self.direction * (self.best_price - self.entry_price) / self.atr_at_entry
                if mfe_atr < np_t:
                    return "no_progress"

        return None

    def mark_deferred_trail(self, bar: dict) -> None:
        trigger_price = float(bar["close"])
        ts = bar.get("timestamp")
        self.deferred_trail_active = True
        self.deferred_trail_trigger_time = ts if isinstance(ts, datetime) else datetime.now(timezone.utc)
        self.deferred_trail_trigger_price = trigger_price
        self.deferred_trail_trigger_pnl_pct = (
            self.direction * (trigger_price - self.entry_price) / self.entry_price
            if self.entry_price else 0.0
        )
        self.deferred_trail_bars = 0
        self.deferred_trail_last_price = trigger_price

    def clear_deferred_trail(self) -> None:
        self.deferred_trail_active = False
        self.deferred_trail_trigger_time = None
        self.deferred_trail_trigger_price = None
        self.deferred_trail_trigger_pnl_pct = None
        self.deferred_trail_bars = 0
        self.deferred_trail_last_price = None

    def update_option_value(self, option_price: float | None) -> str | None:
        price = _as_float(option_price)
        if not math.isfinite(price) or price <= 0.0:
            return None
        self.option_last_price = float(price)
        if self.option_entry_price is None or self.option_entry_price <= 0.0:
            self.option_entry_price = float(price)
            self.option_best_price = float(price)
            return None

        if self.option_best_price is None or price > float(self.option_best_price):
            self.option_best_price = float(price)

        entry = float(self.option_entry_price)
        best = float(self.option_best_price or price)
        current_profit = price - entry
        peak_profit = best - entry
        if entry <= 0.0 or peak_profit <= 0.0:
            return None

        current_pct = current_profit / entry
        peak_pct = peak_profit / entry
        if _OPTION_TAKE_PROFIT_PCT > 0.0 and current_pct >= _OPTION_TAKE_PROFIT_PCT:
            return "option_take_profit"

        if peak_pct >= _OPTION_PROFIT_TRAIL_ARM_PCT:
            self.option_trail_armed = True
        if self.option_trail_armed:
            floor_profit = peak_profit * (1.0 - _OPTION_PROFIT_TRAIL_GIVEBACK_PCT)
            if current_profit <= floor_profit:
                return "option_profit_trail"
        return None

    def deferred_trail_decision(self, bar: dict) -> str | None:
        """Return close reason when a deferred trail should finally exit.

        A late-session trail gets one short next-session recovery window. If price
        resumes in the trade direction, we clear the deferral and let the normal
        trailing logic manage the renewed trend. If the next session opens weak
        or fails to recover quickly, close the position.
        """
        if not self.deferred_trail_active or self.deferred_trail_trigger_price is None:
            return None

        trigger_ts = self.deferred_trail_trigger_time
        bar_ts = bar.get("timestamp")
        if not isinstance(trigger_ts, datetime) or not isinstance(bar_ts, datetime):
            return None

        trigger_day = trigger_ts.astimezone(_ET).date()
        bar_day = bar_ts.astimezone(_ET).date()
        if bar_day <= trigger_day:
            return None

        bar_c = float(bar["close"])
        prior = self.deferred_trail_last_price or self.deferred_trail_trigger_price
        self.deferred_trail_bars += 1
        self.deferred_trail_last_price = bar_c

        favorable_from_trigger = (
            self.direction * (bar_c - self.deferred_trail_trigger_price)
            / self.deferred_trail_trigger_price
        )
        favorable_from_prior = self.direction * (bar_c - prior)

        if favorable_from_trigger >= _DEFER_RECOVERY_PCT and favorable_from_prior >= 0:
            self.clear_deferred_trail()
            return None

        if favorable_from_trigger < 0:
            return "deferred_trail_failed"

        if self.deferred_trail_bars >= _DEFER_RECOVERY_BARS:
            return "deferred_trail_timeout"

        return None

    def _trail_floor(self) -> float | None:
        """Price at which the trailing stop would trigger (None if not yet armed)."""
        if not self.trail_armed:
            return None
        giveback_pct = self.config.giveback_pct if self.config else _GIVEBACK_PCT_DEFAULT
        peak_profit = self.direction * (self.best_price - self.entry_price)
        floor_profit = peak_profit * (1.0 - giveback_pct)
        return self.entry_price + self.direction * floor_profit

    def to_dict(self) -> dict:
        pnl_pct = self.direction * (self.last_price - self.entry_price) / self.entry_price if self.entry_price else 0.0
        return {
            "ticker": self.ticker,
            "direction": int(self.direction),
            "entry_price": float(self.entry_price),
            "entry_time": self.entry_time.astimezone(timezone.utc).isoformat() if self.entry_time else None,
            "last_price": float(self.last_price),
            "best_price": float(self.best_price),
            "pnl_pct": float(pnl_pct),
            "sl_price": float(self.sl_price) if self.sl_price is not None else None,
            "trail_armed": bool(self.trail_armed),
            "trail_floor": float(tf) if (tf := self._trail_floor()) is not None else None,
            "deferred_trail_active": bool(self.deferred_trail_active),
            "option_entry_price": (
                float(self.option_entry_price)
                if self.option_entry_price is not None and math.isfinite(_as_float(self.option_entry_price))
                else None
            ),
            "option_last_price": (
                float(self.option_last_price)
                if self.option_last_price is not None and math.isfinite(_as_float(self.option_last_price))
                else None
            ),
            "option_best_price": (
                float(self.option_best_price)
                if self.option_best_price is not None and math.isfinite(_as_float(self.option_best_price))
                else None
            ),
            "option_trail_armed": bool(self.option_trail_armed),
            "deferred_trail_trigger_time": (
                self.deferred_trail_trigger_time.astimezone(timezone.utc).isoformat()
                if self.deferred_trail_trigger_time else None
            ),
            "deferred_trail_trigger_price": (
                float(self.deferred_trail_trigger_price)
                if self.deferred_trail_trigger_price is not None else None
            ),
            "deferred_trail_trigger_pnl_pct": (
                float(self.deferred_trail_trigger_pnl_pct)
                if self.deferred_trail_trigger_pnl_pct is not None else None
            ),
            "deferred_trail_bars": int(self.deferred_trail_bars),
            "bars_held": int(self.bar_count_5m),
            "atr_at_entry": float(self.atr_at_entry) if not _isnan(self.atr_at_entry) else None,
            "option_symbol": str(self.option_symbol),
            "option_entry_meta": self.option_entry_meta if isinstance(self.option_entry_meta, dict) else None,
            "qty": int(self.qty),
            "tier": int(self.config.tier) if self.config else None,
        }

    def to_chart_dict(self) -> dict:
        """Compact snapshot for position_bar_5m events (just the overlay fields)."""
        return {
            "ticker": self.ticker,
            "direction": int(self.direction),
            "entry_price": float(self.entry_price),
            "entry_time": self.entry_time.astimezone(timezone.utc).isoformat() if self.entry_time else None,
            "sl_price": float(self.sl_price) if self.sl_price is not None else None,
            "trail_armed": bool(self.trail_armed),
            "trail_floor": float(tf) if (tf := self._trail_floor()) is not None else None,
            "deferred_trail_active": bool(self.deferred_trail_active),
            "option_entry_price": (
                float(self.option_entry_price)
                if self.option_entry_price is not None and math.isfinite(_as_float(self.option_entry_price))
                else None
            ),
            "option_last_price": (
                float(self.option_last_price)
                if self.option_last_price is not None and math.isfinite(_as_float(self.option_last_price))
                else None
            ),
            "option_best_price": (
                float(self.option_best_price)
                if self.option_best_price is not None and math.isfinite(_as_float(self.option_best_price))
                else None
            ),
            "option_trail_armed": bool(self.option_trail_armed),
            "deferred_trail_trigger_price": (
                float(self.deferred_trail_trigger_price)
                if self.deferred_trail_trigger_price is not None else None
            ),
            "pnl_pct": float(self.direction * (self.last_price - self.entry_price) / self.entry_price) if self.entry_price else 0.0,
        }


def _isnan(v: float) -> bool:
    return v != v  # float NaN check without importing math


def _safe_response(resp: Any) -> Any:
    if resp is None:
        return None
    if isinstance(resp, (str, int, float, bool)):
        return resp
    if isinstance(resp, dict):
        keep = ("id", "client_order_id", "symbol", "qty", "side", "status", "submitted_at", "filled_avg_price")
        return {k: resp.get(k) for k in keep if k in resp}
    out = {}
    for k in ("id", "client_order_id", "symbol", "qty", "side", "status", "submitted_at", "filled_avg_price"):
        if hasattr(resp, k):
            out[k] = getattr(resp, k)
    return out or str(resp)


def _should_defer_trail_exit(bar: dict) -> bool:
    ts = bar.get("timestamp")
    if not isinstance(ts, datetime):
        return False
    local = ts.astimezone(_ET) if ts.tzinfo else ts.replace(tzinfo=_ET)
    return (
        local.hour > _DEFER_TRAIL_AFTER_HOUR
        or (
            local.hour == _DEFER_TRAIL_AFTER_HOUR
            and local.minute >= _DEFER_TRAIL_AFTER_MINUTE
        )
    )


def _past_expiring_itm_close_cutoff(ts: Any) -> bool:
    if not isinstance(ts, datetime):
        return False
    local = ts.astimezone(_ET) if ts.tzinfo else ts.replace(tzinfo=_ET)
    return (
        local.hour > _EXPIRING_ITM_CLOSE_HOUR
        or (
            local.hour == _EXPIRING_ITM_CLOSE_HOUR
            and local.minute >= _EXPIRING_ITM_CLOSE_MINUTE
        )
    )


def _expiring_itm_exit_reason(pos: SwingPosition, bar: dict) -> str | None:
    parsed = _parse_occ_option_symbol(pos.option_symbol)
    if parsed is None:
        return None
    ts = bar.get("timestamp")
    if not isinstance(ts, datetime):
        return None
    local = ts.astimezone(_ET) if ts.tzinfo else ts.replace(tzinfo=_ET)
    if parsed.expiration != local.date():
        return None
    if not _past_expiring_itm_close_cutoff(ts):
        return None

    close = _as_float(bar.get("close"))
    if not math.isfinite(close):
        return None
    if parsed.call_put == "C" and close > parsed.strike:
        return "expiration_itm_cutoff"
    if parsed.call_put == "P" and close < parsed.strike:
        return "expiration_itm_cutoff"
    return None


def _safe_bar(bar: dict) -> dict[str, Any]:
    ts = bar.get("timestamp")
    return {
        "timestamp": ts.astimezone(timezone.utc).isoformat() if isinstance(ts, datetime) else ts,
        "open": _finite_or_none(bar.get("open")),
        "high": _finite_or_none(bar.get("high")),
        "low": _finite_or_none(bar.get("low")),
        "close": _finite_or_none(bar.get("close")),
        "volume": _finite_or_none(bar.get("volume")),
    }


class SwingPositionManager:
    """
    Manages all open positions across all tickers.
    Thread-safe via a simple lock — updated from the 5m WebSocket callback thread.
    """

    def __init__(
        self,
        alpaca_client: AlpacaOptionsClient,
        dry_run: bool = False,
        event_sink: EventSink | None = None,
        auto_flatten_assigned_equities: bool = True,
    ) -> None:
        self._client = alpaca_client
        self._dry_run = dry_run
        self._sink = event_sink
        self._auto_flatten_assigned_equities = bool(auto_flatten_assigned_equities)
        self._positions: dict[str, SwingPosition] = {}   # ticker → position
        self._last_close_failure_wall: dict[str, float] = {}
        self._pending_close_orders: dict[str, dict[str, Any]] = {}
        self._deferred_trail_cache = self._load_deferred_trail_cache()
        self._worthless_close_abandoned = self._load_worthless_close_abandoned()

    def _emit(self, kind: str, payload: dict) -> None:
        if self._sink is None:
            return
        try:
            self._sink(kind, payload)
        except Exception as exc:
            logger.warning("event_sink raised on %s: %s", kind, exc)

    @property
    def open_tickers(self) -> set[str]:
        return set(self._positions.keys())

    def get_position(self, ticker: str) -> "SwingPosition | None":
        return self._positions.get(ticker)

    def snapshot(self) -> list[dict]:
        return [p.to_dict() for p in self._positions.values()]

    def sync_from_broker(
        self,
        *,
        universe: dict[str, TickerConfig],
        price_lookup: Callable[[str], float | None],
        atr_lookup: Callable[[str], float | None],
    ) -> dict[str, Any]:
        """Seed open swing option positions from Alpaca broker positions.

        Broker positions do not tell us the original underlying entry price or MFE
        trail state, so restored positions are anchored at the current warmed-up
        underlying close and managed from that point forward.
        """
        if self._dry_run:
            return {"synced": True, "restored": 0, "ignored": 0, "simulated": True}

        broker_positions, ignored = self._broker_swing_positions(universe)
        restored: list[dict[str, Any]] = []

        for broker_pos in broker_positions:
            ticker = broker_pos["ticker"]
            symbol = broker_pos["option_symbol"]
            if ticker in self._positions:
                ignored.append({"symbol": symbol, "ticker": ticker, "reason": "already_tracked"})
                continue

            pos = self._restore_broker_position(
                broker_pos,
                universe=universe,
                price_lookup=price_lookup,
                atr_lookup=atr_lookup,
                ignored=ignored,
            )
            if pos is not None:
                self._apply_deferred_trail_cache(pos)
                restored.append(pos.to_dict())

        result = {
            "synced": True,
            "restored": len(restored),
            "ignored": len(ignored),
            "positions": restored,
            "ignored_positions": ignored,
        }
        self._emit("broker_sync", {**result, "positions": restored, "ignored_positions": ignored})
        return result

    def reconcile_with_broker(
        self,
        *,
        universe: dict[str, TickerConfig],
        price_lookup: Callable[[str], float | None],
        atr_lookup: Callable[[str], float | None],
        reason: str = "periodic",
    ) -> dict[str, Any]:
        """Reconcile local swing tracking with current Alpaca option positions."""
        if self._dry_run:
            return {"ok": True, "simulated": True, "reason": reason}

        broker_positions, ignored = self._broker_swing_positions(universe)
        assigned_equities = self._broker_assigned_equity_positions(universe)
        broker_by_ticker: dict[str, dict[str, Any]] = {}
        duplicates: list[dict[str, Any]] = []
        for broker_pos in broker_positions:
            ticker = broker_pos["ticker"]
            if ticker in broker_by_ticker:
                duplicates.append(broker_pos)
                continue
            broker_by_ticker[ticker] = broker_pos

        removed: list[dict[str, Any]] = []
        replaced: list[dict[str, Any]] = []
        qty_updates: list[dict[str, Any]] = []
        restored: list[dict[str, Any]] = []

        for ticker, pos in list(self._positions.items()):
            broker_pos = broker_by_ticker.get(ticker)
            if broker_pos is None:
                payload = {
                    **pos.to_dict(),
                    "reason": "not_found_at_broker",
                    "reconcile_reason": reason,
                }
                removed.append(payload)
                self._positions.pop(ticker, None)
                self._emit("broker_position_missing", payload)
                continue

            broker_symbol = str(broker_pos.get("option_symbol", "")).upper()
            local_symbol = str(pos.option_symbol).upper()
            if broker_symbol != local_symbol:
                old_payload = {
                    **pos.to_dict(),
                    "broker_option_symbol": broker_symbol,
                    "reason": "broker_symbol_changed",
                    "reconcile_reason": reason,
                }
                replaced.append(old_payload)
                self._positions.pop(ticker, None)
                restored_pos = self._restore_broker_position(
                    broker_pos,
                    universe=universe,
                    price_lookup=price_lookup,
                    atr_lookup=atr_lookup,
                    ignored=ignored,
                )
                if restored_pos is not None:
                    restored.append(restored_pos.to_dict())
                continue

            broker_qty = int(broker_pos.get("qty", pos.qty) or pos.qty)
            if broker_qty != pos.qty:
                qty_updates.append({
                    "ticker": ticker,
                    "option_symbol": pos.option_symbol,
                    "old_qty": int(pos.qty),
                    "new_qty": broker_qty,
                })
                pos.qty = broker_qty
            broker_avg = _as_float(broker_pos.get("avg_entry_price"))
            if (
                math.isfinite(broker_avg)
                and broker_avg > 0.0
                and (pos.option_entry_price is None or pos.option_entry_price <= 0.0)
            ):
                pos.option_entry_price = float(broker_avg)
                pos.option_last_price = float(broker_avg)
                pos.option_best_price = float(broker_avg)

        for ticker, broker_pos in broker_by_ticker.items():
            if ticker in self._positions:
                continue
            pos = self._restore_broker_position(
                broker_pos,
                universe=universe,
                price_lookup=price_lookup,
                atr_lookup=atr_lookup,
                ignored=ignored,
            )
            if pos is not None:
                restored.append(pos.to_dict())

        flattened_equities: list[dict[str, Any]] = []
        if assigned_equities and self._auto_flatten_assigned_equities:
            flattened_equities = self.flatten_assigned_equity_positions(assigned_equities)

        result = {
            "ok": True,
            "reason": reason,
            "broker_positions": len(broker_positions),
            "local_positions": len(self._positions),
            "restored": len(restored),
            "removed": len(removed),
            "replaced": len(replaced),
            "qty_updates": qty_updates,
            "positions": restored,
            "removed_positions": removed,
            "replaced_positions": replaced,
            "ignored_positions": ignored,
            "duplicate_positions": duplicates,
            "assigned_equity_positions": assigned_equities,
            "flattened_assigned_equities": flattened_equities,
        }
        if assigned_equities:
            self._emit("assigned_equity_detected", {
                "reason": reason,
                "positions": assigned_equities,
                "count": len(assigned_equities),
            })
        self._emit("broker_reconcile", result)
        return result

    def _broker_swing_positions(
        self,
        universe: dict[str, TickerConfig],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        resp = self._client.get_positions()
        raw_positions = _extract_positions(resp)
        positions: list[dict[str, Any]] = []
        ignored: list[dict[str, Any]] = []

        for raw in raw_positions:
            symbol = str(raw.get("symbol", "")).strip().upper()
            parsed = _parse_swing_option_symbol(symbol, universe)
            if parsed is None:
                continue
            ticker, direction = parsed
            if symbol in self._worthless_close_abandoned:
                ignored.append({"symbol": symbol, "ticker": ticker, "reason": "worthless_close_abandoned"})
                continue

            side_raw = str(raw.get("side", "")).strip().lower()
            qty_val = _as_float(raw.get("qty"))
            side_mult = -1 if side_raw == "short" or (math.isfinite(qty_val) and qty_val < 0) else 1
            qty = int(round(abs(qty_val))) if math.isfinite(qty_val) else 0
            if side_mult <= 0 or qty <= 0:
                ignored.append({"symbol": symbol, "ticker": ticker, "reason": "not_long_option_position"})
                continue

            positions.append({
                "ticker": ticker,
                "direction": direction,
                "option_symbol": symbol,
                "qty": qty,
                "side": side_raw or "long",
                "avg_entry_price": _finite_or_none(raw.get("avg_entry_price")),
                "market_value": _finite_or_none(raw.get("market_value")),
                "unrealized_pl": _finite_or_none(raw.get("unrealized_pl")),
                "unrealized_plpc": _finite_or_none(raw.get("unrealized_plpc")),
            })
        return positions, ignored

    def _broker_assigned_equity_positions(
        self,
        universe: dict[str, TickerConfig],
    ) -> list[dict[str, Any]]:
        """Detect possible exercised/assigned option lots now held as shares.

        Alpaca exposes exercised/assigned options as plain equity positions. The
        live option manager cannot prove intent here, so this method only flags
        whole 100-share lots in the swing universe for operator/audit handling.
        """
        resp = self._client.get_positions()
        raw_positions = _extract_positions(resp)
        positions: list[dict[str, Any]] = []

        for raw in raw_positions:
            symbol = str(raw.get("symbol", "")).strip().upper()
            if symbol not in universe:
                continue
            if _parse_occ_option_symbol(symbol) is not None:
                continue
            qty_val = _as_float(raw.get("qty"))
            if not math.isfinite(qty_val):
                continue
            abs_qty = abs(qty_val)
            if abs_qty < _ASSIGNED_EQUITY_MIN_SHARES:
                continue
            lots = abs_qty / _ASSIGNED_EQUITY_MIN_SHARES
            if not math.isclose(lots, round(lots), rel_tol=0.0, abs_tol=1e-6):
                continue

            side_raw = str(raw.get("side", "")).strip().lower()
            side = side_raw or ("long" if qty_val > 0 else "short")
            close_side = "sell" if qty_val > 0 else "buy"
            positions.append({
                "symbol": symbol,
                "qty": qty_val,
                "side": side,
                "lots_100": int(round(lots)),
                "suggested_close_side": close_side,
                "avg_entry_price": _finite_or_none(raw.get("avg_entry_price")),
                "market_value": _finite_or_none(raw.get("market_value")),
                "unrealized_pl": _finite_or_none(raw.get("unrealized_pl")),
                "unrealized_plpc": _finite_or_none(raw.get("unrealized_plpc")),
                "reason": "possible_option_exercise_assignment",
            })

        return positions

    def flatten_assigned_equity_positions(
        self,
        positions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Submit market orders to close detected assignment/exercise share lots."""
        results: list[dict[str, Any]] = []
        if self._dry_run:
            for pos in positions:
                result = {**pos, "simulated": True, "submitted": False}
                results.append(result)
                self._emit("assigned_equity_flatten_dry_run", result)
            return results

        for pos in positions:
            symbol = str(pos.get("symbol", "")).strip().upper()
            qty_val = _as_float(pos.get("qty"))
            close_side = str(pos.get("suggested_close_side", "")).strip().lower()
            qty = int(round(abs(qty_val))) if math.isfinite(qty_val) else 0
            if not symbol or qty <= 0 or close_side not in {"buy", "sell"}:
                result = {**pos, "submitted": False, "error": "invalid_equity_flatten_position"}
                results.append(result)
                self._emit("assigned_equity_flatten_failed", result)
                continue

            try:
                resp = self._client.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=close_side,
                    order_type="market",
                    time_in_force="day",
                )
                result = {
                    **pos,
                    "submitted": True,
                    "close_side": close_side,
                    "close_qty": qty,
                    "response": _safe_response(resp),
                }
                self._emit("assigned_equity_flatten_order_submitted", result)
            except Exception as exc:
                result = {
                    **pos,
                    "submitted": False,
                    "close_side": close_side,
                    "close_qty": qty,
                    "error": str(exc),
                }
                self._emit("assigned_equity_flatten_failed", result)
            results.append(result)

        return results

    def _restore_broker_position(
        self,
        broker_pos: dict[str, Any],
        *,
        universe: dict[str, TickerConfig],
        price_lookup: Callable[[str], float | None],
        atr_lookup: Callable[[str], float | None],
        ignored: list[dict[str, Any]],
    ) -> SwingPosition | None:
        ticker = str(broker_pos.get("ticker", "")).upper()
        symbol = str(broker_pos.get("option_symbol", "")).upper()
        if ticker not in universe:
            ignored.append({"symbol": symbol, "ticker": ticker, "reason": "outside_universe"})
            return None

        entry_price = price_lookup(ticker)
        atr = atr_lookup(ticker)
        if entry_price is None or not math.isfinite(float(entry_price)):
            ignored.append({"symbol": symbol, "ticker": ticker, "reason": "missing_underlying_price"})
            return None
        if atr is None or not math.isfinite(float(atr)) or float(atr) <= 0:
            ignored.append({"symbol": symbol, "ticker": ticker, "reason": "missing_atr"})
            return None

        pos = SwingPosition(
            ticker=ticker,
            direction=int(broker_pos.get("direction", 1) or 1),
            entry_price=float(entry_price),
            entry_time=datetime.now(timezone.utc),
            atr_at_entry=float(atr),
            option_symbol=symbol,
            qty=int(broker_pos.get("qty", 0) or 0),
            config=universe[ticker],
            option_entry_price=_finite_or_none(broker_pos.get("avg_entry_price")),
        )
        self._apply_deferred_trail_cache(pos)
        self.open_position(pos)
        return pos

    def open_position(self, pos: SwingPosition) -> None:
        """Register a newly entered position."""
        self._positions[pos.ticker] = pos
        logger.info(
            "[%s] OPEN  dir=%+d  entry=%.2f  sl=%s  option=%s  qty=%d",
            pos.ticker, pos.direction, pos.entry_price,
            f"{pos.sl_price:.2f}" if pos.sl_price else "none",
            pos.option_symbol, pos.qty,
        )
        self._emit("position_opened", pos.to_dict())

    def on_5m_bar(self, ticker: str, bar: dict) -> None:
        """
        Called from the 5m bar stream for every bar on every ticker.
        If ticker has an open position, checks exit conditions and closes if triggered.
        """
        pos = self._positions.get(ticker)
        if pos is None:
            return

        expiration_reason = _expiring_itm_exit_reason(pos, bar)
        if expiration_reason:
            self._emit("expiring_itm_exit_triggered", {
                **pos.to_dict(),
                "reason": expiration_reason,
                "bar": _safe_bar(bar),
                "cutoff_et": f"{_EXPIRING_ITM_CLOSE_HOUR:02d}:{_EXPIRING_ITM_CLOSE_MINUTE:02d}",
            })
            self._close_position(pos, expiration_reason, bar)
            return

        option_reason = self._option_value_exit_reason(pos)
        if option_reason:
            self._emit("option_value_exit_triggered", {
                **pos.to_dict(),
                "reason": option_reason,
                "bar": _safe_bar(bar),
                "arm_pct": _OPTION_PROFIT_TRAIL_ARM_PCT,
                "giveback_pct": _OPTION_PROFIT_TRAIL_GIVEBACK_PCT,
                "take_profit_pct": _OPTION_TAKE_PROFIT_PCT,
            })
            self._close_position(pos, option_reason, bar)
            return

        if pos.deferred_trail_active:
            reason = pos.update(bar, allow_trail_exit=False)
            if reason:
                self._close_position(pos, reason, bar)
                return

            was_deferred = pos.deferred_trail_active
            deferred_reason = pos.deferred_trail_decision(bar)
            if deferred_reason:
                self._close_position(pos, deferred_reason, bar)
                return
            if was_deferred and not pos.deferred_trail_active:
                self._emit("position_trail_resumed", {
                    **pos.to_dict(),
                    "reason": "deferred_trail_recovered",
                    "bar": _safe_bar(bar),
                })
                self._remove_deferred_trail_cache(pos)
            return

        reason = pos.update(bar)
        if reason:
            if reason == "trail" and _should_defer_trail_exit(bar):
                pos.mark_deferred_trail(bar)
                self._persist_deferred_trail_cache()
                logger.info(
                    "[%s] DEFER trail exit until next session  pnl=%+.2f%%  bars=%d  option=%s",
                    pos.ticker,
                    (pos.deferred_trail_trigger_pnl_pct or 0.0) * 100.0,
                    pos.bar_count_5m,
                    pos.option_symbol,
                )
                self._emit("position_trail_deferred", {
                    **pos.to_dict(),
                    "reason": "late_session_trail",
                    "bar": _safe_bar(bar),
                    "recovery_bars": _DEFER_RECOVERY_BARS,
                    "recovery_pct": _DEFER_RECOVERY_PCT,
                })
                return
            self._close_position(pos, reason, bar)

    def _option_value_exit_reason(self, pos: SwingPosition) -> str | None:
        if not _OPTION_VALUE_EXIT_ENABLED:
            return None
        symbol = str(pos.option_symbol).strip().upper()
        if not symbol:
            return None
        price = self._get_contract_price(symbol=symbol, mode=_OPTION_VALUE_QUOTE_MODE)
        if not math.isfinite(price) or price <= 0.0:
            price = self._get_contract_price(symbol=symbol, mode="mid")
        return pos.update_option_value(price)

    def _close_position(self, pos: SwingPosition, reason: str, bar: dict) -> None:
        exit_price = float(bar["close"])
        pnl_pct = pos.direction * (exit_price - pos.entry_price) / pos.entry_price
        if str(pos.option_symbol).strip().upper() in self._worthless_close_abandoned:
            self._abandon_worthless_close(
                pos,
                reason=reason,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                close_result=None,
            )
            return
        pending = self._pending_close_orders.get(pos.ticker)
        if pending is not None and not self._dry_run:
            state = self._reconcile_pending_close_order(pos, pending=pending)
            if state.get("closed"):
                self._emit_position_closed(
                    pos,
                    reason=str(pending.get("reason") or reason),
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    order_error=None,
                )
                return
            if state.get("still_pending"):
                self._emit("position_close_pending", {
                    **pos.to_dict(),
                    "exit_price": exit_price,
                    "exit_pnl_pct": float(pnl_pct),
                    "exit_reason": reason,
                    "order_response": _safe_response(state.get("order")),
                    "verification": state.get("verification"),
                    "pending_since": pending.get("created_wall"),
                })
                return
            self._pending_close_orders.pop(pos.ticker, None)

        if not self._dry_run:
            last_failure = self._last_close_failure_wall.get(pos.ticker)
            now = time.monotonic()
            if last_failure is not None and (now - last_failure) < _CLOSE_FAILURE_RETRY_SECS:
                return
        logger.info(
            "[%s] CLOSE  reason=%-12s  pnl=%+.2f%%  bars=%d  option=%s",
            pos.ticker, reason, pnl_pct * 100, pos.bar_count_5m, pos.option_symbol,
        )
        order_resp = None
        order_error: str | None = None
        if not self._dry_run:
            try:
                close_result = self._submit_close_order(pos, reason=reason)
                order_resp = close_result.get("response")
                self._last_close_failure_wall.pop(pos.ticker, None)
                self._emit("order_submitted", {
                    "ticker": pos.ticker,
                    "side": "sell",
                    "option_symbol": pos.option_symbol,
                    "qty": pos.qty,
                    "exit_price": exit_price,
                    "limit_price": close_result.get("limit_price"),
                    "close_quote": close_result.get("close_quote"),
                    "response": _safe_response(order_resp),
                    "verification": close_result.get("verification"),
                })
                if close_result.get("abandoned_worthless"):
                    self._abandon_worthless_close(
                        pos,
                        reason=reason,
                        exit_price=exit_price,
                        pnl_pct=pnl_pct,
                        close_result=close_result,
                    )
                    return
                if not close_result.get("verified", False):
                    self._last_close_failure_wall[pos.ticker] = time.monotonic()
                    order_id = str((close_result.get("verification") or {}).get("order_id") or "").strip()
                    if order_id:
                        self._pending_close_orders[pos.ticker] = {
                            "order_id": order_id,
                            "symbol": pos.option_symbol,
                            "reason": reason,
                            "created_wall": time.monotonic(),
                            "last_checked_wall": time.monotonic(),
                            "order": (close_result.get("verification") or {}).get("order"),
                        }
                    self._emit("position_close_pending", {
                        **pos.to_dict(),
                        "exit_price": exit_price,
                        "exit_pnl_pct": float(pnl_pct),
                        "exit_reason": reason,
                        "order_response": _safe_response(order_resp),
                        "verification": close_result.get("verification"),
                    })
                    return
            except Exception as exc:
                order_error = str(exc)
                self._last_close_failure_wall[pos.ticker] = time.monotonic()
                logger.error("[%s] sell order FAILED: %s", pos.ticker, exc)
                self._emit("order_failed", {
                    "ticker": pos.ticker,
                    "side": "sell",
                    "option_symbol": pos.option_symbol,
                    "qty": pos.qty,
                    "error": order_error,
                })
                self._emit("position_close_failed", {
                    **pos.to_dict(),
                    "exit_price": exit_price,
                    "exit_pnl_pct": float(pnl_pct),
                    "exit_reason": reason,
                    "order_error": order_error,
                })
                return
        else:
            self._emit("order_dry_run", {
                "ticker": pos.ticker,
                "side": "sell",
                "option_symbol": pos.option_symbol,
                "qty": pos.qty,
                "exit_price": exit_price,
            })

        self._emit_position_closed(
            pos,
            reason=reason,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            order_error=order_error,
        )

    def _emit_position_closed(
        self,
        pos: SwingPosition,
        *,
        reason: str,
        exit_price: float,
        pnl_pct: float,
        order_error: str | None,
    ) -> None:
        self._emit("position_closed", {
            **pos.to_dict(),
            "exit_price": exit_price,
            "exit_pnl_pct": float(pnl_pct),
            "exit_reason": reason,
            "order_error": order_error,
        })
        self._last_close_failure_wall.pop(pos.ticker, None)
        self._pending_close_orders.pop(pos.ticker, None)
        self._remove_deferred_trail_cache(pos)
        del self._positions[pos.ticker]

    def _abandon_worthless_close(
        self,
        pos: SwingPosition,
        *,
        reason: str,
        exit_price: float,
        pnl_pct: float,
        close_result: dict[str, Any] | None,
    ) -> None:
        symbol = str(pos.option_symbol).strip().upper()
        self._worthless_close_abandoned.add(symbol)
        self._persist_worthless_close_abandoned()
        self._last_close_failure_wall.pop(pos.ticker, None)
        self._pending_close_orders.pop(pos.ticker, None)
        self._remove_deferred_trail_cache(pos)
        self._emit("position_close_abandoned", {
            **pos.to_dict(),
            "exit_price": exit_price,
            "exit_pnl_pct": float(pnl_pct),
            "exit_reason": reason,
            "option_symbol": symbol,
            "limit_price": _OPTION_TICK,
            "reason": "worthless_after_one_cent_close_attempt",
            "order_response": _safe_response((close_result or {}).get("response")),
            "verification": (close_result or {}).get("verification"),
        })
        self._positions.pop(pos.ticker, None)

    def _load_deferred_trail_cache(self) -> dict[str, dict[str, Any]]:
        try:
            if not _DEFERRED_TRAIL_STATE_PATH.exists():
                return {}
            raw = json.loads(_DEFERRED_TRAIL_STATE_PATH.read_text())
            if isinstance(raw, dict):
                return {
                    str(k).upper(): v
                    for k, v in raw.items()
                    if isinstance(v, dict)
                }
        except Exception as exc:
            logger.warning("deferred trail state load failed: %s", exc)
        return {}

    def _persist_deferred_trail_cache(self) -> None:
        cache: dict[str, dict[str, Any]] = {}
        for pos in self._positions.values():
            if not pos.deferred_trail_active:
                continue
            cache[pos.ticker.upper()] = {
                "ticker": pos.ticker.upper(),
                "option_symbol": str(pos.option_symbol).upper(),
                "trigger_time": (
                    pos.deferred_trail_trigger_time.astimezone(timezone.utc).isoformat()
                    if pos.deferred_trail_trigger_time else None
                ),
                "trigger_price": pos.deferred_trail_trigger_price,
                "trigger_pnl_pct": pos.deferred_trail_trigger_pnl_pct,
                "bars": pos.deferred_trail_bars,
                "last_price": pos.deferred_trail_last_price,
            }
        try:
            _DEFERRED_TRAIL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _DEFERRED_TRAIL_STATE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
            self._deferred_trail_cache = cache
        except Exception as exc:
            logger.warning("deferred trail state persist failed: %s", exc)

    def _load_worthless_close_abandoned(self) -> set[str]:
        try:
            if not _WORTHLESS_CLOSE_STATE_PATH.exists():
                return set()
            raw = json.loads(_WORTHLESS_CLOSE_STATE_PATH.read_text())
            if isinstance(raw, list):
                return {str(item).strip().upper() for item in raw if str(item).strip()}
            if isinstance(raw, dict):
                symbols = raw.get("symbols")
                if isinstance(symbols, list):
                    return {str(item).strip().upper() for item in symbols if str(item).strip()}
        except Exception as exc:
            logger.warning("worthless close state load failed: %s", exc)
        return set()

    def _persist_worthless_close_abandoned(self) -> None:
        try:
            _WORTHLESS_CLOSE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"symbols": sorted(self._worthless_close_abandoned)}
            _WORTHLESS_CLOSE_STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception as exc:
            logger.warning("worthless close state persist failed: %s", exc)

    def _apply_deferred_trail_cache(self, pos: SwingPosition) -> None:
        cached = self._deferred_trail_cache.get(pos.ticker.upper())
        if not cached:
            return
        if str(cached.get("option_symbol", "")).upper() != str(pos.option_symbol).upper():
            return
        trigger_time = _parse_dt(cached.get("trigger_time"))
        trigger_price = _as_float(cached.get("trigger_price"))
        if trigger_time is None or not math.isfinite(trigger_price) or trigger_price <= 0:
            return
        pos.deferred_trail_active = True
        pos.deferred_trail_trigger_time = trigger_time
        pos.deferred_trail_trigger_price = float(trigger_price)
        pos.deferred_trail_trigger_pnl_pct = _finite_or_none(cached.get("trigger_pnl_pct"))
        pos.deferred_trail_bars = int(cached.get("bars") or 0)
        last_price = _as_float(cached.get("last_price"))
        pos.deferred_trail_last_price = float(last_price) if math.isfinite(last_price) else float(trigger_price)

    def _remove_deferred_trail_cache(self, pos: SwingPosition) -> None:
        if pos.ticker.upper() not in self._deferred_trail_cache:
            return
        self._deferred_trail_cache.pop(pos.ticker.upper(), None)
        self._persist_deferred_trail_cache()

    def _submit_close_order(self, pos: SwingPosition, *, reason: str) -> dict[str, Any]:
        symbol = str(pos.option_symbol).strip().upper()
        qty = int(pos.qty)
        quote_meta = self._get_contract_quote_context(symbol=symbol)
        base_limit = self._get_contract_price(symbol=symbol, mode="bid")
        close_bid = base_limit
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            base_limit = self._get_contract_price(symbol=symbol, mode="mid")
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            base_limit = self._get_contract_price(symbol=symbol, mode="mark")
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            base_limit = _OPTION_TICK

        logger.info(
            "[%s] close order pricing symbol=%s source=%s base_limit=%.2f reason=%s",
            pos.ticker,
            symbol,
            "bid" if math.isfinite(close_bid) and close_bid > 0.0 else "fallback",
            base_limit,
            reason,
        )

        limit_prices = _close_limit_ladder(
            base_limit=base_limit,
            close_bid=close_bid,
            reason=reason,
            attempts=_CLOSE_ORDER_ATTEMPTS,
        )
        last_result: dict[str, Any] | None = None
        for attempt, limit_price in enumerate(limit_prices, start=1):
            try:
                resp = self._client.submit_option_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    order_type="limit",
                    time_in_force="day",
                    limit_price=limit_price,
                )
            except Exception as exc:
                if _is_option_tick(limit_price):
                    logger.info(
                        "[%s] close order abandoned after one-cent submit failure symbol=%s error=%s",
                        pos.ticker,
                        symbol,
                        exc,
                    )
                    return {
                        "response": {"error": str(exc), "symbol": symbol, "side": "sell", "qty": qty},
                        "verification": {
                            "verified": False,
                            "status": "submit_failed",
                            "order_id": "",
                            "via": "one_cent_submit_exception",
                            "retryable": False,
                            "error": str(exc),
                        },
                        "limit_price": limit_price,
                        "close_quote": quote_meta,
                        "verified": False,
                        "abandoned_worthless": True,
                    }
                raise
            status = _status_key(resp.get("status") if isinstance(resp, dict) else None)
            order_id = str(resp.get("id", "")).strip() if isinstance(resp, dict) else ""
            logger.info(
                "[%s] close order submitted symbol=%s order_id=%s status=%s limit=%.2f attempt=%d/%d",
                pos.ticker,
                symbol,
                order_id or "n/a",
                status or "n/a",
                limit_price,
                attempt,
                len(limit_prices),
            )

            verify = self._verify_close_order(submitted_resp=resp if isinstance(resp, dict) else {}, symbol=symbol)
            last_result = {
                "response": resp,
                "verification": verify,
                "limit_price": limit_price,
                "close_quote": quote_meta,
                "verified": bool(verify.get("verified")),
            }
            if verify.get("verified"):
                logger.info(
                    "[%s] close order verified symbol=%s status=%s via=%s",
                    pos.ticker,
                    symbol,
                    verify.get("status"),
                    verify.get("via"),
                )
                return last_result
            if _is_option_tick(limit_price):
                last_result["abandoned_worthless"] = True
                logger.info(
                    "[%s] close order abandoned after one-cent attempt symbol=%s status=%s via=%s",
                    pos.ticker,
                    symbol,
                    verify.get("status"),
                    verify.get("via"),
                )
                return last_result

            can_retry = bool(verify.get("retryable")) and attempt < _CLOSE_ORDER_ATTEMPTS
            logger.info(
                "[%s] close order not verified symbol=%s status=%s via=%s retrying=%s",
                pos.ticker,
                symbol,
                verify.get("status"),
                verify.get("via"),
                can_retry,
            )
            if can_retry:
                self._cancel_order_if_needed(verify)
                continue
            # Keep the last live close order working. The manager tracks it in
            # _pending_close_orders and reconciles it before any later retry.
            return last_result

        if last_result is not None:
            return last_result
        raise RuntimeError(f"close_order_submit_failed symbol={symbol}")

    def _get_contract_price(self, *, symbol: str, mode: str) -> float:
        try:
            resp = self._client.get_option_quotes(symbols=symbol, limit=1)
            quotes = _extract_quotes(resp, symbol=symbol)
            if not quotes:
                return float("nan")
            return _quote_price(quotes[-1], mode=mode)
        except Exception as exc:
            logger.warning("quote fetch failed symbol=%s mode=%s: %s", symbol, mode, exc)
        return float("nan")

    def _get_contract_quote_context(self, *, symbol: str) -> dict[str, Any]:
        try:
            resp = self._client.get_option_quotes(symbols=symbol, limit=1)
            quotes = _extract_quotes(resp, symbol=symbol)
            if not quotes:
                return {"quote_error": "no_quotes"}
            return _quote_context(quotes[-1])
        except Exception as exc:
            logger.warning("quote context fetch failed symbol=%s: %s", symbol, exc)
            return {"quote_error": str(exc)}

    def _verify_close_order(self, *, submitted_resp: dict[str, Any], symbol: str) -> dict[str, Any]:
        order_id = str(submitted_resp.get("id", "")).strip()
        last = submitted_resp
        status = _status_key(last.get("status"))
        if _order_is_success(status):
            return {"verified": True, "status": status, "order_id": order_id, "via": "submit_response", "order": last}
        if _order_is_terminal_fail(status):
            return {"verified": False, "status": status, "order_id": order_id, "via": "submit_response", "retryable": True, "cancel_required": False, "order": last}

        deadline = time.monotonic() + _ORDER_VERIFY_TIMEOUT_SECS
        while time.monotonic() < deadline and order_id:
            time.sleep(_ORDER_VERIFY_POLL_SECS)
            try:
                current = self._client.get_order(order_id)
                if isinstance(current, dict):
                    last = current
                    status = _status_key(current.get("status"))
            except Exception as exc:
                logger.warning("close order verify poll warning order_id=%s: %s", order_id, exc)
                continue
            if _order_is_success(status):
                return {"verified": True, "status": status, "order_id": order_id, "via": "order_poll", "order": last}
            if _order_is_terminal_fail(status):
                return {"verified": False, "status": status, "order_id": order_id, "via": "order_poll", "retryable": True, "cancel_required": False, "order": last}

        try:
            if not self._has_open_long_position(symbol=symbol):
                return {"verified": True, "status": status or "unknown", "order_id": order_id, "via": "positions_reconcile", "order": last}
        except Exception:
            pass

        return {
            "verified": False,
            "status": status or "unknown",
            "order_id": order_id,
            "via": "timeout",
            "retryable": bool(order_id),
            "cancel_required": bool(order_id),
            "order": last,
        }

    def _reconcile_pending_close_order(
        self,
        pos: SwingPosition,
        *,
        pending: dict[str, Any],
    ) -> dict[str, Any]:
        order_id = str(pending.get("order_id", "")).strip()
        symbol = str(pending.get("symbol") or pos.option_symbol).strip().upper()
        last = pending.get("order") if isinstance(pending.get("order"), dict) else {}
        status = _status_key(last.get("status"))
        now = time.monotonic()
        if (now - float(pending.get("last_checked_wall") or 0.0)) < _PENDING_CLOSE_REFRESH_SECS:
            return {
                "still_pending": True,
                "verification": {
                    "verified": False,
                    "status": status or "pending",
                    "order_id": order_id,
                    "via": "pending_close_cache",
                    "retryable": False,
                    "order": last,
                },
                "order": last,
            }
        pending["last_checked_wall"] = now

        if order_id:
            try:
                current = self._client.get_order(order_id)
                if isinstance(current, dict):
                    last = current
                    pending["order"] = current
                    status = _status_key(current.get("status"))
            except Exception as exc:
                logger.warning("pending close order poll warning order_id=%s: %s", order_id, exc)

        if _order_is_success(status):
            return {
                "closed": True,
                "verification": {
                    "verified": True,
                    "status": status,
                    "order_id": order_id,
                    "via": "pending_order_poll",
                    "order": last,
                },
                "order": last,
            }
        if _order_is_terminal_fail(status):
            return {
                "terminal": True,
                "verification": {
                    "verified": False,
                    "status": status,
                    "order_id": order_id,
                    "via": "pending_order_poll",
                    "retryable": True,
                    "order": last,
                },
                "order": last,
            }

        try:
            if not self._has_open_long_position(symbol=symbol):
                return {
                    "closed": True,
                    "verification": {
                        "verified": True,
                        "status": status or "unknown",
                        "order_id": order_id,
                        "via": "positions_reconcile",
                        "order": last,
                    },
                    "order": last,
                }
        except Exception:
            pass

        return {
            "still_pending": True,
            "verification": {
                "verified": False,
                "status": status or "unknown",
                "order_id": order_id,
                "via": "pending_order_poll",
                "retryable": False,
                "order": last,
            },
            "order": last,
        }

    def _cancel_order_if_needed(self, verify_result: dict[str, Any]) -> None:
        if not bool(verify_result.get("cancel_required")):
            return
        order_id = str(verify_result.get("order_id", "")).strip()
        if not order_id:
            return
        try:
            self._client.cancel_order(order_id)
            logger.info("close order canceled before retry order_id=%s", order_id)
        except Exception as exc:
            logger.warning("close order cancel warning order_id=%s: %s", order_id, exc)

    def _has_open_long_position(self, *, symbol: str) -> bool:
        target = str(symbol).strip().upper()
        resp = self._client.get_positions()
        for raw in _extract_positions(resp):
            if str(raw.get("symbol", "")).strip().upper() != target:
                continue
            side_raw = str(raw.get("side", "")).strip().lower()
            qty_val = _as_float(raw.get("qty"))
            if side_raw == "short":
                continue
            if side_raw == "long":
                return True
            if math.isfinite(qty_val) and qty_val > 0:
                return True
        return False


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite_or_none(value: Any) -> float | None:
    out = _as_float(value)
    return float(out) if math.isfinite(out) else None


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _status_key(status: Any) -> str:
    return str(status or "").strip().lower()


def _order_is_success(status: Any) -> bool:
    return _status_key(status) in {"filled", "partially_filled"}


def _order_is_terminal_fail(status: Any) -> bool:
    return _status_key(status) in {
        "canceled",
        "cancelled",
        "expired",
        "rejected",
        "failed",
        "suspended",
    }


def _extract_positions(resp: Any) -> list[dict[str, Any]]:
    if isinstance(resp, list):
        return [x for x in resp if isinstance(x, dict)]
    if isinstance(resp, dict):
        for key in ("positions", "data"):
            value = resp.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _extract_quotes(resp: Any, *, symbol: str) -> list[dict[str, Any]]:
    sym = str(symbol).strip().upper()
    if isinstance(resp, dict):
        quotes_obj = resp.get("quotes")
        if isinstance(quotes_obj, dict):
            for key, value in quotes_obj.items():
                if str(key).strip().upper() != sym:
                    continue
                if isinstance(value, list):
                    return [q for q in value if isinstance(q, dict)]
                if isinstance(value, dict):
                    return [value]
        if isinstance(quotes_obj, list):
            return [
                q for q in quotes_obj
                if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
            ]
        value = resp.get("data")
        if isinstance(value, list):
            return [
                q for q in value
                if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
            ]
        if isinstance(value, dict):
            maybe = value.get(sym) or value.get(sym.lower()) or value.get(sym.upper())
            if isinstance(maybe, list):
                return [q for q in maybe if isinstance(q, dict)]
            if isinstance(maybe, dict):
                return [maybe]
    if isinstance(resp, list):
        return [
            q for q in resp
            if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
        ]
    return []


def _quote_price(quote: dict[str, Any], *, mode: str) -> float:
    mode_key = str(mode or "bid").strip().lower()
    ask = _as_float(quote.get("ask_price", quote.get("ap", quote.get("ask"))))
    bid = _as_float(quote.get("bid_price", quote.get("bp", quote.get("bid"))))
    last = _as_float(quote.get("last_price", quote.get("lp")))
    mark = _as_float(quote.get("mark_price", quote.get("mark")))
    if mode_key == "bid":
        return bid
    if mode_key == "ask":
        return ask
    if mode_key == "last":
        return last
    if mode_key == "mark":
        if math.isfinite(mark):
            return mark
        if math.isfinite(bid) and math.isfinite(ask):
            return 0.5 * (bid + ask)
        return float("nan")
    if mode_key == "mid":
        if math.isfinite(bid) and math.isfinite(ask):
            return 0.5 * (bid + ask)
        if math.isfinite(mark):
            return mark
        if math.isfinite(last):
            return last
        if math.isfinite(bid):
            return bid
        if math.isfinite(ask):
            return ask
    if math.isfinite(bid):
        return bid
    if math.isfinite(mark):
        return mark
    if math.isfinite(last):
        return last
    if math.isfinite(ask):
        return ask
    return float("nan")


def _quote_context(quote: dict[str, Any]) -> dict[str, Any]:
    bid = _quote_price(quote, mode="bid")
    ask = _quote_price(quote, mode="ask")
    mid = _quote_price(quote, mode="mid")
    mark = _quote_price(quote, mode="mark")
    last = _quote_price(quote, mode="last")
    spread = ask - bid if math.isfinite(bid) and math.isfinite(ask) else float("nan")
    spread_pct_mid = spread / mid if math.isfinite(spread) and math.isfinite(mid) and mid > 0 else float("nan")
    return {
        "bid": float(bid) if math.isfinite(bid) else None,
        "ask": float(ask) if math.isfinite(ask) else None,
        "mid": float(mid) if math.isfinite(mid) else None,
        "mark": float(mark) if math.isfinite(mark) else None,
        "last": float(last) if math.isfinite(last) else None,
        "spread": float(spread) if math.isfinite(spread) else None,
        "spread_pct_mid": float(spread_pct_mid) if math.isfinite(spread_pct_mid) else None,
        "quote_timestamp": (
            quote.get("timestamp")
            or quote.get("t")
            or quote.get("updated_at")
        ),
    }


def _round_option_limit(value: float) -> float:
    return max(_OPTION_TICK, round(float(value), 2))


def _is_option_tick(value: float) -> bool:
    try:
        return math.isclose(float(value), _OPTION_TICK, rel_tol=0.0, abs_tol=1e-9)
    except Exception:
        return False


def _close_limit_ladder(
    *,
    base_limit: float,
    close_bid: float,
    reason: str,
    attempts: int,
) -> list[float]:
    attempts = max(1, int(attempts))
    base = _round_option_limit(base_limit if math.isfinite(base_limit) and base_limit > 0 else _OPTION_TICK)
    bid = close_bid if math.isfinite(close_bid) and close_bid > 0.0 else float("nan")

    prices: list[float] = []
    if str(reason or "").strip().lower() in _LIQUIDATION_CLOSE_REASONS:
        anchor = bid if math.isfinite(bid) and bid > 0.0 else base
        if anchor <= 0.25:
            raw_prices = [
                base,
                anchor,
                anchor * 0.75,
                anchor * 0.50,
                anchor * 0.25,
                0.02,
                _OPTION_TICK,
            ]
        else:
            raw_prices = [
                base,
                anchor - 0.02,
                anchor - 0.05,
                anchor - 0.10,
                anchor - 0.20,
            ]
        for price in raw_prices:
            rounded = _round_option_limit(price)
            if rounded not in prices:
                prices.append(rounded)
            if len(prices) >= attempts:
                break
        if anchor <= 0.25 and prices[-1] != _OPTION_TICK and len(prices) < attempts:
            prices.append(_OPTION_TICK)
        return prices

    for attempt in range(1, attempts + 1):
        if math.isfinite(bid) and bid > 0.0:
            price = base if attempt == 1 else bid - (attempt - 1) * _OPTION_TICK
        else:
            price = base - (attempt - 1) * _OPTION_TICK * 2.0
        rounded = _round_option_limit(price)
        if rounded not in prices:
            prices.append(rounded)
    return prices or [_OPTION_TICK]


def _parse_occ_option_symbol(symbol: str) -> ParsedOptionSymbol | None:
    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", str(symbol).strip().upper())
    if not m:
        return None
    yy, mm, dd = int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:6])
    try:
        expiration = date(2000 + yy, mm, dd)
    except ValueError:
        return None
    return ParsedOptionSymbol(
        root=m.group(1),
        expiration=expiration,
        call_put=m.group(3),
        strike=int(m.group(4)) / 1000.0,
    )


def _parse_swing_option_symbol(
    symbol: str,
    universe: dict[str, TickerConfig],
) -> tuple[str, int] | None:
    parsed = _parse_occ_option_symbol(symbol)
    if parsed is None:
        return None
    if parsed.root not in universe:
        return None
    return parsed.root, (1 if parsed.call_put == "C" else -1)
