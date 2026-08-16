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
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.calendar import is_market_open_now, next_trading_day
from strategies.multi_ticker_swing.live.universe import TickerConfig

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
_PENDING_CLOSE_CHASE_SECS = 60.0
# When the limit ladder can't get a verified fill, fall back to a MARKET sell so
# the exit actually happens instead of lingering as an unfilled limit. Gated to
# RTH (options market orders need a live market) and skipped for worthless
# one-cent abandons. Env override: SWING_CLOSE_MARKET_FALLBACK=0 to disable.
_CLOSE_MARKET_FALLBACK = os.getenv("SWING_CLOSE_MARKET_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}
_LIQUIDATION_CLOSE_REASONS = {
    "sl",
    "no_progress",
    "deferred_trail_failed",
    "deferred_trail_timeout",
    "expiration_itm_cutoff",
    "expiring_before_closure",
    "restored_unknown_expiring",
    "restored_unknown_loss_cut",
}
_ET = ZoneInfo("America/New_York")
_DEFER_TRAIL_AFTER_HOUR = 15


def _market_is_open(now: datetime | None = None) -> bool:
    """True during US equity/options RTH. Delegates to the shared trading-calendar
    helper (single source of truth — also used by options_exec.equity_order_tif)
    so both correctly treat market holidays as closed, not just weekends."""
    return is_market_open_now(now)
_DEFER_TRAIL_AFTER_MINUTE = 55
_DEFER_RECOVERY_BARS = 3
_DEFER_RECOVERY_PCT = 0.0025
_DEFERRED_TRAIL_STATE_PATH = Path("Data/inference/multi_ticker_swing/deferred_trails.json")
_WORTHLESS_CLOSE_STATE_PATH = Path("Data/inference/multi_ticker_swing/worthless_close_abandoned.json")
_OPEN_POSITION_STATE_PATH = Path("Data/inference/multi_ticker_swing/open_positions.json")
_EXPIRING_ITM_CLOSE_HOUR = 15
_EXPIRING_ITM_CLOSE_MINUTE = 45
_ASSIGNED_EQUITY_MIN_SHARES = 100
_ASSIGNED_EQUITY_FLATTEN_COOLDOWN_S = 180.0  # min seconds between flatten attempts per symbol
_OPTION_VALUE_EXIT_ENABLED = True
_OPTION_VALUE_QUOTE_MODE = "bid"
_OPTION_PROFIT_TRAIL_ARM_PCT = 1.00
_OPTION_PROFIT_TRAIL_GIVEBACK_PCT = 0.25
_OPTION_TAKE_PROFIT_PCT = 3.00
_RESTORED_UNKNOWN_MAX_LOSS_PCT = -0.35

# Exits whose entire purpose is to LOCK IN PROFIT. They are discretionary: the
# position is not in trouble, we are just choosing to stop riding it. Everything
# not listed here (sl, expiry, restored-unknown, worthless, option-value exits)
# is mandatory and must never be delayed or vetoed by the two guards below.
_DISCRETIONARY_PROFIT_EXITS = frozenset({
    "trail", "deferred_trail_failed", "deferred_trail_timeout", "no_progress",
})

# We hold an OPTION but the trail is measured on the UNDERLYING, and the two can
# disagree completely. CALM on 2026-08-03: the underlying trail fired with the
# short still +3.56% in our favour, but the put had decayed from 4.54 to 2.20, so
# a "profit-protecting" exit realized -6.38% (-$345) and gave back $3,335 of
# unrealized gain. When the leg we actually own is not in profit, a profit-taking
# trail has nothing to protect — hold and let the hard stop or the option-value
# exit make the risk decision instead.
_TRAIL_REQUIRES_OPTION_PROFIT = True
_TRAIL_OPTION_PROFIT_FLOOR_PCT = 0.0

# CALM's exit also crossed a bid 2.23 / ask 4.12 quote — a spread 59.5% of mid.
# Entries have had a spread gate since inception (_MAX_ENTRY_SPREAD_PCT_MID in
# runner.py); exits had none, so a discretionary exit would pay any spread the
# book quoted. Deferral is bounded: after _MAX_EXIT_SPREAD_DEFERRALS bars the
# exit goes through regardless, because a permanently wide book must never trap
# a position.
_MAX_EXIT_SPREAD_PCT_MID = 0.35
_MAX_EXIT_SPREAD_DEFERRALS = 12  # 5m bars ≈ 1 hour

# The broker account is shared across every live module. Each one persists its
# OWN managed-position state precisely so a sibling never has to guess at
# another module's book -- so broker reconciliation must exclude anything a
# sibling currently claims before adopting an untracked position or flattening
# an "assigned equity" lot. Without this, Swing's reconcile can silently take
# over (or auto-sell) a position another module is actively managing: on
# 2026-07-21 Swing force-sold HTF Swing's legitimate 100-share FIG position as
# a false-positive option-exercise assignment, because Swing's universe
# happened to also include FIG and nothing checked HTF's own managed state.
_SIBLING_MODULE_STATE_PATHS = (
    Path("strategies/momentum_expansion/live/momentum_live_state.json"),
    Path("strategies/multi_ticker_swing_htf/live/htf_live_state.json"),
    Path("signals/meta_context/meta_ranker/live_state.json"),
    Path("Data/inference/dealer_ranker/live_state.json"),
)


def _sibling_module_owned_symbols() -> set[str]:
    """Equity tickers and option OCC symbols the 4H-family modules and Dealer
    Ranker currently claim in their own persisted ``managed`` state. Best-effort:
    a missing/unreadable state file just contributes nothing, it never blocks
    Swing's own reconciliation."""
    owned: set[str] = set()
    for path in _SIBLING_MODULE_STATE_PATHS:
        try:
            state = json.loads(path.read_text())
        except Exception:
            continue
        for entry in (state.get("managed") or {}).values():
            if not isinstance(entry, dict):
                continue
            symbol = entry.get("symbol")
            if symbol:
                owned.add(str(symbol).strip().upper())
            occ = entry.get("occ")
            if occ:
                owned.add(str(occ).strip().upper())
    return owned


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
    restored_from_broker: bool = False
    restore_source: str | None = None

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

    @property
    def entry_price_is_synthetic(self) -> bool:
        """True when `entry_price` is a restore-time mark, not a real entry.

        A position rebuilt from a broker snapshot with no local state has no
        recoverable underlying entry price, so `_restore_from_broker` stamps the
        price at restore time. Any P&L derived from it measures "move since we
        noticed the position", not "move since entry".
        """
        return bool(self.restored_from_broker) and str(self.restore_source or "") == "broker_snapshot"

    def to_dict(self) -> dict:
        # Reporting a P&L off a synthetic basis fabricates a number: on
        # 2026-08-07 UMC and TKR were both cut at an option-leg -45% while this
        # field read exactly 0.00%, because entry_price had been stamped at the
        # restore-time mark on the same bar. Unknown is reported as unknown; the
        # broker's own unrealized figure is carried alongside as the real one.
        synthetic = self.entry_price_is_synthetic
        pnl_pct = (
            None if synthetic or not self.entry_price
            else self.direction * (self.last_price - self.entry_price) / self.entry_price
        )
        meta = self.option_entry_meta if isinstance(self.option_entry_meta, dict) else {}
        broker_plpc = _finite_or_none(meta.get("broker_unrealized_plpc"))
        return {
            "ticker": self.ticker,
            "direction": int(self.direction),
            "entry_price": float(self.entry_price),
            "entry_price_is_synthetic": synthetic,
            "entry_time": self.entry_time.astimezone(timezone.utc).isoformat() if self.entry_time else None,
            "last_price": float(self.last_price),
            "best_price": float(self.best_price),
            "pnl_pct": None if pnl_pct is None else float(pnl_pct),
            "pnl_pct_source": "restore_time_mark_unusable" if synthetic else "entry_price",
            "broker_unrealized_plpc": broker_plpc,
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
            "restored_from_broker": bool(self.restored_from_broker),
            "restore_source": self.restore_source,
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


def _expiring_before_closure_exit_reason(pos: SwingPosition, bar: dict) -> str | None:
    """Force-close any option that won't see another tradable session.

    Past the EOD cutoff, if the next market session opens *after* the contract's
    expiration (0DTE, or the last session before a weekend/holiday closure), the
    option can't be exited on-screen again — so flatten it now instead of letting
    it expire/auto-exercise into assigned equity over the closure. This is the
    holiday-aware backstop the 2026-06-18 Juneteenth weekend exposed.
    """
    parsed = _parse_occ_option_symbol(pos.option_symbol)
    if parsed is None:
        return None
    ts = bar.get("timestamp")
    if not isinstance(ts, datetime):
        return None
    local = ts.astimezone(_ET) if ts.tzinfo else ts.replace(tzinfo=_ET)
    if not _past_expiring_itm_close_cutoff(ts):
        return None
    # next_trading_day(today) > expiry  ==>  today is the last tradable session.
    if next_trading_day(local.date()) > parsed.expiration:
        return "expiring_before_closure"
    return None


def _restored_unknown_exit_reason(pos: SwingPosition, bar: dict) -> str | None:
    if not bool(pos.restored_from_broker) or str(pos.restore_source or "") != "broker_snapshot":
        return None

    parsed = _parse_occ_option_symbol(pos.option_symbol)
    ts = bar.get("timestamp")
    if parsed is not None and isinstance(ts, datetime):
        local = ts.astimezone(_ET) if ts.tzinfo else ts.replace(tzinfo=_ET)
        if parsed.expiration <= local.date():
            return "restored_unknown_expiring"

    meta = pos.option_entry_meta if isinstance(pos.option_entry_meta, dict) else {}
    broker_plpc = _as_float(meta.get("broker_unrealized_plpc"))
    if math.isfinite(broker_plpc) and broker_plpc <= _RESTORED_UNKNOWN_MAX_LOSS_PCT:
        return "restored_unknown_loss_cut"
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
        self._close_pass_count: dict[str, int] = {}  # ticker → liquidation ladder passes so far
        self._pending_close_orders: dict[str, dict[str, Any]] = {}
        self._assigned_flatten_last_attempt: dict[str, float] = {}  # symbol → wall time
        self._exit_spread_deferrals: dict[str, int] = {}  # ticker → consecutive wide-spread skips
        self._deferred_trail_cache = self._load_deferred_trail_cache()
        self._worthless_close_abandoned = self._load_worthless_close_abandoned()
        self._position_state_cache = self._load_open_position_state_cache()

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
        sibling_owned = _sibling_module_owned_symbols()

        for broker_pos in broker_positions:
            ticker = broker_pos["ticker"]
            symbol = broker_pos["option_symbol"]
            if ticker in self._positions:
                ignored.append({"symbol": symbol, "ticker": ticker, "reason": "already_tracked"})
                continue
            if symbol in sibling_owned or ticker in sibling_owned:
                ignored.append({"symbol": symbol, "ticker": ticker, "reason": "owned_by_other_module"})
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
        # Capture swing's own option-ownership scope BEFORE any reconcile
        # mutation: tickers we currently track plus those persisted from a
        # prior session (an assignment can land while we were offline). This is
        # what keeps the flattener from selling another module's equity.
        owned_tickers = set(self._positions.keys()) | set(self._position_state_cache.keys())
        sibling_owned = _sibling_module_owned_symbols()
        assigned_equities = self._broker_assigned_equity_positions(universe, owned_tickers, sibling_owned)
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
            broker_symbol = str(broker_pos.get("option_symbol", "")).upper()
            if broker_symbol in sibling_owned or ticker in sibling_owned:
                ignored.append({"symbol": broker_symbol, "ticker": ticker, "reason": "owned_by_other_module"})
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
        if removed or replaced or restored or qty_updates:
            self._persist_open_position_state()
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
        owned_tickers: set[str],
        sibling_owned: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect possible exercised/assigned option lots now held as shares.

        Alpaca exposes exercised/assigned options as plain equity positions. The
        account is shared with other strategies (Meta Ranker, Momentum), whose
        deliberate equity lots also live in the swing universe — so universe
        membership alone is NOT enough to claim a lot. We only flag a lot when
        the swing book actually has option-ownership history for that ticker
        (currently tracked, or persisted before a restart). Anything else is
        another module's position and must be left untouched. ``sibling_owned``
        is an absolute veto on top of that: even if Swing's own history looks
        like a match, a symbol a sibling module's OWN managed state currently
        claims is never flattened (2026-07-21: this is exactly how a legitimate
        HTF Swing equity position got force-sold as a false-positive
        assignment).
        """
        sibling_owned = sibling_owned or set()
        resp = self._client.get_positions()
        raw_positions = _extract_positions(resp)
        positions: list[dict[str, Any]] = []

        for raw in raw_positions:
            symbol = str(raw.get("symbol", "")).strip().upper()
            if symbol not in universe:
                continue
            if symbol in sibling_owned:
                continue
            if symbol not in owned_tickers:
                # Equity lot owned by another module (or never an assignment of
                # ours) — out of this module's scope; never flatten it.
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

        now = time.time()
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

            # Don't resubmit the same flatten every reconcile (which spams broker
            # 403s while the prior order is still pending / settling). A market
            # order placed before the open is also rejected, so a cooldown lets
            # the next attempt land after the bell instead of every bar.
            last = self._assigned_flatten_last_attempt.get(symbol)
            if last is not None and (now - last) < _ASSIGNED_EQUITY_FLATTEN_COOLDOWN_S:
                result = {**pos, "submitted": False, "skipped": "flatten_cooldown"}
                results.append(result)
                self._emit("assigned_equity_flatten_skipped", result)
                continue
            self._assigned_flatten_last_attempt[symbol] = now

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

        latest_price = price_lookup(ticker)
        atr = atr_lookup(ticker)
        if latest_price is None or not math.isfinite(float(latest_price)):
            ignored.append({"symbol": symbol, "ticker": ticker, "reason": "missing_underlying_price"})
            return None
        if atr is None or not math.isfinite(float(atr)) or float(atr) <= 0:
            ignored.append({"symbol": symbol, "ticker": ticker, "reason": "missing_atr"})
            return None

        cached = self._cached_position_state(ticker=ticker, symbol=symbol)
        restore_source = "local_state" if cached else "broker_snapshot"
        entry_price = _finite_or_none(cached.get("entry_price")) if cached else None
        if entry_price is None:
            entry_price = float(latest_price)
        cached_atr = _finite_or_none(cached.get("atr_at_entry")) if cached else None
        atr_at_entry = float(cached_atr if cached_atr is not None and cached_atr > 0 else atr)
        cached_entry_time = _parse_dt(cached.get("entry_time")) if cached else None
        entry_time = cached_entry_time or datetime.now(timezone.utc)

        broker_avg = _finite_or_none(broker_pos.get("avg_entry_price"))
        cached_option_entry = _finite_or_none(cached.get("option_entry_price")) if cached else None
        option_entry_price = broker_avg if broker_avg is not None and broker_avg > 0 else cached_option_entry
        option_entry_meta = (
            cached.get("option_entry_meta")
            if cached and isinstance(cached.get("option_entry_meta"), dict)
            else None
        )
        if option_entry_meta is None:
            option_entry_meta = {}
        option_entry_meta = {
            **option_entry_meta,
            "restored_from_broker": True,
            "restore_source": restore_source,
            "missing_local_state": not bool(cached),
            "broker_unrealized_plpc": broker_pos.get("unrealized_plpc"),
            "broker_unrealized_pl": broker_pos.get("unrealized_pl"),
            "broker_market_value": broker_pos.get("market_value"),
        }

        pos = SwingPosition(
            ticker=ticker,
            direction=int(broker_pos.get("direction", 1) or 1),
            entry_price=float(entry_price),
            entry_time=entry_time,
            atr_at_entry=atr_at_entry,
            option_symbol=symbol,
            qty=int(broker_pos.get("qty", 0) or 0),
            config=universe[ticker],
            option_entry_price=option_entry_price,
            option_entry_meta=option_entry_meta,
            restored_from_broker=True,
            restore_source=restore_source,
        )
        if cached:
            self._restore_cached_tracking_state(pos, cached, latest_price=float(latest_price))
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
        self._persist_open_position_state()

    def on_5m_bar(self, ticker: str, bar: dict) -> None:
        """
        Called from the 5m bar stream for every bar on every ticker.
        If ticker has an open position, checks exit conditions and closes if triggered.
        """
        pos = self._positions.get(ticker)
        if pos is None:
            return

        pending = self._pending_close_orders.get(pos.ticker)
        if pending is not None and not self._dry_run:
            self._close_position(pos, str(pending.get("reason") or "pending_close_reconcile"), bar)
            return

        restored_reason = _restored_unknown_exit_reason(pos, bar)
        if restored_reason:
            self._emit("restored_position_defensive_exit_triggered", {
                **pos.to_dict(),
                "reason": restored_reason,
                "bar": _safe_bar(bar),
            })
            self._close_position(pos, restored_reason, bar)
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

        closure_reason = _expiring_before_closure_exit_reason(pos, bar)
        if closure_reason:
            self._emit("expiring_before_closure_exit_triggered", {
                **pos.to_dict(),
                "reason": closure_reason,
                "bar": _safe_bar(bar),
                "cutoff_et": f"{_EXPIRING_ITM_CLOSE_HOUR:02d}:{_EXPIRING_ITM_CLOSE_MINUTE:02d}",
            })
            self._close_position(pos, closure_reason, bar)
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
                if self._gate_discretionary_exit(pos, reason):
                    self._persist_open_position_state()
                    return
                self._close_position(pos, reason, bar)
                return

            was_deferred = pos.deferred_trail_active
            deferred_reason = pos.deferred_trail_decision(bar)
            if deferred_reason:
                if self._gate_discretionary_exit(pos, deferred_reason):
                    self._persist_open_position_state()
                    return
                self._close_position(pos, deferred_reason, bar)
                return
            if was_deferred and not pos.deferred_trail_active:
                self._emit("position_trail_resumed", {
                    **pos.to_dict(),
                    "reason": "deferred_trail_recovered",
                    "bar": _safe_bar(bar),
                })
                self._remove_deferred_trail_cache(pos)
            self._persist_open_position_state()
            return

        reason = pos.update(bar)
        if reason:
            if self._gate_discretionary_exit(pos, reason):
                self._persist_open_position_state()
                return
            if reason == "trail" and _should_defer_trail_exit(bar):
                pos.mark_deferred_trail(bar)
                self._persist_deferred_trail_cache()
                self._persist_open_position_state()
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
            return
        self._persist_open_position_state()

    @staticmethod
    def option_leg_gain_pct(pos: SwingPosition) -> float | None:
        """Current gain on the OPTION we actually hold, or None if unpriced.

        `option_last_price` is refreshed every bar by _option_value_exit_reason ->
        update_option_value, which runs before the underlying exit checks.
        """
        entry = _as_float(pos.option_entry_price)
        last = _as_float(pos.option_last_price)
        if not math.isfinite(entry) or entry <= 0.0:
            return None
        if not math.isfinite(last) or last <= 0.0:
            return None
        return (last - entry) / entry

    def _veto_profit_exit_on_losing_option(self, pos: SwingPosition, reason: str) -> bool:
        """True when a profit-protecting exit should be held back.

        Only applies to _DISCRETIONARY_PROFIT_EXITS. An unpriced option leg never
        vetoes (we cannot prove the exit is wrong, so the exit wins).
        """
        if not _TRAIL_REQUIRES_OPTION_PROFIT or reason not in _DISCRETIONARY_PROFIT_EXITS:
            return False
        gain = self.option_leg_gain_pct(pos)
        if gain is None or gain > _TRAIL_OPTION_PROFIT_FLOOR_PCT:
            return False
        logger.info(
            "[%s] HOLD %s: underlying trail fired but the option leg is %+.2f%% "
            "(entry=%.2f last=%.2f) — no profit to protect",
            pos.ticker, reason, gain * 100.0,
            _as_float(pos.option_entry_price), _as_float(pos.option_last_price),
        )
        self._emit("profit_exit_vetoed_option_at_loss", {
            **pos.to_dict(),
            "reason": reason,
            "option_gain_pct": float(gain),
            "floor_pct": _TRAIL_OPTION_PROFIT_FLOOR_PCT,
        })
        return True

    def _gate_discretionary_exit(self, pos: SwingPosition, reason: str) -> bool:
        """True when a profit-protecting exit must not be submitted on this bar.

        Single choke point for both guards so every discretionary-exit path gets
        the same treatment. Mandatory exits fall straight through.
        """
        if reason not in _DISCRETIONARY_PROFIT_EXITS:
            self._exit_spread_deferrals.pop(pos.ticker, None)
            return False
        if self._veto_profit_exit_on_losing_option(pos, reason):
            return True
        return self._defer_exit_on_wide_spread(pos, reason)

    def _defer_exit_on_wide_spread(self, pos: SwingPosition, reason: str) -> bool:
        """True when a discretionary exit should wait for a tighter book.

        Bounded by _MAX_EXIT_SPREAD_DEFERRALS so a permanently wide quote cannot
        strand the position.
        """
        if reason not in _DISCRETIONARY_PROFIT_EXITS:
            self._exit_spread_deferrals.pop(pos.ticker, None)
            return False
        symbol = str(pos.option_symbol).strip().upper()
        if not symbol:
            return False
        quote_meta = self._get_contract_quote_context(symbol=symbol)
        spread_pct = _as_float((quote_meta or {}).get("spread_pct_mid"))
        if not math.isfinite(spread_pct) or spread_pct <= _MAX_EXIT_SPREAD_PCT_MID:
            self._exit_spread_deferrals.pop(pos.ticker, None)
            return False
        count = self._exit_spread_deferrals.get(pos.ticker, 0) + 1
        if count > _MAX_EXIT_SPREAD_DEFERRALS:
            logger.info(
                "[%s] exit spread gate exhausted after %d bars (spread=%.1f%% of mid) "
                "— submitting %s anyway", pos.ticker, count - 1, spread_pct * 100.0, reason,
            )
            self._exit_spread_deferrals.pop(pos.ticker, None)
            self._emit("exit_spread_gate_exhausted", {
                **pos.to_dict(), "reason": reason,
                "spread_pct_mid": spread_pct, "max_spread_pct_mid": _MAX_EXIT_SPREAD_PCT_MID,
                "deferrals": count - 1,
            })
            return False
        self._exit_spread_deferrals[pos.ticker] = count
        logger.info(
            "[%s] DEFER %s: option spread %.1f%% of mid > %.1f%% cap (defer %d/%d)",
            pos.ticker, reason, spread_pct * 100.0, _MAX_EXIT_SPREAD_PCT_MID * 100.0,
            count, _MAX_EXIT_SPREAD_DEFERRALS,
        )
        self._emit("exit_deferred_wide_spread", {
            **pos.to_dict(), "reason": reason,
            "spread_pct_mid": spread_pct, "max_spread_pct_mid": _MAX_EXIT_SPREAD_PCT_MID,
            "close_quote": quote_meta, "deferrals": count,
        })
        return True

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
                    fill=self._resolve_close_fill(state),
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
            if state.get("retry_now"):
                self._last_close_failure_wall.pop(pos.ticker, None)
                self._emit("position_close_retry", {
                    **pos.to_dict(),
                    "exit_price": exit_price,
                    "exit_pnl_pct": float(pnl_pct),
                    "exit_reason": reason,
                    "retry_reason": state.get("reason"),
                    "verification": state.get("verification"),
                    "order_response": _safe_response(state.get("order")),
                })
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
        fill: dict[str, Any] | None = None
        if not self._dry_run:
            try:
                close_result = self._submit_close_order(pos, reason=reason)
                order_resp = close_result.get("response")
                fill = self._resolve_close_fill(close_result)
                close_result.update(fill)
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
                    **fill,
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
                            "limit_price": close_result.get("limit_price"),
                            "close_quote": close_result.get("close_quote"),
                        }
                    self._emit("position_close_pending", {
                        **pos.to_dict(),
                        "exit_price": exit_price,
                        "exit_pnl_pct": float(pnl_pct),
                        "exit_reason": reason,
                        "limit_price": close_result.get("limit_price"),
                        "close_quote": close_result.get("close_quote"),
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
            fill=fill,
        )

    def _emit_position_closed(
        self,
        pos: SwingPosition,
        *,
        reason: str,
        exit_price: float,
        pnl_pct: float,
        order_error: str | None,
        fill: dict[str, Any] | None = None,
    ) -> None:
        fill = fill or {}
        fill_price = _positive_or_none(fill.get("fill_price"))
        entry_premium = _positive_or_none(pos.option_entry_price)
        payload = {
            **pos.to_dict(),
            "exit_price": exit_price,
            "exit_pnl_pct": float(pnl_pct),
            "exit_reason": reason,
            "order_error": order_error,
            # Realized option economics from the actual fill. `exit_pnl_pct`
            # above is the UNDERLYING move, which is not the trade's P&L.
            "option_exit_price": fill_price,
            "option_exit_filled_qty": fill.get("filled_qty"),
            "option_exit_fill_source": fill.get("fill_source", "unavailable"),
        }
        if fill_price is not None and entry_premium is not None:
            payload["option_realized_pnl"] = round(
                (fill_price - entry_premium) * int(pos.qty) * 100.0, 2)
            payload["option_realized_pct"] = round(fill_price / entry_premium - 1.0, 6)
        else:
            payload["option_realized_pnl"] = None
            payload["option_realized_pct"] = None
        self._emit("position_closed", payload)
        self._last_close_failure_wall.pop(pos.ticker, None)
        self._close_pass_count.pop(pos.ticker, None)
        self._pending_close_orders.pop(pos.ticker, None)
        self._exit_spread_deferrals.pop(pos.ticker, None)
        self._remove_deferred_trail_cache(pos)
        del self._positions[pos.ticker]
        self._persist_open_position_state()

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
        self._close_pass_count.pop(pos.ticker, None)
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
        self._exit_spread_deferrals.pop(pos.ticker, None)
        self._positions.pop(pos.ticker, None)
        self._persist_open_position_state()

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

    def _load_open_position_state_cache(self) -> dict[str, dict[str, Any]]:
        try:
            if not _OPEN_POSITION_STATE_PATH.exists():
                return {}
            raw = json.loads(_OPEN_POSITION_STATE_PATH.read_text())
            rows = raw.get("positions") if isinstance(raw, dict) else raw
            if not isinstance(rows, list):
                return {}
            out: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ticker = str(row.get("ticker", "")).strip().upper()
                symbol = str(row.get("option_symbol", "")).strip().upper()
                if ticker and symbol:
                    out[ticker] = row
            return out
        except Exception as exc:
            logger.warning("open position state load failed: %s", exc)
        return {}

    def _persist_open_position_state(self) -> None:
        try:
            _OPEN_POSITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            positions = [pos.to_dict() for pos in self._positions.values()]
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "positions": positions,
            }
            _OPEN_POSITION_STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
            self._position_state_cache = {
                str(pos.get("ticker", "")).strip().upper(): pos
                for pos in positions
                if str(pos.get("ticker", "")).strip()
            }
        except Exception as exc:
            logger.warning("open position state persist failed: %s", exc)

    def _cached_position_state(self, *, ticker: str, symbol: str) -> dict[str, Any] | None:
        cached = self._position_state_cache.get(str(ticker).strip().upper())
        if not isinstance(cached, dict):
            return None
        cached_symbol = str(cached.get("option_symbol", "")).strip().upper()
        if cached_symbol != str(symbol).strip().upper():
            return None
        return cached

    def _restore_cached_tracking_state(
        self,
        pos: SwingPosition,
        cached: dict[str, Any],
        *,
        latest_price: float,
    ) -> None:
        cached_sl = _finite_or_none(cached.get("sl_price"))
        if cached_sl is not None:
            pos.sl_price = float(cached_sl)

        last_price = latest_price if math.isfinite(latest_price) else _as_float(cached.get("last_price"))
        if math.isfinite(last_price) and last_price > 0.0:
            pos.last_price = float(last_price)

        cached_best = _as_float(cached.get("best_price"))
        if math.isfinite(cached_best) and cached_best > 0.0:
            if pos.direction == 1:
                pos.best_price = max(float(cached_best), float(pos.entry_price), float(pos.last_price))
            else:
                pos.best_price = min(float(cached_best), float(pos.entry_price), float(pos.last_price))

        pos.trail_armed = bool(cached.get("trail_armed"))
        try:
            pos.bar_count_5m = max(0, int(cached.get("bars_held") or 0))
        except Exception:
            pos.bar_count_5m = 0

        option_last = _finite_or_none(cached.get("option_last_price"))
        option_best = _finite_or_none(cached.get("option_best_price"))
        if option_last is not None and option_last > 0:
            pos.option_last_price = float(option_last)
        if option_best is not None and option_best > 0:
            entry = _as_float(pos.option_entry_price)
            if math.isfinite(entry) and entry > 0:
                pos.option_best_price = max(float(option_best), float(entry), float(pos.option_last_price or entry))
            else:
                pos.option_best_price = float(option_best)
        pos.option_trail_armed = bool(cached.get("option_trail_armed"))

    def _submit_close_order(self, pos: SwingPosition, *, reason: str) -> dict[str, Any]:
        symbol = str(pos.option_symbol).strip().upper()
        qty = int(pos.qty)
        quote_meta = self._get_contract_quote_context(symbol=symbol)
        close_bid = self._get_contract_price(symbol=symbol, mode="bid")
        # Anchor on the BID for ordinary exits (trail/take-profit) — resting at the
        # bid is how they fill — but for a forced liquidation anchor on the MID and
        # walk down across passes. Anchoring a liquidation on the bid is what sold
        # 159 VALE260828C00015000 at $0.01 on 2026-08-05 into a 0.01 x 0.74 market:
        # the bid ticking up from 0.00 to 0.01 became the FIRST rung, so a position
        # the broker marked at $3,021 realized $159. A penny bid is a reason to be
        # patient, not a price to hit.
        liquidating = str(reason or "").strip().lower() in _LIQUIDATION_CLOSE_REASONS
        base_limit = float("nan")
        if liquidating:
            for mode in ("mid", "mark"):
                base_limit = self._get_contract_price(symbol=symbol, mode=mode)
                if math.isfinite(base_limit) and base_limit > 0.0:
                    source = mode
                    break
            else:
                source = "fallback"
        else:
            base_limit, source = close_bid, "bid"
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            for mode in ("bid", "mid", "mark"):
                base_limit = self._get_contract_price(symbol=symbol, mode=mode)
                if math.isfinite(base_limit) and base_limit > 0.0:
                    source = mode
                    break
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            base_limit, source = _OPTION_TICK, "fallback"

        # Passes are counted per ticker and cleared only when the position actually
        # leaves the book, so the floor relaxes over successive 60s retries rather
        # than inside one 15-second burst of rungs.
        close_pass = self._close_pass_count.get(pos.ticker, 0) + 1
        self._close_pass_count[pos.ticker] = close_pass

        logger.info(
            "[%s] close order pricing symbol=%s source=%s base_limit=%.2f pass=%d reason=%s",
            pos.ticker,
            symbol,
            source,
            base_limit,
            close_pass,
            reason,
        )

        limit_prices = _close_limit_ladder(
            base_limit=base_limit,
            close_bid=close_bid,
            quote_meta=quote_meta,
            reason=reason,
            attempts=_CLOSE_ORDER_ATTEMPTS,
            close_pass=close_pass,
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
                    position_intent="sell_to_close",
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
            # Limit ladder is out of retries for this attempt — stop chasing and
            # let the market fallback below guarantee the exit.
            break

        # Limit ladder exhausted without a verified fill. Fall back to a MARKET
        # sell (RTH only) so a genuine exit actually closes instead of lingering
        # as an unfilled limit. Skip worthless one-cent abandons.
        if (
            _CLOSE_MARKET_FALLBACK
            and not self._dry_run
            and last_result is not None
            and not last_result.get("abandoned_worthless")
            and not last_result.get("verified")
            and _market_is_open()
        ):
            market_result = self._submit_market_close(
                pos,
                symbol=symbol,
                prior_verify=last_result.get("verification") or {},
                quote_meta=quote_meta,
                reason=reason,
            )
            if market_result is not None:
                return market_result

        if last_result is not None:
            # Keep the last live close order working. The manager tracks it in
            # _pending_close_orders and reconciles it before any later retry.
            return last_result
        raise RuntimeError(f"close_order_submit_failed symbol={symbol}")

    def _submit_market_close(
        self,
        pos: SwingPosition,
        *,
        symbol: str,
        prior_verify: dict[str, Any],
        quote_meta: dict[str, Any],
        reason: str,
    ) -> dict[str, Any] | None:
        """Guarantee an exit by market-selling the currently-held qty.

        Cancels any still-working limit order first (avoid double-sell) and waits
        for that cancel to resolve to a terminal state before re-reading the
        position: cancel_order is async, so reading the position immediately
        after firing it can race a limit that fills anyway, causing the market
        sell below to close more than is actually held. Once the cancel result
        is known, re-reads the live position, then sells the exact remaining
        long qty at market. Returns a close-result dict, or None if there is
        nothing to do / the market submit failed.
        """
        # Cancel the still-working limit and block until it resolves (filled or
        # canceled) so the position read below reflects the true outcome.
        cancel_result = self._cancel_order_if_needed(prior_verify)
        if cancel_result is not None and cancel_result.get("status") == "filled":
            logger.info(
                "[%s] market close skipped — canceled limit filled first symbol=%s order_id=%s",
                pos.ticker, symbol, cancel_result.get("order_id"),
            )
            return {
                "response": cancel_result.get("order") or {},
                "verification": {
                    "verified": True,
                    "status": "filled",
                    "order_id": str(cancel_result.get("order_id", "")),
                    "via": "cancel_race_fill",
                    "order": cancel_result.get("order") or {},
                },
                "limit_price": None,
                "close_quote": quote_meta,
                "verified": True,
            }
        try:
            held_qty = self._open_long_qty(symbol=symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] market close: position re-check failed symbol=%s: %s", pos.ticker, symbol, exc)
            held_qty = int(pos.qty)
        if held_qty <= 0:
            # The limit already closed the position (verify just timed out).
            logger.info("[%s] market close skipped — position already flat symbol=%s", pos.ticker, symbol)
            return {
                "response": prior_verify.get("order") or {},
                "verification": {
                    "verified": True,
                    "status": "closed",
                    "order_id": str(prior_verify.get("order_id", "")),
                    "via": "positions_reconcile_pre_market",
                },
                "limit_price": None,
                "close_quote": quote_meta,
                "verified": True,
            }
        try:
            resp = self._client.submit_option_order(
                symbol=symbol,
                qty=held_qty,
                side="sell",
                order_type="market",
                time_in_force="day",
                position_intent="sell_to_close",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] market close fallback FAILED symbol=%s error=%s", pos.ticker, symbol, exc)
            self._emit("close_market_fallback_failed", {
                "ticker": pos.ticker, "symbol": symbol, "qty": held_qty,
                "reason": reason, "error": str(exc),
            })
            return None
        verify = self._verify_close_order(submitted_resp=resp if isinstance(resp, dict) else {}, symbol=symbol)
        logger.info(
            "[%s] market close fallback submitted symbol=%s order_id=%s status=%s verified=%s qty=%d reason=%s",
            pos.ticker, symbol,
            str(resp.get("id", "n/a")) if isinstance(resp, dict) else "n/a",
            verify.get("status"), verify.get("verified"), held_qty, reason,
        )
        self._emit("close_market_fallback", {
            "ticker": pos.ticker, "symbol": symbol, "qty": held_qty, "reason": reason,
            "status": verify.get("status"), "verified": bool(verify.get("verified")),
        })
        return {
            "response": resp,
            "verification": verify,
            "limit_price": None,
            "close_quote": quote_meta,
            "verified": bool(verify.get("verified")),
            "via_market_fallback": True,
        }

    def _open_long_qty(self, *, symbol: str) -> int:
        """Current long contract qty held for `symbol` (0 if flat/short)."""
        target = str(symbol).strip().upper()
        resp = self._client.get_positions()
        for raw in _extract_positions(resp):
            if str(raw.get("symbol", "")).strip().upper() != target:
                continue
            if str(raw.get("side", "")).strip().lower() == "short":
                return 0
            return int(abs(_as_float(raw.get("qty")) or 0.0))
        return 0

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

    def _resolve_close_fill(self, close_result: dict[str, Any]) -> dict[str, Any]:
        """Resolve the actual exit fill for a close order.

        The submit response is captured at ``pending_new`` with a null
        ``filled_avg_price``, so the audit previously recorded no exit price at
        all and realized P&L had to be proxied from the submitted limit. The
        verification poll already holds the terminal order; when it does not
        (verified via submit_response or positions_reconcile) one extra
        ``get_order`` call fetches it.
        """
        verification = close_result.get("verification") or {}
        order = verification.get("order") if isinstance(verification.get("order"), dict) else {}
        fill_price = _positive_or_none(order.get("filled_avg_price"))
        filled_qty = _finite_or_none(order.get("filled_qty"))
        source = "verification_order"

        order_id = str(verification.get("order_id") or "").strip()
        if fill_price is None and order_id and verification.get("verified"):
            try:
                current = self._client.get_order(order_id)
            except Exception as exc:
                logger.warning("close fill lookup warning order_id=%s: %s", order_id, exc)
            else:
                if isinstance(current, dict):
                    fill_price = _positive_or_none(current.get("filled_avg_price"))
                    filled_qty = _finite_or_none(current.get("filled_qty"))
                    source = "order_refetch"

        if fill_price is None:
            source = "unavailable"
        return {"fill_price": fill_price, "filled_qty": filled_qty, "fill_source": source}

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

        pending_age = now - float(pending.get("created_wall") or now)
        if pending_age >= _PENDING_CLOSE_CHASE_SECS and order_id:
            try:
                self._client.cancel_order(order_id)
            except Exception as exc:
                logger.warning("pending close order chase cancel warning order_id=%s: %s", order_id, exc)
                return {
                    "still_pending": True,
                    "verification": {
                        "verified": False,
                        "status": status or "unknown",
                        "order_id": order_id,
                        "via": "pending_order_chase_cancel_failed",
                        "retryable": False,
                        "error": str(exc),
                        "order": last,
                    },
                    "order": last,
                }
            return {
                "retry_now": True,
                "reason": "stale_pending_close_chase",
                "verification": {
                    "verified": False,
                    "status": status or "unknown",
                    "order_id": order_id,
                    "via": "pending_order_chase_cancel",
                    "retryable": True,
                    "canceled_order_id": order_id,
                    "pending_age_sec": float(pending_age),
                    "order": last,
                },
                "order": last,
            }

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

    def _cancel_order_if_needed(self, verify_result: dict[str, Any]) -> dict[str, Any] | None:
        """Cancel a still-working close order and poll it to a terminal state
        (filled/canceled/rejected/...) before returning.

        cancel_order only requests cancellation — it does not guarantee the
        order is dead by the time it returns. Callers that immediately re-read
        the live position after firing a fire-and-forget cancel can race a
        limit that fills anyway. Polling here (bounded by the same
        verify timeout/poll cadence used elsewhere in this class) closes that
        window. Returns {"order_id", "status", "order"}, or None if there was
        nothing to cancel.
        """
        if not bool(verify_result.get("cancel_required")):
            return None
        order_id = str(verify_result.get("order_id", "")).strip()
        if not order_id:
            return None
        try:
            self._client.cancel_order(order_id)
        except Exception as exc:
            logger.warning("close order cancel warning order_id=%s: %s", order_id, exc)

        last: dict[str, Any] = {}
        status = ""
        deadline = time.monotonic() + _ORDER_VERIFY_TIMEOUT_SECS
        while time.monotonic() < deadline:
            try:
                current = self._client.get_order(order_id)
                if isinstance(current, dict):
                    last = current
                    status = _status_key(current.get("status"))
            except Exception as exc:
                logger.warning("cancel verify poll warning order_id=%s: %s", order_id, exc)
            if _order_is_success(status) or _order_is_terminal_fail(status):
                break
            time.sleep(_ORDER_VERIFY_POLL_SECS)
        logger.info("close order cancel resolved order_id=%s status=%s", order_id, status or "unknown")
        return {"order_id": order_id, "status": status or "unknown", "order": last}

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


def _positive_or_none(value: Any) -> float | None:
    """Finite and strictly positive, so an unfilled order's 0/None price never
    reads as a real fill."""
    out = _finite_or_none(value)
    return out if out is not None and out > 0.0 else None


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


# Floor as a fraction of the mid anchor, by liquidation pass. Passes are ~60s
# apart (_CLOSE_FAILURE_RETRY_SECS), so the ladder spends about three minutes
# above half the mid before it is allowed to touch the bid. Past the end of the
# schedule the floor is the bid itself.
_LIQUIDATION_FLOOR_BY_PASS = (0.85, 0.65, 0.50)
# The one-cent rung is a capitulation for a contract with no book, not a price.
# It needs BOTH a vanished bid and a spread wider than the mid itself, and only
# after the floor schedule is exhausted.
_PENNY_RUNG_MAX_BID = 0.02
_PENNY_RUNG_MIN_SPREAD_PCT_MID = 1.0


def _liquidation_close_ladder(
    *,
    base: float,
    bid: float,
    quote_meta: dict[str, Any] | None,
    attempts: int,
    close_pass: int,
) -> list[float]:
    """Descending sell limits for a forced exit, anchored on the mid.

    `base` is the mid/mark anchor and `bid` the current bid (NaN when the book is
    empty). Early passes refuse to price below a fraction of the anchor; the floor
    relaxes each pass and only reaches the bid once the schedule is exhausted. See
    _submit_close_order for why anchoring on the bid was wrong.
    """
    stage = max(1, int(close_pass))
    has_bid = math.isfinite(bid) and bid > 0.0
    if stage <= len(_LIQUIDATION_FLOOR_BY_PASS):
        floor = base * _LIQUIDATION_FLOOR_BY_PASS[stage - 1]
        # A bid above the scheduled floor is a real, better price — take it.
        if has_bid and bid > floor:
            floor = bid
    else:
        floor = bid if has_bid else _OPTION_TICK
    floor = min(_round_option_limit(floor), base)

    spread_pct_mid = _as_float((quote_meta or {}).get("spread_pct_mid"))
    penny_allowed = (
        stage > len(_LIQUIDATION_FLOOR_BY_PASS)
        and (not has_bid or bid <= _PENNY_RUNG_MAX_BID)
        and math.isfinite(spread_pct_mid)
        and spread_pct_mid > _PENNY_RUNG_MIN_SPREAD_PCT_MID
    )

    rungs = max(1, attempts - (1 if penny_allowed else 0))
    prices: list[float] = []
    step = (base - floor) / (rungs - 1) if rungs > 1 else 0.0
    for i in range(rungs):
        rounded = _round_option_limit(base - step * i)
        if rounded < floor:
            rounded = floor
        if rounded not in prices:
            prices.append(rounded)
    if penny_allowed and _OPTION_TICK not in prices:
        prices.append(_OPTION_TICK)
    return prices or [_OPTION_TICK]


def _close_limit_ladder(
    *,
    base_limit: float,
    close_bid: float,
    quote_meta: dict[str, Any] | None = None,
    reason: str,
    attempts: int,
    close_pass: int = 1,
) -> list[float]:
    attempts = max(1, int(attempts))
    base = _round_option_limit(base_limit if math.isfinite(base_limit) and base_limit > 0 else _OPTION_TICK)
    bid = close_bid if math.isfinite(close_bid) and close_bid > 0.0 else float("nan")
    spread = _as_float((quote_meta or {}).get("spread"))

    prices: list[float] = []
    if str(reason or "").strip().lower() in _LIQUIDATION_CLOSE_REASONS:
        return _liquidation_close_ladder(
            base=base, bid=bid, quote_meta=quote_meta,
            attempts=attempts, close_pass=close_pass,
        )

    anchor = bid if math.isfinite(bid) and bid > 0.0 else base
    if math.isfinite(spread) and spread > _OPTION_TICK:
        offsets = [
            0.0,
            _OPTION_TICK,
            max(_OPTION_TICK * 2.0, spread * 0.25),
            max(_OPTION_TICK * 4.0, spread * 0.50),
            max(_OPTION_TICK * 8.0, spread * 0.75),
        ]
    else:
        offsets = [attempt * _OPTION_TICK for attempt in range(0, attempts)]
    for offset in offsets:
        rounded = _round_option_limit(anchor - offset)
        if rounded not in prices:
            prices.append(rounded)
        if len(prices) >= attempts:
            break
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
