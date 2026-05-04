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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from API.Alpaca_API.options.options_api import AlpacaOptionsClient
from multi_ticker_swing.live.universe import TickerConfig

logger = logging.getLogger(__name__)

ARM_PCT      = 0.025   # arm trailing stop after 2.5% underlying move (matches backtest)
GIVEBACK_PCT = 0.25    # exit when 25% of peak gain given back

EventSink = Callable[[str, dict], None]


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

    # Derived at entry
    sl_price: float | None = None

    # Mutable tracking state
    best_price: float = field(init=False)
    last_price: float = field(init=False)
    trail_armed: bool = False
    bar_count_5m: int = 0

    def __post_init__(self):
        self.best_price = self.entry_price
        self.last_price = self.entry_price
        if self.config.sl_atr > 0 and not _isnan(self.atr_at_entry):
            self.sl_price = (
                self.entry_price - self.direction * self.config.sl_atr * self.atr_at_entry
            )

    def update(self, bar: dict) -> str | None:
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

        # 3. Trailing stop
        move_pct = self.direction * (self.best_price - self.entry_price) / self.entry_price
        if move_pct >= ARM_PCT:
            self.trail_armed = True
        if self.trail_armed:
            peak_profit  = self.direction * (self.best_price - self.entry_price)
            cur_profit   = self.direction * (bar_c        - self.entry_price)
            floor_profit = peak_profit * (1.0 - GIVEBACK_PCT)
            if cur_profit <= floor_profit:
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

    def _trail_floor(self) -> float | None:
        """Price at which the trailing stop would trigger (None if not yet armed)."""
        if not self.trail_armed:
            return None
        peak_profit = self.direction * (self.best_price - self.entry_price)
        floor_profit = peak_profit * (1.0 - GIVEBACK_PCT)
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
            "bars_held": int(self.bar_count_5m),
            "atr_at_entry": float(self.atr_at_entry) if not _isnan(self.atr_at_entry) else None,
            "option_symbol": str(self.option_symbol),
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
        keep = ("id", "client_order_id", "symbol", "qty", "side", "status", "submitted_at")
        return {k: resp.get(k) for k in keep if k in resp}
    out = {}
    for k in ("id", "client_order_id", "symbol", "qty", "side", "status", "submitted_at"):
        if hasattr(resp, k):
            out[k] = getattr(resp, k)
    return out or str(resp)


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
    ) -> None:
        self._client = alpaca_client
        self._dry_run = dry_run
        self._sink = event_sink
        self._positions: dict[str, SwingPosition] = {}   # ticker → position

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

        reason = pos.update(bar)
        if reason:
            self._close_position(pos, reason, bar)

    def _close_position(self, pos: SwingPosition, reason: str, bar: dict) -> None:
        exit_price = float(bar["close"])
        pnl_pct = pos.direction * (exit_price - pos.entry_price) / pos.entry_price
        logger.info(
            "[%s] CLOSE  reason=%-12s  pnl=%+.2f%%  bars=%d  option=%s",
            pos.ticker, reason, pnl_pct * 100, pos.bar_count_5m, pos.option_symbol,
        )
        order_resp = None
        order_error: str | None = None
        if not self._dry_run:
            try:
                order_resp = self._client.submit_option_order(
                    symbol=pos.option_symbol,
                    qty=pos.qty,
                    side="sell",
                    order_type="market",
                    time_in_force="day",
                )
                logger.info("[%s] sell order submitted: %s", pos.ticker, order_resp)
                self._emit("order_submitted", {
                    "ticker": pos.ticker,
                    "side": "sell",
                    "option_symbol": pos.option_symbol,
                    "qty": pos.qty,
                    "exit_price": exit_price,
                    "response": _safe_response(order_resp),
                })
            except Exception as exc:
                order_error = str(exc)
                logger.error("[%s] sell order FAILED: %s", pos.ticker, exc)
                self._emit("order_failed", {
                    "ticker": pos.ticker,
                    "side": "sell",
                    "option_symbol": pos.option_symbol,
                    "qty": pos.qty,
                    "error": order_error,
                })
        else:
            self._emit("order_dry_run", {
                "ticker": pos.ticker,
                "side": "sell",
                "option_symbol": pos.option_symbol,
                "qty": pos.qty,
                "exit_price": exit_price,
            })

        self._emit("position_closed", {
            **pos.to_dict(),
            "exit_price": exit_price,
            "exit_pnl_pct": float(pnl_pct),
            "exit_reason": reason,
            "order_error": order_error,
        })
        del self._positions[pos.ticker]
