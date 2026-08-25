from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from datetime import datetime, time as _time, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from strategies.dealer_positioning.chain import parse_schwab_option_chain, trading_expiration_buckets
from strategies.dealer_positioning.config import DealerPositioningConfig
from strategies.dealer_positioning.execution import DealerTradeExecutor
from strategies.dealer_positioning.levels import compute_gamma_levels
from strategies.dealer_positioning.models import GammaLevels
from strategies.dealer_positioning.schwab_adapter import SchwabDealerDataClient
from strategies.dealer_positioning.signals import DealerSignalEngine, DealerTradeSimulator, PriceCandle
from strategies.dealer_positioning.storage import DealerPositioningWriter

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")
_RTH_START = _time(9, 30)
_RTH_END = _time(16, 0)
EventSink = Callable[[str, dict[str, Any]], None]

#: Schwab's response when the refresh token has expired or been revoked. This is
#: a dead end, not a transient error: renewal is an interactive browser login, so
#: retrying cannot possibly succeed and the cached client in schwab_client.py
#: would not pick up a new token in this process anyway.
_DEAD_CREDENTIAL_MARKERS = (
    "invalid_grant",
    "unsupported_token_type",
    "refresh token is invalid",
    "refresh token is expired",
    "refresh token is revoked",
)


def _is_dead_credential(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _DEAD_CREDENTIAL_MARKERS)


class DealerPositioningRunner:
    def __init__(
        self,
        *,
        config: DealerPositioningConfig | None = None,
        data_client: Any | None = None,
        event_sink: EventSink | None = None,
        bar_queue: queue.Queue | None = None,
    ) -> None:
        self.config = config or DealerPositioningConfig.from_env()
        self._data_client = data_client
        self._event_sink = event_sink
        self._bar_queue = bar_queue
        self._writer = DealerPositioningWriter(self.config)
        self._signals = DealerSignalEngine(self.config)
        self._trade_engine = (
            DealerTradeExecutor(self.config) if self.config.submit_orders else DealerTradeSimulator(self.config)
        )
        self._stop = threading.Event()
        self._latest_levels: dict[str, GammaLevels] = {}
        self._last_candle: dict[str, PriceCandle] = {}
        self._poll_count = 0
        self._last_expiration_finalize_date = None
        #: Set once the Schwab credential is known dead. Polling then stops for
        #: the life of the process instead of reissuing a request that cannot
        #: succeed — on 2026-07-30 this loop logged 535 identical failures across
        #: five symbols in under two hours and buried everything else on the
        #: console. Recovery is a re-auth plus a restart, by design.
        self._auth_dead_reason: str | None = None

    @property
    def stream_symbols(self) -> list[str]:
        return ["SPY"] if "SPY" in self.config.symbols else []

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": {
                "symbols": list(self.config.symbols),
                "dte_offsets": list(self.config.dte_offsets),
                "poll_seconds": self.config.poll_seconds,
                "output_root": str(self.config.output_root),
                "submit_orders": bool(self.config.submit_orders),
                "alpaca_env_file": self.config.alpaca_env_file,
            },
            "poll_count": self._poll_count,
            "levels": {symbol: levels.to_dict() for symbol, levels in self._latest_levels.items()},
            "open_trades": [t.to_dict() for t in self._trade_engine.open_trades],
            "closed_trades": [t.to_dict() for t in self._trade_engine.closed_trades[-100:]],
        }

    def start(self) -> None:
        # Cheap file read, and it must come first: building the Schwab client
        # against a missing or dead token is what turns a knowable credential
        # problem into a startup crash. An already-expired token is knowable
        # before the first request, so there is no reason to discover it from a
        # failed poll -- or from a constructor blowing up.
        self._halt_early_if_token_already_expired()
        if self._auth_dead_reason is None and self._data_client is None:
            self._data_client = SchwabDealerDataClient(self.config)
        self._emit("session_started", self.snapshot())
        while not self._stop.is_set():
            started = time.monotonic()
            self._drain_bar_queue()
            now_et = datetime.now(_ET)
            if self.config.market_hours_only and not _is_market_hours(now_et):
                self._finalize_expired_trades(now_et)
                next_open_et = _next_market_open(now_et)
                sleep_seconds = max(1.0, (next_open_et - now_et).total_seconds())
                self._emit(
                    "market_closed",
                    {
                        "now_et": now_et.isoformat(),
                        "next_open_et": next_open_et.isoformat(),
                        "sleep_seconds": sleep_seconds,
                    },
                )
                self._stop.wait(sleep_seconds)
                continue
            if self._auth_dead_reason is None:
                for symbol in self.config.symbols:
                    try:
                        self._poll_symbol(symbol)
                    except Exception as exc:
                        if _is_dead_credential(exc):
                            self._halt_polling_for_dead_credential(symbol, exc)
                            break
                        logger.warning("[%s] dealer poll failed: %s", symbol, exc)
                        self._emit("poll_error", {"symbol": symbol, "error": str(exc)})
            elapsed = time.monotonic() - started
            self._stop.wait(max(1.0, float(self.config.poll_seconds) - elapsed))
        self._emit("session_ended", self.snapshot())

    def _halt_early_if_token_already_expired(self) -> None:
        """Skip polling entirely when the refresh token is known to be dead."""
        try:
            from core.schwab_token_status import REAUTH_COMMAND, schwab_token_status

            status = schwab_token_status()
        except Exception:  # noqa: BLE001 - an unreadable status must not block startup
            return
        if not status.expired:
            return
        self._auth_dead_reason = status.message
        logger.error(
            "Schwab credential is already expired at dealer startup — polling disabled "
            "for this process (0 requests will be attempted). %s Fix with: %s, then "
            "restart the server.",
            status.message, REAUTH_COMMAND,
        )
        self._emit(
            "auth_halted",
            {
                "symbol": None,
                "error": "refresh_token_expired_at_startup",
                "detail": status.message,
                "reauth_command": REAUTH_COMMAND,
                "halted_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _halt_polling_for_dead_credential(self, symbol: str, exc: BaseException) -> None:
        """Record the failure once, then stop polling until re-auth + restart."""
        reason = str(exc)
        self._auth_dead_reason = reason
        try:
            from core.schwab_token_status import REAUTH_COMMAND, schwab_token_status

            status = schwab_token_status()
            detail = status.message
            command = REAUTH_COMMAND
        except Exception:  # noqa: BLE001 - never let the status read mask the real error
            detail = reason
            command = "core/API/Schwab_API/schwab_client.py --reauth"

        logger.error(
            "[%s] Schwab credential is dead — dealer polling HALTED for this process "
            "(no further attempts will be made). %s Fix with: %s, then restart the server.",
            symbol, detail, command,
        )
        self._emit(
            "auth_halted",
            {
                "symbol": symbol,
                "error": reason,
                "detail": detail,
                "reauth_command": command,
                "halted_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def stop(self) -> None:
        self._stop.set()

    def _poll_symbol(self, symbol: str) -> None:
        symbol = symbol.upper()
        now = datetime.now(timezone.utc)
        expiration_buckets = trading_expiration_buckets(now.astimezone(_ET).date(), self.config.dte_offsets)
        chain = self._data_client.get_option_chain(symbol, now.astimezone(_ET).date())
        spot, rows = parse_schwab_option_chain(
            chain,
            symbol=symbol,
            timestamp=now,
            expiration_buckets=expiration_buckets,
        )
        if spot is None:
            spot = self._data_client.get_quote_price(symbol)
        if spot is None:
            raise RuntimeError("missing spot price from option chain and quote")
        rows = [
            row for row in rows
            if abs(float(row.strike) - float(spot)) <= float(spot) * float(self.config.strike_window_pct)
        ]
        if not rows:
            raise RuntimeError("option chain had no rows inside configured strike window")

        ladder, levels = compute_gamma_levels(
            rows,
            symbol=symbol,
            spot=float(spot),
            magnet_quantile=self.config.magnet_quantile,
        )
        self._latest_levels[symbol] = levels
        candle = self._latest_candle(symbol, float(spot), now)
        self._last_candle[symbol] = candle

        try:
            closed = self._trade_engine.on_candle(symbol, candle)
        except Exception as exc:
            closed = []
            self._emit("trade_update_error", {"symbol": symbol, "error": str(exc)})
        for trade in closed:
            self._writer.append_trade(trade)
            self._emit("trade_closed", trade.to_dict())

        for signal in self._signals.on_candle(symbol, candle, levels):
            self._writer.append_signal(signal)
            self._emit("signal", signal.to_dict())
            try:
                trade = self._trade_engine.on_signal(signal)
            except Exception as exc:
                trade = None
                self._emit(
                    "trade_execution_error",
                    {"symbol": symbol, "signal": signal.to_dict(), "error": str(exc)},
                )
            if trade is not None:
                self._writer.append_trade(trade)
                self._emit("trade_opened", trade.to_dict())

        self._writer.write_snapshot(
            symbol=symbol,
            ladder=ladder,
            levels=levels,
            open_trades=[t for t in self._trade_engine.open_trades if t.symbol == symbol],
            closed_trades=[t for t in self._trade_engine.closed_trades if t.symbol == symbol],
        )
        self._poll_count += 1
        self._emit("levels", {"symbol": symbol, "levels": levels.to_dict(), "rows": int(len(ladder))})

    def _latest_candle(self, symbol: str, spot: float, now: datetime) -> PriceCandle:
        queued = self._last_candle.get(symbol)
        if queued is not None and abs((now - queued.timestamp).total_seconds()) <= self.config.poll_seconds * 2:
            return queued
        return PriceCandle(timestamp=now, open=spot, high=spot, low=spot, close=spot)

    def _drain_bar_queue(self) -> None:
        if self._bar_queue is None:
            return
        drained = 0
        while True:
            try:
                bar = self._bar_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1
            symbol = str(bar.get("symbol") or bar.get("ticker") or "").upper()
            if not symbol:
                continue
            ts = bar.get("timestamp")
            if not isinstance(ts, datetime):
                ts = datetime.now(timezone.utc)
            self._last_candle[symbol] = PriceCandle(
                timestamp=ts,
                open=float(bar.get("open") or bar.get("close") or 0.0),
                high=float(bar.get("high") or bar.get("close") or 0.0),
                low=float(bar.get("low") or bar.get("close") or 0.0),
                close=float(bar.get("close") or 0.0),
                volume=float(bar.get("volume") or 0.0),
            )
        if drained:
            self._emit("bar_queue_drained", {"drained": drained})

    def _finalize_expired_trades(self, now_et: datetime) -> None:
        if self._last_expiration_finalize_date == now_et.date():
            return
        finalize = getattr(self._trade_engine, "finalize_expired", None)
        if finalize is None:
            self._last_expiration_finalize_date = now_et.date()
            return
        last_prices = {
            symbol: float(candle.close)
            for symbol, candle in self._last_candle.items()
        }
        closed = finalize(now_et, last_prices)
        for trade in closed:
            self._writer.append_trade(trade)
            self._emit("trade_closed", trade.to_dict())
        for symbol in self.config.symbols:
            self._writer.write_trades(
                symbol=symbol,
                open_trades=[t for t in self._trade_engine.open_trades if t.symbol == symbol],
                closed_trades=[t for t in self._trade_engine.closed_trades if t.symbol == symbol],
            )
        if closed:
            self._emit("expired_trades_finalized", {"count": len(closed)})
        self._last_expiration_finalize_date = now_et.date()

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_sink is not None:
            self._event_sink(event_type, payload)


def _is_market_hours(now_et: datetime) -> bool:
    if now_et.weekday() >= 5:
        return False
    local_time = now_et.time()
    return _RTH_START <= local_time <= _RTH_END


def _next_market_open(now_et: datetime) -> datetime:
    local_date = now_et.date()
    local_time = now_et.time()

    if now_et.weekday() < 5 and local_time < _RTH_START:
        return datetime.combine(local_date, _RTH_START, tzinfo=_ET)

    days_ahead = 1
    while True:
        candidate = local_date.fromordinal(local_date.toordinal() + days_ahead)
        if candidate.weekday() < 5:
            return datetime.combine(candidate, _RTH_START, tzinfo=_ET)
        days_ahead += 1


def _replace_config(config: DealerPositioningConfig, **updates: Any) -> DealerPositioningConfig:
    return DealerPositioningConfig(**{**config.__dict__, **updates})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run live rule-based dealer positioning simulator.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols, default SPY,SPX.")
    parser.add_argument("--poll-seconds", type=int, default=None, help="Polling interval, default 60.")
    parser.add_argument("--submit-orders", action="store_true", help="Submit Alpaca paper/live option orders instead of sim-only trades.")
    parser.add_argument("--alpaca-env-file", default=None, help="Alpaca env/profile to use for dealer-positioning orders, default .env#paper.")
    args = parser.parse_args()

    config = DealerPositioningConfig.from_env()
    if args.symbols:
        config = DealerPositioningConfig(
            **{**config.__dict__, "symbols": tuple(x.strip().upper() for x in args.symbols.split(",") if x.strip())}
        )
    if args.poll_seconds:
        config = DealerPositioningConfig(**{**config.__dict__, "poll_seconds": int(args.poll_seconds)})
    if args.submit_orders:
        config = DealerPositioningConfig(**{**config.__dict__, "submit_orders": True})
    if args.alpaca_env_file:
        config = DealerPositioningConfig(**{**config.__dict__, "alpaca_env_file": str(args.alpaca_env_file)})
    runner = DealerPositioningRunner(config=config)
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.stop()


if __name__ == "__main__":
    main()
