"""
Multi-ticker swing live runner.

Architecture:
  - Subscribes to 1m WebSocket bars for all 160 universe tickers + 6 context tickers.
  - Aggregates 1m → 5m and 1m → 30m internally.
  - On each 30m close: SwingScanner scores all tickers → top-5 signals by EV score
    enter CONFIRMING state (5m breakout confirmation, up to 6 bars = 30 min).
  - On each 5m close: checks confirmation + manages open position exits.
  - On confirmation: selects ATM option contract → submits buy via AlpacaOptionsClient.
  - Exits: tracked on underlying price (hard SL, trailing stop, ATR no-progress).

Run:
  python -m multi_ticker_swing.live.runner [--dry-run] [--env .env]
"""
from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time as _time, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from API.Alpaca_API.market_data.live_stream import AlpacaBarStreamer
from API.Alpaca_API.market_data.fetch_intraday import fetch_intraday
from API.Alpaca_API.options.options_api import AlpacaOptionsClient
from multi_ticker_swing.config.pipeline_config import CONTEXT_TICKERS, MODEL_PATH
from multi_ticker_swing.live.feature_builder import (
    LiveSwingFeatureBuilder,
    get_shared_feature_builder,
)
from multi_ticker_swing.live.position_manager import SwingPosition, SwingPositionManager
from multi_ticker_swing.live.scanner import Signal, SwingScanner
from multi_ticker_swing.live.universe import load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

BARS_PER_5M  = 5    # aggregate 5 × 1m bars into one 5m bar
BARS_PER_30M = 6    # aggregate 6 × 5m bars into one 30m bar
CONFIRM_MAX_5M = 6  # confirmation window (matches backtest)
DEFAULT_QTY  = 1

_ET = ZoneInfo("America/New_York")

# Market hours gate (ET): confirmations and scans only within this window.
# Exits always run regardless of time.
_CONFIRM_START = _time(10, 0)   # no entries in first 30 min (matches backtest)
_CONFIRM_END   = _time(15, 55)  # last 5m bar of regular session (3:55-3:59 ET)
_SCAN_END_TS   = _time(15, 55)  # skip 30m bar whose last 5m opens at 3:55 (post-close confirm impossible)

# Tickers that may be in the stream but are excluded from entry signals
_CONTEXT_SET = set(CONTEXT_TICKERS)

EventSink = Callable[[str, dict], None]


def _utc_iso(ts: datetime | None = None) -> str:
    return (ts or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Bar aggregators
# ---------------------------------------------------------------------------

class _BarAccumulator:
    """Accumulates N 1m bars into one aggregate OHLCV bar."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._buf: list[dict] = []

    def push(self, bar: dict) -> dict | None:
        self._buf.append(bar)
        if len(self._buf) >= self._n:
            agg = self._aggregate(self._buf[-self._n:])
            self._buf.clear()
            return agg
        return None

    @staticmethod
    def _aggregate(bars: list[dict]) -> dict:
        return {
            "timestamp": bars[-1]["timestamp"],
            "open":      bars[0]["open"],
            "high":      max(b["high"]   for b in bars),
            "low":       min(b["low"]    for b in bars),
            "close":     bars[-1]["close"],
            "volume":    sum(b["volume"] for b in bars),
        }


# ---------------------------------------------------------------------------
# Confirmation state
# ---------------------------------------------------------------------------

@dataclass
class _ConfirmState:
    signal: Signal
    bars_watched: int = 0
    confirmed: bool = False


# ---------------------------------------------------------------------------
# Options contract selection
# ---------------------------------------------------------------------------

_MIN_DTE_DAYS = 7    # skip monthly if fewer than 7 calendar days away
_DELTA_LO    = 0.20  # minimum |delta| for OTM contract selection
_DELTA_HI    = 0.40  # maximum |delta| — stays slightly OTM
_DELTA_TGT   = 0.30  # preferred |delta| within the range

def _next_monthly_expiry(ref_date: date) -> date:
    """Return the next monthly options expiry (third Friday of the month).

    Skips to the following month if the nearest monthly expiry is within
    _MIN_DTE_DAYS (e.g. on May 11 with May 15 expiry only 4 days away,
    returns June 19 instead).
    """
    def _third_friday(year: int, month: int) -> date:
        first = date(year, month, 1)
        days_to_fri = (4 - first.weekday()) % 7   # weekday 4 = Friday
        return first + timedelta(days=days_to_fri) + timedelta(weeks=2)

    year, month = ref_date.year, ref_date.month
    for _ in range(13):
        exp = _third_friday(year, month)
        if (exp - ref_date).days >= _MIN_DTE_DAYS:
            return exp
        month += 1
        if month > 12:
            month, year = 1, year + 1
    raise ValueError(f"Could not determine monthly expiry after {ref_date}")


def _select_contract(
    client: AlpacaOptionsClient,
    ticker: str,
    direction: int,
    current_price: float,
    ref_date: date,
) -> str | None:
    """
    Select a slightly-OTM option contract targeting delta 0.2–0.4 (|delta| ~0.30).

    Strategy:
      1. Fetch option chain snapshots (includes Greeks) for the next monthly expiry.
      2. Filter to contracts where |delta| is in [_DELTA_LO, _DELTA_HI].
      3. Pick the contract closest to _DELTA_TGT.
      4. If Greeks are unavailable (pre-market, API gap), fall back to nearest ATM strike.

    Returns OCC-format symbol string, or None if selection fails.
    """
    cp = "call" if direction == 1 else "put"
    expiry = _next_monthly_expiry(ref_date)
    expiry_str = expiry.strftime("%Y-%m-%d")

    # --- Primary path: delta-based selection via snapshots ---
    try:
        snapshots = client.get_option_snapshots(
            ticker,
            expiration_date=expiry_str,
            type=cp,
        )
        if snapshots:
            candidates = []
            for occ_sym, snap in snapshots.items():
                greeks = snap.get("greeks") or {}
                raw_delta = greeks.get("delta")
                if raw_delta is None:
                    continue
                abs_delta = abs(float(raw_delta))
                candidates.append((occ_sym, abs_delta))

            if candidates:
                # Prefer contracts in [_DELTA_LO, _DELTA_HI], else take closest overall
                in_range = [(s, d) for s, d in candidates if _DELTA_LO <= d <= _DELTA_HI]
                pool = in_range if in_range else candidates
                best_sym, best_d = min(pool, key=lambda x: abs(x[1] - _DELTA_TGT))
                logger.info(
                    "[%s] selected %s %s (|delta|=%.3f, exp=%s)",
                    ticker, cp, best_sym, best_d, expiry_str,
                )
                return best_sym
    except Exception as exc:
        logger.warning("[%s] snapshot delta selection failed (%s); falling back to ATM.", ticker, exc)

    # --- Fallback: nearest ATM strike via contracts list ---
    try:
        strike_lo = round(current_price * 0.90, 0)
        strike_hi = round(current_price * 1.10, 0)
        resp = client.get_option_contracts(
            underlying_symbol=ticker,
            expiration_date=expiry_str,
            type=cp,
            strike_price_gte=int(strike_lo),
            strike_price_lte=int(strike_hi),
        )
        contracts = resp.get("option_contracts") or (resp if isinstance(resp, list) else [])
    except Exception as exc:
        logger.error("[%s] get_option_contracts failed: %s", ticker, exc)
        return None

    if not contracts:
        logger.warning("[%s] no %s contracts found near ATM (%.2f) exp %s",
                       ticker, cp, current_price, expiry_str)
        return None

    best = min(contracts, key=lambda c: abs(float(c.get("strike_price", 0)) - current_price))
    logger.info("[%s] ATM fallback: selected %s %s (exp=%s)", ticker, cp, best.get("symbol"), expiry_str)
    return str(best.get("symbol", ""))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class SwingLiveRunner:
    def __init__(
        self,
        env_file: str = ".env",
        dry_run: bool = False,
        max_entries_per_bar: int = 5,
        event_sink: EventSink | None = None,
        feature_builder: LiveSwingFeatureBuilder | None = None,
        bar_queue: queue.Queue | None = None,
    ) -> None:
        self._dry_run = dry_run
        self._env_file = env_file
        self._sink = event_sink

        self._universe = load_universe()
        self._all_tickers = list(self._universe.keys())
        self._stream_symbols = list(set(self._all_tickers) | _CONTEXT_SET)

        # Reuse the shared (process-wide) feature builder so a Stop → Run cycle
        # skips the parquet load entirely when the in-memory cache is still fresh.
        self._fb = feature_builder if feature_builder is not None else get_shared_feature_builder()
        self._scanner = SwingScanner(
            self._fb,
            model_path=MODEL_PATH,
            max_entries_per_bar=max_entries_per_bar,
        )
        self._client = AlpacaOptionsClient(env_file=env_file)
        self._pos_mgr = SwingPositionManager(
            self._client, dry_run=dry_run, event_sink=self._emit,
        )

        # Per-ticker bar aggregators
        self._acc_5m:  dict[str, _BarAccumulator] = defaultdict(lambda: _BarAccumulator(BARS_PER_5M))
        self._acc_30m: dict[str, _BarAccumulator] = defaultdict(lambda: _BarAccumulator(BARS_PER_30M))

        # Rolling 5m bar history per ticker (60 bars ≈ 5h of market data).
        # Used to seed position charts with pre-entry context.
        self._buf_5m: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))

        # Confirmation watchers: ticker → _ConfirmState
        self._confirming: dict[str, _ConfirmState] = {}

        # If an external queue is provided (e.g. from SharedBarStream) use it directly
        # and skip creating / managing an AlpacaBarStreamer in start().
        self._bar_queue: queue.Queue = bar_queue if bar_queue is not None else queue.Queue(maxsize=10_000)
        self._external_queue: bool = bar_queue is not None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._streamer: AlpacaBarStreamer | None = None
        self._bar_count_total = 0
        self._last_bar_ts: datetime | None = None

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit(self, kind: str, payload: dict) -> None:
        if self._sink is None:
            return
        try:
            self._sink(kind, payload)
        except Exception as exc:
            logger.warning("event_sink raised on %s: %s", kind, exc)

    # ------------------------------------------------------------------
    # Public state accessors (for dashboard snapshots)
    # ------------------------------------------------------------------

    @property
    def stream_symbols(self) -> list[str]:
        return list(self._stream_symbols)

    @property
    def universe_size(self) -> int:
        return len(self._universe)

    def snapshot_confirming(self) -> list[dict]:
        with self._lock:
            out = []
            for tk, st in self._confirming.items():
                out.append({
                    "ticker": tk,
                    "direction": int(st.signal.direction),
                    "p_dir": float(st.signal.p_dir),
                    "ev_score": float(st.signal.ev_score),
                    "bars_watched": int(st.bars_watched),
                    "bars_max": int(CONFIRM_MAX_5M),
                    "ref_high": float(st.signal.ref_high),
                    "ref_low": float(st.signal.ref_low),
                    "atr": float(st.signal.atr),
                    "signal_ts": _utc_iso(st.signal.signal_ts) if isinstance(st.signal.signal_ts, datetime) else None,
                })
            return out

    def snapshot_positions(self) -> list[dict]:
        return self._pos_mgr.snapshot()

    def snapshot_meta(self) -> dict:
        return {
            "stream_symbols": len(self._stream_symbols),
            "universe_size": len(self._universe),
            "bar_count": int(self._bar_count_total),
            "last_bar_ts": _utc_iso(self._last_bar_ts) if self._last_bar_ts else None,
            "confirming_count": len(self._confirming),
            "open_positions_count": len(self._pos_mgr.open_tickers),
        }

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._emit("warmup_start", {
            "tickers": len(self._stream_symbols),
            "message": f"Warming up {len(self._stream_symbols)} symbols (cached → parquet → gap-fetch).",
        })
        logger.info("Warming up %d tickers (cache-aware, parallel gap-fetch)...", len(self._stream_symbols))

        # Counters mostly for runner-side logging; the dashboard tracks its own.
        counts = {"cached": 0, "loaded": 0, "gap_filled": 0, "failed": 0}
        last_progress_idx = 0

        def _on_progress(event: dict) -> None:
            nonlocal last_progress_idx
            phase = event.get("phase")
            if phase == "start":
                self._emit("warmup_progress", {
                    "phase": "start",
                    "total": int(event.get("total", 0)),
                })
                return
            if phase == "cached":
                counts["cached"] += 1
            elif phase == "loaded":
                counts["loaded"] += 1
                if not event.get("ok", False):
                    counts["failed"] += 1
            elif phase == "gap_filled":
                if int(event.get("added", 0)) > 0:
                    counts["gap_filled"] += 1
            elif phase == "gap_fetch_failed":
                counts["failed"] += 1
            # Forward every event to the dashboard for the warmup log
            payload = {
                "phase": phase,
                "ticker": event.get("ticker"),
                "bars": event.get("bars"),
                "added": event.get("added"),
                "ok": event.get("ok"),
                "error": event.get("error"),
                "from": event.get("from"),
                "to": event.get("to"),
            }
            if "index" in event and "total" in event:
                idx = int(event["index"])
                payload["index"] = idx
                payload["total"] = int(event["total"])
                last_progress_idx = max(last_progress_idx, idx)
            self._emit("warmup_progress", payload)

        summary = self._fb.prefill(
            self._stream_symbols,
            gap_fetch=True,
            save_back=True,
            progress=_on_progress,
        )

        logger.info(
            "Warmup complete. cached=%d loaded=%d gap_filled=%d failed=%d",
            counts["cached"], counts["loaded"],
            counts["gap_filled"], counts["failed"],
        )
        self._emit("warmup_done", {
            "tickers": len(self._stream_symbols),
            **counts,
        })

        if self._stop_event.is_set():
            self._emit("stopped", {"reason": "stop requested during warmup"})
            return

        if self._external_queue:
            # Bar queue is owned by the caller (e.g. SharedBarStream in combined_server).
            # We don't create or stop the Alpaca streamer here.
            logger.info("Using shared bar stream for %d symbols.", len(self._stream_symbols))
            self._emit("stream_started", {"symbols": len(self._stream_symbols), "shared": True})
            try:
                self._process_loop()
            finally:
                self._emit("stopped", {"reason": "loop_exit"})
        else:
            streamer = AlpacaBarStreamer(
                symbols=self._stream_symbols,
                env_file=self._env_file,
                queue=self._bar_queue,
            )
            streamer.start_in_thread(daemon=True)
            self._streamer = streamer
            logger.info("WebSocket stream started for %d symbols.", len(self._stream_symbols))
            self._emit("stream_started", {"symbols": len(self._stream_symbols), "shared": False})
            try:
                self._process_loop()
            finally:
                try:
                    streamer.stop()
                except Exception as exc:
                    logger.warning("streamer.stop() failed: %s", exc)
                try:
                    streamer.join(timeout=5.0)
                except Exception:
                    pass
                self._streamer = None
                self._emit("stopped", {"reason": "loop_exit"})

    def stop(self) -> None:
        """Signal the runner to stop. Safe to call from any thread."""
        self._stop_event.set()
        # Best-effort wake the queue.get
        try:
            self._bar_queue.put_nowait({"_sentinel": True})
        except Exception:
            pass

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                bar = self._bar_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            if bar.get("_sentinel"):
                err = bar.get("_error")
                if err:
                    logger.error("Stream fatal error: %s", err)
                    self._emit("stream_error", {
                        "error": err,
                        "hint": (
                            "Alpaca connection limit exceeded. Only one WebSocket stream "
                            "is allowed per account on the IEX free tier. Close any other "
                            "running sessions and wait ~60s for the old connection to expire, "
                            "then click Run again."
                        ),
                    })
                    self._stop_event.set()
                break
            ticker = bar.get("symbol", bar.get("ticker", "")).upper()
            if not ticker:
                continue
            self._bar_count_total += 1
            ts = bar.get("timestamp")
            if isinstance(ts, datetime):
                self._last_bar_ts = ts
            self._on_1m_bar(ticker, bar)

    # ------------------------------------------------------------------
    # 1m → 5m → 30m aggregation
    # ------------------------------------------------------------------

    def _on_1m_bar(self, ticker: str, bar: dict) -> None:
        bar5 = self._acc_5m[ticker].push(bar)
        if bar5 is not None:
            self._on_5m_bar(ticker, bar5)

    def _on_5m_bar(self, ticker: str, bar5: dict) -> None:
        # Maintain rolling 5m history for chart seeding and position_bar_5m events
        self._buf_5m[ticker].append(bar5)

        # Update position manager — exit checks run on underlying 5m bars
        self._pos_mgr.on_5m_bar(ticker, bar5)

        # Push live bar to dashboard if a position is open for this ticker
        pos = self._pos_mgr.get_position(ticker)
        if pos is not None:
            self._emit("position_bar_5m", {
                "ticker": ticker,
                "bar": _bar_to_event(bar5),
                "position": pos.to_chart_dict(),
            })

        # Market hours gate for confirmations and new entries.
        # Exits (handled above by pos_mgr.on_5m_bar) always run regardless of time.
        ts = bar5["timestamp"]
        bar_et_t = ts.astimezone(_ET).time() if hasattr(ts, "astimezone") else None
        if bar_et_t is None or _CONFIRM_START <= bar_et_t <= _CONFIRM_END:
            self._check_confirmation(ticker, bar5)

        # Aggregate 5m → 30m
        bar30 = self._acc_30m[ticker].push(bar5)
        if bar30 is not None:
            # Update feature builder with the new 30m bar
            self._fb.append_bar(ticker, bar30)
            self._on_30m_close(ticker, bar30)

    def _on_30m_close(self, ticker: str, bar30: dict) -> None:
        # Update context tickers silently; only scan universe tickers
        if ticker in _CONTEXT_SET and ticker not in self._universe:
            return

        # Market hours gate: the 30m bar whose last 5m bar opens at 3:55 closes right
        # at the market end — no 5m bars remain to confirm, so skip scanning.
        ts30 = bar30.get("timestamp")
        if hasattr(ts30, "astimezone"):
            if ts30.astimezone(_ET).time() >= _SCAN_END_TS:
                return

        # Run scanner once per 30m close for this ticker only
        # (Full cross-ticker scan triggered by a dedicated timer; per-ticker scan here
        # handles incremental signals as bars close at slightly different times)
        with self._lock:
            busy = set(self._confirming.keys()) | self._pos_mgr.open_tickers
            signals = self._scanner.scan([ticker], skip_tickers=busy)
            self._emit("scan", {
                "ticker": ticker,
                "ts": _utc_iso(bar30.get("timestamp")) if isinstance(bar30.get("timestamp"), datetime) else None,
                "signals": [
                    {
                        "ticker": s.ticker,
                        "direction": int(s.direction),
                        "p_dir": float(s.p_dir),
                        "ev_score": float(s.ev_score),
                    }
                    for s in signals
                ],
            })
            for sig in signals:
                self._confirming[sig.ticker] = _ConfirmState(signal=sig)
                logger.info("[%s] SIGNAL  dir=%+d  p=%.3f  ev=%.4f",
                            sig.ticker, sig.direction, sig.p_dir, sig.ev_score)
                self._emit("signal", {
                    "ticker": sig.ticker,
                    "direction": int(sig.direction),
                    "p_dir": float(sig.p_dir),
                    "ev_score": float(sig.ev_score),
                    "ref_high": float(sig.ref_high),
                    "ref_low": float(sig.ref_low),
                    "atr": float(sig.atr),
                })

    # ------------------------------------------------------------------
    # 5m confirmation: breakout on 5m bar after signal
    # ------------------------------------------------------------------

    def _check_confirmation(self, ticker: str, bar5: dict) -> None:
        state = self._confirming.get(ticker)
        if state is None:
            return

        sig = state.signal
        state.bars_watched += 1

        h, l = float(bar5["high"]), float(bar5["low"])
        o, c = float(bar5["open"]), float(bar5["close"])

        confirmed = False
        if sig.direction == 1:
            # Body-close breakout above signal bar high
            if h >= sig.ref_high and c > o and c > sig.ref_high:
                confirmed = True
        else:
            # Body-close breakout below signal bar low
            if l <= sig.ref_low and c < o and c < sig.ref_low:
                confirmed = True

        if confirmed:
            self._emit("confirmation", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "bars_watched": int(state.bars_watched),
                "close": float(c),
            })
            self._enter_trade(sig, bar5)
            with self._lock:
                self._confirming.pop(ticker, None)
        elif state.bars_watched >= CONFIRM_MAX_5M:
            logger.info("[%s] confirmation expired after %d bars", ticker, state.bars_watched)
            self._emit("confirmation_expired", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "bars_watched": int(state.bars_watched),
            })
            with self._lock:
                self._confirming.pop(ticker, None)

    # ------------------------------------------------------------------
    # Trade entry
    # ------------------------------------------------------------------

    def _enter_trade(self, sig: Signal, conf_bar: dict) -> None:
        ticker = sig.ticker
        # Entry at close of confirmation bar (matches backtest realism)
        entry_price = float(conf_bar["close"])
        today = datetime.now(timezone.utc).date()

        option_symbol = _select_contract(
            self._client, ticker, sig.direction, entry_price, today
        )
        if not option_symbol:
            logger.warning("[%s] no option contract found — skipping entry", ticker)
            self._emit("entry_skipped", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "reason": "no_contract_found",
                "entry_price": entry_price,
            })
            return

        qty = DEFAULT_QTY
        order_resp: Any = None
        order_error: str | None = None
        if not self._dry_run:
            try:
                order_resp = self._client.submit_option_order(
                    symbol=option_symbol,
                    qty=qty,
                    side="buy",
                    order_type="market",
                    time_in_force="day",
                )
                logger.info("[%s] buy order submitted: %s", ticker, order_resp)
            except Exception as exc:
                order_error = str(exc)
                logger.error("[%s] buy order FAILED: %s", ticker, exc)
                self._emit("order_failed", {
                    "ticker": ticker,
                    "side": "buy",
                    "option_symbol": option_symbol,
                    "qty": qty,
                    "error": order_error,
                })
                return
            self._emit("order_submitted", {
                "ticker": ticker,
                "side": "buy",
                "option_symbol": option_symbol,
                "qty": qty,
                "entry_price": entry_price,
                "response": _safe_response(order_resp),
            })
        else:
            logger.info("[DRY RUN] [%s] would BUY %d × %s at entry ~%.2f",
                        ticker, qty, option_symbol, entry_price)
            self._emit("order_dry_run", {
                "ticker": ticker,
                "side": "buy",
                "option_symbol": option_symbol,
                "qty": qty,
                "entry_price": entry_price,
            })

        entry_time = datetime.now(timezone.utc)
        pos = SwingPosition(
            ticker=ticker,
            direction=sig.direction,
            entry_price=entry_price,
            entry_time=entry_time,
            atr_at_entry=sig.atr,
            option_symbol=option_symbol,
            qty=qty,
            config=sig.config,
        )
        self._pos_mgr.open_position(pos)

        # Attach pre-entry 5m bar history so the chart can show context before the entry
        pre_bars = [_bar_to_event(b) for b in self._buf_5m.get(ticker, [])]
        self._emit("position_chart_seed", {
            "ticker": ticker,
            "direction": int(sig.direction),
            "entry_price": float(entry_price),
            "entry_time": int(entry_time.timestamp()),
            "sl_price": float(pos.sl_price) if pos.sl_price is not None else None,
            "pre_entry_bars": pre_bars,
        })


def _bar_to_event(bar: dict) -> dict:
    """Reduce a bar dict to a JSON-safe form suitable for Lightweight Charts."""
    ts = bar.get("timestamp")
    if isinstance(ts, datetime):
        epoch_sec = int(ts.timestamp())
    else:
        try:
            epoch_sec = int(datetime.fromisoformat(str(ts)).timestamp())
        except Exception:
            epoch_sec = 0
    return {
        "time": epoch_sec,
        "open":  float(bar.get("open",  float("nan"))),
        "high":  float(bar.get("high",  float("nan"))),
        "low":   float(bar.get("low",   float("nan"))),
        "close": float(bar.get("close", float("nan"))),
    }


def _safe_response(resp: Any) -> Any:
    """Reduce Alpaca response to JSON-friendly form for events."""
    if resp is None:
        return None
    if isinstance(resp, (str, int, float, bool)):
        return resp
    if isinstance(resp, dict):
        keep = ("id", "client_order_id", "symbol", "qty", "side", "status", "submitted_at")
        return {k: resp.get(k) for k in keep if k in resp}
    # Object with attributes
    out = {}
    for k in ("id", "client_order_id", "symbol", "qty", "side", "status", "submitted_at"):
        if hasattr(resp, k):
            out[k] = getattr(resp, k)
    return out or str(resp)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Multi-ticker swing live runner")
    p.add_argument("--dry-run",       action="store_true",
                   help="Log orders without submitting to Alpaca")
    p.add_argument("--env",           default=".env",
                   help="Path to .env file with Alpaca credentials")
    p.add_argument("--max-entries",   type=int, default=5,
                   help="Max new entries per 30m bar (default 5)")
    args = p.parse_args()

    runner = SwingLiveRunner(
        env_file=args.env,
        dry_run=args.dry_run,
        max_entries_per_bar=args.max_entries,
    )
    runner.start()


if __name__ == "__main__":
    main()
