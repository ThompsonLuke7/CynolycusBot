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
import os
import math
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

from core.API.Alpaca_API.market_data.live_stream import AlpacaBarStreamer
from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday
from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.live_4h_exec import contracts_for_notional
from strategies.dealer_positioning.gate import (
    SCOPE_NEAREST,
    evaluate_dealer_gate,
    gate_enabled as dealer_gate_enabled,
)
from strategies.multi_ticker_swing.config.pipeline_config import CONTEXT_TICKERS, MODEL_PATH
from strategies.multi_ticker_swing.live.feature_builder import (
    LiveSwingFeatureBuilder,
    get_shared_feature_builder,
)
from strategies.multi_ticker_swing.live.position_manager import SwingPosition, SwingPositionManager
from strategies.multi_ticker_swing.live.real_account_policy import (
    RealAccountBookkeeper,
    config_from_env as real_account_policy_from_env,
)
from strategies.multi_ticker_swing.live.risk_profile_policy import RiskProfilePolicy, config_from_env as risk_profile_policy_from_env
from strategies.multi_ticker_swing.live.signal_policy import SignalPolicyDecision, SignalPolicyLayer, config_from_env as signal_policy_from_env
from strategies.multi_ticker_swing.live.scanner import Signal, SwingScanner  # noqa: F401 (Signal used)
from strategies.multi_ticker_swing.live.ranker_scanner import RankerSwingScanner
from strategies.multi_ticker_swing.live.catalyst_signal import LiveCatalystSignal
from strategies.multi_ticker_swing.live.session import (
    confirmation_breakout,
    entry_bucket as _entry_bucket,
    is_log_window,
    is_regular_trading_time,
    should_check_confirmation,
    should_scan_after_30m_close,
)
from strategies.multi_ticker_swing.live.universe import load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

BARS_PER_5M  = 5    # aggregate 5 × 1m bars into one 5m bar
BARS_PER_30M = 6    # aggregate 6 × 5m bars into one 30m bar
CONFIRM_MAX_5M = 6  # confirmation window (matches backtest)
DEFAULT_QTY  = 1        # fallback only if TARGET_NOTIONAL_USD sizing can't price the contract
TARGET_NOTIONAL_USD = 5000.0  # dollar size per new entry when neither real- nor signal-policy sizing is active

_ET = ZoneInfo("America/New_York")

# Market hours gate (ET): confirmations and scans only within this window.
# Exits always run regardless of time.
_CONFIRM_START = _time(10, 0)   # no entries in first 30 min (matches backtest)
_CONFIRM_END   = _time(15, 55)  # last 5m bar of regular session (3:55-3:59 ET)
_SCAN_END_TS   = _time(15, 55)  # skip 30m bar whose last 5m opens at 3:55 (post-close confirm impossible)
_LIVE_OPTION_FILTER_POLICY = "baseline"
_CHALLENGER_OPTION_FILTER_POLICY = "calls_only_best_filter_v1"
_CHALLENGER_ALLOW_SHORT_ENTRIES = False

# LONG-ONLY GATE (independent of the challenger policy above).
#
# This module is options-only, so a short signal is expressed by BUYING PUTS. That
# lost money persistently: across 299 real put fills, -$32,928 vs +$5,827 on 258
# calls. Three independent problems stack (12_dte_and_put_call_study.md):
#   1. downside excursion is smaller AND stalls -- 8.3% median by 30d vs 18.7% for
#      calls, and it plateaus entirely after ~60-90d;
#   2. pure skew cost -- conditioned on the SAME underlying move, puts returned
#      15-18pp less than calls at every move size;
#   3. it is not a regime artifact -- puts lost in all three risk-appetite terciles,
#      and in risk-OFF they had BETTER underlying movement (2.29% vs 1.57%) yet
#      still lost $13,772 while calls made $708.
#
# Deliberately a separate flag rather than enabling _CHALLENGER_OPTION_FILTER_POLICY,
# which would also activate blocked-ticker and blocked-time-bucket rules that are a
# different (and separately-evidenced) decision.
#
# This BLOCKS short entries; it does not convert them to short shares. Converting
# would require an equity execution path this module does not have (it is
# options-only: no share entry, exit, or reconciliation logic). Per user direction,
# shares are not to be added here until they are shown to work.
_ALLOW_SHORT_ENTRIES = False
_CHALLENGER_BLOCKED_LONG_ENTRY_BUCKETS = {"12:30", "15:00", "15:30"}
_CHALLENGER_BLOCKED_LONG_ENTRY_TICKERS = {
    # Challenger policy from live fill audit through 2026-05-26.
    "ADI",
    "SPY",
    "SOXL",
    "RDDT",
    "MRNA",
    "IREN",
    "TGT",
    "ABNB",
    "CVNA",
}


def _challenger_policy_enabled() -> bool:
    return _LIVE_OPTION_FILTER_POLICY == _CHALLENGER_OPTION_FILTER_POLICY

# Regular trading hours gate for bar aggregation (drop pre/post-market bars entirely)
_RTH_START = _time(9, 30)
_RTH_END   = _time(16, 0)

# Stream health: warn if no bar received for this many seconds during RTH.
_STREAM_STALE_SECS = 300

# If we are still dequeuing bars whose timestamps are this far behind wall-clock
# RTH, the shared queue is stale/backlogged.  Drop them rather than trading on
# hours-old data.
_STALE_BAR_LAG_SECS = 10 * 60
_BACKLOG_EVENT_THROTTLE_SECS = 60.0
_HEARTBEAT_SECS = 60.0
_BROKER_RECONCILE_SECS = 60.0
_LOG_END = _time(16, 15)
_ENTRY_ORDER_ATTEMPTS = 3


def _entry_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:  # noqa: BLE001 - a bad override must not stop trading
        return default


# How long a passive rung is given before it is cancelled and the ladder walks
# up. Five seconds is not long enough for a market maker to come to a resting
# order, so the ladder mechanically escalates: across 490 multi-rung ladders in
# UI/swing_audit the mean fill sits 0.326 of the way from mid to ask, for
# $22,025 of cumulative fill-versus-mid slippage and $10,749 of it in August
# alone.
#
# DEFAULTS PRESERVE TODAY'S BEHAVIOUR ON PURPOSE. Whether a longer dwell
# actually fills better cannot be established from this data: Alpaca paper fills
# model something (HII filled 10.70 against a ladder of [9.99, 10.41, 10.84];
# T filled 0.40 below its first rung of 0.41) but they are not evidence about
# live market makers. Raise the dwell and lower the cap deliberately, then
# measure BOTH slippage and the miss rate — the ladder already fails to fill
# more often than it succeeds, so a change that only improves price is not
# obviously a win.
_ENTRY_ORDER_VERIFY_TIMEOUT_SECS = _entry_env_float(
    "MULTITICKER_ENTRY_RUNG_DWELL_SECS", 5.0
)
_ENTRY_ORDER_VERIFY_POLL_SECS = 0.5
# Fraction of the mid->ask distance the last rung is allowed to reach. 1.0 is
# the ask, which is where the ladder tops out today.
_ENTRY_LADDER_MAX_ASK_FRACTION = _entry_env_float(
    "MULTITICKER_ENTRY_LADDER_MAX_ASK_FRACTION", 1.0
)
_OPTION_TICK = 0.01

# Tickers that may be in the stream but are excluded from entry signals
_CONTEXT_SET = set(CONTEXT_TICKERS)

EventSink = Callable[[str, dict], None]


def _utc_iso(ts: datetime | None = None) -> str:
    return (ts or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Bar aggregators
# ---------------------------------------------------------------------------

class _BarAccumulator:
    """Clock-aligned OHLCV accumulator for 1m->5m and 5m->30m bars."""

    def __init__(self, bucket_minutes: int, *, min_bars: int = 1) -> None:
        self._bucket_minutes = int(bucket_minutes)
        self._min_bars = max(1, int(min_bars))
        self._buf: list[dict] = []
        self._bucket_start: datetime | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._bucket_start = None

    def push(self, bar: dict) -> dict | None:
        ts = bar.get("timestamp")
        if not isinstance(ts, datetime):
            return None

        bucket_start = self._bucket_start_for(ts)
        if self._bucket_start is None:
            self._bucket_start = bucket_start
        elif bucket_start != self._bucket_start:
            completed = self._finalize()
            self._bucket_start = bucket_start
            self._buf.append(bar)
            return completed

        self._buf.append(bar)
        if self._is_bucket_end(ts):
            return self._finalize()
        return None

    def _bucket_start_for(self, ts: datetime) -> datetime:
        local = ts.astimezone(_ET) if ts.tzinfo else ts.replace(tzinfo=_ET)
        minute = (local.minute // self._bucket_minutes) * self._bucket_minutes
        return local.replace(minute=minute, second=0, microsecond=0)

    def _is_bucket_end(self, ts: datetime) -> bool:
        local = ts.astimezone(_ET) if ts.tzinfo else ts.replace(tzinfo=_ET)
        return local.minute % self._bucket_minutes == self._bucket_minutes - 1

    def _finalize(self) -> dict | None:
        bars = list(self._buf)
        self.reset()
        if len(bars) < self._min_bars:
            return None
        return self._aggregate(bars)

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

# DTE floor. Was 0 ("allow the nearest listed expiry, including 0DTE/1DTE weeklies"),
# which made the median entry a 2-DTE contract with 56% of entries at <=2 DTE.
#
# Evidence (research/options_experiment/13_dte_floor_and_regime_rules.md, measured on
# this module's own 575 real fills + underlying bars):
#   * 67% of trades eventually reach a +10% favorable underlying move within 60d;
#     time-to-+10% is median 10 days, p75 22, p90 38.
#   * A 2-day floor captures only 19% of those moves. 21 days captures 74%
#     (14d: 60%, 30d: 83%, 45d: 95%) -- 21d is the efficiency knee.
#   * Even the LOSING trades went on to a 9.2% median favorable move by 30d, i.e.
#     the thesis was usually right and the clock was wrong.
#   * Median hold here is only ~20 hours, so the second (and for short holds the
#     larger) benefit is decay: 20h burns ~40% of a 2-DTE contract's remaining life
#     but a trivial fraction of a 21-DTE contract's.
# Nothing observable at signal time predicts move speed (best |r| = 0.22), so a flat
# floor is the right instrument -- a per-trade adaptive/learned DTE is not supportable.
#
# NOT yet proven: that the extra premium of a longer-dated contract is repaid. Option
# marks cannot be reconstructed historically for this universe
# (see 10_RETRACTION_option_pnl_invalid.md), so this is an evidence-backed change to
# the move-capture window, not a validated P&L improvement. Watch realized results.
_MIN_DTE_DAYS = 21
_EXPIRY_LOOKAHEAD_DAYS = 90
_ZERO_DTE_CUTOFF = _time(13, 0)
_FRIDAY_LATE_EXPIRY_CUTOFF = _time(13, 0)
_MAX_ENTRY_SPREAD_PCT_MID = 0.18
_DELTA_LO    = 0.35  # minimum |delta| for contract selection
_DELTA_HI    = 0.60  # maximum |delta| for contract selection
_DELTA_TGT   = 0.45  # preferred |delta| within the range

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


def _contract_strike(contract: dict) -> float:
    try:
        return float(contract.get("strike_price", 0.0))
    except Exception:
        return 0.0


def _contract_expiry(contract: dict) -> date | None:
    raw = contract.get("expiration_date")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def _is_standard_100_contract(contract: dict, ticker: str) -> bool:
    """True for standard, full-size option contracts.

    Alpaca exposes both multiplier and size on option contract records. Minis,
    adjusted contracts, and other non-standard deliverables should not be traded
    by this live runner, so require the normal 100-share contract fields when
    present and keep root_symbol pinned to the underlying ticker.
    """
    root_symbol = str(contract.get("root_symbol") or ticker).upper()
    if root_symbol != ticker.upper():
        return False
    multiplier = contract.get("multiplier")
    if multiplier is not None and str(multiplier) != "100":
        return False
    size = contract.get("size")
    if size is not None and str(size) != "100":
        return False
    return True


def _contract_symbol(contract: dict) -> str:
    return str(contract.get("symbol", "")).strip().upper()


def _contract_selection_meta(
    *,
    ticker: str,
    cp: str,
    symbol: str,
    current_price: float,
    expiry_str: str | None,
    strike: float,
    selected_delta: float | None,
    method: str,
) -> dict[str, Any]:
    dte = None
    if expiry_str:
        try:
            dte = (date.fromisoformat(expiry_str) - datetime.now(_ET).date()).days
        except Exception:
            dte = None
    moneyness_pct = (
        (float(strike) / float(current_price) - 1.0)
        if cp == "call"
        else (float(current_price) / float(strike) - 1.0)
        if strike else None
    )
    return {
        "underlying": ticker.upper(),
        "option_symbol": symbol,
        "option_type": "C" if cp == "call" else "P",
        "selection_method": method,
        "target_delta": float(_DELTA_TGT),
        "delta_min": float(_DELTA_LO),
        "delta_max": float(_DELTA_HI),
        "selected_abs_delta": float(selected_delta) if selected_delta is not None else None,
        "expiration": expiry_str,
        "dte": int(dte) if dte is not None else None,
        "strike": float(strike) if math.isfinite(float(strike or 0.0)) else None,
        "underlying_price_at_selection": float(current_price),
        "moneyness_pct": float(moneyness_pct) if moneyness_pct is not None else None,
    }


def _available_contracts(
    client: AlpacaOptionsClient,
    ticker: str,
    cp: str,
    current_price: float,
    ref_date: date,
) -> tuple[str | None, list[dict]]:
    """Return standard 100-share contracts for the nearest listed expiry.

    Alpaca lists holiday-adjusted expiries by their actual trading date. Looking up
    a hard-coded third Friday can miss the chain when the monthly expiry moves to
    Thursday, so live selection starts from the contracts endpoint and lets the
    broker tell us which expiries exist. With weeklies, the nearest weekly wins;
    without weeklies, this naturally falls back to the closest monthly chain.
    """
    start = ref_date + timedelta(days=_MIN_DTE_DAYS)
    end = ref_date + timedelta(days=_EXPIRY_LOOKAHEAD_DAYS)
    strike_lo = int(round(current_price * 0.90, 0))
    strike_hi = int(round(current_price * 1.10, 0))

    contracts: list[dict] = []
    page_token: str | None = None
    for _ in range(10):
        resp = client.get_option_contracts(
            underlying_symbol=ticker,
            expiration_date_gte=start.strftime("%Y-%m-%d"),
            expiration_date_lte=end.strftime("%Y-%m-%d"),
            type=cp,
            strike_price_gte=strike_lo,
            strike_price_lte=strike_hi,
            status="active",
            page_token=page_token,
        )
        page = resp.get("option_contracts") if isinstance(resp, dict) else resp
        if page:
            contracts.extend(c for c in page if isinstance(c, dict))
        page_token = resp.get("next_page_token") if isinstance(resp, dict) else None
        if not page_token:
            break

    tradable = [
        c for c in contracts
        if (
            c.get("tradable", True)
            and _is_standard_100_contract(c, ticker)
            and (exp := _contract_expiry(c)) is not None
            and exp >= start
        )
    ]
    if not tradable:
        return None, []

    nearest_expiry = min(_contract_expiry(c) for c in tradable if _contract_expiry(c) is not None)
    expiry_str = nearest_expiry.strftime("%Y-%m-%d")
    return expiry_str, [c for c in tradable if c.get("expiration_date") == expiry_str]


def _entry_contract_ref_date(now_et: datetime) -> date:
    ref_date = now_et.date()
    if now_et.weekday() == 4 and now_et.time() >= _FRIDAY_LATE_EXPIRY_CUTOFF:
        return ref_date + timedelta(days=4)
    if now_et.time() >= _ZERO_DTE_CUTOFF:
        return ref_date + timedelta(days=1)
    return ref_date


def _select_contract(
    client: AlpacaOptionsClient,
    ticker: str,
    direction: int,
    current_price: float,
    ref_date: date,
) -> tuple[str | None, dict[str, Any]]:
    """
    Select an option contract targeting delta 0.35–0.60 (|delta| ~0.45).

    Strategy:
      1. Discover the nearest listed expiry with at least _MIN_DTE_DAYS remaining.
      2. Fetch option chain snapshots for that actual expiry (includes Greeks).
      3. Filter to contracts where |delta| is in [_DELTA_LO, _DELTA_HI].
      4. Pick the contract closest to _DELTA_TGT.
      5. If Greeks are unavailable (pre-market, API gap), fall back to nearest ATM strike.

    Returns (OCC-format symbol string, metadata), or (None, metadata) if selection fails.
    """
    cp = "call" if direction == 1 else "put"
    base_meta = {
        "underlying": ticker.upper(),
        "option_type": "C" if cp == "call" else "P",
        "target_delta": float(_DELTA_TGT),
        "delta_min": float(_DELTA_LO),
        "delta_max": float(_DELTA_HI),
        "underlying_price_at_selection": float(current_price),
    }
    try:
        expiry_str, contracts = _available_contracts(client, ticker, cp, current_price, ref_date)
    except Exception as exc:
        logger.error("[%s] get_option_contracts failed: %s", ticker, exc)
        return None, {**base_meta, "selection_error": str(exc)}

    if not expiry_str or not contracts:
        logger.warning(
            "[%s] no standard 100-share %s contracts found near ATM (%.2f)",
            ticker, cp, current_price,
        )
        return None, {**base_meta, "selection_error": "no_standard_contracts"}

    available_symbols = {_contract_symbol(c) for c in contracts}
    contract_by_symbol = {_contract_symbol(c): c for c in contracts}

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
                occ_sym = str(occ_sym).strip().upper()
                if available_symbols and occ_sym not in available_symbols:
                    continue
                greeks = snap.get("greeks") or {}
                raw_delta = greeks.get("delta")
                if raw_delta is None:
                    continue
                abs_delta = abs(float(raw_delta))
                candidates.append((occ_sym, abs_delta, snap))

            if candidates:
                # Prefer contracts in [_DELTA_LO, _DELTA_HI], else take closest overall
                in_range = [(s, d, snap) for s, d, snap in candidates if _DELTA_LO <= d <= _DELTA_HI]
                pool = in_range if in_range else candidates
                best_sym, best_d, best_snap = min(pool, key=lambda x: abs(x[1] - _DELTA_TGT))
                contract = contract_by_symbol.get(best_sym, {})
                strike = _contract_strike(contract)
                meta = _contract_selection_meta(
                    ticker=ticker,
                    cp=cp,
                    symbol=best_sym,
                    current_price=current_price,
                    expiry_str=expiry_str,
                    strike=strike,
                    selected_delta=best_d,
                    method="delta_snapshot",
                )
                greeks = best_snap.get("greeks") if isinstance(best_snap, dict) else None
                if isinstance(greeks, dict):
                    meta["greeks"] = {
                        k: _as_float(greeks.get(k))
                        for k in ("delta", "gamma", "theta", "vega", "rho")
                        if greeks.get(k) is not None
                    }
                if not in_range:
                    meta["selection_warning"] = "no_contract_in_target_delta_range"
                logger.info(
                    "[%s] selected %s %s (|delta|=%.3f, exp=%s)",
                    ticker, cp, best_sym, best_d, expiry_str,
                )
                return best_sym, meta
    except Exception as exc:
        logger.warning("[%s] snapshot delta selection failed (%s); falling back to ATM.", ticker, exc)

    # --- Fallback: nearest ATM strike via contracts list ---
    best = min(contracts, key=lambda c: abs(_contract_strike(c) - current_price))
    best_sym = _contract_symbol(best)
    strike = _contract_strike(best)
    meta = _contract_selection_meta(
        ticker=ticker,
        cp=cp,
        symbol=best_sym,
        current_price=current_price,
        expiry_str=expiry_str,
        strike=strike,
        selected_delta=None,
        method="atm_fallback",
    )
    logger.info("[%s] ATM fallback: selected %s %s (exp=%s)", ticker, cp, best_sym, expiry_str)
    return best_sym, meta


# ---------------------------------------------------------------------------
# Cross-sectional scan batching
# ---------------------------------------------------------------------------

class _ScanBatcher:
    """Coalesce one 30m close wave and scan it off the bar-consumer thread."""

    def __init__(
        self,
        callback: Callable[[list[str]], None],
        *,
        debounce_seconds: float = 0.35,
        max_wait_seconds: float = 3.0,
    ) -> None:
        self._callback = callback
        self._debounce = max(0.01, float(debounce_seconds))
        self._max_wait = max(self._debounce, float(max_wait_seconds))
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="swing-scan-batcher")
        self._thread.start()

    def submit(self, ticker: str) -> None:
        if ticker and not self._stop.is_set():
            self._queue.put_nowait(str(ticker).upper())

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if first is None:
                break
            batch = {first}
            started = time.monotonic()
            deadline = started + self._debounce
            while not self._stop.is_set():
                remaining = min(deadline, started + self._max_wait) - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self._stop.set()
                    break
                batch.add(item)
                deadline = time.monotonic() + self._debounce
            if batch and not self._stop.is_set():
                try:
                    self._callback(sorted(batch))
                except Exception as exc:
                    logger.error("Batched swing scan failed: %s", exc, exc_info=True)


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
        auto_flatten_assigned_equities: bool = True,
        real_account_policy_enabled: bool | None = None,
        real_account_policy_state_path: str | None = None,
        catalyst_tilt_enabled: bool = True,
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
        # Live news tilt: re-ranks signals by fresh catalyst strength from the
        # intraday poller's ledger. Neutral (no effect) until the ledger exists.
        catalyst_signal = LiveCatalystSignal() if catalyst_tilt_enabled else None
        # OOF long+short ranker scanner (v2). Drop-in: same Signal surface. To revert to the
        # classifier, swap RankerSwingScanner -> SwingScanner(..., model_path=MODEL_PATH).
        self._scanner = RankerSwingScanner(
            self._fb,
            max_entries_per_bar=max_entries_per_bar,
            catalyst_signal=catalyst_signal,
        )
        self._client = AlpacaOptionsClient(env_file=env_file)
        self._real_policy = RealAccountBookkeeper(
            real_account_policy_from_env(
                enabled=real_account_policy_enabled,
                state_path=real_account_policy_state_path,
            )
        )
        self._risk_policy = RiskProfilePolicy(risk_profile_policy_from_env())
        self._signal_policy = SignalPolicyLayer(signal_policy_from_env())
        self._pos_mgr = SwingPositionManager(
            self._client,
            dry_run=dry_run,
            event_sink=self._emit,
            auto_flatten_assigned_equities=auto_flatten_assigned_equities,
        )

        # Per-ticker bar aggregators.  These align to wall-clock buckets instead
        # of simple counts, so a restart or skipped stale bars cannot shift the
        # 5m/30m schedule.
        self._acc_5m:  dict[str, _BarAccumulator] = defaultdict(lambda: _BarAccumulator(5))
        self._acc_30m: dict[str, _BarAccumulator] = defaultdict(lambda: _BarAccumulator(30, min_bars=BARS_PER_30M))

        # Rolling 5m bar history per ticker (60 bars ≈ 5h of market data).
        # Used to seed position charts with pre-entry context.
        self._buf_5m: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))

        # Confirmation watchers: ticker → _ConfirmState
        self._confirming: dict[str, _ConfirmState] = {}
        self._signal_policy_decisions: dict[str, SignalPolicyDecision] = {}

        # If an external queue is provided (e.g. from SharedBarStream) use it directly
        # and skip creating / managing an AlpacaBarStreamer in start().
        self._bar_queue: queue.Queue = bar_queue if bar_queue is not None else queue.Queue(maxsize=50_000)
        self._external_queue: bool = bar_queue is not None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._streamer: AlpacaBarStreamer | None = None
        self._bar_count_total = 0
        self._raw_bar_count_total = 0
        self._rth_bar_count_total = 0
        self._non_rth_bar_count_total = 0
        self._five_min_bar_count_total = 0
        self._thirty_min_bar_count_total = 0
        self._scan_count_total = 0
        self._last_bar_ticker: str | None = None
        self._last_heartbeat_wall = 0.0
        self._heartbeat_window_counts: dict[str, int] = defaultdict(int)
        self._recent_symbol_seen: dict[str, float] = {}
        self._last_bar_ts: datetime | None = None
        self._last_bar_lag_secs: int | None = None
        self._last_queue_size: int | None = None
        self._dropped_stale_bars = 0
        self._last_backlog_event_wall = 0.0
        self._last_broker_reconcile_wall = 0.0
        self._last_broker_reconcile_ts: str | None = None
        self._last_broker_reconcile_ok: bool | None = None
        self._last_broker_reconcile_error: str | None = None
        self._scan_batcher = _ScanBatcher(self._scan_tickers)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit(self, kind: str, payload: dict) -> None:
        self._on_internal_event(kind, payload)
        if self._sink is None:
            return
        try:
            self._sink(kind, payload)
        except Exception as exc:
            logger.warning("event_sink raised on %s: %s", kind, exc)

    def _on_internal_event(self, kind: str, payload: dict) -> None:
        if self._dry_run or not getattr(self, "_real_policy", None) or not self._real_policy.enabled:
            return
        if kind not in {"position_closed", "position_close_abandoned", "broker_position_missing"}:
            return
        try:
            self._real_policy.mark_position_closed(
                ticker=str(payload.get("ticker") or ""),
                option_symbol=str(payload.get("option_symbol") or ""),
                entry_premium=payload.get("option_entry_price"),
                qty=payload.get("qty"),
            )
        except Exception as exc:
            logger.warning("real-account policy close bookkeeping failed: %s", exc)

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
            "raw_bar_count": int(self._raw_bar_count_total),
            "rth_bar_count": int(self._rth_bar_count_total),
            "non_rth_bar_count": int(self._non_rth_bar_count_total),
            "five_min_bar_count": int(self._five_min_bar_count_total),
            "thirty_min_bar_count": int(self._thirty_min_bar_count_total),
            "scan_count": int(self._scan_count_total),
            "last_bar_ticker": self._last_bar_ticker,
            "last_bar_ts": _utc_iso(self._last_bar_ts) if self._last_bar_ts else None,
            "last_bar_lag_secs": self._last_bar_lag_secs,
            "queue_size": self._last_queue_size,
            "dropped_stale_bars": int(self._dropped_stale_bars),
            "confirming_count": len(self._confirming),
            "open_positions_count": len(self._pos_mgr.open_tickers),
            "last_broker_reconcile_ts": self._last_broker_reconcile_ts,
            "last_broker_reconcile_ok": self._last_broker_reconcile_ok,
            "last_broker_reconcile_error": self._last_broker_reconcile_error,
            "real_account_policy": self._real_policy.snapshot(),
            "signal_policy": self._signal_policy.snapshot(),
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
        self._emit("risk_profile_policy_config", self._risk_policy.snapshot())
        self._emit("signal_policy_config", self._signal_policy.snapshot())

        if self._stop_event.is_set():
            self._emit("stopped", {"reason": "stop requested during warmup"})
            return

        self._sync_positions_from_broker()
        self._run_startup_scan_from_warmup()
        self._scan_batcher.start()

        if self._external_queue:
            # Bar queue is owned by the caller (e.g. SharedBarStream in combined_server).
            # We don't create or stop the Alpaca streamer here.
            logger.info("Using shared bar stream for %d symbols.", len(self._stream_symbols))
            self._emit("stream_started", {"symbols": len(self._stream_symbols), "shared": True})
            try:
                self._process_loop()
            finally:
                self._scan_batcher.stop()
                self._emit("stopped", {"reason": "loop_exit"})
        else:
            from alpaca.data.enums import DataFeed
            streamer = AlpacaBarStreamer(
                symbols=self._stream_symbols,
                feed=DataFeed.IEX,
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
                self._scan_batcher.stop()
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
        if getattr(self, "_scan_batcher", None) is not None:
            self._scan_batcher.stop()
        # Best-effort wake the queue.get
        try:
            self._bar_queue.put_nowait({"_sentinel": True})
        except Exception:
            pass

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _sync_positions_from_broker(self) -> None:
        if self._dry_run:
            self._emit("broker_sync", {"synced": True, "restored": 0, "ignored": 0, "simulated": True})
            return
        try:
            result = self._pos_mgr.sync_from_broker(
                universe=self._universe,
                price_lookup=self._latest_underlying_close,
                atr_lookup=self._fb.get_atr,
            )
            logger.info(
                "Broker sync complete: restored=%s ignored=%s",
                result.get("restored"),
                result.get("ignored"),
            )
            for pos in result.get("positions") or []:
                if not isinstance(pos, dict):
                    continue
                ticker = str(pos.get("ticker", "")).upper()
                self._emit("position_chart_seed", {
                    "ticker": ticker or pos.get("ticker"),
                    "direction": int(pos.get("direction", 0) or 0),
                    "entry_price": float(pos.get("entry_price", float("nan"))),
                    "entry_time": int(datetime.now(timezone.utc).timestamp()),
                    "sl_price": pos.get("sl_price"),
                    "pre_entry_bars": [],
                    "restored": True,
                })
        except Exception as exc:
            logger.warning("Broker sync failed: %s", exc)
            self._emit("broker_sync", {"synced": False, "error": str(exc)})

    def _maybe_reconcile_broker(self, *, reason: str, force: bool = False) -> None:
        if self._dry_run:
            return
        now_et = datetime.now(_ET)
        if not force and not self._within_log_window(now_et):
            return
        now = time.monotonic()
        if not force and self._last_broker_reconcile_wall and (now - self._last_broker_reconcile_wall) < _BROKER_RECONCILE_SECS:
            return
        self._last_broker_reconcile_wall = now
        self._last_broker_reconcile_ts = _utc_iso()
        try:
            result = self._pos_mgr.reconcile_with_broker(
                universe=self._universe,
                price_lookup=self._latest_underlying_close,
                atr_lookup=self._fb.get_atr,
                reason=reason,
            )
            self._last_broker_reconcile_ok = bool(result.get("ok", False))
            self._last_broker_reconcile_error = None
            restored = result.get("positions") or []
            for pos in restored:
                if not isinstance(pos, dict):
                    continue
                ticker = str(pos.get("ticker", "")).upper()
                self._emit("position_chart_seed", {
                    "ticker": ticker or pos.get("ticker"),
                    "direction": int(pos.get("direction", 0) or 0),
                    "entry_price": float(pos.get("entry_price", float("nan"))),
                    "entry_time": int(datetime.now(timezone.utc).timestamp()),
                    "sl_price": pos.get("sl_price"),
                    "pre_entry_bars": [],
                    "restored": True,
                    "reconciled": True,
                })
        except Exception as exc:
            self._last_broker_reconcile_ok = False
            self._last_broker_reconcile_error = str(exc)
            logger.warning("Broker reconcile failed: %s", exc)
            self._emit("broker_reconcile", {"ok": False, "reason": reason, "error": str(exc)})

    def _latest_underlying_close(self, ticker: str) -> float | None:
        bar = self._fb.get_last_bar(ticker)
        if not bar:
            return None
        try:
            return float(bar.get("close"))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        _last_bar_wall: float = time.monotonic()
        _stale_warned: bool = False
        _rth_anchor_set: bool = False

        while not self._stop_event.is_set():
            try:
                bar = self._bar_queue.get(timeout=2.0)
            except queue.Empty:
                # Health-check: warn if no bar received for >5 min during RTH
                now_et = datetime.now(_ET)
                self._maybe_reconcile_broker(reason="idle")
                if _RTH_START <= now_et.time() < _RTH_END:
                    if not _rth_anchor_set:
                        # Anchor the stale timer to the RTH open so the expected
                        # overnight/pre-open gap (no bars until 09:30) doesn't fire
                        # a false "stream may be stale" warning. Re-armed each day
                        # once the clock leaves RTH (else branch below).
                        _rth_anchor_set = True
                        _last_bar_wall = time.monotonic()
                        _stale_warned = False
                        continue
                    elapsed = time.monotonic() - _last_bar_wall
                    if elapsed >= _STREAM_STALE_SECS and not _stale_warned:
                        _stale_warned = True
                        logger.warning(
                            "No bars received for %.0fs during RTH — stream may be stale. "
                            "Check Alpaca data subscription or restart the server.",
                            elapsed,
                        )
                        self._emit("stream_stale", {
                            "elapsed_secs": round(elapsed),
                            "hint": "No bars received during market hours. Stream may have dropped. Restart to reconnect.",
                        })
                else:
                    _rth_anchor_set = False
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
            if self._bar_after_log_cutoff(bar):
                continue
            self._raw_bar_count_total += 1
            self._last_bar_ticker = ticker
            self._heartbeat_window_counts[ticker] += 1
            self._recent_symbol_seen[ticker] = time.monotonic()
            self._last_queue_size = self._queue_size()
            lag_secs = self._bar_lag_secs(bar)
            if lag_secs is not None:
                self._last_bar_lag_secs = int(round(lag_secs))
            if self._external_queue and self._bar_is_too_stale(lag_secs):
                bar = self._drop_stale_external_backlog(bar)
                if bar is None:
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
                if self._bar_after_log_cutoff(bar):
                    continue
                self._raw_bar_count_total += 1
                self._last_bar_ticker = ticker
                self._heartbeat_window_counts[ticker] += 1
                self._recent_symbol_seen[ticker] = time.monotonic()
                self._last_queue_size = self._queue_size()
                lag_secs = self._bar_lag_secs(bar)
                if lag_secs is not None:
                    self._last_bar_lag_secs = int(round(lag_secs))
            if self._bar_is_too_stale(lag_secs):
                self._dropped_stale_bars += 1
                self._reset_ticker_accumulators(ticker)
                self._emit_backlog_event(ticker=ticker, bar=bar, lag_secs=lag_secs)
                self._maybe_emit_heartbeat(reason="stale_drop")
                continue
            self._bar_count_total += 1
            _last_bar_wall = time.monotonic()
            _stale_warned = False
            ts = bar.get("timestamp")
            if isinstance(ts, datetime):
                self._last_bar_ts = ts
            if self._on_1m_bar(ticker, bar):
                self._rth_bar_count_total += 1
            else:
                self._non_rth_bar_count_total += 1
            self._maybe_reconcile_broker(reason="bar")
            self._maybe_emit_heartbeat(reason="bar")

    def _queue_size(self) -> int | None:
        try:
            return int(self._bar_queue.qsize())
        except Exception:
            return None

    @staticmethod
    def _bar_lag_secs(bar: dict) -> float | None:
        ts = bar.get("timestamp")
        if not isinstance(ts, datetime):
            return None
        ts_utc = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts_utc).total_seconds())

    @staticmethod
    def _bar_is_too_stale(lag_secs: float | None) -> bool:
        if lag_secs is None or lag_secs < _STALE_BAR_LAG_SECS:
            return False
        return is_regular_trading_time(datetime.now(_ET))

    def _drop_stale_external_backlog(self, first_bar: dict) -> dict | None:
        current = first_bar
        last_stale_bar = first_bar
        last_stale_ticker = str(first_bar.get("symbol", first_bar.get("ticker", ""))).upper()
        last_stale_lag = self._bar_lag_secs(first_bar)
        dropped = 0

        while True:
            if current.get("_sentinel"):
                break
            ticker = str(current.get("symbol", current.get("ticker", ""))).upper()
            lag_secs = self._bar_lag_secs(current)
            if not self._bar_is_too_stale(lag_secs):
                break
            dropped += 1
            self._dropped_stale_bars += 1
            if ticker:
                self._reset_ticker_accumulators(ticker)
            last_stale_bar = current
            last_stale_ticker = ticker
            last_stale_lag = lag_secs
            try:
                current = self._bar_queue.get_nowait()
            except queue.Empty:
                current = None
                break
            except Exception:
                current = None
                break

        if dropped:
            self._last_queue_size = self._queue_size()
            self._emit_backlog_event(
                ticker=last_stale_ticker,
                bar=last_stale_bar,
                lag_secs=last_stale_lag,
            )
            self._maybe_emit_heartbeat(reason="stale_drop")
        return current

    @staticmethod
    def _bar_after_log_cutoff(bar: dict) -> bool:
        if not SwingLiveRunner._within_log_window(datetime.now(_ET)):
            return True
        ts = bar.get("timestamp")
        if isinstance(ts, datetime):
            bar_et = ts.astimezone(_ET) if ts.tzinfo else ts.replace(tzinfo=_ET)
            return bar_et.time() >= _LOG_END
        return False

    @staticmethod
    def _within_log_window(now_et: datetime) -> bool:
        return is_log_window(now_et)

    def _reset_ticker_accumulators(self, ticker: str) -> None:
        acc5 = self._acc_5m.get(ticker)
        if acc5 is not None:
            acc5.reset()
        acc30 = self._acc_30m.get(ticker)
        if acc30 is not None:
            acc30.reset()

    def _emit_backlog_event(self, *, ticker: str, bar: dict, lag_secs: float | None) -> None:
        now = time.monotonic()
        if now - self._last_backlog_event_wall < _BACKLOG_EVENT_THROTTLE_SECS:
            return
        self._last_backlog_event_wall = now
        lag = int(round(lag_secs or 0.0))
        qsize = self._queue_size()
        ts = bar.get("timestamp")
        ts_iso = _utc_iso(ts) if isinstance(ts, datetime) else None
        logger.warning(
            "Dropping stale swing bar: ticker=%s ts=%s lag=%ss queue=%s dropped=%d",
            ticker, ts_iso, lag, qsize, self._dropped_stale_bars,
        )
        self._emit("stream_backlog", {
            "ticker": ticker,
            "bar_ts": ts_iso,
            "lag_secs": lag,
            "queue_size": qsize,
            "dropped_stale_bars": int(self._dropped_stale_bars),
            "hint": "Swing queue is behind real time; stale bars are being skipped until fresh data catches up.",
        })

    def _maybe_emit_heartbeat(self, *, reason: str) -> None:
        now = time.monotonic()
        if self._last_heartbeat_wall and (now - self._last_heartbeat_wall) < _HEARTBEAT_SECS:
            return
        self._last_heartbeat_wall = now
        window_counts = dict(self._heartbeat_window_counts)
        self._heartbeat_window_counts.clear()
        top_symbols = sorted(window_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        stream_symbol_set = set(self._stream_symbols)
        window_seen = set(window_counts) & stream_symbol_set
        recent_cutoff = now - 300.0
        recent_seen = {
            ticker for ticker, seen_at in self._recent_symbol_seen.items()
            if seen_at >= recent_cutoff and ticker in stream_symbol_set
        }
        missing_recent = sorted(stream_symbol_set - recent_seen)[:20]
        stream_symbol_count = len(stream_symbol_set)
        payload = self.snapshot_meta()
        payload.update({
            "reason": reason,
            "window_unique_symbols": len(window_counts),
            "window_bar_count": int(sum(window_counts.values())),
            "window_stream_symbols": stream_symbol_count,
            "window_coverage_pct": (
                round((len(window_seen) / stream_symbol_count) * 100.0, 2)
                if stream_symbol_count else None
            ),
            "recent_unique_symbols": len(recent_seen),
            "recent_coverage_pct": (
                round((len(recent_seen) / stream_symbol_count) * 100.0, 2)
                if stream_symbol_count else None
            ),
            "recent_missing_symbols": max(0, stream_symbol_count - len(recent_seen)),
            "recent_missing_sample": missing_recent,
            "window_top_symbols": [
                {"ticker": ticker, "bars": int(count)}
                for ticker, count in top_symbols
            ],
        })
        logger.info(
            "Swing stream heartbeat: raw=%d accepted=%d rth=%d 5m=%d 30m=%d scans=%d "
            "last=%s ts=%s lag=%s queue=%s unique=%d recent=%d/%d",
            self._raw_bar_count_total,
            self._bar_count_total,
            self._rth_bar_count_total,
            self._five_min_bar_count_total,
            self._thirty_min_bar_count_total,
            self._scan_count_total,
            self._last_bar_ticker,
            payload.get("last_bar_ts"),
            self._last_bar_lag_secs,
            self._last_queue_size,
            len(window_counts),
            len(recent_seen),
            stream_symbol_count,
        )
        self._emit("stream_heartbeat", payload)

    # ------------------------------------------------------------------
    # Startup scan from warmed 30m cache
    # ------------------------------------------------------------------

    def _run_startup_scan_from_warmup(self) -> None:
        now_et = datetime.now(_ET)
        if not (_CONFIRM_START <= now_et.time() <= _SCAN_END_TS):
            self._emit("startup_scan_skipped", {
                "reason": "outside_scan_window",
                "now_et": now_et.isoformat(),
            })
            return

        latest_ts: datetime | None = None
        eligible: list[str] = []
        stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=75)
        for ticker in self._all_tickers:
            bar = self._fb.get_last_bar(ticker)
            if not bar:
                continue
            ts = bar.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            ts_utc = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if ts_utc < stale_cutoff:
                continue
            if latest_ts is None or ts_utc > latest_ts:
                latest_ts = ts_utc
            eligible.append(ticker)

        if not eligible:
            self._emit("startup_scan_skipped", {
                "reason": "no_fresh_warmup_bars",
                "freshness_cutoff": stale_cutoff.isoformat(),
            })
            return

        with self._lock:
            busy = set(self._confirming.keys()) | self._pos_mgr.open_tickers

        signals = self._scanner.scan(eligible, skip_tickers=busy)
        self._scan_count_total += 1
        self._emit("startup_scan", {
            "candidate_count": len(eligible),
            "latest_bar_ts": _utc_iso(latest_ts),
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
        self._accept_signals(signals)

    def _accept_signals(self, signals: list[Signal]) -> None:
        spy_p_long, spy_p_short = None, None
        filtered_signals = []
        for sig in signals:
            signal_decision = self._signal_policy.evaluate_signal(sig)
            self._signal_policy_decisions[sig.ticker] = signal_decision
            self._emit("signal_policy_decision", {
                "ticker": sig.ticker,
                "direction": int(sig.direction),
                "p_dir": float(sig.p_dir),
                "ev_score": float(sig.ev_score),
                "decision": signal_decision.to_dict(),
                "enforced": bool(self._signal_policy.config.enforce),
            })
            if self._signal_policy.config.enforce and signal_decision.action == "BLOCK":
                logger.info("[%s] VETOED by signal policy: %s", sig.ticker, signal_decision.reason)
                self._emit("entry_skipped", {
                    "ticker": sig.ticker,
                    "direction": int(sig.direction),
                    "reason": signal_decision.reason,
                    "signal_policy": signal_decision.to_dict(),
                })
                continue
            # Long-only gate: applies unconditionally, not just under the challenger
            # policy. See _ALLOW_SHORT_ENTRIES for the evidence.
            if sig.direction < 0 and not _ALLOW_SHORT_ENTRIES:
                logger.info("[%s] VETOED: long-only gate (put entries disabled)", sig.ticker)
                self._emit("entry_skipped", {
                    "ticker": sig.ticker,
                    "direction": int(sig.direction),
                    "reason": "short_entries_disabled_long_only",
                })
                continue
            if _challenger_policy_enabled() and sig.direction < 0 and not _CHALLENGER_ALLOW_SHORT_ENTRIES:
                logger.info("[%s] VETOED by live option policy (short/put entries disabled)", sig.ticker)
                self._emit("entry_skipped", {
                    "ticker": sig.ticker,
                    "direction": int(sig.direction),
                    "reason": "short_entries_disabled",
                })
                continue
            if (
                _challenger_policy_enabled()
                and sig.direction > 0
                and sig.ticker.upper() in _CHALLENGER_BLOCKED_LONG_ENTRY_TICKERS
            ):
                logger.info("[%s] VETOED by live option policy (call ticker blocked)", sig.ticker)
                self._emit("entry_skipped", {
                    "ticker": sig.ticker,
                    "direction": int(sig.direction),
                    "reason": "long_ticker_blocked",
                })
                continue
            risk_decision = self._risk_policy.evaluate(sig)
            self._emit("risk_profile_policy_decision", {
                "ticker": sig.ticker,
                "direction": int(sig.direction),
                "p_dir": float(sig.p_dir),
                "ev_score": float(sig.ev_score),
                "allowed": bool(risk_decision.allowed),
                "reason": risk_decision.reason,
                "profile": risk_decision.profile,
                "details": risk_decision.details,
            })
            if not risk_decision.allowed:
                logger.info("[%s] VETOED by risk profile policy: %s", sig.ticker, risk_decision.reason)
                self._emit("entry_skipped", {
                    "ticker": sig.ticker,
                    "direction": int(sig.direction),
                    "reason": risk_decision.reason,
                    "risk_profile_policy": {
                        "profile": risk_decision.profile,
                        "details": risk_decision.details,
                    },
                })
                continue
            spy_min = sig.config.spy_min if sig.config else 0.0
            if spy_min > 0:
                if spy_p_long is None:
                    spy_p_long, spy_p_short = self._scanner.get_directional_p("SPY")
                spy_p = spy_p_long if sig.direction == 1 else spy_p_short
                if math.isnan(spy_p) or spy_p < spy_min:
                    logger.info("[%s] VETOED by SPY filter (spy_p=%.3f < min=%.2f)",
                                sig.ticker, spy_p if not math.isnan(spy_p) else -1, spy_min)
                    continue
            filtered_signals.append(sig)

        accepted_signals: list[Signal] = []
        with self._lock:
            busy_now = set(self._confirming.keys()) | self._pos_mgr.open_tickers
            for sig in filtered_signals:
                if sig.ticker in busy_now:
                    continue
                self._confirming[sig.ticker] = _ConfirmState(signal=sig)
                busy_now.add(sig.ticker)
                accepted_signals.append(sig)

        for sig in accepted_signals:
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
                "risk_profile_policy": self._risk_policy.evaluate(sig).details,
                "signal_policy": self._signal_policy_decisions.get(sig.ticker).to_dict()
                if self._signal_policy_decisions.get(sig.ticker) is not None
                else None,
            })

    # ------------------------------------------------------------------
    # 1m → 5m → 30m aggregation
    # ------------------------------------------------------------------

    def _on_1m_bar(self, ticker: str, bar: dict) -> bool:
        # Drop pre-market and post-market bars — only aggregate RTH bars so the
        # 30m accumulator aligns cleanly with the 9:30/10:00/.../15:30 schedule.
        ts = bar.get("timestamp")
        if hasattr(ts, "astimezone") and not is_regular_trading_time(ts):
            return False

        bar5 = self._acc_5m[ticker].push(bar)
        if bar5 is not None:
            self._on_5m_bar(ticker, bar5)
        return True

    def _on_5m_bar(self, ticker: str, bar5: dict) -> None:
        self._five_min_bar_count_total += 1
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
        if should_check_confirmation(ts):
            self._check_confirmation(ticker, bar5)

        # Aggregate 5m → 30m
        bar30 = self._acc_30m[ticker].push(bar5)
        if bar30 is not None:
            self._thirty_min_bar_count_total += 1
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
        if not should_scan_after_30m_close(ts30):
            return

        # Queue the ticker for one vectorized cross-sectional scan after this
        # 30m wave settles.  Previously the bar thread ran ~925 separate model
        # scans here, blocking one-minute bar consumption for 6-8 minutes at
        # every boundary and eventually dropping stale bars.
        self._scan_batcher.submit(ticker)

    def _scan_tickers(self, tickers: list[str]) -> None:
        """Run one cross-sectional scan without blocking the bar-consumer loop."""
        if self._stop_event.is_set() or not tickers:
            return
        # Important: never call the dashboard event sink while holding
        # self._lock. The sink refreshes runner snapshots, which also acquire
        # this lock.
        with self._lock:
            busy = set(self._confirming.keys()) | self._pos_mgr.open_tickers
        active = sorted(set(tickers))
        signals = self._scanner.scan(active, skip_tickers=busy)
        self._scan_count_total += len(active)
        self._emit("scan", {
            "tickers": active,
            "ticker_count": len(active),
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

        self._accept_signals(signals)

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

        confirmed = confirmation_breakout(
            direction=sig.direction,
            ref_high=sig.ref_high,
            ref_low=sig.ref_low,
            bar=bar5,
        )

        if confirmed:
            candle_metrics = _bar_shape_metrics(bar5, atr=sig.atr)
            self._emit("confirmation", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "bars_watched": int(state.bars_watched),
                "close": float(c),
                **candle_metrics,
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
        now_et = datetime.now(_ET)
        bucket = _entry_bucket(now_et)
        if (
            _challenger_policy_enabled()
            and sig.direction > 0
            and bucket in _CHALLENGER_BLOCKED_LONG_ENTRY_BUCKETS
        ):
            logger.info("[%s] entry skipped by live option policy bucket=%s", ticker, bucket)
            self._emit("entry_skipped", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "reason": "long_entry_time_blocked",
                "entry_price": entry_price,
                "entry_bucket": bucket,
            })
            return
        contract_ref_date = _entry_contract_ref_date(now_et)
        if contract_ref_date != now_et.date():
            logger.info(
                "[%s] entry time requires later expiry; selecting next listed expiry >= %s",
                ticker,
                contract_ref_date.isoformat(),
            )

        option_symbol, option_selection_meta = _select_contract(
            self._client, ticker, sig.direction, entry_price, contract_ref_date
        )
        if not option_symbol:
            logger.warning("[%s] no option contract found — skipping entry", ticker)
            self._emit("entry_skipped", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "reason": "no_contract_found",
                "entry_price": entry_price,
                "option_selection": option_selection_meta,
            })
            return

        # Dealer-positioning structural gate. Nearest-expiry swing -> daily_week
        # scope, proximity recomputed against the LIVE entry price (absolute
        # strikes). Fails open on missing/stale data. The swing has no share
        # lifecycle, so an enforced veto SKIPS the entry (vs the 4H modules,
        # which fall back to shares). Observe-only unless DEALER_GATE_ENABLED.
        dealer_verdict = evaluate_dealer_gate(
            ticker, int(sig.direction), entry_price, SCOPE_NEAREST
        )
        if dealer_verdict.vetoed and dealer_gate_enabled():
            logger.info("[%s] entry skipped by dealer gate: %s", ticker, dealer_verdict.reason)
            self._emit("entry_skipped", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "reason": f"dealer_veto:{dealer_verdict.reason}",
                "entry_price": entry_price,
                "option_symbol": option_symbol,
                "dealer_gate": dealer_verdict.to_dict(),
            })
            return

        order_resp: Any = None
        order_error: str | None = None
        verification: dict[str, Any] | None = None
        limit_prices, quote_meta = _entry_buy_limit_ladder(
            self._client,
            symbol=option_symbol,
            attempts=_ENTRY_ORDER_ATTEMPTS,
        )
        qty = _entry_contracts_for_quote(quote_meta)
        option_entry_meta = {
            **(option_selection_meta or {}),
            "confirmation_bar": _bar_to_event(conf_bar),
            "confirmation_metrics": _bar_shape_metrics(conf_bar, atr=sig.atr),
            "entry_quote": quote_meta,
            "entry_limit_prices": limit_prices,
            "risk_profile_policy": self._risk_policy.evaluate(sig).details,
            "dealer_gate": dealer_verdict.to_dict(),
        }
        signal_policy_decision = self._signal_policy_decisions.get(ticker) or self._signal_policy.evaluate_signal(sig)
        entry_policy_decision = self._signal_policy.with_entry_context(
            signal_policy_decision,
            signal=sig,
            option_meta=option_selection_meta or {},
            quote_meta=quote_meta or {},
        )
        self._signal_policy_decisions[ticker] = entry_policy_decision
        option_entry_meta["signal_policy"] = entry_policy_decision.to_dict()
        self._emit("signal_policy_entry_decision", {
            "ticker": ticker,
            "direction": int(sig.direction),
            "option_symbol": option_symbol,
            "decision": entry_policy_decision.to_dict(),
            "enforced": bool(self._signal_policy.config.enforce),
            "apply_sizing": bool(self._signal_policy.config.apply_sizing),
        })
        if self._signal_policy.config.enforce and entry_policy_decision.action == "BLOCK":
            logger.info("[%s] entry skipped by signal policy: %s", ticker, entry_policy_decision.reason)
            self._emit("entry_skipped", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "reason": entry_policy_decision.reason,
                "entry_price": entry_price,
                "option_symbol": option_symbol,
                "option_entry_meta": option_entry_meta,
                "signal_policy": entry_policy_decision.to_dict(),
            })
            return
        if not limit_prices:
            logger.warning("[%s] no quote for buy limit price; skipping entry symbol=%s", ticker, option_symbol)
            self._emit("entry_skipped", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "reason": "no_quote_for_buy_limit",
                "entry_price": entry_price,
                "option_symbol": option_symbol,
                "option_entry_meta": option_entry_meta,
            })
            return
        spread_ok, spread_reason, spread_pct = _entry_quote_spread_ok(quote_meta)
        if not spread_ok:
            logger.info(
                "[%s] entry skipped by option spread gate: %s symbol=%s spread_pct_mid=%s max=%.3f",
                ticker,
                spread_reason,
                option_symbol,
                f"{spread_pct:.4f}" if math.isfinite(spread_pct) else "nan",
                _MAX_ENTRY_SPREAD_PCT_MID,
            )
            self._emit("entry_skipped", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "reason": spread_reason,
                "entry_price": entry_price,
                "option_symbol": option_symbol,
                "option_entry_meta": option_entry_meta,
                "spread_pct_mid": spread_pct if math.isfinite(spread_pct) else None,
                "max_spread_pct_mid": _MAX_ENTRY_SPREAD_PCT_MID,
            })
            return
        account_snapshot = None
        if self._real_policy.enabled and not self._dry_run:
            try:
                account_resp = self._client.get_account()
                account_snapshot = account_resp if isinstance(account_resp, dict) else None
            except Exception as exc:
                logger.warning("[%s] real account policy account fetch failed: %s", ticker, exc)
                self._emit("entry_skipped", {
                    "ticker": ticker,
                    "direction": int(sig.direction),
                    "reason": "real_policy_account_unavailable",
                    "entry_price": entry_price,
                    "option_symbol": option_symbol,
                    "option_entry_meta": option_entry_meta,
                    "error": str(exc),
                })
                return
        real_decision = self._real_policy.evaluate_entry(
            signal=sig,
            option_symbol=option_symbol,
            option_meta=option_selection_meta or {},
            quote_meta=quote_meta or {},
            limit_prices=limit_prices,
            open_positions_count=len(self._pos_mgr.open_tickers),
            account=account_snapshot,
            entry_quality=option_entry_meta.get("confirmation_metrics") or {},
        )
        option_entry_meta["real_account_policy"] = {
            "enabled": self._real_policy.enabled,
            "allowed": bool(real_decision.allowed),
            "reason": real_decision.reason,
            "qty": int(real_decision.qty),
            "premium_at_risk": float(real_decision.premium_at_risk),
            "details": real_decision.details,
        }
        if not real_decision.allowed:
            logger.info("[%s] entry skipped by real-account policy: %s", ticker, real_decision.reason)
            self._emit("entry_skipped", {
                "ticker": ticker,
                "direction": int(sig.direction),
                "reason": real_decision.reason,
                "entry_price": entry_price,
                "option_symbol": option_symbol,
                "option_entry_meta": option_entry_meta,
                "real_account_policy": option_entry_meta["real_account_policy"],
            })
            return
        if self._real_policy.enabled:
            qty = int(real_decision.qty)
        elif self._signal_policy.config.apply_sizing:
            qty = int(entry_policy_decision.recommended_qty)
            if qty < 1:
                logger.info("[%s] entry skipped by signal policy sizing qty=0", ticker)
                self._emit("entry_skipped", {
                    "ticker": ticker,
                    "direction": int(sig.direction),
                    "reason": "signal_policy_size_zero",
                    "entry_price": entry_price,
                    "option_symbol": option_symbol,
                    "option_entry_meta": option_entry_meta,
                    "signal_policy": entry_policy_decision.to_dict(),
                })
                return
        if not self._dry_run:
            for attempt, limit_price in enumerate(limit_prices, start=1):
                try:
                    order_resp = self._client.submit_option_order(
                        symbol=option_symbol,
                        qty=qty,
                        side="buy",
                        order_type="limit",
                        time_in_force="day",
                        limit_price=limit_price,
                    )
                    logger.info(
                        "[%s] buy limit order submitted limit=%.2f attempt=%d/%d: %s",
                        ticker,
                        limit_price,
                        attempt,
                        len(limit_prices),
                        order_resp,
                    )
                except Exception as exc:
                    order_error = str(exc)
                    logger.error("[%s] buy order FAILED: %s", ticker, exc)
                    self._emit("order_failed", {
                        "ticker": ticker,
                        "side": "buy",
                        "option_symbol": option_symbol,
                        "qty": qty,
                        "order_type": "limit",
                        "limit_price": limit_price,
                        "attempt": attempt,
                        "attempts": len(limit_prices),
                        "error": order_error,
                        "option_entry_meta": option_entry_meta,
                    })
                    return
                verification = _verify_entry_order(
                    self._client,
                    submitted_resp=order_resp if isinstance(order_resp, dict) else {},
                    symbol=option_symbol,
                )
                self._emit("order_submitted", {
                    "ticker": ticker,
                    "side": "buy",
                    "option_symbol": option_symbol,
                    "qty": qty,
                    "order_type": "limit",
                    "limit_price": limit_price,
                    "attempt": attempt,
                    "attempts": len(limit_prices),
                    "entry_price": entry_price,
                    "option_entry_meta": option_entry_meta,
                    "response": _safe_response(order_resp),
                    "verification": verification,
                })
                if verification.get("verified", False):
                    break
                order_id = str(verification.get("order_id") or "").strip()
                can_retry = bool(verification.get("retryable")) and attempt < len(limit_prices)
                if can_retry and order_id:
                    _cancel_entry_order(self._client, order_id=order_id)
                    logger.info(
                        "[%s] buy limit not filled; retrying at next limit symbol=%s status=%s limit=%.2f",
                        ticker,
                        option_symbol,
                        verification.get("status"),
                        limit_price,
                    )
                    continue
                if can_retry:
                    continue
                break
            if verification is None:
                return
            if not verification.get("verified", False):
                order_id = str(verification.get("order_id") or "").strip()
                if order_id and bool(verification.get("retryable")):
                    _cancel_entry_order(self._client, order_id=order_id)
                logger.warning(
                    "[%s] buy order not verified; skipping local position open symbol=%s status=%s via=%s",
                    ticker,
                    option_symbol,
                    verification.get("status"),
                    verification.get("via"),
                )
                self._emit("entry_skipped", {
                    "ticker": ticker,
                    "direction": int(sig.direction),
                    "reason": "buy_order_not_verified",
                    "entry_price": entry_price,
                    "option_symbol": option_symbol,
                    "option_entry_meta": option_entry_meta,
                    "verification": verification,
                })
                return
        else:
            logger.info(
                "[DRY RUN] [%s] would BUY %d × %s limit ladder=%s at entry ~%.2f",
                ticker, qty, option_symbol, limit_prices, entry_price,
            )
            self._emit("order_dry_run", {
                "ticker": ticker,
                "side": "buy",
                "option_symbol": option_symbol,
                "qty": qty,
                "order_type": "limit",
                "limit_prices": limit_prices,
                "entry_price": entry_price,
                "option_entry_meta": option_entry_meta,
            })

        entry_time = datetime.now(timezone.utc)
        verified_order = verification.get("order") if verification else None
        option_entry_price = (
            _response_float(verified_order, "filled_avg_price")
            if verified_order is not None
            else _response_float(order_resp, "filled_avg_price")
        )
        pos = SwingPosition(
            ticker=ticker,
            direction=sig.direction,
            entry_price=entry_price,
            entry_time=entry_time,
            atr_at_entry=sig.atr,
            option_symbol=option_symbol,
            qty=qty,
            config=sig.config,
            option_entry_price=option_entry_price,
            option_entry_meta=option_entry_meta,
        )
        self._pos_mgr.open_position(pos)
        if not self._dry_run:
            self._real_policy.record_entry(
                ticker=ticker,
                option_symbol=option_symbol,
                qty=qty,
                premium_at_risk=float(real_decision.premium_at_risk) if self._real_policy.enabled else 0.0,
                reason=real_decision.reason,
            )

        # Attach pre-entry 5m bar history so the chart can show context before the entry
        pre_bars = [_bar_to_event(b) for b in self._buf_5m.get(ticker, [])]
        self._emit("position_chart_seed", {
            "ticker": ticker,
            "direction": int(sig.direction),
            "entry_price": float(entry_price),
            "entry_time": int(entry_time.timestamp()),
            "sl_price": float(pos.sl_price) if pos.sl_price is not None else None,
            "option_symbol": option_symbol,
            "option_entry_meta": option_entry_meta,
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
        "volume": float(bar.get("volume", 0.0) or 0.0),
    }


def _bar_shape_metrics(bar: dict, *, atr: float | None = None) -> dict[str, float | None]:
    o = float(bar.get("open", float("nan")))
    h = float(bar.get("high", float("nan")))
    l = float(bar.get("low", float("nan")))
    c = float(bar.get("close", float("nan")))
    volume_raw = bar.get("volume")
    volume = float(volume_raw) if volume_raw is not None else None
    if not all(math.isfinite(v) for v in (o, h, l, c)):
        return {
            "range": None,
            "range_atr": None,
            "body_frac": None,
            "upper_wick_frac": None,
            "lower_wick_frac": None,
            "volume": volume,
        }
    bar_range = h - l
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    atr_value = float(atr) if atr is not None else float("nan")
    return {
        "range": float(bar_range),
        "range_atr": float(bar_range / atr_value) if math.isfinite(atr_value) and atr_value > 0 else None,
        "body_frac": float(body / bar_range) if bar_range > 0 else None,
        "upper_wick_frac": float(upper / bar_range) if bar_range > 0 else None,
        "lower_wick_frac": float(lower / bar_range) if bar_range > 0 else None,
        "volume": volume,
    }


def _safe_response(resp: Any) -> Any:
    """Reduce Alpaca response to JSON-friendly form for events."""
    if resp is None:
        return None
    if isinstance(resp, (str, int, float, bool)):
        return resp
    if isinstance(resp, dict):
        keep = (
            "id",
            "client_order_id",
            "symbol",
            "qty",
            "side",
            "type",
            "order_type",
            "limit_price",
            "status",
            "submitted_at",
            "filled_avg_price",
        )
        return {k: resp.get(k) for k in keep if k in resp}
    # Object with attributes
    out = {}
    for k in (
        "id",
        "client_order_id",
        "symbol",
        "qty",
        "side",
        "type",
        "order_type",
        "limit_price",
        "status",
        "submitted_at",
        "filled_avg_price",
    ):
        if hasattr(resp, k):
            out[k] = getattr(resp, k)
    return out or str(resp)


def _entry_buy_limit_ladder(
    client: AlpacaOptionsClient,
    *,
    symbol: str,
    attempts: int,
) -> tuple[list[float], dict[str, Any]]:
    try:
        resp = client.get_option_quotes(symbols=symbol, limit=1)
        quotes = _extract_option_quotes(resp, symbol=symbol)
    except Exception as exc:
        logger.warning("entry quote fetch failed symbol=%s: %s", symbol, exc)
        return [], {"quote_error": str(exc)}
    if not quotes:
        return [], {"quote_error": "no_quotes"}
    quote = quotes[-1]
    bid = _option_quote_price(quote, mode="bid")
    ask = _option_quote_price(quote, mode="ask")
    mid = _option_quote_price(quote, mode="mid")
    quote_meta = _option_quote_context(quote)
    if not math.isfinite(ask) or ask <= 0.0:
        quote_meta["quote_error"] = "missing_ask"
        return [], quote_meta
    if not math.isfinite(mid) or mid <= 0.0:
        mid = ask
    mid = min(mid, ask)

    count = max(1, int(attempts))
    if count == 1 or math.isclose(mid, ask, rel_tol=0.0, abs_tol=1e-9):
        prices = [_round_option_limit(mid)]
        quote_meta["limit_prices"] = prices
        return prices, quote_meta

    # The ladder tops out at mid + fraction*(ask-mid). At the default 1.0 that
    # is the ask, exactly as before; below 1.0 the last rung rests short of the
    # ask rather than crossing the whole spread, and an unfilled ladder simply
    # leaves that rung working.
    fraction = min(1.0, max(0.0, _ENTRY_LADDER_MAX_ASK_FRACTION))
    top = mid + (ask - mid) * fraction
    prices: list[float] = []
    distance = top - mid
    for idx in range(count):
        raw = mid + distance * (idx / (count - 1))
        rounded = _round_option_limit(raw)
        if rounded not in prices:
            prices.append(rounded)
    top_limit = _round_option_limit(top)
    if prices[-1] != top_limit:
        prices.append(top_limit)
    logger.info(
        "entry buy limit ladder symbol=%s bid=%s mid=%.2f ask=%.2f limits=%s",
        symbol,
        f"{bid:.2f}" if math.isfinite(bid) else "nan",
        mid,
        ask,
        prices,
    )
    quote_meta["limit_prices"] = prices
    return prices, quote_meta


def _extract_option_quotes(resp: Any, *, symbol: str) -> list[dict[str, Any]]:
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


def _option_quote_context(quote: dict[str, Any]) -> dict[str, Any]:
    bid = _option_quote_price(quote, mode="bid")
    ask = _option_quote_price(quote, mode="ask")
    mid = _option_quote_price(quote, mode="mid")
    mark = _option_quote_price(quote, mode="mark")
    last = _option_quote_price(quote, mode="last")
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


def _entry_quote_spread_ok(quote_meta: dict[str, Any] | None) -> tuple[bool, str, float]:
    spread_pct = _as_float((quote_meta or {}).get("spread_pct_mid"))
    if not math.isfinite(spread_pct):
        return False, "entry_spread_missing", spread_pct
    if spread_pct >= _MAX_ENTRY_SPREAD_PCT_MID:
        return False, "entry_spread_too_wide", spread_pct
    return True, "entry_spread_ok", spread_pct


def _entry_contracts_for_quote(quote_meta: dict[str, Any] | None) -> int:
    """Contracts sized to ``TARGET_NOTIONAL_USD`` off the entry quote's premium.

    Falls back to ``DEFAULT_QTY`` when the quote didn't carry a usable mid/ask
    (this is only the fallback path taken when neither the real-account nor
    signal-policy sizing overrides it downstream).
    """
    premium = (quote_meta or {}).get("mid") or (quote_meta or {}).get("ask")
    if not premium:
        return DEFAULT_QTY
    return contracts_for_notional(premium, TARGET_NOTIONAL_USD)


def _option_quote_price(quote: dict[str, Any], *, mode: str) -> float:
    mode_key = str(mode or "ask").strip().lower()
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
        if math.isfinite(ask):
            return ask
        return bid
    if math.isfinite(ask):
        return ask
    if math.isfinite(mark):
        return mark
    if math.isfinite(last):
        return last
    if math.isfinite(bid):
        return bid
    return float("nan")


def _round_option_limit(value: float) -> float:
    return max(_OPTION_TICK, round(float(value), 2))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


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


def _has_open_long_option_position(client: AlpacaOptionsClient, *, symbol: str) -> bool:
    target = str(symbol).strip().upper()
    try:
        resp = client.get_positions()
    except Exception:
        return False
    for raw in _extract_positions(resp):
        if str(raw.get("symbol", "")).strip().upper() != target:
            continue
        if str(raw.get("side", "")).strip().lower() == "short":
            continue
        qty = _response_float(raw, "qty")
        if qty is None or qty > 0:
            return True
    return False


def _verify_entry_order(
    client: AlpacaOptionsClient,
    *,
    submitted_resp: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    order_id = str(submitted_resp.get("id", "")).strip()
    last: dict[str, Any] = submitted_resp
    status = _status_key(last.get("status"))
    if _order_is_success(status):
        return {"verified": True, "status": status, "order_id": order_id, "via": "submit_response", "order": last}
    if _order_is_terminal_fail(status):
        return {"verified": False, "status": status, "order_id": order_id, "via": "submit_response", "retryable": False, "order": last}

    deadline = time.monotonic() + _ENTRY_ORDER_VERIFY_TIMEOUT_SECS
    while time.monotonic() < deadline and order_id:
        time.sleep(_ENTRY_ORDER_VERIFY_POLL_SECS)
        try:
            current = client.get_order(order_id)
            if isinstance(current, dict):
                last = current
                status = _status_key(current.get("status"))
        except Exception as exc:
            logger.warning("entry order verify poll warning order_id=%s: %s", order_id, exc)
            continue
        if _order_is_success(status):
            return {"verified": True, "status": status, "order_id": order_id, "via": "order_poll", "order": last}
        if _order_is_terminal_fail(status):
            return {"verified": False, "status": status, "order_id": order_id, "via": "order_poll", "retryable": False, "order": last}

    if _has_open_long_option_position(client, symbol=symbol):
        return {"verified": True, "status": status or "unknown", "order_id": order_id, "via": "positions_reconcile", "order": last}

    return {
        "verified": False,
        "status": status or "unknown",
        "order_id": order_id,
        "via": "timeout",
        "retryable": bool(order_id),
        "order": last,
    }


def _cancel_entry_order(client: AlpacaOptionsClient, *, order_id: str) -> None:
    """Cancel one working entry order, and say so.

    The cancel itself was already correct; it was silent, which is a different
    problem. An unverified entry that walked its whole ladder logs a warning and
    nothing after it, so the log reads as an order left working at the broker
    when it was in fact cancelled a second later. Reviewing 2026-08-19 that
    ambiguity had to be resolved by querying the broker for three order IDs.
    """
    try:
        client.cancel_order(order_id)
    except Exception as exc:
        logger.warning("entry order cancel warning order_id=%s: %s", order_id, exc)
    else:
        logger.info("entry order cancel requested order_id=%s", order_id)


def _response_float(resp: Any, key: str) -> float | None:
    try:
        if isinstance(resp, dict):
            value = resp.get(key)
        else:
            value = getattr(resp, key)
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) and out > 0.0 else None


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
    p.add_argument("--auto-flatten-assigned-equities", action=argparse.BooleanOptionalAction, default=True,
                   help="Market-close possible exercised/assigned 100-share equity lots detected during broker reconcile")
    p.add_argument("--real-account-policy", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable real-money bookkeeping/risk policy for new option entries")
    p.add_argument("--real-account-policy-state", default=None,
                   help="Path for persistent real-account bookkeeping state")
    args = p.parse_args()

    runner = SwingLiveRunner(
        env_file=args.env,
        dry_run=args.dry_run,
        max_entries_per_bar=args.max_entries,
        auto_flatten_assigned_equities=args.auto_flatten_assigned_equities,
        real_account_policy_enabled=args.real_account_policy,
        real_account_policy_state_path=args.real_account_policy_state,
    )
    runner.start()


if __name__ == "__main__":
    main()
