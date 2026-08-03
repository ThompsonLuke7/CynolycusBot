"""Live-entry readiness and freshness gates.

Nightly/pre-open jobs do the expensive cache preparation. Live order paths use
this module to require a recent successful readiness stamp before opening new
positions. Risk-reducing sells are allowed to continue; buys are removed from the
plan when readiness is not proven.

Two levels of proof, checked in that order:

1. **The global stamp** — written only when every stage of
   ``nightly_data_readiness.sh`` succeeded. Fast path: if it is current,
   everything is authorized without touching the filesystem further.

2. **Per-ticker bar freshness** — the fallback. The stamp is all-or-nothing, and
   its last stage is a ~30-minute full-universe feature rebuild, so losing that
   one stage discards proof for data that is provably fine. On 2026-07-30 stage
   1 (the shared 4H bar catch-up that live inference actually reads) finished
   ``exit=0`` at 11:07 and every ticker's bars were current to the 14:00 UTC
   bar — yet no stamp was written, and a third consecutive session opened with
   every 4H entry blocked. A ticker whose own bars cover the last completed
   session has the data its decision needs, regardless of whether an unrelated
   ticker later in the same batch failed.

The fallback is deliberately *narrow*: it proves freshness of the shared 4H bar
cache, which is what the momentum and HTF live paths build their features from
at decision time. Modules whose inference reads a prebuilt artifact instead —
Meta Ranker scores ``meta_ranker_matrix.parquet`` — must pass
``per_ticker_fallback=False``, because fresh bars say nothing about that file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, time, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

from core.calendar import prev_trading_day

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READINESS_PATH = _REPO_ROOT / "Data/readiness/latest_success.json"
#: Shared 4H bar cache refreshed by stage 1 of nightly_data_readiness.sh. Named
#: here rather than imported from strategies/ to keep core/ free of a dependency
#: on the strategy packages that consume it.
DEFAULT_BARS_4H_DIR = _REPO_ROOT / "Data/shared/bars/4h"
DEFAULT_MAX_AGE_HOURS = 96.0
_ET = ZoneInfo("America/New_York")

#: OCC option symbol: root + YYMMDD + C/P + 8-digit strike (e.g. ZS260821C00150000).
_OCC = re.compile(r"^(?P<root>[A-Z]{1,6})\d{6}[CP]\d{8}$")


def write_readiness_success(*, job: str, path: Path | None = None) -> dict:
    path = path or DEFAULT_READINESS_PATH
    payload = {
        "job": job,
        "status": "success",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return payload


def readiness_status(
    *,
    path: Path | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> tuple[bool, str, dict]:
    path = path or DEFAULT_READINESS_PATH
    if os.getenv("CYNOLYCUS_READINESS_REQUIRED", "1") == "0":
        return True, "readiness gate disabled by CYNOLYCUS_READINESS_REQUIRED=0", {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return False, f"missing readiness stamp {path}: {exc}", {}
    if payload.get("status") != "success":
        return False, f"readiness status is {payload.get('status')!r}", payload
    raw_ts = payload.get("completed_at_utc")
    try:
        completed = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
    except Exception as exc:
        return False, f"invalid readiness timestamp {raw_ts!r}: {exc}", payload
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    completed_utc = completed.astimezone(timezone.utc)
    age_hours = (now_utc - completed_utc).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return False, f"readiness stamp is {age_hours:.1f}h old (> {max_age_hours:.1f}h)", payload

    # A broad hour limit is needed across weekends, but by itself it lets a
    # Sunday stamp authorize Tuesday entries after Monday's refresh failed.
    # Require proof generated after the most recently completed trading
    # session's regular close.  Weekend/holiday stamps naturally satisfy the
    # preceding session threshold.
    now_et = now_utc.astimezone(_ET)
    prior_session = prev_trading_day(now_et.date())
    required_after = datetime.combine(prior_session, time(16, 0), tzinfo=_ET).astimezone(timezone.utc)
    if completed_utc < required_after:
        return False, (
            "readiness stamp predates latest completed trading session "
            f"({prior_session.isoformat()} 16:00 ET)"
        ), payload
    return True, f"readiness stamp OK ({age_hours:.1f}h old)", payload


def underlying_for_symbol(symbol: str) -> str:
    """Map an order symbol to the ticker whose data backs the decision.

    Equity orders are already the ticker; option orders carry the underlying in
    the OCC root, so ``ZS260821C00150000`` resolves to ``ZS``. Anything
    unrecognised is returned unchanged and will simply fail the freshness check.
    """
    text = str(symbol or "").strip().upper()
    match = _OCC.match(text)
    return match.group("root") if match else text


@lru_cache(maxsize=4096)
def _latest_bar_utc(path_str: str, mtime_ns: int) -> datetime | None:
    """Newest bar timestamp in a cached parquet.

    ``mtime_ns`` is part of the cache key, not the body: it makes the entry
    self-invalidate the moment the bar catch-up rewrites the file, so a live
    process never answers from a cache older than the data on disk.
    """
    try:
        import pandas as pd

        df = pd.read_parquet(path_str)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        for col in ("timestamp", "time", "date"):
            if col in df.columns:
                index = pd.to_datetime(df[col], utc=True, errors="coerce")
                break
        else:
            return None
    index = pd.to_datetime(index, utc=True, errors="coerce").dropna()
    if len(index) == 0:
        return None
    return index.max().to_pydatetime()


def ticker_data_status(
    ticker: str,
    *,
    bars_dir: Path | None = None,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Is this one ticker's 4H bar cache current enough to open a position?

    The bar must cover the most recently completed trading session — the same
    threshold the global stamp is held to, applied to one ticker's own data.
    Fails closed on a missing, empty or unreadable file.
    """
    directory = Path(bars_dir) if bars_dir is not None else DEFAULT_BARS_4H_DIR
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return False, "empty ticker"

    path = directory / f"{symbol}.parquet"
    if not path.exists():
        return False, f"no 4H bar cache for {symbol}"

    try:
        stat = path.stat()
    except OSError as exc:
        return False, f"cannot stat 4H bars for {symbol}: {exc}"

    latest = _latest_bar_utc(str(path), stat.st_mtime_ns)
    if latest is None:
        return False, f"unreadable or empty 4H bars for {symbol}"

    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    prior_session = prev_trading_day(now_utc.astimezone(_ET).date())
    # Any bar dated on or after the prior session's open proves that session was
    # captured; 4H bars are stamped at their open, so the session's first bar is
    # 13:30 UTC and comparing against midnight ET is the safe lower bound.
    required_after = datetime.combine(prior_session, time(0, 0), tzinfo=_ET).astimezone(timezone.utc)
    if latest < required_after:
        return False, (
            f"{symbol} 4H bars end {latest.date().isoformat()}, "
            f"before the last completed session ({prior_session.isoformat()})"
        )
    return True, f"{symbol} 4H bars current to {latest.isoformat()}"


def _ticker_for_order_symbol(
    symbol: str,
    *,
    symbol_tickers: Mapping[str, str] | None,
    new_managed: dict | None,
) -> str:
    """Best available ticker for an order symbol, preferring what the caller knows."""
    if symbol_tickers and symbol in symbol_tickers:
        return str(symbol_tickers[symbol]).strip().upper()
    for ticker, state in (new_managed or {}).items():
        route = (state or {}).get("route", "equity")
        order_symbol = state.get("occ") if route == "option" else state.get("symbol", ticker)
        if str(order_symbol) == str(symbol):
            return str(ticker).strip().upper()
    return underlying_for_symbol(symbol)


def _without_skipped_buys(items: list[tuple], skipped: set[str]) -> list[tuple]:
    """Drop only the *buy* rows for skipped symbols.

    Filtering the plan by symbol alone would also remove a sell on the same
    symbol. Exits reduce risk and must never be withheld because the data behind
    an entry could not be proven — the gate exists to stop new exposure, not to
    trap existing exposure.
    """
    return [
        row
        for row in items
        if not (str(row[0]) in skipped and len(row) >= 3 and str(row[1]).lower() == "buy")
    ]


def _drop_skipped_new_entries(new_managed: dict | None, skipped_symbols: set[str]) -> None:
    if not new_managed or not skipped_symbols:
        return
    for ticker, state in list(new_managed.items()):
        route = state.get("route", "equity")
        order_symbol = state.get("occ") if route == "option" else state.get("symbol", ticker)
        if order_symbol in skipped_symbols and int(state.get("runs_held", 0) or 0) == 0:
            new_managed.pop(ticker, None)


def filter_entry_orders_for_readiness(
    plan: Iterable[tuple],
    *,
    new_managed: dict | None = None,
    symbol_tickers: Mapping[str, str] | None = None,
    per_ticker_fallback: bool = True,
    bars_dir: Path | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> tuple[list[tuple], set[str], str]:
    """Remove BUY orders whose backing data is not proven current.

    With a current global stamp every buy passes. Without one, and when
    ``per_ticker_fallback`` is set, each buy is judged on its own ticker's 4H bar
    freshness instead of being dropped wholesale — one ticker's missing data no
    longer blocks the rest of the book.

    Set ``per_ticker_fallback=False`` for modules whose inference reads a
    prebuilt artifact rather than the bar cache; fresh bars are not evidence
    about that artifact.
    """
    items = list(plan or [])
    buy_symbols = {str(p[0]) for p in items if len(p) >= 3 and str(p[1]).lower() == "buy"}
    if not buy_symbols:
        return items, set(), "no entry orders"

    ok, reason, _payload = readiness_status(max_age_hours=max_age_hours)
    if ok:
        return items, set(), reason

    if not per_ticker_fallback:
        skipped = buy_symbols
        _drop_skipped_new_entries(new_managed, skipped)
        return _without_skipped_buys(items, skipped), skipped, reason

    skipped = set()
    failures: list[str] = []
    for symbol in sorted(buy_symbols):
        ticker = _ticker_for_order_symbol(
            symbol, symbol_tickers=symbol_tickers, new_managed=new_managed
        )
        ticker_ok, ticker_reason = ticker_data_status(ticker, bars_dir=bars_dir)
        if not ticker_ok:
            skipped.add(symbol)
            failures.append(ticker_reason)

    allowed = len(buy_symbols) - len(skipped)
    detail = f"stamp stale ({reason}); per-ticker bars authorized {allowed}/{len(buy_symbols)} entries"
    if failures:
        shown = "; ".join(sorted(set(failures))[:3])
        detail = f"{detail} — blocked: {shown}"

    _drop_skipped_new_entries(new_managed, skipped)
    return _without_skipped_buys(items, skipped), skipped, detail


def main() -> None:
    ap = argparse.ArgumentParser(description="Read/write live data-readiness stamp.")
    ap.add_argument("--write-success", action="store_true")
    ap.add_argument("--job", default="nightly_data_readiness")
    args = ap.parse_args()
    if args.write_success:
        print(json.dumps(write_readiness_success(job=args.job), indent=2, sort_keys=True))
    else:
        ok, reason, payload = readiness_status()
        print(json.dumps({"ok": ok, "reason": reason, "payload": payload}, indent=2, sort_keys=True))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
