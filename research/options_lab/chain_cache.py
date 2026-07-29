"""Historical options chain data layer over Alpaca — read-only, disk-cached.

Part of the options-instrument-routing experiment
(``docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md``,
Phase 1). This module is the sole place that talks to Alpaca for historical
options data. It answers three questions from cache first, and only calls out
to Alpaca for the parts of a request that are not yet cached:

- ``discover_contracts``: what contracts (OSI symbol, strike, right, OI)
  existed for a ticker in an expiry window (merges ``active`` + ``inactive``
  status so expired contracts are included).
- ``fetch_bars``: OHLCV+VWAP+trade-count bars for a set of OSI symbols.
- ``fetch_trades``: executed trade prints for a set of OSI symbols.

Verified endpoint behavior (session of 2026-07-25, paper-api credentials)
---------------------------------------------------------------------------
- ``GET /v2/options/contracts`` (trading API): ``status=active`` and
  ``status=inactive`` are two disjoint queries — an expired contract never
  shows up under ``active``, so both must be fetched and unioned to get a
  full historical strike grid. Paginates via ``page_token`` /
  ``next_page_token``, max ``limit=10000``.
- ``GET /v1beta1/options/bars`` and ``/v1beta1/options/trades`` (data API):
  accept a comma-separated ``symbols`` list (multi-symbol batched request),
  paginate via ``next_page_token``. ``timeframe`` in {1Min, 30Min, 1Day}.
- ``GET /v1beta1/options/quotes`` (historical bid/ask) returns 404 — **not
  available**. Do not build against it; ``fills.py``/spread estimation must
  use bars + trade prints only.
- History depth: verified empirically back to at least 2024-02 for liquid
  names; thin names can start later or have sparse/no data for a given
  expiry (see smoke-test notes in the Phase-1 report).

Missing-data semantics
-----------------------
A contract/date genuinely having no data (delisted before Alpaca's history
starts, zero volume day, thin name never listed) must come back as an empty
frame / ``None`` and be logged — never coerced to zero. ``core/option_liquidity.py``
documents the exact failure mode this guards against: a prior bug silently
turned "unknown OI" into "OI=0", which disabled all option routing without
anyone noticing (294 equity / 0 option live tally). Every liquidity floor in
``liquidity.py`` is written to treat ``None`` as "cannot evaluate", not as a
failing value.

Cache layout (parquet, atomic writes)
--------------------------------------
- ``Data/options_history/contracts/{ticker}.parquet`` (+ ``.meta.json``)
- ``Data/options_history/contracts_full/{ticker}.parquet`` (+ ``.meta.json``) --
  same contracts, plus ``open_interest_date``/``close_price``/``close_price_date``
  (see ``discover_contracts_full``), needed for GEX reconstruction.
- ``Data/options_history/bars/{timeframe}/{ticker}/{expiry}.parquet`` (+ ``.meta.json``)
- ``Data/options_history/trades/{ticker}/{expiry}.parquet`` (+ ``.meta.json``)

Each ``.meta.json`` sidecar records, per symbol key (or ``"_all_"`` for the
ticker-wide contracts file), the disjoint list of ``[start, end]`` windows
that have actually been fetched from Alpaca. A new request only re-fetches
the gaps not already covered — a partially-fetched window can never be
mistaken for a complete one, and an interrupted fetch is safe to resume.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from core.API.Alpaca_API.core.config import AlpacaConfig
from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("Data/options_history")
CONTRACTS_DIR = CACHE_ROOT / "contracts"
BARS_DIR = CACHE_ROOT / "bars"
TRADES_DIR = CACHE_ROOT / "trades"

VALID_TIMEFRAMES = ("1Min", "30Min", "1Day")

_CONTRACTS_PAGE_LIMIT = 10000
_BARS_PAGE_LIMIT = 10000
_SYMBOL_BATCH_SIZE = 50  # keep multi-symbol query strings and result pages sane
_ALL_KEY = "_all_"

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 30.0

_OSI_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<yymmdd>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


# --------------------------------------------------------------------------
# OSI symbol parsing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedOsi:
    root: str
    expiry: str  # YYYY-MM-DD
    right: str   # "C" or "P"
    strike: float


def parse_osi_symbol(osi_symbol: str) -> ParsedOsi:
    """Parse an Alpaca/OCC-style OSI option symbol, e.g. ``AAPL240216C00185000``.

    Raises ``ValueError`` on malformed input — callers must not swallow this
    into a silent skip, since a parse failure means the caller is about to
    look up the wrong cache file.
    """
    sym = str(osi_symbol or "").strip().upper()
    m = _OSI_RE.match(sym)
    if not m:
        raise ValueError(f"not a valid OSI option symbol: {osi_symbol!r}")
    yymmdd = m.group("yymmdd")
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    year = 2000 + yy
    expiry = f"{year:04d}-{mm:02d}-{dd:02d}"
    strike = int(m.group("strike")) / 1000.0
    return ParsedOsi(root=m.group("root"), expiry=expiry, right=m.group("right"), strike=strike)


# --------------------------------------------------------------------------
# Credentials / HTTP with 429 backoff (modeled on
# strategies/dealer_positioning/schwab_adapter.py: retry the transient
# rate-limit case with a short backoff instead of dropping the whole
# request; honor Retry-After when the server sends one).
# --------------------------------------------------------------------------

def _credentials(env_file: str | None = ".env") -> tuple[str, str, str]:
    cfg = AlpacaConfig.from_env(env_file)
    data_base = AlpacaOptionsClient(env_file=env_file)._data_base  # reuse resolved data-API base URL
    return cfg.key_id, cfg.secret_key, data_base


def _resolve_trading_base(env_file: str | None = ".env") -> str:
    # The contracts endpoint lives on the trading API, not the data API.
    # Resolve it the same way AlpacaOptionsClient does so paper/live env
    # overrides are respected. Split out as its own function (rather than
    # inlined in _fetch_contracts_page) so tests can monkeypatch it without
    # needing a real .env file.
    return AlpacaOptionsClient(env_file=env_file)._trading_base


def _sleep_backoff(attempt: int, retry_after: str | None) -> None:
    if retry_after:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            delay = _BASE_BACKOFF_SECONDS * (2 ** attempt)
    else:
        delay = _BASE_BACKOFF_SECONDS * (2 ** attempt)
    time.sleep(min(delay, _MAX_BACKOFF_SECONDS))


def _get_json(url: str, params: dict[str, Any], *, key: str, secret: str) -> dict[str, Any]:
    """GET with 429 exponential backoff. Raises on non-transient HTTP errors
    and after the retry budget is exhausted on a persistent 429."""
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    full_url = f"{url}?{query}" if query else url
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        req = urllib.request.Request(full_url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 and attempt < _MAX_RETRIES:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                logger.warning(
                    "options data 429 (attempt %d/%d), backing off: %s",
                    attempt + 1, _MAX_RETRIES, full_url.split("?")[0],
                )
                _sleep_backoff(attempt, retry_after)
                continue
            body = ""
            try:
                body = (exc.read() or b"").decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise urllib.error.HTTPError(exc.url, exc.code, f"{exc.reason}: {body}".strip(), exc.headers, None) from exc
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------
# Atomic parquet write (temp file + os.replace) — avoids the concurrent-writer
# race a plain ``to_parquet`` would have (see strategies/intraday_structure/
# state_store.py::JsonStateStore.save for the same pattern used for JSON).
# --------------------------------------------------------------------------

def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _load_meta(cache_path: Path) -> dict[str, list[list[str]]]:
    meta_path = _meta_path(cache_path)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("options cache: corrupt meta %s (%s); treating as empty", meta_path, exc)
        return {}


def _save_meta(cache_path: Path, meta: dict[str, list[list[str]]]) -> None:
    _atomic_write_json(meta, _meta_path(cache_path))


# --------------------------------------------------------------------------
# Interval (window) bookkeeping — the resumability core.
# --------------------------------------------------------------------------

def _to_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _merge_intervals(windows: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = sorted(windows, key=lambda w: w[0])
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract_interval(
    target: tuple[pd.Timestamp, pd.Timestamp],
    covered: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Gaps of ``target`` not covered by any interval in ``covered``."""
    gaps = [target]
    for c_start, c_end in _merge_intervals(covered):
        next_gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for g_start, g_end in gaps:
            if c_end <= g_start or c_start >= g_end:
                next_gaps.append((g_start, g_end))
                continue
            if c_start > g_start:
                next_gaps.append((g_start, c_start))
            if c_end < g_end:
                next_gaps.append((c_end, g_end))
        gaps = next_gaps
    return gaps


def _gaps_for_key(
    meta: dict[str, list[list[str]]], key: str, start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    raw = meta.get(key, [])
    covered = [(_to_ts(w[0]), _to_ts(w[1])) for w in raw]
    return _subtract_interval((start, end), covered)


def _record_window(meta: dict[str, list[list[str]]], key: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    raw = meta.get(key, [])
    covered = [(_to_ts(w[0]), _to_ts(w[1])) for w in raw]
    covered.append((start, end))
    merged = _merge_intervals(covered)
    meta[key] = [[s.isoformat(), e.isoformat()] for s, e in merged]


# --------------------------------------------------------------------------
# discover_contracts
# --------------------------------------------------------------------------

_CONTRACT_COLUMNS = ["osi_symbol", "ticker", "expiry", "strike", "right", "open_interest"]


def _empty_contracts_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_CONTRACT_COLUMNS)


def _fetch_contracts_page(
    ticker: str, status: str, expiry_start: str, expiry_end: str, *, env_file: str | None
) -> list[dict[str, Any]]:
    key, secret, _ = _credentials(env_file)
    trading_base = _resolve_trading_base(env_file)
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params = {
            "underlying_symbols": ticker,
            "status": status,
            "expiration_date_gte": expiry_start,
            "expiration_date_lte": expiry_end,
            "limit": _CONTRACTS_PAGE_LIMIT,
            "page_token": page_token,
        }
        payload = _get_json(f"{trading_base}/v2/options/contracts", params, key=key, secret=secret)
        page = payload.get("option_contracts") if isinstance(payload, dict) else None
        rows.extend(page or [])
        page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
        if not page_token:
            break
    return rows


def _normalize_contract_rows(raw_rows: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    if not raw_rows:
        return _empty_contracts_frame()
    out = []
    for c in raw_rows:
        osi = c.get("symbol")
        expiry = c.get("expiration_date")
        strike = c.get("strike_price")
        right = c.get("type")
        oi_raw = c.get("open_interest")
        if osi is None or expiry is None or strike is None or right is None:
            logger.warning("options contracts: skipping malformed contract row for %s: %r", ticker, c)
            continue
        oi = None
        if oi_raw is not None:
            try:
                oi = int(float(oi_raw))
            except (TypeError, ValueError):
                oi = None
        out.append(
            {
                "osi_symbol": str(osi).upper(),
                "ticker": ticker.upper(),
                "expiry": str(expiry)[:10],
                "strike": float(strike),
                "right": "C" if str(right).lower().startswith("c") else "P",
                "open_interest": oi,
            }
        )
    return pd.DataFrame(out, columns=_CONTRACT_COLUMNS)


def discover_contracts(
    ticker: str, expiry_start: str, expiry_end: str, *, env_file: str | None = ".env"
) -> pd.DataFrame:
    """Full strike grid (active + expired) for ``ticker`` between
    ``expiry_start`` and ``expiry_end`` (inclusive, ``YYYY-MM-DD``).

    Cached at ``Data/options_history/contracts/{ticker}.parquet``. Only the
    portion of ``[expiry_start, expiry_end]`` not already recorded in the
    sidecar meta file is fetched from Alpaca. Returns an empty DataFrame
    (never ``None``, never rows with a coerced 0 open_interest) when a
    ticker genuinely has no contracts in the window.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    start_ts, end_ts = _to_ts(expiry_start), _to_ts(expiry_end)
    if end_ts < start_ts:
        raise ValueError(f"expiry_end {expiry_end} before expiry_start {expiry_start}")

    cache_path = CONTRACTS_DIR / f"{ticker}.parquet"
    existing = pd.read_parquet(cache_path) if cache_path.exists() else _empty_contracts_frame()
    meta = _load_meta(cache_path)

    gaps = _gaps_for_key(meta, _ALL_KEY, start_ts, end_ts)
    if gaps:
        fetched_frames = []
        for gap_start, gap_end in gaps:
            g_start_s, g_end_s = gap_start.strftime("%Y-%m-%d"), gap_end.strftime("%Y-%m-%d")
            logger.info("discover_contracts: fetching %s gap %s..%s", ticker, g_start_s, g_end_s)
            rows: list[dict[str, Any]] = []
            for status in ("active", "inactive"):
                rows.extend(_fetch_contracts_page(ticker, status, g_start_s, g_end_s, env_file=env_file))
            fetched_frames.append(_normalize_contract_rows(rows, ticker))
            _record_window(meta, _ALL_KEY, gap_start, gap_end)
        non_empty = [f for f in ([existing] + fetched_frames) if not f.empty]
        new_df = pd.concat(non_empty, ignore_index=True) if non_empty else _empty_contracts_frame()
        new_df = new_df.drop_duplicates(subset=["osi_symbol"], keep="last").reset_index(drop=True)
        _atomic_write_parquet(new_df, cache_path)
        _save_meta(cache_path, meta)
        existing = new_df

    if existing.empty:
        logger.info("discover_contracts: no contracts found for %s in %s..%s", ticker, expiry_start, expiry_end)
        return _empty_contracts_frame()

    expiry_col = pd.to_datetime(existing["expiry"])
    mask = (expiry_col >= start_ts.tz_localize(None)) & (expiry_col <= end_ts.tz_localize(None))
    return existing.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# discover_contracts_full -- same endpoint/pagination/gap-cache machinery as
# discover_contracts, but keeps the extra fields that endpoint returns and
# discover_contracts's narrow schema drops: open_interest_date (the date the
# OI figure was actually measured -- NOT the query date), close_price, and
# close_price_date. These three are required by
# research/options_lab/gex_reconstruct.py: open_interest on
# ``status=inactive`` (expired) contracts is a single value dated near
# expiry, not a daily time series, and any historical-GEX reconstruction
# from it must know exactly which date that value applies to. Kept as a
# separate cache (``Data/options_history/contracts_full``) rather than
# widening ``discover_contracts``'s own cache/schema in place, so existing
# callers/cached files of the narrower function are untouched.
# --------------------------------------------------------------------------

CONTRACTS_FULL_DIR = CACHE_ROOT / "contracts_full"
_CONTRACT_FULL_COLUMNS = _CONTRACT_COLUMNS + ["open_interest_date", "close_price", "close_price_date"]


def _empty_contracts_full_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_CONTRACT_FULL_COLUMNS)


def _normalize_contract_rows_full(raw_rows: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    if not raw_rows:
        return _empty_contracts_full_frame()
    out = []
    for c in raw_rows:
        osi = c.get("symbol")
        expiry = c.get("expiration_date")
        strike = c.get("strike_price")
        right = c.get("type")
        oi_raw = c.get("open_interest")
        if osi is None or expiry is None or strike is None or right is None:
            logger.warning("options contracts (full): skipping malformed contract row for %s: %r", ticker, c)
            continue
        oi = None
        if oi_raw is not None:
            try:
                oi = int(float(oi_raw))
            except (TypeError, ValueError):
                oi = None
        close_price_raw = c.get("close_price")
        close_price = None
        if close_price_raw is not None:
            try:
                close_price = float(close_price_raw)
            except (TypeError, ValueError):
                close_price = None
        out.append(
            {
                "osi_symbol": str(osi).upper(),
                "ticker": ticker.upper(),
                "expiry": str(expiry)[:10],
                "strike": float(strike),
                "right": "C" if str(right).lower().startswith("c") else "P",
                "open_interest": oi,
                "open_interest_date": (str(c.get("open_interest_date"))[:10] if c.get("open_interest_date") else None),
                "close_price": close_price,
                "close_price_date": (str(c.get("close_price_date"))[:10] if c.get("close_price_date") else None),
            }
        )
    return pd.DataFrame(out, columns=_CONTRACT_FULL_COLUMNS)


def discover_contracts_full(
    ticker: str, expiry_start: str, expiry_end: str, *, env_file: str | None = ".env"
) -> pd.DataFrame:
    """Like ``discover_contracts``, but retains ``open_interest_date``,
    ``close_price``, and ``close_price_date`` from the raw Alpaca contract
    record. Cached at ``Data/options_history/contracts_full/{ticker}.parquet``
    with the same gap-tracking sidecar semantics (a re-request only fetches
    the portion of ``[expiry_start, expiry_end]`` not already cached).

    Verified (2026-07-27): ``open_interest``/``open_interest_date``/
    ``close_price``/``close_price_date`` are populated on essentially all
    ``status=inactive`` (expired) contracts, and are ``null`` on
    ``status=active`` contracts -- Alpaca does not carry a live/current OI
    figure for contracts that have not yet expired. Callers needing OI must
    therefore only trust it for contracts where ``expiry`` is already in
    the past relative to whenever this was fetched.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    start_ts, end_ts = _to_ts(expiry_start), _to_ts(expiry_end)
    if end_ts < start_ts:
        raise ValueError(f"expiry_end {expiry_end} before expiry_start {expiry_start}")

    cache_path = CONTRACTS_FULL_DIR / f"{ticker}.parquet"
    existing = pd.read_parquet(cache_path) if cache_path.exists() else _empty_contracts_full_frame()
    meta = _load_meta(cache_path)

    gaps = _gaps_for_key(meta, _ALL_KEY, start_ts, end_ts)
    if gaps:
        fetched_frames = []
        for gap_start, gap_end in gaps:
            g_start_s, g_end_s = gap_start.strftime("%Y-%m-%d"), gap_end.strftime("%Y-%m-%d")
            logger.info("discover_contracts_full: fetching %s gap %s..%s", ticker, g_start_s, g_end_s)
            rows: list[dict[str, Any]] = []
            for status in ("active", "inactive"):
                rows.extend(_fetch_contracts_page(ticker, status, g_start_s, g_end_s, env_file=env_file))
            fetched_frames.append(_normalize_contract_rows_full(rows, ticker))
            _record_window(meta, _ALL_KEY, gap_start, gap_end)
        non_empty = [f for f in ([existing] + fetched_frames) if not f.empty]
        new_df = pd.concat(non_empty, ignore_index=True) if non_empty else _empty_contracts_full_frame()
        new_df = new_df.drop_duplicates(subset=["osi_symbol"], keep="last").reset_index(drop=True)
        _atomic_write_parquet(new_df, cache_path)
        _save_meta(cache_path, meta)
        existing = new_df

    if existing.empty:
        logger.info("discover_contracts_full: no contracts found for %s in %s..%s", ticker, expiry_start, expiry_end)
        return _empty_contracts_full_frame()

    expiry_col = pd.to_datetime(existing["expiry"])
    mask = (expiry_col >= start_ts.tz_localize(None)) & (expiry_col <= end_ts.tz_localize(None))
    return existing.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# fetch_bars / fetch_trades (shared implementation)
# --------------------------------------------------------------------------

def _empty_bars_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["osi_symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])


def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["osi_symbol", "t", "x", "p", "s", "c"])


def _batched(seq: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _fetch_bars_or_trades_page(
    kind: str,
    symbols: list[str],
    *,
    timeframe: str | None,
    start: str,
    end: str,
    env_file: str | None,
) -> dict[str, list[dict[str, Any]]]:
    key, secret, data_base = _credentials(env_file)
    endpoint = "bars" if kind == "bars" else "trades"
    url = f"{data_base}/v1beta1/options/{endpoint}"
    out: dict[str, list[dict[str, Any]]] = {}
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "start": start,
            "end": end,
            "limit": _BARS_PAGE_LIMIT,
            "page_token": page_token,
        }
        if timeframe:
            params["timeframe"] = timeframe
        payload = _get_json(url, params, key=key, secret=secret)
        page = payload.get(endpoint) if isinstance(payload, dict) else None
        if isinstance(page, dict):
            for sym, rows in page.items():
                out.setdefault(sym, []).extend(rows or [])
        page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
        if not page_token:
            break
    return out


def _fetch_grouped(
    kind: str,
    osi_symbols: Sequence[str],
    *,
    timeframe: str | None,
    start: str,
    end: str,
    cache_root: Path,
    env_file: str | None,
) -> pd.DataFrame:
    if not osi_symbols:
        return _empty_bars_frame() if kind == "bars" else _empty_trades_frame()
    start_ts, end_ts = _to_ts(start), _to_ts(end)
    if end_ts < start_ts:
        raise ValueError(f"end {end} before start {start}")

    parsed: dict[str, ParsedOsi] = {}
    for sym in osi_symbols:
        parsed[sym.upper()] = parse_osi_symbol(sym)

    groups: dict[tuple[str, str], list[str]] = {}
    for sym, p in parsed.items():
        groups.setdefault((p.root, p.expiry), []).append(sym)

    out_frames: list[pd.DataFrame] = []
    for (ticker, expiry), group_symbols in groups.items():
        if timeframe:
            cache_path = cache_root / timeframe / ticker / f"{expiry}.parquet"
        else:
            cache_path = cache_root / ticker / f"{expiry}.parquet"
        existing = pd.read_parquet(cache_path) if cache_path.exists() else (
            _empty_bars_frame() if kind == "bars" else _empty_trades_frame()
        )
        meta = _load_meta(cache_path)

        gaps_by_symbol: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for sym in group_symbols:
            gaps = _gaps_for_key(meta, sym, start_ts, end_ts)
            if gaps:
                gaps_by_symbol[sym] = gaps

        if gaps_by_symbol:
            buckets: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = {}
            for sym, gaps in gaps_by_symbol.items():
                for gap in gaps:
                    buckets.setdefault(gap, []).append(sym)

            new_rows: list[dict[str, Any]] = []
            for (gap_start, gap_end), gap_symbols in buckets.items():
                g_start_s = gap_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                g_end_s = gap_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                logger.info(
                    "fetch_%s: fetching %s/%s gap %s..%s for %d symbol(s)",
                    kind, ticker, expiry, g_start_s, g_end_s, len(gap_symbols),
                )
                for batch in _batched(gap_symbols, _SYMBOL_BATCH_SIZE):
                    result = _fetch_bars_or_trades_page(
                        kind, batch, timeframe=timeframe, start=g_start_s, end=g_end_s, env_file=env_file
                    )
                    for sym in batch:
                        rows = result.get(sym, [])
                        for row in rows:
                            new_rows.append({"osi_symbol": sym, **row})
                for sym in gap_symbols:
                    _record_window(meta, sym, gap_start, gap_end)

            new_df = pd.DataFrame(new_rows) if new_rows else (
                _empty_bars_frame() if kind == "bars" else _empty_trades_frame()
            )
            non_empty = [f for f in (existing, new_df) if not f.empty]
            combined = pd.concat(non_empty, ignore_index=True) if non_empty else (
                _empty_bars_frame() if kind == "bars" else _empty_trades_frame()
            )
            dedupe_cols = [c for c in ("osi_symbol", "t") if c in combined.columns]
            if dedupe_cols:
                combined = combined.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
            _atomic_write_parquet(combined, cache_path)
            _save_meta(cache_path, meta)
            existing = combined

        if existing.empty:
            continue
        sub = existing[existing["osi_symbol"].isin(group_symbols)]
        if sub.empty:
            continue
        if "t" in sub.columns:
            t_col = pd.to_datetime(sub["t"], utc=True, errors="coerce")
            sub = sub.loc[(t_col >= start_ts) & (t_col <= end_ts)]
        out_frames.append(sub)

    if not out_frames:
        logger.info("fetch_%s: no data found for requested symbols in %s..%s", kind, start, end)
        return _empty_bars_frame() if kind == "bars" else _empty_trades_frame()
    return pd.concat(out_frames, ignore_index=True)


def fetch_bars(
    osi_symbols: Sequence[str], timeframe: str, start: str, end: str, *, env_file: str | None = ".env"
) -> pd.DataFrame:
    """OHLCV+VWAP+trade-count bars for ``osi_symbols`` between ``start`` and
    ``end`` (ISO datetimes/dates). Batched multi-symbol requests, paginated,
    cached per ``(ticker, expiry, timeframe)``. Empty frame (never ``None``
    coerced to zeros) when nothing exists for the request.
    """
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {VALID_TIMEFRAMES}, got {timeframe!r}")
    return _fetch_grouped(
        "bars", osi_symbols, timeframe=timeframe, start=start, end=end, cache_root=BARS_DIR, env_file=env_file
    )


def fetch_trades(
    osi_symbols: Sequence[str], start: str, end: str, *, env_file: str | None = ".env"
) -> pd.DataFrame:
    """Executed trade prints for ``osi_symbols`` between ``start`` and ``end``.
    Same batching/pagination/cache/gap-refetch semantics as ``fetch_bars``."""
    return _fetch_grouped(
        "trades", osi_symbols, timeframe=None, start=start, end=end, cache_root=TRADES_DIR, env_file=env_file
    )
