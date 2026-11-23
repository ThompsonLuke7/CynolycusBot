"""
Bar download + cache + 1H -> 4H resampler for momentum_expansion.

Wraps the existing Alpaca fetcher (`API.Alpaca_API.market_data.fetch_intraday`)
so credentials, pagination, and adjustment policy match the SPY live system.

Storage layout:
  Data/shared/bars/1h/{TICKER}.parquet   — native 1H pull
  Data/shared/bars/4h/{TICKER}.parquet   — derived from 1H
  Data/shared/bars/1d/{TICKER}.parquet   — native daily pull
  Data/shared/bars/context/{TICKER}.parquet — SPY/QQQ/IWM/VIXY/sector ETFs at 1H

All parquets store UTC tz-aware timestamps in column `timestamp` (or index
fall-through). The 4H resample anchors to NY local 09:30 / 13:30 so each
RTH session yields exactly two 4H bars (the second may be partial).
"""
from __future__ import annotations

import logging
import time as time_mod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday, keep_regular_hours
from core.shared_universe.delisted import is_delisted

from strategies.momentum_expansion.config.momentum_config import (
    BAR_CONFIG,
    CONTEXT_TICKERS,
    RAW_1D_DIR,
    RAW_1H_DIR,
    RAW_4H_DIR,
    RAW_CONTEXT_DIR,
    SECTOR_ETFS,
    TRAIN_END,
    TRAIN_START,
)

logger = logging.getLogger(__name__)

_DAILY_REFRESH_OVERLAP = pd.Timedelta(days=5)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _path_for(ticker: str, kind: str) -> Path:
    base = {
        "1h":      RAW_1H_DIR,
        "4h":      RAW_4H_DIR,
        "1d":      RAW_1D_DIR,
        "context": RAW_CONTEXT_DIR,
    }[kind]
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{ticker}.parquet"


def _to_utc(values) -> pd.Series:
    """``pd.to_datetime(values, utc=True, errors="coerce")`` that skips the per-element
    cache probe when the column is already datetime (~22% of a catch-up's CPU)."""
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("UTC")
    return pd.to_datetime(values, utc=True, errors="coerce", cache=False)


def _load_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = _to_utc(df["timestamp"])
    return df


def _atomic_to_parquet(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    """Write via temp file + atomic rename so a concurrent reader (SharedDataRefresher,
    startup readiness, nightly job — all can overlap) never sees a torn/partial cache."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, **kwargs)
    tmp.replace(out_path)


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ny_session_start(ts: pd.Timestamp) -> pd.Timestamp:
    """UTC instant of NY midnight on ``ts``'s New York calendar date."""
    return _as_utc_timestamp(ts).tz_convert("America/New_York").normalize().tz_convert("UTC")


def _intraday_bar_length(timeframe: str) -> pd.Timedelta | None:
    """Bar length for an Alpaca intraday timeframe string ("1Hour", "15Min"); None otherwise."""
    tf = str(timeframe).strip().lower()
    for suffix, unit in (("hour", "h"), ("min", "min")):
        if tf.endswith(suffix) and tf[: -len(suffix)].isdigit():
            return pd.Timedelta(int(tf[: -len(suffix)]), unit=unit)
    return None


def _next_fetch_start(*, kind: str, start: str, last_ts: pd.Timestamp) -> str:
    """Return the incremental fetch start; every kind re-fetches an overlap for revision.

    Daily bars overlap 5 days. Intraday bars re-fetch the whole NY session of the last
    cached bar: before 2026-11-22 the intraday catch-ups cached the still-forming hour and
    resumed at ``last + 1 min``, so a partial bar was never revised (13:00-15:00 ET hours
    partial on ~75% of sessions since 2026-07). The overlap heals the latest session.
    """
    last = _as_utc_timestamp(last_ts)
    start_bound = _as_utc_timestamp(start)
    if kind == "1d":
        fetch_ts = max(start_bound, last - _DAILY_REFRESH_OVERLAP)
    else:
        fetch_ts = max(start_bound, _ny_session_start(last))
    return fetch_ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _complete_bars_only(df: pd.DataFrame, *, timeframe: str, end: str | pd.Timestamp) -> pd.DataFrame:
    """Drop intraday bars whose interval has not closed by ``end``.

    Alpaca returns the bucket containing ``end`` aggregated from whatever has printed so
    far; caching it would freeze a partial bar. Daily timeframes pass through unchanged.
    """
    bar_len = _intraday_bar_length(timeframe)
    if bar_len is None or df is None or df.empty or "timestamp" not in df.columns:
        return df
    ts = _to_utc(df["timestamp"])
    return df.loc[(ts + bar_len) <= _as_utc_timestamp(end)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Single-ticker fetcher (incremental, resumable)
# ---------------------------------------------------------------------------

def fetch_with_retry(
    *,
    ticker: str,
    timeframe: str,
    start: str,
    end: str,
    label: str = "",
    max_attempts: int = 5,
    limit: int | None = 10_000,
) -> pd.DataFrame:
    """One Alpaca bar pull with the cache's adjustment/feed, retrying on rate limits.

    Retry with exponential backoff on rate limits (Alpaca 429 / "too many
    requests"). A whole-universe catch-up otherwise gives up on hundreds of
    tickers under load and silently leaves their bars stale. Any other error,
    or a rate limit on the last attempt, is raised to the caller.

    ``limit`` caps the TOTAL bars returned (alpaca-py 0.43); pass None for long ranges.
    """
    return _with_rate_limit_retry(
        lambda: fetch_intraday(
            ticker=ticker,
            start=start,
            end=end,
            timeframe=timeframe,
            limit=limit,
            adjustment=BAR_CONFIG["adjustment"],
            feed=BAR_CONFIG.get("feed", "sip"),
            save_path="",
        ),
        label=label or ticker, max_attempts=max_attempts,
    )


def _with_rate_limit_retry(call, *, label: str, max_attempts: int = 5):
    """Run ``call()``; on an Alpaca rate limit back off 2, 4, 8, 16 s and retry."""
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception as exc:
            msg = str(exc).lower()
            rate_limited = "429" in msg or "too many requests" in msg or "rate limit" in msg
            if not (rate_limited and attempt < max_attempts):
                raise
            backoff = min(60.0, 2.0 ** attempt)
            logger.info("[%s] rate limited (attempt %d/%d) — backing off %.0fs",
                        label, attempt, max_attempts, backoff)
            time_mod.sleep(backoff)
    raise RuntimeError("unreachable")  # pragma: no cover


def _merge_into_cache(cached: pd.DataFrame | None, df_new: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Cached rows plus ``df_new`` (fresh row wins on a duplicate timestamp), written atomically."""
    if cached is not None and not cached.empty:
        merged = pd.concat([cached, df_new], axis=0, ignore_index=True)
    else:
        merged = df_new
    merged = (
        merged.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
              .sort_values("timestamp")
              .reset_index(drop=True)
    )
    _atomic_to_parquet(merged, out_path, index=False)
    return merged


def _last_cached_ts(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    ts = _to_utc(pd.read_parquet(path, columns=["timestamp"])["timestamp"])
    last = ts.max()
    return None if pd.isna(last) else last


def _fetch_bars_multi(symbols: list[str], *, timeframe: str, start: str, end: str, client) -> pd.DataFrame:
    """One multi-symbol Alpaca bar request (all pages), with the cache's adjustment/feed."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    length = _intraday_bar_length(timeframe)
    if length is None:
        raise ValueError(f"batched fetch is intraday-only, got {timeframe!r}")
    tf = (TimeFrame(int(length / pd.Timedelta(hours=1)), TimeFrameUnit.Hour)
          if length % pd.Timedelta(hours=1) == pd.Timedelta(0)
          else TimeFrame(int(length / pd.Timedelta(minutes=1)), TimeFrameUnit.Minute))
    feed = DataFeed.SIP if str(BAR_CONFIG.get("feed", "sip")).upper() == "SIP" else DataFeed.IEX
    frame = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=tf,
        start=_as_utc_timestamp(start).to_pydatetime(), end=_as_utc_timestamp(end).to_pydatetime(),
        adjustment=Adjustment(BAR_CONFIG["adjustment"]), feed=feed, limit=None,
    )).df
    if frame is None or frame.empty:
        return pd.DataFrame()
    return keep_regular_hours(frame.reset_index())


def fetch_intraday_batched(
    tickers: Iterable[str],
    *,
    kind: str = "1h",
    end: str,
    batch_size: int = 500,
    workers: int = 6,
    executor=None,
    client=None,
) -> dict[str, str]:
    """Incremental intraday catch-up for many tickers with multi-symbol requests.

    Same cache semantics as ``fetch_one``: re-fetch the last cached NY session, never
    cache a bar still forming at ``end``, fresh row wins on a duplicate. Tickers sharing
    a fetch start go into one request: ~3 s per 1,000 symbols, against ~800 s for the
    eligible universe one request at a time at the Alpaca rate limit (2026-11-22), which
    made the 14:25 ET 4H passes read bars mid-catch-up.

    Returns ``{ticker: status}``: ``fresh`` | ``updated`` | ``unchanged`` | ``no_new_bars`` | ``delisted`` |
    ``uncached`` (no cache: fetch its full history with ``fetch_one``) | ``error``
    (batch request failed: retry with ``fetch_one``).
    """
    timeframe = {"1h": BAR_CONFIG["primary_timeframe"],
                 "context": BAR_CONFIG["context_timeframe"]}[kind]
    end_ts = _as_utc_timestamp(end)
    status: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for t in dict.fromkeys(str(x).strip().upper() for x in tickers if str(x).strip()):
        if is_delisted(t):
            # fetch_one (via fetch_intraday) never requests these; Alpaca can still serve
            # bars for a reused symbol, which must not land in the dead name's cache.
            status[t] = "delisted"
            continue
        try:
            last = _last_cached_ts(_path_for(t, kind))
        except Exception as exc:  # noqa: BLE001 - unreadable cache: let fetch_one decide
            logger.warning("[%s/%s] cache unreadable for batched fetch: %s", t, kind, exc)
            status[t] = "error"
            continue
        if last is None:
            status[t] = "uncached"
        elif last + pd.Timedelta(minutes=1) >= end_ts:
            status[t] = "fresh"
        else:
            groups.setdefault(_next_fetch_start(kind=kind, start=TRAIN_START, last_ts=last), []).append(t)

    if groups and client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        from core.API.Alpaca_API.core.config import AlpacaConfig

        cfg = AlpacaConfig.from_env()
        client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)

    for start, members in sorted(groups.items()):
        for i in range(0, len(members), batch_size):
            batch = members[i:i + batch_size]
            try:
                frame = _with_rate_limit_retry(
                    lambda: _fetch_bars_multi(batch, timeframe=timeframe, start=start, end=end, client=client),
                    label=f"batch {kind} {start} [{batch[0]}..{batch[-1]}]")
            except Exception as exc:  # noqa: BLE001 - reported per ticker, caller falls back
                logger.warning("[batch %s] %d symbols from %s failed: %s", kind, len(batch), start, exc)
                status.update({t: "error" for t in batch})
                continue
            frame = _complete_bars_only(frame, timeframe=timeframe, end=end_ts)
            by_symbol = ({str(s).upper(): g for s, g in frame.groupby("symbol")}
                         if frame is not None and not frame.empty else {})
            # Per-ticker read/merge/write is ~35 ms of mostly GIL-bound pandas work, ~100 s
            # for the universe serially. Pass a process pool as `executor` to spread it
            # over cores (catchup_shared_bars does); threads are the default.
            rows = [by_symbol.get(t) for t in batch]
            if executor is not None:
                results = executor.map(_merge_batched_rows, batch, [kind] * len(batch), rows, chunksize=25)
            else:
                with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                    results = list(pool.map(_merge_batched_rows, batch, [kind] * len(batch), rows))
            status.update(zip(batch, results))
    return status


def _same_rows(cached: pd.DataFrame | None, new: pd.DataFrame) -> bool:
    """True when the cache already holds exactly ``new``'s rows from its first timestamp on.
    Any doubt (dtype mismatch, odd frame) answers False, which just means a rewrite."""
    if cached is None or cached.empty:
        return False
    try:
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        tail = cached.loc[cached["timestamp"] >= new["timestamp"].min(), cols].reset_index(drop=True)
        fresh = new[cols].reset_index(drop=True)
        fresh["timestamp"] = _to_utc(fresh["timestamp"])
        return (len(tail) == len(fresh)
                and (tail["timestamp"].to_numpy() == fresh["timestamp"].to_numpy()).all()
                and np.allclose(tail[cols[1:]].to_numpy(float), fresh[cols[1:]].to_numpy(float),
                                rtol=0, atol=0, equal_nan=True))
    except Exception:  # noqa: BLE001
        return False


def _merge_batched_rows(ticker: str, kind: str, new: pd.DataFrame | None) -> str:
    """Merge one ticker's rows from a batched pull; skip the write when nothing changed."""
    if new is None or new.empty:
        return "no_new_bars"
    path = _path_for(ticker, kind)
    try:
        cached = _load_cached(path)
        new = new.reset_index(drop=True)
        if _same_rows(cached, new):
            return "unchanged"
        _merge_into_cache(cached, new, path)
        return "updated"
    except Exception as exc:  # noqa: BLE001 - reported per ticker, caller falls back
        logger.warning("[%s/%s] batched merge failed: %s", ticker, kind, exc)
        return "error"

def fetch_one(
    *,
    ticker: str,
    kind: str,
    start: str = TRAIN_START,
    end: str = TRAIN_END,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch a single ticker at the given timeframe. If a cache exists, only a
    revision overlap plus new bars are fetched (incremental; see
    ``_next_fetch_start``). Intraday bars still forming at ``end`` are never cached.

    kind: "1h", "1d", or "context" (1H bars for context tickers).
    Returns the on-disk DataFrame after merge, or None on failure.
    """
    timeframe = {
        "1h":      BAR_CONFIG["primary_timeframe"],
        "1d":      BAR_CONFIG["daily_timeframe"],
        "context": BAR_CONFIG["context_timeframe"],
    }[kind]

    out_path = _path_for(ticker, kind)
    cached = None if force else _load_cached(out_path)

    fetch_start = start
    if cached is not None and "timestamp" in cached.columns and len(cached):
        last_ts = cached["timestamp"].max()
        if pd.notna(last_ts):
            fetch_start = _next_fetch_start(kind=kind, start=start, last_ts=last_ts)
            # Intraday starts overlap the last session, so "fresh" means no bar can
            # follow the cached one before `end`, not "the fetch start is past end".
            probe = (fetch_start if kind == "1d"
                     else _as_utc_timestamp(last_ts) + pd.Timedelta(minutes=1))
            if _as_utc_timestamp(probe) >= _as_utc_timestamp(end):
                logger.info("[%s/%s] cache fresh (last=%s) — skip", ticker, kind, last_ts)
                return cached

    try:
        df_new = fetch_with_retry(ticker=ticker, timeframe=timeframe, start=fetch_start,
                                  end=end, label=f"{ticker}/{kind}")
    except Exception as exc:
        logger.warning("[%s/%s] fetch failed: %s", ticker, kind, exc)
        return cached

    df_new = _complete_bars_only(df_new, timeframe=timeframe, end=end)
    if df_new is None or df_new.empty:
        if cached is not None:
            return cached
        logger.info("[%s/%s] no bars returned and no cache", ticker, kind)
        return None

    merged = _merge_into_cache(cached, df_new, out_path)
    logger.info("[%s/%s] cached %d bars -> %s", ticker, kind, len(merged), out_path)
    return merged


# ---------------------------------------------------------------------------
# 1H -> 4H RTH-aligned resample
# ---------------------------------------------------------------------------

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def resample_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1H RTH bars to 4H bars anchored at NY 09:30 and 13:30.

    Rules:
      - Keep RTH only (already filtered upstream by fetch_intraday, but enforced).
      - Each RTH session contributes up to two 4H bars:
          bar A: 09:30 - 13:30 ET  (4 hourly bars)
          bar B: 13:30 - 17:30 ET  (the remaining hourly bars; may be 2-3)
      - Bar timestamp = bar start, in UTC.
    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
    df = _ensure_utc_index(df_1h.copy())
    df.columns = [c.lower() for c in df.columns]

    idx_ny = df.index.tz_convert("America/New_York")
    minutes = idx_ny.hour * 60 + idx_ny.minute

    # Slot index per row: 0 for [09:30, 13:30), 1 for [13:30, 17:30)
    slot = np.where(minutes < 13 * 60 + 30, 0, 1)
    session_date = pd.Series(idx_ny.date, index=df.index).astype("string")

    bucket = session_date + "_" + pd.Series(slot, index=df.index).astype(str)
    # Carry the bar-start timestamp as a column so it can be aggregated with a
    # vectorized "first" instead of a per-group Python apply (the latter cost
    # ~140ms/ticker = ~7 min serialized across the whole universe per catch-up).
    df = df.assign(_bucket=bucket.values, _ts=df.index)

    grouped = df.groupby("_bucket", sort=True)
    agg = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        _ts=("_ts", "first"),  # df is sorted ascending, so first = bar start
    )
    agg.index = pd.DatetimeIndex(agg.pop("_ts").values, tz="UTC")
    agg = agg.sort_index()
    agg.index.name = "timestamp"
    return agg


def build_4h_for(ticker: str, *, force: bool = False) -> pd.DataFrame | None:
    """Resample cached 1H bars to 4H and persist.

    Incremental: when a 4H cache already exists, the 1H bars from the start of the
    NY session of the last cached 4H bar onward are re-resampled and spliced over
    that session, instead of re-resampling the whole multi-year history.

    Always re-resample that session. Either of its two bars may have been built
    from hourly bars that were not yet cached, or were partial and have since been
    re-fetched (``fetch_one`` overlaps the last session). The old early return
    (``last_4h >= last_1h - 4h``) skipped exactly the case where hours were still
    arriving, so a 4H bar built after its first hour kept only that hour: the
    2026-11-20 14:32 ET momentum pass scored a 10:00 bar with ~30% of its volume.
    """
    out_path = _path_for(ticker, "4h")
    src = _path_for(ticker, "1h")
    if out_path.exists() and not force:
        try:
            cached_4h = pd.read_parquet(out_path)
            if not src.exists() or cached_4h.empty:
                return cached_4h
            cached_4h["timestamp"] = _to_utc(cached_4h["timestamp"])
            since = _ny_session_start(cached_4h["timestamp"].max())
            df_1h = pd.read_parquet(src)
            df_1h["timestamp"] = _to_utc(df_1h["timestamp"])
            tail_4h = resample_1h_to_4h(df_1h[df_1h["timestamp"] >= since])
            if tail_4h.empty:
                return cached_4h
            kept = cached_4h[cached_4h["timestamp"] < since]
            merged_4h = (
                pd.concat([kept, tail_4h.reset_index()], axis=0, ignore_index=True)
                  .drop_duplicates(subset=["timestamp"], keep="last")
                  .sort_values("timestamp")
                  .reset_index(drop=True)
            )
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            unchanged = (
                len(merged_4h) == len(cached_4h)
                and set(cols).issubset(cached_4h.columns)
                and merged_4h[cols].reset_index(drop=True).equals(
                    cached_4h.sort_values("timestamp")[cols].reset_index(drop=True))
            )
            if not unchanged:
                _atomic_to_parquet(merged_4h, out_path, index=False)
            return merged_4h.set_index("timestamp")
        except Exception as exc:
            logger.warning("[%s] incremental 4h rebuild failed, full rebuild: %s", ticker, exc)

    if not src.exists():
        logger.warning("[%s] no 1h cache to resample to 4h", ticker)
        return None
    df_1h = pd.read_parquet(src)
    df_4h = resample_1h_to_4h(df_1h)
    if df_4h.empty:
        logger.warning("[%s] 1h->4h yielded 0 bars", ticker)
        return None
    _atomic_to_parquet(df_4h.reset_index(), out_path, index=False)
    return df_4h


# ---------------------------------------------------------------------------
# Multi-ticker orchestrators
# ---------------------------------------------------------------------------

def fetch_universe_bars(
    *,
    tickers: Iterable[str],
    fetch_1h: bool = True,
    fetch_1d: bool = True,
    build_4h: bool = True,
    sleep_between: float = 0.05,
    force: bool = False,
) -> dict:
    """
    Pull 1H, 1D bars for a list of tickers, then resample to 4H.

    Returns a status dict {ticker: {"1h": n_bars, "1d": n_bars, "4h": n_bars}}.
    """
    out: dict[str, dict] = {}
    tickers = list(tickers)
    n = len(tickers)
    for i, t in enumerate(tickers, 1):
        st: dict = {"1h": 0, "1d": 0, "4h": 0}
        if fetch_1h:
            df = fetch_one(ticker=t, kind="1h", force=force)
            st["1h"] = 0 if df is None else len(df)
        if fetch_1d:
            df = fetch_one(ticker=t, kind="1d", force=force)
            st["1d"] = 0 if df is None else len(df)
        if build_4h:
            df = build_4h_for(t, force=force)
            st["4h"] = 0 if df is None else len(df)
        out[t] = st
        if i % 20 == 0 or i == n:
            logger.info("(%d/%d) bars cached for %s -> %s", i, n, t, st)
        time_mod.sleep(sleep_between)
    return out


def fetch_context_bars(
    *,
    tickers: Iterable[str] = tuple(CONTEXT_TICKERS) + tuple(SECTOR_ETFS),
    force: bool = False,
    end: str | None = None,
) -> dict:
    """Pull current 1H bars for context tickers (regime + sector ETFs).

    ``fetch_one`` defaults to the fixed historical ``TRAIN_END`` so training
    datasets remain reproducible.  Context caches serve live regime features,
    though, and must instead extend through the latest completed interval.
    """
    if end is None:
        # Match the shared-universe catch-up's SIP safety buffer.  The latest
        # 20 minutes may be unavailable on the subscription, while the last
        # completed 1H bar is sufficient for 4H feature construction.
        end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).isoformat()
    out: dict[str, int] = {}
    for t in tickers:
        df = fetch_one(ticker=t, kind="context", force=force, end=end)
        out[t] = 0 if df is None else len(df)
        # Also build 4H for context tickers so feature builder can use them at the
        # same cadence as the names being scored.
        try:
            df_1h = pd.read_parquet(_path_for(t, "context"))
            df_4h = resample_1h_to_4h(df_1h)
            if not df_4h.empty:
                _atomic_to_parquet(df_4h.reset_index(), _path_for(t, "4h"), index=False)
        except Exception as exc:
            logger.warning("[%s] context 4h build failed: %s", t, exc)
    logger.info("context bars cached: %s", out)
    return out


def fetch_daily_for_universe_scoring(
    *,
    tickers: Iterable[str],
    force: bool = False,
) -> dict:
    """Pull 1D only — used by the universe selector before any intraday cost."""
    out: dict[str, int] = {}
    for t in tickers:
        df = fetch_one(ticker=t, kind="1d", force=force)
        out[t] = 0 if df is None else len(df)
    logger.info("daily bars cached for %d tickers", len(out))
    return out
