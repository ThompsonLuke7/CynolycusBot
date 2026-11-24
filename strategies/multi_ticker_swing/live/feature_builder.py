"""
LiveSwingFeatureBuilder — maintains rolling 30m bar windows per ticker and computes
features on demand using the same build_ticker_features() pipeline used for training.

Usage:
    builder = get_shared_feature_builder()           # process-wide cache
    builder.prefill(tickers, gap_fetch=True)          # parquet → in-memory + gap fill
    feature_row = builder.update("AAPL", bar_dict)    # returns latest feature Series

Caching strategy (so warmup is as fast as possible):
  1. Already-loaded tickers (in-memory deque has data) are skipped entirely.
  2. Otherwise, the parquet at RAW_30M_DIR / "{ticker}.parquet" is read once.
  3. If the loaded data ends before "now" (older than `freshness_minutes`), a
     gap fetch from Alpaca fills the missing 30m bars in parallel.
  4. The merged dataset is written back to parquet so the *next* cold start is
     also fast.

The builder is held as a process-wide singleton (`get_shared_feature_builder`)
so that Stop → Run cycles in the dashboard reuse the in-memory cache.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import (
    CONTEXT_TICKERS,
    FEATURE_COLUMNS,
    MIN_BARS,
    RAW_30M_DIR,
    UNIVERSE_CSV,
)
from strategies.multi_ticker_swing.config.ticker_meta import training_meta
from strategies.multi_ticker_swing.features.build_features import build_ticker_features, _build_sector_cap_encodings

logger = logging.getLogger(__name__)

# Keep this many 30m bars in memory per ticker (well above MIN_BARS=200)
WINDOW_SIZE = 500

# All tickers that need bars (universe + context)
_CONTEXT_SET = set(CONTEXT_TICKERS)

BAR_MINUTES = 30
NY_TZ = "America/New_York"
# A first sync replaces this much of the deque tail with REST bars: covers the previous
# session's 16:00 bucket across a weekend and any partial bar a mid-session restart froze.
SYNC_FIRST_LOOKBACK = timedelta(days=5)
SYNC_BATCH = 500                    # symbols per multi-symbol request
# Daily-scope features (daily EMA/RSI, 52-week and 60-day levels, gap fill rates) need far
# more than WINDOW_SIZE bars; their prior-session daily bars are kept separately.
DAILY_HISTORY_SESSIONS = 500
_OHLCV = ["open", "high", "low", "close", "volume"]

# Default knobs for the warmup
DEFAULT_FRESHNESS_MINUTES = 30      # if last bar is younger than this, no gap fetch
DEFAULT_GAP_LOOKBACK_DAYS = 60      # cap a gap fetch to this many days even if parquet is older
DEFAULT_FETCH_WORKERS = 8           # parallel Alpaca fetch workers


def bucket_start(ts) -> pd.Timestamp:
    """UTC start of the NY-clock 30m bucket holding ``ts`` (naive = UTC)."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_convert(NY_TZ).floor(f"{BAR_MINUTES}min").tz_convert("UTC")


def fetch_research_bars_multi(symbols: list[str], start: datetime, end: datetime,
                              client=None) -> pd.DataFrame:
    """30m bars for many symbols under the TRAINING fetch rule (fetch_data.fetch_ticker):
    IEX feed, split adjustment, bar starts 09:30-16:00 ET. Long frame: symbol, timestamp, OHLCV."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from core.API.Alpaca_API.market_data.fetch_intraday import keep_regular_hours

    if client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        from core.API.Alpaca_API.core.config import AlpacaConfig
        cfg = AlpacaConfig.from_env()
        client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)
    frames = []
    for i in range(0, len(symbols), SYNC_BATCH):
        df = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbols[i:i + SYNC_BATCH], timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
            start=start, end=end, adjustment=Adjustment.SPLIT, feed=DataFeed.IEX, limit=None)).df
        if df is not None and not df.empty:
            frames.append(keep_regular_hours(df.reset_index()))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Session (NY date, as a tz-aware midnight) daily bars from 30m bars, the way the
    research daily features aggregate them: first open, max high, min low, last close, sum volume."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLCV)
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    session = idx.tz_convert(NY_TZ).normalize()
    agg = df[_OHLCV].assign(_s=session).groupby("_s").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"))
    agg.index.name = None
    return agg.sort_index()


def daily_scope_features(frame: pd.DataFrame, history: pd.DataFrame | None, *,
                         cache: dict | None = None, cache_key: str | None = None) -> dict[str, float]:
    """Last-bar values of the research daily-scope features over the full daily history.

    ``frame``: the live deque (UTC index, OHLCV). ``history``: daily bars for sessions up
    to and including the deque's first session (which the deque may hold only part of).
    Each prior session enters as ONE synthetic bar carrying its daily OHLCV, so the
    research functions' session aggregation reproduces the true daily series exactly
    while the intraday frame stays at WINDOW_SIZE bars."""
    from signals.location_features import add_daily_high_low_gap_features
    from strategies.multi_ticker_swing.features.build_features import _add_daily_context

    if history is None or history.empty or frame is None or frame.empty:
        return {}
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    sess = idx.tz_convert(NY_TZ).normalize()
    first = sess.min()
    hist = history[history.index <= first]
    synth = hist[_OHLCV].copy()
    synth.index = (synth.index + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
    live = frame[_OHLCV].copy()
    live.index = idx
    long = pd.concat([synth, live[sess > first]]).astype(float)
    out: dict[str, float] = {}
    # daily_* for the current bar are built from sessions BEFORE it (shifted one day), so
    # they are fixed for the session; ``cache`` keys them on the prior session's bar, so
    # a REST fix to yesterday (its 16:00 bucket) still recomputes.
    long_sess = long.index.tz_convert(NY_TZ).normalize()
    current = long_sess.max()
    prior = long[long_sess < current]
    key = (current, len(prior), hash(prior.to_numpy(float)[-28:].tobytes()))
    hit = cache.get(cache_key) if cache is not None else None
    if hit is not None and hit[0] == key:
        out.update(hit[1])
    else:
        sd = pd.Series(long.index.tz_convert(NY_TZ).date, index=long.index)
        daily = _add_daily_context(long.copy(), sd)
        vals = {c: (float(daily[c].iloc[-1]) if pd.notna(daily[c].iloc[-1]) else float("nan"))
                for c in daily.columns if c.startswith("daily_")}
        out.update(vals)
        if cache is not None:
            cache[cache_key] = (key, vals)
    loc, loc_cols = add_daily_high_low_gap_features(long.copy())
    for c in loc_cols:
        v = loc[c].iloc[-1]
        out[c] = float(v) if pd.notna(v) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_SHARED_BUILDER: "LiveSwingFeatureBuilder | None" = None
_SHARED_LOCK = threading.Lock()


def get_shared_feature_builder() -> "LiveSwingFeatureBuilder":
    """Return a process-wide LiveSwingFeatureBuilder; constructs it lazily.

    Reusing the singleton across SwingLiveRunner sessions means a Stop → Run cycle
    skips parquet I/O entirely if the cache is still warm.
    """
    global _SHARED_BUILDER
    with _SHARED_LOCK:
        if _SHARED_BUILDER is None:
            _SHARED_BUILDER = LiveSwingFeatureBuilder()
        return _SHARED_BUILDER


def reset_shared_feature_builder() -> None:
    """Drop the singleton (forces re-load on next get). Mainly for tests."""
    global _SHARED_BUILDER
    with _SHARED_LOCK:
        _SHARED_BUILDER = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class LiveSwingFeatureBuilder:
    """
    Maintains a rolling WINDOW_SIZE-bar 30m DataFrame per ticker.
    On each new bar appended, recomputes features and returns the latest row.
    """

    def __init__(self) -> None:
        self._bars: dict[str, deque[dict]] = {}
        self._last_ts: dict[str, datetime] = {}
        self._universe_meta: pd.DataFrame = pd.read_csv(UNIVERSE_CSV)
        sec_ids, cap_ids, type_ids = _build_sector_cap_encodings(self._universe_meta)
        self._sec_ids = sec_ids
        self._cap_ids = cap_ids
        self._type_ids = type_ids
        self._lock = threading.Lock()
        # Last 30m bucket each ticker was synced through from REST (sync_recent_bars).
        self._synced_through: dict[str, pd.Timestamp] = {}
        self.last_sync: dict = {}
        # Prior-session daily bars per ticker (daily_scope_features), from the full parquet.
        self._daily_hist: dict[str, pd.DataFrame] = {}
        self._daily_scope_cache: dict = {}

    # ------------------------------------------------------------------
    # Cache state
    # ------------------------------------------------------------------

    def cache_status(self, ticker: str) -> dict:
        """Return current in-memory state for `ticker`."""
        with self._lock:
            buf = self._bars.get(ticker)
            n = len(buf) if buf else 0
            last_ts = self._last_ts.get(ticker)
        return {"bars": n, "last_ts": last_ts}

    def has_fresh_cache(
        self,
        ticker: str,
        *,
        min_bars: int = MIN_BARS,
        freshness_minutes: int = DEFAULT_FRESHNESS_MINUTES,
        ref_now: datetime | None = None,
    ) -> bool:
        """
        True iff the in-memory deque has enough bars AND the latest bar is
        younger than `freshness_minutes`. Used to skip prefill entirely.
        """
        st = self.cache_status(ticker)
        if st["bars"] < min_bars:
            return False
        last_ts = st["last_ts"]
        if last_ts is None:
            return False
        ref = ref_now or datetime.now(timezone.utc)
        return (ref - last_ts) <= timedelta(minutes=freshness_minutes)

    # ------------------------------------------------------------------
    # Prefill (cache-aware, optionally gap-fetched in parallel)
    # ------------------------------------------------------------------

    def prefill_from_parquet(
        self,
        tickers: list[str],
        progress_callback: Callable[[int, int, str, int, bool], None] | None = None,
    ) -> None:
        """
        Backwards-compatible wrapper used by older callers — just delegates to
        `prefill()` with default settings (cache + gap fetch + write-back enabled).
        The progress_callback is invoked with the legacy (idx, total, ticker, bars, ok)
        signature so existing dashboards keep working.
        """
        def _legacy_cb(event: dict) -> None:
            if event.get("phase") in ("done_ticker", "cached", "gap_fetch_failed"):
                progress_callback(
                    int(event["index"]),
                    int(event["total"]),
                    str(event["ticker"]),
                    int(event.get("bars", 0)),
                    bool(event.get("ok", False)),
                )

        cb = _legacy_cb if progress_callback is not None else None
        self.prefill(
            tickers,
            gap_fetch=True,
            save_back=True,
            progress=cb,
        )

    def prefill(
        self,
        tickers: list[str],
        *,
        gap_fetch: bool = True,
        save_back: bool = True,
        max_workers: int = DEFAULT_FETCH_WORKERS,
        freshness_minutes: int = DEFAULT_FRESHNESS_MINUTES,
        gap_lookback_days: int = DEFAULT_GAP_LOOKBACK_DAYS,
        progress: Callable[[dict], None] | None = None,
        ref_now: datetime | None = None,
    ) -> dict[str, dict]:
        """
        Cache-aware prefill. For each ticker:

          * if the in-memory deque is already fresh → skip ("cached" event).
          * else read parquet (one shot per ticker).
          * if `gap_fetch=True` and the latest bar is older than `freshness_minutes`,
            queue a gap-fetch on Alpaca (parallel, up to `max_workers`).
          * if `save_back=True`, the merged data is written back to parquet so the
            next cold start is also fast.

        `progress` receives structured events:

            {"phase": "start", "total": int}
            {"phase": "cached",        "index": i, "total": N, "ticker": str, "bars": int, "ok": True}
            {"phase": "loaded",        "index": i, "total": N, "ticker": str, "bars": int, "ok": bool}
            {"phase": "gap_fetching",  "ticker": str, "from": iso, "to": iso}
            {"phase": "gap_filled",    "ticker": str, "added": int, "bars": int}
            {"phase": "gap_fetch_failed", "ticker": str, "error": str}
            {"phase": "done_ticker",   "index": i, "total": N, "ticker": str, "bars": int, "ok": bool}
            {"phase": "done", "loaded": int, "cached": int, "gap_filled": int, "failed": int}

        Returns a per-ticker summary dict.
        """
        ref = ref_now or datetime.now(timezone.utc)
        all_needed = sorted(set(tickers) | _CONTEXT_SET)
        total = len(all_needed)
        summary: dict[str, dict] = {}

        if progress:
            try:
                progress({"phase": "start", "total": total})
            except Exception as exc:
                logger.warning("progress raised: %s", exc)

        # --- Step 1: figure out which tickers are already cached fresh ----
        to_load_idx: list[tuple[int, str]] = []
        cached_count = 0
        for idx, ticker in enumerate(all_needed, start=1):
            if self.has_fresh_cache(
                ticker, min_bars=MIN_BARS,
                freshness_minutes=freshness_minutes, ref_now=ref,
            ):
                st = self.cache_status(ticker)
                summary[ticker] = {"source": "cached", "bars": st["bars"], "ok": True}
                cached_count += 1
                self._notify(progress, {
                    "phase": "cached", "index": idx, "total": total,
                    "ticker": ticker, "bars": st["bars"], "ok": True,
                })
                self._notify(progress, {
                    "phase": "done_ticker", "index": idx, "total": total,
                    "ticker": ticker, "bars": st["bars"], "ok": True,
                })
            else:
                to_load_idx.append((idx, ticker))

        # --- Step 2: load parquets sequentially (fast — pure disk I/O) ----
        gap_candidates: list[tuple[int, str, datetime | None]] = []
        loaded_count = 0
        failed_count = 0
        for idx, ticker in to_load_idx:
            n_loaded, last_ts, ok = self._load_parquet(ticker)
            summary[ticker] = {
                "source": "parquet" if ok else "missing",
                "bars": n_loaded, "ok": ok,
            }
            self._notify(progress, {
                "phase": "loaded", "index": idx, "total": total,
                "ticker": ticker, "bars": n_loaded, "ok": ok,
            })
            if not ok:
                failed_count += 1
            else:
                loaded_count += 1

            # Decide if a gap fetch is needed
            need_gap = False
            if gap_fetch:
                if last_ts is None:
                    need_gap = True
                else:
                    need_gap = (ref - last_ts) > timedelta(minutes=freshness_minutes)
            if need_gap:
                gap_candidates.append((idx, ticker, last_ts))
            else:
                self._notify(progress, {
                    "phase": "done_ticker", "index": idx, "total": total,
                    "ticker": ticker, "bars": n_loaded, "ok": ok,
                })

        # --- Step 3: parallel gap-fetch -----------------------------------
        gap_filled_count = 0
        if gap_candidates:
            from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday
            min_start = ref - timedelta(days=gap_lookback_days)

            def _fetch(ticker: str, last_ts: datetime | None):
                start = (last_ts + timedelta(minutes=1)) if last_ts else min_start
                if start < min_start:
                    start = min_start
                self._notify(progress, {
                    "phase": "gap_fetching",
                    "ticker": ticker,
                    "from": start.isoformat(),
                    "to": ref.isoformat(),
                })
                try:
                    df = fetch_intraday(
                        ticker=ticker,
                        start=start,
                        end=ref,
                        timeframe="30Min",
                        adjustment="split",   # the training fetch rule (fetch_data.fetch_ticker)
                        save_path="",  # don't write the per-ticker side parquet
                    )
                except Exception as exc:
                    return ticker, None, str(exc)
                return ticker, df, None

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {
                    pool.submit(_fetch, t, lt): (i, t)
                    for (i, t, lt) in gap_candidates
                }
                for fut in as_completed(futs):
                    idx, ticker = futs[fut]
                    try:
                        _, df, err = fut.result()
                    except Exception as exc:
                        df, err = None, str(exc)
                    if err:
                        self._notify(progress, {
                            "phase": "gap_fetch_failed",
                            "index": idx, "total": total,
                            "ticker": ticker, "error": err,
                        })
                        s = summary.setdefault(ticker, {})
                        s["gap_error"] = err
                    else:
                        added = self._merge_fetched(ticker, df)
                        if added > 0:
                            gap_filled_count += 1
                            if save_back:
                                try:
                                    self._save_parquet(ticker, complete_before=ref)
                                except Exception as exc:
                                    logger.warning("[%s] parquet save_back failed: %s", ticker, exc)
                        st = self.cache_status(ticker)
                        s = summary.setdefault(ticker, {})
                        s["gap_added"] = added
                        s["bars"] = st["bars"]
                        self._notify(progress, {
                            "phase": "gap_filled",
                            "ticker": ticker,
                            "added": added,
                            "bars": st["bars"],
                        })
                    # Mark the ticker complete regardless of fetch outcome
                    st = self.cache_status(ticker)
                    self._notify(progress, {
                        "phase": "done_ticker",
                        "index": idx, "total": total,
                        "ticker": ticker, "bars": st["bars"],
                        "ok": err is None,
                    })

        if progress:
            self._notify(progress, {
                "phase": "done",
                "loaded": loaded_count,
                "cached": cached_count,
                "gap_filled": gap_filled_count,
                "failed": failed_count,
            })

        return summary

    # ------------------------------------------------------------------
    # Disk I/O helpers
    # ------------------------------------------------------------------

    def _load_parquet(self, ticker: str) -> tuple[int, datetime | None, bool]:
        path = RAW_30M_DIR / f"{ticker}.parquet"
        if not path.exists():
            logger.warning("[%s] no 30m parquet for prefill, skipping", ticker)
            return 0, None, False
        try:
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            hist = daily_ohlcv(df).tail(DAILY_HISTORY_SESSIONS)
            df = df.tail(WINDOW_SIZE)
            buf = deque(
                ({"timestamp": ts, **row.to_dict()} for ts, row in df.iterrows()),
                maxlen=WINDOW_SIZE,
            )
            with self._lock:
                self._bars[ticker] = buf
                self._last_ts[ticker] = buf[-1]["timestamp"] if buf else None
                self._daily_hist[ticker] = hist
            return len(buf), (buf[-1]["timestamp"] if buf else None), True
        except Exception as exc:
            logger.warning("[%s] parquet load failed: %s", ticker, exc)
            return 0, None, False

    def _merge_fetched(self, ticker: str, df: pd.DataFrame | None) -> int:
        """Append fetched 30m bars (newer than last cached) into the deque."""
        if df is None or df.empty:
            return 0
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" not in df.columns:
            return 0
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        with self._lock:
            buf = self._bars.setdefault(ticker, deque(maxlen=WINDOW_SIZE))
            last_ts = self._last_ts.get(ticker)
            added = 0
            for _, row in df.iterrows():
                ts = row["timestamp"]
                if last_ts is not None and ts <= last_ts:
                    continue
                bar = {
                    "timestamp": ts,
                    "open": float(row.get("open", float("nan"))),
                    "high": float(row.get("high", float("nan"))),
                    "low":  float(row.get("low",  float("nan"))),
                    "close": float(row.get("close", float("nan"))),
                    "volume": float(row.get("volume", 0.0)),
                }
                buf.append(bar)
                last_ts = ts
                added += 1
            self._last_ts[ticker] = last_ts
        return added

    def _save_parquet(self, ticker: str, complete_before: datetime | None = None) -> None:
        """
        Append new in-memory bars to the existing on-disk parquet.

        ``complete_before``: write only bars that had closed by then. A gap fetch through
        "now" returns the still-forming bar; writing it froze a partial bar into the
        research raw store at every mid-session restart (2026-11 audit: 5-9 per ticker).

        Reads the existing parquet (if any), merges with the in-memory deque,
        deduplicates by timestamp, sorts, and writes the full history back.
        This preserves the complete bar history for future training use rather
        than truncating to the in-memory WINDOW_SIZE window.
        """
        with self._lock:
            buf = self._bars.get(ticker)
            if not buf:
                return
            rows = list(buf)

        new_df = pd.DataFrame(rows)
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)
        if complete_before is not None:
            cutoff = pd.Timestamp(complete_before)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            new_df = new_df[new_df["timestamp"] + pd.Timedelta(minutes=BAR_MINUTES) <= cutoff]
            if new_df.empty:
                return
        new_df = new_df.set_index("timestamp")

        out_path = RAW_30M_DIR / f"{ticker}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                existing.columns = [c.lower() for c in existing.columns]
                if "timestamp" in existing.columns:
                    existing = existing.set_index("timestamp")
                existing.index = pd.to_datetime(existing.index, utc=True)
                merged = pd.concat([existing, new_df])
            except Exception as exc:
                logger.warning("[%s] could not read existing parquet for append, overwriting: %s", ticker, exc)
                merged = new_df
        else:
            merged = new_df

        merged = (
            merged[~merged.index.duplicated(keep="last")]
            .sort_index()
        )

        tmp = out_path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        tmp.replace(out_path)

    # ------------------------------------------------------------------
    # Online state mutators
    # ------------------------------------------------------------------

    def append_bar(self, ticker: str, bar: dict) -> None:
        """
        Append a new 30m bar (from the aggregator) without computing features.
        bar must have keys: timestamp, open, high, low, close, volume.

        The bar is stamped at its 30m bucket START, the stamp research bars carry; the
        aggregator stamps it with its last 1m bar (15:29, or 15:27 for a name that did not
        print at :29), which shifted the time features and the context-ticker alignment.
        A bucket already at the tail (the forming bar a prefill fetched) is merged into, not
        appended a second time; a bucket REST already synced is left to the REST bar.
        """
        ts = bar.get("timestamp")
        if ts is None:
            return
        b = bucket_start(ts)
        new = {**bar, "timestamp": b.to_pydatetime()}
        with self._lock:
            synced = self._synced_through.get(ticker)
            if synced is not None and b <= synced:
                return
            buf = self._bars.setdefault(ticker, deque(maxlen=WINDOW_SIZE))
            if buf and bucket_start(buf[-1]["timestamp"]) == b:
                old = buf[-1]
                buf[-1] = {**old, "high": max(old["high"], new["high"]), "low": min(old["low"], new["low"]),
                           "close": new["close"], "volume": float(old.get("volume", 0.0)) + float(new.get("volume", 0.0))}
            else:
                buf.append(new)
            self._last_ts[ticker] = b.to_pydatetime()

    def sync_recent_bars(self, tickers: list[str], *, now: datetime | None = None,
                         fetch: Callable[..., pd.DataFrame] = fetch_research_bars_multi) -> dict:
        """Replace each ticker's recent 30m bars with REST bars under the training fetch rule.

        The live aggregator's bars are not the bars the model was trained on: stamps differ,
        a restart leaves a partial (or duplicated) bucket, live-built days have no 16:00
        bucket, and stream gaps go unnoticed. Parity on 2026-11-17..23 failed on ~70% of
        rows for core trend/volatility features from this alone. Complete REST bars replace
        every deque bar of the buckets they cover; aggregator bars newer than REST's last
        bucket (REST latency) are kept. First sync per ticker covers SYNC_FIRST_LOOKBACK,
        later ones only buckets after the last synced. Never raises: on failure the deque is
        left as it was and ``last_sync["error"]`` says why.
        """
        import time as _time

        t0 = _time.time()
        ref = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        ref = ref.tz_localize("UTC") if ref.tzinfo is None else ref.tz_convert("UTC")
        closed = bucket_start(ref) - pd.Timedelta(minutes=BAR_MINUTES)
        groups: dict[pd.Timestamp, list[str]] = {}
        for t in dict.fromkeys(tickers):
            if not self._bars.get(t):
                continue
            synced = self._synced_through.get(t)
            if synced is not None and synced >= closed:
                continue
            start = (synced + pd.Timedelta(minutes=BAR_MINUTES)) if synced is not None else ref - SYNC_FIRST_LOOKBACK
            groups.setdefault(start, []).append(t)
        stats = {"at": ref.isoformat(), "requested": sum(map(len, groups.values())), "synced": 0,
                 "rest_bars": 0, "kept_newer": 0, "error": None}
        try:
            for start, group in groups.items():
                df = fetch(sorted(group), start.to_pydatetime(), ref.to_pydatetime())
                if df is None or df.empty:
                    continue
                df = df.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df[df["timestamp"] + pd.Timedelta(minutes=BAR_MINUTES) <= ref]
                for sym, rows in df.groupby("symbol"):
                    stats["kept_newer"] += self._splice_rest_bars(str(sym), rows.sort_values("timestamp"))
                    stats["synced"] += 1
                    stats["rest_bars"] += len(rows)
        except Exception as exc:  # noqa: BLE001 - the scan runs on the deque as it is
            stats["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("30m bar sync failed (scan uses the stream-built bars): %s", stats["error"])
        stats["seconds"] = round(_time.time() - t0, 2)
        self.last_sync = stats
        return stats

    def _splice_rest_bars(self, ticker: str, rows: pd.DataFrame) -> int:
        """REST rows (one ticker, sorted, complete) into the deque; returns bars kept past REST."""
        rest = [{"timestamp": ts.to_pydatetime(), "symbol": ticker,
                 **{c: float(r[c]) for c in ("open", "high", "low", "close", "volume", "trade_count", "vwap")
                    if c in rows.columns and pd.notna(r[c])}}
                for ts, (_, r) in zip(rows["timestamp"], rows.iterrows())]
        if not rest:
            return 0
        first, last = bucket_start(rest[0]["timestamp"]), bucket_start(rest[-1]["timestamp"])
        with self._lock:
            buf = list(self._bars.get(ticker, ()))
            older = [x for x in buf if bucket_start(x["timestamp"]) < first]
            newer = [x for x in buf if bucket_start(x["timestamp"]) > last]
            merged = older + rest + newer
            self._bars[ticker] = deque(merged[-WINDOW_SIZE:], maxlen=WINDOW_SIZE)
            self._last_ts[ticker] = bucket_start(merged[-1]["timestamp"]).to_pydatetime()
            self._synced_through[ticker] = last
        return len(newer)

    def update(self, ticker: str, bar: dict) -> pd.Series | None:
        """
        Append bar and return the latest feature row for `ticker`, or None if
        not enough history or feature computation fails.
        """
        self.append_bar(ticker, bar)
        return self._compute_latest(ticker)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _to_df(self, ticker: str) -> pd.DataFrame | None:
        if ticker not in self._bars or len(self._bars[ticker]) < MIN_BARS:
            return None
        rows = list(self._bars[ticker])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _feature_frame(self, ticker: str) -> pd.DataFrame | None:
        """Full per-bar feature frame (OHLCV + atr_14 + base features), indexed by timestamp."""
        df_ticker = self._to_df(ticker)
        if df_ticker is None:
            return None

        # Build context DataFrames
        context_data: dict[str, pd.DataFrame] = {}
        for ctx in CONTEXT_TICKERS:
            df_ctx = self._to_df(ctx)
            if df_ctx is not None:
                context_data[ctx] = df_ctx

        # Resolve sector/cap meta for this ticker
        row = self._universe_meta[self._universe_meta["ticker"] == ticker]
        if row.empty:
            meta = {"sector_id": 0, "market_cap_bucket": 0, "asset_type": 0}
        else:
            r = row.iloc[0]
            meta = {
                "sector_id":        self._sec_ids.get(r.get("sector", ""), 0),
                "market_cap_bucket": self._cap_ids.get(r.get("market_cap_bucket", ""), 0),
                "asset_type":        self._type_ids.get(r.get("type", ""), 0),
            }
        # The values the ranker was trained on win (config/ticker_meta.py).
        trained = training_meta(ticker)
        if trained is not None:
            meta = {**meta, **trained}

        try:
            feat_df = build_ticker_features(ticker, df_ticker, context_data, meta)
        except Exception as exc:
            logger.warning("[%s] feature build failed: %s", ticker, exc)
            return None
        return feat_df if feat_df is not None and not feat_df.empty else None

    def _compute_latest(self, ticker: str) -> pd.Series | None:
        feat_df = self._feature_frame(ticker)
        if feat_df is None:
            return None
        last = feat_df.iloc[-1]
        # Keep only model feature columns that are present
        avail = [c for c in FEATURE_COLUMNS if c in last.index]
        return last[avail]

    def compute_latest_full(self, ticker: str) -> pd.Series | None:
        """Latest row with the FULL 192-feature ranker set: base features + live bar-location
        (validated identical to the offline parquet) + as-of daily context. Used by the
        RankerSwingScanner so the live feature set has no gap vs training."""
        from strategies.multi_ticker_swing.live.context_features import (
            compute_bar_location, DailyContextLookup,
        )
        if not hasattr(self, "_daily_ctx"):
            self._daily_ctx = DailyContextLookup()   # training export + nightly live rows
        frame = self._feature_frame(ticker)
        if frame is None:
            return None
        out = frame.iloc[-1].copy()
        try:
            for k, v in compute_bar_location(frame, self._spy_feature_frame()).items():
                out[k] = v
        except Exception as exc:
            logger.warning("[%s] bar-location build failed: %s", ticker, exc)
        try:
            for k, v in daily_scope_features(frame, self._daily_history(ticker, frame),
                                             cache=self._daily_scope_cache, cache_key=ticker).items():
                out[k] = v
        except Exception as exc:
            logger.warning("[%s] daily-scope features failed (500-bar values kept): %s", ticker, exc)
        for k, v in self._daily_ctx.get(ticker, frame.index[-1]).items():
            out[k] = v
        return out

    def _spy_feature_frame(self) -> pd.DataFrame | None:
        """SPY's feature frame, rebuilt only when SPY's or a context ticker's bars change.
        It was rebuilt for every scanned ticker (~0.2 s x ~900 names per boundary scan)."""
        key = tuple((t, len(self._bars.get(t, ())), self._last_ts.get(t)) for t in ["SPY", *CONTEXT_TICKERS])
        cached = getattr(self, "_spy_frame_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        frame = self._feature_frame("SPY")
        self._spy_frame_cache = (key, frame)
        return frame

    def _daily_history(self, ticker: str, frame: pd.DataFrame) -> pd.DataFrame | None:
        """Prior-session daily bars, extended with the deque's complete sessions so a
        process that runs for days keeps the sessions that roll out of the deque."""
        hist = self._daily_hist.get(ticker)
        if hist is None or hist.empty:
            return None
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        sess = idx.tz_convert(NY_TZ).normalize()
        done = frame[(sess > sess.min()) & (sess < sess.max())]   # skip partial first + current
        if not done.empty:
            hist = daily_ohlcv(done).combine_first(hist).tail(DAILY_HISTORY_SESSIONS)
            with self._lock:
                self._daily_hist[ticker] = hist
        return hist

    def get_atr(self, ticker: str) -> float:
        """Return the ATR value at the last available bar (for SL computation)."""
        df = self._to_df(ticker)
        if df is None:
            return float("nan")
        if "atr_14" in df.columns:
            val = df["atr_14"].iloc[-1]
            return float(val) if pd.notna(val) else float("nan")

        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            return float("nan")

        try:
            import pandas_ta as ta
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
            if atr_series is None or atr_series.empty:
                return float("nan")
            val = atr_series.iloc[-1]
        except Exception:
            high_low = df["high"] - df["low"]
            high_prev_close = (df["high"] - df["close"].shift(1)).abs()
            low_prev_close = (df["low"] - df["close"].shift(1)).abs()
            true_range = pd.concat(
                [high_low, high_prev_close, low_prev_close],
                axis=1,
            ).max(axis=1)
            val = true_range.rolling(14).mean().iloc[-1]
        return float(val) if pd.notna(val) else float("nan")

    def get_last_bar(self, ticker: str) -> dict | None:
        """Return the most recently appended bar dict."""
        if ticker not in self._bars or not self._bars[ticker]:
            return None
        return self._bars[ticker][-1]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _notify(cb: Callable[[dict], None] | None, event: dict) -> None:
        if cb is None:
            return
        try:
            cb(event)
        except Exception as exc:
            logger.warning("prefill progress callback raised: %s", exc)
