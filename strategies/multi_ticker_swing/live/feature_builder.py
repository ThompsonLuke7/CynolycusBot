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
from strategies.multi_ticker_swing.features.build_features import build_ticker_features, _build_sector_cap_encodings

logger = logging.getLogger(__name__)

# Keep this many 30m bars in memory per ticker (well above MIN_BARS=200)
WINDOW_SIZE = 500

# All tickers that need bars (universe + context)
_CONTEXT_SET = set(CONTEXT_TICKERS)

# Default knobs for the warmup
DEFAULT_FRESHNESS_MINUTES = 30      # if last bar is younger than this, no gap fetch
DEFAULT_GAP_LOOKBACK_DAYS = 60      # cap a gap fetch to this many days even if parquet is older
DEFAULT_FETCH_WORKERS = 8           # parallel Alpaca fetch workers


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
                                    self._save_parquet(ticker)
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
            df = df.sort_index().tail(WINDOW_SIZE)
            buf = deque(
                ({"timestamp": ts, **row.to_dict()} for ts, row in df.iterrows()),
                maxlen=WINDOW_SIZE,
            )
            with self._lock:
                self._bars[ticker] = buf
                self._last_ts[ticker] = buf[-1]["timestamp"] if buf else None
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

    def _save_parquet(self, ticker: str) -> None:
        """
        Append new in-memory bars to the existing on-disk parquet.

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
        """
        with self._lock:
            if ticker not in self._bars:
                self._bars[ticker] = deque(maxlen=WINDOW_SIZE)
            self._bars[ticker].append(bar)
            ts = bar.get("timestamp")
            if isinstance(ts, datetime):
                self._last_ts[ticker] = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

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

    def _compute_latest(self, ticker: str) -> pd.Series | None:
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

        try:
            feat_df = build_ticker_features(ticker, df_ticker, context_data, meta)
        except Exception as exc:
            logger.warning("[%s] feature build failed: %s", ticker, exc)
            return None

        if feat_df is None or feat_df.empty:
            return None

        last = feat_df.iloc[-1]
        # Keep only model feature columns that are present
        avail = [c for c in FEATURE_COLUMNS if c in last.index]
        return last[avail]

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
