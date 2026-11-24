"""Research-side feature vectors for archived 30m swing decisions (decision-state parity).

The reference is what the TRAINING pipeline computes for a (30m bar, ticker), built from a
fresh Alpaca pull, never from the live builder's deque or ``RAW_30M_DIR``: a reference built
from the cache the live path reads cannot see a cache defect (the 2026-11 partial-bar lesson).

  * base features   build_ticker_features over the full fresh history (training semantics,
                    not the live 500-bar window), research fetch rule: 30Min, IEX, split
                    adjustment, bar starts 09:30-16:00 ET (fetch_data.fetch_ticker).
  * bar location    export_for_colab._build_ticker_location_frame / _build_spy_location_context
                    on that frame (the training export's own code).
  * daily context   export_for_colab._build_daily_context_features for the (ticker, date) keys,
                    joined on the bar's UTC date as the training export joins it.

Keys are 30m bucket STARTS (Alpaca's stamp). Live archives the stamp of the last 1m bar in
the bucket (e.g. 15:29), so callers map with ``bucket_start``.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_DAYS = 800      # daily EMA-50 / 52-week levels converge (training used history from 2021)
NY = "America/New_York"


def bucket_start(ts: pd.Series) -> pd.Series:
    """UTC 30m bucket start (NY clock) of each timestamp."""
    t = pd.to_datetime(ts, utc=True)
    return t.dt.tz_convert(NY).dt.floor("30min").dt.tz_convert("UTC")


def fetch_research_30m(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fresh 30m bars under the research fetch rule, UTC DatetimeIndex."""
    from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

    df = fetch_intraday(ticker=ticker, start=start.to_pydatetime(), end=end.to_pydatetime(),
                        timeframe="30Min", limit=None, adjustment="split", save_path="")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns=["timestamp"], errors="ignore")
    return df.sort_index()


def _meta_for(ticker: str, universe: pd.DataFrame, enc: tuple[dict, dict, dict]) -> dict:
    """Static ids as the TRAINING matrix holds them (config/ticker_meta.py); the universe
    CSV encoding only for a ticker the model never saw."""
    from strategies.multi_ticker_swing.config.ticker_meta import training_meta

    sec, cap, typ = enc
    row = universe[universe["ticker"] == ticker]
    if row.empty:
        meta = {"sector_id": 0, "market_cap_bucket": 0, "asset_type": 0}
    else:
        r = row.iloc[0]
        meta = {"sector_id": sec.get(r.get("sector", ""), 0),
                "market_cap_bucket": cap.get(r.get("market_cap_bucket", ""), 0),
                "asset_type": typ.get(r.get("type", ""), 0)}
    return {**meta, **(training_meta(ticker) or {})}


def build_reference(keys: pd.DataFrame, *,
                    fetch: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame] = fetch_research_30m,
                    daily_context: bool = True) -> tuple[pd.DataFrame, dict]:
    """Research vectors for ``keys`` (columns bar_ts = bucket start UTC, ticker).

    Returns (frame keyed by bar_ts/ticker, info). Tickers whose fresh pull or feature build
    fails are listed in info["failed"], never silently dropped."""
    from strategies.multi_ticker_swing.config.pipeline_config import CONTEXT_TICKERS, UNIVERSE_CSV
    from strategies.multi_ticker_swing.features.build_features import (
        _build_sector_cap_encodings, build_ticker_features)
    from strategies.multi_ticker_swing.models import export_for_colab as ex

    keys = keys[["bar_ts", "ticker"]].drop_duplicates().copy()
    keys["bar_ts"] = pd.to_datetime(keys["bar_ts"], utc=True)
    start = keys["bar_ts"].min() - pd.Timedelta(days=HISTORY_DAYS)
    end = keys["bar_ts"].max() + pd.Timedelta(minutes=30)
    tickers = sorted(keys["ticker"].astype(str).unique())
    info: dict = {"tickers": len(tickers), "failed": {}, "history_start": str(start), "end": str(end)}

    raw = {}
    for t in sorted(set(tickers) | set(CONTEXT_TICKERS)):
        try:
            raw[t] = fetch(t, start, end)
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            info["failed"][t] = f"fetch: {type(exc).__name__}: {exc}"
    ctx = {c: raw[c] for c in CONTEXT_TICKERS if c in raw and not raw[c].empty}
    universe = pd.read_csv(UNIVERSE_CSV)
    enc = _build_sector_cap_encodings(universe)

    frames: dict[str, pd.DataFrame] = {}
    for t in sorted(set(tickers) | {"SPY"}):
        if t not in raw or raw[t].empty:
            info["failed"].setdefault(t, "no fresh bars")
            continue
        try:
            f = build_ticker_features(t, raw[t], ctx, _meta_for(t, universe, enc))
        except Exception as exc:  # noqa: BLE001
            info["failed"][t] = f"features: {type(exc).__name__}: {exc}"
            continue
        if f is None or f.empty:
            info["failed"].setdefault(t, "feature build returned nothing")
            continue
        frames[t] = f

    spy_ctx = None
    if "SPY" in frames:
        idx = pd.DatetimeIndex(keys["bar_ts"].unique()).sort_values()
        spy_ctx = ex._build_spy_location_context(idx, frame=frames["SPY"])
    out = []
    for t, group in keys.groupby("ticker"):
        if t not in frames:
            continue
        f = frames[t]
        base = f.reindex(pd.DatetimeIndex(group["bar_ts"]))
        base.index.name = "bar_ts"
        base = base.reset_index()
        loc = ex._build_ticker_location_frame(t, group["bar_ts"], frame=f)
        if not loc.empty:
            loc = loc.rename(columns={"timestamp": "bar_ts"}).drop(columns=["ticker"])
            base = base.drop(columns=[c for c in loc.columns if c != "bar_ts" and c in base.columns])
            base = base.merge(loc, on="bar_ts", how="left")
        if spy_ctx is not None and not spy_ctx.empty:
            s = spy_ctx.rename(columns={"timestamp": "bar_ts"})
            base = base.drop(columns=[c for c in s.columns if c != "bar_ts" and c in base.columns])
            base = base.merge(s, on="bar_ts", how="left")
        base["ticker"] = t
        out.append(base)
    if not out:
        return pd.DataFrame(columns=["bar_ts", "ticker"]), info
    ref = pd.concat(out, ignore_index=True)

    if daily_context:
        ctx_df, cols, sources = ex._build_daily_context_features(
            ref[["ticker", "bar_ts"]].rename(columns={"bar_ts": "timestamp"}))
        info["daily_context_sources"] = sources
        ref["date"] = ex._norm_date(ref["bar_ts"])
        ref = ref.drop(columns=[c for c in cols if c in ref.columns])
        ref = ref.merge(ctx_df[["ticker", "date"] + cols], on=["ticker", "date"], how="left").drop(columns=["date"])
    for c in ref.columns:
        if c not in ("bar_ts", "ticker") and ref[c].dtype == object:
            ref[c] = pd.to_numeric(ref[c], errors="coerce")
    return ref.replace([np.inf, -np.inf], np.nan), info
