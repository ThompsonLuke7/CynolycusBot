"""
Weekly universe selector for momentum_expansion.

Approach:
  - Start from a curated candidate pool of liquid US tickers (reuses
    multi_ticker_swing's universe CSV — 213 names — as the seed pool).
  - On each rebuild date, score every name with daily bars available
    on (avg dollar volume, 5/20/60d relative strength vs SPY, ATR
    expansion).
  - Take the top N as that week's universe and write a snapshot CSV.
  - Backtest reads the snapshot dated <= the bar in question (no future
    constituents).

We intentionally accept survivorship bias from Alpaca's currently-listed
universe; flagged in the plan / report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from momentum_expansion.config.momentum_config import (
    RAW_1D_DIR,
    UNIVERSE_CONFIG,
    UNIVERSE_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def get_candidate_pool() -> list[str]:
    """
    Default candidate pool: union of curated multi_ticker_swing universe
    plus megacaps that aren't in there. Stocks only (ETFs handled via context).
    """
    swing_csv = (
        Path(__file__).resolve().parents[2]
        / "multi_ticker_swing" / "config" / "swing_trader_universe_v3.csv"
    )
    pool: set[str] = set()
    if swing_csv.exists():
        df = pd.read_csv(swing_csv)
        type_col = "type" if "type" in df.columns else "asset_type"
        # exclude ETFs from the candidate pool (we use those as context)
        stocks = df[df[type_col].astype(str).str.upper() != "ETF"]["ticker"].tolist()
        pool.update(stocks)

    # Hand-curated megacaps + high-beta liquid names that may not be in the
    # swing CSV. Safe to add even if duplicates — set semantics dedupes.
    pool.update([
        "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA",
        "AVGO", "AMD", "NFLX", "ADBE", "CRM", "ORCL", "INTC", "MU",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
        "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT",
        "XOM", "CVX", "COP", "OXY", "SLB", "EOG", "MPC", "PSX",
        "CAT", "DE", "BA", "GE", "HON", "LMT", "RTX",
        "WMT", "COST", "TGT", "HD", "LOW", "NKE", "MCD", "SBUX",
        "PG", "KO", "PEP", "PM", "MO",
        "DIS", "CMCSA", "T", "VZ", "TMUS",
        "F", "GM", "RIVN", "LCID",
        "COIN", "PYPL", "SQ", "SHOP",
        "PLTR", "NET", "DDOG", "SNOW", "CRWD", "PANW", "ZS",
        "UBER", "LYFT", "DASH", "ABNB",
        "BABA", "JD", "PDD", "BIDU",
        "TSM", "ASML", "ARM",
        "SMCI", "MSTR", "MARA", "RIOT",
    ])
    return sorted(pool)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class TickerSnapshotMetrics:
    ticker:        str
    last_close:    float
    avg_dollar_vol: float
    rs_5:          float
    rs_20:         float
    rs_60:         float
    atr_expand:    float
    rvol:          float
    composite:     float


def _load_daily(ticker: str) -> pd.DataFrame | None:
    p = RAW_1D_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _normalized_rank(values: pd.Series) -> pd.Series:
    """Rank to [0, 1] with NaN preserved."""
    ranked = values.rank(pct=True, method="average")
    return ranked


def score_universe(
    *,
    as_of: pd.Timestamp,
    candidates: Iterable[str],
    benchmark_ticker: str = "SPY",
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Score every candidate that has enough daily history as of `as_of`.

    Returns a DataFrame indexed by ticker with metric columns and a
    `composite` score (higher = better momentum/expansion candidate).
    """
    cfg = {**UNIVERSE_CONFIG, **(cfg or {})}
    rs_lookbacks = tuple(cfg["rs_lookbacks"])
    min_history_days = int(cfg["min_history_days"])
    score_weights = cfg["score_weights"]

    bench = _load_daily(benchmark_ticker)
    if bench is None:
        raise FileNotFoundError(
            f"Benchmark daily bars missing for {benchmark_ticker}. Run bar download first."
        )
    bench = bench.loc[bench.index <= as_of]
    if len(bench) < max(rs_lookbacks) + 5:
        raise ValueError(f"Not enough benchmark history before {as_of}.")

    bench_close = bench["close"]
    bench_rs = {
        n: bench_close.iloc[-1] / bench_close.iloc[-1 - n] - 1.0
        for n in rs_lookbacks
        if len(bench_close) > n + 1
    }

    rows: list[dict] = []
    for ticker in candidates:
        df = _load_daily(ticker)
        if df is None:
            continue
        df = df.loc[df.index <= as_of]
        if len(df) < min_history_days:
            continue

        c = df["close"]
        v = df["volume"]
        last_close = float(c.iloc[-1])
        if not (cfg["min_price"] <= last_close <= cfg["max_price"]):
            continue

        dollar_vol = (c * v).rolling(30).mean().iloc[-1]
        if not np.isfinite(dollar_vol) or dollar_vol < cfg["min_avg_dollar_vol"]:
            continue

        # Relative strength vs benchmark over each lookback
        rs_vals: dict[int, float] = {}
        for n in rs_lookbacks:
            if len(c) <= n + 1 or n not in bench_rs:
                rs_vals[n] = np.nan
                continue
            stk_ret = c.iloc[-1] / c.iloc[-1 - n] - 1.0
            rs_vals[n] = stk_ret - bench_rs[n]

        # ATR expansion: ATR(14) / ATR(60)
        h, lo, c_p = df["high"], df["low"], c.shift(1)
        tr = pd.concat([(h - lo), (h - c_p).abs(), (lo - c_p).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        atr_60 = tr.rolling(60).mean()
        atr_expand = float(atr_14.iloc[-1] / atr_60.iloc[-1]) if atr_60.iloc[-1] > 0 else np.nan

        # RVOL: today vs 20-day mean
        rvol_window = int(cfg["rvol_window"])
        rvol = float(v.iloc[-1] / v.rolling(rvol_window).mean().iloc[-1]) if v.rolling(rvol_window).mean().iloc[-1] > 0 else np.nan

        rows.append({
            "ticker":         ticker,
            "last_close":     last_close,
            "avg_dollar_vol": float(dollar_vol),
            "rs_5":           rs_vals.get(rs_lookbacks[0], np.nan),
            "rs_20":          rs_vals.get(rs_lookbacks[1], np.nan) if len(rs_lookbacks) > 1 else np.nan,
            "rs_60":          rs_vals.get(rs_lookbacks[2], np.nan) if len(rs_lookbacks) > 2 else np.nan,
            "atr_expand":     atr_expand,
            "rvol":           rvol,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "last_close", "avg_dollar_vol",
            "rs_5", "rs_20", "rs_60", "atr_expand", "rvol", "composite",
        ]).set_index("ticker")

    df_metrics = pd.DataFrame(rows).set_index("ticker")

    # Convert each metric to a [0, 1] cross-sectional rank, then weighted sum
    rs5_r   = _normalized_rank(df_metrics["rs_5"])
    rs20_r  = _normalized_rank(df_metrics["rs_20"])
    rs60_r  = _normalized_rank(df_metrics["rs_60"])
    atr_r   = _normalized_rank(df_metrics["atr_expand"])
    dvol_r  = _normalized_rank(df_metrics["avg_dollar_vol"])

    composite = (
        score_weights["rs_5"]       * rs5_r.fillna(0.5)
        + score_weights["rs_20"]    * rs20_r.fillna(0.5)
        + score_weights["rs_60"]    * rs60_r.fillna(0.5)
        + score_weights["atr_expand"] * atr_r.fillna(0.5)
        + score_weights["dollar_vol"] * dvol_r.fillna(0.5)
    )
    df_metrics["composite"] = composite
    return df_metrics.sort_values("composite", ascending=False)


# ---------------------------------------------------------------------------
# Snapshot writer / reader
# ---------------------------------------------------------------------------

def _snapshot_path(as_of: pd.Timestamp) -> Path:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    return UNIVERSE_DIR / f"universe_{as_of.strftime('%Y-%m-%d')}.csv"


def write_weekly_snapshot(
    *,
    as_of: pd.Timestamp,
    candidates: Iterable[str] | None = None,
    cfg: dict | None = None,
) -> Path:
    """
    Build and persist the weekly universe snapshot.

    `as_of` should be a Sunday-of-week (or any date — the snapshot is keyed
    by that date). Backtests look up the snapshot dated <= the bar.
    """
    cfg = {**UNIVERSE_CONFIG, **(cfg or {})}
    if candidates is None:
        candidates = get_candidate_pool()
    scored = score_universe(as_of=as_of, candidates=candidates, cfg=cfg)
    if scored.empty:
        logger.warning("No candidates scored for as_of=%s — wrote empty snapshot", as_of)

    top = scored.head(int(cfg["max_universe_size"])).reset_index()
    top["as_of"] = as_of.strftime("%Y-%m-%d")
    out = _snapshot_path(as_of)
    top.to_csv(out, index=False)
    logger.info("Universe snapshot %s written (%d names)", out.name, len(top))
    return out


def list_snapshots() -> list[Path]:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(UNIVERSE_DIR.glob("universe_*.csv"))


def load_snapshot_for(date_or_ts: pd.Timestamp | str) -> pd.DataFrame:
    """
    Load the most recent universe snapshot dated <= the given timestamp.

    Returns an empty DataFrame if no snapshot exists at or before that date.
    """
    target = pd.Timestamp(date_or_ts).normalize()
    candidates = list_snapshots()
    if not candidates:
        return pd.DataFrame()

    chosen: Path | None = None
    for p in candidates:
        # filename: universe_YYYY-MM-DD.csv
        try:
            stem_date = pd.Timestamp(p.stem.replace("universe_", ""))
        except Exception:
            continue
        if stem_date <= target:
            chosen = p
        else:
            break
    if chosen is None:
        return pd.DataFrame()
    return pd.read_csv(chosen)


def build_snapshots_over_range(
    *,
    start: str | pd.Timestamp,
    end:   str | pd.Timestamp,
    candidates: Iterable[str] | None = None,
    weekday: int = 6,
    cfg: dict | None = None,
) -> list[Path]:
    """
    Build a snapshot for every `weekday` (default Sunday) between start and end.

    Used for backtest setup — writes one snapshot per week so the backtester
    can read the appropriate one at each test bar.
    """
    if candidates is None:
        candidates = get_candidate_pool()
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    out: list[Path] = []
    cur = start_ts
    while cur.weekday() != weekday:
        cur = cur + pd.Timedelta(days=1)
    while cur <= end_ts:
        try:
            p = write_weekly_snapshot(as_of=cur, candidates=candidates, cfg=cfg)
            out.append(p)
        except Exception as exc:
            logger.warning("snapshot %s failed: %s", cur.date(), exc)
        cur = cur + pd.Timedelta(days=7)
    return out
