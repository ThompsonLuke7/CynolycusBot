"""
4H feature matrix for momentum_expansion.

All features are CAUSAL (no forward leakage) and cross-ticker comparable
(ATR-normalized / pct-change / cross-sectional rank).

Feature categories:
  trend        — EMA distance/slope, ADX, MA stack
  momentum     — multi-bar returns, RSI, MACD-hist (ATR-norm)
  rs           — relative strength vs SPY/QQQ/sector ETF, beta-like
  volume       — RVOL, dollar-volume rank
  volatility   — ATR pct, ATR expansion ratio, vol-of-vol
  structure    — distance to 52-week high, base/breakout, range position
  regime       — SPY trend state, VIX bucket
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from strategies.momentum_expansion.config.momentum_config import (
    CONTEXT_TICKERS,
    FEATURES_COMBINED,
    MIN_4H_BARS,
    PROCESSED_FEAT_DIR,
    RAW_4H_DIR,
    SECTOR_ETFS,
)
from strategies.momentum_expansion.data.load_bars import load_1d, load_4h
from strategies.momentum_expansion.data.universe import load_candidate_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector mapping (used for sector relative-strength feature)
# ---------------------------------------------------------------------------

# Coarse mapping of common tickers -> sector ETF. Tickers absent from this
# map fall back to XLK (broad). Light-weight; the universe selector / live
# system can override via config later.
SECTOR_MAP: dict[str, str] = {
    # Tech
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK", "INTC": "XLK", "MU": "XLK",
    "TSM": "XLK", "ASML": "XLK", "ARM": "XLK", "SMCI": "XLK", "PANW": "XLK",
    "CRWD": "XLK", "NET": "XLK", "DDOG": "XLK", "SNOW": "XLK", "PLTR": "XLK",
    "ZS": "XLK", "MSTR": "XLK",
    # Comm / Internet
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "CMCSA": "XLC", "T": "XLC", "VZ": "XLC", "TMUS": "XLC",
    # Cons Disc
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "LOW": "XLY", "NKE": "XLY",
    "MCD": "XLY", "SBUX": "XLY", "F": "XLY", "GM": "XLY", "RIVN": "XLY",
    "LCID": "XLY", "ABNB": "XLY", "UBER": "XLY", "LYFT": "XLY", "DASH": "XLY",
    "BABA": "XLY", "JD": "XLY", "PDD": "XLY", "SHOP": "XLY",
    # Cons Staples
    "WMT": "XLP", "COST": "XLP", "TGT": "XLP", "PG": "XLP", "KO": "XLP",
    "PEP": "XLP", "PM": "XLP", "MO": "XLP",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "V": "XLF", "MA": "XLF", "AXP": "XLF", "PYPL": "XLF",
    "SQ": "XLF", "COIN": "XLF", "SCHW": "XLF",
    # Healthcare
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV",
    "ABBV": "XLV", "TMO": "XLV", "ABT": "XLV",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "OXY": "XLE", "SLB": "XLE",
    "EOG": "XLE", "MPC": "XLE", "PSX": "XLE", "MARA": "XLE", "RIOT": "XLE",
    # Industrials
    "CAT": "XLI", "DE": "XLI", "BA": "XLI", "GE": "XLI", "HON": "XLI",
    "LMT": "XLI", "RTX": "XLI",
}


def _sector_etf_for(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), "XLK")


# ---------------------------------------------------------------------------
# Market-regime / sector-context join (WS-B — see
# docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md §4).
# `signals/market_regime/` (WS-A) is a sibling package owned by another
# workstream and is only READ here (its own config/build/sector_map), never
# edited. Every helper below fails soft: if the WS-A tables are absent,
# unreadable, or the package itself is unimportable, the new columns come out
# NaN (regime_available=0), never zero-filled and never a raised exception —
# a missing/mid-rebuild regime table must not break feature building or the
# live runner.
# ---------------------------------------------------------------------------

_REGIME_TABLE_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_REGIME_WARNED_MISSING: set[str] = set()
_REGIME_WARNED_STALE: set[tuple[str, int]] = set()

# A daily_regime.parquet whose most-recently-joined row is older than this many
# calendar days from a bar's own session date is almost certainly a stale
# build (the table is meant to be refreshed daily; a multi-day gap usually
# means `python -m signals.market_regime.build` hasn't run recently), not a
# legitimate as-of lag (which is normally 0-1 sessions, a few more across a
# weekend/holiday). Surfaced as a throttled log warning, never raised —
# `regime_stale_days` is also exposed as a feature so the model itself sees it.
_REGIME_STALE_WARN_DAYS = 5


def _read_cached_parquet(path: Path) -> pd.DataFrame | None:
    """mtime-cached parquet read so a multi-hundred-ticker batch build reads
    daily_regime.parquet / sector_state.parquet once, not once per ticker."""
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = str(path)
    cached = _REGIME_TABLE_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - corrupt/partial file
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    _REGIME_TABLE_CACHE[key] = (mtime, df)
    return df


def _load_daily_regime_table() -> pd.DataFrame | None:
    try:
        from signals.market_regime.config import DAILY_REGIME_PATH
    except Exception as exc:  # pragma: no cover - WS-A package unavailable
        logger.warning("signals.market_regime unavailable (%s) — regime_* features will be NaN", exc)
        return None
    df = _read_cached_parquet(DAILY_REGIME_PATH)
    if df is None:
        if str(DAILY_REGIME_PATH) not in _REGIME_WARNED_MISSING:
            logger.warning(
                "Daily regime table missing at %s — risk_appetite_z/liquidity_stress_z/"
                "credit_risk_z/sector_dispersion_21d/regime_* features will be NaN "
                "(regime_available=0). Build it with `python -m signals.market_regime.build`.",
                DAILY_REGIME_PATH,
            )
            _REGIME_WARNED_MISSING.add(str(DAILY_REGIME_PATH))
        return None
    required = {"date", "available_at", "risk_appetite_z", "liquidity_stress_z",
                "credit_risk_z", "sector_dispersion_raw", "breadth_z"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Daily regime table missing columns %s — regime_* features will be NaN", sorted(missing))
        return None
    return df


def _load_sector_state_table() -> pd.DataFrame | None:
    try:
        from signals.market_regime.config import SECTOR_STATE_PATH
    except Exception as exc:  # pragma: no cover - WS-A package unavailable
        logger.warning("signals.market_regime unavailable (%s) — sector_* features will be NaN", exc)
        return None
    df = _read_cached_parquet(SECTOR_STATE_PATH)
    if df is None:
        if str(SECTOR_STATE_PATH) not in _REGIME_WARNED_MISSING:
            logger.warning(
                "Sector state table missing at %s — sector_* features will be NaN. "
                "Build it with `python -m signals.market_regime.build`.", SECTOR_STATE_PATH,
            )
            _REGIME_WARNED_MISSING.add(str(SECTOR_STATE_PATH))
        return None
    required = {"date", "sector_etf", "available_at", "excess_21d", "excess_63d",
                "rank_21d", "rank_63d", "rs_accel", "above_20d", "above_50d"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Sector state table missing columns %s — sector_* features will be NaN", sorted(missing))
        return None
    return df


def _regime_sector_etf_for(ticker: str) -> str | None:
    """Curated-map-only sector resolution for the NEW sector_* columns.

    Deliberately NOT `_sector_etf_for` above — that function owns
    `rs_sector_5`/`rs_sector_20` and must keep its XLK fallback byte-for-byte
    (deployed boosters were trained under that semantics). `signals.market_regime`'s
    empirical resolver ships with `SECTOR_RESOLVER_ENABLED=False` by default
    (plan defect D2), so an unresolved ticker here is `None` -> NaN sector_*
    features, never a silent XLK guess.
    """
    try:
        from signals.market_regime.sector_map import sector_etf_for
    except Exception as exc:  # pragma: no cover - WS-A package unavailable
        logger.warning("signals.market_regime.sector_map unavailable (%s)", exc)
        return None
    try:
        return sector_etf_for(ticker)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[%s] sector_etf_for failed: %s", ticker, exc)
        return None


def _asof_join_regime(
    bar_index: pd.DatetimeIndex, table: pd.DataFrame | None, value_cols: list[str],
) -> pd.DataFrame:
    """Backward as-of join of a regime/sector table onto 4H bar timestamps.

    Each bar receives the last table row whose ``available_at`` is <= the
    bar's own UTC timestamp — never a future row. Same backward-as-of intent
    as ``_map_to_4h`` below, applied to ``available_at`` instead of calendar
    ``date``: a naive date merge would leak (a 09:30 ET bar on session D must
    see only D-1's regime row, not same-day D, since D's row isn't knowable
    until D's close + settle margin).

    Returns a frame aligned to ``bar_index`` with ``value_cols`` plus
    ``_regime_date`` (the matched row's session date, used for staleness);
    all NaN/NaT if ``table`` is ``None``, empty, or every row's
    ``available_at`` is after every bar timestamp.
    """
    out_cols = list(value_cols) + ["_regime_date"]
    if table is None or table.empty:
        empty = pd.DataFrame(index=bar_index)
        for col in value_cols:
            empty[col] = np.nan
        empty["_regime_date"] = pd.Series(pd.NaT, index=bar_index, dtype="datetime64[ns]")
        return empty[out_cols]

    left = pd.DataFrame({"__bar_ts": pd.DatetimeIndex(bar_index).tz_convert("UTC")})
    left["__order"] = np.arange(len(left))
    left = left.sort_values("__bar_ts")

    right = table[["available_at", "date"] + list(value_cols)].copy()
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True)
    right = right.sort_values("available_at").rename(columns={"date": "_regime_date"})

    merged = pd.merge_asof(
        left, right, left_on="__bar_ts", right_on="available_at", direction="backward",
    )
    merged = merged.sort_values("__order")
    merged.index = bar_index
    return merged[out_cols]


def _build_metadata_encodings(meta: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return stable categorical encodings for sector, cap bucket, and asset type."""
    if meta is None or meta.empty:
        return {"Unknown": 0}, {"Unknown": 0}, {"Stock": 0}
    sectors = sorted(meta.get("sector", pd.Series(["Unknown"])).fillna("Unknown").astype(str).unique())
    caps = sorted(meta.get("market_cap_bucket", pd.Series(["Unknown"])).fillna("Unknown").astype(str).unique())
    types = sorted(meta.get("type", pd.Series(["Stock"])).fillna("Stock").astype(str).unique())
    if "Unknown" not in sectors:
        sectors.append("Unknown")
    if "Unknown" not in caps:
        caps.append("Unknown")
    if "Stock" not in types:
        types.append("Stock")
    return (
        {name: i for i, name in enumerate(sectors)},
        {name: i for i, name in enumerate(caps)},
        {name: i for i, name in enumerate(types)},
    )


def _metadata_for_ticker(
    *,
    ticker: str,
    meta_lookup: dict[str, dict] | None,
    sector_enc: dict[str, int],
    cap_enc: dict[str, int],
    type_enc: dict[str, int],
) -> dict[str, float]:
    row = (meta_lookup or {}).get(ticker.upper(), {})
    sector = str(row.get("sector", "Unknown"))
    cap = str(row.get("market_cap_bucket", "Unknown"))
    asset_type = str(row.get("type", "Stock"))
    is_etf = 1.0 if asset_type.upper() == "ETF" else 0.0
    return {
        "sector_id": float(sector_enc.get(sector, sector_enc.get("Unknown", 0))),
        "market_cap_bucket": float(cap_enc.get(cap, cap_enc.get("Unknown", 0))),
        "asset_type": float(type_enc.get(asset_type, type_enc.get("Stock", 0))),
        "is_etf": is_etf,
    }


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, lo, c_p = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - lo), (h - c_p).abs(), (lo - c_p).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    try:
        import pandas_ta as ta
        out = ta.adx(df["high"], df["low"], df["close"], length=length)
        if out is None or out.empty:
            return pd.Series(np.nan, index=df.index)
        return out.iloc[:, 0].clip(0, 100)
    except Exception:
        return pd.Series(np.nan, index=df.index)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    try:
        import pandas_ta as ta
        out = ta.rsi(close, length=length)
        return out if out is not None else pd.Series(np.nan, index=close.index)
    except Exception:
        return pd.Series(np.nan, index=close.index)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    try:
        import pandas_ta as ta
        out = ta.macd(close, fast=fast, slow=slow, signal=signal)
        if out is None or out.empty:
            return pd.Series(np.nan, index=close.index)
        col = [c for c in out.columns if c.startswith("MACDh")]
        if not col:
            return pd.Series(np.nan, index=close.index)
        return out[col[0]]
    except Exception:
        return pd.Series(np.nan, index=close.index)


def _bars_since_true(mask: pd.Series, *, cap: int = 252) -> pd.Series:
    """Number of bars since a boolean condition last occurred, capped."""
    out: list[float] = []
    last_seen: int | None = None
    values = mask.fillna(False).astype(bool).to_numpy()
    for i, flag in enumerate(values):
        if flag:
            last_seen = i
            out.append(0.0)
        elif last_seen is None:
            # Condition has never occurred yet (e.g. a chronic downtrend that
            # never prints a new 52w high). Treat as maximally stale rather than
            # NaN so a single never-true column doesn't drop the whole ticker
            # in build_training_matrix's dropna(how="any").
            out.append(float(cap))
        else:
            out.append(float(min(i - last_seen, cap)))
    return pd.Series(out, index=mask.index)


# ---------------------------------------------------------------------------
# Per-ticker builder
# ---------------------------------------------------------------------------

def build_ticker_features_4h(
    *,
    ticker: str,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame | None,
    ctx_4h: dict[str, pd.DataFrame],
    ticker_meta: dict[str, float] | None = None,
) -> pd.DataFrame | None:
    """
    Build 4H features for one ticker.

    Parameters
    ----------
    df_4h : 4H OHLCV with UTC DatetimeIndex (from data.load_bars.load_4h)
    df_1d : daily OHLCV (used for 52-week high distance / daily trend)
    ctx_4h: {"SPY": df, "QQQ": df, "IWM": df, "VIXY": df, "XLK": df, ...} on 4H
    """
    if df_4h is None or len(df_4h) < MIN_4H_BARS:
        return None

    df = df_4h.copy()
    df.columns = [c.lower() for c in df.columns]
    if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        return None

    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # --- ATR (used everywhere)
    atr14 = _atr(df, 14)
    atr60 = _atr(df, 60)
    df["atr_14"] = atr14
    df["atr_pct_14"] = atr14 / c.replace(0, np.nan)

    # --- TREND ---
    ema10 = _ema(c, 10)
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema100 = _ema(c, 100)
    df["ema_dist_10"] = ((c - ema10) / atr14).clip(-10, 10)
    df["ema_dist_20"] = ((c - ema20) / atr14).clip(-10, 10)
    df["ema_dist_50"] = ((c - ema50) / atr14).clip(-10, 10)
    df["ema_dist_100"] = ((c - ema100) / atr14).clip(-15, 15)
    df["ema_slope_10"] = ema10.pct_change(1, fill_method=None)
    df["ema_slope_20"] = ema20.pct_change(1, fill_method=None)
    df["ema_slope_50"] = ema50.pct_change(1, fill_method=None)
    df["adx_14"] = _adx(df, 14)
    # Stack: bit0 ema10>ema20, bit1 ema20>ema50, bit2 ema50>ema100
    b0 = (ema10 > ema20).astype(int)
    b1 = (ema20 > ema50).astype(int)
    b2 = (ema50 > ema100).astype(int)
    df["ema_stack_4"] = b0 + 2 * b1 + 4 * b2

    # --- MOMENTUM ---
    for n in (1, 3, 5, 10, 20):
        df[f"ret_{n}"] = c.pct_change(n, fill_method=None)
    df["rsi_14"] = _rsi(c, 14)
    df["macd_hist_norm"] = (_macd_hist(c) / atr14).clip(-5, 5)

    # --- VOLATILITY ---
    df["atr_expand_14_60"] = (atr14 / atr60).clip(0, 5)
    # Guard zero/negative closes (halted or bad bars): a 0 price makes the ratio 0
    # → log(0) = -inf and floods the logs with divide-by-zero RuntimeWarnings. A
    # return across a non-positive price is undefined, so emit NaN instead.
    _c_ratio = c / c.shift(1)
    log_ret = np.log(_c_ratio.where(_c_ratio > 0))
    rv5 = log_ret.rolling(5).std()
    rv20 = log_ret.rolling(20).std()
    rv60 = log_ret.rolling(60).std()
    df["realized_vol_5"] = rv5
    df["realized_vol_20"] = rv20
    df["vol_regime_5_60"] = (rv5 / rv60.replace(0, np.nan)).clip(0, 5)
    df["vol_of_vol_20"] = rv5.rolling(20).std()

    # --- VOLUME ---
    vol_mean20 = v.rolling(20).mean().replace(0, np.nan)
    df["rvol_20"] = (v / vol_mean20).clip(0, 20)
    df["dollar_volume"] = (v * c)
    df["dollar_vol_pctile_252"] = (v * c).rolling(252, min_periods=50).rank(pct=True)
    df["dollar_vol_surge_20"] = ((v * c) / (v * c).rolling(20).mean().replace(0, np.nan)).clip(0, 20)
    vol_std20 = v.rolling(20).std().replace(0, np.nan)
    dollar_vol = v * c
    dollar_vol_mean60 = dollar_vol.rolling(60).mean()
    dollar_vol_std60 = dollar_vol.rolling(60).std().replace(0, np.nan)
    df["volume_z_20"] = ((v - vol_mean20) / vol_std20).clip(-5, 10)
    df["dollar_vol_z_60"] = ((dollar_vol - dollar_vol_mean60) / dollar_vol_std60).clip(-5, 10)
    df["volume_spike_20"] = (df["rvol_20"] >= 2.0).astype(float)
    df["bars_since_volume_spike"] = _bars_since_true(df["volume_spike_20"] > 0, cap=252)

    # --- STRUCTURE ---
    # 52-week high distance — at 2 4H bars/day * ~252 trading days = ~504 bars.
    # min_periods=252 (~6 months) lets young listings compute a partial-window
    # "high so far" instead of NaN (NaN would drop the whole ticker in the
    # build_training_matrix dropna). insufficient_52w_history flags those rows so
    # the model can discount the approximate value.
    roll_high_252d = h.rolling(504, min_periods=252).max()
    roll_low_252d = lo.rolling(504, min_periods=252).min()
    df["insufficient_52w_history"] = (c.rolling(504, min_periods=1).count() < 504).astype(float)
    df["dist_to_52w_high_atr"] = ((roll_high_252d - c) / atr14).clip(0, 50)
    df["dist_to_52w_low_atr"] = ((c - roll_low_252d) / atr14).clip(0, 50)

    # 20-bar range position
    rh20 = h.rolling(20).max()
    rl20 = lo.rolling(20).min()
    rr20 = (rh20 - rl20).replace(0, np.nan)
    df["range_pos_20"] = ((c - rl20) / rr20).clip(0, 1)
    df["dist_20bar_high_atr"] = ((rh20 - c) / atr14).clip(0, 20)

    # Compression: vol_5 / vol_20 — lower = tighter
    df["compression_5_20"] = (rv5 / rv20.replace(0, np.nan)).clip(0, 5)
    df["is_compressed_5_20"] = (df["compression_5_20"] <= 0.65).astype(float)
    df["compression_count_20"] = df["is_compressed_5_20"].rolling(20).sum()

    # Breakout flag — close > 20-bar high (causal: shifted by 1 to avoid same-bar)
    df["breakout_20"] = (c > rh20.shift(1)).astype(float)
    df["breakout_attempts_60"] = df["breakout_20"].rolling(60).sum()
    df["bars_since_breakout_20"] = _bars_since_true(df["breakout_20"] > 0, cap=252)

    # Base depth: 60-bar drawdown from rolling 60-bar high
    rh60 = h.rolling(60).max()
    df["drawdown_from_60h"] = ((rh60 - c) / rh60.replace(0, np.nan)).clip(0, 1)
    base_range_60 = (rh60 - lo.rolling(60).min()).replace(0, np.nan)
    df["base_range_60_atr"] = (base_range_60 / atr14.replace(0, np.nan)).clip(0, 100)
    df["base_position_60"] = ((c - lo.rolling(60).min()) / base_range_60).clip(0, 1)
    df["close_tightness_10"] = (
        (c.diff().abs().rolling(10).mean() / atr14.replace(0, np.nan))
        .clip(0, 10)
    )
    df["range_contraction_20_60"] = (
        (h - lo).rolling(20).mean() / (h - lo).rolling(60).mean().replace(0, np.nan)
    ).clip(0, 5)
    df["near_52w_high"] = (df["dist_to_52w_high_atr"] <= 2.0).astype(float)
    df["bars_since_52w_high"] = _bars_since_true(c >= roll_high_252d.shift(1), cap=504)
    df["open_to_prev_close_ret"] = (df["open"] / c.shift(1).replace(0, np.nan) - 1.0).clip(-1, 1)

    # --- RELATIVE STRENGTH ---
    def _ctx_close(name: str) -> pd.Series:
        df_ctx = ctx_4h.get(name)
        if df_ctx is None or df_ctx.empty:
            return pd.Series(np.nan, index=df.index)
        return df_ctx["close"].reindex(df.index, method="ffill")

    spy_c = _ctx_close("SPY")
    qqq_c = _ctx_close("QQQ")
    iwm_c = _ctx_close("IWM")
    sector_etf = _sector_etf_for(ticker)
    sec_c = _ctx_close(sector_etf)

    for n in (1, 5, 20):
        spy_ret = spy_c.pct_change(n, fill_method=None)
        qqq_ret = qqq_c.pct_change(n, fill_method=None)
        sec_ret = sec_c.pct_change(n, fill_method=None)
        df[f"rs_spy_{n}"] = (df[f"ret_{n}"] - spy_ret).clip(-1, 1) if f"ret_{n}" in df.columns else np.nan
        if n in (1, 5, 20):
            df[f"rs_qqq_{n}"] = (df[f"ret_{n}"] - qqq_ret).clip(-1, 1)
        if n in (5, 20):
            df[f"rs_sector_{n}"] = (df[f"ret_{n}"] - sec_ret).clip(-1, 1)

    # Beta & corr to SPY (60-bar)
    own_lr = log_ret
    _spy_ratio = spy_c / spy_c.shift(1)
    spy_lr = np.log(_spy_ratio.where(_spy_ratio > 0))
    if ticker == "SPY":
        df["beta_spy_60"] = 1.0
        df["corr_spy_60"] = 1.0
    else:
        cov60 = own_lr.rolling(60).cov(spy_lr)
        var60 = spy_lr.rolling(60).var().replace(0, np.nan)
        df["beta_spy_60"] = (cov60 / var60).clip(-5, 5)
        df["corr_spy_60"] = own_lr.rolling(60).corr(spy_lr)

    # --- REGIME ---
    spy_trend = np.sign(_ema(spy_c, 20) - _ema(spy_c, 50))
    df["regime_spy_trend"] = spy_trend
    spy_ret_20 = spy_c.pct_change(20, fill_method=None)
    df["regime_spy_ret_20"] = spy_ret_20.fillna(0)

    vixy_c = _ctx_close("VIXY")
    if vixy_c.notna().any():
        vixy_z = (vixy_c - vixy_c.rolling(60).mean()) / vixy_c.rolling(60).std().replace(0, np.nan)
        df["regime_vix_z"] = vixy_z.clip(-5, 5)
        df["regime_vix_high"] = (vixy_z > 1.0).astype(float)
    else:
        df["regime_vix_z"] = 0.0
        df["regime_vix_high"] = 0.0

    # --- DAILY HTF context (shifted -1 to avoid leakage) ---
    if df_1d is not None and len(df_1d) > 60:
        d = df_1d.copy()
        d.columns = [x.lower() for x in d.columns]
        dc = d["close"]
        d_ret_1 = dc.pct_change(1, fill_method=None)
        d_ret_5 = dc.pct_change(5, fill_method=None)
        d_atr = _atr(d, 14)
        d_atr_pct = (d_atr / dc).shift(1)
        d_rsi = _rsi(dc, 14).shift(1)
        d_ema20 = _ema(dc, 20)
        d_ema50 = _ema(dc, 50)
        d_ema100 = _ema(dc, 100)
        d_ema200 = _ema(dc, 200)
        d_trend = np.sign(d_ema20 - d_ema50).shift(1)
        d_stack = (
            (d_ema20 > d_ema50).astype(int)
            + 2 * (d_ema50 > d_ema100).astype(int)
            + 4 * (d_ema100 > d_ema200).astype(int)
        ).shift(1)
        d_high_252 = d["high"].rolling(252, min_periods=126).max()  # ~6mo partial window for young names
        d_dist_52w = ((d_high_252 - dc) / d_atr.replace(0, np.nan)).shift(1).clip(0, 50)
        d_dist_200 = ((dc - d_ema200) / d_atr.replace(0, np.nan)).shift(1).clip(-50, 50)
        d_new_high_252 = (dc >= d_high_252.shift(1)).astype(float)
        d_gap = (d["open"] / dc.shift(1).replace(0, np.nan) - 1.0).clip(-1, 1)
        d_gap_z_60 = (
            (d_gap - d_gap.rolling(60).mean()) / d_gap.rolling(60).std().replace(0, np.nan)
        ).clip(-10, 10)

        weekly_close = dc.resample("W-FRI").last()
        weekly_ema10 = _ema(weekly_close, 10)
        weekly_ema30 = _ema(weekly_close, 30)
        weekly = pd.DataFrame({
            "weekly_ret_1": weekly_close.pct_change(1, fill_method=None).shift(1),
            "weekly_ret_4": weekly_close.pct_change(4, fill_method=None).shift(1),
            "weekly_trend_state": np.sign(weekly_ema10 - weekly_ema30).shift(1),
        })
        weekly_daily = weekly.reindex(d.index, method="ffill")

        # Map daily series to 4H bars by date
        ny = df.index.tz_convert("America/New_York")
        date_index = pd.Series(ny.date, index=df.index)
        d.index = pd.to_datetime(d.index)
        d_dates = pd.Series(d.index.tz_convert("America/New_York").date if d.index.tz else pd.to_datetime(d.index).date,
                            index=d.index)

        def _map_to_4h(daily_series: pd.Series) -> pd.Series:
            mapping = pd.Series(
                daily_series.values,
                index=pd.to_datetime(pd.Index(d_dates.values)),
            )
            mapping = mapping[~mapping.index.duplicated(keep="last")].sort_index()
            target = pd.to_datetime(pd.Index(date_index.values))
            # Backward as-of: a 4H bar whose calendar date has no daily row yet
            # (the live edge, where today's daily bar isn't cached) carries the most
            # recent completed daily value instead of NaN. Identical to exact-date
            # match whenever the daily series covers the bar's date (all historical
            # dates), so backtest/training features are unchanged. Daily inputs are
            # already .shift(1)-lagged above, so this stays leak-free.
            pos = mapping.index.searchsorted(target, side="right") - 1
            vals = np.where(pos >= 0, mapping.to_numpy()[pos.clip(min=0)], np.nan)
            return pd.Series(vals, index=df.index, dtype=float)

        df["daily_ret_1"] = _map_to_4h(d_ret_1.shift(1))
        df["daily_ret_5"] = _map_to_4h(d_ret_5.shift(1))
        df["daily_atr_pct"] = _map_to_4h(d_atr_pct)
        df["daily_rsi_14"] = _map_to_4h(d_rsi)
        df["daily_trend_state"] = _map_to_4h(d_trend)
        df["daily_dist_52w_atr"] = _map_to_4h(d_dist_52w)
        df["daily_ema_stack"] = _map_to_4h(d_stack)
        df["daily_dist_200dma_atr"] = _map_to_4h(d_dist_200)
        df["daily_new_high_252"] = _map_to_4h(d_new_high_252.shift(1))
        df["daily_gap_pct"] = _map_to_4h(d_gap)
        df["daily_gap_z_60"] = _map_to_4h(d_gap_z_60)
        df["weekly_ret_1"] = _map_to_4h(weekly_daily["weekly_ret_1"])
        df["weekly_ret_4"] = _map_to_4h(weekly_daily["weekly_ret_4"])
        df["weekly_trend_state"] = _map_to_4h(weekly_daily["weekly_trend_state"])
    else:
        df["daily_ret_1"] = np.nan
        df["daily_ret_5"] = np.nan
        df["daily_atr_pct"] = np.nan
        df["daily_rsi_14"] = np.nan
        df["daily_trend_state"] = np.nan
        df["daily_dist_52w_atr"] = np.nan
        df["daily_ema_stack"] = np.nan
        df["daily_dist_200dma_atr"] = np.nan
        df["daily_new_high_252"] = np.nan
        df["daily_gap_pct"] = np.nan
        df["daily_gap_z_60"] = np.nan
        df["weekly_ret_1"] = np.nan
        df["weekly_ret_4"] = np.nan
        df["weekly_trend_state"] = np.nan

    # --- MARKET REGIME / SECTOR CONTEXT (WS-B) ---
    # Additive, opt-in-only columns (see REGIME_FEATURE_COLUMNS_4H below) —
    # NOT part of FEATURE_COLUMNS_4H / the deployed training matrix (plan D3).
    regime_table = _load_daily_regime_table()
    regime_join = _asof_join_regime(
        df.index, regime_table,
        ["risk_appetite_z", "liquidity_stress_z", "credit_risk_z", "sector_dispersion_raw", "breadth_z"],
    )
    df["risk_appetite_z"] = regime_join["risk_appetite_z"]
    df["liquidity_stress_z"] = regime_join["liquidity_stress_z"]
    df["credit_risk_z"] = regime_join["credit_risk_z"]
    df["sector_dispersion_21d"] = regime_join["sector_dispersion_raw"]
    # Genuine market-wide breadth (plan §4 WS-A: fraction of the 11 sector
    # ETFs above their own 20d/50d SMA, z-scored) -- named `market_breadth_z`,
    # never `sector_breadth_*`, so it can't be confused with the per-ticker
    # `sector_above_20d/50d` flags below (which are NOT breadth: they're one
    # sector's own above/below-SMA state, not a cross-sector fraction).
    df["market_breadth_z"] = regime_join["breadth_z"]

    bar_session_date = pd.to_datetime(
        pd.Series(df.index.tz_convert("America/New_York").date, index=df.index)
    )
    regime_row_date = pd.to_datetime(regime_join["_regime_date"])
    df["regime_available"] = regime_row_date.notna().astype(float)
    df["regime_stale_days"] = (bar_session_date - regime_row_date).dt.days.astype(float)

    sector_etf_ctx = _regime_sector_etf_for(ticker)
    sector_table = _load_sector_state_table()
    sector_rows = (
        sector_table[sector_table["sector_etf"] == sector_etf_ctx]
        if sector_etf_ctx is not None and sector_table is not None
        else None
    )
    sector_join = _asof_join_regime(
        df.index, sector_rows,
        ["excess_21d", "excess_63d", "rank_21d", "rank_63d", "rs_accel", "above_20d", "above_50d"],
    )
    df["sector_excess_21d"] = sector_join["excess_21d"]
    df["sector_excess_63d"] = sector_join["excess_63d"]
    df["sector_rank_21d"] = sector_join["rank_21d"]
    df["sector_rank_63d"] = sector_join["rank_63d"]
    df["sector_rs_accel"] = sector_join["rs_accel"]
    # NOT breadth (renamed from sector_breadth_20/50): this is the ticker's OWN
    # resolved sector ETF's above/below-its-own-SMA state (sector_state.parquet's
    # above_20d/above_50d), a per-ticker binary that varies by sector on a given
    # bar -- see `market_breadth_z` above for the genuine cross-sector fraction.
    df["sector_above_20d"] = sector_join["above_20d"]
    df["sector_above_50d"] = sector_join["above_50d"]

    # Interactions (plan §4 WS-B). NaN-propagating; no extra clipping beyond
    # what rs_spy_20 / breakout_20 already apply upstream.
    df["int_rs_spy_20_x_sector_rank_63d"] = df["rs_spy_20"] * df["sector_rank_63d"]
    df["int_ret_20_x_risk_appetite_z"] = df["ret_20"] * df["risk_appetite_z"]
    df["int_breakout_20_x_liquidity_stress_z"] = df["breakout_20"] * df["liquidity_stress_z"]

    # Loud (throttled) staleness warning — constraint: the live path must not
    # silently score on a stale regime table. Warn, never raise (must not
    # break the live runner); regime_stale_days is also exposed as a feature.
    if len(df) and pd.notna(df["regime_stale_days"].iloc[-1]):
        last_stale = int(df["regime_stale_days"].iloc[-1])
        if last_stale > _REGIME_STALE_WARN_DAYS:
            warn_key = (ticker, last_stale // _REGIME_STALE_WARN_DAYS)
            if warn_key not in _REGIME_WARNED_STALE:
                logger.warning(
                    "[%s] most recent 4H bar's joined market-regime row is %d days stale "
                    "(> %d day tolerance) — daily_regime.parquet likely needs a rebuild "
                    "(`python -m signals.market_regime.build`). Scoring proceeds on stale "
                    "regime context; regime_stale_days=%d is exposed as a feature.",
                    ticker, last_stale, _REGIME_STALE_WARN_DAYS, last_stale,
                )
                _REGIME_WARNED_STALE.add(warn_key)

    # --- BAR TIME (within session) ---
    ny = df.index.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    df["is_morning_4h"] = (minutes < 13 * 60 + 30).astype(float)
    df["dow"] = pd.Series(ny.dayofweek.astype(float), index=df.index)
    month = pd.Series(ny.month.astype(float), index=df.index)
    quarter = pd.Series(ny.quarter.astype(float), index=df.index)
    week = pd.Series(ny.isocalendar().week.astype(float), index=df.index)
    day = pd.Series(ny.day.astype(float), index=df.index)
    df["month"] = month
    df["quarter"] = quarter
    df["week_of_year"] = week
    df["day_of_month"] = day
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 5.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 5.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.0)
    df["is_month_start"] = pd.Series(ny.is_month_start.astype(float), index=df.index)
    df["is_month_end"] = pd.Series(ny.is_month_end.astype(float), index=df.index)

    # --- STATIC IDENTITY / TRADABILITY CONTEXT ---
    # These are intentionally simple categorical/numeric fields, mirroring the
    # multi-ticker swing model's ability to separate ETFs, mega caps, small caps,
    # and unknown scanner names.
    ticker_meta = ticker_meta or {}
    df["sector_id"] = float(ticker_meta.get("sector_id", 0.0))
    df["market_cap_bucket"] = float(ticker_meta.get("market_cap_bucket", 0.0))
    df["asset_type"] = float(ticker_meta.get("asset_type", 0.0))
    df["is_etf"] = float(ticker_meta.get("is_etf", 0.0))
    df["low_price_flag"] = (c < 5.0).astype(float)

    return df


# ---------------------------------------------------------------------------
# Final ordered feature list (used by training + inference)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS_4H: list[str] = [
    # ATR baseline
    "atr_pct_14", "atr_expand_14_60",
    # Trend
    "ema_dist_10", "ema_dist_20", "ema_dist_50", "ema_dist_100",
    "ema_slope_10", "ema_slope_20", "ema_slope_50",
    "adx_14", "ema_stack_4",
    # Momentum
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "rsi_14", "macd_hist_norm",
    # Volatility
    "realized_vol_5", "realized_vol_20", "vol_regime_5_60", "vol_of_vol_20",
    # Volume
    "rvol_20", "dollar_vol_pctile_252", "dollar_vol_surge_20",
    "volume_z_20", "dollar_vol_z_60", "volume_spike_20", "bars_since_volume_spike",
    # Structure
    "dist_to_52w_high_atr", "dist_to_52w_low_atr",
    "range_pos_20", "dist_20bar_high_atr",
    "compression_5_20", "breakout_20", "drawdown_from_60h",
    "is_compressed_5_20", "compression_count_20",
    "breakout_attempts_60", "bars_since_breakout_20",
    "base_range_60_atr", "base_position_60", "close_tightness_10",
    "range_contraction_20_60", "near_52w_high", "bars_since_52w_high",
    "insufficient_52w_history",
    "open_to_prev_close_ret",
    # Relative strength
    "rs_spy_1", "rs_spy_5", "rs_spy_20",
    "rs_qqq_1", "rs_qqq_5", "rs_qqq_20",
    "rs_sector_5", "rs_sector_20",
    "beta_spy_60", "corr_spy_60",
    # Regime
    "regime_spy_trend", "regime_spy_ret_20",
    "regime_vix_z", "regime_vix_high",
    # Daily HTF
    "daily_ret_1", "daily_ret_5", "daily_atr_pct", "daily_rsi_14",
    "daily_trend_state", "daily_dist_52w_atr",
    "daily_ema_stack", "daily_dist_200dma_atr", "daily_new_high_252",
    "daily_gap_pct", "daily_gap_z_60",
    "weekly_ret_1", "weekly_ret_4", "weekly_trend_state",
    # Time
    "is_morning_4h", "dow", "month", "quarter", "week_of_year", "day_of_month",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos",
    "is_month_start", "is_month_end",
    # Earnings proximity (point-in-time from the earnings calendar; known at decision time)
    "days_to_earnings", "days_since_earnings",
    "is_pre_earnings_3d", "is_post_earnings_3d", "earnings_in_fwd_window",
    # Identity / tradability context
    "sector_id", "market_cap_bucket", "asset_type", "is_etf", "low_price_flag",
    # Cross-sectional ranks added after all tickers are combined
    "xsec_ret_5_rank", "xsec_ret_20_rank", "xsec_rs_spy_20_rank",
    "xsec_rvol_20_rank", "xsec_dollar_vol_surge_20_rank",
    "xsec_atr_expand_rank", "xsec_atr_pct_rank", "xsec_near_high_rank",
]


# ---------------------------------------------------------------------------
# Market-regime / sector-context features (WS-B). Deliberately NOT part of
# FEATURE_COLUMNS_4H (plan D3): daily_regime.parquet starts 2020-07-27 and its
# z-scores need a 252-session warmup, so these columns are NaN across a real,
# large slice of history and for any ticker whose sector can't be resolved.
# FEATURE_COLUMNS_4H feeds build_training_matrix's
# dropna(subset=feature_cols, how="any") (expansion_labels.py); if these were
# in that list, that dropna would silently delete most of the pre-2021
# training set. Opt in explicitly (e.g. an ablation trainer) via this list.
# ---------------------------------------------------------------------------
REGIME_FEATURE_COLUMNS_4H: list[str] = [
    # Sector context (signals.market_regime.sector_state, joined on the
    # ticker's curated-map sector ETF). sector_above_20d/50d are NOT breadth
    # -- they're one sector's own above/below-its-SMA state, a per-ticker
    # binary that varies by sector on a given bar. See market_breadth_z below
    # for the genuine cross-sector fraction (was misnamed sector_breadth_20/50
    # until this was caught during verification; renamed to avoid confusion
    # with the real breadth_z composite).
    "sector_excess_21d", "sector_excess_63d",
    "sector_rank_21d", "sector_rank_63d", "sector_rs_accel",
    "sector_above_20d", "sector_above_50d",
    # Daily market-wide regime composites (signals.market_regime.daily_regime)
    "sector_dispersion_21d",
    "risk_appetite_z", "liquidity_stress_z", "credit_risk_z", "market_breadth_z",
    "regime_available", "regime_stale_days",
    # Interactions
    "int_rs_spy_20_x_sector_rank_63d",
    "int_ret_20_x_risk_appetite_z",
    "int_breakout_20_x_liquidity_stress_z",
]


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def _add_cross_sectional_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Add same-timestamp ranks across the active ticker universe."""
    if combined.empty or not isinstance(combined.index, pd.MultiIndex):
        return combined
    ts_level = combined.index.names[0]
    rank_specs = {
        "ret_5": "xsec_ret_5_rank",
        "ret_20": "xsec_ret_20_rank",
        "rs_spy_20": "xsec_rs_spy_20_rank",
        "rvol_20": "xsec_rvol_20_rank",
        "dollar_vol_surge_20": "xsec_dollar_vol_surge_20_rank",
        "atr_expand_14_60": "xsec_atr_expand_rank",
        "atr_pct_14": "xsec_atr_pct_rank",
    }
    for source, target in rank_specs.items():
        if source in combined.columns:
            combined[target] = combined.groupby(level=ts_level)[source].rank(pct=True)
        else:
            combined[target] = np.nan
    if "dist_to_52w_high_atr" in combined.columns:
        # Lower distance means closer to a high, so rank the negative distance.
        combined["xsec_near_high_rank"] = (
            (-combined["dist_to_52w_high_atr"]).groupby(level=ts_level).rank(pct=True)
        )
    else:
        combined["xsec_near_high_rank"] = np.nan
    return combined

def _load_context_4h() -> dict[str, pd.DataFrame]:
    ctx: dict[str, pd.DataFrame] = {}
    for sym in list(CONTEXT_TICKERS) + list(SECTOR_ETFS):
        path = RAW_4H_DIR / f"{sym}.parquet"
        if not path.exists():
            logger.warning("Context 4h missing for %s — features that need it will be NaN", sym)
            continue
        try:
            ctx[sym] = load_4h(sym)
        except Exception as exc:
            logger.warning("[%s] load_4h failed: %s", sym, exc)
    return ctx


# Momentum's expansion label looks ~25x4H bars ≈ 10 trading days forward.
_EARNINGS_FWD_WINDOW_DAYS = 10


def _add_earnings_features_4h(combined: pd.DataFrame) -> pd.DataFrame:
    """Join point-in-time earnings-proximity features onto the (timestamp, ticker) matrix.

    Earnings act as a binary wall on momentum (kill / continue / reset), so the
    model gets days_to/since_earnings, pre/post flags, and an earnings_in_fwd_window
    flag marking bars whose forward label straddles a report (event-driven, not
    momentum-driven). No-op (NaN columns) if the calendar isn't fetched yet.
    """
    try:
        from signals.events.earnings_calendar import add_earnings_features
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("earnings_calendar unavailable (%s) — skipping earnings features", exc)
        return combined

    idx_names = list(combined.index.names)
    flat = combined.reset_index()
    ts_col = idx_names[0] if idx_names and idx_names[0] in flat.columns else "timestamp"
    flat["__date"] = pd.to_datetime(flat[ts_col], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    flat = add_earnings_features(
        flat, date_col="__date", ticker_col="ticker",
        fwd_window_days=_EARNINGS_FWD_WINDOW_DAYS,
    )
    flat = flat.drop(columns="__date")
    out = flat.set_index(idx_names).sort_index()
    added = ["days_to_earnings", "days_since_earnings", "is_pre_earnings_3d", "is_post_earnings_3d", "earnings_in_fwd_window"]
    cov = out["days_to_earnings"].notna().mean() if "days_to_earnings" in out.columns else 0.0
    logger.info("Added earnings features %s (days_to_earnings coverage %.1f%%)", added, 100 * cov)
    return out


def build_all_features_4h(
    *,
    tickers: Iterable[str],
    out_path: Path = FEATURES_COMBINED,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Build 4H features for every ticker that has cached 4H data, persist
    per-ticker parquets, and combine into one (timestamp, ticker)
    MultiIndex parquet.
    """
    PROCESSED_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        logger.info("Features already exist at %s. Use force=True to rebuild.", out_path)
        return pd.read_parquet(out_path)

    ctx_4h = _load_context_4h()
    meta = load_candidate_metadata()
    sector_enc, cap_enc, type_enc = _build_metadata_encodings(meta)
    meta_lookup = meta.set_index("ticker").to_dict(orient="index") if not meta.empty else {}
    frames: list[pd.DataFrame] = []
    tickers = list(tickers)
    for i, t in enumerate(tickers, 1):
        per_path = PROCESSED_FEAT_DIR / f"{t}_features.parquet"
        if per_path.exists() and not force:
            df_feat = pd.read_parquet(per_path)
            ticker_meta = _metadata_for_ticker(
                ticker=t,
                meta_lookup=meta_lookup,
                sector_enc=sector_enc,
                cap_enc=cap_enc,
                type_enc=type_enc,
            )
            for col, value in ticker_meta.items():
                if col not in df_feat.columns:
                    df_feat[col] = value
            if "low_price_flag" not in df_feat.columns and "close" in df_feat.columns:
                df_feat["low_price_flag"] = (df_feat["close"] < 5.0).astype(float)
            df_feat["ticker"] = t
            frames.append(df_feat)
            continue

        try:
            df_4h = load_4h(t)
        except FileNotFoundError:
            logger.info("[%s] no 4h data — skip", t)
            continue
        try:
            df_1d = load_1d(t)
        except FileNotFoundError:
            df_1d = None

        feats = build_ticker_features_4h(
            ticker=t,
            df_4h=df_4h,
            df_1d=df_1d,
            ctx_4h=ctx_4h,
            ticker_meta=_metadata_for_ticker(
                ticker=t,
                meta_lookup=meta_lookup,
                sector_enc=sector_enc,
                cap_enc=cap_enc,
                type_enc=type_enc,
            ),
        )
        if feats is None:
            logger.info("[%s] insufficient bars or missing OHLCV — skip", t)
            continue
        feats.to_parquet(per_path)
        feats["ticker"] = t
        frames.append(feats)
        if i % 25 == 0:
            logger.info("(%d/%d) features built for %s", i, len(tickers), t)

    if not frames:
        logger.error("No features built.")
        return None

    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().rename(columns={"index": "timestamp"})
    ts_col = "timestamp" if "timestamp" in combined.columns else combined.columns[0]
    combined = combined.set_index([ts_col, "ticker"]).sort_index()
    combined = _add_cross_sectional_features(combined)
    combined = _add_earnings_features_4h(combined)
    combined.to_parquet(out_path)
    logger.info("Saved combined 4H features (%d rows) -> %s", len(combined), out_path)
    return combined
