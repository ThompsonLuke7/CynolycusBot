#!/usr/bin/env python
"""Phase 3 counterfactual sweep for the options-instrument-routing experiment.

Plan: docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md
(Phase 3). Pre-registration: docs/superpowers/plans/2026-07-25-options-routing-preregistration.md.
Gate results this script's scope depends on: research/options_experiment/01_gate_g1_verdict.md
(mid-priced results forbidden; relative ranking only) and 02_spread_model.md (Roll spread
estimator scaled to the realized-level anchor; Corwin-Schultz excluded).

Only the three powered modules from the pre-registration's power table may be swept:
multi_ticker_swing (30m), multi_ticker_swing_htf, momentum_expansion. dealer_ranker/
meta_ranker/intraday_structure are excluded entirely (declared underpowered up front).

Replays each sampled spine trade through B0 (long shares) and seven option structures
from research/options_lab/strategies.py, at two DTE buckets (near = nearest listed expiry
with DTE >= holding_period + 5 calendar days; far = nearest listed expiry with DTE >=
2x that) and two sizing modes (matched notional, matched max-loss), pricing every leg from
real cached Alpaca option bars (chain_cache.fetch_bars) and estimating spread cost from real
trade prints (spread_estimators.combine_spread_estimates). Never touches
research/options_lab/*.py's own logic -- this script is a consumer only.

No signal re-tuning: entry/exit timestamps and prices are frozen from the spine, taken
verbatim from the modules' own backtests. Only the instrument changes.

Sampling: stratified by (module, week, direction) so every calendar week is represented
(pre-registration's block-bootstrap unit), targeting ~2,000 trades/module. Recorded once to
research/options_experiment/data/phase3_sample.parquet (seed + coverage in
phase3_sample_coverage.json) so a restart replays the exact same sample rather than
re-drawing -- required for resumability to mean anything.

Checkpointing: output is flushed to research/options_experiment/data/phase3_counterfactual.parquet
every `--flush-every` trades (atomic temp-file + os.replace, mirroring chain_cache.py's own
pattern). On restart, trade_ids already present in the output are skipped, so a long run that
dies partway loses at most one ticker's in-flight work, never completed work.

Run: .venv/bin/python scripts/run_options_counterfactual_sweep.py [--max-tickers N]
     [--max-trades N] [--flush-every N] [--seed N] [--target-per-module N]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.options_lab import chain_cache, fills, liquidity, pricing, spread_estimators, strategies  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase3_sweep")

# --------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------

SPINE_PATH = REPO_ROOT / "research" / "options_experiment" / "data" / "signal_spine.parquet"
DATA_DIR = REPO_ROOT / "research" / "options_experiment" / "data"
SAMPLE_PATH = DATA_DIR / "phase3_sample.parquet"
COVERAGE_PATH = DATA_DIR / "phase3_sample_coverage.json"
OUT_PARQUET = DATA_DIR / "phase3_counterfactual.parquet"
BARS_1D_DIR = REPO_ROOT / "Data" / "shared" / "bars" / "1d"

# Per the pre-registration's power table: only these three modules are powered.
# dealer_ranker (n=70, 2 weeks), meta_ranker (n=3), intraday_structure (no ledger)
# are excluded here, not merely down-weighted.
POWERED_MODULES = ("multi_ticker_swing", "multi_ticker_swing_htf", "momentum_expansion")

SAMPLE_SEED = 20260726
TARGET_PER_MODULE = 2000

ASSUMPTIONS = ("optimistic", "calibrated", "pessimistic")
COMMISSION_RATE_PER_CONTRACT = 0.65
TARGET_NOTIONAL = 5000.0
TARGET_MAX_LOSS = 5000.0

# 02_spread_model.md: Roll's median estimate is 5.1% vs a 10.8% realized-level anchor on the
# same contracts (its own results table). Roll is the only estimator in the ladder that is
# correctly *ranked* (Spearman 0.523) but it underestimates the *level* -- so Roll-sourced
# spread_pct values are scaled up to the realized anchor before use. Values from any other
# method in the ladder (clustering, regression) are left as-is; only Roll is level-biased.
ROLL_LEVEL_SCALE = 10.8 / 5.1
SPREAD_PCT_CAP = 3.0  # matches fills.SpreadCalibration.max_spread_pct -- guards runaway scaling

# Strike band around spot used to bound the candidate-contract universe fetched per ticker.
# Generous enough to cover the widest structure in the sweep (deep ITM at ~0.80 delta,
# verticals with wings a few ATR out) without pulling an entire small-cap chain.
STRIKE_BAND_LOW = 0.55
STRIKE_BAND_HIGH = 1.50

MAX_PRICE_LOOKBACK_MIN = 4320.0  # 3 days: tolerance for "last real bar at/before this timestamp"
LIQUIDITY_LOOKBACK_DAYS = 5

# Loaded once; passed explicitly to every pricing.risk_free_rate call to avoid re-reading the
# treasury parquet from disk on every one of tens of thousands of calls.
TREASURY_CURVE = pricing.load_treasury_curve()

_UNDERLYING_CACHE: dict[str, Optional[pd.DataFrame]] = {}

# --------------------------------------------------------------------------
# Shares (B0) risk config -- max_loss for a share position must reflect the
# module's actual exit rule (the stop it really trades to), not "stock goes to
# zero." Every one of the three powered modules exits at a stop; using the
# to-zero figure would overstate shares' risk 5-33x under the registered
# matched-max-loss metric (momentum_expansion's real median stop is ~19.8% of
# entry, multi_ticker_swing_htf's ~3.0%, multi_ticker_swing's ~4.8% -- measured
# from real sl_price/atr_at_entry where available) and bias every comparison
# toward "options look more capital-efficient," which is exactly the false
# positive the pre-registration exists to prevent.
#
# momentum_expansion and multi_ticker_swing_htf carry a real sl_price on 100%
# of spine rows (verified) -- use it directly: |entry_px - sl_price|.
#
# multi_ticker_swing carries sl_price on only ~2.8% of rows (the live-real
# leg from paired_option_trades.csv); the much larger OOF-backtest leg
# (strategies/multi_ticker_swing/backtest/results_oof/{long,short}/trades.parquet)
# was not built with atr_at_entry/sl_price attached at all (verified in
# build_options_experiment_spine.py). For those rows we reconstruct the
# module's OWN documented stop convention instead of inventing one:
# live/position_manager.py sets sl_price = entry_price - direction * sl_atr *
# atr_at_entry, where sl_atr is a per-ticker multiple (mostly 4.0, a few 0.0
# for trail-only names) from config/trading_universe.json, and atr_at_entry is
# a 14-bar rolling mean of true range (features/build_features.py's
# `atr_14 = tr.rolling(14).mean()` convention, reproduced exactly here).
# Data/shared/bars has no 30-minute bars locally (only 1h/4h/1d), so the ATR
# is computed on 1h bars as the closest available proxy to the module's real
# 30m-bar ATR -- NOT identical, documented explicitly, and recorded per-row
# via `shares_stop_source` so the real-vs-proxy split is reportable, per the
# instruction to never silently fall back to notional.
MTS_UNIVERSE_PATH = REPO_ROOT / "strategies" / "multi_ticker_swing" / "config" / "trading_universe.json"
DEFAULT_SL_ATR_MULT = 4.0  # modal value across trading_universe.json's tickers
MTS_STOP_PCT_FALLBACK = 0.05  # last-resort proxy only when no 1h bars exist at all for the ticker
STOP_DISTANCE_PCT_FLOOR = 0.005  # clamps: a near-zero stop distance must not blow matched-max-loss sizing up
STOP_DISTANCE_PCT_CEIL = 0.95

_HOURLY_BARS_CACHE: dict[str, Optional[pd.DataFrame]] = {}


def _load_mts_universe() -> dict[str, float]:
    if not MTS_UNIVERSE_PATH.exists():
        logger.warning("trading_universe.json not found at %s; all multi_ticker_swing stop proxies "
                        "will use the default %.1fx ATR fallback", MTS_UNIVERSE_PATH, DEFAULT_SL_ATR_MULT)
        return {}
    try:
        data = json.loads(MTS_UNIVERSE_PATH.read_text())
    except Exception as exc:
        logger.warning("trading_universe.json unreadable (%s); using default sl_atr for all tickers", exc)
        return {}
    return {k: float(v.get("sl_atr", DEFAULT_SL_ATR_MULT)) for k, v in data.items()}


_MTS_UNIVERSE = _load_mts_universe()


def _load_hourly_bars(ticker: str) -> Optional[pd.DataFrame]:
    if ticker in _HOURLY_BARS_CACHE:
        return _HOURLY_BARS_CACHE[ticker]
    path = REPO_ROOT / "Data" / "shared" / "bars" / "1h" / f"{ticker}.parquet"
    df = None
    if path.exists():
        try:
            df = pd.read_parquet(path, columns=["timestamp", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            prev_close = df["close"].shift(1)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(14).mean()
        except Exception as exc:
            logger.warning("1h bars unreadable for %s: %s", ticker, exc)
            df = None
    _HOURLY_BARS_CACHE[ticker] = df
    return df


def stop_distance_for_trade(module: str, ticker: str, entry_ts, entry_px: float, sl_price) -> tuple[float, str]:
    """Real risk-per-share in dollars, for the shares (B0) matched-max-loss sizing/reporting.
    Returns (distance_dollars, source) where source is one of:
    'sl_price' (real, from the spine), 'atr_proxy_ticker_mult' (real per-ticker sl_atr x
    reconstructed 1h ATR-14), 'atr_proxy_default_4x' (ticker missing from trading_universe.json
    or sl_atr<=0 there, defaulted to 4.0x), 'atr_proxy_fallback_no_1h_bars' /
    'atr_proxy_fallback_missing_sl' (no usable ATR data at all -- flat 5% of entry price)."""
    if pd.notna(sl_price):
        return abs(float(entry_px) - float(sl_price)), "sl_price"
    if module != "multi_ticker_swing":
        # momentum_expansion / multi_ticker_swing_htf carry sl_price on 100% of spine rows
        # (verified) -- a miss here is an unexpected data gap, not the expected OOF-backtest gap.
        return entry_px * MTS_STOP_PCT_FALLBACK, "atr_proxy_fallback_missing_sl"
    bars = _load_hourly_bars(ticker)
    atr_val = None
    if bars is not None:
        prior = bars[bars["timestamp"] < entry_ts]
        if not prior.empty and pd.notna(prior["atr_14"].iloc[-1]):
            atr_val = float(prior["atr_14"].iloc[-1])
    mult = _MTS_UNIVERSE.get(ticker)
    if mult and mult > 0:
        source = "atr_proxy_ticker_mult"
    else:
        mult = DEFAULT_SL_ATR_MULT
        source = "atr_proxy_default_4x"
    if atr_val is None or atr_val <= 0:
        return entry_px * MTS_STOP_PCT_FALLBACK, "atr_proxy_fallback_no_1h_bars"
    return mult * atr_val, source


# --------------------------------------------------------------------------
# Time-correctness helpers (mirrors the _normalize_ts convention already used
# identically in surface.py / strategies.py -- kept local per those modules'
# own precedent of not sharing this trivial helper across files).
# --------------------------------------------------------------------------


def _naive_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def dte_days_for_expiry(entry_ts, expiry_str: str) -> int:
    """Calendar days from `entry_ts` to `expiry_str`, computed with the EXACT same
    convention as strategies.py's internal `_dte_days` (expiry midnight minus the full,
    non-midnight-truncated entry timestamp, `.days` truncation). Using a different
    convention here (e.g. midnight-normalizing entry_ts first) would compute a target
    DTE that is off by one from what `_filter_dte_bucket` actually selects on, silently
    excluding the very expiry this script picked -- verified against strategies.py before
    writing this function this way."""
    asof_naive = _naive_ts(entry_ts)
    expiry_naive = pd.Timestamp(expiry_str)
    return (expiry_naive - asof_naive).days


def _prep_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    return df


# --------------------------------------------------------------------------
# Spine load + stratified sample
# --------------------------------------------------------------------------


def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(SPINE_PATH)
    df = df[df["module"].isin(POWERED_MODULES)].copy()
    required = ["direction", "entry_px_underlying", "exit_px_underlying", "entry_ts", "exit_ts"]
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped:
        logger.warning("load_spine: dropped %d/%d rows missing required fields", dropped, before)
    df["direction"] = df["direction"].astype(int)
    iso = df["entry_ts"].dt.isocalendar()
    df["week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, seed: int, target_per_module: int) -> tuple[pd.DataFrame, dict]:
    """Stratify by (module, week, direction). Every stratum contributes >=1 row (or all of
    itself if smaller than its quota), so every calendar week is represented in every module
    -- required because the registered significance test block-bootstraps over weeks, so power
    is governed by week count, not raw n."""
    rng = np.random.default_rng(seed)
    parts = []
    coverage = {}
    for module, mdf in df.groupby("module"):
        n_weeks = mdf["week_key"].nunique()
        picked = []
        for (_week, _direction), sdf in mdf.groupby(["week_key", "direction"]):
            quota = max(1, round(target_per_module * len(sdf) / len(mdf)))
            if len(sdf) <= quota:
                picked.append(sdf)
            else:
                idx = rng.choice(sdf.index.to_numpy(), size=quota, replace=False)
                picked.append(sdf.loc[idx])
        sampled = pd.concat(picked, ignore_index=False) if picked else mdf.iloc[0:0]
        coverage[module] = {
            "weeks_total": int(n_weeks),
            "weeks_sampled": int(sampled["week_key"].nunique()),
            "n_total": int(len(mdf)),
            "n_sampled": int(len(sampled)),
        }
        parts.append(sampled)
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    return out, coverage


def build_or_load_sample(seed: int, target_per_module: int) -> pd.DataFrame:
    if SAMPLE_PATH.exists():
        sample = pd.read_parquet(SAMPLE_PATH)
        logger.info("loaded existing sample: %d trades from %s", len(sample), SAMPLE_PATH)
        return sample
    spine = load_spine()
    sample, coverage = stratified_sample(spine, seed, target_per_module)
    sample = sample.copy()
    sample["trade_id"] = (
        sample["module"].astype(str) + "|" + sample["ticker"].astype(str) + "|"
        + sample["entry_ts"].astype(str) + "|" + sample["exit_ts"].astype(str) + "|"
        + sample["direction"].astype(str)
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PATH, index=False)
    with open(COVERAGE_PATH, "w") as f:
        json.dump({"seed": seed, "target_per_module": target_per_module, "coverage": coverage}, f, indent=2)
    logger.info("built new sample: %d trades. coverage=%s", len(sample), coverage)
    return sample


# --------------------------------------------------------------------------
# Strategy menu (Phase 3 spec: B0 shares, B1 ATM naked long, ~30-delta OTM naked long,
# deep ITM (~0.80 delta) stock replacement, debit spread at 2 widths, credit spread,
# extended-DTE long). Every option strategy is called with the SAME per-trade,
# per-bucket dte_min=dte_max=target_dte -- deliberately overriding each function's own
# hardcoded default DTE range (e.g. extended_dte_long's 46-90), so the near/far bucket
# comparison (H3) is apples-to-apples across every strategy including this one.
# --------------------------------------------------------------------------


def strategy_specs(direction: str, dte_min: int, dte_max: int, width1: float, width2: float):
    opt_fn = strategies.long_call if direction == "long" else strategies.long_put
    deep_fn = strategies.deep_itm_call if direction == "long" else strategies.deep_itm_put
    call_or_put = "call" if direction == "long" else "put"
    return [
        (f"long_{call_or_put}_atm", opt_fn, dict(dte_min=dte_min, dte_max=dte_max, target_delta=0.50)),
        (f"long_{call_or_put}_otm30", opt_fn, dict(dte_min=dte_min, dte_max=dte_max, target_delta=0.30)),
        ("deep_itm", deep_fn, dict(dte_min=dte_min, dte_max=dte_max, target_delta=0.80)),
        ("vertical_debit_w1", strategies.vertical_debit_spread,
         dict(dte_min=dte_min, dte_max=dte_max, long_delta=0.60, width=width1)),
        ("vertical_debit_w2", strategies.vertical_debit_spread,
         dict(dte_min=dte_min, dte_max=dte_max, long_delta=0.60, width=width2)),
        ("vertical_credit", strategies.vertical_credit_spread,
         dict(dte_min=dte_min, dte_max=dte_max, short_delta=0.30, width=width1)),
        ("extended_dte_long", strategies.extended_dte_long,
         dict(dte_min=dte_min, dte_max=dte_max, target_delta=0.50)),
    ]


def spread_widths(spot: float, atr) -> tuple[float, float]:
    if atr is not None and pd.notna(atr) and float(atr) > 0:
        return max(float(atr), 0.5), max(2.0 * float(atr), 1.0)
    return max(spot * 0.025, 0.5), max(spot * 0.05, 1.0)


# --------------------------------------------------------------------------
# Underlying ADV (for fills.estimate_spread's regression fallback input only)
# --------------------------------------------------------------------------


def _load_underlying_daily(ticker: str) -> Optional[pd.DataFrame]:
    if ticker in _UNDERLYING_CACHE:
        return _UNDERLYING_CACHE[ticker]
    path = BARS_1D_DIR / f"{ticker}.parquet"
    df = None
    if path.exists():
        try:
            df = pd.read_parquet(path, columns=["timestamp", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except Exception as exc:
            logger.warning("underlying bars unreadable for %s: %s", ticker, exc)
            df = None
    _UNDERLYING_CACHE[ticker] = df
    return df


def underlying_adv(ticker: str, entry_ts, window: int = 20) -> Optional[float]:
    bars = _load_underlying_daily(ticker)
    if bars is None:
        return None
    prior = bars[bars["timestamp"] < entry_ts]
    if len(prior) < 5:
        return None
    return float(prior["volume"].tail(window).mean())


# --------------------------------------------------------------------------
# Real-bar price / liquidity lookups
# --------------------------------------------------------------------------


def price_at(sub: Optional[pd.DataFrame], ts) -> tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]:
    """Last real bar at/before `ts` within MAX_PRICE_LOOKBACK_MIN. Returns
    (price, bar_timestamp, gap_minutes); price is None (never fabricated/interpolated)
    when no usable bar exists in range."""
    if sub is None or sub.empty:
        return None, None, None
    at_or_before = sub[sub["t"] <= ts]
    if at_or_before.empty:
        return None, None, None
    last = at_or_before.iloc[-1]
    gap = (ts - last["t"]).total_seconds() / 60.0
    if gap > MAX_PRICE_LOOKBACK_MIN:
        return None, None, gap
    price = last["vw"] if pd.notna(last.get("vw")) and last["vw"] > 0 else last["c"]
    if pd.isna(price) or price <= 0:
        return None, None, gap
    return float(price), last["t"], gap


def liquidity_from_1d(sub: Optional[pd.DataFrame], asof_ts, lookback_days: int = LIQUIDITY_LOOKBACK_DAYS):
    if sub is None or sub.empty:
        return None, None
    start = asof_ts - pd.Timedelta(days=lookback_days)
    win = sub[(sub["t"] <= asof_ts) & (sub["t"] > start)]
    if win.empty:
        return None, None
    bv = int(win["v"].fillna(0).sum()) if "v" in win.columns and win["v"].notna().any() else None
    tc = int(win["n"].fillna(0).sum()) if "n" in win.columns and win["n"].notna().any() else None
    return bv, tc


def pick_expiry(expiries_sorted: list[str], entry_ts, target_min_days: float) -> tuple[Optional[str], Optional[int]]:
    for exp in expiries_sorted:
        dte = dte_days_for_expiry(entry_ts, exp)
        if dte >= target_min_days:
            return exp, dte
    return None, None


def build_chain(entry_ts, spot: float, exp: str, dte: int, contracts: pd.DataFrame,
                 bars30_by_sym: dict, bars1d_by_sym: dict, r: float) -> pd.DataFrame:
    lo, hi = spot * STRIKE_BAND_LOW, spot * STRIKE_BAND_HIGH
    sub = contracts[(contracts["expiry"] == exp) & (contracts["strike"] >= lo) & (contracts["strike"] <= hi)]
    T = dte / 365.25
    rows = []
    for _, c in sub.iterrows():
        sym = c["osi_symbol"]
        price, bar_ts, _gap = price_at(bars30_by_sym.get(sym), entry_ts)
        if price is None:
            continue
        iv = None
        vega = None
        if T > 0:
            iv = pricing.implied_vol(price=price, S=spot, K=float(c["strike"]), T=T, r=r, q=0.0,
                                      right=c["right"], american=False)
            if iv is not None:
                g = pricing.bsm_greeks(spot, float(c["strike"]), T, r, 0.0, iv, c["right"])
                vega = float(g.vega)
        bv, tc = liquidity_from_1d(bars1d_by_sym.get(sym), entry_ts)
        oi_raw = c.get("open_interest")
        oi = None if pd.isna(oi_raw) else int(oi_raw)
        rows.append({
            "osi_symbol": sym, "ticker": c["ticker"], "expiry": c["expiry"], "strike": float(c["strike"]),
            "right": c["right"], "price": float(price), "iv": iv, "vega": vega,
            "quote_asof": bar_ts, "open_interest": oi, "bar_volume": bv, "trade_count": tc,
        })
    if not rows:
        return pd.DataFrame(columns=strategies.CHAIN_COLUMNS)
    return pd.DataFrame(rows, columns=strategies.CHAIN_COLUMNS)


# --------------------------------------------------------------------------
# Row construction
# --------------------------------------------------------------------------

_ROW_DEFAULTS = dict(
    selection_method=None, n_legs=np.nan, entry_cost=np.nan, max_loss=np.nan, max_gain=np.nan,
    buying_power_required=np.nan, unavailable_reason=None, executable=False,
    gross_pnl=np.nan, commission_dollars=np.nan,
    spread_cost_optimistic=np.nan, spread_cost_calibrated=np.nan, spread_cost_pessimistic=np.nan,
    net_pnl_optimistic=np.nan, net_pnl_calibrated=np.nan, net_pnl_pessimistic=np.nan,
    spread_methods="", entry_liquidity_ok=False, exit_liquidity_ok=False,
    shares_stop_distance_dollars=np.nan, shares_stop_source=None,
)


def trade_base(t: pd.Series) -> dict:
    atr = t.get("atr_at_entry")
    realized_move_atr = np.nan
    try:
        if pd.notna(atr) and float(atr) > 0:
            realized_move_atr = abs(float(t["exit_px_underlying"]) - float(t["entry_px_underlying"])) / float(atr)
    except (TypeError, ValueError):
        pass
    return dict(
        trade_id=t["trade_id"], module=t["module"], ticker=t["ticker"], direction=int(t["direction"]),
        entry_ts=t["entry_ts"], exit_ts=t["exit_ts"],
        holding_days=(t["exit_ts"] - t["entry_ts"]).total_seconds() / 86400.0,
        week_key=t["week_key"],
        entry_px_underlying=float(t["entry_px_underlying"]), exit_px_underlying=float(t["exit_px_underlying"]),
        atr_at_entry=(float(atr) if pd.notna(atr) else np.nan),
        score=t.get("score"), exit_reason=t.get("exit_reason"), bars_held=t.get("bars_held"),
        provenance=t.get("provenance"), realized_move_atr=realized_move_atr,
    )


def unavailable_row(base: dict, strategy: str, dte_bucket: str, target_dte, expiry_date, reason: str,
                     sizing_mode: Optional[str] = None) -> dict:
    row = dict(base)
    row.update(dict(strategy=strategy, dte_bucket=dte_bucket, target_dte=target_dte, expiry_date=expiry_date,
                     sizing_mode=sizing_mode))
    row.update(_ROW_DEFAULTS)
    row["unavailable_reason"] = reason
    return row


def emit_shares_row(base: dict, ticker: str, entry_ts, exit_underlying: float, spot: float, direction_str: str,
                     sizing_mode: str, target_notional: float, stop_distance_dollars: float, stop_source: str) -> dict:
    """B0 baseline. No commission (zero-commission equities, matching the module's own live
    sizing convention) and no explicit equity bid/ask spread is modeled (documented assumption:
    this experiment is about OPTION structure cost, and equity spread is a rounding error next
    to option spread cost for the liquid names in scope) -- so gross == net under all three cost
    assumptions for shares. `max_loss` is the REAL risk taken (stop distance x shares), not the
    to-zero figure strategies.Structure.max_loss would report for a naked share leg -- see the
    module-level comment block above `stop_distance_for_trade` for why that distinction matters."""
    sel = strategies.long_shares(ticker, entry_ts, spot, direction_str, target_notional=target_notional)
    structure = sel.structure
    leg = structure.legs[0]
    gross_pnl = leg.quantity * leg.multiplier * (exit_underlying - leg.entry_price)
    realistic_max_loss = abs(leg.quantity) * stop_distance_dollars

    row = dict(base)
    row.update(_ROW_DEFAULTS)
    row.update(dict(
        strategy="long_shares", dte_bucket="n/a", target_dte=np.nan, expiry_date=None,
        sizing_mode=sizing_mode, selection_method=None, n_legs=1,
        entry_cost=structure.entry_cost, max_loss=realistic_max_loss, max_gain=None,
        buying_power_required=structure.buying_power_required,
        unavailable_reason=None, executable=True,
        gross_pnl=gross_pnl, commission_dollars=0.0,
        spread_cost_optimistic=0.0, spread_cost_calibrated=0.0, spread_cost_pessimistic=0.0,
        net_pnl_optimistic=gross_pnl, net_pnl_calibrated=gross_pnl, net_pnl_pessimistic=gross_pnl,
        spread_methods="n/a_shares", entry_liquidity_ok=True, exit_liquidity_ok=True,
        shares_stop_distance_dollars=stop_distance_dollars, shares_stop_source=stop_source,
    ))
    return row


def price_and_emit(base: dict, strategy: str, dte_bucket: str, target_dte, expiry_date: str, sizing_mode: str,
                    sel: strategies.Selection, entry_ts, exit_ts, exit_underlying: float,
                    bars30_by_sym: dict, tape_by_sym: dict, chain_indexed: Optional[pd.DataFrame],
                    adv: Optional[float]) -> dict:
    structure = sel.structure
    resize_fn = strategies.resize_to_notional if sizing_mode == "matched_notional" else strategies.resize_to_max_loss
    target = TARGET_NOTIONAL if sizing_mode == "matched_notional" else TARGET_MAX_LOSS
    try:
        sized = resize_fn(structure, target)
    except ValueError as exc:
        return unavailable_row(base, strategy, dte_bucket, target_dte, expiry_date,
                                f"resize_failed:{exc}", sizing_mode=sizing_mode)

    exit_prices: dict[str, Optional[float]] = {}
    for leg in sized.legs:
        if leg.right == "S":
            exit_prices[leg.osi_symbol] = exit_underlying
        else:
            price, _bar_ts, _gap = price_at(bars30_by_sym.get(leg.osi_symbol), exit_ts)
            exit_prices[leg.osi_symbol] = price

    row = dict(base)
    row.update(dict(
        strategy=strategy, dte_bucket=dte_bucket, target_dte=target_dte, expiry_date=expiry_date,
        sizing_mode=sizing_mode, selection_method=sel.selection_method, n_legs=len(sized.legs),
        entry_cost=sized.entry_cost, max_loss=sized.max_loss, max_gain=sized.max_gain,
        buying_power_required=sized.buying_power_required, entry_liquidity_ok=True,
    ))

    missing = [leg.osi_symbol for leg in sized.legs if exit_prices.get(leg.osi_symbol) is None]
    if missing:
        row.update(dict(
            unavailable_reason=f"exit_unpriceable:{missing[0]}", executable=False,
            gross_pnl=np.nan, commission_dollars=np.nan,
            spread_cost_optimistic=np.nan, spread_cost_calibrated=np.nan, spread_cost_pessimistic=np.nan,
            net_pnl_optimistic=np.nan, net_pnl_calibrated=np.nan, net_pnl_pessimistic=np.nan,
            spread_methods="", exit_liquidity_ok=False,
        ))
        return row

    spreads: dict[str, tuple[float, str]] = {}
    exit_liq_stats = []
    for leg in sized.legs:
        if leg.right == "S":
            continue
        feat = None
        if chain_indexed is not None and leg.osi_symbol in chain_indexed.index:
            feat = chain_indexed.loc[leg.osi_symbol]
        moneyness = abs(math.log(leg.strike / structure.entry_spot)) if leg.strike else None
        oi = None if feat is None or pd.isna(feat.get("open_interest")) else float(feat["open_interest"])
        bv = None if feat is None or pd.isna(feat.get("bar_volume")) else float(feat["bar_volume"])

        tape = tape_by_sym.get(leg.osi_symbol)
        prices: list[float] = []
        if tape is not None and not tape.empty:
            w_lo = entry_ts - pd.Timedelta(days=3)
            w_hi = exit_ts + pd.Timedelta(days=3)
            w = tape[(tape["t"] >= w_lo) & (tape["t"] <= w_hi)]
            if "p" in w.columns:
                prices = w["p"].dropna().tolist()
        roll_pct = spread_estimators.roll_effective_spread_pct(prices)
        clustering_pct = spread_estimators.price_clustering_spread_pct(prices)
        regression_pct = fills.estimate_spread(
            moneyness, float(target_dte) if target_dte is not None and pd.notna(target_dte) else None,
            oi, bv, adv,
        )
        est = spread_estimators.combine_spread_estimates(
            roll_pct=roll_pct, clustering_pct=clustering_pct, regression_pct=regression_pct,
        )
        spread_pct = est.spread_pct
        method = est.method
        # 02_spread_model.md: Roll underestimates the spread LEVEL (median 5.1% vs 10.8%
        # realized on the same contracts) even though it ranks contracts correctly -- scale
        # to the realized-level anchor before use. Other methods are not level-biased in the
        # same documented way and are left unscaled.
        if method == "roll" and spread_pct is not None:
            spread_pct = min(spread_pct * ROLL_LEVEL_SCALE, SPREAD_PCT_CAP)
        spreads[leg.osi_symbol] = (spread_pct if spread_pct is not None else 0.0, method)

        exit_liq_stats.append(liquidity.LiquidityStats(
            osi_symbol=leg.osi_symbol, asof=str(exit_ts), lookback_days=LIQUIDITY_LOOKBACK_DAYS,
            open_interest=int(oi) if oi is not None else None,
            bar_volume=int(bv) if bv is not None else None,
            trade_count=None,
        ))

    # Exit-time liquidity re-check reuses the entry-window bar volume as a proxy (a fully
    # independent exit-day volume recompute would need a second per-leg 1Day-bar slice; not
    # done here -- documented limitation, see 03_phase3_results.md). Open interest is whatever
    # Alpaca reports at contract-discovery time, not a point-in-time historical OI series --
    # a data-source limitation inherited from liquidity.py, not introduced here.
    exit_liquidity_ok = liquidity.structure_tradable(exit_liq_stats) if exit_liq_stats else True

    gross_pnl = 0.0
    commission_dollars = 0.0
    net_by_assumption = {a: 0.0 for a in ASSUMPTIONS}
    for leg in sized.legs:
        exit_mid = exit_prices[leg.osi_symbol]
        gross_pnl += leg.quantity * leg.multiplier * (exit_mid - leg.entry_price)
        if leg.right != "S":
            # $0.65/contract, per leg, charged at both entry and exit (round trip).
            commission_dollars += fills.commission_cost(abs(leg.quantity), rate=COMMISSION_RATE_PER_CONTRACT) * 2.0
        # No explicit equities bid/ask spread is modeled for share legs (B0): the routing
        # question this experiment answers is about OPTION structure costs specifically, and
        # for the liquid underlyings in scope the equity spread is a rounding error next to
        # option spread costs. Documented assumption, not an oversight.
        spread_pct = spreads.get(leg.osi_symbol, (0.0, "none"))[0] if leg.right != "S" else 0.0
        entry_side = "buy" if leg.quantity > 0 else "sell"
        exit_side = "sell" if leg.quantity > 0 else "buy"
        for a in ASSUMPTIONS:
            entry_fill = fills.apply_fill(leg.entry_price, entry_side, spread_pct, a)
            exit_fill = fills.apply_fill(exit_mid, exit_side, spread_pct, a)
            net_by_assumption[a] += leg.quantity * leg.multiplier * (exit_fill - entry_fill)

    for a in ASSUMPTIONS:
        net_by_assumption[a] -= commission_dollars
    spread_cost = {a: gross_pnl - commission_dollars - net_by_assumption[a] for a in ASSUMPTIONS}
    spread_methods = "+".join(sorted({m for _, m in spreads.values()})) if spreads else "n/a_shares"

    row.update(dict(
        unavailable_reason=None, executable=bool(exit_liquidity_ok),
        gross_pnl=gross_pnl, commission_dollars=commission_dollars,
        spread_cost_optimistic=spread_cost["optimistic"], spread_cost_calibrated=spread_cost["calibrated"],
        spread_cost_pessimistic=spread_cost["pessimistic"],
        net_pnl_optimistic=net_by_assumption["optimistic"], net_pnl_calibrated=net_by_assumption["calibrated"],
        net_pnl_pessimistic=net_by_assumption["pessimistic"],
        spread_methods=spread_methods, exit_liquidity_ok=bool(exit_liquidity_ok),
    ))
    return row


def process_trade(t: pd.Series, meta: dict, contracts: pd.DataFrame, bars30_by_sym: dict, bars1d_by_sym: dict,
                   tape_by_sym: dict) -> list[dict]:
    base = trade_base(t)
    direction_str = "long" if t["direction"] == 1 else "short"
    spot = float(t["entry_px_underlying"])
    entry_ts, exit_ts = t["entry_ts"], t["exit_ts"]
    rows: list[dict] = []

    # --- B0: shares, priced from the trade's own real entry/exit underlying marks --
    # already in the spine (no chain, no DTE bucket -- needs no option data at all, so it is
    # computed unconditionally, even for tickers whose option chain is unavailable).
    # max_loss is the module's REAL stop distance (real sl_price, or a documented ATR proxy
    # for multi_ticker_swing's OOF-backtest leg), never the naive "stock to zero" figure --
    # see the config block above stop_distance_for_trade for why. matched_notional sizes to
    # $5,000 of underlying exposure as usual; matched_max_loss sizes shares so a stop-out
    # would lose $5,000, using that same real stop distance (a materially larger position
    # than $5,000 notional whenever the stop is tight, which is the point: shares are far
    # less risk-hungry per dollar of max_loss than the naive to-zero framing implied).
    stop_distance, stop_source = stop_distance_for_trade(t["module"], t["ticker"], entry_ts, spot, t.get("sl_price"))
    stop_distance = min(max(stop_distance, spot * STOP_DISTANCE_PCT_FLOOR), spot * STOP_DISTANCE_PCT_CEIL)
    stop_pct = stop_distance / spot
    exit_underlying = float(t["exit_px_underlying"])
    rows.append(emit_shares_row(base, t["ticker"], entry_ts, exit_underlying, spot, direction_str,
                                 "matched_notional", TARGET_NOTIONAL, stop_distance, stop_source))
    matched_max_loss_notional = TARGET_MAX_LOSS / max(stop_pct, STOP_DISTANCE_PCT_FLOOR)
    rows.append(emit_shares_row(base, t["ticker"], entry_ts, exit_underlying, spot, direction_str,
                                 "matched_max_loss", matched_max_loss_notional, stop_distance, stop_source))

    w1, w2 = spread_widths(spot, t.get("atr_at_entry"))
    adv = underlying_adv(t["ticker"], entry_ts)

    for bucket in ("near", "far"):
        exp, dte = meta[f"{bucket}_exp"], meta[f"{bucket}_dte"]
        specs = strategy_specs(direction_str, dte or 0, dte or 0, w1, w2)
        if exp is None:
            for name, _fn, _kwargs in specs:
                rows.append(unavailable_row(base, name, bucket, np.nan, None, "no_expiry_available"))
            continue

        T = dte / 365.25
        try:
            r = pricing.risk_free_rate(entry_ts, tenor_years=max(T, 0.01), curve=TREASURY_CURVE)
        except Exception as exc:
            for name, _fn, _kwargs in specs:
                rows.append(unavailable_row(base, name, bucket, dte, exp, f"risk_free_rate_failed:{exc}"))
            continue

        chain = build_chain(entry_ts, spot, exp, dte, contracts, bars30_by_sym, bars1d_by_sym, r)
        chain_indexed = chain.set_index("osi_symbol") if not chain.empty else None

        for name, fn, kwargs in specs:
            try:
                sel = fn(t["ticker"], entry_ts, spot, direction_str, chain, r=r, q=0.0, **kwargs)
            except Exception as exc:
                rows.append(unavailable_row(base, name, bucket, dte, exp, f"selection_error:{type(exc).__name__}:{exc}"))
                continue
            if sel.structure is None:
                rows.append(unavailable_row(base, name, bucket, dte, exp, sel.reason))
                continue
            for sizing_mode in ("matched_notional", "matched_max_loss"):
                rows.append(price_and_emit(base, name, bucket, dte, exp, sizing_mode, sel,
                                            entry_ts, exit_ts, exit_underlying,
                                            bars30_by_sym, tape_by_sym, chain_indexed, adv))
    return rows


def process_ticker(ticker: str, trades_df: pd.DataFrame) -> list[dict]:
    trades_df = trades_df.copy()
    trades_df["near_min_dte"] = (
        (trades_df["exit_ts"] - trades_df["entry_ts"]).dt.total_seconds() / 86400.0
    ) + 5.0
    trades_df["far_min_dte"] = 2.0 * trades_df["near_min_dte"]

    win_start = trades_df["entry_ts"].min().strftime("%Y-%m-%d")
    max_far = int(math.ceil(trades_df["far_min_dte"].max())) + 15
    win_end = (trades_df["entry_ts"].max() + pd.Timedelta(days=max_far)).strftime("%Y-%m-%d")

    try:
        contracts = chain_cache.discover_contracts(ticker, win_start, win_end)
    except Exception as exc:
        logger.warning("discover_contracts(%s) failed, treating as no chain: %s", ticker, exc)
        contracts = chain_cache._empty_contracts_frame()  # noqa: SLF001 -- documented empty-frame constructor

    expiries_str = sorted(contracts["expiry"].unique().tolist(), key=lambda s: pd.Timestamp(s)) if not contracts.empty else []

    trade_meta: dict[int, dict] = {}
    candidate_syms: set[str] = set()
    for idx, t in trades_df.iterrows():
        near_exp, near_dte = pick_expiry(expiries_str, t["entry_ts"], t["near_min_dte"])
        far_exp, far_dte = pick_expiry(expiries_str, t["entry_ts"], t["far_min_dte"])
        lo, hi = t["entry_px_underlying"] * STRIKE_BAND_LOW, t["entry_px_underlying"] * STRIKE_BAND_HIGH
        for exp in (near_exp, far_exp):
            if exp is None:
                continue
            sub = contracts[(contracts["expiry"] == exp) & (contracts["strike"] >= lo) & (contracts["strike"] <= hi)]
            candidate_syms.update(sub["osi_symbol"].tolist())
        trade_meta[idx] = dict(near_exp=near_exp, near_dte=near_dte, far_exp=far_exp, far_dte=far_dte)

    candidate_syms_sorted = sorted(candidate_syms)
    ts_min = trades_df["entry_ts"].min() - pd.Timedelta(days=2)
    ts_max = trades_df["exit_ts"].max() + pd.Timedelta(days=2)
    start_s, end_s = ts_min.strftime("%Y-%m-%dT%H:%M:%SZ"), ts_max.strftime("%Y-%m-%dT%H:%M:%SZ")

    bars30 = pd.DataFrame(columns=["osi_symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])
    bars1d = bars30.copy()
    trades_tape = pd.DataFrame(columns=["osi_symbol", "t", "x", "p", "s", "c"])
    if candidate_syms_sorted:
        try:
            bars30 = chain_cache.fetch_bars(candidate_syms_sorted, "30Min", start_s, end_s)
        except Exception as exc:
            logger.warning("fetch_bars(30Min) failed for %s: %s", ticker, exc)
        try:
            bars1d = chain_cache.fetch_bars(candidate_syms_sorted, "1Day", start_s, end_s)
        except Exception as exc:
            logger.warning("fetch_bars(1Day) failed for %s: %s", ticker, exc)
        try:
            trades_tape = chain_cache.fetch_trades(candidate_syms_sorted, start_s, end_s)
        except Exception as exc:
            logger.warning("fetch_trades failed for %s: %s", ticker, exc)

    bars30, bars1d, trades_tape = _prep_ts(bars30), _prep_ts(bars1d), _prep_ts(trades_tape)
    bars30_by_sym = {s: g.sort_values("t") for s, g in bars30.groupby("osi_symbol")} if not bars30.empty else {}
    bars1d_by_sym = {s: g.sort_values("t") for s, g in bars1d.groupby("osi_symbol")} if not bars1d.empty else {}
    tape_by_sym = {s: g.sort_values("t") for s, g in trades_tape.groupby("osi_symbol")} if not trades_tape.empty else {}

    out_rows: list[dict] = []
    for idx, t in trades_df.iterrows():
        out_rows.extend(process_trade(t, trade_meta[idx], contracts, bars30_by_sym, bars1d_by_sym, tape_by_sym))
    return out_rows


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def flush_checkpoint(buffer: list[dict], out_path: Path) -> None:
    if not buffer:
        return
    new_df = pd.DataFrame(buffer)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    logger.info("checkpoint: +%d rows, %d total, %d distinct trade_ids -> %s",
                len(new_df), len(combined), combined["trade_id"].nunique(), out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=None, help="debug: limit tickers processed this run")
    ap.add_argument("--max-trades", type=int, default=None, help="debug: limit trades processed this run")
    ap.add_argument("--target-per-module", type=int, default=TARGET_PER_MODULE)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--flush-every", type=int, default=25, help="flush checkpoint every N trades")
    args = ap.parse_args()

    sample = build_or_load_sample(args.seed, args.target_per_module)

    if OUT_PARQUET.exists():
        completed = set(pd.read_parquet(OUT_PARQUET, columns=["trade_id"])["trade_id"].unique())
    else:
        completed = set()
    todo = sample[~sample["trade_id"].isin(completed)].copy()
    logger.info("todo: %d / %d trades remaining (%d already checkpointed)", len(todo), len(sample), len(completed))

    if args.max_trades:
        todo = todo.head(args.max_trades)

    tickers = todo["ticker"].unique().tolist()
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
    logger.info("processing %d tickers", len(tickers))

    buffer: list[dict] = []
    n_since_flush = 0
    t_start = time.time()
    for i, ticker in enumerate(tickers):
        ticker_trades = todo[todo["ticker"] == ticker]
        t0 = time.time()
        try:
            rows = process_ticker(ticker, ticker_trades)
        except Exception:
            logger.exception("process_ticker(%s) failed; its trades remain unflushed and will retry on resume", ticker)
            rows = []
        buffer.extend(rows)
        n_since_flush += len(ticker_trades)
        elapsed = time.time() - t0
        logger.info("[%d/%d] %s: %d trades, %d rows, %.1fs (elapsed total %.0fs)",
                    i + 1, len(tickers), ticker, len(ticker_trades), len(rows), elapsed, time.time() - t_start)
        if n_since_flush >= args.flush_every:
            flush_checkpoint(buffer, OUT_PARQUET)
            buffer = []
            n_since_flush = 0
    flush_checkpoint(buffer, OUT_PARQUET)
    logger.info("done. total elapsed %.0fs", time.time() - t_start)


if __name__ == "__main__":
    main()
