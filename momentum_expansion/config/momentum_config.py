"""
Central configuration for the momentum_expansion trader.

All paths, thresholds, walk-forward windows, capital sleeve params,
and option-policy defaults live here. Override via env or by editing.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT   = Path(__file__).resolve().parents[2]

RAW_DIR              = MODULE_ROOT / "data" / "raw"
RAW_1H_DIR           = RAW_DIR / "1h"
RAW_4H_DIR           = RAW_DIR / "4h"
RAW_1D_DIR           = RAW_DIR / "1d"
RAW_CONTEXT_DIR      = RAW_DIR / "context"

UNIVERSE_DIR         = MODULE_ROOT / "data" / "universe_snapshots"
PROCESSED_DIR        = MODULE_ROOT / "data" / "processed"
PROCESSED_FEAT_DIR   = PROCESSED_DIR / "features_4h"
FEATURES_COMBINED    = PROCESSED_DIR / "features_4h.parquet"
LABELS_COMBINED      = PROCESSED_DIR / "expansion_labels_4h.parquet"
TRAINING_MATRIX      = PROCESSED_DIR / "training_matrix_4h.parquet"

TRAINING_EXPORT_DIR  = MODULE_ROOT / "data" / "training_export"

MODELS_DIR           = MODULE_ROOT / "models" / "expansion_v1"
MODEL_PATH           = MODELS_DIR / "expansion_xgb.json"
FEATURE_MANIFEST     = MODELS_DIR / "feature_manifest.json"
EVAL_METRICS_PATH    = MODELS_DIR / "eval_metrics.json"

BACKTEST_RESULTS_DIR = MODULE_ROOT / "backtest" / "results"
PLOTS_DIR            = MODULE_ROOT / "plots" / "output"

# ---------------------------------------------------------------------------
# Date range — Alpaca historical depth on most names is reliable from ~2016
# ---------------------------------------------------------------------------
TRAIN_START = "2017-01-01"
TRAIN_END   = "2026-04-30"

# ---------------------------------------------------------------------------
# Context tickers — fetched once, reused by every feature build
# ---------------------------------------------------------------------------
CONTEXT_TICKERS:  list[str] = ["SPY", "QQQ", "IWM", "VIXY", "TLT"]
SECTOR_ETFS:      list[str] = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Healthcare
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLB",  # Materials
    "XLRE", # Real Estate
    "XLC",  # Communications
]

# ---------------------------------------------------------------------------
# Universe selector defaults
# ---------------------------------------------------------------------------
UNIVERSE_CONFIG: dict = {
    "candidate_pool_csv":   None,            # if None, pulls from get_candidate_pool()
    "min_avg_dollar_vol":   5_000_000.0,     # 30-day avg dollar volume floor (lowered to catch emerging breakouts faster)
    "min_price":            1.0,             # allow low-priced names; option chain gate handles real illiquidity
    "max_price":            1000.0,
    "rvol_window":          20,              # bars used for relative volume
    "rs_lookbacks":         (5, 20, 60),     # daily lookbacks for relative strength
    "atr_expansion_window": 20,              # ATR(14)/ATR(60) used as expansion proxy
    "min_history_days":     200,             # require this many daily bars before scoring
    "max_universe_size":    500,             # cap weekly universe size
    "rebuild_day_of_week":  6,               # 6 = Sunday
    "score_weights": {
        "rs_5":         0.20,
        "rs_20":        0.30,
        "rs_60":        0.20,
        "atr_expand":   0.15,
        "dollar_vol":   0.15,
    },
}

# ---------------------------------------------------------------------------
# Bar download / resampling
# ---------------------------------------------------------------------------
BAR_CONFIG: dict = {
    "primary_timeframe": "1Hour",   # native pull, then resample to 4H
    "context_timeframe": "1Hour",   # context tickers stored at same cadence
    "daily_timeframe":   "1Day",
    "adjustment":        "split",
    # 4H aggregation: align to RTH so each session yields exactly two 4H bars
    # (09:30-13:30 and 13:30-17:30 ET, last bar can be partial). We label by
    # the bar's *start* timestamp.
    "rth_4h_anchor_minutes": [9 * 60 + 30, 13 * 60 + 30],   # in NY local
}

# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
MIN_4H_BARS = 250  # ~6 months of 4H bars before features are usable

# ---------------------------------------------------------------------------
# Forward expansion label
# ---------------------------------------------------------------------------
LABEL_CONFIG: dict = {
    "forward_window_4h_bars": 25,    # ~10 trading days @ 2 bars/day
    "alpha_benchmark":        "SPY",
    "atr_window":             14,
    "trend_persistence_window": 10,  # bars closing above entry within forward window
    # Composite score weights (higher = more "expansion-like")
    "composite_weights": {
        "fwd_max_alpha":      0.40,   # forward max return minus SPY
        "fwd_atr_adj_return": 0.25,   # forward return / ATR(14)
        "trend_persistence":  0.20,   # fraction of forward bars closing above entry
        "fwd_max_drawdown":   0.15,   # negative MDD penalty
    },
    # Binary target: top-decile-or-quintile of composite score within rolling window
    "binary_top_quantile":    0.20,   # top 20% = positive class
    "binary_window_bars":     2000,   # ~2 years of 4H bars for percentile context
}

# ---------------------------------------------------------------------------
# Walk-forward training (used by Colab notebook)
# ---------------------------------------------------------------------------
WALK_FORWARD_CONFIG: dict = {
    "train_years":   2.0,
    "embargo_days":  21,
    "test_months":   6,
    "min_train_rows": 50_000,
}

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
RANKING_CONFIG: dict = {
    "top_n":         20,
    "top_pct":       0.10,    # use min(top_n, top_pct * universe_size)
    "min_score":     0.55,    # absolute floor — name is rejected even if in top-N
    "tie_break":     "expansion_score",
}

# ---------------------------------------------------------------------------
# 1H entry rules
# ---------------------------------------------------------------------------
ENTRY_RULES_CONFIG: dict = {
    "ema_fast":               10,
    "ema_slow":               20,
    "pullback_min_atr":       0.4,    # min pullback depth to be a "pullback continuation"
    "pullback_max_atr":       2.5,    # too deep -> probably a real reversal
    "flag_consolidation_bars": 4,
    "flag_breakout_atr":      0.25,   # bar must close >= flag_high + this*ATR
    "volume_confirm_mult":    1.5,    # 1H bar volume must exceed this * 20-bar avg
    "ema_reclaim_lookback":   8,      # bars allowed to be below ema_slow before reclaim
    # Bar must be in RTH (no after-hours triggers)
    "rth_only":               True,
}

# ---------------------------------------------------------------------------
# Capital sleeve + MomentumOptionPolicy defaults
# ---------------------------------------------------------------------------
CAPITAL_CONFIG: dict = {
    "sleeve_dollars":        25_000.0,    # isolated risk budget for momentum book
    "max_concurrent":        5,
    "per_trade_risk_pct":    0.015,       # 1.5% of sleeve risked per trade (ATR stop)
    "min_per_trade_dollars": 250.0,
}

OPTION_POLICY_CONFIG: dict = {
    # Strike & DTE selection
    "target_dte_min_days": 30,
    "target_dte_max_days": 60,
    "target_delta_long":   0.55,        # slightly ITM call/put for swing momentum
    "delta_tolerance":     0.10,
    # Liquidity gate (chain rejection — name is dropped if it fails)
    "min_open_interest":   500,
    "min_chain_volume":    100,
    "max_bid_ask_spread_pct": 0.10,
    # Pricing / sizing
    "price_mode":          "mid",       # ask|mid|bid|last|mark
    "max_contracts_cap":   25,
    # IV regime gate
    "max_ivr":             0.85,        # don't buy calls when IV is at the top of its range
    # Exits
    "atr_stop_mult":       1.5,         # initial underlying stop in ATR units
    "atr_trail_arm":       1.0,         # trail arms after this much favorable underlying move
    "atr_trail_distance":  1.2,         # trailing distance once armed
    "score_decay_exit":    0.40,        # exit if 4H score drops below this
    "trend_break_atr":     1.0,         # exit if underlying breaks below ema_slow by this much
    "max_holding_4h_bars": 30,          # ~6 trading weeks; let winners run, kill stale
}

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
BACKTEST_CONFIG: dict = {
    "initial_capital":       25_000.0,
    "commission_pct":        0.001,
    "use_synthetic_options": True,        # if False, P&L is on the underlying only
    "synthetic_iv_proxy":    "vix_scaled",# vix_scaled | constant_30 | per_ticker_hist
    "constant_iv":           0.30,
    "report_underlying_pnl": True,        # always include "underlying-only" baseline metrics
}

# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------
LIVE_CONFIG: dict = {
    "universe_refresh_dow":  6,           # 0=Mon … 6=Sun
    "universe_refresh_hhmm": "10:00",     # ET
    "score_on_4h_close":     True,
    "trigger_on_1h_close":   True,
    "auto_trade":            False,       # alerts-only by default
    "alert_log_path":        MODULE_ROOT / "live" / "alerts.jsonl",
}
