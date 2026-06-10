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
REPO_ROOT   = Path(__file__).resolve().parents[3]
SHARED_DATA_ROOT = REPO_ROOT / "Data" / "shared"

RAW_DIR              = SHARED_DATA_ROOT / "bars"
RAW_1H_DIR           = RAW_DIR / "1h"
RAW_4H_DIR           = RAW_DIR / "4h"
RAW_1D_DIR           = RAW_DIR / "1d"
RAW_CONTEXT_DIR      = RAW_DIR / "context"
DEFAULT_SCAN_UNIVERSE = SHARED_DATA_ROOT / "universe" / "MomentumExpansionUniverse.xlsx"

UNIVERSE_DIR         = MODULE_ROOT / "data" / "universe_snapshots"
PROCESSED_DIR        = MODULE_ROOT / "data" / "processed"
PROCESSED_FEAT_DIR   = PROCESSED_DIR / "features_4h"
FEATURES_COMBINED    = PROCESSED_DIR / "features_4h.parquet"
LABELS_COMBINED      = PROCESSED_DIR / "expansion_labels_4h.parquet"
TRAINING_MATRIX      = PROCESSED_DIR / "training_matrix_4h.parquet"
CORRELATED_FEATURE_REPORT = PROCESSED_DIR / "correlated_features_dropped.csv"

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
TRAIN_START = "2020-01-01"
# Date-only fetch endpoints are effectively end-exclusive in the current cache,
# so 2026-06-02 includes bars through 2026-06-01.
TRAIN_END   = "2026-06-02"

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

# Instruments used as cross-market context/regime inputs only. They should be
# fetched for features, but not labeled as expansion-training examples or used
# as live trade candidates by default.
CONTEXT_ONLY_TICKERS: list[str] = sorted(set(CONTEXT_TICKERS) | set(SECTOR_ETFS))

# ---------------------------------------------------------------------------
# Universe selector defaults
# ---------------------------------------------------------------------------
UNIVERSE_CONFIG: dict = {
    "candidate_pool_csv":   DEFAULT_SCAN_UNIVERSE,  # TOS export; unioned with swing universe
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
    "feed":              "sip",
    # 4H aggregation: align to RTH so each session yields exactly two 4H bars
    # (09:30-13:30 and 13:30-17:30 ET, last bar can be partial). We label by
    # the bar's *start* timestamp.
    "rth_4h_anchor_minutes": [9 * 60 + 30, 13 * 60 + 30],   # in NY local
}

# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
MIN_4H_BARS = 250  # ~6 months of 4H bars before features are usable
CORRELATION_PRUNE_THRESHOLD = 0.995

# ---------------------------------------------------------------------------
# Momentum candidate filter
# ---------------------------------------------------------------------------
# Coarse live/execution gate for actionable momentum-continuation setups.
# Keep this separate from training scope: the May 31, 2026 cached-data sweep
# found broad cross-sectional ranker training beat training only on this gate.
MOMENTUM_CANDIDATE_FILTER_CONFIG: dict = {
    "enabled": True,
    "exclude_low_price": True,
    "min_dollar_vol_pctile_252": 0.20,
    "max_dist_to_52w_high_atr": 8.0,
    "min_xsec_near_high_rank": 0.60,
    "min_rs_spy_20": 0.0,
    "min_xsec_ret_20_rank": 0.60,
    "min_range_pos_20": 0.45,
}

# ---------------------------------------------------------------------------
# Training matrix
# ---------------------------------------------------------------------------
TRAINING_MATRIX_CONFIG: dict = {
    "target_column": "expansion_survival_score",
    "target_kind": "regression",
    # Train the ranker on broad valid rows, then apply candidate/setup filters
    # after scoring for execution.
    "apply_momentum_candidate_filter": False,
}

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
    # Live default from the May 24, 2026 entry-timing replay:
    # enter the first chronological 1H confirmation after the 4H score clears
    # this floor. This avoids waiting for the hindsight-best bar inside a
    # momentum campaign while keeping lower-quality early campaigns out.
    "min_score":     0.85,    # absolute floor — name is rejected even if in top-N
    "tie_break":     "expansion_score",
    "use_momentum_candidate_filter": True,
}

# ---------------------------------------------------------------------------
# 1H entry rules
# ---------------------------------------------------------------------------
ENTRY_RULES_CONFIG: dict = {
    # Default confirmation from the May 2026 trigger sweep:
    # use SPY-style 1H body breakout as the main trigger and pullback reclaim
    # as the higher-quality secondary continuation trigger.
    "enabled_rules":           ("break_body_prev_high", "pullback_continuation"),
    # Default confirmation from the May 2026 trigger sweep:
    # use SPY-style 1H body breakout as the main trigger and pullback reclaim
    # as the higher-quality secondary continuation trigger.
    "enabled_rules":           ("break_body_prev_high", "pullback_continuation"),
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

CAMPAIGN_CONFIG: dict = {
    # Defaults from the May 2026 adaptive-campaign and entry-timing replays:
    # first entry uses the same 0.85 live floor as the ranking layer; take up
    # to two entries in an expansion campaign normally; allow extra
    # continuation legs only after the campaign has already paid us and the
    # survival score is exceptional again.
    "enabled": True,
    "reset_gap_days": 15,
    "max_entries": 4,
    "entry_1_min_score": 0.85,
    "entry_2_min_score": 0.50,
    "entry_3_min_score": 0.80,
    "entry_4_min_score": 0.85,
    "entry_3_min_cumulative_return": 0.20,
}

OPTION_POLICY_CONFIG: dict = {
    # Strike & DTE selection
    "target_dte_min_days": 30,
    "target_dte_max_days": 60,
    "target_delta_long":   0.45,        # default fallback; adaptive delta below controls live selection
    "target_delta_long":   0.45,        # default fallback; adaptive delta below controls live selection
    "delta_tolerance":     0.10,
    # Adaptive option moneyness from the May 2026 theoretical delta grid:
    # use more convexity only when the expansion score is exceptional and the
    # campaign is still early; use more intrinsic/delta when confidence is
    # lower or the campaign is already mature.
    "adaptive_delta_enabled": True,
    "adaptive_delta_high_score": 0.85,
    "adaptive_delta_mid_score": 0.65,
    "adaptive_delta_high": 0.35,
    "adaptive_delta_mid": 0.45,
    "adaptive_delta_low": 0.55,
    "adaptive_delta_late_campaign_entry": 3,
    # Adaptive option moneyness from the May 2026 theoretical delta grid:
    # use more convexity only when the expansion score is exceptional and the
    # campaign is still early; use more intrinsic/delta when confidence is
    # lower or the campaign is already mature.
    "adaptive_delta_enabled": True,
    "adaptive_delta_high_score": 0.85,
    "adaptive_delta_mid_score": 0.65,
    "adaptive_delta_high": 0.35,
    "adaptive_delta_mid": 0.45,
    "adaptive_delta_low": 0.55,
    "adaptive_delta_late_campaign_entry": 3,
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
    "atr_trail_distance":  2.4,         # best replay: give expansion pullbacks more room once armed
    "score_decay_exit":    0.30,        # best replay: wait for deeper survival-score decay before exiting
    "trend_break_atr":     None,        # best replay: disabled; EMA trend breaks cut runners too early
    "atr_trail_distance":  2.4,         # best replay: give expansion pullbacks more room once armed
    "score_decay_exit":    0.30,        # best replay: wait for deeper survival-score decay before exiting
    "trend_break_atr":     None,        # best replay: disabled; EMA trend breaks cut runners too early
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
