"""
Central configuration for the multi-ticker swing trading pipeline.

All paths, dates, feature lists, and model/backtest parameters live here.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

UNIVERSE_CSV = MODULE_ROOT / "config" / "swing_trader_universe_v3.csv"

RAW_30M_DIR = MODULE_ROOT / "data" / "raw" / "30m"
RAW_10M_DIR = MODULE_ROOT / "data" / "raw" / "10m"
PROCESSED_DIR = MODULE_ROOT / "data" / "processed"
PROCESSED_30M_DIR = PROCESSED_DIR / "30m"        # per-ticker feature parquets
PROCESSED_LBL_DIR = PROCESSED_DIR / "labels"     # per-ticker label parquets

FEATURES_COMBINED = PROCESSED_DIR / "features_30m.parquet"
LABELS_COMBINED   = PROCESSED_DIR / "labels_30m.parquet"
TRAINING_MATRIX   = PROCESSED_DIR / "training_matrix.parquet"

MODELS_DIR  = MODULE_ROOT / "models"
MODEL_PATH  = MODELS_DIR / "swing_xgb_model.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
EVAL_METRICS_PATH = MODELS_DIR / "eval_metrics.json"

BACKTEST_RESULTS_DIR = MODULE_ROOT / "backtest" / "results"
PLOTS_DIR = MODULE_ROOT / "plots" / "output"

# ---------------------------------------------------------------------------
# Data range
# ---------------------------------------------------------------------------
TRAIN_START = "2021-01-01"
TRAIN_END   = "2026-04-15"

# ---------------------------------------------------------------------------
# Context / market tickers (always fetched; used for relative-strength features)
# ---------------------------------------------------------------------------
CONTEXT_TICKERS: list[str] = ["SPY", "QQQ", "IWM"]

# ---------------------------------------------------------------------------
# Feature list  (exact column names produced by build_features.py)
# ---------------------------------------------------------------------------
# Cat 1 – Price / return
FEATURES_PRICE = [
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_16",
    "log_ret_1",
    "close_to_close_z_20",
    "open_to_close_ret",
    "high_low_range_pct",
    "close_location_in_bar",
]

# Cat 2 – Volatility / expansion
FEATURES_VOL = [
    "atr_pct_14",
    "atr_pct_z_64",
    "realized_vol_4",
    "realized_vol_16",
    "realized_vol_32",
    "range_expansion_4",
    "range_expansion_16",
    "range_regime_8_32",
    "true_range_pct",
    "vol_of_vol_16",
]

# Cat 3 – Trend / structure
FEATURES_TREND = [
    "ema_dist_10", "ema_dist_20", "ema_dist_50",
    "ema_slope_10", "ema_slope_20",
    "ema_stack_state",
    "adx_14", "dmp_14", "dmn_14", "adxr_14",
    "trend_phase_m", "trend_phase_a",
]

# Cat 4 – Swing / setup structure
FEATURES_SWING = [
    "swing_setup_score_long",
    "swing_setup_score_short",
    "bars_since_long_setup",
    "bars_since_short_setup",
    "dist_to_recent_swing_high",
    "dist_to_recent_swing_low",
    "range_pos_20",
    "dist_20bar_high",
    "dist_20bar_low",
    "compression_score",
    "breakout_pressure_score",
]

# Cat 5 – Prior model outputs / lags  (populated once GA-XGBoost pivot models exist)
FEATURES_MODEL_LAGS = [
    "p_setup_long",   "p_setup_short",
    "p_setup_long_lag1",  "p_setup_long_lag2",
    "p_setup_short_lag1", "p_setup_short_lag2",
    "p_setup_long_delta_1",  "p_setup_short_delta_1",
    "p_setup_long_max_last_4",  "p_setup_short_max_last_4",
]

# Cat 6 – Volume / participation
FEATURES_VOLUME = [
    "volume_z_20",
    "volume_rel_20",
    "dollar_volume",
    "dollar_volume_rel_20",
    "dollar_volume_pctile",
    "obv_slope_10",
    "accdist_slope_10",
    "relative_volume_open_window",
]

# Cat 7 – Gap / overnight context
FEATURES_GAP = [
    "gap_pct",
    "gap_to_atr",
    "overnight_ret",
    "prev_day_ret",
    "prev_day_range_pct",
    "gap_followthrough_1",
    "gap_followthrough_2",
]

# Cat 8 – Relative strength / market context
FEATURES_RS = [
    "spy_ret_1",  "spy_ret_4",  "spy_ret_16",
    "qqq_ret_1",  "qqq_ret_4",  "qqq_ret_16",
    "iwm_ret_1",  "iwm_ret_4",  "iwm_ret_16",
    "rel_str_spy_4",  "rel_str_spy_16",
    "rel_str_qqq_4",
    "beta_like_spy_64",
    "corr_spy_64",
    "market_regime_proxy",
]

# Cat 9 – Location / mean-reversion context
FEATURES_LOCATION = [
    "dist_to_vwap",
    "dist_to_pdh", "dist_to_pdl", "dist_to_pdc",
    "dist_to_rolling_mean_20",
    "zscore_close_20", "zscore_close_64",
    "percentile_close_20", "percentile_close_64",
]

# Cat 10 – Time features
FEATURES_TIME = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "is_monday", "is_friday",
    "bars_from_open", "bars_to_close",
]

# Cat 11 – Behavioral identity
FEATURES_BEHAVIORAL = [
    "sector_id",
    "market_cap_bucket",
    "volatility_pctile_rolling",
    "dollar_vol_pctile_rolling",
    "gap_frequency_rolling",
    "trendiness_score_rolling",
    "stock_beta_bucket",
]

# Complete ordered feature list used for model training
# Cat 5 (model lags) excluded by default until models exist; add manually when ready.
FEATURE_COLUMNS: list[str] = (
    FEATURES_PRICE
    + FEATURES_VOL
    + FEATURES_TREND
    + FEATURES_SWING
    + FEATURES_VOLUME
    + FEATURES_GAP
    + FEATURES_RS
    + FEATURES_LOCATION
    + FEATURES_TIME
    + FEATURES_BEHAVIORAL
)

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
TRIPLE_BARRIER_CONFIG: dict = {
    "k_up": 2.5,            # TP in ATR units
    "k_dn": 2.5,            # SL in ATR units
    "max_holding": 48,      # 48 × 30min ≈ 3 trading days
    "chop_atr_mult": 0.5,   # time-expiry label as chop if move < 0.5×ATR
    "atr_length": 14,
}

SWING_STATE_CONFIG: dict = {
    "threshold": 0.65,
    "confirm_threshold": 0.75,
    "confirm_mult": 0.12,
    "pending_max_bars": 10,
    "tp_mult": 0.5,
    "sl_mult": 1.0,
    "max_holding": 48,
    "cooldown_bars": 3,
    "session_open_minutes": 30,
    "session_late_minutes": 30,
}

# Minimum bars required before a ticker's features are considered valid
MIN_BARS = 200

# ---------------------------------------------------------------------------
# Time-based train / val / test split
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15 (remainder)

# ---------------------------------------------------------------------------
# XGBoost multiclass model
# Classes: 0 = short  |  1 = neutral  |  2 = long
# ---------------------------------------------------------------------------
XGBOOST_CONFIG: dict = {
    "objective":          "multi:softprob",
    "num_class":          3,
    "n_estimators":       400,
    "max_depth":          5,
    "learning_rate":      0.04,
    "subsample":          0.80,
    "colsample_bytree":   0.75,
    "min_child_weight":   10,
    "gamma":              0.1,
    "reg_alpha":          0.05,
    "reg_lambda":         1.0,
    "random_state":       42,
    "n_jobs":             -1,
    "eval_metric":        "mlogloss",
    "early_stopping_rounds": 40,
}

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
BACKTEST_CONFIG: dict = {
    "tp_atr_mult":          2.5,
    "sl_atr_mult":          1.0,
    "entry_bars_max":       3,       # max 10m bars after signal before giving up
    "prob_threshold_long":  0.50,    # model P(class=long) to enter
    "prob_threshold_short": 0.50,
    "swing_gate_required":  True,    # only trade when swing state gate is active
    "max_concurrent":       5,       # portfolio-level position limit
    "position_size_pct":    0.02,    # 2 % of equity per trade
    "initial_capital":      100_000.0,
    "commission_pct":       0.001,   # 0.10 % round-trip
}
