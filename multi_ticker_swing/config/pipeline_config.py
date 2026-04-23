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
RAW_5M_DIR  = MODULE_ROOT / "data" / "raw" / "5m"
RAW_10M_DIR = MODULE_ROOT / "data" / "raw" / "10m"   # deprecated — switched to 5m for entry confirmation
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
CONTEXT_TICKERS: list[str] = ["SPY", "QQQ", "IWM", "TLT", "USO", "GLD"]

# Tickers kept in universe for training features but excluded from live trading / backtests
TRADING_BLACKLIST: list[str] = [
    # Rates-driven — equity momentum features have no causal relationship to price drivers
    "TLT",   # 20yr Treasury — use as regime feature only
    "HYG",   # high yield bonds — same issue
    "LQD",   # investment grade bonds — same issue
    "SHY",   # short-duration Treasury — same issue, near-zero ATR
    # Rate proxies — price driven by Fed expectations, not price momentum
    "XLU",   # utilities — interest-rate proxy
    "SCHW",  # Charles Schwab — NII-driven like a bond; 66% SL rate, WR=34%
    "BAC",   # Bank of America — extreme NII sensitivity same as SCHW; 66% SL rate, WR=34%
    # Inverse/leveraged products with structural decay — timing mismatch
    "SCO",   # 2x inverse crude oil — contango decay + lead/lag vs USO; 52% SL rate
    # Broad sector ETFs — diversification kills fractal momentum; model systematically wrong
    "XLP",   # Consumer Staples ETF — WR=23%, SL%=77%
    "XLI",   # Industrials ETF — WR=25%, SL%=71%
    "XLB",   # Materials ETF — WR=27%, SL%=73%
    "PG",    # Procter & Gamble — quintessential anti-momentum stock; WR=28%, SL%=69%
    # Hyper-efficient benchmark — fractal momentum approach doesn't apply
    "SPY",   # S&P 500 ETF — use as context feature only; WR=30%, SL%=70%
]

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
    # Equity index context
    "spy_ret_1",  "spy_ret_4",  "spy_ret_16",
    "qqq_ret_1",  "qqq_ret_4",  "qqq_ret_16",
    "iwm_ret_1",  "iwm_ret_4",  "iwm_ret_16",
    "rel_str_spy_4",  "rel_str_spy_16",
    "rel_str_qqq_4",
    "beta_like_spy_64",
    "corr_spy_64",
    "market_regime_proxy",
    # Rate context — TLT moves tell the model when rate pressure is driving sectors
    "tlt_ret_1",  "tlt_ret_4",  "tlt_ret_16",
    # Crude oil context — causal driver for energy names (XOM, XLE, HAL, SLB, etc.)
    "uso_ret_1",  "uso_ret_4",  "uso_ret_16",
    "rel_str_uso_4",   # ticker's 4-bar return relative to USO (is it outperforming crude?)
    "corr_uso_64",     # 64-bar rolling correlation to USO (is this an oil-beta name?)
    # Gold context — causal driver for metals/miners (GLD, GDX, SLV, COPX, PAAS, etc.)
    "gld_ret_1",  "gld_ret_4",  "gld_ret_16",
    "rel_str_gld_4",   # ticker's 4-bar return relative to GLD
    "corr_gld_64",     # 64-bar rolling correlation to GLD
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
# Note: hour_sin/hour_cos (24h encoding) removed — RTH spans only hour_cos ∈ [-1, -0.5],
# providing nearly zero signal.  Replaced with session-relative fraction.
FEATURES_TIME = [
    "session_frac", "session_frac_sin",
    "day_of_week_sin", "day_of_week_cos",
    "is_monday", "is_friday",
    "bars_from_open", "bars_to_close",
]

# Cat 12 – Daily (HTF) context — shift(1) applied; no leakage into intraday bars
FEATURES_DAILY = [
    "daily_ret_1",
    "daily_atr_pct",
    "daily_rsi_14",
    "daily_ema_dist_20",
    "daily_ema_dist_50",
    "daily_trend_state",
    "daily_range_pos_20",
    "daily_vol_rel_20",
]

# Cat 3b – Momentum oscillators
FEATURES_OSCILLATORS = [
    "rsi_14",
    "macd_hist_12_26_9",
    "bb_pct_b_20",
    "stoch_k_14",
    "mfi_14",
    "chop_14",
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
    + FEATURES_OSCILLATORS
    + FEATURES_SWING
    + FEATURES_VOLUME
    + FEATURES_GAP
    + FEATURES_RS
    + FEATURES_LOCATION
    + FEATURES_TIME
    + FEATURES_DAILY
    + FEATURES_BEHAVIORAL
)

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
# Soft swing zone label config (30m primary training target)
# Mirrors generate_soft_swing_30m_plots.py constants — keep in sync.
SWING_LABEL_30M_CONFIG: dict = {
    "sequence_count": 3,  # fractal pivot: 3 bars each side
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
    "n_estimators":       1500,   # loss was still descending at 600 — needs more trees
    "max_depth":          6,      # slightly deeper for 187-ticker dataset
    "learning_rate":      0.03,   # lower LR paired with more estimators
    "subsample":          0.75,   # tighter than 10m; cross-ticker noise
    "colsample_bytree":   0.70,   # ~70% of ~90 features per tree = ~63
    "min_child_weight":   20,     # discourage fine splits on noisy pivot labels
    "gamma":              0.1,
    "reg_alpha":          0.10,   # L1 sparsity — more features than 10m
    "reg_lambda":         1.0,
    "random_state":       42,
    "n_jobs":             -1,
    "eval_metric":        "mlogloss",
    "early_stopping_rounds": 60,  # more patience at lower LR
}

# OOF cross-validation folds (time-series sequential)
OOF_N_FOLDS: int = 5

# Neutral class downweight — neutral is ~56% of bars; multiply its sample
# weight by this factor so the model doesn't predict neutral for everything.
NEUTRAL_WEIGHT_FACTOR: float = 0.40

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
BACKTEST_CONFIG: dict = {
    "tp_atr_mult":          2.5,
    "sl_atr_mult":          4.0,     # 4.0×ATR stop — wider stop improves WR and trail exit rate
    "entry_bars_max":       3,       # max 5m bars after signal before giving up
    "prob_threshold_long":  0.70,    # directional-conditional P(long|dir) threshold
    "prob_threshold_short": 0.70,
    "arm_pct":              0.025,   # underlying move to arm trailing exit (2.5%)
    "giveback_pct":         0.25,    # fraction of peak profit given back before exit (25%)
    "confirm_max_5m":       6,       # 30-minute confirmation window (6 × 5m bars)
    "swing_gate_required":  True,    # only trade when swing state gate is active
    "max_concurrent":       5,       # portfolio-level position limit
    "position_size_pct":    0.02,    # 2 % of equity per trade
    "initial_capital":      100_000.0,
    "commission_pct":       0.001,   # 0.10 % round-trip
}
