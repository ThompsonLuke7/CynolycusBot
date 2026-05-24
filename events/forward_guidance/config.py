"""Central configuration for the forward-guidance earnings pipeline."""

from __future__ import annotations

from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[1]

DATA_DIR = MODULE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MARKET_WINDOWS_DIR = DATA_DIR / "market_windows"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

EVENTS_PATH = PROCESSED_DIR / "earnings_events.parquet"
DISCOVERED_EVENTS_CSV = PROCESSED_DIR / "discovered_earnings_events.csv"
FEATURES_PATH = PROCESSED_DIR / "feature_matrix.parquet"
LABELS_PATH = PROCESSED_DIR / "labels.parquet"
TRAINING_MATRIX = PROCESSED_DIR / "training_matrix.parquet"

MODELS_DIR = MODULE_ROOT / "models"
XGB_MODEL_PATH = MODELS_DIR / "forward_guidance_xgb.json"
LGB_MODEL_PATH = MODELS_DIR / "forward_guidance_lgb.txt"
MODEL_META_PATH = MODELS_DIR / "forward_guidance_model_meta.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
EVAL_METRICS_PATH = MODELS_DIR / "eval_metrics.json"

INFERENCE_DIR = MODULE_ROOT / "inference" / "output"
RANKED_OUTPUT_PARQUET = INFERENCE_DIR / "ranked_opportunities.parquet"
RANKED_OUTPUT_CSV = INFERENCE_DIR / "ranked_opportunities.csv"
DASHBOARD_STATE_PATH = INFERENCE_DIR / "dashboard_state.json"

BACKTEST_RESULTS_DIR = MODULE_ROOT / "backtests" / "results"

DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_MARKET_TIMEFRAME = "30Min"
DEFAULT_ALPACA_FEED = "IEX"
MARKET_LOOKBACK_DAYS = 35
MARKET_FORWARD_DAYS = 130

CONTEXT_TICKERS = ["SPY", "QQQ", "VIXY"]

SECTOR_ETFS: dict[str, str] = {
    "communication_services": "XLC",
    "communications": "XLC",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "energy": "XLE",
    "financials": "XLF",
    "health_care": "XLV",
    "healthcare": "XLV",
    "industrials": "XLI",
    "materials": "XLB",
    "real_estate": "XLRE",
    "technology": "XLK",
    "information_technology": "XLK",
    "utilities": "XLU",
}

DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"
ALT_FINBERT_MODEL = "yiyanghkust/finbert-tone"
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEC_BERT_MODEL = "nlpaueb/sec-bert-num"

PRIMARY_TARGET = "target"
PRIMARY_PROBABILITY = "future_60d_outperformance_probability"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
OOF_N_FOLDS = 5

XGBOOST_CONFIG: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.04,
    "n_estimators": 600,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": 4,
}

LIGHTGBM_CONFIG: dict[str, object] = {
    "objective": "binary",
    "n_estimators": 600,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": 4,
}

BACKTEST_CONFIG: dict[str, object] = {
    "prob_threshold": 0.70,
    "min_guidance_strength": 0.25,
    "require_bad_reaction": True,
    "require_stabilization": True,
    "hold_days": 60,
    "position_size_pct": 0.05,
    "atr_trail_mult": None,
}


def existing_swing_universe_csv() -> Path | None:
    """Return the existing swing universe CSV path when that package is available."""
    try:
        from multi_ticker_swing.config.pipeline_config import UNIVERSE_CSV
    except Exception:
        return None
    path = Path(UNIVERSE_CSV)
    return path if path.exists() else None


UNIVERSE_CSV = existing_swing_universe_csv() or (DATA_DIR / "universe.csv")


def ensure_data_dirs() -> None:
    """Create runtime cache/output directories used by the package."""
    for path in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        MARKET_WINDOWS_DIR,
        EMBEDDINGS_DIR,
        MODELS_DIR,
        INFERENCE_DIR,
        BACKTEST_RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
