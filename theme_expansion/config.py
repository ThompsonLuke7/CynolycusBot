from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DAILY_DIR = DATA_DIR / "daily_bars"
OUTPUT_DIR = ROOT / "outputs"
BACKTEST_DIR = OUTPUT_DIR / "backtests"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"

DEFAULT_THEME_MAP_PATH = DATA_DIR / "default_theme_map.csv"
THEME_DEFINITION_PATH = DATA_DIR / "theme_definition.csv"
THEME_MAP_V3_1_PATH = DATA_DIR / "theme_map_v3_1.csv"
THEME_DEFINITION_V3_1_PATH = DATA_DIR / "theme_definition_v3_1.csv"
THEME_MAP_V2_PATH = DATA_DIR / "theme_map_v2.csv"
THEME_DEFINITION_V2_PATH = DATA_DIR / "theme_definition_v2.csv"
THEME_MAP_PATH = (
    THEME_MAP_V3_1_PATH
    if THEME_MAP_V3_1_PATH.exists()
    else THEME_MAP_V2_PATH
    if THEME_MAP_V2_PATH.exists()
    else DEFAULT_THEME_MAP_PATH
)
ACTIVE_THEME_DEFINITION_PATH = (
    THEME_DEFINITION_V3_1_PATH
    if THEME_DEFINITION_V3_1_PATH.exists()
    else THEME_DEFINITION_V2_PATH
    if THEME_DEFINITION_V2_PATH.exists()
    else THEME_DEFINITION_PATH
)
LEGACY_THEME_MAP_PATH = DATA_DIR / "ticker_theme_map.csv"
DAILY_BARS_PATH = OUTPUT_DIR / "daily_bars.parquet"
UNIVERSE_FILTER_PATH = OUTPUT_DIR / "universe_filter.csv"
THEME_DAILY_PATH = OUTPUT_DIR / "theme_daily.parquet"
THEME_SCORES_PATH = OUTPUT_DIR / "theme_scores.parquet"
THEME_LEADERS_PATH = OUTPUT_DIR / "theme_leaders.parquet"
THEME_MEMBER_GRAPH_PATH = OUTPUT_DIR / "theme_member_graph.parquet"
LIVE_RANKING_PATH = OUTPUT_DIR / "live_theme_ranking.csv"
THEME_SIGNAL_LABELS_PATH = OUTPUT_DIR / "theme_signal_labels.parquet"
LIVE_THEME_SIGNAL_RANKING_PATH = OUTPUT_DIR / "live_theme_signal_ranking.csv"

START_DATE = "2018-01-01"
# yfinance treats end as exclusive, so this includes bars through 2026-05-21.
END_DATE = "2026-05-22"
BENCHMARK_TICKERS = ("SPY", "QQQ")

TOP_N_THEMES = 3
LEADERS_PER_THEME = 3
HOLD_DAYS = 5
REBALANCE_EVERY_DAYS = 1
TRANSACTION_COST_BPS = 5.0
MIN_PRICE = 3.0
MIN_MARKET_CAP = 300_000_000
MIN_AVG_DOLLAR_VOLUME_20D = 20_000_000
FILTER_EXCEPTIONS = {"RKLB", "LUNR", "OKLO", "SMR", "HOOD", "PLTR"}
THEME_FILTER_OVERRIDES = {
    # Emerging themes can be investable before their constituents satisfy the
    # broad liquidity/price profile of mature large-cap themes.
    "additive_manufacturing": {
        "min_price": 1.50,
        "min_avg_dollar_volume_20d": 4_000_000,
        "min_market_cap": 300_000_000,
    },
}

THEME_SCORE_WEIGHTS = {
    "rank_5d": 30.0,
    "theme_breadth": 25.0,
    "entropy_score": 15.0,
    "theme_rvol": 15.0,
    "rank_change_5d": 15.0,
    "rank_stability_5d": 10.0,
}


def ensure_dirs() -> None:
    for path in (DATA_DIR, RAW_DAILY_DIR, OUTPUT_DIR, BACKTEST_DIR, PLOTS_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace("$", "")


def theme_columns() -> list[str]:
    return ["theme_1", "theme_2", "theme_3"]


def _parse_bool(value: object, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def load_theme_definitions(path: Path | None = None) -> pd.DataFrame:
    """Load theme metadata and tolerate allow_etfs/benchmark column swaps."""
    src = path or ACTIVE_THEME_DEFINITION_PATH
    if not src.exists():
        return pd.DataFrame(
            columns=[
                "theme",
                "category",
                "min_constituents",
                "max_constituents",
                "allow_etfs",
                "benchmark",
                "is_tradable",
                "is_watchlist_only",
            ]
        )
    defs = pd.read_csv(src)
    defs.columns = [c.strip().lower() for c in defs.columns]
    if "theme" not in defs.columns:
        raise ValueError(f"{THEME_DEFINITION_PATH} must include a theme column")

    defs["theme"] = defs["theme"].astype(str).str.strip().str.lower()
    if "allow_etfs" not in defs.columns:
        defs["allow_etfs"] = False
    if "benchmark" not in defs.columns:
        defs["benchmark"] = "SPY"

    # Current file shape is theme,...,allow_etfs,benchmark but values look like
    # allow_etfs=SPY and benchmark=False. Treat that as benchmark=SPY,
    # allow_etfs=False.
    allow_text = defs["allow_etfs"].astype(str).str.upper()
    bench_text = defs["benchmark"].astype(str).str.lower()
    swapped = allow_text.str.fullmatch(r"[A-Z.]{1,8}", na=False) & bench_text.isin(["true", "false"])
    if swapped.any():
        old_allow = defs.loc[swapped, "allow_etfs"].astype(str).copy()
        old_benchmark = defs.loc[swapped, "benchmark"].astype(str).copy()
        defs["allow_etfs"] = defs["allow_etfs"].astype(object)
        defs["benchmark"] = defs["benchmark"].astype(object)
        defs.loc[swapped, "allow_etfs"] = old_benchmark
        defs.loc[swapped, "benchmark"] = old_allow

    defs["allow_etfs"] = defs["allow_etfs"].map(_parse_bool)
    defs["benchmark"] = defs["benchmark"].fillna("SPY").astype(str).str.strip().str.upper()
    if "is_tradable" not in defs.columns:
        defs["is_tradable"] = True
    if "is_watchlist_only" not in defs.columns:
        defs["is_watchlist_only"] = False
    defs["is_tradable"] = defs["is_tradable"].map(_parse_bool)
    defs["is_watchlist_only"] = defs["is_watchlist_only"].map(_parse_bool)
    for col in ("min_constituents", "max_constituents"):
        if col in defs.columns:
            defs[col] = pd.to_numeric(defs[col], errors="coerce")
    return defs.drop_duplicates(subset=["theme"])


def load_theme_memberships(path: Path | None = None) -> pd.DataFrame:
    """
    Return one row per ticker/theme with columns ticker, asset_type, theme.

    Supports the new simple schema:
      ticker, asset_type, theme_1[, theme_2, theme_3]

    Also supports the older wide map:
      ticker, sector, industry, subindustry, theme_1, theme_2, theme_3
    """
    src = path or THEME_MAP_PATH
    df = pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError(f"{src} must include a ticker column")
    df["ticker"] = df["ticker"].map(clean_ticker)
    df = df[df["ticker"].ne("")]
    if "asset_type" not in df.columns:
        df["asset_type"] = "stock"
    df["asset_type"] = df["asset_type"].fillna("stock").astype(str).str.strip().str.lower()

    theme_cols = [c for c in theme_columns() if c in df.columns]
    if not theme_cols and "theme" in df.columns:
        theme_cols = ["theme"]
    if not theme_cols:
        raise ValueError(f"{src} must include at least one theme column")

    id_cols = [c for c in ["ticker", "asset_type", "sector", "industry", "subindustry"] if c in df.columns]
    if theme_cols == ["theme"]:
        long_map = df[id_cols + ["theme"]].copy()
    else:
        long_map = df.melt(id_vars=id_cols, value_vars=theme_cols, value_name="theme")
        long_map = long_map.drop(columns=["variable"], errors="ignore")
    long_map["theme"] = long_map["theme"].fillna("").astype(str).str.strip().str.lower()
    long_map = long_map[long_map["theme"].ne("")]

    defs = load_theme_definitions()
    if not defs.empty:
        meta_cols = [
            "theme",
            "category",
            "min_constituents",
            "max_constituents",
            "allow_etfs",
            "benchmark",
            "is_tradable",
            "is_watchlist_only",
        ]
        long_map = long_map.merge(defs[[c for c in meta_cols if c in defs.columns]], on="theme", how="left")
        unknown = sorted(long_map[long_map["category"].isna()]["theme"].unique())
        if unknown:
            raise ValueError(f"themes missing from {ACTIVE_THEME_DEFINITION_PATH}: {unknown}")
        long_map = long_map[(long_map["asset_type"].ne("etf")) | (long_map["allow_etfs"].fillna(False))]
        counts = long_map.groupby("theme")["ticker"].transform("nunique")
        min_counts = pd.to_numeric(long_map["min_constituents"], errors="coerce").fillna(5)
        max_counts = pd.to_numeric(long_map["max_constituents"], errors="coerce").fillna(25)
        long_map["theme_constituent_count"] = counts
        auto_watchlist = counts < min_counts
        long_map["is_below_min_constituents"] = auto_watchlist
        long_map["is_above_max_constituents"] = counts > max_counts
        long_map["is_watchlist_only"] = long_map["is_watchlist_only"].map(_parse_bool).fillna(False) | auto_watchlist
        long_map["is_tradable"] = (
            long_map["is_tradable"].map(lambda value: _parse_bool(value, True)).fillna(True)
            & ~long_map["is_watchlist_only"]
        )
    if "is_tradable" not in long_map.columns:
        long_map["is_tradable"] = True
    if "is_watchlist_only" not in long_map.columns:
        long_map["is_watchlist_only"] = False
    if "theme_constituent_count" not in long_map.columns:
        long_map["theme_constituent_count"] = long_map.groupby("theme")["ticker"].transform("nunique")
    long_map["is_tradable"] = long_map["is_tradable"].map(lambda value: _parse_bool(value, True)).fillna(True)
    long_map["is_watchlist_only"] = long_map["is_watchlist_only"].map(_parse_bool).fillna(False)
    long_map["theme_constituent_count"] = (
        pd.to_numeric(long_map["theme_constituent_count"], errors="coerce")
        .fillna(long_map.groupby("theme")["ticker"].transform("nunique"))
    )
    return long_map.drop_duplicates(subset=["ticker", "theme"]).sort_values(["theme", "ticker"]).reset_index(drop=True)
