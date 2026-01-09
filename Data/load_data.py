from pathlib import Path

import pandas as pd

from Data.retrieve_data import get_output_path, normalize_ticker, retrieve_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"


def get_ticker_data_dir(ticker: str, base_dir: Path | None = None) -> Path:
    base = base_dir if base_dir is not None else DATA_DIR
    slug = normalize_ticker(ticker).lower()
    return base / slug


def get_ticker_raw_dir(ticker: str, base_dir: Path | None = None) -> Path:
    return get_ticker_data_dir(ticker, base_dir) / "raw"


def get_ticker_processed_dir(ticker: str, base_dir: Path | None = None) -> Path:
    return get_ticker_data_dir(ticker, base_dir) / "processed"


def get_ticker_processed_base_dir(ticker: str, base_dir: Path | None = None) -> Path:
    return get_ticker_processed_dir(ticker, base_dir) / "base"


def get_ticker_processed_split_dir(ticker: str, base_dir: Path | None = None) -> Path:
    return get_ticker_processed_dir(ticker, base_dir) / "splits"


def get_ticker_processed_stats_dir(ticker: str, base_dir: Path | None = None) -> Path:
    return get_ticker_processed_dir(ticker, base_dir) / "stats"


def get_ticker_plots_dir(ticker: str, base_dir: Path | None = None) -> Path:
    return get_ticker_data_dir(ticker, base_dir) / "plots"


def ensure_ticker_dirs(ticker: str, base_dir: Path | None = None) -> dict[str, Path]:
    """
    Ensure the standard directory structure exists for a ticker.
    """
    data_dir = get_ticker_data_dir(ticker, base_dir)
    raw_dir = get_ticker_raw_dir(ticker, base_dir)
    processed_dir = get_ticker_processed_dir(ticker, base_dir)
    processed_base_dir = get_ticker_processed_base_dir(ticker, base_dir)
    processed_split_dir = get_ticker_processed_split_dir(ticker, base_dir)
    processed_stats_dir = get_ticker_processed_stats_dir(ticker, base_dir)
    plots_dir = get_ticker_plots_dir(ticker, base_dir)

    for path in [
        data_dir,
        raw_dir,
        processed_dir,
        processed_base_dir,
        processed_split_dir,
        processed_stats_dir,
        plots_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "data": data_dir,
        "raw": raw_dir,
        "processed": processed_dir,
        "processed_base": processed_base_dir,
        "processed_splits": processed_split_dir,
        "processed_stats": processed_stats_dir,
        "plots": plots_dir,
    }


def get_raw_data_path(ticker: str) -> Path:
    """
    Build the CSV path for the requested ticker under Data/.
    """
    return get_output_path(ticker, base_dir=DATA_DIR)


def ensure_raw_data(ticker: str) -> Path:
    """
    Make sure we have a CSV for the ticker; download it if missing.
    """
    raw_path = get_raw_data_path(ticker)
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw = retrieve_data(ticker)
        df_raw.to_csv(raw_path)
        print(f"Downloaded {normalize_ticker(ticker)} data to {raw_path}")
    return raw_path


def load_ticker_csv(ticker: str) -> pd.DataFrame:
    """
    Load a daily OHLCV CSV for the ticker and normalize columns/index.
    """
    path = ensure_raw_data(ticker)
    df = pd.read_csv(path, index_col=0, parse_dates=[0], date_format="%Y-%m-%d")
    df.index = pd.to_datetime(df.index, errors="coerce", format="%Y-%m-%d")
    df = df[df.index.notna()]

    keep_cols = [
        c
        for c in df.columns
        if c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    ]
    df = df[keep_cols]

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_ticker_parquet(
    ticker: str,
    parquet_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load a parquet of OHLCV data for a ticker and normalize columns/index.

    If parquet_path is None, defaults to Data/{ticker}_intraday.parquet.
    """
    slug = normalize_ticker(ticker).lower()
    path = resolve_intraday_parquet_path(ticker, parquet_path=parquet_path)

    df = pd.read_parquet(path)

    rename_map = {
        "timestamp": "date",
        "Date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "volume": "volume",
    }
    df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.set_index("date")
    elif df.index.name:
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)

    df = df[df.index.notna()]

    keep_cols = [
        c
        for c in ["open", "high", "low", "close", "adj_close", "volume"]
        if c in df.columns
    ]
    df = df[keep_cols]

    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resolve_intraday_parquet_path(
    ticker: str, parquet_path: str | Path | None = None
) -> Path:
    """
    Resolve the parquet path for a ticker's intraday data.
    """
    slug = normalize_ticker(ticker).lower()
    if parquet_path is not None:
        return Path(parquet_path)

    raw_dir = get_ticker_raw_dir(ticker)
    candidates = sorted(raw_dir.glob(f"{slug}_intraday_*.parquet"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    path = raw_dir / f"{slug}_intraday_1hr.parquet"
    legacy_path = DATA_DIR / f"{slug}_intraday_1hr.parquet"
    if not path.exists() and legacy_path.exists():
        return legacy_path
    return path


def get_processed_feature_path(ticker: str, prefix: str | None = None) -> Path:
    """
    Build the path where the processed feature/label matrix cache is stored.
    Ensures the base processed directory exists so we can write the cache.
    """
    slug = normalize_ticker(ticker).lower()
    processed_dir = get_ticker_processed_base_dir(ticker)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        return processed_dir / f"{slug}_features_with_labels.parquet"
    return processed_dir / f"{prefix}_features_with_labels.parquet"


def load_cached_features(cache_path: Path):
    """
    Load a previously cached feature matrix if present, otherwise return None.
    """
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index, errors="coerce")
        cached = cached[cached.index.notna()]
        print(f"Loaded cached features + labels from {cache_path}")
        return cached
    return None
