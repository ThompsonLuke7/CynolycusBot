import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Data.retrieve_data import get_output_path, normalize_ticker, retrieve_data
from Features.custom_indicators import (
    add_atr_swing_state_features,
    add_fractal_pivots,
    add_rsilg_fe_gauss,
    add_tmo,
    add_vmd_return_features,
)
from Features.label_generations import add_all_labels, plot_zig_zag
from Features.pandas_ta_indicators import add_all_pandasta_indicators

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
global_file_path = str(PROJECT_ROOT)


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
        df_raw = retrieve_data(ticker)
        df_raw.to_csv(raw_path)
        print(f"Downloaded {normalize_ticker(ticker)} data to {raw_path}")
    return raw_path


def load_ticker_csv(ticker: str) -> pd.DataFrame:
    path = ensure_raw_data(ticker)
    df = pd.read_csv(path, index_col=0, parse_dates=[0], date_format="%Y-%m-%d")
    df.index = pd.to_datetime(df.index, errors="coerce", format="%Y-%m-%d")
    # Drop rows where index is NaT (these are 'Ticker', 'Date', etc.)
    df = df[df.index.notna()]
    # Keep only the columns we care about initially (drop any 'Symbol', etc.)
    keep_cols = [
        c
        for c in df.columns
        if c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    ]
    df = df[keep_cols]

    # Rename to lowercase OHLCV for pandas_ta
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Force OHLCV columns to numeric
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
    path = (
        Path(parquet_path)
        if parquet_path is not None
        else DATA_DIR / f"{slug}_intraday_1hr.parquet"
    )

    df = pd.read_parquet(path)

    # Standardize column names
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

    # Set datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.set_index("date")
    elif df.index.name:
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)

    df = df[df.index.notna()]

    # Keep only standard OHLCV columns if present
    keep_cols = [
        c
        for c in ["open", "high", "low", "close", "adj_close", "volume"]
        if c in df.columns
    ]
    df = df[keep_cols]

    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df.index.dayofweek  # 0 = Monday … 4 = Friday
    df["day_of_month"] = df.index.day  # sometimes helps
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_month_end"] = df.index.is_month_end.astype(int)
    df["is_month_start"] = df.index.is_month_start.astype(int)
    return df


def add_all_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_tmo(df)
    df = add_rsilg_fe_gauss(df)
    df = add_fractal_pivots(df)
    df = add_atr_swing_state_features(df)
    df = add_vmd_return_features(df)
    return df


def get_processed_feature_path(ticker: str) -> Path:
    """
    Build the path where the processed feature/label matrix is stored.
    Ensures the directory exists so we can write the cache.
    """
    slug = normalize_ticker(ticker).lower()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / f"{slug}_features_with_labels.parquet"


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


def get_default_plot_path(ticker: str) -> Path:
    slug = normalize_ticker(ticker).lower()
    plots_dir = DATA_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    filename = "atr_swing_plot.png" if slug == "spy" else f"{slug}_atr_swing_plot.png"
    return plots_dir / filename


def plot_atr_swing_signals(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Plot OHLC candles with ATR swing labels, pivots, and flip markers for a quick visual check.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos = np.arange(len(df))  # compressed positions
    open_y = df["open"].to_numpy()
    high_y = df["high"].to_numpy()
    low_y = df["low"].to_numpy()
    close_y = df["close"].to_numpy()
    if "atr" in df.columns:
        marker_offset = np.nanmedian(df["atr"].to_numpy())
    else:
        marker_offset = np.nanmedian(high_y - low_y)
    if not np.isfinite(marker_offset) or marker_offset <= 0:
        marker_offset = np.nanmax(high_y) * 0.001
    up_offset = marker_offset * 0.6
    down_offset = marker_offset * 0.6

    up = close_y >= open_y
    down = ~up
    wick_color = "#444444"
    up_color = "#1976D2"
    down_color = "#E53935"

    width = 0.8

    # Marker positions
    pivot_up_y = high_y + up_offset * 1.2
    pivot_dn_y = low_y - down_offset * 1.2
    swing_pos_y = close_y + up_offset
    swing_neg_y = close_y - down_offset

    # Wick lines
    ax.vlines(pos, low_y, high_y, color=wick_color, linewidth=1.0, zorder=1)
    # Candle bodies
    ax.bar(
        pos[up],
        close_y[up] - open_y[up],
        width=width,
        bottom=open_y[up],
        color=up_color,
        edgecolor="none",
        label="Bull candle",
        zorder=1.2,
    )
    ax.bar(
        pos[down],
        close_y[down] - open_y[down],
        width=width,
        bottom=open_y[down],
        color=down_color,
        edgecolor="none",
        label="Bear candle",
        zorder=1.2,
    )

    if "atr_swing_label" in df.columns:
        mask_pos = (df["atr_swing_label"].fillna(0) == 1).to_numpy()
        mask_neg = (df["atr_swing_label"].fillna(0) == -1).to_numpy()
        pos_idx = pos[mask_pos]
        neg_idx = pos[mask_neg]
        # Align with pivots when both occur
        if "pivot_down" in df.columns:
            mask_pivot_down = (df["pivot_down"].fillna(0).astype(int) == 1).to_numpy()
            coincide = mask_pos & mask_pivot_down
            swing_pos_y[coincide] = pivot_dn_y[coincide]
        if "pivot_up" in df.columns:
            mask_pivot_up = (df["pivot_up"].fillna(0).astype(int) == 1).to_numpy()
            coincide = mask_neg & mask_pivot_up
            swing_neg_y[coincide] = pivot_up_y[coincide]
        if len(pos_idx) > 0:
            ax.scatter(
                pos_idx,
                swing_pos_y[mask_pos],
                color="#1976D2",
                marker="^",
                s=42,
                label="atr_swing_label = +1",
                alpha=0.96,
                zorder=2,
            )
        if len(neg_idx) > 0:
            ax.scatter(
                neg_idx,
                swing_neg_y[mask_neg],
                color="#E53935",
                marker="v",
                s=42,
                label="atr_swing_label = -1",
                alpha=0.96,
                zorder=2,
            )

    if "pivot_down" in df.columns:
        mask_pivot_down = (df["pivot_down"].fillna(0).astype(int) == 1).to_numpy()
        pivot_idx = pos[mask_pivot_down]
        if len(pivot_idx) > 0:
            ax.scatter(
                pivot_idx,
                pivot_dn_y[mask_pivot_down],
                color="#2E7D32",
                marker="v",
                s=52,
                label="pivot_down",
                alpha=0.95,
                zorder=2.2,
            )

    if "pivot_up" in df.columns:
        mask_pivot_up = (df["pivot_up"].fillna(0).astype(int) == 1).to_numpy()
        pivot_idx = pos[mask_pivot_up]
        if len(pivot_idx) > 0:
            ax.scatter(
                pivot_idx,
                pivot_up_y[mask_pivot_up],
                color="#1E0D32",
                marker="^",
                s=52,
                label="pivot_up",
                alpha=0.95,
                zorder=2.2,
            )

    ax.set_title(
        "Close with atr_swing_label (+/-1 markers) & atr_swing_flip (vlines)",
        fontsize=14,
    )
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=11, ncol=3)
    ax.set_xlabel("Date")

    dates = pd.Series(date_index)
    day_start = dates.dt.normalize().ne(dates.dt.normalize().shift())
    tick_positions = pos[day_start.to_numpy()]
    tick_labels = dates[day_start].dt.strftime("%Y-%m-%d").to_list()
    if len(tick_positions) > 25:
        step = int(np.ceil(len(tick_positions) / 25))
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    # Light vertical lines for each new day to improve readability
    for x in tick_positions:
        ax.axvline(
            x, color="#d0d0d000", linestyle="--", linewidth=1, alpha=0.7, zorder=0.5
        )

    plt.tight_layout()
    plt.suptitle(
        "Close Price with ATR Swing Label & Flip Points - Last Year",
        fontsize=17,
        y=1.02,
    )
    plt.subplots_adjust(top=0.93)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()


def main(
    ticker: str = "$SPY",
    use_cached: bool = True,
    save_processed: bool = True,
    save_plot_path: str | Path | None = None,
) -> None:
    """
    Build the full feature/label matrix once for the chosen ticker, cache it, and reuse it for plotting.
    Set use_cached=False to force a recompute, or save_processed=False to skip writing.
    Ticker defaults to $SPY; caches and plot outputs are separated per ticker.
    """
    clean_ticker = normalize_ticker(ticker)
    cache_path = get_processed_feature_path(clean_ticker)
    df = load_cached_features(cache_path) if use_cached else None

    if df is None:
        df = load_ticker_parquet(clean_ticker)
        print(df.head())
        print(df.info())

        # Add ALL pandas_ta indicators except statistics & performance
        df = add_all_pandasta_indicators(df, verbose=True)
        print(df.head())
        # Drop NaNs introduced by indicators
        # 1) Drop columns that are completely NaN
        df = df.dropna(axis=1, how="all")
        # Add date features
        df = add_all_custom_indicators(df)
        df = add_date_features(df)

        if save_processed:
            df.to_parquet(cache_path, index=True)
            print(f"Saved processed features + labels to {cache_path}")
    # Filter to keep only the last year of data
    if isinstance(df.index, pd.DatetimeIndex):
        last_date = df.index[-1]
        one_year_ago = last_date - pd.DateOffset(years=1)
        df = df[df.index >= one_year_ago]

    df = add_all_labels(df)

    if save_plot_path is None:
        save_plot_path = get_default_plot_path(clean_ticker)
    plot_atr_swing_signals(df, save_path=str(save_plot_path))


"""    #drop any rows that have NA in these columns
    df = df.dropna(subset=["atr_swing_label", "close", "open", "high", "low", "volume"])

    # --- NEW: binary labels for two-model swing detector ---
    df["long_swing_label"] = (df["atr_swing_label"] == 1.0).astype(np.int64)
    df["short_swing_label"] = (df["atr_swing_label"] == -1.0).astype(np.int64)

    labels = [
        "atr_swing_label",
        "long_swing_label",
        "short_swing_label",
        "atr_entry_price",
        "atr_exit_price",
        "atr_holding_bars",
        "atr_realized_return",
    ]
    # Features = everything except labels
    feature_cols = [c for c in df.columns if c not in labels]

    
    # Start with only feature columns so labels never leak back in
    feature_df = df[feature_cols]

    X = feature_df.to_numpy(dtype=np.float32)

    y_long = df["long_swing_label"].to_numpy(dtype=np.int64)
    y_short = df["short_swing_label"].to_numpy(dtype=np.int64)

    close = df["close"].to_numpy(dtype=float)

    # Save outputs to a dedicated directory under the project path
    output_dir = os.path.join(global_file_path, "Data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "close_spy_daily.npy"), close)
    np.save(os.path.join(output_dir, "X_spy_daily.npy"), X)
    np.save(os.path.join(output_dir, "y_spy_daily_long.npy"), y_long)
    np.save(os.path.join(output_dir, "y_spy_daily_short.npy"), y_short)
    
    X_df = feature_df.astype(np.float32)
    y_long_df = df[["long_swing_label"]].astype("int64")
    y_short_df = df[["short_swing_label"]].astype("int64")
    close_df = df[["close"]].astype(float)

    X_df.to_parquet(os.path.join(output_dir, "X_spy_daily.parquet"), index=False)
    y_long_df.to_parquet(os.path.join(output_dir, "y_spy_daily_long.parquet"), index=False)
    y_short_df.to_parquet(os.path.join(output_dir, "y_spy_daily_short.parquet"), index=False)
    close_df.to_parquet(os.path.join(output_dir, "close_spy_daily.parquet"), index=False)
    labels_df = df[labels]
    labels_df.to_parquet(os.path.join(output_dir, "labels_spy_daily.parquet"), index=False)
    # Optional: keep feature names for reference
    with open(os.path.join(output_dir, "features_spy_daily.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print("X shape:", X.shape)
    print("y_long shape:", y_long.shape)
    print("y_short shape:", y_short.shape)
    
    print("describe:")
    print(df.describe().T)
    print("corr:")
    print(df.corr().abs())
    print("high_corr:")
    corr = feature_df.corr().abs()
    high_corr = corr[corr > 0.98]  # overly similar features
    print(high_corr)
    print("atr_swing_label value counts:")
    print(df["atr_swing_label"].value_counts())


    corr = feature_df.corr().abs()

    # Upper triangle mask
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    for col in ['high','low','open','atr','tmo_main','tmo_signal','pivot_up','pivot_down']:
        print(f"\nHighly correlated with {col}:")
        print(
            corr[col][corr[col] > 0.999].sort_values(ascending=False)
        )

    core_keep = ['high', 'low', 'open', 'close', 'volume']

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > 0.999) and col not in core_keep
    ]


    print("Dropping redundant features:", len(to_drop))
    print(to_drop)

    feature_df_reduced = feature_df.drop(columns=to_drop)
    print("Original feature shape:", feature_df.shape)
    print("Reduced feature shape:", feature_df_reduced.shape)
    
    constant_cols = [c for c in feature_df_reduced.columns if feature_df_reduced[c].nunique() <= 1]
    print("constant cols:")
    print(constant_cols)
    
    feature_df_final = feature_df_reduced.drop(columns=constant_cols)
    
    X = feature_df_final.to_numpy(dtype=np.float32)
    np.save(os.path.join(output_dir, "X_spy_daily.npy"), X)
    
    X_df = feature_df_final.astype(np.float32)
    X_df.to_parquet(os.path.join(output_dir, "X_spy_daily.parquet"), index=False)
    """

if __name__ == "__main__":
    main()
