import pandas as pd
import os
from Features.pandas_ta_indicators import add_all_pandasta_indicators
from Features.label_generations import add_all_labels, plot_zig_zag
from Features.custom_indicators import add_tmo, add_rsilg_fe_gauss, add_fractal_pivots, add_atr_swing_state_features, add_vmd_return_features
import matplotlib.pyplot as plt

global_file_path = "C:/Users/luket/CynolycusBot"

def load_spy_csv() -> pd.DataFrame:
    path = os.path.join(global_file_path, "Data", "spy_data.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=[0], date_format="%Y-%m-%d")
    df.index = pd.to_datetime(df.index, errors="coerce", format="%Y-%m-%d")
    # 2) Drop rows where index is NaT (these are 'Ticker', 'Date', etc.)
    df = df[df.index.notna()]
    # 1) Keep only the columns we care about initially (drop any 'Symbol', etc.)
    keep_cols = [c for c in df.columns if c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df = df[keep_cols]

    # 2) Rename to lowercase OHLCV for pandas_ta
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # 3) Force OHLCV columns to numeric
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df.index.dayofweek        # 0 = Monday … 4 = Friday
    df["day_of_month"] = df.index.day            # sometimes helps
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


def get_processed_feature_path(filename: str = "spy_features_with_labels.parquet") -> str:
    """
    Build the path where the processed feature/label matrix is stored.
    Ensures the directory exists so we can write the cache.
    """
    output_dir = os.path.join(global_file_path, "Data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def load_cached_features(cache_path: str):
    """
    Load a previously cached feature matrix if present, otherwise return None.
    """
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index, errors="coerce")
        cached = cached[cached.index.notna()]
        print(f"Loaded cached features + labels from {cache_path}")
        return cached
    return None


def main(use_cached: bool = True, save_processed: bool = True) -> None:
    """
    Build the full feature/label matrix once, cache it, and reuse it for plotting.
    Set use_cached=False to force a recompute, or save_processed=False to skip writing.
    """
    cache_path = get_processed_feature_path()
    df = load_cached_features(cache_path) if use_cached else None

    if df is None:
        df = load_spy_csv()
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

        df = add_all_labels(df)

        if save_processed:
            df.to_parquet(cache_path, index=True)
            print(f"Saved processed features + labels to {cache_path}")
    # Filter to keep only the last year of data
    if isinstance(df.index, pd.DatetimeIndex):
        last_date = df.index[-1]
        one_year_ago = last_date - pd.DateOffset(years=1)
        df = df[df.index >= one_year_ago]

    # Plot close with atr_swing_label (+1/-1 markers) and atr_swing_flip (vertical lines)
    fig, ax = plt.subplots(figsize=(18, 6))
    close_y = df["close"].to_numpy()
    ax.plot(df.index, close_y, label="Close", color="black", linewidth=1.6, zorder=1)

    # Plot atr_swing_label as ±1 markers on close
    if "atr_swing_label" in df.columns:
        mask_pos = (df["atr_swing_label"].fillna(0) == 1).to_numpy()
        mask_neg = (df["atr_swing_label"].fillna(0) == -1).to_numpy()
        pos_idx = df.index[mask_pos]
        neg_idx = df.index[mask_neg]
        if len(pos_idx) > 0:
            ax.scatter(
                pos_idx, close_y[mask_pos], color="#1976D2", marker="^", s=42,
                label="atr_swing_label = +1", alpha=0.96, zorder=2
            )
        if len(neg_idx) > 0:
            ax.scatter(
                neg_idx, close_y[mask_neg], color="#E53935", marker="v", s=42,
                label="atr_swing_label = -1", alpha=0.96, zorder=2
            )

    # Plot pivot_down markers in green
    if "pivot_down" in df.columns:
        mask_pivot_down = (df["pivot_down"].fillna(0).astype(int) == 1).to_numpy()
        pivot_idx = df.index[mask_pivot_down]
        if len(pivot_idx) > 0:
            ax.scatter(
                pivot_idx, close_y[mask_pivot_down], color="#2E7D32", marker="v", s=52,
                label="pivot_down", alpha=0.95, zorder=2.2
            )
            
    if "pivot_up" in df.columns:
        mask_pivot_up = (df["pivot_up"].fillna(0).astype(int) == 1).to_numpy()
        pivot_idx = df.index[mask_pivot_up]
        if len(pivot_idx) > 0:
            ax.scatter(
                pivot_idx, close_y[mask_pivot_up], color="#1E0D32", marker="^", s=52,
                label="pivot_up", alpha=0.95, zorder=2.2
            )

    # Plot atr_swing_flip as vertical lines on the close price chart
    if "atr_swing_flip" in df.columns:
        flip_idx = df.index[df["atr_swing_flip"].fillna(0).astype(int) == 1]
        for flip_time in flip_idx:
            ax.axvline(flip_time, color="#43A047", alpha=0.55, linestyle="--", linewidth=1.3, label="atr_swing_flip" if flip_time==flip_idx[0] else "")

    ax.set_title("Close with atr_swing_label (±1 markers) & atr_swing_flip (vlines)", fontsize=14)
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=11, ncol=3)
    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.suptitle("Close Price with ATR Swing Label & Flip Points - Last Year", fontsize=17, y=1.02)
    plt.subplots_adjust(top=0.93)
    plt.show()

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
