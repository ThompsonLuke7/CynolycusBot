import os
from pathlib import Path

import numpy as np
import pandas as pd
from Data.load_data import (
    get_processed_feature_path,
    load_cached_features,
    load_ticker_parquet,
)
from Data.plots.atr_swing_plot import get_default_plot_path, plot_atr_swing_signals
from Data.retrieve_data import normalize_ticker
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
        save_plot_path = get_default_plot_path(clean_ticker, DATA_DIR)
    plot_atr_swing_signals(df, save_path=str(save_plot_path))

    # drop any rows that have NA in these columns
    df = df.dropna(subset=["atr_swing_label", "close", "open", "high", "low", "volume"])

    # --- binary labels for two-model swing detector ---
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
    y_long_df.to_parquet(
        os.path.join(output_dir, "y_spy_daily_long.parquet"), index=False
    )
    y_short_df.to_parquet(
        os.path.join(output_dir, "y_spy_daily_short.parquet"), index=False
    )
    close_df.to_parquet(
        os.path.join(output_dir, "close_spy_daily.parquet"), index=False
    )
    labels_df = df[labels]
    labels_df.to_parquet(
        os.path.join(output_dir, "labels_spy_daily.parquet"), index=False
    )
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
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    for col in [
        "high",
        "low",
        "open",
        "atr",
        "tmo_main",
        "tmo_signal",
        "pivot_up",
        "pivot_down",
    ]:
        print(f"\nHighly correlated with {col}:")
        print(corr[col][corr[col] > 0.999].sort_values(ascending=False))

    core_keep = ["high", "low", "open", "close", "volume"]

    to_drop = [
        col for col in upper.columns if any(upper[col] > 0.999) and col not in core_keep
    ]

    print("Dropping redundant features:", len(to_drop))
    print(to_drop)

    feature_df_reduced = feature_df.drop(columns=to_drop)
    print("Original feature shape:", feature_df.shape)
    print("Reduced feature shape:", feature_df_reduced.shape)

    constant_cols = [
        c for c in feature_df_reduced.columns if feature_df_reduced[c].nunique() <= 1
    ]
    print("constant cols:")
    print(constant_cols)

    feature_df_final = feature_df_reduced.drop(columns=constant_cols)

    X = feature_df_final.to_numpy(dtype=np.float32)
    np.save(os.path.join(output_dir, "X_spy_daily.npy"), X)

    X_df = feature_df_final.astype(np.float32)
    X_df.to_parquet(os.path.join(output_dir, "X_spy_daily.parquet"), index=False)


if __name__ == "__main__":
    main()
