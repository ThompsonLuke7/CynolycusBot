from pathlib import Path

import numpy as np
import pandas as pd
from Data.load_data import (
    get_processed_feature_path,
    get_ticker_data_dir,
    get_ticker_processed_features_dir,
    resolve_intraday_parquet_path,
    load_cached_features,
    load_ticker_parquet,
)
from Data.plots.all_labels_plot import plot_all_labels
from Data.plots.swing_state_machine_plot import plot_swing_state_machine_signals
from Data.plots.leg_segmentation_plot import plot_leg_segmentation_signals
from Data.plots.atr_swing_plot import get_default_plot_path, plot_atr_swing_signals
from Data.retrieve_data import normalize_ticker
from Features.feature_sets.feature_constants import (
    LEAKY_FEATURE_COLUMNS,
    SWING_LABEL_COLUMNS,
)
from Features.feature_scaling import (
    apply_scaler_from_stats,
    normalize_continuous_features,
    save_normalization_stats,
)
from Features.feature_sets.custom_indicators import (
    add_atr_swing_state_features,
    add_fractal_pivots,
    add_rsilg_fe_gauss,
    add_tmo,
    add_vmd_return_features,
)
from Features.label_generations import add_all_labels, add_all_labels_on_timeframe
from Features.multi_timeframe_features import (
    add_multi_timeframe_features,
    ensure_time_index,
)
from Features.feature_sets.pandas_ta_indicators import add_all_pandasta_indicators

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_BASE_DIR = PROCESSED_DIR / "base"
PROCESSED_SPLIT_DIR = PROCESSED_DIR / "splits"
PROCESSED_STATS_DIR = PROCESSED_DIR / "stats"


def _base_feature_name(name: str) -> str:
    return name.split("__", 1)[0]


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


def add_binary_swing_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=["atr_swing_label", "close", "open", "high", "low", "volume"]
    ).copy()
    df["long_swing_label"] = (df["atr_swing_label"] == 1.0).astype(np.int64)
    df["short_swing_label"] = (df["atr_swing_label"] == -1.0).astype(np.int64)
    return df


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        c
        for c in df.columns
        if _base_feature_name(c) not in SWING_LABEL_COLUMNS
        and _base_feature_name(c) not in LEAKY_FEATURE_COLUMNS
    ]
    feature_df = df[feature_cols]
    return feature_df, feature_cols


def run_feature_diagnostics(df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    print("describe:")
    print(df.describe().T)
    print("corr:")
    corr_all = df.corr().abs()
    print(corr_all)

    corr_features = feature_df.corr().abs()
    print("high_corr:")
    high_corr = corr_features[corr_features > 0.98]  # overly similar features
    print(high_corr)
    print("atr_swing_label value counts:")
    print(df["atr_swing_label"].value_counts())

    return corr_features


def drop_correlated_and_constant_features(
    feature_df: pd.DataFrame,
    corr_features: pd.DataFrame,
    report_path: Path | None = None,
) -> pd.DataFrame:
    upper = corr_features.where(np.triu(np.ones(corr_features.shape), k=1).astype(bool))

    corr_threshold = 0.999
    to_drop = [
        col
        for col in upper.columns
        if col and (upper[col] > corr_threshold).any()
    ]

    report_path = report_path or (PROCESSED_STATS_DIR / "dropped_correlated_features.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        f"correlation_threshold={corr_threshold}",
        f"to_drop_count={len(to_drop)}",
        "",
        "to_drop:",
        *to_drop,
        "",
        "correlated_to:",
    ]
    for col in to_drop:
        matches = upper[col][upper[col] > corr_threshold].sort_values(ascending=False)
        if matches.empty:
            report_lines.append(f"{col}\t(no matches)")
            continue
        match_str = ", ".join(f"{name} ({value:.6f})" for name, value in matches.items())
        report_lines.append(f"{col}\t{match_str}")
    report_path.write_text("\n".join(report_lines) + "\n")

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

    return feature_df_reduced.drop(columns=constant_cols)


def clean_nan_inf_entries(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Count NaN/Inf entries, convert Inf to NaN, and zero-fill NaNs.
    """
    nan_count = int(df.isna().sum().sum())
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df.copy(), {"nan_count": nan_count, "inf_count": 0}

    numeric_frame = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric_array = numeric_frame.to_numpy(dtype="float64", na_value=np.nan)
    inf_count = int(np.isinf(numeric_array).sum())
    cleaned = df.copy()
    cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)
    cleaned[numeric_cols] = cleaned[numeric_cols].ffill().fillna(0)
    return cleaned, {"nan_count": nan_count, "inf_count": inf_count}


def drop_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_base = {"open", "high", "low", "close", "volume"}
    drop_cols = [
        col for col in df.columns if col.split("__", 1)[0] in drop_base
    ]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def save_feature_outputs(
    output_dir: Path,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    df: pd.DataFrame,
    *,
    prefix: str = "spy_daily",
    label_mode: str = "swing",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    X = feature_df.to_numpy(dtype=np.float32)
    if label_mode == "swing":
        long_col = "long_swing_label"
        short_col = "short_swing_label"
        label_suffix = "swing"
    elif label_mode == "leg":
        long_col = "leg_up_label"
        short_col = "leg_down_label"
        label_suffix = "leg"
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    y_long = df[long_col].to_numpy(dtype=np.int64)
    y_short = df[short_col].to_numpy(dtype=np.int64)
    close = df["close"].to_numpy(dtype=float)

    np.save(output_dir / f"close_{prefix}.npy", close)
    np.save(output_dir / f"X_{prefix}.npy", X)
    np.save(output_dir / f"y_{prefix}_{label_suffix}_long.npy", y_long)
    np.save(output_dir / f"y_{prefix}_{label_suffix}_short.npy", y_short)

    X_df = feature_df.astype(np.float32)
    y_long_df = df[[long_col]].astype("int64")
    y_short_df = df[[short_col]].astype("int64")
    close_df = df[["close"]].astype(float)

    X_df.to_parquet(output_dir / f"X_{prefix}.parquet", index=False)
    y_long_df.to_parquet(
        output_dir / f"y_{prefix}_{label_suffix}_long.parquet", index=False
    )
    y_short_df.to_parquet(
        output_dir / f"y_{prefix}_{label_suffix}_short.parquet", index=False
    )
    close_df.to_parquet(output_dir / f"close_{prefix}.parquet", index=False)
    label_cols = [c for c in SWING_LABEL_COLUMNS if c in df.columns]
    labels_df = df[label_cols]
    labels_df.to_parquet(output_dir / f"labels_{prefix}.parquet", index=False)
    with open(output_dir / f"features_{prefix}.txt", "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print("X shape:", X.shape)
    print("y_long shape:", y_long.shape)
    print("y_short shape:", y_short.shape)


def main(
    ticker: str = "$SPY",
    use_cached: bool = True,
    save_processed: bool = True,
    save_plot_path: str | Path | None = None,
    label_mode: str = "leg",
    multi_timeframes: dict[str, str] | None = None,
    label_timeframe: str | None = "15T",
    tz: str | None = "America/New_York",
) -> None:
    """
    Build the full feature/label matrix once for the chosen ticker, cache it, and reuse it for plotting.
    Set use_cached=False to force a recompute, or save_processed=False to skip writing.
    Ticker defaults to $SPY; caches and plot outputs are separated per ticker.
    """
    clean_ticker = normalize_ticker(ticker)
    ticker_slug = clean_ticker.lower()
    raw_parquet_path = resolve_intraday_parquet_path(clean_ticker)
    prefix = (
        raw_parquet_path.stem
        if raw_parquet_path is not None
        else f"{ticker_slug}_daily"
    )
    ticker_data_dir = get_ticker_data_dir(clean_ticker)
    cache_path = get_processed_feature_path(clean_ticker, prefix=prefix)
    df = load_cached_features(cache_path) if use_cached else None

    if df is None:
        df = load_ticker_parquet(clean_ticker, parquet_path=raw_parquet_path)
        if multi_timeframes:
            df = ensure_time_index(df, tz=tz)
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

        if multi_timeframes:
            tf_features = add_multi_timeframe_features(
                df,
                timeframes=multi_timeframes,
                tz=tz,
                include_custom=True,
                verbose=False,
            )
            df = pd.concat([df, tf_features], axis=1)

        if save_processed:
            df.to_parquet(cache_path, index=True)
            print(f"Saved processed features + labels to {cache_path}")
    # Filter to keep only the last year of data
    if isinstance(df.index, pd.DatetimeIndex):
        last_date = df.index[-1]
        one_year_ago = last_date - pd.DateOffset(years=1)
        df = df[df.index >= one_year_ago]

    if label_timeframe:
        df = add_all_labels_on_timeframe(
            df,
            label_timeframe=label_timeframe,
            tz=tz,
            swing_state_machine_kwargs={},
        )
    else:
        df = add_all_labels(df, swing_state_machine_kwargs={})

    if save_plot_path is None:
        save_plot_path = get_default_plot_path(clean_ticker, ticker_data_dir)
    # plot_all_labels(
    #    df, save_path=str(save_plot_path).replace("atr_swing", "all_labels")
    # )
    # plot_swing_state_machine_signals(
    #    df, save_path=str(save_plot_path).replace("atr_swing", "swing_state_machine")
    # )
    #  plot_leg_segmentation_signals(
    #    df, save_path=str(save_plot_path).replace("atr_swing", "leg_segmentation")
    # )
    plot_atr_swing_signals(df, save_path=str(save_plot_path))

    df = add_binary_swing_labels(df)
    feature_df, feature_cols = build_feature_frame(df)

    corr_features = run_feature_diagnostics(df, feature_df)
    feature_df_final = drop_correlated_and_constant_features(feature_df, corr_features)

    output_dir = get_ticker_processed_features_dir(clean_ticker)
    final_feature_cols = list(feature_df_final.columns)
    save_feature_outputs(
        output_dir,
        feature_df_final,
        final_feature_cols,
        df,
        prefix=prefix,
        label_mode=label_mode,
    )


if __name__ == "__main__":
    main()
