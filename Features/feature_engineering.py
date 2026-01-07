import json
from pathlib import Path

import numpy as np
import pandas as pd
from Data.load_data import (
    get_processed_feature_path,
    get_ticker_data_dir,
    get_ticker_processed_base_dir,
    resolve_intraday_parquet_path,
    load_cached_features,
    load_ticker_parquet,
)
from Data.plots.all_labels_plot import plot_all_labels
from Data.plots.swing_state_machine_plot import plot_swing_state_machine_signals
from Data.plots.leg_segmentation_plot import plot_leg_segmentation_signals
from Data.plots.atr_swing_plot import get_default_plot_path, plot_atr_swing_signals
from Data.retrieve_data import normalize_ticker
from Features.custom_indicators import (
    add_atr_swing_state_features,
    add_fractal_pivots,
    add_rsilg_fe_gauss,
    add_tmo,
    add_vmd_return_features,
)
from Features.label_generations import add_all_labels
from Features.pandas_ta_indicators import add_all_pandasta_indicators

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_BASE_DIR = PROCESSED_DIR / "base"
PROCESSED_SPLIT_DIR = PROCESSED_DIR / "splits"
PROCESSED_STATS_DIR = PROCESSED_DIR / "stats"
LEAKY_FEATURE_COLUMNS = {
    # Pivot detections are inherently lookahead-based; keep them out of features
    "pivot_up",
    "pivot_down",
    "super_pivot_up",
    "super_pivot_down",
}
SWING_LABEL_COLUMNS = [
    "atr_swing_label",
    "long_swing_label",
    "short_swing_label",
    "atr_entry_price",
    "atr_exit_price",
    "atr_holding_bars",
    "atr_realized_return",
    # Continuation labels
    "atr_cont_label",
    "long_cont_label",
    "short_cont_label",
    "atr_cont_label_entry_price",
    "atr_cont_label_exit_price",
    "atr_cont_label_holding_bars",
    "atr_cont_label_realized_return",
    # Leg segmentation labels
    "atr_leg_label",
    "leg_up_label",
    "leg_down_label",
    # State machine gates (label-like, avoid leakage)
    "p_long_state_gate",
    "p_short_state_gate",
    "p_long_pending",
    "p_short_pending",
    "p_flat_state_gate",
]
SCALE_FEATURE_COLUMNS = {
    # Raw price & volume (only if kept as-is)
    "open",
    "high",
    "low",
    "close",
    "volume",
    # Returns / log / volatility
    "LOGRET_1",
    "TRUERANGE_1",
    "NATR_14",
    "STDEV_30",
    "VAR_30",
    "UI_14",
    "KURT_30",
    "SKEW_30",
    "rel_vol_z",
    # Trend, momentum, oscillators
    "ADX_14",
    "ADXR_14_2",
    "AO_5_34",
    "APO_12_26",
    "AROOND_14",
    "AROONU_14",
    "AROONOSC_14",
    "BIAS_SMA_26",
    "BOP",
    "CCI_14_0.015",
    "CHOP_14_1_100.0",
    "CMO_14",
    "COPC_11_14_10",
    "CRSI_3_2_100",
    "CTI_12",
    "DPO_20",
    "ER_10",
    "FISHERT_9_1",
    "FISHERTs_9_1",
    "INERTIA_20_14",
    "K_9_3",
    "D_9_3",
    "J_9_3",
    "KST_10_15_20_30_10_10_10_15",
    "KSTs_9",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "MASSI_9_25",
    "MFI_14",
    "MOM_10",
    "PGO_14",
    "PPOh_12_26_9",
    "PPOs_12_26_9",
    "PSL_12",
    "QQE_14_5_4.236",
    "QQE_14_5_4.236_RSIMA",
    "QS_10",
    "REFLEX_20_20_0.04",
    "RSX_14",
    "RVGI_14_4",
    "RVGIs_14_4",
    "RVI_14",
    "RWIh_14",
    "RWIl_14",
    "SMI_5_20_5_1.0",
    "SMIs_5_20_5_1.0",
    "SMIo_5_20_5_1.0",
    "STC_10_12_26_0.5",
    "STCstoch_10_12_26_0.5",
    "STOCHk_14_3_3",
    "STOCHd_14_3_3",
    "STOCHh_14_3_3",
    "STOCHFk_14_3",
    "STOCHRSIk_14_14_3_3",
    "STOCHRSId_14_14_3_3",
    "TRIX_30_9",
    "TRIXs_30_9",
    "TSI_13_25_13",
    "TSIs_13_25_13",
    "UO_7_14_28",
    "VHF_28",
    "VIDYA_14",
    "VTXP_14",
    "VTXM_14",
    # Bands / channels / distances
    "BBL_5_2.0_2.0",
    "BBU_5_2.0_2.0",
    "BBB_5_2.0_2.0",
    "BBP_5_2.0_2.0",
    "KCLe_20_2",
    "PDIST",
    # Volume / flow
    "AD",
    "ADOSC_3_10",
    "OBV",
    "OBVe_12",
    "AOBV_LR_2",
    "AOBV_SR_2",
    "CMF_20",
    "EFI_13",
    "EOM_14_100000000",
    "KVO_34_55_13",
    "KVOs_34_55_13",
    "NVI_1",
    "PVI",
    "PVIe_255",
    "PVO_12_26_9",
    "PVOh_12_26_9",
    "PVOs_12_26_9",
    "PVOL",
    "PVR",
    "PVT",
    "TSV_18_10",
    "TSVs_18_10",
    "TSVr_18_10",
    # Regression / smoothing / statistical
    "ALPHATl_14_1_50_2",
    "AMATe_LR_8_21_2",
    "AMATe_SR_8_21_2",
    "CHDLREXTl_22_22_14_2.0",
    "CHDLREXTs_22_22_14_2.0",
    "CHDLREXTd_22_22_14_2.0",
    "ENTP_10",
    "FAMA_0.5_0.05",
    "HT_TL",
    "ISA_9",
    "ISB_26",
    "IKS_26",
    "ICS_26",
    "MAD_30",
    "MEDIAN_30",
    "TRENDFLEX_20_20_0.04",
    "TOS_STDEVALL_LR",
    # ATR swing distances
    "atr_swing_dist_from_extreme_atr",
    "atr_swing_bars_since_flip",
    # VMD features
    "vmd_r_mode1_last",
    "vmd_r_mode1_energy_20",
    "vmd_r_mode2_last",
    "vmd_r_mode2_energy_20",
    "vmd_r_mode3_last",
    "vmd_r_mode3_energy_20",
    "vmd_r_mode4_last",
    "vmd_r_mode4_energy_20",
}


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
        if c not in SWING_LABEL_COLUMNS and c not in LEAKY_FEATURE_COLUMNS
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

    for col in [
        "high",
        "low",
        "open",
        "atr",
    ]:
        print(f"\nHighly correlated with {col}:")
        print(
            corr_features[col][corr_features[col] > 0.999].sort_values(ascending=False)
        )

    return corr_features


def drop_correlated_and_constant_features(
    feature_df: pd.DataFrame, corr_features: pd.DataFrame
) -> pd.DataFrame:
    upper = corr_features.where(np.triu(np.ones(corr_features.shape), k=1).astype(bool))

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

    return feature_df_reduced.drop(columns=constant_cols)


def normalize_continuous_features(
    feature_df: pd.DataFrame, scale_cols: set[str] | None = None
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Standardize ONLY the vetted continuous magnitude features.
    Any column not explicitly listed in scale_cols is left untouched.
    """
    target_cols = set(scale_cols) if scale_cols is not None else SCALE_FEATURE_COLUMNS
    norm_df = feature_df.copy()
    stats: dict[str, dict[str, float]] = {}

    for col in norm_df.columns:
        if col not in target_cols:
            continue
        if not pd.api.types.is_numeric_dtype(norm_df[col]):
            continue
        mean = float(norm_df[col].mean())
        std = float(norm_df[col].std(ddof=0))
        if std == 0.0 or np.isnan(std):
            continue
        norm_df[col] = (norm_df[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}

    return norm_df, stats


def apply_scaler_from_stats(
    feature_df: pd.DataFrame, stats: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """
    Apply precomputed mean/std stats to a feature frame without refitting.
    """
    norm_df = feature_df.copy()
    for col, vals in stats.items():
        if col not in norm_df.columns:
            continue
        std = vals.get("std")
        mean = vals.get("mean")
        if std is None or std == 0.0 or np.isnan(std) or mean is None:
            continue
        norm_df[col] = (norm_df[col] - mean) / std
    return norm_df


def save_normalization_stats(
    output_dir: Path,
    stats: dict[str, dict[str, float]],
    filename: str = "norm_stats_spy_daily.json",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, "w") as f:
        json.dump(stats, f, indent=2)


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

    output_dir = get_ticker_processed_base_dir(clean_ticker)
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
