from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from Data.load_data import get_ticker_processed_base_dir, load_ticker_parquet
from Data.retrieve_data import normalize_ticker
from Features.custom_indicators import (
    add_atr_swing_state_features,
    add_fractal_pivots,
    add_rsilg_fe_gauss,
    add_tmo,
    add_vmd_return_features,
    add_s_r_features,
)
from Features.feature_constants import SWING_LABEL_COLUMNS
from Features.feature_engineering import (
    add_binary_swing_labels,
    add_date_features,
    build_feature_frame,
    clean_nan_inf_entries,
    drop_correlated_and_constant_features,
    run_feature_diagnostics,
)
from Features.label_generations import add_all_labels
from Features.multi_timeframe_features import ensure_time_index, resample_ohlcv
from Features.pandas_ta_indicators import add_all_pandasta_indicators

DEFAULT_FEATURE_TIMEFRAMES: dict[str, str] = {
    "30m": "30T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}
DATASETS_DIRNAME = "datasets"
PIVOT_LABEL_COLUMNS = (
    "pivot_up",
    "pivot_down",
    "super_pivot_up",
    "super_pivot_down",
)


def _normalize_timeframe_label(label_timeframe: str) -> str:
    tf = label_timeframe.strip().lower()
    if tf.endswith("min"):
        value = tf.replace("min", "") or "1"
        return f"{value}min"
    if tf.endswith("t"):
        value = tf[:-1] or "1"
        return f"{value}min"
    if tf.endswith("hour"):
        value = tf.replace("hour", "") or "1"
        return f"{value}h"
    if tf.endswith("h"):
        value = tf[:-1] or "1"
        return f"{value}h"
    if tf.endswith("day"):
        value = tf.replace("day", "") or "1"
        return f"{value}d"
    if tf.endswith("d"):
        value = tf[:-1] or "1"
        return f"{value}d"
    return tf


def _collect_label_columns(df: pd.DataFrame) -> list[str]:
    label_cols: list[str] = []
    for col in PIVOT_LABEL_COLUMNS:
        if col in df.columns:
            label_cols.append(col)
    for col in SWING_LABEL_COLUMNS:
        if col in df.columns and col not in label_cols:
            label_cols.append(col)
    return label_cols


def _add_feature_set(
    df: pd.DataFrame,
    *,
    include_custom: bool,
    include_date_features: bool,
    verbose: bool,
) -> pd.DataFrame:
    df = add_all_pandasta_indicators(df, verbose=verbose)
    df = df.dropna(axis=1, how="all")
    if include_custom:
        df = add_tmo(df)
        df = add_rsilg_fe_gauss(df)
        df = add_atr_swing_state_features(df)
        df = add_vmd_return_features(df)
        df = add_s_r_features(df)
    if include_date_features:
        df = add_date_features(df)
    return df


def _align_htf_features(
    df: pd.DataFrame,
    *,
    base_index: pd.DatetimeIndex,
    suffix: str,
    shift_bars: int,
) -> pd.DataFrame:
    aligned = df.add_suffix(f"__{suffix}")
    if shift_bars:
        aligned = aligned.shift(shift_bars)
    aligned = aligned.reindex(base_index, method="ffill")
    return aligned.dropna(axis=1, how="all")


def build_training_matrix(
    parquet_path: str | Path,
    *,
    ticker: str = "$SPY",
    tz: str | None = "America/New_York",
    label_timeframe: str = "15T",
    feature_timeframes: Mapping[str, str] | None = None,
    pivot_kwargs: dict | None = None,
    label_kwargs: dict | None = None,
    include_custom: bool = True,
    include_date_features: bool = True,
    include_htf_date_features: bool = False,
    verbose: bool = False,
    shift_htf_bars: int = 0,
    resample_label: str = "left",
    resample_closed: str = "left",
) -> pd.DataFrame:
    """
    Build a 15m training matrix with labels on 15m only and HTF context features.

    - Resamples 1m data to 15m for labels/features.
    - Computes labels (pivot/continuation/leg) on 15m only.
    - Computes features on 15m + 30m/1h/4h/1d (aligned to 15m).
    - Does not forward-fill labels to 1m.
    """
    feature_timeframes = feature_timeframes or DEFAULT_FEATURE_TIMEFRAMES
    pivot_kwargs = pivot_kwargs or {}
    label_kwargs = label_kwargs or {}

    df_1m = load_ticker_parquet(ticker, parquet_path=parquet_path)
    df_1m = ensure_time_index(df_1m, tz=tz)

    df_15m = resample_ohlcv(
        df_1m, label_timeframe, label=resample_label, closed=resample_closed
    )
    df_15m = _add_feature_set(
        df_15m,
        include_custom=include_custom,
        include_date_features=include_date_features,
        verbose=verbose,
    )
    df_15m = add_fractal_pivots(df_15m, **pivot_kwargs)
    df_15m = add_all_labels(df_15m, **label_kwargs)

    frames = [df_15m]
    for tf_label, tf_rule in feature_timeframes.items():
        tf_df = resample_ohlcv(
            df_1m, tf_rule, label=resample_label, closed=resample_closed
        )
        if tf_df.empty:
            continue
        tf_df = _add_feature_set(
            tf_df,
            include_custom=include_custom,
            include_date_features=include_htf_date_features,
            verbose=verbose,
        )
        aligned = _align_htf_features(
            tf_df,
            base_index=df_15m.index,
            suffix=tf_label,
            shift_bars=shift_htf_bars,
        )
        frames.append(aligned)

    out = pd.concat(frames, axis=1)
    out.attrs["source_parquet"] = str(parquet_path)
    out.attrs["ticker"] = ticker
    out.attrs["label_timeframe"] = label_timeframe
    return out


def clean_training_matrix(
    df: pd.DataFrame,
    *,
    run_diagnostics: bool = False,
    drop_correlated: bool = True,
    save_outputs: bool = False,
    output_dir: Path | None = None,
    ticker: str = "$SPY",
    dataset_name: str | None = None,
    label_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Clean a training matrix using the same steps as feature_engineering.main.
    Returns the cleaned matrix, reduced feature frame, and feature columns.
    When save_outputs is True, writes X.parquet and y.parquet under
    processed/base/datasets/<dataset_name>/.
    """
    source_attrs = dict(df.attrs)
    cleaned = df.copy()
    if "atr_swing_label" in cleaned.columns:
        cleaned = add_binary_swing_labels(cleaned)

    feature_df, feature_cols = build_feature_frame(cleaned)

    if drop_correlated:
        if run_diagnostics and "atr_swing_label" in cleaned.columns:
            corr_features = run_feature_diagnostics(cleaned, feature_df)
        else:
            corr_features = feature_df.corr().abs()

        feature_df = drop_correlated_and_constant_features(feature_df, corr_features)
        kept_cols = list(feature_df.columns)
        drop_cols = [c for c in feature_cols if c not in kept_cols]
        if drop_cols:
            cleaned = cleaned.drop(columns=drop_cols)
        feature_cols = kept_cols
        cleaned, _ = clean_nan_inf_entries(cleaned)
        feature_df = cleaned[feature_cols]

    if save_outputs:
        clean_ticker = normalize_ticker(ticker)
        resolved_output_dir = (
            output_dir
            if output_dir is not None
            else get_ticker_processed_base_dir(clean_ticker)
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        datasets_root = resolved_output_dir / DATASETS_DIRNAME
        datasets_root.mkdir(parents=True, exist_ok=True)

        if dataset_name is None:
            label_tf = source_attrs.get("label_timeframe")
            if isinstance(label_tf, str) and label_tf:
                dataset_name = _normalize_timeframe_label(label_tf)
            else:
                dataset_name = "dataset"

        dataset_dir = datasets_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        selected_label_cols = (
            label_cols if label_cols is not None else _collect_label_columns(cleaned)
        )
        labels_df = (
            cleaned[selected_label_cols].copy()
            if selected_label_cols
            else pd.DataFrame(index=cleaned.index)
        )

        feature_df.astype("float32").to_parquet(
            dataset_dir / "X.parquet", index=False
        )
        labels_df.to_parquet(dataset_dir / "y.parquet", index=False)
        with open(dataset_dir / "features.txt", "w") as f:
            for col in feature_cols:
                f.write(col + "\n")

        features_dir = resolved_output_dir / "features"
        labels_dir = resolved_output_dir / "labels"
        features_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        with open(features_dir / f"{dataset_name}_features.txt", "w") as f:
            for col in feature_cols:
                f.write(col + "\n")
        labels_df.to_parquet(labels_dir / f"{dataset_name}_labels.parquet", index=False)

    return cleaned, feature_df, feature_cols


def main() -> None:
    raise SystemExit(
        "Use build_training_matrix(parquet_path=...) from another script."
    )


if __name__ == "__main__":
    main()
