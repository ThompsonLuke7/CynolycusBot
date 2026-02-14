from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from Data.load_data import get_ticker_processed_base_dir, load_ticker_parquet
from Data.retrieve_data import normalize_ticker
from Features.feature_sets.custom_indicators import (
    add_atr_swing_state_features,
    add_fractal_pivots,
    add_rsilg_fe_gauss,
    add_tmo,
    add_vmd_return_features,
    add_s_r_features,
)
from Features.feature_sets.feature_constants import SWING_LABEL_COLUMNS
from Features.feature_engineering import (
    add_binary_swing_labels,
    add_date_features,
    build_feature_frame,
    clean_nan_inf_entries,
    drop_ohlcv_columns,
    drop_correlated_and_constant_features,
    run_feature_diagnostics,
)
from Features.label_generations import add_all_labels
from Features.multi_timeframe_features import ensure_time_index, resample_ohlcv
from Features.feature_sets.pandas_ta_indicators import add_all_pandasta_indicators
from Features.feature_sets.LSTM_features import add_lstm_features

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
VIX_SUITE_COLUMNS = [
    "vix_close",
    "vix_ret_1",
    "vix_ret_4",
    "vix_ret_16",
    "vix_range_pct",
    "vix_atr_pct",
    "vix_trend_ema_8_21",
    "vix_z_20",
    "vix_vol_of_vol_20",
    "ret_1_x_vix",
    "atr_pct_x_vix",
]


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
    model: str = "lstm",
) -> pd.DataFrame:
    model_key = (model or "tree").strip().lower()
    if model_key == "lstm":
        return add_lstm_features(df, include_time_features=include_date_features)
    if model_key != "tree":
        raise ValueError(f"Unsupported model: {model}")

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


def _add_lstm_features_for_tree(
    df: pd.DataFrame,
    *,
    include_time_features: bool,
    tz: str | None,
) -> pd.DataFrame:
    base_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if len(base_cols) < 5:
        return df

    base = df[base_cols].copy()
    time_kwargs = {"tz": tz} if tz is not None else None
    lstm_df = add_lstm_features(
        base,
        include_time_features=include_time_features,
        time_kwargs=time_kwargs,
    )

    new_cols = [c for c in lstm_df.columns if c not in df.columns]
    if not new_cols:
        return df
    return pd.concat([df, lstm_df[new_cols]], axis=1)


def _ensure_vix_suite_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in VIX_SUITE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _load_vix_1m(
    *,
    vix_ticker: str,
    vix_parquet_path: str | Path | None,
    tz: str | None,
) -> pd.DataFrame:
    vix_df = load_ticker_parquet(vix_ticker, parquet_path=vix_parquet_path)
    return ensure_time_index(vix_df, tz=tz)


def _compute_vix_suite(
    *,
    base_df: pd.DataFrame,
    vix_ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(index=base_df.index)

    if vix_ohlcv.empty or "close" not in vix_ohlcv.columns:
        for col in VIX_SUITE_COLUMNS:
            out[col] = np.nan
        return out

    vix_close = pd.to_numeric(vix_ohlcv.get("close"), errors="coerce")
    vix_high = pd.to_numeric(vix_ohlcv.get("high"), errors="coerce")
    vix_low = pd.to_numeric(vix_ohlcv.get("low"), errors="coerce")

    prev_close = vix_close.shift(1)
    tr = pd.concat(
        [
            (vix_high - vix_low).abs(),
            (vix_high - prev_close).abs(),
            (vix_low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    vix_atr = tr.rolling(14, min_periods=1).mean()

    out["vix_close"] = vix_close
    out["vix_ret_1"] = vix_close.pct_change(1)
    out["vix_ret_4"] = vix_close.pct_change(4)
    out["vix_ret_16"] = vix_close.pct_change(16)
    out["vix_range_pct"] = (vix_high - vix_low) / vix_close.replace(0, np.nan)
    out["vix_atr_pct"] = vix_atr / vix_close.replace(0, np.nan)
    out["vix_trend_ema_8_21"] = (
        vix_close.ewm(span=8, adjust=False).mean()
        - vix_close.ewm(span=21, adjust=False).mean()
    ) / vix_close.replace(0, np.nan)
    vix_mean_20 = vix_close.rolling(20, min_periods=20).mean()
    vix_std_20 = vix_close.rolling(20, min_periods=20).std(ddof=0)
    out["vix_z_20"] = (vix_close - vix_mean_20) / vix_std_20.replace(0, np.nan)
    out["vix_vol_of_vol_20"] = out["vix_ret_1"].rolling(20, min_periods=20).std(ddof=0)

    base_ret_1 = pd.to_numeric(base_df.get("ret_1"), errors="coerce")
    base_atr_pct = pd.to_numeric(base_df.get("atr_pct"), errors="coerce")
    out["ret_1_x_vix"] = base_ret_1 * vix_close
    out["atr_pct_x_vix"] = base_atr_pct * vix_close
    return out


def _add_vix_suite_to_frame(
    df: pd.DataFrame,
    *,
    vix_1m: pd.DataFrame | None,
    timeframe_rule: str,
    resample_label: str,
    resample_closed: str,
) -> pd.DataFrame:
    out = df.copy()
    if vix_1m is None or vix_1m.empty:
        return _ensure_vix_suite_cols(out)

    vix_tf = resample_ohlcv(
        vix_1m,
        timeframe_rule,
        label=resample_label,
        closed=resample_closed,
    )
    if vix_tf.empty:
        return _ensure_vix_suite_cols(out)

    vix_suite = _compute_vix_suite(base_df=out, vix_ohlcv=vix_tf)
    vix_suite = vix_suite.reindex(out.index, method="ffill")
    for col in VIX_SUITE_COLUMNS:
        out[col] = vix_suite.get(col)
    return out


def build_feature_matrix(
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
    model: str = "tree",
    shift_htf_bars: int = 1,
    resample_label: str = "left",
    resample_closed: str = "left",
    include_vix_features: bool = True,
    vix_ticker: str = "$VIX",
    vix_parquet_path: str | Path | None = None,
    vix_warn_on_missing: bool = True,
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
    vix_1m: pd.DataFrame | None = None
    if include_vix_features:
        try:
            vix_1m = _load_vix_1m(
                vix_ticker=vix_ticker,
                vix_parquet_path=vix_parquet_path,
                tz=tz,
            )
        except Exception as exc:
            if vix_warn_on_missing:
                print(f"[feature_matrix] VIX suite unavailable: {exc}")
            vix_1m = None

    df_15m = resample_ohlcv(
        df_1m, label_timeframe, label=resample_label, closed=resample_closed
    )
    df_15m = _add_feature_set(
        df_15m,
        include_custom=include_custom,
        include_date_features=include_date_features,
        verbose=verbose,
        model=model,
    )
    model_key = (model or "tree").strip().lower()
    if model_key == "tree":
        df_15m = _add_lstm_features_for_tree(
            df_15m,
            include_time_features=True,
            tz=tz,
        )
    if include_vix_features:
        df_15m = _add_vix_suite_to_frame(
            df_15m,
            vix_1m=vix_1m,
            timeframe_rule=label_timeframe,
            resample_label=resample_label,
            resample_closed=resample_closed,
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
            model=model,
        )
        if model_key == "tree":
            tf_df = _add_lstm_features_for_tree(
                tf_df,
                include_time_features=include_htf_date_features,
                tz=tz,
            )
        if include_vix_features:
            tf_df = _add_vix_suite_to_frame(
                tf_df,
                vix_1m=vix_1m,
                timeframe_rule=tf_rule,
                resample_label=resample_label,
                resample_closed=resample_closed,
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

def build_feature_matrices(
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
    models: Iterable[str] = ("LSTM",),
    shift_htf_bars: int = 1,
    resample_label: str = "left",
    resample_closed: str = "left",
    include_vix_features: bool = True,
    vix_ticker: str = "$VIX",
    vix_parquet_path: str | Path | None = None,
    vix_warn_on_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build multiple model-specific feature matrices in one pass.

    - Common work is done once: load 1m, resample 15m + HTFs, pivots+labels on 15m.
    - Per-model work: compute feature sets, align HTFs, attach SAME labels.
    """
    feature_timeframes = feature_timeframes or DEFAULT_FEATURE_TIMEFRAMES
    pivot_kwargs = pivot_kwargs or {}
    label_kwargs = label_kwargs or {}

    # ---------- common: load + resample once ----------
    df_1m = load_ticker_parquet(ticker, parquet_path=parquet_path)
    df_1m = ensure_time_index(df_1m, tz=tz)
    vix_1m: pd.DataFrame | None = None
    if include_vix_features:
        try:
            vix_1m = _load_vix_1m(
                vix_ticker=vix_ticker,
                vix_parquet_path=vix_parquet_path,
                tz=tz,
            )
        except Exception as exc:
            if vix_warn_on_missing:
                print(f"[feature_matrix] VIX suite unavailable: {exc}")
            vix_1m = None

    df_15m_ohlcv = resample_ohlcv(
        df_1m, label_timeframe, label=resample_label, closed=resample_closed
    )

    # labels computed ONCE on a “label frame”
    label_frame = df_15m_ohlcv.copy()
    label_frame = add_fractal_pivots(label_frame, **pivot_kwargs)
    label_frame = add_all_labels(label_frame, **label_kwargs)

    label_cols = _collect_label_columns(label_frame)
    y_15m = label_frame[label_cols].copy() if label_cols else pd.DataFrame(index=label_frame.index)

    # HTF OHLCV cached once (feature computation still per model)
    htf_ohlcv: dict[str, pd.DataFrame] = {}
    for tf_label, tf_rule in feature_timeframes.items():
        tf_df = resample_ohlcv(df_1m, tf_rule, label=resample_label, closed=resample_closed)
        if not tf_df.empty:
            htf_ohlcv[tf_label] = tf_df

    # ---------- per-model feature build ----------
    out: dict[str, pd.DataFrame] = {}
    for model in models:
        # Base 15m features
        f15 = df_15m_ohlcv.copy()
        f15 = _add_feature_set(
            f15,
            include_custom=include_custom,
            include_date_features=include_date_features,
            verbose=verbose,
            model=model,
        )
        model_key = (model or "tree").strip().lower()
        if model_key == "tree":
            f15 = _add_lstm_features_for_tree(
                f15,
                include_time_features=True,
                tz=tz,
            )
        if include_vix_features:
            f15 = _add_vix_suite_to_frame(
                f15,
                vix_1m=vix_1m,
                timeframe_rule=label_timeframe,
                resample_label=resample_label,
                resample_closed=resample_closed,
            )

        # Attach labels (same for all models)
        f15 = pd.concat([f15, y_15m], axis=1)

        frames = [f15]

        # HTF features (aligned)
        for tf_label, tf_df in htf_ohlcv.items():
            tf_feat = tf_df.copy()
            tf_feat = _add_feature_set(
                tf_feat,
                include_custom=include_custom,
                include_date_features=include_htf_date_features,
                verbose=verbose,
                model=model,
            )
            if model_key == "tree":
                tf_feat = _add_lstm_features_for_tree(
                    tf_feat,
                    include_time_features=include_htf_date_features,
                    tz=tz,
                )
            if include_vix_features:
                tf_feat = _add_vix_suite_to_frame(
                    tf_feat,
                    vix_1m=vix_1m,
                    timeframe_rule=feature_timeframes[tf_label],
                    resample_label=resample_label,
                    resample_closed=resample_closed,
                )
            aligned = _align_htf_features(
                tf_feat,
                base_index=f15.index,
                suffix=tf_label,
                shift_bars=shift_htf_bars,
            )
            frames.append(aligned)

        model_df = pd.concat(frames, axis=1)
        model_df.attrs["source_parquet"] = str(parquet_path)
        model_df.attrs["ticker"] = ticker
        model_df.attrs["label_timeframe"] = label_timeframe
        model_df.attrs["model"] = str(model)
        out[str(model)] = model_df

    return out


def clean_feature_matrix(
    df: pd.DataFrame,
    *,
    run_diagnostics: bool = False,
    drop_correlated: bool = True,
    save_outputs: bool = False,
    output_dir: Path | None = None,
    ticker: str = "$SPY",
    dataset_name: str | None = None,
    label_cols: list[str] | None = None,
    x_filename: str = "X.parquet",
    write_y: bool = True,
    align_index: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Clean a training matrix using the same steps as feature_engineering.main.
    Returns the cleaned matrix, reduced feature frame, and feature columns.
    When save_outputs is True, writes X.parquet and y.parquet under
    processed/base/datasets/<dataset_name>/.
    """
    source_attrs = dict(df.attrs)
    plot_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    plot_frame = df[plot_cols].copy() if plot_cols else None
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

    cleaned = drop_ohlcv_columns(cleaned)
    feature_df = drop_ohlcv_columns(feature_df)
    feature_cols = list(feature_df.columns)

    if align_index is not None:
        missing = align_index.difference(cleaned.index)
        if len(missing) > 0:
            raise ValueError(
                "align_index contains rows not present in the cleaned frame. "
                "Regenerate features or relax feature NaN handling."
            )
        cleaned = cleaned.loc[align_index]
        feature_df = feature_df.loc[align_index]

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

        # X: model-specific filename
        feature_df.astype("float32").to_parquet(dataset_dir / x_filename, index=False)

        # Plot frame: save OHLCV aligned to cleaned rows if available.
        if plot_frame is not None and not plot_frame.empty:
            aligned_plot = plot_frame
            if not plot_frame.index.equals(cleaned.index):
                try:
                    aligned_plot = plot_frame.loc[cleaned.index]
                except KeyError:
                    missing = cleaned.index.difference(plot_frame.index)
                    if len(missing) > 0:
                        print(
                            f"[warn] plot_frame missing {len(missing)} rows; reindexing with NaNs."
                        )
                    aligned_plot = plot_frame.reindex(cleaned.index)
            aligned_plot.to_parquet(dataset_dir / "plot_frame.parquet", index=True)

        # y: shared, write once (or force if you want)
        if write_y or not (dataset_dir / "y.parquet").exists():
            labels_df.to_parquet(dataset_dir / "y.parquet", index=False)

        # features list: also model-specific so you don't clobber
        features_txt = f"features_{Path(x_filename).stem}.txt"  # e.g. features_X_15min_tree.txt
        with open(dataset_dir / features_txt, "w") as f:
            for col in feature_cols:
                f.write(col + "\n")

        features_dir = resolved_output_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        with open(features_dir / f"{dataset_name}_features.txt", "w") as f:
            for col in feature_cols:
                f.write(col + "\n")

    return cleaned, feature_df, feature_cols


def main() -> None:
    raise SystemExit(
        "Use build_feature_matrix(parquet_path=...) from another script."
    )


if __name__ == "__main__":
    main()
