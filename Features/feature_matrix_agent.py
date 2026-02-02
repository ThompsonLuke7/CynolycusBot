from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta  # registers df.ta accessor

from Data.load_data import get_ticker_processed_base_dir
from Data.retrieve_data import normalize_ticker


@dataclass(frozen=True)
class AgentFeatureConfig:
    ticker: str = "$SPY"
    dataset_name: str = "15min"
    model_name: str = "ga_xgboost"
    processed_root: str | Path | None = None
    model_root: str | Path | None = None
    pivot_label_dir: str = "pivots"
    tb_label_dir: str = "tb"
    include_pivot_probs: bool = True
    include_tb_probs: bool = True
    tz: str | None = "America/New_York"
    session_open: str = "09:30"
    session_close: str = "16:00"
    drop_na: bool = False
    include_state_placeholders: bool = True


def _series_from_ta(
    result: pd.Series | pd.DataFrame, *, prefix: str | None = None
) -> pd.Series:
    if isinstance(result, pd.Series):
        return result
    if isinstance(result, pd.DataFrame):
        if prefix:
            for col in result.columns:
                if col.startswith(prefix):
                    return result[col]
        return result.iloc[:, 0]
    raise TypeError("Unexpected pandas_ta return type")


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour_str, minute_str = value.split(":")
    return int(hour_str), int(minute_str)


def _compute_time_sin_cos(
    index: pd.DatetimeIndex,
    *,
    tz: str | None,
    session_open: str,
    session_close: str,
) -> tuple[pd.Series, pd.Series]:
    idx = index
    if tz:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)

    open_hour, open_minute = _parse_hhmm(session_open)
    close_hour, close_minute = _parse_hhmm(session_close)
    open_minutes = open_hour * 60 + open_minute
    close_minutes = close_hour * 60 + close_minute
    session_minutes = max(1, close_minutes - open_minutes)

    minutes = idx.hour * 60 + idx.minute + idx.second / 60.0
    minutes = np.asarray(minutes, dtype=float)
    minutes_since_open = minutes - open_minutes
    minutes_from_open = np.clip(minutes_since_open / session_minutes, 0.0, 1.0)

    sin_time = np.sin(2 * np.pi * minutes_from_open)
    cos_time = np.cos(2 * np.pi * minutes_from_open)
    return (
        pd.Series(sin_time, index=index, name="sin_time_of_day"),
        pd.Series(cos_time, index=index, name="cos_time_of_day"),
    )


def _load_plot_frame(
    *,
    ticker: str,
    dataset_name: str,
    processed_root: Path | None = None,
) -> pd.DataFrame:
    if processed_root is None:
        clean = normalize_ticker(ticker)
        dataset_dir = get_ticker_processed_base_dir(clean) / "datasets" / dataset_name
    else:
        dataset_dir = processed_root / "datasets" / dataset_name
    plot_path = dataset_dir / "plot_frame.parquet"
    if not plot_path.exists():
        raise FileNotFoundError(f"Missing plot_frame.parquet in {dataset_dir}")
    df = pd.read_parquet(plot_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("plot_frame.parquet must use a DatetimeIndex")
    return df


def _load_prob_series(
    *,
    model_root: Path,
    side: str,
    column: str,
    target_index: pd.DatetimeIndex,
    label_dir: str | None = None,
    fallback_to_root: bool = True,
) -> pd.Series:
    probs_root = model_root / side.lower() / "probs"
    probe_dirs = []
    if label_dir:
        probe_dirs.append(probs_root / label_dir)
    if fallback_to_root or not label_dir:
        probe_dirs.append(probs_root)

    for probs_dir in probe_dirs:
        parquet_path = probs_dir / f"{column.split('_full')[0]}_probs.parquet"
        npy_path = probs_dir / f"{column}.npy"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if column not in df.columns:
                raise KeyError(f"Missing {column} in {parquet_path}")
            return df[column].reindex(target_index)

        if npy_path.exists():
            arr = np.load(npy_path)
            if arr.shape[0] != len(target_index):
                raise ValueError(
                    f"{npy_path.name} length {arr.shape[0]} does not match data length {len(target_index)}"
                )
            return pd.Series(arr, index=target_index, name=column)

    searched = ", ".join(str(d) for d in probe_dirs)
    raise FileNotFoundError(f"Missing probs for {side} ({column}) in: {searched}")


def _compute_prior_day_high(df: pd.DataFrame) -> pd.Series:
    session_key = df.index.normalize()
    day_high = df.groupby(session_key)["high"].max().sort_index()
    pdh = session_key.map(day_high.shift(1))
    return pd.Series(pdh, index=df.index)


def _add_pivot_features(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    df[f"{base_col}_lag1"] = df[base_col].shift(1)
    df[f"{base_col}_lag2"] = df[base_col].shift(2)
    df[f"{base_col}_max_last_4"] = df[base_col].rolling(4, min_periods=1).max()
    df[f"{base_col}_delta_1"] = df[base_col] - df[base_col].shift(1)
    return df


def build_agent_feature_matrix(
    *,
    config: AgentFeatureConfig | None = None,
) -> pd.DataFrame:
    cfg = config or AgentFeatureConfig()
    processed_root = Path(cfg.processed_root) if cfg.processed_root else None
    plot_df = _load_plot_frame(
        ticker=cfg.ticker,
        dataset_name=cfg.dataset_name,
        processed_root=processed_root,
    )
    df = plot_df.copy()

    if cfg.model_root is not None:
        model_root = Path(cfg.model_root)
    else:
        model_root = (
            Path(__file__).resolve().parents[1]
            / "Data"
            / "models"
            / cfg.model_name
            / cfg.dataset_name
        )

    if cfg.include_pivot_probs:
        p_long = _load_prob_series(
            model_root=model_root,
            side="long",
            column="p_long_full",
            target_index=df.index,
            label_dir=cfg.pivot_label_dir,
        )
        p_short = _load_prob_series(
            model_root=model_root,
            side="short",
            column="p_short_full",
            target_index=df.index,
            label_dir=cfg.pivot_label_dir,
        )

        df["p_pivot_long"] = p_long
        df["p_pivot_short"] = p_short
        df = _add_pivot_features(df, "p_pivot_long")
        df = _add_pivot_features(df, "p_pivot_short")

    if cfg.include_tb_probs:
        tb_long = _load_prob_series(
            model_root=model_root,
            side="long",
            column="p_long_full",
            target_index=df.index,
            label_dir=cfg.tb_label_dir,
        )
        tb_short = _load_prob_series(
            model_root=model_root,
            side="short",
            column="p_short_full",
            target_index=df.index,
            label_dir=cfg.tb_label_dir,
        )
        df["p_tb_long"] = tb_long
        df["p_tb_short"] = tb_short
        df = _add_pivot_features(df, "p_tb_long")
        df = _add_pivot_features(df, "p_tb_short")

    sin_time, cos_time = _compute_time_sin_cos(
        df.index, tz=cfg.tz, session_open=cfg.session_open, session_close=cfg.session_close
    )
    df["sin_time_of_day"] = sin_time
    df["cos_time_of_day"] = cos_time

    atr = _series_from_ta(df.ta.atr(length=14, append=False))
    df["atr_pct"] = atr / df["close"].replace(0, np.nan)

    vwap = _series_from_ta(df.ta.vwap(append=False, anchor="D"))
    df["dist_to_vwap"] = (df["close"] - vwap) / df["close"].replace(0, np.nan)

    pdh = _compute_prior_day_high(df)
    df["dist_to_pdh"] = df["close"] - pdh

    adx_df = df.ta.adx(length=14, append=False)
    df["trend_strength"] = _series_from_ta(adx_df, prefix="ADX")

    df["timestamp"] = df.index
    df["day_id"] = pd.Series(df.index.normalize()).factorize()[0]

    close = df["close"].replace(0, np.nan).astype(float)
    for lag in (1, 2, 4, 8, 16):
        df[f"ret_{lag}"] = close.pct_change(lag)

    if cfg.include_state_placeholders:
        df["current_position"] = 0.0
        df["time_in_position"] = 0.0
        df["bars_since_last_trade"] = 0.0
        df["unrealized_pnl"] = 0.0
        df["realized_pnl_today"] = 0.0

    cols = [
        "timestamp",
        "day_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sin_time_of_day",
        "cos_time_of_day",
        "atr_pct",
        "dist_to_vwap",
        "dist_to_pdh",
        "trend_strength",
        "ret_1",
        "ret_2",
        "ret_4",
        "ret_8",
        "ret_16",
    ]
    if cfg.include_pivot_probs:
        cols.extend(
            [
                "p_pivot_long",
                "p_pivot_long_lag1",
                "p_pivot_long_lag2",
                "p_pivot_long_max_last_4",
                "p_pivot_long_delta_1",
                "p_pivot_short",
                "p_pivot_short_lag1",
                "p_pivot_short_lag2",
                "p_pivot_short_max_last_4",
                "p_pivot_short_delta_1",
            ]
        )
    if cfg.include_tb_probs:
        cols.extend(
            [
                "p_tb_long",
                "p_tb_long_lag1",
                "p_tb_long_lag2",
                "p_tb_long_max_last_4",
                "p_tb_long_delta_1",
                "p_tb_short",
                "p_tb_short_lag1",
                "p_tb_short_lag2",
                "p_tb_short_max_last_4",
                "p_tb_short_delta_1",
            ]
        )

    if cfg.include_state_placeholders:
        cols.extend(
            [
                "current_position",
                "time_in_position",
                "bars_since_last_trade",
                "unrealized_pnl",
                "realized_pnl_today",
            ]
        )

    out = df[cols].copy()
    if cfg.drop_na:
        out = out.dropna()
    return out


def main() -> None:
    df = build_agent_feature_matrix()
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()
