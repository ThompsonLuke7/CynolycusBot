from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta  # registers df.ta accessor

from Data.load_data import (
    get_ticker_processed_base_dir,
    load_ticker_csv,
    load_ticker_parquet,
)
from Data.retrieve_data import normalize_ticker

VIX_FEATURE_COLUMNS = [
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
    include_vix_features: bool = True
    vix_ticker: str = "VIXY"
    vix_parquet_path: str | Path | None = "Data/raw/vix/vixy_15min.parquet"
    vix_fetch_if_missing: bool = True
    vix_fetch_timeframe: str = "15Min"
    vix_fetch_limit: int = 100000
    vix_fetch_start: str | None = None
    vix_fetch_end: str | None = None
    vix_resample_rule: str | None = None
    vix_max_lag: str = "2h"
    vix_ffill_limit: int | None = 256
    vix_warn_on_missing: bool = True
    vix_allow_daily_fallback: bool = True
    vix_daily_symbol: str = "VIXY"
    vix_daily_max_lag: str = "7d"


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


def _ensure_vix_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in VIX_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.suffix == "":
        p = p.with_suffix(".parquet")
    if p.is_absolute():
        return p
    return _repo_root() / p


def _infer_vix_fetch_start(
    *,
    target_index: pd.DatetimeIndex,
    explicit_start: str | None,
) -> str:
    if explicit_start:
        return str(explicit_start)
    if len(target_index) == 0:
        return "2021-01-01T00:00:00Z"
    first_ts = pd.to_datetime(target_index.min(), utc=True, errors="coerce")
    if pd.isna(first_ts):
        return "2021-01-01T00:00:00Z"
    start_ts = first_ts - pd.Timedelta(days=30)
    return start_ts.isoformat().replace("+00:00", "Z")


def _infer_vix_fetch_end(
    *,
    target_index: pd.DatetimeIndex,
    explicit_end: str | None,
) -> str | None:
    if explicit_end:
        return str(explicit_end)
    if len(target_index) == 0:
        return None
    last_ts = pd.to_datetime(target_index.max(), utc=True, errors="coerce")
    if pd.isna(last_ts):
        return None
    # Alpaca `end` is exclusive; add a tiny buffer so the final target bar is included.
    end_ts = last_ts + pd.Timedelta(minutes=1)
    return end_ts.isoformat().replace("+00:00", "Z")


def _load_align_vix_ohlcv(
    *,
    cfg: AgentFeatureConfig,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    target_max = pd.to_datetime(target_index.max(), utc=True, errors="coerce")

    def _align_frame(
        source: pd.DataFrame,
        *,
        tolerance_text: str | None,
        ffill_limit: int | None,
    ) -> pd.DataFrame:
        if not isinstance(source.index, pd.DatetimeIndex):
            raise ValueError("VIX data must have a DatetimeIndex.")
        out = source.sort_index()
        if cfg.tz:
            if out.index.tz is None:
                out.index = out.index.tz_localize(cfg.tz)
            else:
                out.index = out.index.tz_convert(cfg.tz)
        # Strict no-lookahead: never use any source row later than the max target bar.
        if not pd.isna(target_max):
            cutoff = target_max
            if out.index.tz is not None:
                cutoff = cutoff.tz_convert(out.index.tz)
            out = out.loc[out.index <= cutoff]
        tolerance = pd.Timedelta(tolerance_text) if tolerance_text else None
        return out.reindex(
            target_index,
            method="ffill",
            limit=ffill_limit,
            tolerance=tolerance,
        )

    vix_df: pd.DataFrame | None = None
    intraday_error: Exception | None = None

    preferred_path: Path | None = None
    if cfg.vix_parquet_path is not None:
        preferred_path = _resolve_path(cfg.vix_parquet_path)

    if preferred_path is not None and not preferred_path.exists() and cfg.vix_fetch_if_missing:
        try:
            from API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

            preferred_path.parent.mkdir(parents=True, exist_ok=True)
            fetch_start = _infer_vix_fetch_start(
                target_index=target_index,
                explicit_start=cfg.vix_fetch_start,
            )
            fetch_end = _infer_vix_fetch_end(
                target_index=target_index,
                explicit_end=cfg.vix_fetch_end,
            )
            print(
                f"[agent_matrix] Missing intraday VIX file at {preferred_path}; "
                "fetching fresh data."
            )
            fetched = fetch_intraday(
                ticker=cfg.vix_ticker,
                start=fetch_start,
                end=fetch_end,
                timeframe=cfg.vix_fetch_timeframe,
                limit=int(cfg.vix_fetch_limit),
                adjustment="raw",
                save_path=str(preferred_path),
            )
            if fetched is None or fetched.empty:
                intraday_error = RuntimeError(
                    "Intraday VIX fetch returned no rows."
                )
        except Exception as exc:  # noqa: BLE001
            intraday_error = exc

    try:
        if preferred_path is not None:
            vix_df = load_ticker_parquet(cfg.vix_ticker, parquet_path=str(preferred_path))
        else:
            vix_df = load_ticker_parquet(cfg.vix_ticker)
    except Exception as exc:  # noqa: BLE001
        if intraday_error is None:
            intraday_error = exc
        else:
            intraday_error = RuntimeError(
                f"{intraday_error}; load_ticker_parquet failed: {exc}"
            )

    if vix_df is not None:
        if cfg.vix_resample_rule:
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            keep = [c for c in agg if c in vix_df.columns]
            if "close" not in keep:
                raise ValueError("VIX frame must include a close column.")
            resample_agg = {k: v for k, v in agg.items() if k in keep}
            vix_df = (
                vix_df[keep]
                .resample(cfg.vix_resample_rule, label="left", closed="left")
                .agg(resample_agg)
            )
            close_cols = [c for c in ("open", "high", "low", "close") if c in vix_df.columns]
            if close_cols:
                vix_df = vix_df.dropna(subset=close_cols)
        aligned_intraday = _align_frame(
            vix_df,
            tolerance_text=cfg.vix_max_lag,
            ffill_limit=cfg.vix_ffill_limit,
        )
        close_valid = (
            aligned_intraday["close"].notna().sum()
            if "close" in aligned_intraday.columns
            else 0
        )
        if close_valid > 0:
            return aligned_intraday
        intraday_error = ValueError(
            "Aligned intraday VIX close has no valid rows after reindex."
        )

    if not cfg.vix_allow_daily_fallback:
        if intraday_error is not None:
            raise intraday_error
        raise FileNotFoundError("Intraday VIX source unavailable.")

    if intraday_error is not None and cfg.vix_warn_on_missing:
        print(
            "[agent_matrix] Intraday VIX unavailable; falling back to daily source. "
            f"Reason: {intraday_error}"
        )

    daily = load_ticker_csv(cfg.vix_daily_symbol)
    keep_daily = [c for c in ("open", "high", "low", "close", "volume") if c in daily.columns]
    if "close" not in keep_daily:
        raise ValueError("Daily VIX fallback is missing close column.")
    daily = daily[keep_daily]
    aligned_daily = _align_frame(
        daily,
        tolerance_text=cfg.vix_daily_max_lag,
        ffill_limit=None,
    )
    close_valid_daily = (
        aligned_daily["close"].notna().sum()
        if "close" in aligned_daily.columns
        else 0
    )
    if close_valid_daily == 0:
        if intraday_error is not None:
            raise RuntimeError(
                f"Intraday VIX failed ({intraday_error}); daily fallback also empty after align."
            ) from intraday_error
        raise RuntimeError("Daily VIX fallback produced no aligned rows.")
    print(
        f"[agent_matrix] Using daily VIX fallback ({cfg.vix_daily_symbol}) "
        f"with max_lag={cfg.vix_daily_max_lag}."
    )
    return aligned_daily


def _add_vix_feature_suite(
    df: pd.DataFrame,
    *,
    vix_ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    vix_close = pd.to_numeric(vix_ohlcv.get("close"), errors="coerce")
    if vix_close is None:
        raise ValueError("Aligned VIX frame is missing close.")
    vix_high = pd.to_numeric(vix_ohlcv.get("high"), errors="coerce")
    vix_low = pd.to_numeric(vix_ohlcv.get("low"), errors="coerce")

    vix_ret_1 = vix_close.pct_change(1)
    vix_ret_4 = vix_close.pct_change(4)
    vix_ret_16 = vix_close.pct_change(16)
    vix_range_pct = (vix_high - vix_low) / vix_close.replace(0, np.nan)
    vix_atr_raw = _series_from_ta(vix_ohlcv.ta.atr(length=14, append=False))
    vix_atr_pct = vix_atr_raw / vix_close.replace(0, np.nan)
    vix_ema_8 = vix_close.ewm(span=8, adjust=False).mean()
    vix_ema_21 = vix_close.ewm(span=21, adjust=False).mean()
    vix_trend = (vix_ema_8 - vix_ema_21) / vix_close.replace(0, np.nan)
    vix_mean_20 = vix_close.rolling(20, min_periods=20).mean()
    vix_std_20 = vix_close.rolling(20, min_periods=20).std(ddof=0)
    vix_z_20 = (vix_close - vix_mean_20) / vix_std_20.replace(0, np.nan)
    vix_vol_of_vol_20 = vix_ret_1.rolling(20, min_periods=20).std(ddof=0)

    df["vix_close"] = vix_close
    df["vix_ret_1"] = vix_ret_1
    df["vix_ret_4"] = vix_ret_4
    df["vix_ret_16"] = vix_ret_16
    df["vix_range_pct"] = vix_range_pct
    df["vix_atr_pct"] = vix_atr_pct
    df["vix_trend_ema_8_21"] = vix_trend
    df["vix_z_20"] = vix_z_20
    df["vix_vol_of_vol_20"] = vix_vol_of_vol_20
    df["ret_1_x_vix"] = df["ret_1"] * vix_close
    df["atr_pct_x_vix"] = df["atr_pct"] * vix_close
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

    if cfg.include_vix_features:
        try:
            vix_ohlcv = _load_align_vix_ohlcv(cfg=cfg, target_index=df.index)
            df = _add_vix_feature_suite(df, vix_ohlcv=vix_ohlcv)
        except Exception as exc:
            if cfg.vix_warn_on_missing:
                print(f"[agent_matrix] VIX feature suite unavailable: {exc}")
            df = _ensure_vix_feature_cols(df)

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
    if cfg.include_vix_features:
        cols.extend(VIX_FEATURE_COLUMNS)

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
