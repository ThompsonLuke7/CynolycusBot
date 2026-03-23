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
from Data.retrieve_data import get_output_path, normalize_ticker, retrieve_data

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
    vix_parquet_path: str | Path | None = "Data/raw/vix/vixy_10min.parquet"
    vix_fetch_if_missing: bool = True
    vix_fetch_timeframe: str = "10Min"
    vix_fetch_limit: int = 100000
    vix_fetch_start: str | None = None
    vix_fetch_end: str | None = None
    vix_resample_rule: str | None = None
    vix_max_lag: str = "2h"
    vix_ffill_limit: int | None = 256
    vix_min_coverage_ratio: float = 0.90
    vix_refetch_if_low_coverage: bool = True
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


def _compute_time_features(
    index: pd.DatetimeIndex,
    *,
    tz: str | None,
    session_open: str,
    session_close: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
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
    minutes_since_open_raw = minutes - open_minutes
    minutes_since_open = np.clip(minutes_since_open_raw, 0.0, float(session_minutes))
    minutes_to_close = np.clip(float(session_minutes) - minutes_since_open, 0.0, float(session_minutes))
    session_progress = np.clip(minutes_since_open / float(session_minutes), 0.0, 1.0)

    sin_time = np.sin(2 * np.pi * session_progress)
    cos_time = np.cos(2 * np.pi * session_progress)
    day_of_week = np.asarray(idx.dayofweek, dtype=float)
    day_of_week_sin = np.sin(2 * np.pi * (day_of_week / 7.0))
    day_of_week_cos = np.cos(2 * np.pi * (day_of_week / 7.0))
    return (
        pd.Series(sin_time, index=index, name="sin_time_of_day"),
        pd.Series(cos_time, index=index, name="cos_time_of_day"),
        pd.Series(minutes_since_open, index=index, name="minutes_since_open"),
        pd.Series(minutes_to_close, index=index, name="minutes_to_close"),
        pd.Series(day_of_week_sin, index=index, name="day_of_week_sin"),
        pd.Series(day_of_week_cos, index=index, name="day_of_week_cos"),
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
    side_root = model_root / side.lower()
    probe_dirs: list[Path] = []
    if label_dir:
        probe_dirs.append(side_root / label_dir)
        # Backward compatibility with previous nested layout.
        probe_dirs.append(side_root / "probs" / label_dir)
    if fallback_to_root or not label_dir:
        probe_dirs.append(side_root)
        probe_dirs.append(side_root / "probs")

    for probs_dir in probe_dirs:
        parquet_path = probs_dir / f"{column.split('_full')[0]}_probs.parquet"
        npy_path = probs_dir / f"{column}.npy"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if column not in df.columns:
                raise KeyError(f"Missing {column} in {parquet_path}")
            series = pd.to_numeric(df[column], errors="coerce")
            aligned = series.reindex(target_index)
            if aligned.notna().any():
                return aligned
            # Backward compatibility: older probability parquet files may have
            # non-datetime/default indices; align positionally when lengths match.
            if len(series) == len(target_index):
                return pd.Series(series.to_numpy(dtype=float), index=target_index, name=column)
            return aligned

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


def _localize_index(index: pd.DatetimeIndex, tz: str | None) -> pd.DatetimeIndex:
    idx = index
    if tz:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
    return idx


def _add_intraday_sr_distance_features(
    df: pd.DataFrame,
    *,
    atr: pd.Series,
    tz: str | None,
    session_open: str,
    open_range_minutes: int = 30,
) -> pd.DataFrame:
    high = pd.to_numeric(df.get("high"), errors="coerce")
    low = pd.to_numeric(df.get("low"), errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce")
    atr_safe = pd.to_numeric(atr, errors="coerce").replace(0.0, np.nan)

    idx_local = _localize_index(df.index, tz)
    session_key = pd.Series(idx_local.normalize(), index=df.index)
    minute_of_day = pd.Series(
        idx_local.hour * 60 + idx_local.minute + idx_local.second / 60.0,
        index=df.index,
        dtype=float,
    )

    day_high_so_far = high.groupby(session_key).cummax()
    day_low_so_far = low.groupby(session_key).cummin()
    df["dist_to_day_high_so_far_atr"] = (day_high_so_far - close) / atr_safe
    df["dist_to_day_low_so_far_atr"] = (close - day_low_so_far) / atr_safe

    open_hour, open_minute = _parse_hhmm(session_open)
    open_min_total = open_hour * 60 + open_minute
    or_end = open_min_total + max(1, int(open_range_minutes))
    in_opening_range = (minute_of_day >= float(open_min_total)) & (minute_of_day < float(or_end))

    or_high_partial = high.where(in_opening_range).groupby(session_key).cummax()
    or_low_partial = low.where(in_opening_range).groupby(session_key).cummin()
    or_high_final = high.where(in_opening_range).groupby(session_key).transform("max")
    or_low_final = low.where(in_opening_range).groupby(session_key).transform("min")
    or_high = or_high_partial.where(in_opening_range, or_high_final)
    or_low = or_low_partial.where(in_opening_range, or_low_final)

    df["dist_to_or_high_30m_atr"] = (or_high - close) / atr_safe
    df["dist_to_or_low_30m_atr"] = (close - or_low) / atr_safe
    return df


def _add_pivot_features(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    df[f"{base_col}_lag1"] = df[base_col].shift(1)
    df[f"{base_col}_lag2"] = df[base_col].shift(2)
    df[f"{base_col}_max_last_4"] = df[base_col].rolling(4, min_periods=1).max()
    df[f"{base_col}_delta_1"] = df[base_col] - df[base_col].shift(1)
    return df


def _rolling_zscore(
    series: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    win = max(2, int(window))
    minp = int(min_periods) if min_periods is not None else win
    rolling_mean = series.rolling(win, min_periods=minp).mean()
    rolling_std = series.rolling(win, min_periods=minp).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def _rolling_last_percentile(
    series: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    win = max(2, int(window))
    minp = int(min_periods) if min_periods is not None else win

    def _percentile_last(arr: np.ndarray) -> float:
        if arr.size == 0:
            return np.nan
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan
        last = arr[-1]
        if not np.isfinite(last):
            return np.nan
        return float((valid <= last).sum()) / float(valid.size)

    return series.rolling(win, min_periods=minp).apply(_percentile_last, raw=True)


def _add_probability_confidence_features(df: pd.DataFrame) -> pd.DataFrame:
    has_pivot = {"p_pivot_long", "p_pivot_short"}.issubset(df.columns)
    has_tb = {"p_tb_long", "p_tb_short"}.issubset(df.columns)

    if has_pivot:
        pivot_edge = df["p_pivot_long"] - df["p_pivot_short"]
        df["pivot_edge"] = pivot_edge
        df["pivot_edge_abs"] = pivot_edge.abs()
    if has_tb:
        tb_edge = df["p_tb_long"] - df["p_tb_short"]
        df["tb_edge"] = tb_edge
        df["tb_edge_abs"] = tb_edge.abs()

    if has_pivot and has_tb:
        pivot_edge = pd.to_numeric(df["pivot_edge"], errors="coerce")
        tb_edge = pd.to_numeric(df["tb_edge"], errors="coerce")
        df["edge_disagreement_abs"] = (pivot_edge - tb_edge).abs()
        pivot_sign = np.sign(pivot_edge.to_numpy(dtype=float, copy=False))
        tb_sign = np.sign(tb_edge.to_numpy(dtype=float, copy=False))
        valid_sign = np.isfinite(pivot_sign) & np.isfinite(tb_sign)
        sign_disagree = np.full(len(df), np.nan, dtype=float)
        sign_disagree[valid_sign] = (pivot_sign[valid_sign] != tb_sign[valid_sign]).astype(float)
        df["edge_sign_disagreement"] = sign_disagree
    return df


def _add_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    atr_pct = pd.to_numeric(df.get("atr_pct"), errors="coerce")
    df["atr_pct_z_64"] = _rolling_zscore(atr_pct, window=64, min_periods=32)
    df["atr_pct_rank_64"] = _rolling_last_percentile(atr_pct, window=64, min_periods=32)

    ret_1 = pd.to_numeric(df.get("ret_1"), errors="coerce")
    df["realized_vol_4"] = ret_1.rolling(4, min_periods=4).std(ddof=0)
    df["realized_vol_16"] = ret_1.rolling(16, min_periods=16).std(ddof=0)
    df["realized_vol_32"] = ret_1.rolling(32, min_periods=32).std(ddof=0)

    close = pd.to_numeric(df.get("close"), errors="coerce")
    high = pd.to_numeric(df.get("high"), errors="coerce")
    low = pd.to_numeric(df.get("low"), errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_pct = tr / close.replace(0.0, np.nan)
    tr_ema_fast = tr_pct.ewm(span=8, adjust=False).mean()
    tr_ema_slow = tr_pct.ewm(span=32, adjust=False).mean()
    tr_avg_32 = tr_pct.rolling(32, min_periods=16).mean()
    df["range_regime_8_32"] = tr_ema_fast / tr_ema_slow.replace(0.0, np.nan) - 1.0
    df["range_expansion_32"] = tr_pct / tr_avg_32.replace(0.0, np.nan) - 1.0
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


def _infer_vix_daily_start(
    *,
    target_index: pd.DatetimeIndex,
) -> str:
    if len(target_index) == 0:
        return "2015-01-01"
    first_ts = pd.to_datetime(target_index.min(), utc=True, errors="coerce")
    if pd.isna(first_ts):
        return "2015-01-01"
    # Keep some buffer for rolling features.
    return (first_ts - pd.Timedelta(days=30)).strftime("%Y-%m-%d")


def _load_align_vix_ohlcv(
    *,
    cfg: AgentFeatureConfig,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    target_max = pd.to_datetime(target_index.max(), utc=True, errors="coerce")
    required_coverage = float(np.clip(cfg.vix_min_coverage_ratio, 0.0, 1.0))

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

    def _coverage_ratio(frame: pd.DataFrame) -> float:
        if len(target_index) == 0:
            return 1.0
        if "close" not in frame.columns:
            return 0.0
        return float(frame["close"].notna().mean())

    def _prepare_intraday_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame
        if cfg.vix_resample_rule:
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            keep = [c for c in agg if c in out.columns]
            if "close" not in keep:
                raise ValueError("VIX frame must include a close column.")
            resample_agg = {k: v for k, v in agg.items() if k in keep}
            out = (
                out[keep]
                .resample(cfg.vix_resample_rule, label="left", closed="left")
                .agg(resample_agg)
            )
            close_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
            if close_cols:
                out = out.dropna(subset=close_cols)
        return out

    def _fetch_to_preferred(reason: str) -> Exception | None:
        if preferred_path is None or not cfg.vix_fetch_if_missing:
            return RuntimeError(reason)
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
                f"[agent_matrix] Fetching intraday VIX ({cfg.vix_ticker}) "
                f"to {preferred_path} because: {reason}"
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
                return RuntimeError("Intraday VIX fetch returned no rows.")
            return None
        except Exception as exc:  # noqa: BLE001
            return exc

    def _refresh_daily_fallback(reason: str) -> Exception | None:
        try:
            start = _infer_vix_daily_start(target_index=target_index)
            print(
                f"[agent_matrix] Refreshing daily VIX ({cfg.vix_daily_symbol}) "
                f"from {start} because: {reason}"
            )
            daily_df = retrieve_data(cfg.vix_daily_symbol, start=start, interval="1d")
            if daily_df is None or daily_df.empty:
                return RuntimeError("Daily VIX refresh returned no rows.")
            out_path = get_output_path(cfg.vix_daily_symbol)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            daily_df.to_csv(out_path)
            return None
        except Exception as exc:  # noqa: BLE001
            return exc

    vix_df: pd.DataFrame | None = None
    intraday_error: Exception | None = None

    preferred_path: Path | None = None
    if cfg.vix_parquet_path is not None:
        preferred_path = _resolve_path(cfg.vix_parquet_path)

    if preferred_path is not None and not preferred_path.exists() and cfg.vix_fetch_if_missing:
        intraday_error = _fetch_to_preferred(
            f"missing intraday file at {preferred_path}"
        )

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
        try:
            vix_df = _prepare_intraday_frame(vix_df)
        except Exception as exc:  # noqa: BLE001
            intraday_error = exc
            vix_df = None

    if vix_df is not None:
        aligned_intraday = _align_frame(
            vix_df,
            tolerance_text=cfg.vix_max_lag,
            ffill_limit=cfg.vix_ffill_limit,
        )
        coverage = _coverage_ratio(aligned_intraday)
        if cfg.vix_warn_on_missing:
            print(f"[agent_matrix] Intraday VIX aligned coverage={coverage:.1%}")

        if (
            coverage < required_coverage
            and preferred_path is not None
            and cfg.vix_refetch_if_low_coverage
            and cfg.vix_fetch_if_missing
        ):
            refetch_error = _fetch_to_preferred(
                f"insufficient aligned coverage {coverage:.1%} (< {required_coverage:.1%})"
            )
            if refetch_error is None:
                try:
                    vix_df_refreshed = load_ticker_parquet(
                        cfg.vix_ticker, parquet_path=str(preferred_path)
                    )
                    vix_df_refreshed = _prepare_intraday_frame(vix_df_refreshed)
                    aligned_intraday = _align_frame(
                        vix_df_refreshed,
                        tolerance_text=cfg.vix_max_lag,
                        ffill_limit=cfg.vix_ffill_limit,
                    )
                    coverage = _coverage_ratio(aligned_intraday)
                except Exception as exc:  # noqa: BLE001
                    intraday_error = exc
            else:
                intraday_error = refetch_error

        if coverage >= required_coverage:
            return aligned_intraday
        intraday_error = ValueError(
            "Aligned intraday VIX coverage too low after reindex "
            f"({coverage:.1%} < {required_coverage:.1%})."
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
    daily_coverage = _coverage_ratio(aligned_daily)
    close_valid_daily = aligned_daily["close"].notna().sum() if "close" in aligned_daily.columns else 0
    if (
        daily_coverage < required_coverage
        and cfg.vix_fetch_if_missing
    ):
        refresh_error = _refresh_daily_fallback(
            f"daily aligned coverage {daily_coverage:.1%} (< {required_coverage:.1%})"
        )
        if refresh_error is None:
            daily = load_ticker_csv(cfg.vix_daily_symbol)
            keep_daily = [c for c in ("open", "high", "low", "close", "volume") if c in daily.columns]
            if "close" not in keep_daily:
                raise ValueError("Daily VIX fallback is missing close column after refresh.")
            daily = daily[keep_daily]
            aligned_daily = _align_frame(
                daily,
                tolerance_text=cfg.vix_daily_max_lag,
                ffill_limit=None,
            )
            daily_coverage = _coverage_ratio(aligned_daily)
            close_valid_daily = aligned_daily["close"].notna().sum() if "close" in aligned_daily.columns else 0
        elif cfg.vix_warn_on_missing:
            print(f"[agent_matrix] Daily VIX refresh failed: {refresh_error}")
    if close_valid_daily == 0:
        if intraday_error is not None:
            raise RuntimeError(
                f"Intraday VIX failed ({intraday_error}); daily fallback also empty after align."
            ) from intraday_error
        raise RuntimeError("Daily VIX fallback produced no aligned rows.")
    if cfg.vix_warn_on_missing:
        print(f"[agent_matrix] Daily VIX aligned coverage={daily_coverage:.1%}")
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
    if not isinstance(vix_close, pd.Series):
        vix_close = pd.Series(np.nan, index=df.index, dtype=float)
    if not isinstance(vix_high, pd.Series):
        vix_high = pd.Series(np.nan, index=df.index, dtype=float)
    if not isinstance(vix_low, pd.Series):
        vix_low = pd.Series(np.nan, index=df.index, dtype=float)
    vix_close = vix_close.reindex(df.index)
    vix_high = vix_high.reindex(df.index)
    vix_low = vix_low.reindex(df.index)

    # Preserve the historical behavior explicitly: forward-fill prior observed
    # VIX values first, then compute pct_change without pandas' deprecated
    # implicit fill_method default.
    vix_close_ffill = vix_close.ffill()
    vix_ret_1 = vix_close_ffill.pct_change(1, fill_method=None)
    vix_ret_4 = vix_close_ffill.pct_change(4, fill_method=None)
    vix_ret_16 = vix_close_ffill.pct_change(16, fill_method=None)
    vix_range_pct = (vix_high - vix_low) / vix_close.replace(0, np.nan)
    # Use a direct ATR implementation to avoid TA statefulness turning nearly all rows NaN.
    vix_prev_close = vix_close.shift(1)
    vix_tr = pd.concat(
        [
            (vix_high - vix_low).abs(),
            (vix_high - vix_prev_close).abs(),
            (vix_low - vix_prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    vix_atr_raw = vix_tr.rolling(14, min_periods=14).mean()
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

    df = _add_probability_confidence_features(df)

    (
        sin_time,
        cos_time,
        minutes_since_open,
        minutes_to_close,
        day_of_week_sin,
        day_of_week_cos,
    ) = _compute_time_features(
        df.index, tz=cfg.tz, session_open=cfg.session_open, session_close=cfg.session_close
    )
    df["sin_time_of_day"] = sin_time
    df["cos_time_of_day"] = cos_time
    df["minutes_since_open"] = minutes_since_open
    df["minutes_to_close"] = minutes_to_close
    df["day_of_week_sin"] = day_of_week_sin
    df["day_of_week_cos"] = day_of_week_cos

    atr = _series_from_ta(df.ta.atr(length=14, append=False))
    df["atr_pct"] = atr / df["close"].replace(0, np.nan)

    vwap = _series_from_ta(df.ta.vwap(append=False, anchor="D"))
    df["dist_to_vwap"] = (df["close"] - vwap) / df["close"].replace(0, np.nan)

    pdh = _compute_prior_day_high(df)
    df["dist_to_pdh"] = df["close"] - pdh
    df = _add_intraday_sr_distance_features(
        df,
        atr=atr,
        tz=cfg.tz,
        session_open=cfg.session_open,
        open_range_minutes=30,
    )

    adx_df = df.ta.adx(length=14, append=False)
    df["trend_strength"] = _series_from_ta(adx_df, prefix="ADX")

    df["timestamp"] = df.index
    df["day_id"] = pd.Series(df.index.normalize()).factorize()[0]

    close = df["close"].replace(0, np.nan).astype(float)
    for lag in (1, 2, 4, 8, 16):
        df[f"ret_{lag}"] = close.pct_change(lag)

    df = _add_volatility_regime_features(df)

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
        "minutes_since_open",
        "minutes_to_close",
        "day_of_week_sin",
        "day_of_week_cos",
        "atr_pct",
        "dist_to_vwap",
        "dist_to_pdh",
        "dist_to_day_high_so_far_atr",
        "dist_to_day_low_so_far_atr",
        "dist_to_or_high_30m_atr",
        "dist_to_or_low_30m_atr",
        "trend_strength",
        "ret_1",
        "ret_2",
        "ret_4",
        "ret_8",
        "ret_16",
        "atr_pct_z_64",
        "atr_pct_rank_64",
        "realized_vol_4",
        "realized_vol_16",
        "realized_vol_32",
        "range_regime_8_32",
        "range_expansion_32",
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
                "pivot_edge",
                "pivot_edge_abs",
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
                "tb_edge",
                "tb_edge_abs",
            ]
        )
    if cfg.include_pivot_probs and cfg.include_tb_probs:
        cols.extend(
            [
                "edge_disagreement_abs",
                "edge_sign_disagreement",
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
