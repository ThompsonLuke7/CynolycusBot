import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..inference.live_inference import (
    LiveInferenceEngine,
    LiveMetaXGBAgent,
    LivePPOAgent,
    build_meta_feature_frame_from_1m,
    build_tree_feature_frame_from_1m,
)
from .live_runner import (
    LiveBarProcessor,
    _action_to_position,
    _fmt_prob,
    _format_ts_local,
    _load_test_split_warmup_1m,
    _make_1m_handler,
)
from Policy.execution_latch import DirectionExecutionLatch
from Policy.order_policy import OptionOrderPolicy, OptionOrderPolicyConfig


def _load_history(path: Path, *, assume_tz: str = "UTC") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("History file must be .csv or .parquet")

    rename_map = {
        "Date": "timestamp",
        "date": "timestamp",
        "Datetime": "timestamp",
        "datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        else:
            raise ValueError("History data must include a timestamp column or DatetimeIndex.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(assume_tz)
    df = df.dropna(subset=["timestamp"])
    return df


def _apply_regular_hours(df: pd.DataFrame, *, tz: str = "America/New_York") -> pd.DataFrame:
    ts = df["timestamp"].dt.tz_convert(tz)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    regular_mask = minutes.between(570, 960)
    return df.loc[regular_mask].copy()


def _print_meta_prob_log(*, prefix: str, probs: dict[str, float | None] | None, thresholds: dict[str, float] | None) -> None:
    if not probs and not thresholds:
        return
    probs = probs or {}
    thresholds = thresholds or {}
    print(
        f"{prefix} "
        f"p_enter_long={_fmt_prob(probs.get('p_enter_long'))} thr_enter_long={_fmt_prob(thresholds.get('enter_long'))} "
        f"p_enter_short={_fmt_prob(probs.get('p_enter_short'))} thr_enter_short={_fmt_prob(thresholds.get('enter_short'))} "
        f"p_exit_long={_fmt_prob(probs.get('p_exit_long'))} thr_exit_long={_fmt_prob(thresholds.get('exit_long'))} "
        f"p_exit_short={_fmt_prob(probs.get('p_exit_short'))} thr_exit_short={_fmt_prob(thresholds.get('exit_short'))}"
    )


def _print_trace_prob_diagnostics(trace_df: pd.DataFrame) -> None:
    if trace_df.empty:
        return

    prob_cols = (
        "p_enter_long",
        "p_enter_short",
        "p_exit_long",
        "p_exit_short",
        "p_pivot_long",
        "p_pivot_short",
        "p_tb_long",
        "p_tb_short",
    )
    for col in prob_cols:
        if col not in trace_df.columns:
            continue
        series = pd.to_numeric(trace_df[col], errors="coerce")
        valid = series[np.isfinite(series)]
        if valid.empty:
            print(f"[replay] Trace probs {col}: valid=0/{len(series):,} (all NaN/non-numeric)")
            continue
        zero_count = int((valid == 0.0).sum())
        print(
            f"[replay] Trace probs {col}: valid={len(valid):,}/{len(series):,} "
            f"zero={zero_count:,} ({zero_count / len(valid):.1%})"
        )

    source_cols = (
        "p_pivot_long_source",
        "p_pivot_short_source",
        "p_tb_long_source",
        "p_tb_short_source",
    )
    for col in source_cols:
        if col not in trace_df.columns:
            continue
        src = trace_df[col].astype("string").str.lower()
        valid = src[src.notna()]
        if valid.empty:
            print(f"[replay] Trace sources {col}: no values")
            continue
        counts = valid.value_counts(dropna=False)
        pieces = [f"{idx}={int(val):,}" for idx, val in counts.items()]
        fill_ratio = float((valid == "fill").mean())
        print(
            f"[replay] Trace sources {col}: {', '.join(pieces)} "
            f"(fill={fill_ratio:.1%})"
        )


def _save_trace_plot(
    *,
    trace_df: pd.DataFrame,
    save_path: Path,
    tz: str,
) -> None:
    if trace_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[replay] Plot skipped (matplotlib unavailable): {exc}")
        return

    df = trace_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return

    symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist())
    n = max(1, len(symbols))
    fig, axes = plt.subplots(
        n * 2,
        1,
        figsize=(18, max(7.5, 6.0 * n)),
        sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1.0] * n},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for row_idx, symbol in enumerate(symbols):
        sdf = df[df["symbol"].astype(str) == str(symbol)].copy()
        if sdf.empty:
            continue

        sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True, errors="coerce")
        sdf = sdf.dropna(subset=["timestamp"]).sort_values("timestamp")
        if sdf.empty:
            continue

        plot_df = sdf.copy()
        plot_df["ts_local"] = plot_df["timestamp"].dt.tz_convert(tz)
        plot_df = plot_df.dropna(subset=["ts_local"]).sort_values("ts_local")
        plot_df = plot_df.drop_duplicates(subset=["ts_local"], keep="last").set_index("ts_local")
        if plot_df.empty:
            continue

        ax_price = axes[row_idx * 2]
        ax_probs = axes[row_idx * 2 + 1]

        pos = np.arange(len(plot_df))
        close = pd.to_numeric(plot_df["close"], errors="coerce").to_numpy()
        open_ = pd.to_numeric(plot_df.get("open"), errors="coerce").to_numpy() if "open" in plot_df.columns else None
        high = pd.to_numeric(plot_df.get("high"), errors="coerce").to_numpy() if "high" in plot_df.columns else None
        low = pd.to_numeric(plot_df.get("low"), errors="coerce").to_numpy() if "low" in plot_df.columns else None
        has_ohlc = open_ is not None and high is not None and low is not None

        if has_ohlc:
            valid_mask = np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
        else:
            valid_mask = np.isfinite(close)
        if not valid_mask.any():
            continue

        if has_ohlc:
            up = close >= open_
            up_mask = up & valid_mask
            down_mask = (~up) & valid_mask
            ax_price.vlines(pos[valid_mask], low[valid_mask], high[valid_mask], color="#4a4a4a", linewidth=1.0, zorder=1)
            ax_price.bar(
                pos[up_mask],
                close[up_mask] - open_[up_mask],
                width=0.8,
                bottom=open_[up_mask],
                color="#1976D2",
                edgecolor="none",
                zorder=1.2,
                label="bull candle",
            )
            ax_price.bar(
                pos[down_mask],
                close[down_mask] - open_[down_mask],
                width=0.8,
                bottom=open_[down_mask],
                color="#E53935",
                edgecolor="none",
                zorder=1.2,
                label="bear candle",
            )
            spread = (high - low)[valid_mask]
            marker_offset = np.nanmedian(spread)
            if not np.isfinite(marker_offset) or marker_offset <= 0:
                marker_offset = np.nanmax(high[valid_mask]) * 0.002
            y_enter_long = low - marker_offset * 1.8
            y_exit_long = high + marker_offset * 1.2
            y_enter_short = high + marker_offset * 1.8
            y_exit_short = low - marker_offset * 1.2
        else:
            ax_price.plot(pos, close, color="#1f77b4", linewidth=1.4, label="close")
            clean_close = close[valid_mask]
            marker_offset = np.nanmedian(np.abs(np.diff(clean_close)))
            if not np.isfinite(marker_offset) or marker_offset <= 0:
                marker_offset = np.nanmax(clean_close) * 0.002
            y_enter_long = close - marker_offset * 1.8
            y_exit_long = close + marker_offset * 1.2
            y_enter_short = close + marker_offset * 1.8
            y_exit_short = close - marker_offset * 1.2

        exec_pos = pd.to_numeric(plot_df.get("exec_pos"), errors="coerce").fillna(0.0).astype(int)
        prev_exec = exec_pos.shift(1).fillna(0).astype(int)
        entry_long_mask = ((exec_pos == 1) & (prev_exec != 1)).to_numpy() & valid_mask
        exit_long_mask = ((prev_exec == 1) & (exec_pos != 1)).to_numpy() & valid_mask
        entry_short_mask = ((exec_pos == -1) & (prev_exec != -1)).to_numpy() & valid_mask
        exit_short_mask = ((prev_exec == -1) & (exec_pos != -1)).to_numpy() & valid_mask

        if entry_long_mask.any():
            ax_price.scatter(pos[entry_long_mask], y_enter_long[entry_long_mask], color="#2E7D32", marker="^", s=58, label="enter long", zorder=2.1)
        if exit_long_mask.any():
            ax_price.scatter(pos[exit_long_mask], y_exit_long[exit_long_mask], color="#8c564b", marker="v", s=54, label="exit long", zorder=2.1)
        if entry_short_mask.any():
            ax_price.scatter(pos[entry_short_mask], y_enter_short[entry_short_mask], color="#C62828", marker="v", s=58, label="enter short", zorder=2.1)
        if exit_short_mask.any():
            ax_price.scatter(pos[exit_short_mask], y_exit_short[exit_short_mask], color="#9467bd", marker="^", s=54, label="exit short", zorder=2.1)

        ax_price.set_title(f"{symbol} | meta entries/exits")
        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.25)
        ax_price.legend(loc="upper left", fontsize=8)

        prob_specs = (
            ("p_enter_long", "#2ca02c", "p_enter_long"),
            ("p_enter_short", "#d62728", "p_enter_short"),
            ("p_exit_long", "#17becf", "p_exit_long"),
            ("p_exit_short", "#ff7f0e", "p_exit_short"),
        )
        thr_specs = (
            ("thr_enter_long", "#2ca02c", "thr_enter_long"),
            ("thr_enter_short", "#d62728", "thr_enter_short"),
            ("thr_exit_long", "#17becf", "thr_exit_long"),
            ("thr_exit_short", "#ff7f0e", "thr_exit_short"),
        )
        plotted_any = False
        for col, color, label in prob_specs:
            if col in plot_df:
                series = pd.to_numeric(plot_df[col], errors="coerce").to_numpy()
                if np.isfinite(series).any():
                    ax_probs.plot(pos, series, color=color, linewidth=1.3, label=label)
                    plotted_any = True
        for col, color, label in thr_specs:
            if col in plot_df:
                series = pd.to_numeric(plot_df[col], errors="coerce")
                finite = series[np.isfinite(series)]
                if finite.size:
                    ax_probs.axhline(
                        float(finite.iloc[-1]),
                        color=color,
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.85,
                        label=label,
                    )
                    plotted_any = True
        if plotted_any:
            ax_probs.set_ylim(-0.02, 1.02)
            ax_probs.legend(loc="upper right", fontsize=8)
        ax_probs.set_title(f"{symbol} | meta probabilities")
        ax_probs.set_ylabel("Probability")
        ax_probs.grid(True, alpha=0.25)

        dates = pd.Series(plot_df.index)
        day_start = dates.dt.normalize().ne(dates.dt.normalize().shift())
        tick_positions = pos[day_start.to_numpy()]
        tick_labels = dates[day_start].dt.strftime("%Y-%m-%d").to_list()
        if len(tick_positions) > 25:
            step = int(np.ceil(len(tick_positions) / 25))
            tick_positions = tick_positions[::step]
            tick_labels = tick_labels[::step]
        if len(tick_positions) > 0:
            ax_probs.set_xticks(tick_positions)
            ax_probs.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
            ax_probs.set_xlabel("Session")
            for x in tick_positions:
                ax_price.axvline(x, color="#cfd8dc", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0.5)
                ax_probs.axvline(x, color="#cfd8dc", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0.5)
        else:
            ax_probs.set_xlabel("Bar")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _write_frame(df: pd.DataFrame, *, path_no_ext: Path, fmt: str) -> Path:
    fmt_l = str(fmt).strip().lower()
    if fmt_l == "csv":
        out = path_no_ext.with_suffix(".csv")
        df.to_csv(out, index=True)
        return out
    out = path_no_ext.with_suffix(".parquet")
    df.to_parquet(out)
    return out


def _dump_live_inference_matrices(
    *,
    raw_df: pd.DataFrame,
    symbols: list[str],
    agent: object | None,
    trace_df: pd.DataFrame | None,
    interval_minutes: int,
    resample_label: str,
    resample_closed: str,
    tz: str | None,
    assume_tz: str,
    session_open: str,
    session_close: str,
    include_pivot_probs: bool,
    include_tb_probs: bool,
    fill_missing_prob: float,
    out_dir: Path,
    out_fmt: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_agent = agent if isinstance(agent, LiveMetaXGBAgent) else None
    ga_predictor = getattr(meta_agent, "_ga_predictor", None) if meta_agent is not None else None
    ga_feature_list = getattr(ga_predictor, "_feature_list", None) if ga_predictor is not None else None
    include_vix_features = bool(getattr(meta_agent, "_include_vix_features", True)) if meta_agent is not None else True

    for symbol in symbols:
        one_min = raw_df[raw_df["symbol"].astype(str) == str(symbol)].copy()
        if one_min.empty:
            continue
        one_min["timestamp"] = pd.to_datetime(one_min["timestamp"], utc=True, errors="coerce")
        one_min = one_min.dropna(subset=["timestamp"]).sort_values("timestamp")
        one_min = one_min[["timestamp", "open", "high", "low", "close", "volume"]]
        one_min = one_min.set_index("timestamp")

        tf_rule = f"{int(interval_minutes)}min"
        x_tree = build_tree_feature_frame_from_1m(
            one_min,
            label_timeframe=tf_rule,
            resample_label=resample_label,
            resample_closed=resample_closed,
            tz=tz or "America/New_York",
            assume_tz=assume_tz,
        )

        symbol_dir = out_dir / str(symbol).lower()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        x_tree_path = _write_frame(
            x_tree,
            path_no_ext=symbol_dir / "live_ga_tree_matrix_full",
            fmt=out_fmt,
        )
        print(f"[replay] Dumped live GA tree matrix (full): {x_tree_path}")

        if isinstance(ga_feature_list, list) and len(ga_feature_list) > 0:
            x_tree_sel = x_tree.reindex(columns=ga_feature_list)
            x_tree_sel_path = _write_frame(
                x_tree_sel,
                path_no_ext=symbol_dir / "live_ga_tree_matrix_selected",
                fmt=out_fmt,
            )
            print(f"[replay] Dumped live GA tree matrix (selected): {x_tree_sel_path}")

        ga_probs = pd.DataFrame(index=x_tree.index)
        if ga_predictor is not None and not x_tree.empty:
            ga_probs = ga_predictor.predict_frame(x_tree)
            ga_probs_path = _write_frame(
                ga_probs,
                path_no_ext=symbol_dir / "live_ga_probs",
                fmt=out_fmt,
            )
            print(f"[replay] Dumped live GA probs: {ga_probs_path}")

        meta_frame = build_meta_feature_frame_from_1m(
            one_min,
            rule=tf_rule,
            label=resample_label,
            closed=resample_closed,
            tz=tz or "America/New_York",
            assume_tz=assume_tz,
            include_pivot_probs=include_pivot_probs,
            include_tb_probs=include_tb_probs,
            include_vix_features=include_vix_features,
            fill_missing_prob=float(fill_missing_prob),
            session_open=session_open,
            session_close=session_close,
            ga_predictor=ga_predictor,
            ga_probs_frame=None if ga_probs.empty else ga_probs,
            ga_probs_mode="xgb",
        )
        meta_path = _write_frame(
            meta_frame,
            path_no_ext=symbol_dir / "live_meta_matrix",
            fmt=out_fmt,
        )
        print(f"[replay] Dumped live meta matrix: {meta_path}")

        if trace_df is None or trace_df.empty:
            continue
        trace_symbol = trace_df[trace_df["symbol"].astype(str) == str(symbol)].copy()
        if trace_symbol.empty:
            continue
        trace_ts = pd.to_datetime(trace_symbol["timestamp"], utc=True, errors="coerce").dropna()
        if trace_ts.empty:
            continue
        trace_idx = pd.DatetimeIndex(trace_ts.unique()).sort_values()
        if meta_frame.index.tz is not None:
            trace_idx = trace_idx.tz_convert(meta_frame.index.tz)
        elif trace_idx.tz is not None:
            trace_idx = trace_idx.tz_localize(None)

        meta_on_trace = meta_frame.reindex(trace_idx)
        meta_trace_path = _write_frame(
            meta_on_trace,
            path_no_ext=symbol_dir / "live_meta_matrix_on_trace_ts",
            fmt=out_fmt,
        )
        print(f"[replay] Dumped live meta matrix on trace timestamps: {meta_trace_path}")

        if not x_tree.empty:
            x_tree_on_trace = x_tree.reindex(trace_idx)
            x_tree_trace_path = _write_frame(
                x_tree_on_trace,
                path_no_ext=symbol_dir / "live_ga_tree_matrix_on_trace_ts",
                fmt=out_fmt,
            )
            print(f"[replay] Dumped live GA tree matrix on trace timestamps: {x_tree_trace_path}")


def _make_close_handler(
    *,
    inference: LiveInferenceEngine,
    interval_minutes: int,
    print_close: bool,
    print_tz: str,
    quiet_inference_logs: bool,
    execution_latches: dict[str, DirectionExecutionLatch],
    order_policies: dict[str, OptionOrderPolicy] | None = None,
    trace_rows: list[dict] | None = None,
):
    def _handler(symbol: str, closed_bar: dict, buffer) -> None:
        if print_close:
            ts = _format_ts_local(closed_bar.get("timestamp"), tz=print_tz)
            print(
                f"{symbol} {interval_minutes}m closed: {ts} "
                f"o={closed_bar.get('open')} h={closed_bar.get('high')} "
                f"l={closed_bar.get('low')} c={closed_bar.get('close')} v={closed_bar.get('volume')}"
            )
        if order_policies is not None and symbol in order_policies:
            order_policies[symbol].on_15m_bar(closed_bar=closed_bar)
        action = inference.on_15m_close(df_1m=buffer.to_dataframe(), closed_bar=closed_bar)
        if action is not None:
            ts = _format_ts_local(closed_bar.get("timestamp"), tz=print_tz)
            raw_action = float(action)
            raw_pos = _action_to_position(raw_action)
            gate = execution_latches[symbol].step(raw_pos)
            exec_pos = int(gate.executed_pos)
            probs = inference.last_probs() or {}
            prob_sources = inference.last_prob_sources() or {}
            thresholds = inference.last_thresholds() or {}
            if not quiet_inference_logs:
                _print_meta_prob_log(
                    prefix=f"{symbol} meta [{ts}]:",
                    probs=probs,
                    thresholds=thresholds,
                )
                print(
                    f"{symbol} inference ts={ts} raw={raw_action:+.4f} raw_pos={raw_pos:+d} "
                    f"exec={exec_pos:+d} gate={gate.status}"
                )
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": closed_bar.get("timestamp"),
                        "open": closed_bar.get("open"),
                        "high": closed_bar.get("high"),
                        "low": closed_bar.get("low"),
                        "close": closed_bar.get("close"),
                        "volume": closed_bar.get("volume"),
                        "raw_action": raw_action,
                        "raw_pos": int(raw_pos),
                        "exec_pos": int(exec_pos),
                        "gate_status": str(gate.status),
                        "p_enter_long": probs.get("p_enter_long"),
                        "p_enter_short": probs.get("p_enter_short"),
                        "p_exit_long": probs.get("p_exit_long"),
                        "p_exit_short": probs.get("p_exit_short"),
                        "p_pivot_long": probs.get("p_pivot_long"),
                        "p_pivot_short": probs.get("p_pivot_short"),
                        "p_tb_long": probs.get("p_tb_long"),
                        "p_tb_short": probs.get("p_tb_short"),
                        "p_pivot_long_source": prob_sources.get("p_pivot_long_source"),
                        "p_pivot_short_source": prob_sources.get("p_pivot_short_source"),
                        "p_tb_long_source": prob_sources.get("p_tb_long_source"),
                        "p_tb_short_source": prob_sources.get("p_tb_short_source"),
                        "thr_enter_long": thresholds.get("enter_long"),
                        "thr_enter_short": thresholds.get("enter_short"),
                        "thr_exit_long": thresholds.get("exit_long"),
                        "thr_exit_short": thresholds.get("exit_short"),
                    }
                )
            if order_policies is not None and symbol in order_policies:
                policy_bar = dict(closed_bar)
                policy_bar.update({k: v for k, v in probs.items() if v is not None})
                policy_bar.update(
                    {
                        "thr_enter_long": thresholds.get("enter_long"),
                        "thr_enter_short": thresholds.get("enter_short"),
                        "thr_exit_long": thresholds.get("exit_long"),
                        "thr_exit_short": thresholds.get("exit_short"),
                    }
                )
                result = order_policies[symbol].on_decision(
                    action=float(exec_pos),
                    closed_bar=policy_bar,
                    update_bar_state=False,
                )
                event = str(result.get("event", "unknown"))
                if event not in {"hold", "no_change"}:
                    print(f"{symbol} order_policy event={event} details={result}")

    return _handler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay historical 1m bars through the live inference pipeline."
    )
    parser.add_argument(
        "--data-path",
        default="Data/raw/spy/inference_buffer_1m.parquet",
        help="CSV/Parquet with 1m bars.",
    )
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbols.")
    parser.add_argument("--start", default=None, help="Optional ISO start timestamp (UTC).")
    parser.add_argument("--end", default=None, help="Optional ISO end timestamp (UTC).")
    parser.add_argument("--regular-only", action="store_true", help="Filter to 9:30-16:00 ET.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between bars.")
    parser.add_argument("--max-bars", type=int, default=None, help="Max bars to replay.")
    parser.add_argument("--trace-out", default=None, help="Optional CSV path to save per-bar meta trace.")
    parser.add_argument("--plot-out", default=None, help="Optional PNG path to save entries/exits + probability plot.")
    parser.add_argument(
        "--dump-live-matrix-dir",
        default=None,
        help="Optional directory to dump live GA tree + meta feature matrices used for replay.",
    )
    parser.add_argument(
        "--dump-live-matrix-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format for --dump-live-matrix-dir artifacts.",
    )
    parser.add_argument(
        "--quiet-inference-logs",
        action="store_true",
        help="Suppress per-bar inference/probability logs and keep summary output only.",
    )
    parser.add_argument("--buffer-size", type=int, default=5000, help="Ring buffer size.")
    parser.add_argument("--print-1m", action="store_true", help="Print each 1m bar.")
    parser.add_argument("--print-15m", action="store_true", help="Print completed interval bars.")
    parser.add_argument("--resample-label", default="left", help="Resample label (left/right).")
    parser.add_argument("--resample-closed", default="left", help="Resample closed (left/right).")
    parser.add_argument("--tz", default="America/New_York", help="Timezone for resampling.")
    parser.add_argument("--assume-tz", default="UTC", help="Assume timezone for naive timestamps.")
    parser.add_argument("--inference-mode", choices=["meta", "ppo", "none"], default="meta", help="Inference controller to run.")
    parser.add_argument("--interval", type=int, default=10, help="Aggregation interval in minutes.")
    parser.add_argument("--model-path", default="Data/outputs/agent/ppo_model.pt", help="PPO model checkpoint.")
    parser.add_argument("--no-agent", action="store_true", help="Disable PPO inference.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions from policy (default is deterministic mean).")
    parser.add_argument("--device", default="auto", help="Device for inference (auto/cpu/cuda/mps).")
    parser.add_argument("--min-15m-bars", type=int, default=20, help="Minimum 15m bars before inference.")
    parser.add_argument("--no-pivot-probs", action="store_true", help="Disable pivot probability features.")
    parser.add_argument("--no-tb-probs", action="store_true", help="Disable triple-barrier probability features.")
    parser.add_argument("--fill-missing-prob", type=float, default=0.0, help="Value for missing prob features.")
    parser.add_argument("--session-open", default="09:30", help="Session open for time features.")
    parser.add_argument("--session-close", default="16:00", help="Session close for time features.")
    parser.add_argument("--ga-model-root", default="Data/models/ga_xgboost/10min", help="GA-XGB model root.")
    parser.add_argument("--ga-feature-list", default=None, help="Path to GA-XGB feature list txt.")
    parser.add_argument("--ga-dataset-name", default="10min", help="Dataset name for split-warmup lookup.")
    parser.add_argument(
        "--split-x-filename",
        default="X_10min_tree.parquet",
        help="Feature filename stem used to locate split indices for test warmup preload.",
    )
    parser.add_argument("--ga-pivot-label-dir", default="swing", help="Label dir for pivot GA-XGB models.")
    parser.add_argument("--ga-tb-label-dir", default="tb", help="Label dir for TB GA-XGB models.")
    parser.add_argument("--meta-model-root", default="Data/models/meta_xgboost/10min", help="Meta-XGB model root.")
    parser.add_argument(
        "--meta-entry-threshold",
        type=float,
        default=None,
        help="Optional execution threshold override for both meta long/short entries.",
    )
    parser.add_argument(
        "--meta-exit-threshold",
        type=float,
        default=None,
        help="Optional execution threshold override for both meta long/short exits.",
    )
    parser.add_argument("--meta-trail-activate-atr", type=float, default=2.0, help="Trail activation ATR used to build live exit context.")
    parser.add_argument("--meta-trail-atr", type=float, default=1.0, help="Base trail ATR used to build live exit context.")
    parser.add_argument("--meta-trail-atr-after-tp", type=float, default=0.8, help="Tightened trail ATR after TP is seen.")
    parser.add_argument("--meta-use-tp-to-tighten-trail", action=argparse.BooleanOptionalAction, default=True, help="Mirror training trail-tightening behavior in replay exit context.")
    parser.add_argument("--env-file", default=".env", help="Path to .env with Alpaca credentials.")
    parser.add_argument(
        "--enable-option-orders",
        action="store_true",
        help="Enable option order policy execution on each 15m inference action.",
    )
    parser.add_argument(
        "--option-order-qty",
        type=int,
        default=1,
        help="Fallback max contracts when account/quote sizing is unavailable.",
    )
    parser.add_argument(
        "--option-price-mode",
        default="ask",
        choices=["ask", "mid", "bid", "last", "mark"],
        help="Price input for sizing max contracts (ask is conservative, mid for sim).",
    )
    parser.add_argument(
        "--option-action-ema-alpha",
        type=float,
        default=0.85,
        help="EMA alpha for action smoothing (higher = smoother).",
    )
    parser.add_argument(
        "--option-rebalance-deadband",
        type=float,
        default=0.10,
        help="Ignore action changes smaller than this after smoothing.",
    )
    parser.add_argument(
        "--option-max-step-contracts",
        type=int,
        default=2,
        help="Max absolute signed-contract change per decision step.",
    )
    parser.add_argument(
        "--option-max-contracts-cap",
        type=int,
        default=0,
        help="Optional hard cap on max contracts (<=0 disables cap).",
    )
    parser.add_argument(
        "--option-atr-mult",
        type=float,
        default=1.0,
        help="ATR multiplier for target strike distance (default 1.0 ATR).",
    )
    parser.add_argument(
        "--option-dte-cutoff",
        default="14:00",
        help="Local HH:MM cutoff; before cutoff use 0DTE, otherwise 1DTE.",
    )
    parser.add_argument(
        "--simulate-orders",
        action="store_true",
        help="Do not submit to Alpaca; print intended order payloads only.",
    )
    parser.add_argument(
        "--option-no-close-on-flat",
        action="store_true",
        help="Do not auto close open option when agent action goes flat.",
    )
    parser.add_argument(
        "--option-no-close-on-flip",
        action="store_true",
        help="Do not auto close existing option before flipping side.",
    )
    parser.add_argument(
        "--no-prepend-split-test-warmup",
        action="store_true",
        help="Disable prepending test-split 1m warmup bars before replay data.",
    )
    parser.add_argument(
        "--exec-entry-confirm-bars",
        type=int,
        default=1,
        help="Consecutive bars required to confirm a new entry while flat.",
    )
    parser.add_argument(
        "--exec-exit-confirm-bars",
        type=int,
        default=2,
        help="Consecutive bars required to confirm exit/flip while in-position.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    df = _load_history(Path(args.data_path), assume_tz=args.assume_tz)
    if args.regular_only:
        df = _apply_regular_hours(df, tz=args.tz)

    if "symbol" not in df.columns:
        if len(symbols) != 1:
            raise ValueError("History data missing symbol column; use single --symbols or add symbol column.")
        df["symbol"] = symbols[0]
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df[df["symbol"].isin(symbols)]

    if args.start:
        start = pd.to_datetime(args.start, utc=True, errors="coerce")
        df = df[df["timestamp"] >= start]
    if args.end:
        end = pd.to_datetime(args.end, utc=True, errors="coerce")
        df = df[df["timestamp"] <= end]

    if not args.no_prepend_split_test_warmup:
        warm_frames = []
        for symbol in symbols:
            warm_df = _load_test_split_warmup_1m(
                symbol=symbol,
                dataset_name=args.ga_dataset_name,
                x_filename=args.split_x_filename,
            )
            if warm_df is None or warm_df.empty:
                continue
            if args.regular_only:
                warm_df = _apply_regular_hours(warm_df, tz=args.tz)
            warm_frames.append(warm_df)
        if warm_frames:
            warmup = pd.concat(warm_frames, axis=0, ignore_index=True)
            df = pd.concat([warmup, df], axis=0, ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp")
            # Prefer bars from explicit replay file when timestamps overlap.
            df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            print(
                f"[replay] Prepended split-test warmup bars: {len(warmup):,} "
                f"(combined rows: {len(df):,})"
            )
        else:
            print("[replay] Split-test warmup not found; replaying provided data only.")

    required = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"History data missing required columns: {missing}")

    df = df.sort_values("timestamp")

    if args.max_bars is not None:
        keep_n = int(args.max_bars)
        if keep_n > 0:
            df = df.tail(keep_n).copy()

    inference_mode = "none" if args.no_agent else str(args.inference_mode).strip().lower()
    agent = None
    if inference_mode != "none" and args.ga_feature_list is None and not (args.no_pivot_probs and args.no_tb_probs):
        try:
            from Data.load_data import get_ticker_processed_base_dir
            from Data.retrieve_data import normalize_ticker

            ticker = normalize_ticker(symbols[0])
            dataset_name = args.ga_dataset_name
            candidate = (
                get_ticker_processed_base_dir(ticker)
                / "datasets"
                / dataset_name
                / f"features_X_{dataset_name}_tree.txt"
            )
            if candidate.exists():
                args.ga_feature_list = str(candidate)
        except Exception:
            args.ga_feature_list = None

    if inference_mode != "none" and args.ga_feature_list is None and not (args.no_pivot_probs and args.no_tb_probs):
        print("[replay] Warning: GA-XGB feature list not found; pivot/TB probs will be filled with defaults.")

    if inference_mode == "ppo":
        agent = LivePPOAgent(
            model_path=args.model_path,
            deterministic=not args.stochastic,
            device=args.device,
            include_pivot_probs=not args.no_pivot_probs,
            include_tb_probs=not args.no_tb_probs,
            tz=args.tz or "America/New_York",
            assume_tz=args.assume_tz,
            session_open=args.session_open,
            session_close=args.session_close,
            min_15m_bars=args.min_15m_bars,
            fill_missing_prob=args.fill_missing_prob,
            ga_model_root=args.ga_model_root if args.ga_feature_list else None,
            ga_feature_list_path=args.ga_feature_list,
            ga_pivot_label_dir=args.ga_pivot_label_dir,
            ga_tb_label_dir=args.ga_tb_label_dir,
            resample_label=args.resample_label,
            resample_closed=args.resample_closed,
            label_timeframe_rule=f"{args.interval}min",
        )
    elif inference_mode == "meta":
        if int(args.interval) != 10:
            print(f"[replay] Warning: meta inference is trained for 10min bars; current --interval={args.interval}.")
        agent = LiveMetaXGBAgent(
            model_root=args.meta_model_root,
            ga_model_root=args.ga_model_root if args.ga_feature_list else None,
            ga_feature_list_path=args.ga_feature_list,
            include_pivot_probs=not args.no_pivot_probs,
            include_tb_probs=not args.no_tb_probs,
            pivot_label_dir=args.ga_pivot_label_dir,
            tb_label_dir=args.ga_tb_label_dir,
            tz=args.tz or "America/New_York",
            assume_tz=args.assume_tz,
            session_open=args.session_open,
            session_close=args.session_close,
            min_15m_bars=args.min_15m_bars,
            fill_missing_prob=args.fill_missing_prob,
            resample_label=args.resample_label,
            resample_closed=args.resample_closed,
            label_timeframe_rule=f"{args.interval}min",
            trail_activate_atr=float(args.meta_trail_activate_atr),
            trail_atr=float(args.meta_trail_atr),
            trail_atr_after_tp=float(args.meta_trail_atr_after_tp),
            use_tp_to_tighten_trail=bool(args.meta_use_tp_to_tighten_trail),
            entry_threshold_override=args.meta_entry_threshold,
            exit_threshold_override=args.meta_exit_threshold,
        )
        print(
            f"[replay] Meta-XGB inference enabled: model_root={args.meta_model_root} "
            f"timeframe={args.interval}min"
        )

    inference = LiveInferenceEngine(
        agent=agent,
        label=args.resample_label,
        closed=args.resample_closed,
        rule=f"{args.interval}min",
        tz=args.tz,
        assume_tz=args.assume_tz,
    )
    execution_latches: dict[str, DirectionExecutionLatch] = {
        symbol: DirectionExecutionLatch(
            entry_confirm_bars=max(1, int(args.exec_entry_confirm_bars)),
            exit_confirm_bars=max(1, int(args.exec_exit_confirm_bars)),
            initial_position=0,
        )
        for symbol in symbols
    }

    order_policies: dict[str, OptionOrderPolicy] | None = None
    if args.enable_option_orders:
        order_policies = {}
        for symbol in symbols:
            cfg = OptionOrderPolicyConfig(
                underlying=symbol,
                env_file=args.env_file,
                tz_name=args.tz or "America/New_York",
                atr_multiplier=float(args.option_atr_mult),
                dte_cutoff_hhmm=args.option_dte_cutoff,
                qty=int(args.option_order_qty),
                close_on_flat=not args.option_no_close_on_flat,
                close_on_flip=not args.option_no_close_on_flip,
                submit_orders=not args.simulate_orders,
                ema_alpha=float(args.option_action_ema_alpha),
                rebalance_deadband=float(args.option_rebalance_deadband),
                max_step_contracts=int(args.option_max_step_contracts),
                price_mode=str(args.option_price_mode),
                max_contracts_fallback=int(args.option_order_qty),
                max_contracts_cap=int(args.option_max_contracts_cap),
                meta_trailing_stop_enabled=True,
                meta_trail_activate_atr=float(args.meta_trail_activate_atr),
                meta_trail_atr=float(args.meta_trail_atr),
                meta_trail_atr_after_tp=float(args.meta_trail_atr_after_tp),
                meta_use_tp_to_tighten_trail=bool(args.meta_use_tp_to_tighten_trail),
            )
            order_policies[symbol] = OptionOrderPolicy(cfg)
        mode = "SIMULATED" if args.simulate_orders else "LIVE"
        print(f"[replay] Option order policy enabled ({mode}) for symbols: {', '.join(symbols)}")

    trace_rows: list[dict] | None = [] if (args.trace_out or args.plot_out) else None

    processor = LiveBarProcessor(
        interval_minutes=args.interval,
        buffer_size=args.buffer_size,
        agg_label=args.resample_label,
        on_1m=(
            _make_1m_handler(
                print_tz=args.tz or "America/New_York",
                print_1m=bool(args.print_1m),
                order_policies=order_policies,
            )
            if (args.print_1m or order_policies is not None)
            else None
        ),
        on_15m_close=_make_close_handler(
            inference=inference,
            interval_minutes=int(args.interval),
            print_close=args.print_15m,
            print_tz=args.tz or "America/New_York",
            quiet_inference_logs=bool(args.quiet_inference_logs),
            execution_latches=execution_latches,
            order_policies=order_policies,
            trace_rows=trace_rows,
        ),
    )

    count = 0
    for row in df.itertuples(index=False):
        bar = {
            "symbol": getattr(row, "symbol"),
            "timestamp": getattr(row, "timestamp"),
            "open": float(getattr(row, "open")),
            "high": float(getattr(row, "high")),
            "low": float(getattr(row, "low")),
            "close": float(getattr(row, "close")),
            "volume": float(getattr(row, "volume")),
        }
        processor.handle_bar(bar)
        count += 1
        if args.max_bars and count >= args.max_bars:
            break
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"[replay] Done. Bars processed: {count:,}.")
    trace_df: pd.DataFrame | None = None
    if (args.trace_out or args.plot_out) and isinstance(trace_rows, list):
        trace_df = pd.DataFrame(trace_rows)
        _print_trace_prob_diagnostics(trace_df)
        if args.trace_out:
            trace_path = Path(args.trace_out)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_df.to_csv(trace_path, index=False)
            print(f"[replay] Saved trace: {trace_path}")
        if args.plot_out:
            plot_path = Path(args.plot_out)
            _save_trace_plot(trace_df=trace_df, save_path=plot_path, tz=args.tz or "America/New_York")
            print(f"[replay] Saved plot: {plot_path}")

    if args.dump_live_matrix_dir:
        _dump_live_inference_matrices(
            raw_df=df,
            symbols=symbols,
            agent=agent,
            trace_df=trace_df,
            interval_minutes=int(args.interval),
            resample_label=args.resample_label,
            resample_closed=args.resample_closed,
            tz=args.tz,
            assume_tz=args.assume_tz,
            session_open=args.session_open,
            session_close=args.session_close,
            include_pivot_probs=not args.no_pivot_probs,
            include_tb_probs=not args.no_tb_probs,
            fill_missing_prob=float(args.fill_missing_prob),
            out_dir=Path(args.dump_live_matrix_dir),
            out_fmt=str(args.dump_live_matrix_format),
        )


if __name__ == "__main__":
    main()
