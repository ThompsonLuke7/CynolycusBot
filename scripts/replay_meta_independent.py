from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay cached meta matrix with independent long/short sides and recomputed exit probabilities."
    )
    parser.add_argument(
        "--meta-matrix",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts.parquet",
        help="Cached meta matrix parquet.",
    )
    parser.add_argument(
        "--model-root",
        default="Data/models/meta_xgboost/10min",
        help="Meta model root directory.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol label for output.")
    parser.add_argument("--start", default="2026-02-13T00:00:00Z", help="UTC start timestamp.")
    parser.add_argument("--end", default="2026-03-13T23:59:59Z", help="UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument(
        "--trace-out",
        default="Data/inference/spy/10min/meta/meta_trace_independent_last_month.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot-out",
        default="Data/inference/spy/10min/plots/meta_entries_exits_probs_last_month_independent_replay.png",
        help="Output PNG path.",
    )
    parser.add_argument("--entry-threshold", type=float, default=None, help="Optional override for both entry thresholds.")
    parser.add_argument("--exit-threshold", type=float, default=None, help="Optional override for both exit thresholds.")
    parser.add_argument(
        "--min-hold-bars",
        type=int,
        default=2,
        help="Minimum number of 10m bars to hold a side before honoring its exit model.",
    )
    parser.add_argument(
        "--exit-entry-delta",
        type=float,
        default=0.15,
        help="Required exit-vs-entry dominance margin when the same-side entry probability is still above threshold.",
    )
    return parser.parse_args()


def _normalize_bounds(start: str | None, end: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    out_start = pd.Timestamp(start) if start else None
    out_end = pd.Timestamp(end) if end else None
    if out_start is not None and out_start.tzinfo is None:
        out_start = out_start.tz_localize("UTC")
    elif out_start is not None:
        out_start = out_start.tz_convert("UTC")
    if out_end is not None and out_end.tzinfo is None:
        out_end = out_end.tz_localize("UTC")
    elif out_end is not None:
        out_end = out_end.tz_convert("UTC")
    return out_start, out_end


def _load_meta_matrix(path: Path, *, start: pd.Timestamp | None, end: pd.Timestamp | None, tz: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.loc[ts.notna()].copy()
        df.index = ts[ts.notna()]
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Meta matrix at {path} has no DatetimeIndex or timestamp column.")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz).tz_convert("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    if df.empty:
        raise ValueError("Meta matrix is empty after date filtering.")
    return df


def _score_exit(agent: LiveMetaXGBAgent, row: pd.Series, *, side: str) -> float:
    ctx_row = agent._annotate_current_context(row)
    exit_df = pd.DataFrame([ctx_row], index=[row.name])
    if side == "long":
        return float(agent._exit_long.predict_row(exit_df, target_ts=row.name)) if agent._state.position > 0 else float("nan")
    return float(agent._exit_short.predict_row(exit_df, target_ts=row.name)) if agent._state.position < 0 else float("nan")


def _run_independent_replay(
    *,
    meta_df: pd.DataFrame,
    model_root: Path,
    symbol: str,
    entry_threshold: float | None,
    exit_threshold: float | None,
    min_hold_bars: int,
    exit_entry_delta: float,
    tz: str,
) -> pd.DataFrame:
    long_agent = LiveMetaXGBAgent(
        model_root=model_root,
        precomputed_base_frame=meta_df,
        entry_threshold_override=entry_threshold,
        exit_threshold_override=exit_threshold,
    )
    short_agent = LiveMetaXGBAgent(
        model_root=model_root,
        precomputed_base_frame=meta_df,
        entry_threshold_override=entry_threshold,
        exit_threshold_override=exit_threshold,
    )

    entry_long_probs = long_agent._entry_long.predict_frame(meta_df)
    entry_short_probs = long_agent._entry_short.predict_frame(meta_df)
    thresholds = long_agent.last_thresholds() or {
        "enter_long": np.nan,
        "enter_short": np.nan,
        "exit_long": np.nan,
        "exit_short": np.nan,
    }

    rows: list[dict[str, object]] = []
    long_active = False
    short_active = False
    long_bars_held = -1
    short_bars_held = -1

    for idx, (_, row) in enumerate(meta_df.iterrows()):
        ts = pd.Timestamp(row.name)
        p_enter_long = float(entry_long_probs[idx]) if idx < entry_long_probs.size else float("nan")
        p_enter_short = float(entry_short_probs[idx]) if idx < entry_short_probs.size else float("nan")

        work_row = row.copy()
        work_row["p_enter_long_oof"] = p_enter_long
        work_row["p_enter_short_oof"] = p_enter_short

        p_exit_long = _score_exit(long_agent, work_row, side="long") if long_active else float("nan")
        p_exit_short = _score_exit(short_agent, work_row, side="short") if short_active else float("nan")

        long_exit_threshold_hit = bool(long_active and np.isfinite(p_exit_long) and p_exit_long >= float(thresholds["exit_long"]))
        short_exit_threshold_hit = bool(short_active and np.isfinite(p_exit_short) and p_exit_short >= float(thresholds["exit_short"]))

        long_hold_ready = bool(long_active and long_bars_held >= int(min_hold_bars))
        short_hold_ready = bool(short_active and short_bars_held >= int(min_hold_bars))

        long_entry_still_supports = bool(
            np.isfinite(p_enter_long)
            and p_enter_long >= float(thresholds["enter_long"])
            and (not np.isfinite(p_exit_long) or (p_exit_long - p_enter_long) < float(exit_entry_delta))
        )
        short_entry_still_supports = bool(
            np.isfinite(p_enter_short)
            and p_enter_short >= float(thresholds["enter_short"])
            and (not np.isfinite(p_exit_short) or (p_exit_short - p_enter_short) < float(exit_entry_delta))
        )

        do_exit_long = bool(long_exit_threshold_hit and long_hold_ready and not long_entry_still_supports)
        do_exit_short = bool(short_exit_threshold_hit and short_hold_ready and not short_entry_still_supports)
        do_entry_long = bool((not long_active) and np.isfinite(p_enter_long) and p_enter_long >= float(thresholds["enter_long"]))
        do_entry_short = bool((not short_active) and np.isfinite(p_enter_short) and p_enter_short >= float(thresholds["enter_short"]))

        next_long_active = bool((long_active and not do_exit_long) or do_entry_long)
        next_short_active = bool((short_active and not do_exit_short) or do_entry_short)

        long_action = 1 if next_long_active else 0
        short_action = -1 if next_short_active else 0
        long_agent._advance_state(action=long_action, row=work_row)
        short_agent._advance_state(action=short_action, row=work_row)
        long_active = next_long_active
        short_active = next_short_active
        if do_entry_long:
            long_bars_held = 0
        elif long_active:
            long_bars_held = max(0, long_bars_held + 1)
        else:
            long_bars_held = -1
        if do_entry_short:
            short_bars_held = 0
        elif short_active:
            short_bars_held = max(0, short_bars_held + 1)
        else:
            short_bars_held = -1

        rows.append(
            {
                "symbol": str(symbol),
                "timestamp": ts,
                "open": float(row.get("open", np.nan)),
                "high": float(row.get("high", np.nan)),
                "low": float(row.get("low", np.nan)),
                "close": float(row.get("close", np.nan)),
                "volume": float(row.get("volume", np.nan)),
                "p_pivot_long": float(row.get("p_pivot_long", np.nan)),
                "p_pivot_short": float(row.get("p_pivot_short", np.nan)),
                "p_tb_long": float(row.get("p_tb_long", np.nan)),
                "p_tb_short": float(row.get("p_tb_short", np.nan)),
                "p_enter_long": p_enter_long,
                "p_enter_short": p_enter_short,
                "p_exit_long": p_exit_long,
                "p_exit_short": p_exit_short,
                "thr_enter_long": float(thresholds["enter_long"]),
                "thr_enter_short": float(thresholds["enter_short"]),
                "thr_exit_long": float(thresholds["exit_long"]),
                "thr_exit_short": float(thresholds["exit_short"]),
                "long_bars_held": int(long_bars_held) if long_bars_held >= 0 else 0,
                "short_bars_held": int(short_bars_held) if short_bars_held >= 0 else 0,
                "long_exit_threshold_hit": bool(long_exit_threshold_hit),
                "short_exit_threshold_hit": bool(short_exit_threshold_hit),
                "long_entry_still_supports": bool(long_entry_still_supports),
                "short_entry_still_supports": bool(short_entry_still_supports),
                "ind_entry_long": bool(do_entry_long),
                "ind_exit_long": bool(do_exit_long),
                "ind_entry_short": bool(do_entry_short),
                "ind_exit_short": bool(do_exit_short),
                "ind_long_active": int(long_active),
                "ind_short_active": int(short_active),
            }
        )

    trace = pd.DataFrame(rows)
    trace["timestamp"] = pd.to_datetime(trace["timestamp"], utc=True, errors="coerce")
    trace["ts_local"] = trace["timestamp"].dt.tz_convert(tz)
    return trace


def _save_plot(trace: pd.DataFrame, *, save_path: Path, symbol: str) -> None:
    fig, (ax_price, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(18, 10),
        sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    df = trace.copy().sort_values("timestamp")
    x = np.arange(len(df), dtype=float)
    open_ = pd.to_numeric(df["open"], errors="coerce").to_numpy()
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy()
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy()
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy()
    up = close >= open_
    down = ~up
    candle_width = 0.82

    ax_price.vlines(x, low, high, color="#4a4a4a", linewidth=0.8, zorder=1)
    ax_price.bar(x[up], close[up] - open_[up], width=candle_width, bottom=open_[up], color="#1976D2", edgecolor="none", zorder=1.2, label="bull candle")
    ax_price.bar(x[down], close[down] - open_[down], width=candle_width, bottom=open_[down], color="#E53935", edgecolor="none", zorder=1.2, label="bear candle")

    session_change = df["ts_local"].dt.normalize().ne(df["ts_local"].dt.normalize().shift(1)).fillna(False)
    session_starts = np.flatnonzero(session_change.to_numpy())
    for idx in session_starts[1:]:
        ax_price.axvline(idx - 0.5, color="#cfcfcf", linestyle="--", linewidth=0.9, alpha=0.7, zorder=0.5)
        ax_prob.axvline(idx - 0.5, color="#cfcfcf", linestyle="--", linewidth=0.9, alpha=0.7, zorder=0.5)

    spread = high - low
    offset = pd.Series(spread).replace(0, pd.NA).dropna().median()
    if pd.isna(offset) or offset <= 0:
        offset = max(close) * 0.0015
    y_enter_long = low - offset * 1.2
    y_exit_long = high + offset * 0.8
    y_enter_short = high + offset * 1.2
    y_exit_short = low - offset * 0.8

    for col, yvals, color, marker, label in (
        ("ind_entry_long", y_enter_long, "#2E7D32", "^", "enter long"),
        ("ind_exit_long", y_exit_long, "#8c564b", "v", "exit long"),
        ("ind_entry_short", y_enter_short, "#C62828", "v", "enter short"),
        ("ind_exit_short", y_exit_short, "#9467bd", "^", "exit short"),
    ):
        mask = df[col].fillna(False).to_numpy()
        if mask.any():
            ax_price.scatter(x[mask], yvals[mask], color=color, marker=marker, s=58, label=label, zorder=2.0)

    ax_price.set_title(f"{symbol} | independent meta entries/exits")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.25)
    handles, labels = ax_price.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax_price.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=9)

    for col, color, label in (
        ("p_enter_long", "#2ca02c", "p_enter_long"),
        ("p_enter_short", "#d62728", "p_enter_short"),
        ("p_exit_long", "#17becf", "p_exit_long"),
        ("p_exit_short", "#ff7f0e", "p_exit_short"),
    ):
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().any():
            ax_prob.plot(x, y, color=color, linewidth=1.3, label=label)

    for col, color, label in (
        ("thr_enter_long", "#2ca02c", "thr_enter_long"),
        ("thr_enter_short", "#d62728", "thr_enter_short"),
        ("thr_exit_long", "#17becf", "thr_exit_long"),
        ("thr_exit_short", "#ff7f0e", "thr_exit_short"),
    ):
        y = pd.to_numeric(df[col], errors="coerce").dropna()
        if not y.empty:
            ax_prob.axhline(float(y.iloc[-1]), color=color, linestyle="--", linewidth=1.0, alpha=0.85, label=label)

    ax_prob.set_title(f"{symbol} | meta probabilities")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlabel("Session")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.grid(True, alpha=0.25)
    handles, labels = ax_prob.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax_prob.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)
    ax_price.tick_params(axis="x", labelbottom=False)
    ax_price.set_xlim(x.min() - 0.8, x.max() + 0.8)
    ax_prob.set_xlim(x.min() - 0.8, x.max() + 0.8)

    if session_starts.size:
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        sessions = df["ts_local"].dt.normalize()
        for start_idx in session_starts:
            tick_positions.append(float(start_idx))
            tick_labels.append(sessions.iloc[start_idx].strftime("%Y-%m-%d"))
        max_ticks = 12
        if len(tick_positions) > max_ticks:
            stride = int(np.ceil(len(tick_positions) / max_ticks))
            tick_positions = tick_positions[::stride]
            tick_labels = tick_labels[::stride]
        ax_prob.set_xticks(tick_positions)
        ax_prob.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    start, end = _normalize_bounds(args.start, args.end)
    meta_df = _load_meta_matrix(Path(args.meta_matrix), start=start, end=end, tz=args.tz)
    trace = _run_independent_replay(
        meta_df=meta_df,
        model_root=Path(args.model_root),
        symbol=args.symbol,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        min_hold_bars=max(0, int(args.min_hold_bars)),
        exit_entry_delta=float(args.exit_entry_delta),
        tz=args.tz,
    )
    trace_out = Path(args.trace_out)
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(trace_out, index=False)
    _save_plot(trace, save_path=Path(args.plot_out), symbol=args.symbol)
    print(trace_out)
    print(Path(args.plot_out))


if __name__ == "__main__":
    main()
