from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 10m meta trace with independent long/short side state derived from entry/exit probabilities."
    )
    parser.add_argument(
        "--trace",
        default="Data/inference/spy/10min/meta/meta_trace_warmup.csv",
        help="Meta trace CSV containing 10m probabilities.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to plot.")
    parser.add_argument("--start", default="2026-02-13T00:00:00Z", help="UTC start timestamp.")
    parser.add_argument("--end", default="2026-03-13T23:59:59Z", help="UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument(
        "--out",
        default="Data/inference/spy/10min/plots/meta_entries_exits_probs_last_month_independent.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def _load_trace(path: Path, *, symbol: str, start: str | None, end: str | None, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df[df["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    if df.empty:
        raise ValueError("No trace rows remain after filtering.")
    df["ts_local"] = df["timestamp"].dt.tz_convert(tz)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").copy()
    return df


def _derive_independent_markers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    long_active = False
    short_active = False
    entry_long = []
    exit_long = []
    entry_short = []
    exit_short = []
    long_state = []
    short_state = []

    for row in out.itertuples(index=False):
        p_enter_long = pd.to_numeric(pd.Series([getattr(row, "p_enter_long", float("nan"))]), errors="coerce").iloc[0]
        p_enter_short = pd.to_numeric(pd.Series([getattr(row, "p_enter_short", float("nan"))]), errors="coerce").iloc[0]
        p_exit_long = pd.to_numeric(pd.Series([getattr(row, "p_exit_long", float("nan"))]), errors="coerce").iloc[0]
        p_exit_short = pd.to_numeric(pd.Series([getattr(row, "p_exit_short", float("nan"))]), errors="coerce").iloc[0]
        thr_enter_long = pd.to_numeric(pd.Series([getattr(row, "thr_enter_long", float("nan"))]), errors="coerce").iloc[0]
        thr_enter_short = pd.to_numeric(pd.Series([getattr(row, "thr_enter_short", float("nan"))]), errors="coerce").iloc[0]
        thr_exit_long = pd.to_numeric(pd.Series([getattr(row, "thr_exit_long", float("nan"))]), errors="coerce").iloc[0]
        thr_exit_short = pd.to_numeric(pd.Series([getattr(row, "thr_exit_short", float("nan"))]), errors="coerce").iloc[0]

        do_exit_long = bool(long_active and pd.notna(p_exit_long) and pd.notna(thr_exit_long) and p_exit_long >= thr_exit_long)
        do_exit_short = bool(short_active and pd.notna(p_exit_short) and pd.notna(thr_exit_short) and p_exit_short >= thr_exit_short)
        do_entry_long = bool((not long_active) and pd.notna(p_enter_long) and pd.notna(thr_enter_long) and p_enter_long >= thr_enter_long)
        do_entry_short = bool((not short_active) and pd.notna(p_enter_short) and pd.notna(thr_enter_short) and p_enter_short >= thr_enter_short)

        # For each side independently, an active side only listens to its exit model.
        if do_exit_long:
            long_active = False
        if do_exit_short:
            short_active = False
        if do_entry_long and not do_exit_long:
            long_active = True
        if do_entry_short and not do_exit_short:
            short_active = True

        entry_long.append(do_entry_long and not do_exit_long)
        exit_long.append(do_exit_long)
        entry_short.append(do_entry_short and not do_exit_short)
        exit_short.append(do_exit_short)
        long_state.append(int(long_active))
        short_state.append(int(short_active))

    out["ind_entry_long"] = entry_long
    out["ind_exit_long"] = exit_long
    out["ind_entry_short"] = entry_short
    out["ind_exit_short"] = exit_short
    out["ind_long_active"] = long_state
    out["ind_short_active"] = short_state
    return out


def _save_plot(df: pd.DataFrame, *, save_path: Path, symbol: str, tz: str) -> None:
    fig, (ax_price, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(18, 10),
        sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    x = np.arange(len(df), dtype=float)
    open_ = pd.to_numeric(df["open"], errors="coerce").to_numpy()
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy()
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy()
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy()
    up = close >= open_
    down = ~up
    candle_width = 0.82

    ax_price.vlines(x, low, high, color="#4a4a4a", linewidth=0.8, zorder=1)
    ax_price.bar(
        x[up],
        close[up] - open_[up],
        width=candle_width,
        bottom=open_[up],
        color="#1976D2",
        edgecolor="none",
        zorder=1.2,
        label="bull candle",
    )
    ax_price.bar(
        x[down],
        close[down] - open_[down],
        width=candle_width,
        bottom=open_[down],
        color="#E53935",
        edgecolor="none",
        zorder=1.2,
        label="bear candle",
    )

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
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce")
            if y.notna().any():
                ax_prob.plot(x, y, color=color, linewidth=1.3, label=label)

    for col, color, label in (
        ("thr_enter_long", "#2ca02c", "thr_enter_long"),
        ("thr_enter_short", "#d62728", "thr_enter_short"),
        ("thr_exit_long", "#17becf", "thr_exit_long"),
        ("thr_exit_short", "#ff7f0e", "thr_exit_short"),
    ):
        if col in df.columns:
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
    df = _load_trace(
        Path(args.trace),
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        tz=args.tz,
    )
    df = _derive_independent_markers(df)
    _save_plot(df, save_path=Path(args.out), symbol=args.symbol, tz=args.tz)
    print(Path(args.out))


if __name__ == "__main__":
    main()
