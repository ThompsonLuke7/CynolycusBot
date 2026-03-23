from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 10m independent meta candles and probabilities with 1m breakout-entry events."
    )
    parser.add_argument(
        "--trace",
        default="Data/inference/spy/10min/meta/meta_trace_independent_1m_entry_last_month.csv",
        help="10m independent meta trace CSV.",
    )
    parser.add_argument(
        "--events",
        default="Data/inference/spy/10min/meta/meta_events_independent_1m_entry_last_month.csv",
        help="1m execution events CSV.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to plot.")
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp.")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument("--sessions-per-fig", type=int, default=4, help="Sessions per output PNG.")
    parser.add_argument(
        "--out-dir",
        default="Data/inference/spy/10min/plots/independent_1m_entry_sessions",
        help="Directory for output PNGs.",
    )
    return parser.parse_args()


def _load_inputs(
    trace_path: Path,
    events_path: Path,
    *,
    symbol: str,
    start: str | None,
    end: str | None,
    tz: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = pd.read_csv(trace_path)
    trace["timestamp"] = pd.to_datetime(trace["timestamp"], utc=True, errors="coerce")
    trace = trace.dropna(subset=["timestamp"]).copy()
    trace = trace[trace["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()

    events = pd.read_csv(events_path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["timestamp"]).copy()
    events = events[events["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()

    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        trace = trace[trace["timestamp"] >= start_ts]
        events = events[events["timestamp"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
        trace = trace[trace["timestamp"] <= end_ts]
        events = events[events["timestamp"] <= end_ts]

    if trace.empty:
        raise ValueError("No trace rows remain after filtering.")

    trace["ts_local"] = trace["timestamp"].dt.tz_convert(tz)
    trace["session_date"] = trace["ts_local"].dt.normalize()
    events["ts_local"] = events["timestamp"].dt.tz_convert(tz)
    events["session_date"] = events["ts_local"].dt.normalize()
    return trace.sort_values("timestamp"), events.sort_values("timestamp")


def _plot_sessions(
    trace: pd.DataFrame,
    events: pd.DataFrame,
    *,
    out_dir: Path,
    sessions_per_fig: int,
    tz: str,
    symbol: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dates = sorted(trace["session_date"].dropna().unique().tolist())
    if not session_dates:
        raise ValueError("No sessions found to plot.")

    outputs: list[Path] = []
    chunks = int(ceil(len(session_dates) / max(1, sessions_per_fig)))
    candle_width = 8.0 / (24 * 60)

    marker_specs = {
        "enter_long": ("#2E7D32", "^", "enter long"),
        "exit_long": ("#8c564b", "v", "exit long"),
        "enter_short": ("#C62828", "v", "enter short"),
        "exit_short": ("#9467bd", "^", "exit short"),
    }

    for chunk_idx in range(chunks):
        dates = session_dates[chunk_idx * sessions_per_fig:(chunk_idx + 1) * sessions_per_fig]
        n = len(dates)
        fig, axes = plt.subplots(
            n * 2,
            1,
            figsize=(18, max(7.0, 5.0 * n)),
            sharex=False,
            gridspec_kw={"height_ratios": [2.3, 1.0] * n},
        )
        if n == 1:
            axes = [axes[0], axes[1]]

        for idx, session_date in enumerate(dates):
            ax_price = axes[idx * 2]
            ax_prob = axes[idx * 2 + 1]
            met = trace[trace["session_date"] == session_date].copy().sort_values("ts_local")
            evt = events[events["session_date"] == session_date].copy().sort_values("ts_local")
            if met.empty:
                continue

            x = mdates.date2num(met["ts_local"].to_list())
            open_ = pd.to_numeric(met["open"], errors="coerce").to_numpy()
            high = pd.to_numeric(met["high"], errors="coerce").to_numpy()
            low = pd.to_numeric(met["low"], errors="coerce").to_numpy()
            close = pd.to_numeric(met["close"], errors="coerce").to_numpy()
            up = close >= open_
            down = ~up

            ax_price.vlines(x, low, high, color="#4a4a4a", linewidth=0.8, zorder=1)
            ax_price.bar(
                x[up],
                close[up] - open_[up],
                width=candle_width,
                bottom=open_[up],
                color="#1976D2",
                edgecolor="none",
                zorder=1.2,
                label="10m bull",
            )
            ax_price.bar(
                x[down],
                close[down] - open_[down],
                width=candle_width,
                bottom=open_[down],
                color="#E53935",
                edgecolor="none",
                zorder=1.2,
                label="10m bear",
            )

            if not evt.empty:
                ex = mdates.date2num(evt["ts_local"].to_list())
                ey = pd.to_numeric(evt["price"], errors="coerce").to_numpy()
                for event, (color, marker, label) in marker_specs.items():
                    mask = evt["event"].astype(str).eq(event).to_numpy()
                    if mask.any():
                        ax_price.scatter(ex[mask], ey[mask], color=color, marker=marker, s=48, label=label, zorder=2.2)

            ax_price.set_title(f"{symbol} | {session_date.date()} | 10m bars + 1m breakout execution")
            ax_price.set_ylabel("Price")
            ax_price.grid(True, alpha=0.25)
            handles, labels = ax_price.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_price.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=8, ncol=3)

            mx = mdates.date2num(met["ts_local"].to_list())
            for col, color, label in (
                ("p_enter_long", "#2ca02c", "p_enter_long"),
                ("p_enter_short", "#d62728", "p_enter_short"),
                ("p_exit_long", "#17becf", "p_exit_long"),
                ("p_exit_short", "#ff7f0e", "p_exit_short"),
            ):
                if col in met.columns:
                    y = pd.to_numeric(met[col], errors="coerce")
                    if y.notna().any():
                        ax_prob.step(mx, y.to_numpy(), where="post", color=color, linewidth=1.2, label=label)

            for col, color, label in (
                ("thr_enter_long", "#2ca02c", "thr_enter_long"),
                ("thr_enter_short", "#d62728", "thr_enter_short"),
                ("thr_exit_long", "#17becf", "thr_exit_long"),
                ("thr_exit_short", "#ff7f0e", "thr_exit_short"),
            ):
                if col in met.columns:
                    y = pd.to_numeric(met[col], errors="coerce").dropna()
                    if not y.empty:
                        ax_prob.axhline(float(y.iloc[-1]), color=color, linestyle="--", linewidth=1.0, alpha=0.85, label=label)

            ax_prob.set_ylim(-0.02, 1.02)
            ax_prob.set_ylabel("Probability")
            ax_prob.set_xlabel(f"Session Time ({tz})")
            ax_prob.grid(True, alpha=0.25)
            handles, labels = ax_prob.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_prob.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8, ncol=4)

            session_tz = met["ts_local"].dt.tz
            locator = mdates.HourLocator(interval=1, tz=session_tz)
            formatter = mdates.DateFormatter("%H:%M", tz=session_tz)
            ax_price.xaxis.set_major_locator(locator)
            ax_price.xaxis.set_major_formatter(formatter)
            ax_prob.xaxis.set_major_locator(locator)
            ax_prob.xaxis.set_major_formatter(formatter)
            ax_price.tick_params(axis="x", labelbottom=False)
            ax_price.set_xlim(x.min() - candle_width * 2, x.max() + candle_width * 2)
            ax_prob.set_xlim(x.min() - candle_width * 2, x.max() + candle_width * 2)

        fig.tight_layout()
        out_path = out_dir / f"{symbol.lower()}_independent_1m_entry_part{chunk_idx + 1}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def main() -> None:
    args = _parse_args()
    trace, events = _load_inputs(
        Path(args.trace),
        Path(args.events),
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        tz=args.tz,
    )
    outputs = _plot_sessions(
        trace,
        events,
        out_dir=Path(args.out_dir),
        sessions_per_fig=max(1, int(args.sessions_per_fig)),
        tz=args.tz,
        symbol=args.symbol,
    )
    print("Generated plots:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
