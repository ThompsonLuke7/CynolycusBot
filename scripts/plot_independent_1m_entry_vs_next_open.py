from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 1m candles with actual breakout entries versus naive next-10m-open entries."
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
    parser.add_argument(
        "--one-min-data",
        default="Data/raw/spy/spy_intraday_1min.parquet",
        help="Raw 1m parquet.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to plot.")
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp.")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument("--sessions-per-fig", type=int, default=2, help="Sessions per output PNG.")
    parser.add_argument(
        "--out-dir",
        default="Data/inference/spy/10min/plots/independent_1m_entry_vs_next_open",
        help="Directory for output PNGs.",
    )
    return parser.parse_args()


def _load_inputs(
    trace_path: Path,
    events_path: Path,
    one_min_path: Path,
    *,
    symbol: str,
    start: str | None,
    end: str | None,
    tz: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace = pd.read_csv(trace_path)
    trace["timestamp"] = pd.to_datetime(trace["timestamp"], utc=True, errors="coerce")
    trace = trace.dropna(subset=["timestamp"]).copy()
    trace = trace[trace["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()

    events = pd.read_csv(events_path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["timestamp"]).copy()
    events = events[events["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()

    one = pd.read_parquet(one_min_path)
    one["timestamp"] = pd.to_datetime(one["timestamp"], utc=True, errors="coerce")
    one = one.dropna(subset=["timestamp"]).copy()
    one = one[one["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()

    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        trace = trace[trace["timestamp"] >= start_ts]
        events = events[events["timestamp"] >= start_ts]
        one = one[one["timestamp"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
        trace = trace[trace["timestamp"] <= end_ts]
        events = events[events["timestamp"] <= end_ts]
        one = one[one["timestamp"] <= end_ts]

    if trace.empty:
        raise ValueError("No trace rows remain after filtering.")
    if one.empty:
        raise ValueError("No 1m rows remain after filtering.")

    for frame in (trace, events, one):
        frame["ts_local"] = frame["timestamp"].dt.tz_convert(tz)
        frame["session_date"] = frame["ts_local"].dt.normalize()

    return trace.sort_values("timestamp"), events.sort_values("timestamp"), one.sort_values("timestamp")


def _derive_entry_diagnostics(trace: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    entry_events = events[events["event"].isin(["enter_long", "enter_short"])].copy()
    trace = trace.sort_values("timestamp").reset_index(drop=True)

    for _, event in entry_events.iterrows():
        side = "long" if event["event"] == "enter_long" else "short"
        valid_col = f"{side}_valid_signal"
        ref_col = "long_ref_high" if side == "long" else "short_ref_low"
        event_ts = event["timestamp"]
        event_bar_ts = event_ts.floor("10min")

        cand = trace[trace["timestamp"] <= event_bar_ts].copy()
        cand = cand[cand[valid_col].fillna(False)]
        if cand.empty:
            continue

        # Use the latest valid signal bar that was still supporting the side when the entry fired.
        signal_row = cand.iloc[-1]
        signal_ts = pd.Timestamp(signal_row["timestamp"])
        decision_ts = signal_ts + pd.Timedelta(minutes=10)
        ref_price = float(signal_row[ref_col]) if pd.notna(signal_row[ref_col]) else float(event["price"])

        next_rows = trace[trace["timestamp"] > signal_ts].copy()
        next_open_ts = pd.NaT
        next_open_price = float("nan")
        if not next_rows.empty:
            next_open_ts = pd.Timestamp(next_rows.iloc[0]["timestamp"])
            next_open_price = float(next_rows.iloc[0]["open"])

        rows.append(
            {
                "symbol": event["symbol"],
                "side": side,
                "event_ts": event_ts,
                "event_price": float(event["price"]),
                "signal_ts": signal_ts,
                "decision_ts": decision_ts,
                "signal_ref_price": ref_price,
                "next_open_ts": next_open_ts,
                "next_open_price": next_open_price,
                "session_date": event["session_date"],
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["ts_local"] = out["event_ts"].dt.tz_convert(events["ts_local"].dt.tz)
        out["signal_local"] = out["signal_ts"].dt.tz_convert(events["ts_local"].dt.tz)
        out["decision_local"] = out["decision_ts"].dt.tz_convert(events["ts_local"].dt.tz)
        out["next_open_local"] = pd.to_datetime(out["next_open_ts"], utc=True, errors="coerce").dt.tz_convert(
            events["ts_local"].dt.tz
        )
    return out


def _plot_sessions(
    trace: pd.DataFrame,
    one: pd.DataFrame,
    events: pd.DataFrame,
    entry_diag: pd.DataFrame,
    *,
    out_dir: Path,
    sessions_per_fig: int,
    tz: str,
    symbol: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dates = sorted(one["session_date"].dropna().unique().tolist())
    outputs: list[Path] = []
    chunks = int(ceil(len(session_dates) / max(1, sessions_per_fig)))
    candle_width = 0.65 / (24 * 60)

    for chunk_idx in range(chunks):
        dates = session_dates[chunk_idx * sessions_per_fig:(chunk_idx + 1) * sessions_per_fig]
        n = len(dates)
        fig, axes = plt.subplots(
            n * 2,
            1,
            figsize=(18, max(7.0, 5.5 * n)),
            sharex=False,
            gridspec_kw={"height_ratios": [2.5, 1.0] * n},
        )
        if n == 1:
            axes = [axes[0], axes[1]]

        for idx, session_date in enumerate(dates):
            ax_price = axes[idx * 2]
            ax_prob = axes[idx * 2 + 1]

            one_s = one[one["session_date"] == session_date].copy().sort_values("ts_local")
            tr_s = trace[trace["session_date"] == session_date].copy().sort_values("ts_local")
            ev_s = events[events["session_date"] == session_date].copy().sort_values("ts_local")
            dg_s = entry_diag[entry_diag["session_date"] == session_date].copy().sort_values("event_ts")
            if one_s.empty or tr_s.empty:
                continue

            x = mdates.date2num(one_s["ts_local"].to_list())
            open_ = pd.to_numeric(one_s["open"], errors="coerce").to_numpy()
            high = pd.to_numeric(one_s["high"], errors="coerce").to_numpy()
            low = pd.to_numeric(one_s["low"], errors="coerce").to_numpy()
            close = pd.to_numeric(one_s["close"], errors="coerce").to_numpy()
            up = close >= open_
            down = ~up

            ax_price.vlines(x, low, high, color="#4a4a4a", linewidth=0.5, zorder=1)
            ax_price.bar(
                x[up],
                close[up] - open_[up],
                width=candle_width,
                bottom=open_[up],
                color="#1976D2",
                edgecolor="none",
                zorder=1.2,
                label="1m bull",
            )
            ax_price.bar(
                x[down],
                close[down] - open_[down],
                width=candle_width,
                bottom=open_[down],
                color="#E53935",
                edgecolor="none",
                zorder=1.2,
                label="1m bear",
            )

            # Actual entries/exits
            if not ev_s.empty:
                ex = mdates.date2num(ev_s["ts_local"].to_list())
                ey = pd.to_numeric(ev_s["price"], errors="coerce").to_numpy()
                specs = {
                    "enter_long": ("#2E7D32", "^", "actual enter long"),
                    "exit_long": ("#8c564b", "v", "actual exit long"),
                    "enter_short": ("#C62828", "v", "actual enter short"),
                    "exit_short": ("#9467bd", "^", "actual exit short"),
                }
                for event_name, (color, marker, label) in specs.items():
                    mask = ev_s["event"].astype(str).eq(event_name).to_numpy()
                    if mask.any():
                        ax_price.scatter(ex[mask], ey[mask], color=color, marker=marker, s=42, label=label, zorder=2.5)

            # Entry diagnostics: signal arm, breakout ref, and naive next open.
            if not dg_s.empty:
                for _, row in dg_s.iterrows():
                    side = row["side"]
                    ref_color = "#2E7D32" if side == "long" else "#C62828"
                    sig_x = mdates.date2num(row["decision_local"])
                    evt_x = mdates.date2num(row["ts_local"])
                    ax_price.axvline(sig_x, color=ref_color, linestyle="--", linewidth=0.8, alpha=0.45)
                    ax_price.hlines(
                        float(row["signal_ref_price"]),
                        sig_x,
                        evt_x,
                        color=ref_color,
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.75,
                    )
                    if pd.notna(row["next_open_local"]) and pd.notna(row["next_open_price"]):
                        nx = mdates.date2num(row["next_open_local"])
                        marker = "^" if side == "long" else "v"
                        label = "next 10m open long" if side == "long" else "next 10m open short"
                        ax_price.scatter(
                            [nx],
                            [float(row["next_open_price"])],
                            facecolors="none",
                            edgecolors=ref_color,
                            marker=marker,
                            s=52,
                            linewidths=1.3,
                            label=label,
                            zorder=2.3,
                        )

            ax_price.set_title(f"{symbol} | {session_date.date()} | actual 1m breakout vs next 10m open")
            ax_price.set_ylabel("Price")
            ax_price.grid(True, alpha=0.25)
            handles, labels = ax_price.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_price.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=8, ncol=4)

            mx = mdates.date2num(tr_s["ts_local"].to_list())
            for col, color, label in (
                ("p_enter_long", "#2ca02c", "p_enter_long"),
                ("p_enter_short", "#d62728", "p_enter_short"),
                ("p_exit_long", "#17becf", "p_exit_long"),
                ("p_exit_short", "#ff7f0e", "p_exit_short"),
            ):
                y = pd.to_numeric(tr_s[col], errors="coerce")
                if y.notna().any():
                    ax_prob.step(mx, y.to_numpy(), where="post", color=color, linewidth=1.2, label=label)
            for col, color, label in (
                ("thr_enter_long", "#2ca02c", "thr_enter_long"),
                ("thr_enter_short", "#d62728", "thr_enter_short"),
                ("thr_exit_long", "#17becf", "thr_exit_long"),
                ("thr_exit_short", "#ff7f0e", "thr_exit_short"),
            ):
                y = pd.to_numeric(tr_s[col], errors="coerce").dropna()
                if not y.empty:
                    ax_prob.axhline(float(y.iloc[-1]), color=color, linestyle="--", linewidth=1.0, alpha=0.85, label=label)

            ax_prob.set_ylim(-0.02, 1.02)
            ax_prob.set_ylabel("Probability")
            ax_prob.set_xlabel(f"Session Time ({tz})")
            ax_prob.grid(True, alpha=0.25)
            handles, labels = ax_prob.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_prob.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8, ncol=4)

            session_tz = one_s["ts_local"].dt.tz
            locator = mdates.HourLocator(interval=1, tz=session_tz)
            formatter = mdates.DateFormatter("%H:%M", tz=session_tz)
            ax_price.xaxis.set_major_locator(locator)
            ax_price.xaxis.set_major_formatter(formatter)
            ax_prob.xaxis.set_major_locator(locator)
            ax_prob.xaxis.set_major_formatter(formatter)
            ax_price.tick_params(axis="x", labelbottom=False)
            ax_price.set_xlim(x.min() - candle_width * 6, x.max() + candle_width * 6)
            ax_prob.set_xlim(x.min() - candle_width * 6, x.max() + candle_width * 6)

        fig.tight_layout()
        out_path = out_dir / f"{symbol.lower()}_independent_1m_vs_next_open_part{chunk_idx + 1}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def main() -> None:
    args = _parse_args()
    trace, events, one = _load_inputs(
        Path(args.trace),
        Path(args.events),
        Path(args.one_min_data),
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        tz=args.tz,
    )
    entry_diag = _derive_entry_diagnostics(trace, events)
    outputs = _plot_sessions(
        trace,
        one,
        events,
        entry_diag,
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
