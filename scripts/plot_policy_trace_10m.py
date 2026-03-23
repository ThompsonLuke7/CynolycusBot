from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 10m meta candles and probabilities with actual policy entry/exit markers."
    )
    parser.add_argument(
        "--trace",
        default="Data/inference/spy/10min/meta/meta_trace_policy_last_month.csv",
        help="Mixed replay trace CSV containing policy_1m and meta_10m rows.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to plot.")
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp.")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument("--sessions-per-fig", type=int, default=4, help="Sessions per output PNG.")
    parser.add_argument(
        "--out-dir",
        default="Data/inference/spy/10min/plots/policy_sessions_10m",
        help="Directory for output PNGs.",
    )
    return parser.parse_args()


def _load_trace(
    path: Path,
    *,
    symbol: str,
    start: str | None,
    end: str | None,
    tz: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    if "trace_kind" not in df.columns:
        df["trace_kind"] = "meta_10m"
    else:
        df["trace_kind"] = df["trace_kind"].fillna("meta_10m")

    df["ts_local"] = df["timestamp"].dt.tz_convert(tz)
    df["session_date"] = df["ts_local"].dt.normalize()
    meta = df[df["trace_kind"].astype(str) == "meta_10m"].copy().sort_values("timestamp")
    policy = df[df["trace_kind"].astype(str) == "policy_1m"].copy().sort_values("timestamp")
    if meta.empty:
        raise ValueError("Trace does not contain meta_10m rows.")
    if policy.empty:
        raise ValueError("Trace does not contain policy_1m rows.")
    return meta, policy


def _derive_policy_markers(policy_df: pd.DataFrame) -> pd.DataFrame:
    out = policy_df.sort_values("timestamp").copy()
    long_pos = pd.to_numeric(out.get("policy_long_contracts"), errors="coerce").fillna(0).astype(int)
    short_pos = pd.to_numeric(out.get("policy_short_contracts"), errors="coerce").fillna(0).astype(int)
    prev_long = long_pos.shift(1).fillna(0).astype(int)
    prev_short = short_pos.shift(1).fillna(0).astype(int)
    out["entry_long"] = (long_pos > 0) & (prev_long <= 0)
    out["exit_long"] = (prev_long > 0) & (long_pos <= 0)
    out["entry_short"] = (short_pos > 0) & (prev_short <= 0)
    out["exit_short"] = (prev_short > 0) & (short_pos <= 0)
    return out


def _plot_sessions(
    meta: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    out_dir: Path,
    sessions_per_fig: int,
    tz: str,
    symbol: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = _derive_policy_markers(policy)
    session_dates = sorted(meta["session_date"].dropna().unique().tolist())
    if not session_dates:
        raise ValueError("No meta_10m session rows found to plot.")

    outputs: list[Path] = []
    chunks = int(ceil(len(session_dates) / max(1, sessions_per_fig)))
    candle_width = 8.0 / (24 * 60)

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
            met = meta[meta["session_date"] == session_date].copy().sort_values("ts_local")
            pol = policy[policy["session_date"] == session_date].copy().sort_values("ts_local")
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

            spread = high - low
            offset = pd.Series(spread).replace(0, pd.NA).dropna().median()
            if pd.isna(offset) or offset <= 0:
                offset = max(close) * 0.0015

            if not pol.empty:
                px = mdates.date2num(pol["ts_local"].to_list())
                low_p = pd.to_numeric(pol["low"], errors="coerce").to_numpy()
                high_p = pd.to_numeric(pol["high"], errors="coerce").to_numpy()
                y_enter_long = low_p - offset * 1.2
                y_exit_long = high_p + offset * 0.8
                y_enter_short = high_p + offset * 1.2
                y_exit_short = low_p - offset * 0.8

                for col, yvals, color, marker, label in (
                    ("entry_long", y_enter_long, "#2E7D32", "^", "enter long"),
                    ("exit_long", y_exit_long, "#8c564b", "v", "exit long"),
                    ("entry_short", y_enter_short, "#C62828", "v", "enter short"),
                    ("exit_short", y_exit_short, "#9467bd", "^", "exit short"),
                ):
                    mask = pol[col].fillna(False).to_numpy()
                    if mask.any():
                        ax_price.scatter(px[mask], yvals[mask], color=color, marker=marker, s=34, label=label, zorder=2.0)

            ax_price.set_title(f"{symbol} | {session_date.date()} | 10m bars + policy execution")
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
        out_path = out_dir / f"{symbol.lower()}_policy_sessions_10m_part{chunk_idx + 1}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def main() -> None:
    args = _parse_args()
    meta, policy = _load_trace(
        Path(args.trace),
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        tz=args.tz,
    )
    outputs = _plot_sessions(
        meta,
        policy,
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
