from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot equity curves for independent long/short meta trace using next-bar-open fills."
    )
    parser.add_argument(
        "--trace",
        default="Data/inference/spy/10min/meta/meta_trace_independent_last_month.csv",
        help="Independent replay trace CSV.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol label for the chart.")
    parser.add_argument(
        "--out",
        default="Data/inference/spy/10min/plots/meta_independent_equity_curve.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def _load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("ind_entry_long", "ind_exit_long", "ind_entry_short", "ind_exit_short"):
        df[col] = df[col].fillna(False).astype(bool)
    return df


def _build_equity_curves(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        raise ValueError("Trace must contain at least two bars.")

    records: list[dict[str, float | pd.Timestamp]] = []
    buy_hold = 1.0
    long_equity = 1.0
    short_equity = 1.0
    net_1x_equity = 1.0

    long_on = False
    short_on = False

    for i in range(len(df) - 1):
        open_i = float(df.loc[i, "open"])
        open_n = float(df.loc[i + 1, "open"])
        ts_n = pd.Timestamp(df.loc[i + 1, "timestamp"])
        if not (np.isfinite(open_i) and np.isfinite(open_n) and open_i > 0.0 and open_n > 0.0):
            continue

        bar_ret = open_n / open_i - 1.0
        buy_hold *= 1.0 + bar_ret
        if long_on:
            long_equity *= 1.0 + bar_ret
        if short_on:
            short_equity *= 1.0 - bar_ret

        active_sleeves = int(long_on) + int(short_on)

        # Net 1x curve collapses simultaneous long+short to flat exposure.
        if long_on and not short_on:
            net_1x_equity *= 1.0 + bar_ret
        elif short_on and not long_on:
            net_1x_equity *= 1.0 - bar_ret

        combined_normalized = 0.5 * long_equity + 0.5 * short_equity
        combined_full_gross = long_equity + short_equity - 1.0

        records.append(
            {
                "timestamp": ts_n,
                "buy_hold": buy_hold,
                "long_only": long_equity,
                "short_only": short_equity,
                "combined_independent_normalized": combined_normalized,
                "combined_independent_full_gross": combined_full_gross,
                "net_1x_style": net_1x_equity,
                "active_sleeves": float(active_sleeves),
            }
        )

        if df.loc[i, "ind_exit_long"] and long_on:
            long_on = False
        if df.loc[i, "ind_exit_short"] and short_on:
            short_on = False
        if df.loc[i, "ind_entry_long"] and not long_on:
            long_on = True
        if df.loc[i, "ind_entry_short"] and not short_on:
            short_on = True

    equity_df = pd.DataFrame(records)
    if equity_df.empty:
        raise ValueError("No equity points were generated from the trace.")
    return equity_df


def _save_plot(equity_df: pd.DataFrame, *, save_path: Path, symbol: str) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [2.4, 0.8]},
        sharex=True,
    )
    ax = axes[0]
    ax_aux = axes[1]

    x = pd.to_datetime(equity_df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    for col, color, label, width in (
        ("buy_hold", "#444444", "buy_hold", 1.7),
        ("long_only", "#2E7D32", "long_only", 1.5),
        ("short_only", "#C62828", "short_only", 1.5),
        ("combined_independent_normalized", "#6A1B9A", "combined_independent_normalized", 1.5),
        ("combined_independent_full_gross", "#8E24AA", "combined_independent_full_gross", 2.0),
        ("net_1x_style", "#1565C0", "net_1x_style", 1.5),
    ):
        ax.plot(x, equity_df[col], color=color, linewidth=width, label=label)

    ax.set_title(f"{symbol} | next-bar-open equity curves")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    ax_aux.plot(x, equity_df["active_sleeves"], color="#ff7f0e", linewidth=1.3, label="active_sleeves")
    ax_aux.set_ylabel("Sleeves")
    ax_aux.set_xlabel("Session Time (America/New_York)")
    ax_aux.grid(True, alpha=0.25)
    ax_aux.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    trace = _load_trace(Path(args.trace))
    equity_df = _build_equity_curves(trace)
    _save_plot(equity_df, save_path=Path(args.out), symbol=args.symbol)
    print(Path(args.out))


if __name__ == "__main__":
    main()
