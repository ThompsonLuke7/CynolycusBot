from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.shared_plotting import (
    apply_mpl_defaults,
    compute_marker_offset,
    plot_candles_from_frame,
    save_figure,
    style_figure,
    time_to_position,
)
from strategies.dealer_positioning.levels import _core_levels_from_ladder


def _load_level_history(snapshot_dir: Path, *, session_date: date, magnet_quantile: float) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(snapshot_dir.glob("gamma_ladder_*.csv")):
        ladder = pd.read_csv(path)
        if ladder.empty or "timestamp" not in ladder or "spot" not in ladder:
            continue
        ts = pd.to_datetime(ladder["timestamp"].iloc[0], utc=True, errors="coerce")
        if pd.isna(ts) or ts.date() != session_date:
            continue
        spot = float(pd.to_numeric(ladder["spot"], errors="coerce").dropna().iloc[0])
        numeric_cols = [
            "strike",
            "call_gex",
            "put_gex",
            "net_gex",
            "abs_net_gex",
            "total_abs_gex",
        ]
        for col in numeric_cols:
            ladder[col] = pd.to_numeric(ladder[col], errors="coerce")
        ladder = ladder.dropna(subset=["strike", "net_gex", "abs_net_gex"])
        if ladder.empty:
            continue
        core = _core_levels_from_ladder(ladder, spot, magnet_quantile)
        rows.append(
            {
                "timestamp": ts,
                "spot": spot,
                "source_file": path.name,
                **core,
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _load_bars(path: Path, *, session_date: date) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    bars = pd.read_parquet(path)
    if bars.empty or "timestamp" not in bars:
        return pd.DataFrame()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["timestamp"])
    bars = bars[bars["timestamp"].dt.date == session_date].copy()
    for col in ("open", "high", "low", "close"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    return bars.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)


def _load_orders(path: Path, *, session_date: date) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    orders = pd.read_csv(path)
    if orders.empty or "opened_at" not in orders:
        return pd.DataFrame()
    orders["opened_at"] = pd.to_datetime(orders["opened_at"], utc=True, errors="coerce")
    orders["closed_at"] = pd.to_datetime(orders.get("closed_at"), utc=True, errors="coerce")
    orders = orders[orders["opened_at"].dt.date == session_date].copy()
    for col in ("entry", "exit_price", "pnl_dollars", "entry_contract_price", "exit_contract_price"):
        if col in orders:
            orders[col] = pd.to_numeric(orders[col], errors="coerce")
    return orders.reset_index(drop=True)


def _plot_orders(ax, bars: pd.DataFrame, orders: pd.DataFrame) -> None:
    if bars.empty or orders.empty:
        return
    bar_index = pd.DatetimeIndex(bars["timestamp"])
    marker_offset = compute_marker_offset(bars, bars["high"].to_numpy(dtype=float), bars["low"].to_numpy(dtype=float))
    entries = orders[orders["status"].eq("open")].copy()
    exits = orders[orders["status"].eq("closed")].copy()

    if not entries.empty:
        entries["x"] = time_to_position(bar_index, entries["opened_at"]).to_numpy()
        calls = entries["contract_side"].astype(str).str.upper().eq("C")
        puts = entries["contract_side"].astype(str).str.upper().eq("P")
        ax.scatter(
            entries.loc[calls, "x"],
            entries.loc[calls, "entry"] + marker_offset * 0.55,
            marker="^",
            s=72,
            color="#22c55e",
            edgecolor="#0f172a",
            linewidth=0.6,
            label="paper call buy",
            zorder=5,
        )
        ax.scatter(
            entries.loc[puts, "x"],
            entries.loc[puts, "entry"] - marker_offset * 0.55,
            marker="v",
            s=72,
            color="#f59e0b",
            edgecolor="#0f172a",
            linewidth=0.6,
            label="paper put buy",
            zorder=5,
        )

    if not exits.empty:
        exits["x"] = time_to_position(bar_index, exits["closed_at"]).to_numpy()
        wins = exits["pnl_dollars"].fillna(0.0) >= 0.0
        ax.scatter(
            exits.loc[wins, "x"],
            exits.loc[wins, "exit_price"],
            marker="x",
            s=55,
            color="#86efac",
            linewidth=1.5,
            label="paper exit win",
            zorder=5.2,
        )
        ax.scatter(
            exits.loc[~wins, "x"],
            exits.loc[~wins, "exit_price"],
            marker="x",
            s=55,
            color="#fb7185",
            linewidth=1.5,
            label="paper exit loss",
            zorder=5.2,
        )


def _plot_levels(
    levels: pd.DataFrame,
    bars: pd.DataFrame,
    orders: pd.DataFrame,
    output_path: Path,
    *,
    symbol: str,
    session_date: date,
) -> None:
    theme = apply_mpl_defaults()
    fig, ax = plt.subplots(figsize=(16, 8))
    style_figure(fig, ax, theme)
    plot_levels = levels
    if not bars.empty:
        plot_levels = levels[
            (levels["timestamp"] >= bars["timestamp"].min()) & (levels["timestamp"] <= bars["timestamp"].max())
        ].copy()
        if plot_levels.empty:
            plot_levels = levels
    if not bars.empty:
        plot_candles_from_frame(ax, bars, time_col="timestamp", compressed=True, theme=theme, label=False)
        x_levels = time_to_position(pd.DatetimeIndex(bars["timestamp"]), plot_levels["timestamp"])
        x_spot = x_levels
        tick_rows = bars.iloc[::30]
        ax.set_xticks(tick_rows.index.to_numpy(dtype=float))
        ax.set_xticklabels(
            pd.to_datetime(tick_rows["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M"),
            rotation=45,
            ha="right",
        )
    else:
        x_levels = np.arange(len(plot_levels), dtype=float)
        x_spot = x_levels
        ax.plot(x_spot, plot_levels["spot"], color=theme.text, linewidth=1.2, label="spot")

    series = [
        ("call_wall", "#ef4444", "call wall"),
        ("put_wall", "#22c55e", "put wall"),
        ("nearest_magnet", "#60a5fa", "magnet"),
        ("next_magnet_above", "#a78bfa", "next magnet above"),
        ("next_magnet_below", "#f97316", "next magnet below"),
        ("gamma_flip", "#fbbf24", "gamma flip"),
    ]
    for col, color, label in series:
        if col in plot_levels:
            ax.step(x_levels, plot_levels[col], where="post", color=color, linewidth=1.35, label=label, alpha=0.92)

    ax.plot(x_spot, plot_levels["spot"], color=theme.text, linewidth=1.0, alpha=0.82, label="dealer spot")
    _plot_orders(ax, bars, orders)
    ax.set_title(f"{symbol} dealer positioning levels + paper option orders - {session_date.isoformat()}")
    ax.set_xlabel("time ET")
    ax.set_ylabel("price / strike")
    legend = ax.legend(loc="upper left", ncols=2, fontsize=9, framealpha=0.78)
    legend.get_frame().set_facecolor(theme.axes_bg)
    legend.get_frame().set_edgecolor(theme.spine)
    for text in legend.get_texts():
        text.set_color(theme.text)
    save_figure(fig, output_path, close=True)


def build_report(
    *,
    symbol: str,
    session_date: date,
    data_root: Path,
    bars_path: Path,
    output_dir: Path,
    magnet_quantile: float,
    orders_path: Path | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    snapshot_dir = data_root / symbol.upper() / "snapshots"
    levels = _load_level_history(snapshot_dir, session_date=session_date, magnet_quantile=magnet_quantile)
    if levels.empty:
        raise RuntimeError(f"no dealer snapshots found for {symbol} on {session_date.isoformat()}")
    bars = _load_bars(bars_path, session_date=session_date)
    orders = _load_orders(orders_path, session_date=session_date) if orders_path else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{symbol.upper()}_dealer_levels_{session_date.isoformat()}.csv"
    png_path = output_dir / f"{symbol.upper()}_dealer_levels_{session_date.isoformat()}.png"
    levels.to_csv(csv_path, index=False)
    _plot_levels(levels, bars, orders, png_path, symbol=symbol.upper(), session_date=session_date)
    return csv_path, png_path, levels


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dealer positioning levels over intraday bars.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--date", default=None, help="Session date, YYYY-MM-DD. Defaults to latest snapshot date.")
    parser.add_argument("--data-root", default="Data/dealer_positioning")
    parser.add_argument("--bars-path", default="Data/raw/spy/spy_intraday_1min_2026_06_12.parquet")
    parser.add_argument("--orders-path", default="Data/dealer_positioning/SPY/live_trade_log.csv")
    parser.add_argument("--output-dir", default="Data/dealer_positioning/reports")
    parser.add_argument("--magnet-quantile", type=float, default=0.90)
    args = parser.parse_args()

    session_date = date.fromisoformat(args.date) if args.date else _latest_snapshot_date(Path(args.data_root), args.symbol)
    csv_path, png_path, levels = build_report(
        symbol=args.symbol,
        session_date=session_date,
        data_root=Path(args.data_root),
        bars_path=Path(args.bars_path),
        output_dir=Path(args.output_dir),
        magnet_quantile=float(args.magnet_quantile),
        orders_path=Path(args.orders_path) if args.orders_path else None,
    )
    print(f"rows={len(levels)}")
    print(f"csv={csv_path}")
    print(f"plot={png_path}")
    print(levels[["timestamp", "spot", "call_wall", "put_wall", "nearest_magnet", "gamma_flip"]].tail(5).to_string(index=False))


def _latest_snapshot_date(data_root: Path, symbol: str) -> date:
    files = sorted((data_root / symbol.upper() / "snapshots").glob("gamma_ladder_*.csv"))
    for path in reversed(files):
        frame = pd.read_csv(path, nrows=1)
        if "timestamp" not in frame or frame.empty:
            continue
        ts = pd.to_datetime(frame["timestamp"].iloc[0], utc=True, errors="coerce")
        if not pd.isna(ts):
            return ts.date()
    raise RuntimeError(f"no dated snapshots found for {symbol}")


if __name__ == "__main__":
    main()
