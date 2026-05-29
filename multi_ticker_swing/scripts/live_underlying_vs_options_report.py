from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_ticker_swing.scripts.analyze_live_trade_fills import load_audit


BASE = Path("Data/analysis/multi_ticker_swing_live")
OUT = BASE / "underlying_vs_options"


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(BASE / "paired_option_trades.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades = _num(
        trades,
        [
            "direction",
            "entry_price_underlying",
            "exit_price",
            "exit_pnl_pct",
            "pnl_dollars",
            "pnl_pct_option",
            "entry_price_option",
            "holding_minutes",
        ],
    )
    _opens, _closes, bars, _signals = load_audit(Path("UI/swing_audit"))
    bars = _num(
        bars,
        [
            "direction",
            "entry_price",
            "underlying_close",
            "underlying_high",
            "underlying_low",
            "option_last_price",
            "pnl_pct_underlying_mark",
        ],
    )
    if not bars.empty:
        bars["audit_ts"] = pd.to_datetime(bars["audit_ts"], utc=True, errors="coerce")
    spy = pd.read_parquet(BASE / "spy_1min_20260501_20260526.parquet")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True, errors="coerce")
    return trades, bars, spy


def short_path_stats(trades: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shorts = trades.dropna(
        subset=["direction", "entry_price_underlying", "exit_price", "exit_time"]
    )
    shorts = shorts[shorts["direction"] == -1].copy()
    rows: list[dict] = []
    horizons = [5, 15, 30, 60, 120, 240, 390, 780, 1440]
    horizon_rows: list[dict] = []
    for trade in shorts.itertuples(index=False):
        path = bars[
            (bars["option_symbol"] == trade.symbol)
            & (bars["audit_ts"] >= trade.entry_time)
            & (bars["audit_ts"] <= trade.exit_time)
        ].sort_values("audit_ts")
        path = path.dropna(subset=["underlying_close"])
        entry = float(trade.entry_price_underlying)
        final_signed = -1.0 * (float(trade.exit_price) - entry) / entry
        rec = {
            "symbol": trade.symbol,
            "ticker": trade.ticker,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "bars": len(path),
            "holding_minutes": trade.holding_minutes,
            "final_signed_ret": final_signed,
            "stock_pnl_100": final_signed * entry * 100.0,
            "option_pnl": trade.pnl_dollars,
        }
        if not path.empty:
            signed_close = -1.0 * (path["underlying_close"] - entry) / entry
            signed_high = -1.0 * (path["underlying_low"] - entry) / entry
            signed_low = -1.0 * (path["underlying_high"] - entry) / entry
            best_series = pd.concat([signed_close, signed_high], axis=0)
            worst_series = pd.concat([signed_close, signed_low], axis=0)
            rec.update(
                {
                    "mfe_signed_ret": best_series.max(),
                    "mae_signed_ret": worst_series.min(),
                    "range_signed_ret": best_series.max() - worst_series.min(),
                    "first_5m_signed_ret": signed_close.iloc[0],
                    "last_mark_signed_ret": signed_close.iloc[-1],
                    "hit_1pct_favorable": bool(best_series.max() >= 0.01),
                    "hit_2pct_favorable": bool(best_series.max() >= 0.02),
                    "hit_3pct_favorable": bool(best_series.max() >= 0.03),
                    "hit_1pct_adverse": bool(worst_series.min() <= -0.01),
                    "hit_2pct_adverse": bool(worst_series.min() <= -0.02),
                    "hit_3pct_adverse": bool(worst_series.min() <= -0.03),
                }
            )
            for minutes in horizons:
                target = trade.entry_time + pd.Timedelta(minutes=minutes)
                after = path[path["audit_ts"] >= target]
                if after.empty:
                    continue
                px = float(after.iloc[0]["underlying_close"])
                ret = -1.0 * (px - entry) / entry
                horizon_rows.append(
                    {
                        "horizon_min": minutes,
                        "symbol": trade.symbol,
                        "ticker": trade.ticker,
                        "signed_ret": ret,
                        "favorable": ret > 0,
                    }
                )
        rows.append(rec)
    per_trade = pd.DataFrame(rows)
    horizons_df = pd.DataFrame(horizon_rows)
    return per_trade, horizons_df


def _summary(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in value_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append(
            {
                "metric": col,
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "min": s.min(),
                "max": s.max(),
                "p10": s.quantile(0.10),
                "p25": s.quantile(0.25),
                "p75": s.quantile(0.75),
                "p90": s.quantile(0.90),
            }
        )
    return pd.DataFrame(rows)


def build_performance_curves(trades: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    calls = trades.dropna(subset=["exit_time", "pnl_dollars"]).copy()
    calls = calls[calls["option_type"] == "C"]
    valid_underlying = trades.dropna(
        subset=["exit_time", "direction", "entry_price_underlying", "exit_price", "pnl_dollars"]
    ).copy()
    longs = valid_underlying[valid_underlying["direction"] == 1].copy()
    call_events = calls[["exit_time", "pnl_dollars"]].rename(columns={"pnl_dollars": "call_pnl"})
    call_events["exit_time"] = call_events["exit_time"].dt.floor("min")
    stock_events = longs[["exit_time"]].copy()
    stock_events["exit_time"] = stock_events["exit_time"].dt.floor("min")
    stock_events["stock_long_pnl"] = (
        (longs["exit_price"] - longs["entry_price_underlying"]) * 100.0
    )
    start = min(trades["entry_time"].min(), spy["timestamp"].min())
    end = max(trades["exit_time"].max(), spy["timestamp"].max())
    idx = pd.date_range(start.floor("min"), end.ceil("min"), freq="1min", tz="UTC")
    curve = pd.DataFrame(index=idx)
    curve["call_buying_pnl"] = call_events.groupby("exit_time")["call_pnl"].sum().cumsum()
    curve["stock_long_100sh_pnl"] = (
        stock_events.groupby("exit_time")["stock_long_pnl"].sum().cumsum()
    )
    curve = curve.ffill().fillna(0.0)
    spy_window = spy[(spy["timestamp"] >= start) & (spy["timestamp"] <= end)].copy()
    spy_window = spy_window.dropna(subset=["close"]).sort_values("timestamp")
    if not spy_window.empty:
        first = float(spy_window.iloc[0]["close"])
        spy_series = spy_window.set_index("timestamp")["close"]
        curve["spy_100sh_buy_hold_pnl"] = (spy_series - first) * 100.0
        curve["spy_100sh_buy_hold_pnl"] = curve["spy_100sh_buy_hold_pnl"].ffill().fillna(0.0)
    else:
        curve["spy_100sh_buy_hold_pnl"] = 0.0
    curve["call_buying_equity_10k"] = 10_000.0 + curve["call_buying_pnl"]
    curve["stock_long_equity_10k"] = 10_000.0 + curve["stock_long_100sh_pnl"]
    curve["spy_equity_10k_proxy"] = 10_000.0 + curve["spy_100sh_buy_hold_pnl"]
    return curve.reset_index(names="timestamp")


def plot_curves(curve: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(curve["timestamp"], curve["call_buying_pnl"], label="Calls bought, actual fills")
    axes[0].plot(curve["timestamp"], curve["stock_long_100sh_pnl"], label="Long signals as 100-share stock")
    axes[0].plot(curve["timestamp"], curve["spy_100sh_buy_hold_pnl"], label="SPY buy-hold, 100 shares")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Live Multi-Ticker Swing: Calls vs Stock Longs vs SPY")
    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(curve["timestamp"], curve["call_buying_equity_10k"] / 10_000 - 1, label="Calls equity / 10k")
    axes[1].plot(curve["timestamp"], curve["stock_long_equity_10k"] / 10_000 - 1, label="Stock longs equity / 10k")
    axes[1].plot(curve["timestamp"], curve["spy_equity_10k_proxy"] / 10_000 - 1, label="SPY 100sh proxy / 10k")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Return on 10k proxy")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades, bars, spy = _load()
    short_trades, horizon = short_path_stats(trades, bars)
    short_summary = _summary(
        short_trades,
        [
            "final_signed_ret",
            "mfe_signed_ret",
            "mae_signed_ret",
            "range_signed_ret",
            "first_5m_signed_ret",
            "stock_pnl_100",
            "option_pnl",
        ],
    )
    horizon_summary = (
        horizon.groupby("horizon_min")
        .agg(
            n=("signed_ret", "size"),
            mean=("signed_ret", "mean"),
            median=("signed_ret", "median"),
            min=("signed_ret", "min"),
            max=("signed_ret", "max"),
            p25=("signed_ret", lambda s: s.quantile(0.25)),
            p75=("signed_ret", lambda s: s.quantile(0.75)),
            favorable_rate=("favorable", "mean"),
        )
        .reset_index()
        if not horizon.empty
        else pd.DataFrame()
    )
    hit_rates = pd.DataFrame(
        [
            {
                "n": len(short_trades),
                "hit_1pct_favorable": short_trades["hit_1pct_favorable"].mean(),
                "hit_2pct_favorable": short_trades["hit_2pct_favorable"].mean(),
                "hit_3pct_favorable": short_trades["hit_3pct_favorable"].mean(),
                "hit_1pct_adverse": short_trades["hit_1pct_adverse"].mean(),
                "hit_2pct_adverse": short_trades["hit_2pct_adverse"].mean(),
                "hit_3pct_adverse": short_trades["hit_3pct_adverse"].mean(),
            }
        ]
    )
    curve = build_performance_curves(trades, spy)
    plot_path = OUT / "call_vs_stock_vs_spy.png"
    plot_curves(curve, plot_path)

    short_trades.to_csv(OUT / "short_entry_path_metrics.csv", index=False)
    short_summary.to_csv(OUT / "short_entry_summary.csv", index=False)
    horizon_summary.to_csv(OUT / "short_entry_horizon_summary.csv", index=False)
    hit_rates.to_csv(OUT / "short_entry_hit_rates.csv", index=False)
    curve.to_csv(OUT / "call_vs_stock_vs_spy_curve.csv", index=False)

    lines = [
        "# Live Underlying vs Options Report",
        "",
        f"Plot: {plot_path}",
        "",
        "## Short Entry Summary",
        short_summary.to_string(index=False),
        "",
        "## Short Horizon Summary",
        horizon_summary.to_string(index=False),
        "",
        "## Short Hit Rates",
        hit_rates.to_string(index=False),
        "",
        "## Final Curve Values",
        curve.tail(1)[
            [
                "call_buying_pnl",
                "stock_long_100sh_pnl",
                "spy_100sh_buy_hold_pnl",
                "call_buying_equity_10k",
                "stock_long_equity_10k",
                "spy_equity_10k_proxy",
            ]
        ].to_string(index=False),
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "report.md").as_posix())
    print(plot_path.as_posix())
    print(curve.tail(1)[["call_buying_pnl", "stock_long_100sh_pnl", "spy_100sh_buy_hold_pnl"]].to_string(index=False))


if __name__ == "__main__":
    main()
