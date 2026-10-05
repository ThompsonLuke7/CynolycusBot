"""Equity and drawdown chart for the 05 rotation backtest: monthly rebalance, phase 0 (2026-09-29).

It uses the same simulator, costs and fills as 05 and shows only a few arms, so the risk behind
each CAGR is visible. Output: 09_equity_monthly.png.

    .venv/bin/python research/long_horizon_discount_2026-09-25/09_plot_equity.py
"""
from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core.shared_plotting.figures import save_figure  # noqa: E402
from core.shared_plotting.theme import apply_mpl_defaults, style_figure  # noqa: E402

_spec = spec_from_file_location("rb", HERE / "05_rotation_backtest.py")
rb = module_from_spec(_spec)
_spec.loader.exec_module(rb)

ARMS = {"spy": "SPY buy & hold", "qqq": "QQQ buy & hold", "mom_top300": "12-1 momentum, top-300 liquid",
        "mom_12_1": "12-1 momentum, liquid-1000", "core_satellite": "70% SPY + 30% momentum",
        "hand_rule": "fuzzy 200-SMA + TMO rule"}


def main() -> None:
    p = pd.read_parquet(rb.pb.OUT)
    p = rb.build_scores(p[p["date"] >= rb.START - pd.Timedelta(days=7)])
    O_df, C_df = rb.price_matrices(sorted(set(p["ticker"]) | set(rb.BENCH)))
    C_df = C_df.ffill()
    O_df = O_df.fillna(C_df.shift(1))
    O_df["CASH"] = C_df["CASH"] = 1.0
    O_df = O_df[C_df.columns]
    cal, col = C_df.index, {t: i for i, t in enumerate(C_df.columns)}
    end = len(cal) - 1
    ds = pd.DatetimeIndex(sorted(p.loc[p["date"] >= rb.START, "date"].unique()))[::rb.FREQS["monthly"]]
    by_date = dict(tuple(p.groupby("date")))
    entries = [e for e in (cal.searchsorted(d, side="right") for d in ds) if e <= end]
    curves = {}
    for arm, label in ARMS.items():
        ws = [rb.weights_for(arm, by_date[d], "monthly") for d in ds[:len(entries)]]
        v, _ = rb.simulate(ws, entries, O_df.values, C_df.values, col, end)
        curves[label] = pd.Series(v, index=cal[entries[0]:])

    theme = apply_mpl_defaults()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    colors = [theme.neutral, theme.purple, theme.blue, theme.amber, theme.bull, theme.warning]
    for (label, s), c in zip(curves.items(), colors):
        lw = 2.2 if label.startswith("SPY") else 1.4
        ax1.plot(s.index, s.values, color=c, lw=lw, label=f"{label}  (x{s.iloc[-1]:.1f})")
        ax2.plot(s.index, (s / s.cummax() - 1) * 100, color=c, lw=lw)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title(f"Rotation backtest, monthly rebalance (phase 0), {cal[entries[0]].date()} to {cal[end].date()}\n"
                  "Rule-based, no model fitting; 20 names equal weight; 20 bp round trip; survivor-shaped universe "
                  "(today's tickers), which flatters the stock-picking lines")
    ax1.legend(loc="upper left", fontsize=9, facecolor=theme.axes_bg, labelcolor=theme.text)
    ax2.set_ylabel("drawdown from peak (%)")
    style_figure(fig, [ax1, ax2], theme)
    out = save_figure(fig, HERE / "09_equity_monthly.png", close=True)
    print(out)


if __name__ == "__main__":
    main()
