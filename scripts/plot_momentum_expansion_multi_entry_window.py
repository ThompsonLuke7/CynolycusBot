from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.momentum_expansion.data.load_bars import load_1h, load_4h
from scripts.plot_momentum_expansion_order_policy_replay import (
    _add_1h_indicators,
    _add_4h_indicators,
    _entry_trigger,
    _exit_trade,
)


DEFAULT_MATRIX = Path("strategies/momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_OUT = Path("strategies/momentum_expansion/plots/output/multi_entry_replay")


@dataclass
class Trade:
    ticker: str
    watch_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry_price: float
    entry_rule: str
    entry_score: float
    exit_ts: pd.Timestamp
    exit_price: float
    exit_reason: str
    return_pct: float


def _score_at(matrix_ticker: pd.DataFrame, ts: pd.Timestamp) -> float:
    idx = matrix_ticker.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(matrix_ticker["expansion_survival_score"].iloc[idx])


def _next_watch_ts(matrix_ticker: pd.DataFrame, after_ts: pd.Timestamp, end_ts: pd.Timestamp, min_score: float) -> pd.Timestamp | None:
    sub = matrix_ticker.loc[(matrix_ticker.index > after_ts) & (matrix_ticker.index <= end_ts)]
    sub = sub[sub["expansion_survival_score"] >= min_score]
    if sub.empty:
        return None
    return pd.Timestamp(sub.index[0])


def _simulate(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    matrix_ticker: pd.DataFrame,
    bars_1h: pd.DataFrame,
    bars_4h: pd.DataFrame,
    min_score: float,
    max_hours: int,
) -> list[Trade]:
    trades: list[Trade] = []
    watch_ts = _next_watch_ts(matrix_ticker, start - pd.Timedelta(nanoseconds=1), end, min_score)
    while watch_ts is not None and watch_ts <= end:
        trig = _entry_trigger(bars_1h, watch_ts, max_hours=max_hours)
        if trig is None:
            watch_ts = _next_watch_ts(matrix_ticker, watch_ts, end, min_score)
            continue
        entry_ts, entry_price, entry_rule = trig
        if entry_ts > end:
            break
        exit_ts, exit_price, exit_reason = _exit_trade(
            bars_4h,
            matrix_ticker,
            entry_ts,
            entry_price,
            score_decay_exit=0.30,
            score_decay_min_bars=0,
            atr_trail_arm=1.0,
            atr_trail_distance=2.4,
            trend_break_atr=None,
            max_holding_4h_bars=30,
        )
        trades.append(
            Trade(
                ticker=ticker,
                watch_ts=watch_ts,
                entry_ts=entry_ts,
                entry_price=float(entry_price),
                entry_rule=entry_rule,
                entry_score=_score_at(matrix_ticker, entry_ts),
                exit_ts=exit_ts,
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                return_pct=float(exit_price / entry_price - 1.0),
            )
        )
        watch_ts = _next_watch_ts(matrix_ticker, exit_ts, end, min_score)
    return trades


def _plot(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bars_4h: pd.DataFrame,
    matrix_ticker: pd.DataFrame,
    trades: list[Trade],
    out_dir: Path,
) -> Path:
    df = bars_4h.loc[start - pd.Timedelta(days=5) : end + pd.Timedelta(days=5)].copy()
    scores = matrix_ticker["expansion_survival_score"].reindex(df.index, method="ffill")
    x = np.arange(len(df))
    fig, (ax, ax_score) = plt.subplots(2, 1, figsize=(17, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    for axis in (ax, ax_score):
        axis.set_facecolor("#0d1117")
        axis.grid(True, color="#30363d", alpha=0.5)
        axis.tick_params(colors="#8b949e")
        for spine in axis.spines.values():
            spine.set_edgecolor("#30363d")

    for i, row in enumerate(df.itertuples()):
        color = "#2dd4bf" if row.close >= row.open else "#fb7185"
        ax.vlines(i, row.low, row.high, color=color, linewidth=1)
        ax.vlines(i, row.open, row.close, color=color, linewidth=4)

    def pos(ts: pd.Timestamp) -> int:
        return min(max(int(df.index.searchsorted(ts, side="left")), 0), len(df) - 1)

    for n, tr in enumerate(trades, 1):
        ex = pos(tr.entry_ts)
        xx = pos(tr.exit_ts)
        ax.scatter(ex, tr.entry_price, marker="^", s=100, color="#38bdf8", edgecolor="#111827", zorder=5)
        ax.scatter(xx, tr.exit_price, marker="v", s=100, color="#f97316", edgecolor="#111827", zorder=5)
        ax.plot([ex, xx], [tr.entry_price, tr.exit_price], color="#facc15", linewidth=1.4, alpha=0.8)
        ax.text(ex, tr.entry_price, f" E{n}", color="#c9d1d9", fontsize=9, va="bottom")
        ax.text(xx, tr.exit_price, f" X{n} {tr.return_pct:.0%}", color="#c9d1d9", fontsize=9, va="top")

    ax_score.plot(x, scores.to_numpy(dtype=float), color="#a78bfa", linewidth=1.4)
    ax_score.axhline(0.30, color="#f97316", linestyle="--", linewidth=0.9)
    ax_score.set_ylim(-0.03, 1.03)
    ticks = np.linspace(0, max(len(df) - 1, 0), min(10, len(df)), dtype=int)
    ax_score.set_xticks(ticks)
    ax_score.set_xticklabels([df.index[i].strftime("%Y-%m-%d") for i in ticks], rotation=30, ha="right")
    total = sum(t.return_pct for t in trades)
    ax.set_title(
        f"{ticker} multi-entry replay | {start.date()} to {end.date()} | trades={len(trades)} | sum returns={total:.1%}",
        color="#c9d1d9",
    )
    ax.set_ylabel("Price", color="#c9d1d9")
    ax_score.set_ylabel("Survival score", color="#c9d1d9")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_multi_entry.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run(matrix_path: Path, out_dir: Path) -> None:
    matrix = pd.read_parquet(matrix_path, columns=["expansion_survival_score", "fwd_max_return"])
    cases = [
        ("MU", pd.Timestamp("2025-01-03 00:00:00+00:00"), pd.Timestamp("2025-01-24 23:59:00+00:00")),
        ("AAOI", pd.Timestamp("2024-08-28 00:00:00+00:00"), pd.Timestamp("2024-09-25 23:59:00+00:00")),
    ]
    rows = []
    for ticker, start, end in cases:
        mt = matrix.xs(ticker, level="ticker").sort_index()
        b1 = _add_1h_indicators(load_1h(ticker))
        b4 = _add_4h_indicators(load_4h(ticker))
        trades = _simulate(ticker, start, end, matrix_ticker=mt, bars_1h=b1, bars_4h=b4, min_score=0.30, max_hours=8)
        path = _plot(ticker, start, end, b4, mt, trades, out_dir)
        for tr in trades:
            rows.append({**tr.__dict__, "plot": str(path)})
    summary = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "multi_entry_replay_trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.matrix, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
