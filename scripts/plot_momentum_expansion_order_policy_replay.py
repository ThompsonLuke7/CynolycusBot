from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.momentum_expansion.config.momentum_config import OPTION_POLICY_CONFIG
from strategies.momentum_expansion.data.load_bars import load_1h, load_4h


DEFAULT_MATRIX = Path("strategies/momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_OUT = Path("strategies/momentum_expansion/plots/output/order_policy_replay")


@dataclass
class ReplayTrade:
    ticker: str
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    score: float
    fwd_max_return: float
    return_pct: float


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def _is_rth(index: pd.DatetimeIndex) -> np.ndarray:
    ny = index.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    return np.asarray((minutes >= 9 * 60 + 30) & (minutes <= 16 * 60))


def _add_1h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out["ema10"] = out["close"].ewm(span=10, adjust=False).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["atr14"] = _atr(out, 14)
    return out


def _add_4h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["atr14"] = _atr(out, 14)
    return out


def _entry_trigger(
    bars_1h: pd.DataFrame,
    signal_ts: pd.Timestamp,
    *,
    max_hours: int,
) -> tuple[pd.Timestamp, float, str] | None:
    pos = bars_1h.index.searchsorted(signal_ts, side="right")
    window = bars_1h.iloc[pos:].loc[: signal_ts + pd.Timedelta(hours=max_hours)]
    window = window.loc[_is_rth(window.index)]
    if window.empty:
        return None
    for ts, row in window.iterrows():
        if not np.isfinite(row["atr14"]) or row["atr14"] <= 0:
            continue
        prev_high = float(bars_1h.loc[:ts]["high"].iloc[-2]) if len(bars_1h.loc[:ts]) >= 2 else np.nan
        if np.isfinite(prev_high) and row["high"] >= prev_high and row["close"] > row["open"]:
            return ts, float(row["close"]), "break_body_prev_high"

        prior = bars_1h.loc[:ts].tail(13).iloc[:-1]
        if len(prior) >= 12:
            trend_ok = bool((prior["ema10"].tail(5) > prior["ema20"].tail(5)).all())
            look_high = prior["high"].iloc[-11:-1].max()
            pullback_atr = (look_high - prior["low"].tail(3).min()) / row["atr14"]
            if trend_ok and 0.4 <= pullback_atr <= 2.5 and row["close"] > row["ema10"]:
                return ts, float(row["close"]), "pullback_continuation"
    return None


def _score_at(matrix_ticker: pd.DataFrame, ts: pd.Timestamp) -> float:
    idx = matrix_ticker.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(matrix_ticker["expansion_survival_score"].iloc[idx])


def _exit_trade(
    bars_4h: pd.DataFrame,
    matrix_ticker: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    *,
    score_decay_exit: float | None,
    score_decay_min_bars: int,
    atr_trail_arm: float,
    atr_trail_distance: float,
    trend_break_atr: float | None,
    max_holding_4h_bars: int,
) -> tuple[pd.Timestamp, float, str]:
    cfg = OPTION_POLICY_CONFIG
    pos = bars_4h.index.searchsorted(entry_ts, side="right")
    if pos >= len(bars_4h):
        return entry_ts, entry_price, "no_future_bars"
    entry_row = bars_4h.iloc[max(0, pos - 1)]
    entry_atr = float(entry_row["atr14"])
    if not np.isfinite(entry_atr) or entry_atr <= 0:
        entry_atr = max(entry_price * 0.04, 0.01)
    initial_stop = entry_price - float(cfg["atr_stop_mult"]) * 1.1 * entry_atr
    trail_armed = False
    trail_high = entry_price
    last_ts = bars_4h.index[pos]
    last_close = float(bars_4h["close"].iloc[pos])

    for held, (ts, row) in enumerate(bars_4h.iloc[pos:].iterrows(), start=1):
        close = float(row["close"])
        last_ts, last_close = ts, close
        favorable = close - entry_price
        if not trail_armed and favorable >= float(atr_trail_arm) * entry_atr:
            trail_armed = True
            trail_high = close
        if trail_armed:
            trail_high = max(trail_high, close)
            trail_stop = trail_high - float(atr_trail_distance) * entry_atr
            if close <= trail_stop:
                return ts, close, "trail_stop"
        if close <= initial_stop:
            return ts, close, "initial_stop"
        if trend_break_atr is not None and close < float(row["ema20"]) - float(trend_break_atr) * float(row["atr14"]):
            return ts, close, "trend_break"
        if score_decay_exit is not None and held >= int(score_decay_min_bars):
            score = _score_at(matrix_ticker, ts)
            if np.isfinite(score) and score < float(score_decay_exit):
                return ts, close, "score_decay"
        if held >= int(max_holding_4h_bars):
            return ts, close, "time_stop"
    return last_ts, last_close, "end_of_data"


def _select_signals(matrix_ticker: pd.DataFrame, max_plots: int) -> pd.DataFrame:
    work = matrix_ticker.dropna(subset=["expansion_survival_score", "fwd_max_return"]).copy()
    work = work.sort_values("expansion_survival_score", ascending=False)
    chosen = []
    used: list[pd.Timestamp] = []
    for ts, row in work.iterrows():
        if any(abs((ts - old).total_seconds()) < 20 * 24 * 3600 for old in used):
            continue
        chosen.append((ts, row))
        used.append(ts)
        if len(chosen) >= max_plots:
            break
    return pd.DataFrame([r for _, r in chosen], index=[ts for ts, _ in chosen])


def _plot_trade(
    ticker: str,
    bars_4h: pd.DataFrame,
    matrix_ticker: pd.DataFrame,
    trade: ReplayTrade,
    out_dir: Path,
) -> Path:
    start = trade.signal_ts - pd.Timedelta(days=35)
    end = trade.exit_ts + pd.Timedelta(days=10)
    df = bars_4h.loc[start:end].copy()
    score = matrix_ticker["expansion_survival_score"].reindex(df.index, method="ffill")
    x = np.arange(len(df))

    fig, (ax_price, ax_score) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax_price, ax_score):
        ax.set_facecolor("#0d1117")
        ax.grid(True, color="#30363d", alpha=0.5)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    for i, row in enumerate(df.itertuples()):
        color = "#2dd4bf" if row.close >= row.open else "#fb7185"
        ax_price.vlines(i, row.low, row.high, color=color, linewidth=1.0, alpha=0.9)
        ax_price.vlines(i, row.open, row.close, color=color, linewidth=4.0, alpha=0.9)

    def _mark(ts: pd.Timestamp, y: float, label: str, color: str, marker: str) -> None:
        if ts < df.index.min() or ts > df.index.max():
            return
        loc = int(df.index.searchsorted(ts, side="left"))
        loc = min(max(loc, 0), len(df) - 1)
        ax_price.scatter(loc, y, color=color, marker=marker, s=90, edgecolor="#111827", zorder=5, label=label)

    _mark(trade.signal_ts, float(df["low"].min()), "label/prediction bar", "#facc15", "^")
    _mark(trade.entry_ts, trade.entry_price, "1H confirmed entry", "#38bdf8", "^")
    _mark(trade.exit_ts, trade.exit_price, f"exit: {trade.exit_reason}", "#f97316", "v")
    ax_price.axhline(trade.entry_price, color="#38bdf8", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_price.axhline(trade.exit_price, color="#f97316", linestyle="--", linewidth=0.9, alpha=0.7)

    ax_score.plot(x, score.to_numpy(dtype=float), color="#a78bfa", linewidth=1.4)
    ax_score.axhline(float(OPTION_POLICY_CONFIG["score_decay_exit"]), color="#f97316", linestyle="--", linewidth=0.9)
    ax_score.set_ylim(-0.03, 1.03)

    tick_locs = np.linspace(0, max(len(df) - 1, 0), min(10, len(df)), dtype=int)
    ax_score.set_xticks(tick_locs)
    ax_score.set_xticklabels([df.index[i].strftime("%Y-%m-%d") for i in tick_locs], rotation=30, ha="right")
    ret = trade.return_pct * 100.0
    ax_price.set_title(
        f"{ticker} labelled-bar replay | signal {trade.signal_ts.date()} | "
        f"score={trade.score:.3f} | fwd max={trade.fwd_max_return:.1%} | "
        f"policy return={ret:.1f}% | exit={trade.exit_reason}",
        color="#c9d1d9",
    )
    ax_price.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax_price.set_ylabel("Price", color="#c9d1d9")
    ax_score.set_ylabel("Survival score", color="#c9d1d9")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_{trade.signal_ts.strftime('%Y%m%d_%H%M')}_{trade.exit_reason}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run(tickers: list[str], matrix_path: Path, out_dir: Path, max_plots_per_ticker: int, max_hours: int) -> None:
    matrix = pd.read_parquet(
        matrix_path,
        columns=["expansion_survival_score", "fwd_max_return"],
    )
    summary_rows = []
    paths = []
    for ticker in tickers:
        try:
            bars_4h = _add_4h_indicators(load_4h(ticker))
            bars_1h = _add_1h_indicators(load_1h(ticker))
            mt = matrix.xs(ticker, level="ticker").sort_index()
        except Exception as exc:
            summary_rows.append({"ticker": ticker, "error": str(exc)})
            continue
        signals = _select_signals(mt, max_plots_per_ticker)
        for signal_ts, signal_row in signals.iterrows():
            trig = _entry_trigger(bars_1h, signal_ts, max_hours=max_hours)
            if trig is None:
                summary_rows.append({"ticker": ticker, "signal_ts": signal_ts, "status": "no_entry_trigger"})
                continue
            entry_ts, entry_price, _entry_rule = trig
            exit_ts, exit_price, reason = _exit_trade(
                bars_4h,
                mt,
                entry_ts,
                entry_price,
                score_decay_exit=run.score_decay_exit,
                score_decay_min_bars=run.score_decay_min_bars,
                atr_trail_arm=run.atr_trail_arm,
                atr_trail_distance=run.atr_trail_distance,
                trend_break_atr=run.trend_break_atr,
                max_holding_4h_bars=run.max_holding_4h_bars,
            )
            trade = ReplayTrade(
                ticker=ticker,
                signal_ts=signal_ts,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                entry_price=entry_price,
                exit_price=exit_price,
                exit_reason=reason,
                score=float(signal_row["expansion_survival_score"]),
                fwd_max_return=float(signal_row["fwd_max_return"]),
                return_pct=exit_price / entry_price - 1.0,
            )
            path = _plot_trade(ticker, bars_4h, mt, trade, out_dir)
            paths.append(path)
            summary_rows.append(
                {
                    "ticker": ticker,
                    "signal_ts": signal_ts,
                    "entry_ts": entry_ts,
                    "entry_rule": _entry_rule,
                    "exit_ts": exit_ts,
                    "exit_reason": reason,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": trade.return_pct,
                    "score": trade.score,
                    "fwd_max_return": trade.fwd_max_return,
                    "plot": str(path),
                }
            )
    summary = pd.DataFrame(summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "order_policy_replay_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {len(paths)} plots -> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=["DELL", "MU", "AAOI"])
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-plots-per-ticker", type=int, default=3)
    parser.add_argument("--max-hours", type=int, default=8)
    parser.add_argument("--score-decay-exit", type=float, default=0.30)
    parser.add_argument("--score-decay-min-bars", type=int, default=0)
    parser.add_argument("--atr-trail-arm", type=float, default=1.0)
    parser.add_argument("--atr-trail-distance", type=float, default=2.4)
    parser.add_argument("--trend-break-atr", type=float, default=None)
    parser.add_argument("--max-holding-4h-bars", type=int, default=30)
    args = parser.parse_args()
    run.score_decay_exit = args.score_decay_exit
    run.score_decay_min_bars = args.score_decay_min_bars
    run.atr_trail_arm = args.atr_trail_arm
    run.atr_trail_distance = args.atr_trail_distance
    run.trend_break_atr = args.trend_break_atr
    run.max_holding_4h_bars = args.max_holding_4h_bars
    run(args.tickers, args.matrix, args.out, args.max_plots_per_ticker, args.max_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
