from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

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
    _select_signals,
    _score_at,
)


DEFAULT_MATRIX = Path("strategies/momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_OUT = Path("strategies/momentum_expansion/data/processed/exit_policy_sweep")


@dataclass(frozen=True)
class ExitConfig:
    name: str
    atr_stop_mult: float
    atr_trail_arm: float
    atr_trail_distance: float
    trend_break_atr: float | None
    score_decay_exit: float | None
    score_decay_min_bars: int
    max_holding_4h_bars: int


def _exit_configs() -> list[ExitConfig]:
    configs: list[ExitConfig] = []
    score_thresholds: list[float | None] = [0.40, 0.30, 0.20, None]
    score_min_bars = [0, 4, 8, 12]
    trail_arms = [1.0, 1.5]
    trail_distances = [1.2, 1.8, 2.4]
    trend_breaks: list[float | None] = [1.0, 1.5, None]
    max_holds = [20, 30, 40]

    for score, min_bars, arm, dist, trend, hold in product(
        score_thresholds, score_min_bars, trail_arms, trail_distances, trend_breaks, max_holds
    ):
        if score is None and min_bars != 0:
            continue
        name = (
            f"score_{'off' if score is None else str(score).replace('.', 'p')}"
            f"_min{min_bars}_trail{str(arm).replace('.', 'p')}x{str(dist).replace('.', 'p')}"
            f"_trend{'off' if trend is None else str(trend).replace('.', 'p')}_hold{hold}"
        )
        configs.append(
            ExitConfig(
                name=name,
                atr_stop_mult=1.5,
                atr_trail_arm=arm,
                atr_trail_distance=dist,
                trend_break_atr=trend,
                score_decay_exit=score,
                score_decay_min_bars=min_bars,
                max_holding_4h_bars=hold,
            )
        )
    return configs


def _exit_trade(
    bars_4h: pd.DataFrame,
    matrix_ticker: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    cfg: ExitConfig,
) -> tuple[pd.Timestamp, float, str, int, float]:
    pos = bars_4h.index.searchsorted(entry_ts, side="right")
    if pos >= len(bars_4h):
        return entry_ts, entry_price, "no_future_bars", 0, 0.0
    entry_row = bars_4h.iloc[max(0, pos - 1)]
    entry_atr = float(entry_row["atr14"])
    if not np.isfinite(entry_atr) or entry_atr <= 0:
        entry_atr = max(entry_price * 0.04, 0.01)
    initial_stop = entry_price - cfg.atr_stop_mult * 1.1 * entry_atr
    trail_armed = False
    trail_high = entry_price
    best_close = entry_price
    last_ts = bars_4h.index[pos]
    last_close = float(bars_4h["close"].iloc[pos])

    for held, (ts, row) in enumerate(bars_4h.iloc[pos:].iterrows(), start=1):
        close = float(row["close"])
        last_ts, last_close = ts, close
        best_close = max(best_close, close)
        favorable = close - entry_price
        if not trail_armed and favorable >= cfg.atr_trail_arm * entry_atr:
            trail_armed = True
            trail_high = close
        if trail_armed:
            trail_high = max(trail_high, close)
            trail_stop = trail_high - cfg.atr_trail_distance * entry_atr
            if close <= trail_stop:
                return ts, close, "trail_stop", held, best_close / entry_price - 1.0
        if close <= initial_stop:
            return ts, close, "initial_stop", held, best_close / entry_price - 1.0
        if cfg.trend_break_atr is not None:
            atr = float(row["atr14"])
            if np.isfinite(atr) and atr > 0 and close < float(row["ema20"]) - cfg.trend_break_atr * atr:
                return ts, close, "trend_break", held, best_close / entry_price - 1.0
        if cfg.score_decay_exit is not None and held >= cfg.score_decay_min_bars:
            score = _score_at(matrix_ticker, ts)
            if np.isfinite(score) and score < cfg.score_decay_exit:
                return ts, close, "score_decay", held, best_close / entry_price - 1.0
        if held >= cfg.max_holding_4h_bars:
            return ts, close, "time_stop", held, best_close / entry_price - 1.0
    return last_ts, last_close, "end_of_data", len(bars_4h.iloc[pos:]), best_close / entry_price - 1.0


def _build_entries(
    tickers: list[str],
    matrix: pd.DataFrame,
    *,
    max_signals_per_ticker: int,
    max_hours: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    entries: list[dict[str, object]] = []
    bars4: dict[str, pd.DataFrame] = {}
    matrices: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            b4 = _add_4h_indicators(load_4h(ticker))
            b1 = _add_1h_indicators(load_1h(ticker))
            mt = matrix.xs(ticker, level="ticker").sort_index()
        except Exception as exc:
            print(f"[skip] {ticker}: {exc}")
            continue
        bars4[ticker] = b4
        matrices[ticker] = mt
        signals = _select_signals(mt, max_signals_per_ticker)
        for signal_ts, signal_row in signals.iterrows():
            trig = _entry_trigger(b1, signal_ts, max_hours=max_hours)
            if trig is None:
                continue
            entry_ts, entry_price, entry_rule = trig
            pos = b4.index.searchsorted(entry_ts, side="right")
            if pos >= len(b4):
                continue
            entries.append(
                {
                    "ticker": ticker,
                    "signal_ts": signal_ts,
                    "entry_ts": entry_ts,
                    "entry_price": entry_price,
                    "entry_rule": entry_rule,
                    "score": float(signal_row["expansion_survival_score"]),
                    "label_fwd_max_return": float(signal_row["fwd_max_return"]),
                }
            )
    return pd.DataFrame(entries), bars4, matrices


def _summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cfg_name, g in trades.groupby("exit_policy"):
        winners = g.loc[g["return_pct"] > 0, "return_pct"]
        losers = g.loc[g["return_pct"] <= 0, "return_pct"]
        reasons = g["exit_reason"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "exit_policy": cfg_name,
                "trades": int(len(g)),
                "avg_return": float(g["return_pct"].mean()),
                "median_return": float(g["return_pct"].median()),
                "total_return_units": float(g["return_pct"].sum()),
                "win_rate": float((g["return_pct"] > 0).mean()),
                "avg_winner": float(winners.mean()) if len(winners) else np.nan,
                "avg_loser": float(losers.mean()) if len(losers) else np.nan,
                "avg_mfe_close": float(g["mfe_close_pct"].mean()),
                "avg_capture": float((g["return_pct"] / g["mfe_close_pct"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).mean()),
                "pct_return_gt_10": float((g["return_pct"] >= 0.10).mean()),
                "pct_return_gt_20": float((g["return_pct"] >= 0.20).mean()),
                "avg_bars_held": float(g["bars_held"].mean()),
                "score_decay_share": float(reasons.get("score_decay", 0.0)),
                "trail_stop_share": float(reasons.get("trail_stop", 0.0)),
                "trend_break_share": float(reasons.get("trend_break", 0.0)),
                "time_stop_share": float(reasons.get("time_stop", 0.0)),
                "initial_stop_share": float(reasons.get("initial_stop", 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["avg_return", "total_return_units", "avg_capture"], ascending=False
    )


def run(
    tickers: list[str],
    matrix_path: Path,
    out_dir: Path,
    max_signals_per_ticker: int,
    max_hours: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_parquet(matrix_path, columns=["expansion_survival_score", "fwd_max_return"])
    entries, bars4, matrices = _build_entries(
        tickers,
        matrix,
        max_signals_per_ticker=max_signals_per_ticker,
        max_hours=max_hours,
    )
    if entries.empty:
        raise RuntimeError("No confirmed entries found")

    rows: list[dict[str, object]] = []
    configs = _exit_configs()
    for cfg in configs:
        for entry in entries.itertuples(index=False):
            b4 = bars4[entry.ticker]
            mt = matrices[entry.ticker]
            exit_ts, exit_price, reason, bars_held, mfe_close_pct = _exit_trade(
                b4,
                mt,
                pd.Timestamp(entry.entry_ts),
                float(entry.entry_price),
                cfg,
            )
            ret = float(exit_price / float(entry.entry_price) - 1.0)
            rows.append(
                {
                    "exit_policy": cfg.name,
                    "ticker": entry.ticker,
                    "signal_ts": entry.signal_ts,
                    "entry_ts": entry.entry_ts,
                    "entry_price": entry.entry_price,
                    "entry_rule": entry.entry_rule,
                    "score": entry.score,
                    "label_fwd_max_return": entry.label_fwd_max_return,
                    "exit_ts": exit_ts,
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "bars_held": bars_held,
                    "mfe_close_pct": mfe_close_pct,
                    "return_pct": ret,
                }
            )
    trades = pd.DataFrame(rows)
    summary = _summarize(trades)
    reason_pivot = (
        trades.pivot_table(index="exit_policy", columns="exit_reason", values="ticker", aggfunc="count", fill_value=0)
        .reset_index()
    )
    best = summary.head(25).copy()
    trades.to_parquet(out_dir / "exit_policy_trades.parquet")
    summary.to_csv(out_dir / "exit_policy_summary.csv", index=False)
    best.to_csv(out_dir / "exit_policy_top25.csv", index=False)
    reason_pivot.to_csv(out_dir / "exit_policy_reason_counts.csv", index=False)
    entries.to_csv(out_dir / "confirmed_entries.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tickers": tickers,
                "matrix": str(matrix_path),
                "confirmed_entries": int(len(entries)),
                "exit_configs": int(len(configs)),
                "max_signals_per_ticker": int(max_signals_per_ticker),
                "max_entry_confirmation_hours": int(max_hours),
                "note": "Labelled-bar replay, not a real no-lookahead model backtest.",
            },
            indent=2,
            default=str,
        )
    )
    print("Confirmed entries")
    print(entries.groupby("ticker").size().to_string())
    print()
    print("Top exit policies")
    print(best.to_string(index=False))
    print()
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=["DELL", "MU", "AAOI"])
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-signals-per-ticker", type=int, default=35)
    parser.add_argument("--max-hours", type=int, default=8)
    args = parser.parse_args()
    run(args.tickers, args.matrix, args.out, args.max_signals_per_ticker, args.max_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
