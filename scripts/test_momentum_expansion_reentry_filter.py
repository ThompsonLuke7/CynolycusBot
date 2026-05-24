from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momentum_expansion.data.load_bars import load_1h, load_4h
from scripts.plot_momentum_expansion_order_policy_replay import (
    _add_1h_indicators,
    _add_4h_indicators,
    _entry_trigger,
    _exit_trade,
)


DEFAULT_MATRIX = Path("momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_OUT = Path("momentum_expansion/data/processed/reentry_filter_experiment")


@dataclass(frozen=True)
class Trigger:
    ts: pd.Timestamp
    price: float
    rule: str
    score: float


def _is_rth(index: pd.DatetimeIndex) -> np.ndarray:
    ny = index.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    return np.asarray((minutes >= 9 * 60 + 30) & (minutes <= 16 * 60))


def _score_at(matrix_ticker: pd.DataFrame, ts: pd.Timestamp) -> float:
    idx = matrix_ticker.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(matrix_ticker["expansion_survival_score"].iloc[idx])


def _next_watch_ts(
    matrix_ticker: pd.DataFrame,
    after_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    min_score: float,
) -> pd.Timestamp | None:
    sub = matrix_ticker.loc[(matrix_ticker.index > after_ts) & (matrix_ticker.index <= end_ts)]
    sub = sub[sub["expansion_survival_score"] >= min_score]
    if sub.empty:
        return None
    return pd.Timestamp(sub.index[0])


def _strict_reentry_trigger(
    bars_1h: pd.DataFrame,
    matrix_ticker: pd.DataFrame,
    watch_ts: pd.Timestamp,
    *,
    max_hours: int,
    min_score: float,
) -> Trigger | None:
    pos = bars_1h.index.searchsorted(watch_ts, side="right")
    window = bars_1h.iloc[pos:].loc[: watch_ts + pd.Timedelta(hours=max_hours)]
    window = window.loc[_is_rth(window.index)]
    if window.empty:
        return None

    for ts, row in window.iterrows():
        score = _score_at(matrix_ticker, ts)
        if not np.isfinite(score) or score < min_score:
            continue
        if not np.isfinite(row["atr14"]) or row["atr14"] <= 0:
            continue

        prior_all = bars_1h.loc[:ts]
        if len(prior_all) < 2:
            continue
        prev_high = float(prior_all["high"].iloc[-2])
        if np.isfinite(prev_high) and row["high"] >= prev_high and row["close"] > row["open"]:
            return Trigger(ts=ts, price=float(row["close"]), rule="break_body_prev_high", score=score)

        prior = prior_all.tail(13).iloc[:-1]
        if len(prior) < 12:
            continue
        trend_ok = bool((prior["ema10"].tail(5) > prior["ema20"].tail(5)).all())
        look_high = prior["high"].iloc[-11:-1].max()
        pullback_atr = (look_high - prior["low"].tail(3).min()) / row["atr14"]
        pullback_ok = bool(
            trend_ok
            and 0.4 <= pullback_atr <= 2.5
            and row["close"] > row["ema10"]
            and row["close"] > row["open"]
        )
        if pullback_ok:
            return Trigger(ts=ts, price=float(row["close"]), rule="pullback_continuation_strict", score=score)
    return None


def _baseline_trigger(
    bars_1h: pd.DataFrame,
    matrix_ticker: pd.DataFrame,
    watch_ts: pd.Timestamp,
    *,
    max_hours: int,
) -> Trigger | None:
    trig = _entry_trigger(bars_1h, watch_ts, max_hours=max_hours)
    if trig is None:
        return None
    ts, price, rule = trig
    return Trigger(ts=ts, price=float(price), rule=rule, score=_score_at(matrix_ticker, ts))


def _simulate_ticker(
    ticker: str,
    matrix_ticker: pd.DataFrame,
    bars_1h: pd.DataFrame,
    bars_4h: pd.DataFrame,
    *,
    variant: str,
    initial_min_score: float,
    reentry_min_score: float,
    max_hours: int,
    max_campaign_entries: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp(matrix_ticker.index.min())
    end = pd.Timestamp(matrix_ticker.index.max())
    watch_ts = _next_watch_ts(matrix_ticker, start - pd.Timedelta(nanoseconds=1), end, initial_min_score)
    in_campaign_entries = 0

    while watch_ts is not None and watch_ts <= end:
        is_reentry = in_campaign_entries > 0
        if is_reentry and variant == "strict_reentry":
            trig = _strict_reentry_trigger(
                bars_1h,
                matrix_ticker,
                watch_ts,
                max_hours=max_hours,
                min_score=reentry_min_score,
            )
        else:
            trig = _baseline_trigger(bars_1h, matrix_ticker, watch_ts, max_hours=max_hours)
            if trig is not None and is_reentry and trig.score < reentry_min_score:
                trig = None

        if trig is None:
            # If no re-entry appears soon, end the current campaign and wait
            # for a fresh high-score setup.
            next_min = reentry_min_score if is_reentry else initial_min_score
            next_ts = _next_watch_ts(matrix_ticker, watch_ts, end, next_min)
            if next_ts is None:
                break
            if is_reentry and next_ts - watch_ts > pd.Timedelta(days=15):
                in_campaign_entries = 0
                next_ts = _next_watch_ts(matrix_ticker, watch_ts, end, initial_min_score)
                if next_ts is None:
                    break
            watch_ts = next_ts
            continue

        if trig.ts > end:
            break

        exit_ts, exit_price, exit_reason = _exit_trade(
            bars_4h,
            matrix_ticker,
            trig.ts,
            trig.price,
            score_decay_exit=0.30,
            score_decay_min_bars=0,
            atr_trail_arm=1.0,
            atr_trail_distance=2.4,
            trend_break_atr=None,
            max_holding_4h_bars=30,
        )
        ret = float(exit_price / trig.price - 1.0)
        rows.append(
            {
                "variant": variant,
                "ticker": ticker,
                "entry_num_in_campaign": int(in_campaign_entries + 1),
                "is_reentry": bool(is_reentry),
                "watch_ts": watch_ts,
                "entry_ts": trig.ts,
                "entry_price": trig.price,
                "entry_rule": trig.rule,
                "entry_score": trig.score,
                "exit_ts": exit_ts,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "return_pct": ret,
            }
        )
        in_campaign_entries += 1
        if in_campaign_entries >= max_campaign_entries:
            in_campaign_entries = 0
            watch_ts = _next_watch_ts(matrix_ticker, exit_ts, end, initial_min_score)
        else:
            watch_ts = _next_watch_ts(matrix_ticker, exit_ts, end, reentry_min_score)
    return rows


def _summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, subset), g in [
        ((v, "all"), gg) for v, gg in trades.groupby("variant")
    ] + [
        ((v, "reentries"), gg[gg["is_reentry"]]) for v, gg in trades.groupby("variant")
    ]:
        if g.empty:
            continue
        losers = g.loc[g["return_pct"] <= 0, "return_pct"]
        winners = g.loc[g["return_pct"] > 0, "return_pct"]
        rows.append(
            {
                "variant": variant,
                "subset": subset,
                "trades": int(len(g)),
                "avg_return": float(g["return_pct"].mean()),
                "median_return": float(g["return_pct"].median()),
                "total_return_units": float(g["return_pct"].sum()),
                "win_rate": float((g["return_pct"] > 0).mean()),
                "avg_winner": float(winners.mean()) if len(winners) else np.nan,
                "avg_loser": float(losers.mean()) if len(losers) else np.nan,
                "pct_loss_gt_8": float((g["return_pct"] <= -0.08).mean()),
                "pct_return_gt_20": float((g["return_pct"] >= 0.20).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["subset", "avg_return"], ascending=[True, False])


def run(
    tickers: list[str],
    matrix_path: Path,
    out_dir: Path,
    initial_min_score: float,
    reentry_min_score: float,
    max_hours: int,
    max_campaign_entries: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_parquet(matrix_path, columns=["expansion_survival_score", "fwd_max_return"])
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        try:
            mt = matrix.xs(ticker, level="ticker").sort_index()
            b1 = _add_1h_indicators(load_1h(ticker))
            b4 = _add_4h_indicators(load_4h(ticker))
        except Exception as exc:
            print(f"[skip] {ticker}: {exc}")
            continue
        for variant in ("baseline", "strict_reentry"):
            rows.extend(
                _simulate_ticker(
                    ticker,
                    mt,
                    b1,
                    b4,
                    variant=variant,
                    initial_min_score=initial_min_score,
                    reentry_min_score=reentry_min_score,
                    max_hours=max_hours,
                    max_campaign_entries=max_campaign_entries,
                )
            )
    trades = pd.DataFrame(rows)
    summary = _summarize(trades)
    trades.to_csv(out_dir / "reentry_filter_trades.csv", index=False)
    summary.to_csv(out_dir / "reentry_filter_summary.csv", index=False)
    by_ticker = (
        trades.groupby(["variant", "ticker", "is_reentry"])
        .agg(
            trades=("ticker", "size"),
            avg_return=("return_pct", "mean"),
            win_rate=("return_pct", lambda s: float((s > 0).mean())),
            loss_gt_8=("return_pct", lambda s: float((s <= -0.08).mean())),
        )
        .reset_index()
    )
    by_ticker.to_csv(out_dir / "reentry_filter_by_ticker.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tickers": tickers,
                "initial_min_score": initial_min_score,
                "reentry_min_score": reentry_min_score,
                "max_hours": max_hours,
                "max_campaign_entries": max_campaign_entries,
                "note": "Labelled-score replay, not a no-lookahead trained-model backtest.",
            },
            indent=2,
            default=str,
        )
    )
    print("Summary")
    print(summary.to_string(index=False))
    print()
    print("By ticker")
    print(by_ticker.to_string(index=False))
    print()
    print("Late AAOI-style losing reentries")
    bad = trades[(trades["is_reentry"]) & (trades["return_pct"] <= -0.08)]
    print(bad[["variant", "ticker", "watch_ts", "entry_ts", "entry_rule", "entry_score", "exit_reason", "return_pct"]].to_string(index=False))
    print()
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=["DELL", "MU", "AAOI"])
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--initial-min-score", type=float, default=0.90)
    parser.add_argument("--reentry-min-score", type=float, default=0.50)
    parser.add_argument("--max-hours", type=int, default=8)
    parser.add_argument("--max-campaign-entries", type=int, default=3)
    args = parser.parse_args()
    run(
        args.tickers,
        args.matrix,
        args.out,
        args.initial_min_score,
        args.reentry_min_score,
        args.max_hours,
        args.max_campaign_entries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
