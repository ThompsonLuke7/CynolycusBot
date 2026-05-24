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
DEFAULT_OUT = Path("momentum_expansion/data/processed/campaign_guard_experiment")


@dataclass(frozen=True)
class Variant:
    name: str
    max_entries_per_campaign: int
    initial_min_score: float
    reentry_min_score: float
    after_two_min_score: float | None
    reset_gap_days: int


def _variants() -> list[Variant]:
    return [
        Variant("baseline_max3_reentry50", 3, 0.90, 0.50, None, 15),
        Variant("max2_reentry50", 2, 0.90, 0.50, None, 15),
        Variant("max3_after2_score70", 3, 0.90, 0.50, 0.70, 15),
        Variant("max3_after2_score80", 3, 0.90, 0.50, 0.80, 15),
        Variant("max2_after2_score70", 2, 0.90, 0.50, 0.70, 15),
        Variant("baseline_max4_reentry50", 4, 0.90, 0.50, None, 15),
        Variant("max4_after2_score70", 4, 0.90, 0.50, 0.70, 15),
        Variant("max4_after2_score80", 4, 0.90, 0.50, 0.80, 15),
    ]


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


def _simulate(
    ticker: str,
    mt: pd.DataFrame,
    b1: pd.DataFrame,
    b4: pd.DataFrame,
    variant: Variant,
    *,
    max_hours: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp(mt.index.min())
    end = pd.Timestamp(mt.index.max())
    campaign_entries = 0
    profitable_legs = 0
    watch_ts = _next_watch_ts(mt, start - pd.Timedelta(nanoseconds=1), end, variant.initial_min_score)

    while watch_ts is not None and watch_ts <= end:
        is_reentry = campaign_entries > 0
        if is_reentry and campaign_entries >= variant.max_entries_per_campaign:
            campaign_entries = 0
            profitable_legs = 0
            watch_ts = _next_watch_ts(mt, watch_ts, end, variant.initial_min_score)
            continue

        min_score = variant.initial_min_score if not is_reentry else variant.reentry_min_score
        if is_reentry and variant.after_two_min_score is not None and profitable_legs >= 2:
            min_score = max(min_score, variant.after_two_min_score)
        if _score_at(mt, watch_ts) < min_score:
            next_ts = _next_watch_ts(mt, watch_ts, end, min_score)
            if next_ts is None:
                if is_reentry:
                    campaign_entries = 0
                    profitable_legs = 0
                    watch_ts = _next_watch_ts(mt, watch_ts, end, variant.initial_min_score)
                    continue
                break
            if is_reentry and next_ts - watch_ts > pd.Timedelta(days=variant.reset_gap_days):
                campaign_entries = 0
                profitable_legs = 0
                watch_ts = _next_watch_ts(mt, watch_ts, end, variant.initial_min_score)
                continue
            watch_ts = next_ts
            continue

        trig = _entry_trigger(b1, watch_ts, max_hours=max_hours)
        if trig is None:
            next_ts = _next_watch_ts(mt, watch_ts, end, min_score)
            if next_ts is None:
                break
            if is_reentry and next_ts - watch_ts > pd.Timedelta(days=variant.reset_gap_days):
                campaign_entries = 0
                profitable_legs = 0
                watch_ts = _next_watch_ts(mt, watch_ts, end, variant.initial_min_score)
                continue
            watch_ts = next_ts
            continue

        entry_ts, entry_price, entry_rule = trig
        entry_score = _score_at(mt, entry_ts)
        if entry_score < min_score:
            watch_ts = _next_watch_ts(mt, watch_ts, end, min_score)
            continue

        exit_ts, exit_price, exit_reason = _exit_trade(
            b4,
            mt,
            entry_ts,
            entry_price,
            score_decay_exit=0.30,
            score_decay_min_bars=0,
            atr_trail_arm=1.0,
            atr_trail_distance=2.4,
            trend_break_atr=None,
            max_holding_4h_bars=30,
        )
        ret = float(exit_price / entry_price - 1.0)
        rows.append(
            {
                "variant": variant.name,
                "ticker": ticker,
                "campaign_entry_num": int(campaign_entries + 1),
                "is_reentry": bool(is_reentry),
                "watch_ts": watch_ts,
                "entry_ts": entry_ts,
                "entry_price": float(entry_price),
                "entry_rule": entry_rule,
                "entry_score": entry_score,
                "exit_ts": exit_ts,
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "return_pct": ret,
                "profitable_legs_before_entry": int(profitable_legs),
            }
        )
        campaign_entries += 1
        if ret > 0:
            profitable_legs += 1
        watch_ts = _next_watch_ts(mt, exit_ts, end, variant.reentry_min_score)
    return rows


def _summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, g in trades.groupby("variant"):
        losers = g.loc[g["return_pct"] <= 0, "return_pct"]
        winners = g.loc[g["return_pct"] > 0, "return_pct"]
        late = g[g["campaign_entry_num"] >= 3]
        rows.append(
            {
                "variant": variant,
                "trades": int(len(g)),
                "avg_return": float(g["return_pct"].mean()),
                "median_return": float(g["return_pct"].median()),
                "total_return_units": float(g["return_pct"].sum()),
                "win_rate": float((g["return_pct"] > 0).mean()),
                "avg_winner": float(winners.mean()) if len(winners) else np.nan,
                "avg_loser": float(losers.mean()) if len(losers) else np.nan,
                "pct_loss_gt_8": float((g["return_pct"] <= -0.08).mean()),
                "pct_return_gt_20": float((g["return_pct"] >= 0.20).mean()),
                "reentry_trades": int(g["is_reentry"].sum()),
                "late_entry_trades": int(len(late)),
                "late_avg_return": float(late["return_pct"].mean()) if len(late) else np.nan,
                "aaoi_late_losses": int(
                    len(g[(g["ticker"] == "AAOI") & (g["campaign_entry_num"] >= 3) & (g["return_pct"] <= -0.08)])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_return", "total_return_units"], ascending=False)


def run(tickers: list[str], matrix_path: Path, out_dir: Path, max_hours: int) -> None:
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
        for variant in _variants():
            rows.extend(_simulate(ticker, mt, b1, b4, variant, max_hours=max_hours))
    trades = pd.DataFrame(rows)
    summary = _summarize(trades)
    by_ticker = (
        trades.groupby(["variant", "ticker"])
        .agg(
            trades=("ticker", "size"),
            avg_return=("return_pct", "mean"),
            win_rate=("return_pct", lambda s: float((s > 0).mean())),
            loss_gt_8=("return_pct", lambda s: float((s <= -0.08).mean())),
        )
        .reset_index()
    )
    trades.to_csv(out_dir / "campaign_guard_trades.csv", index=False)
    summary.to_csv(out_dir / "campaign_guard_summary.csv", index=False)
    by_ticker.to_csv(out_dir / "campaign_guard_by_ticker.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tickers": tickers,
                "max_hours": max_hours,
                "note": "Labelled-score campaign replay, not a no-lookahead trained-model backtest.",
            },
            indent=2,
            default=str,
        )
    )
    print("Summary")
    print(summary.to_string(index=False))
    print()
    print("AAOI Aug-Oct 2024")
    tmp = trades.copy()
    tmp["entry_ts"] = pd.to_datetime(tmp["entry_ts"], utc=True)
    aaoi = tmp[
        (tmp["ticker"] == "AAOI")
        & (tmp["entry_ts"].between("2024-08-20", "2024-10-01"))
    ]
    print(
        aaoi[
            [
                "variant",
                "campaign_entry_num",
                "entry_ts",
                "entry_rule",
                "entry_score",
                "exit_reason",
                "return_pct",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=["DELL", "MU", "AAOI"])
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-hours", type=int, default=8)
    args = parser.parse_args()
    run(args.tickers, args.matrix, args.out, args.max_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
