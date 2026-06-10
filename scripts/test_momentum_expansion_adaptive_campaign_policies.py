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

from strategies.momentum_expansion.data.load_bars import load_1h, load_4h
from scripts.plot_momentum_expansion_order_policy_replay import (
    _add_1h_indicators,
    _add_4h_indicators,
    _entry_trigger,
    _exit_trade,
)


DEFAULT_MATRIX = Path("strategies/momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_OUT = Path("strategies/momentum_expansion/data/processed/adaptive_campaign_policy_experiment")
DEFAULT_ARCHETYPE_TRADES = Path(
    "strategies/momentum_expansion/data/processed/campaign_guard_experiment_top200/campaign_guard_trades.csv"
)


@dataclass(frozen=True)
class AdaptivePolicy:
    name: str
    max_entries: int
    entry2_min_score: float
    entry3_min_score: float
    entry4_min_score: float
    require_quality_after_two: bool = False
    quality_min_cum_return: float = 0.20
    runner_max_entries: int | None = None
    runner_entry3_min_score: float | None = None
    theme_proxy_max_entries: int | None = None
    theme_proxy_score: float | None = None


def _policies() -> list[AdaptivePolicy]:
    return [
        AdaptivePolicy(
            name="static_max2_reentry50",
            max_entries=2,
            entry2_min_score=0.50,
            entry3_min_score=9.99,
            entry4_min_score=9.99,
        ),
        AdaptivePolicy(
            name="score_ladder_50_75_85",
            max_entries=4,
            entry2_min_score=0.50,
            entry3_min_score=0.75,
            entry4_min_score=0.85,
        ),
        AdaptivePolicy(
            name="campaign_quality_20pct_80_85",
            max_entries=4,
            entry2_min_score=0.50,
            entry3_min_score=0.80,
            entry4_min_score=0.85,
            require_quality_after_two=True,
            quality_min_cum_return=0.20,
        ),
        AdaptivePolicy(
            name="ticker_runner_or_max2",
            max_entries=2,
            entry2_min_score=0.50,
            entry3_min_score=9.99,
            entry4_min_score=9.99,
            runner_max_entries=4,
            runner_entry3_min_score=0.70,
        ),
        AdaptivePolicy(
            name="theme_proxy_score80_allows4",
            max_entries=2,
            entry2_min_score=0.50,
            entry3_min_score=0.70,
            entry4_min_score=0.80,
            theme_proxy_max_entries=4,
            theme_proxy_score=0.80,
        ),
        AdaptivePolicy(
            name="combined_runner_theme_quality",
            max_entries=2,
            entry2_min_score=0.50,
            entry3_min_score=0.80,
            entry4_min_score=0.85,
            require_quality_after_two=True,
            quality_min_cum_return=0.20,
            runner_max_entries=4,
            runner_entry3_min_score=0.70,
            theme_proxy_max_entries=4,
            theme_proxy_score=0.80,
        ),
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


def _load_runner_tickers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    trades = pd.read_csv(path)
    if trades.empty or "ticker" not in trades.columns or "return_pct" not in trades.columns:
        return set()
    if "variant" in trades.columns:
        trades = trades[trades["variant"] == "baseline_max4_reentry50"].copy()
    if "is_reentry" in trades.columns:
        trades = trades[trades["is_reentry"].astype(bool)].copy()
    if trades.empty:
        return set()
    rows = []
    for ticker, g in trades.groupby("ticker"):
        if len(g) < 5:
            continue
        loss_gt8 = float((g["return_pct"] <= -0.08).mean())
        score = float(g["return_pct"].mean() + 0.05 * (g["return_pct"] > 0).mean() - 0.10 * loss_gt8)
        rows.append({"ticker": ticker, "archetype_score": score, "reentry_trades": len(g)})
    archetypes = pd.DataFrame(rows)
    if archetypes.empty:
        return set()
    cutoff = float(archetypes["archetype_score"].quantile(0.70))
    return set(archetypes.loc[archetypes["archetype_score"] >= cutoff, "ticker"].astype(str))


def _campaign_max_entries(policy: AdaptivePolicy, ticker: str, runner_tickers: set[str], score: float) -> int:
    max_entries = policy.max_entries
    if policy.runner_max_entries is not None and ticker in runner_tickers:
        max_entries = max(max_entries, policy.runner_max_entries)
    if (
        policy.theme_proxy_max_entries is not None
        and policy.theme_proxy_score is not None
        and np.isfinite(score)
        and score >= policy.theme_proxy_score
    ):
        max_entries = max(max_entries, policy.theme_proxy_max_entries)
    return max_entries


def _min_score_for_entry(
    policy: AdaptivePolicy,
    *,
    ticker: str,
    runner_tickers: set[str],
    campaign_entry_num: int,
    profitable_legs: int,
    cumulative_return: float,
) -> float:
    if campaign_entry_num <= 1:
        return 0.90
    if campaign_entry_num == 2:
        return policy.entry2_min_score
    if policy.require_quality_after_two and profitable_legs >= 2 and cumulative_return < policy.quality_min_cum_return:
        return 9.99
    if campaign_entry_num == 3:
        if policy.runner_entry3_min_score is not None and ticker in runner_tickers:
            return min(policy.entry3_min_score, policy.runner_entry3_min_score)
        return policy.entry3_min_score
    return policy.entry4_min_score


def _simulate(
    ticker: str,
    mt: pd.DataFrame,
    b1: pd.DataFrame,
    b4: pd.DataFrame,
    policy: AdaptivePolicy,
    runner_tickers: set[str],
    *,
    max_hours: int,
    reset_gap_days: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp(mt.index.min())
    end = pd.Timestamp(mt.index.max())
    campaign_entries = 0
    profitable_legs = 0
    cumulative_return = 0.0
    watch_ts = _next_watch_ts(mt, start - pd.Timedelta(nanoseconds=1), end, 0.90)

    while watch_ts is not None and watch_ts <= end:
        current_score = _score_at(mt, watch_ts)
        max_entries = _campaign_max_entries(policy, ticker, runner_tickers, current_score)
        if campaign_entries >= max_entries:
            campaign_entries = 0
            profitable_legs = 0
            cumulative_return = 0.0
            watch_ts = _next_watch_ts(mt, watch_ts, end, 0.90)
            continue

        next_entry_num = campaign_entries + 1
        min_score = _min_score_for_entry(
            policy,
            ticker=ticker,
            runner_tickers=runner_tickers,
            campaign_entry_num=next_entry_num,
            profitable_legs=profitable_legs,
            cumulative_return=cumulative_return,
        )
        if min_score > 1:
            campaign_entries = 0
            profitable_legs = 0
            cumulative_return = 0.0
            watch_ts = _next_watch_ts(mt, watch_ts, end, 0.90)
            continue

        if current_score < min_score:
            next_ts = _next_watch_ts(mt, watch_ts, end, min_score)
            if next_ts is None:
                if campaign_entries > 0:
                    campaign_entries = 0
                    profitable_legs = 0
                    cumulative_return = 0.0
                    watch_ts = _next_watch_ts(mt, watch_ts, end, 0.90)
                    continue
                break
            if campaign_entries > 0 and next_ts - watch_ts > pd.Timedelta(days=reset_gap_days):
                campaign_entries = 0
                profitable_legs = 0
                cumulative_return = 0.0
                watch_ts = _next_watch_ts(mt, watch_ts, end, 0.90)
                continue
            watch_ts = next_ts
            continue

        trig = _entry_trigger(b1, watch_ts, max_hours=max_hours)
        if trig is None:
            next_ts = _next_watch_ts(mt, watch_ts, end, min_score)
            if next_ts is None:
                break
            if campaign_entries > 0 and next_ts - watch_ts > pd.Timedelta(days=reset_gap_days):
                campaign_entries = 0
                profitable_legs = 0
                cumulative_return = 0.0
                watch_ts = _next_watch_ts(mt, watch_ts, end, 0.90)
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
                "variant": policy.name,
                "ticker": ticker,
                "is_runner_ticker": ticker in runner_tickers,
                "campaign_entry_num": next_entry_num,
                "is_reentry": next_entry_num > 1,
                "watch_ts": watch_ts,
                "entry_ts": entry_ts,
                "entry_price": float(entry_price),
                "entry_rule": entry_rule,
                "entry_score": entry_score,
                "min_score_required": min_score,
                "max_entries_allowed": max_entries,
                "exit_ts": exit_ts,
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "return_pct": ret,
                "profitable_legs_before_entry": profitable_legs,
                "cumulative_return_before_entry": cumulative_return,
            }
        )
        campaign_entries += 1
        cumulative_return += ret
        if ret > 0:
            profitable_legs += 1
        watch_ts = _next_watch_ts(mt, exit_ts, end, 0.50)
    return rows


def _summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, g in trades.groupby("variant"):
        winners = g.loc[g["return_pct"] > 0, "return_pct"]
        losers = g.loc[g["return_pct"] <= 0, "return_pct"]
        late = g[g["campaign_entry_num"] >= 3]
        runner = g[g["is_runner_ticker"]]
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
                "runner_trades": int(len(runner)),
                "runner_avg_return": float(runner["return_pct"].mean()) if len(runner) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_return", "total_return_units"], ascending=False)


def run(
    tickers: list[str],
    matrix_path: Path,
    out_dir: Path,
    archetype_trades: Path,
    max_hours: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_tickers = _load_runner_tickers(archetype_trades)
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
        for policy in _policies():
            rows.extend(_simulate(ticker, mt, b1, b4, policy, runner_tickers, max_hours=max_hours, reset_gap_days=15))

    trades = pd.DataFrame(rows)
    summary = _summarize(trades)
    by_ticker = (
        trades.groupby(["variant", "ticker"])
        .agg(
            trades=("ticker", "size"),
            avg_return=("return_pct", "mean"),
            win_rate=("return_pct", lambda s: float((s > 0).mean())),
            late_entries=("campaign_entry_num", lambda s: int((s >= 3).sum())),
        )
        .reset_index()
    )
    trades.to_csv(out_dir / "adaptive_campaign_trades.csv", index=False)
    summary.to_csv(out_dir / "adaptive_campaign_summary.csv", index=False)
    by_ticker.to_csv(out_dir / "adaptive_campaign_by_ticker.csv", index=False)
    (out_dir / "runner_tickers.txt").write_text("\n".join(sorted(runner_tickers)))
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tickers": tickers,
                "runner_tickers": len(runner_tickers),
                "archetype_source": str(archetype_trades),
                "max_hours": max_hours,
                "theme_proxy": "current survival_score >= 0.80; placeholder until real theme_expansion signal is wired",
                "note": "Labelled-score replay; not a no-lookahead trained-model backtest.",
            },
            indent=2,
            default=str,
        )
    )
    print("Summary")
    print(summary.to_string(index=False))
    print()
    print(f"Runner tickers loaded: {len(runner_tickers)}")
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--archetype-trades", type=Path, default=DEFAULT_ARCHETYPE_TRADES)
    parser.add_argument("--max-hours", type=int, default=8)
    args = parser.parse_args()
    run(args.tickers, args.matrix, args.out, args.archetype_trades, args.max_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
