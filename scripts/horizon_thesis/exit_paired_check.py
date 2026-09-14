"""Paired exit-policy comparison on the untouched window: same entries, different exits.

`exit_policy_fresh_window_check.py` compares policies on the trade sets each one
generates, so momentum's "deployed" row is 328 trades and id4's is 251 — different
entries, different dates, unpaired means. This pairs them on identical
(ticker, entry bar) and reports the paired difference, which isolates the exit rule.

Two numbers per policy, because id4 holds ~2x longer than the old default and a
per-trade mean mechanically favours longer holds in a rising tape:
  * return per trade
  * return per BAR held (capital-time)

SAME CAVEAT AS THE SOURCE SCRIPT: the fresh window is scored with the DEPLOYED
boosters, which have very likely seen it during training. This is a direction
check, not validation. HTF is excluded — its features_4h.parquet ends 2026-06-02.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/capstone"))

import exit_policy_segmentation as seg  # noqa: E402

DATA = REPO / "research/execution_quality/data"
FRESH = REPO / "research/capstone/segmentation/fresh_window_directional_check"
MAX_RANK_KEPT = 50
TOPK = seg.TOPK
SEED = 19


def stream_from_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["rk"] = df.groupby("timestamp")["score"].rank(ascending=False, method="first")
    df = df[df["rk"] <= MAX_RANK_KEPT].copy()
    df["in_top"] = df["rk"] <= TOPK
    return df[["timestamp", "ticker", "in_top"]]


def day_cluster_boot(v: np.ndarray, days: np.ndarray, rng, n=3000) -> tuple[float, float, float]:
    uniq, inv = np.unique(days, return_inverse=True)
    groups = [v[inv == i] for i in range(len(uniq))]
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        out[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), float(2 * min((out <= 0).mean(), (out >= 0).mean()))


def main() -> None:
    rng = np.random.default_rng(SEED)
    results: dict = {}
    for module in ("momentum", "meta"):
        path = FRESH / f"deployed_scores_{module}.csv"
        if not path.exists():
            print(f"{module}: no scores file, skipped")
            continue
        member = stream_from_scores(path)
        print(f"\n{'=' * 96}\n{module}: {member['timestamp'].nunique()} bars, "
              f"{member['ticker'].nunique()} tickers (DIRECTIONAL ONLY — deployed-score leakage)")
        trades = {}
        for name, cfg in seg.POLICIES.items():
            t = seg.all_trades(member, **cfg)
            if t.empty:
                continue
            trades[name] = t.set_index(["ticker", "entry_ts"])
            print(f"  {name:14s} n={len(t):5d} mean/trade={t['ret'].mean() * 100:+7.2f}% "
                  f"mean/bar={(t['ret'] / t['bars_held'].clip(lower=1)).mean() * 100:+6.3f}% "
                  f"hold={t['bars_held'].mean():5.1f}")
        rows = []
        for a, b in (("deployed", "id4"), ("deployed", "g284"), ("current-live", "id4")):
            if a not in trades or b not in trades:
                continue
            common = trades[a].index.intersection(trades[b].index)
            if len(common) < 30:
                print(f"  {b} vs {a}: only {len(common)} shared entries, skipped")
                continue
            ra, rb = trades[a].loc[common], trades[b].loc[common]
            d_trade = (rb["ret"] - ra["ret"]).to_numpy()
            d_bar = ((rb["ret"] / rb["bars_held"].clip(lower=1))
                     - (ra["ret"] / ra["bars_held"].clip(lower=1))).to_numpy()
            days = np.array([pd.Timestamp(ts).normalize() for _t, ts in common])
            lo, hi, p = day_cluster_boot(d_trade, days, rng)
            blo, bhi, bp = day_cluster_boot(d_bar, days, rng)
            print(f"  PAIRED {b} - {a} on {len(common)} shared entries:")
            print(f"     per trade {d_trade.mean() * 100:+6.2f}pp [{lo * 100:+6.2f},{hi * 100:+6.2f}] p={p:.3f}"
                  f"   ({a} {ra['ret'].mean() * 100:+.2f}% vs {b} {rb['ret'].mean() * 100:+.2f}%)")
            print(f"     per bar   {d_bar.mean() * 100:+6.3f}pp [{blo * 100:+6.3f},{bhi * 100:+6.3f}] p={bp:.3f}"
                  f"   (holds {ra['bars_held'].mean():.1f} vs {rb['bars_held'].mean():.1f} bars)")
            rows.append(dict(a=a, b=b, n=int(len(common)), d_trade=float(d_trade.mean()),
                             ci_trade=[lo, hi], p_trade=p, d_bar=float(d_bar.mean()),
                             ci_bar=[blo, bhi], p_bar=bp,
                             mean_a=float(ra["ret"].mean()), mean_b=float(rb["ret"].mean()),
                             hold_a=float(ra["bars_held"].mean()), hold_b=float(rb["bars_held"].mean())))
        results[module] = rows
    out = DATA / "exit_paired_fresh_window.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
