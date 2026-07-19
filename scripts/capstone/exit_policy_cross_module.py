"""
Cross-module exit-policy comparison: run the SAME exit mechanic (from
backtest_exits.py / exit_policy_guided_search.py) against each module's OWN
top-10 OOF entry stream, not just Meta's.

Why OOF and not the deployed/live scores: leakage_audit.md §4.3 documents that
meta's live scores partially in-sample-trained a prefix of the 2025-07-01+
holdout; the same risk applies to momentum/HTF's deployed scores. Each
module's own oof_preds.parquet (walk-forward, held out) avoids that, so all
three modules are compared on equally clean ground. Momentum/HTF's OOF is
scored on their OWN historical universe/labels, not re-derived from meta's
matrix, so entry SETS will differ from what meta selects for the same names.

Same val/test split throughout this thread:
  VAL  2025-07-01 -> 2026-01-15
  TEST 2026-01-15 -> 2026-05-15

Policies compared (fixed, not re-searched — this is a cross-module transfer
check of round-1/round-2's winners, not a new per-module search):
  current-live, deployed, g284 (harvester), id4 (tail-rider)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_cross_module.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "signals/meta_context/meta_ranker"))
import backtest_exits as be  # noqa: E402

VAL_START = pd.Timestamp("2025-07-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-15", tz="UTC")
TEST_END = pd.Timestamp("2026-05-15", tz="UTC")
TOPK = 10

OOF_SOURCES = {
    "momentum": REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet",
    "htf": REPO / "strategies/multi_ticker_swing_htf/models/oof_preds.parquet",
}
META_SCORED = be.SCORED  # /tmp/meta_scored.parquet, score col = s_combo

POLICIES = {
    "current-live": dict(stop=None, trail=None, target=None, scale_frac=1.0, horizon=None, grace=0),
    "deployed": dict(stop=0.50, trail=0.35, target=0.20, scale_frac=0.5, horizon=25, grace=None),
    "g284 harvester": dict(stop=0.59, trail=None, target=0.07, scale_frac=1.0, horizon=60, grace=None),
    "id4 tail-rider": dict(stop=0.39, trail=None, target=0.30, scale_frac=0.16, horizon=53, grace=None),
}

_BAR_CACHE: dict[str, pd.DataFrame | None] = {}


def _bars(ticker: str) -> pd.DataFrame | None:
    if ticker not in _BAR_CACHE:
        _BAR_CACHE[ticker] = be._ticker_path(ticker, None)
    return _BAR_CACHE[ticker]


def load_member(module: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if module == "meta":
        df = pd.read_parquet(META_SCORED).dropna(subset=["s_combo"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        score_col = "s_combo"
    else:
        df = pd.read_parquet(OOF_SOURCES[module]).reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["score"])
        score_col = "score"
    df["rk"] = df.groupby("timestamp")[score_col].rank(ascending=False, method="first")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["in_top"] = df["rk"] <= TOPK
    return df[["timestamp", "ticker", "in_top"]]


def simulate(member: pd.DataFrame, *, stop=None, target=None, scale_frac=1.0,
             trail=None, grace=0, horizon=None) -> dict:
    rets, holds = [], []
    for ticker, g in member.groupby("ticker"):
        g = g.sort_values("timestamp")
        bars = _bars(ticker)
        if bars is None:
            continue
        m = g.set_index("timestamp")["in_top"].reindex(bars.index).fillna(False).astype(bool).values
        close, high, low = bars["close"].values, bars["high"].values, bars["low"].values
        n = len(bars)
        i = 0
        while i < n - 1:
            if not m[i]:
                i += 1
                continue
            entry = close[i]
            if entry <= 0:
                i += 1
                continue
            peak = entry
            realized = 0.0
            remaining = 1.0
            trimmed = False
            out = 0
            j = i + 1
            exit_ret = None
            while j < n and (j - i) <= be.MAX_HOLD:
                peak = max(peak, high[j])
                lo_ret = low[j] / entry - 1
                hi_ret = high[j] / entry - 1
                if stop is not None and lo_ret <= -stop:
                    exit_ret = -stop
                    break
                if trail is not None and low[j] <= peak * (1 - trail):
                    exit_ret = peak * (1 - trail) / entry - 1
                    break
                if target is not None and not trimmed and hi_ret >= target:
                    if scale_frac >= 1.0:
                        exit_ret = target
                        break
                    realized += scale_frac * target
                    remaining = 1.0 - scale_frac
                    trimmed = True
                out = out + 1 if not m[j] else 0
                if grace is not None and out > grace:
                    exit_ret = close[j] / entry - 1
                    break
                if horizon is not None and (j - i) >= horizon:
                    exit_ret = close[j] / entry - 1
                    break
                j += 1
            if exit_ret is None:
                jj = min(j, n - 1)
                exit_ret = close[jj] / entry - 1
            total = realized + remaining * exit_ret
            rets.append(total)
            holds.append(min(j, n - 1) - i)
            i = min(j, n - 1) + 1
    r = np.array(rets)
    h = np.array(holds)
    if len(r) == 0:
        return {}
    return {
        "n": len(r), "mean": r.mean(), "median": np.median(r), "win": (r > 0).mean(),
        "std": r.std(), "avg_hold": h.mean(), "ret_per_bar": (r / np.maximum(h, 1)).mean(),
        "total_ret": r.sum(),
    }


def main() -> None:
    rows = []
    for module in ("momentum", "htf", "meta"):
        val_m = load_member(module, VAL_START, VAL_END)
        test_m = load_member(module, VAL_END, TEST_END)
        print(f"{module}: val entries-in-top10 rows={val_m['in_top'].sum()}  "
              f"test entries-in-top10 rows={test_m['in_top'].sum()}  "
              f"unique tickers ever top10 (val+test)={pd.concat([val_m, test_m])[lambda d: d['in_top']]['ticker'].nunique()}")
        for pname, cfg in POLICIES.items():
            sv = simulate(val_m, **cfg)
            st = simulate(test_m, **cfg)
            if not sv or not st:
                continue
            rows.append(dict(module=module, policy=pname,
                              val_n=sv["n"], val_mean=sv["mean"], val_median=sv["median"],
                              val_win=sv["win"], val_rpb=sv["ret_per_bar"], val_tot=sv["total_ret"],
                              val_hold=sv["avg_hold"],
                              test_n=st["n"], test_mean=st["mean"], test_median=st["median"],
                              test_win=st["win"], test_rpb=st["ret_per_bar"], test_tot=st["total_ret"],
                              test_hold=st["avg_hold"]))
            print(f"  [{module:9s}] {pname:16s} val(n={sv['n']:4d} mean={sv['mean']*100:6.2f}% "
                  f"med={sv['median']*100:6.2f}% win={sv['win']*100:5.1f}% rpb={sv['ret_per_bar']:.4f}) "
                  f"test(n={st['n']:4d} mean={st['mean']*100:6.2f}% med={st['median']*100:6.2f}% "
                  f"win={st['win']*100:5.1f}% rpb={st['ret_per_bar']:.4f} hold={st['avg_hold']:5.1f})")

    out = pd.DataFrame(rows)
    out.to_csv(REPO / "research/capstone/exit_policy_cross_module.csv", index=False)
    print(f"\nsaved research/capstone/exit_policy_cross_module.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
