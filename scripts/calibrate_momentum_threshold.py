"""Calibrate the Momentum Expansion selection floor (RANKING_CONFIG['min_score']).

The live ranker's model outputs P(expansion) in ~[0, 0.6], but the shipped floor
is 0.85 -> nothing ever clears it -> the module never trades. This derives a
data-driven floor from the model's OWN score distribution and the realized
forward outcome, WITHOUT tuning on the final test set:

  * Temporal split (60/20/20) of the labeled training matrix.
  * VALIDATION is used to choose the threshold; TEST is a single confirm.
  * Population = candidate-filtered rows (what the live ranker actually scores).
  * Per threshold report: coverage, avg names/bar, precision (win-rate on the
    expansion_target relevance label), realized forward return, and expectancy.
  * Stability check across time sub-periods.

Run: .venv/bin/python scripts/calibrate_momentum_threshold.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from strategies.momentum_expansion.config.momentum_config import TRAINING_MATRIX
from strategies.momentum_expansion.inference.candidate_filter import filter_momentum_candidates
from strategies.momentum_expansion.inference.ranker import ExpansionRanker

# Outcome columns present in the labeled matrix.
LABEL_BINARY = "expansion_target"        # relevance label (top-quantile forward expansion)
FWD_RETURN = "fwd_max_return"            # best forward move over the 25-bar window
FWD_ALPHA = "fwd_max_alpha"              # forward move minus SPY
FWD_DD = "fwd_max_drawdown"             # worst forward drawdown (positive number)
FWD_CLOSE = "fwd_close_return"           # buy-and-hold close-to-horizon return

THRESHOLDS = [round(x, 3) for x in np.arange(0.20, 0.61, 0.025)]


def _score(ranker: ExpansionRanker, df: pd.DataFrame) -> pd.Series:
    return ranker.score(df)


def _sweep(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    n_bars = df.index.get_level_values("timestamp").nunique()
    base_rate = float(df[LABEL_BINARY].mean())
    rows = []
    for thr in THRESHOLDS:
        sel = df[df["_score"] >= thr]
        if sel.empty:
            rows.append({"thr": thr, "n": 0, "names_per_bar": 0.0, "coverage_%": 0.0,
                         "precision": np.nan, "lift": np.nan, "avg_fwd_ret_%": np.nan,
                         "avg_fwd_alpha_%": np.nan, "avg_fwd_dd_%": np.nan,
                         "expectancy_%": np.nan})
            continue
        prec = float(sel[LABEL_BINARY].mean())
        rows.append({
            "thr": thr,
            "n": int(len(sel)),
            "names_per_bar": round(len(sel) / max(1, n_bars), 2),
            "coverage_%": round(100 * len(sel) / len(df), 2),
            "precision": round(prec, 3),
            "lift": round(prec / base_rate, 2) if base_rate > 0 else np.nan,
            "avg_fwd_ret_%": round(100 * float(sel[FWD_RETURN].mean()), 2),
            "avg_fwd_alpha_%": round(100 * float(sel[FWD_ALPHA].mean()), 2),
            "avg_fwd_dd_%": round(100 * float(sel[FWD_DD].mean()), 2),
            # Expectancy of the realized close-to-horizon return (buy-and-hold proxy).
            "expectancy_%": round(100 * float(sel[FWD_CLOSE].mean()), 2),
        })
    out = pd.DataFrame(rows)
    print(f"\n=== {label}: threshold sweep "
          f"(rows={len(df):,}, bars={n_bars:,}, base positive-rate={base_rate:.3%}) ===")
    print(out.to_string(index=False))
    return out


def main() -> None:
    print(f"loading {TRAINING_MATRIX} ...")
    df = pd.read_parquet(TRAINING_MATRIX)
    df = df.sort_index(level="timestamp")
    ts = df.index.get_level_values("timestamp")
    print(f"matrix: {df.shape[0]:,} rows | {df.index.get_level_values('ticker').nunique():,} tickers "
          f"| {ts.min().date()} -> {ts.max().date()}")

    # Temporal 60/20/20 split by UNIQUE timestamp (all rows of a bar stay together).
    uts = pd.Index(ts.unique()).sort_values()
    q60 = uts[int(len(uts) * 0.60)]
    q80 = uts[int(len(uts) * 0.80)]
    train = df[ts <= q60]
    val = df[(ts > q60) & (ts <= q80)]
    test = df[ts > q80]
    print(f"split: train<= {q60.date()} ({len(train):,}) | "
          f"val {q60.date()}..{q80.date()} ({len(val):,}) | test > {q80.date()} ({len(test):,})")

    # Population = candidate-filtered (what the live ranker actually scores).
    val = filter_momentum_candidates(val)
    test = filter_momentum_candidates(test)
    val = val.dropna(subset=[c for c in [LABEL_BINARY, FWD_RETURN, FWD_CLOSE] if c in val.columns])
    test = test.dropna(subset=[c for c in [LABEL_BINARY, FWD_RETURN, FWD_CLOSE] if c in test.columns])
    print(f"after candidate filter + outcome dropna: val={len(val):,}  test={len(test):,}")

    ranker = ExpansionRanker()
    val = val.copy(); val["_score"] = _score(ranker, val).values
    test = test.copy(); test["_score"] = _score(ranker, test).values

    for name, d in [("VAL", val), ("TEST", test)]:
        s = d["_score"]
        print(f"{name} score dist: min={s.min():.3f} p50={s.median():.3f} "
              f"p90={s.quantile(.9):.3f} p99={s.quantile(.99):.3f} max={s.max():.3f}")

    val_sweep = _sweep(val, label="VALIDATION")

    # Stability: split VAL into 3 equal time sub-periods, sweep a few thresholds.
    vts = val.index.get_level_values("timestamp")
    vuts = pd.Index(vts.unique()).sort_values()
    edges = [vuts[0], vuts[int(len(vuts) / 3)], vuts[int(2 * len(vuts) / 3)], vuts[-1]]
    print("\n=== VALIDATION stability across 3 time sub-periods (precision @ thr) ===")
    stab = []
    for thr in [0.30, 0.35, 0.40, 0.45]:
        row = {"thr": thr}
        for i in range(3):
            lo, hi = edges[i], edges[i + 1]
            sub = val[(vts > lo) & (vts <= hi)] if i > 0 else val[vts <= hi]
            sel = sub[sub["_score"] >= thr]
            row[f"P{i+1}_prec"] = round(float(sel[LABEL_BINARY].mean()), 3) if len(sel) else np.nan
            row[f"P{i+1}_n/bar"] = round(len(sel) / max(1, sub.index.get_level_values('timestamp').nunique()), 2)
        stab.append(row)
    print(pd.DataFrame(stab).to_string(index=False))

    # Single confirm on TEST.
    _sweep(test, label="TEST (confirm only)")

    # Per-bar top-N ranking view (how the live system actually selects). This
    # isolates RANKING skill from absolute-probability calibration.
    for name, d in [("VALIDATION", val), ("TEST", test)]:
        base = float(d[LABEL_BINARY].mean())
        print(f"\n=== {name}: per-bar top-N ranking (base positive-rate={base:.3%}) ===")
        rows = []
        g = d.groupby(level="timestamp", sort=False)
        for topn in [1, 3, 5, 10, 20]:
            picks = g.apply(lambda x: x.nlargest(topn, "_score"))
            prec = float(picks[LABEL_BINARY].mean())
            rows.append({
                "top_n": topn,
                "precision": round(prec, 3),
                "lift": round(prec / base, 2) if base > 0 else np.nan,
                "avg_fwd_ret_%": round(100 * float(picks[FWD_RETURN].mean()), 2),
                "avg_fwd_alpha_%": round(100 * float(picks[FWD_ALPHA].mean()), 2),
                "expectancy_%": round(100 * float(picks[FWD_CLOSE].mean()), 2),
            })
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
