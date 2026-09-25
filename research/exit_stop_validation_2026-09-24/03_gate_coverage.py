"""Step 3 -- WHY do 20% of pool-ticker days have no 4H feature row?

Stage 1 measured the hole: the momentum candidate gate is evaluated on only 80.2% of
pool-ticker sessions, and the P1 tail rate is 1.101% on evaluated rows against 1.507% across
all pool rows -- the tail is ~37% DENSER on the rows the pipeline has no entry for. That is a
retrieval loss that happens before any model or gate speaks.

This classifies the missing (ticker, session) pairs into causes, so the dominant one can be
fixed. Candidate causes, checked in order of precedence:

  no_4h_bars        the ticker has no 4H bar file at all
  before_first_bar  the session predates the ticker's first 4H bar
  after_last_bar    the session postdates its last 4H bar (delisted / halted / stale cache)
  warmup            inside the feature warm-up window after the first bar (features need
                    ~6 months of history; expansion_labels documents min_periods ~window/8)
  missing_bar       the 4H cache has no bar for that session though it spans it
  feature_nan       bars exist, the matrix has no row -> features were NaN and dropped
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/rl_reintroduction_2026-09-23"))

MATRIX = REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet"
POP = REPO / "research/rl_reintroduction_2026-09-23/data/stage1_population.parquet"
BARS_4H = REPO / "Data/shared/bars/4h"
WINDOW_START = pd.Timestamp("2020-09-08")
WINDOW_END = pd.Timestamp("2025-08-07")
WARMUP_SESSIONS = 126          # ~6 months, the documented feature warm-up
EVENT_R = 4.0


def main() -> None:
    out: dict = {}
    print("loading the Stage 1 population and the gate's own matrix ...", flush=True)
    pop = pd.read_parquet(POP)
    pop = pop[(~pop["ca_flagged"]) & (pop["session_date"] >= WINDOW_START)
              & (pop["session_date"] <= WINDOW_END)]

    g = pd.read_parquet(MATRIX, columns=["atr_pct_14"]).reset_index()
    g["decision_day"] = (g["timestamp"].dt.tz_convert("America/New_York")
                         .dt.normalize().dt.tz_localize(None))
    g = g[(g["decision_day"] >= WINDOW_START) & (g["decision_day"] <= WINDOW_END)]
    pool = set(g["ticker"].unique())
    pop = pop[pop["ticker"].isin(pool)].copy()

    # a gate decision on day D acts at the next session's open, so the population row it
    # covers is the first session strictly after D -- same join Stage 1 used.
    gd = g[["ticker", "decision_day"]].drop_duplicates().copy()
    gd["_key"] = gd["decision_day"] + pd.Timedelta(days=1)
    j = pd.merge_asof(gd.sort_values("_key"),
                      pop[["ticker", "session_date"]].drop_duplicates().sort_values("session_date"),
                      left_on="_key", right_on="session_date", by="ticker",
                      direction="forward", tolerance=pd.Timedelta(days=7)).dropna(subset=["session_date"])
    covered = set(map(tuple, j[["ticker", "session_date"]].drop_duplicates().to_numpy()))

    pop["covered"] = [(t, s) in covered for t, s in zip(pop["ticker"], pop["session_date"])]
    pop["event"] = pop["rmfe_10"] >= EVENT_R
    n, ncov = len(pop), int(pop["covered"].sum())
    out["coverage"] = {"pool_rows": n, "covered": ncov, "share": ncov / max(1, n),
                       "event_rate_all": float(pop["event"].mean()),
                       "event_rate_covered": float(pop.loc[pop["covered"], "event"].mean()),
                       "event_rate_missing": float(pop.loc[~pop["covered"], "event"].mean())}
    print(json.dumps(out["coverage"], indent=2))

    miss = pop[~pop["covered"]].copy()
    print(f"\nclassifying {len(miss):,} missing (ticker, session) pairs over "
          f"{miss['ticker'].nunique()} tickers ...", flush=True)

    # per-ticker 4H bar spans and session sets
    spans: dict[str, tuple] = {}
    sessions: dict[str, set] = {}
    for i, t in enumerate(sorted(miss["ticker"].unique()), 1):
        f = BARS_4H / f"{t}.parquet"
        if not f.exists():
            spans[t] = None
            continue
        try:
            b = pd.read_parquet(f, columns=["timestamp"])
        except Exception:
            spans[t] = None
            continue
        d = (pd.to_datetime(b["timestamp"], utc=True).dt.tz_convert("America/New_York")
             .dt.normalize().dt.tz_localize(None))
        spans[t] = (d.min(), d.max())
        sessions[t] = set(d.unique())
        if i % 200 == 0:
            print(f"  bars {i}/{miss['ticker'].nunique()}", flush=True)

    # The matrix's ticker set is NOT a constant pool: names enter and leave the momentum
    # universe over time (TM appears only from 2025-08, NNDM only 2021-01..2022-03). A
    # session outside a ticker's own matrix span therefore means "not in the universe then",
    # which is correct behaviour, NOT a coverage hole. Conflating the two was the first
    # version's bug and it inflated `feature_nan` badly.
    span_lo = g.groupby("ticker")["decision_day"].min().to_dict()
    span_hi = g.groupby("ticker")["decision_day"].max().to_dict()

    def classify(t, s) -> str:
        lo_m, hi_m = span_lo.get(t), span_hi.get(t)
        if lo_m is not None and s < lo_m:
            return "not_in_universe_yet"
        if hi_m is not None and s > hi_m:
            return "left_universe"
        sp = spans.get(t)
        if sp is None:
            return "no_4h_bars"
        lo, hi = sp
        if s < lo:
            return "before_first_bar"
        if s > hi:
            return "after_last_bar"
        if (s - lo).days <= WARMUP_SESSIONS * 2 and lo_m is not None and s < lo_m:
            return "warmup"
        if s not in sessions.get(t, set()):
            return "missing_bar"
        return "feature_nan_in_span"

    miss["cause"] = [classify(t, s) for t, s in zip(miss["ticker"], miss["session_date"])]
    tab = (miss.groupby("cause")
           .agg(rows=("event", "size"), events=("event", "sum"),
                tickers=("ticker", "nunique"))
           .assign(share_of_missing=lambda x: x["rows"] / len(miss),
                   event_rate=lambda x: x["events"] / x["rows"])
           .sort_values("rows", ascending=False).round(4))
    out["causes"] = tab.reset_index().to_dict("records")
    print("\n" + "=" * 78 + "\nWHY THE ROWS ARE MISSING\n" + "=" * 78)
    print(tab.to_string())

    print("\nevent rate by cause vs the covered baseline "
          f"({out['coverage']['event_rate_covered']:.3%}):")
    for r in out["causes"]:
        lift = r["event_rate"] / max(1e-12, out["coverage"]["event_rate_covered"])
        print(f"  {r['cause']:<18} {r['event_rate']:.3%}  ({lift:.2f}x)  "
              f"{int(r['rows']):>7,} rows")

    in_span = {"feature_nan_in_span", "missing_bar", "warmup", "no_4h_bars",
               "before_first_bar", "after_last_bar"}
    real = miss[miss["cause"].isin(in_span)]
    covered_plus = ncov + int((~miss["cause"].isin(in_span)).sum())
    print("\n" + "=" * 78 + "\nTHE ACTIONABLE HOLE (sessions INSIDE the ticker's own universe span)\n"
          + "=" * 78)
    print(f"  pool rows                                  {n:,}")
    print(f"  outside the ticker's universe span         "
          f"{int((~miss['cause'].isin(in_span)).sum()):,}  (expected, not a hole)")
    print(f"  effective denominator                      {covered_plus:,}")
    print(f"  covered by the gate                        {ncov:,} "
          f"({ncov / max(1, covered_plus):.2%})")
    print(f"  GENUINE in-span gap                        {len(real):,} "
          f"({len(real) / max(1, covered_plus):.2%})")
    if len(real):
        print(f"  tail events in the genuine gap             {int(real['event'].sum()):,} "
              f"(rate {real['event'].mean():.3%}, "
              f"{real['event'].mean() / max(1e-12, out['coverage']['event_rate_covered']):.2f}x covered)")
    out["actionable"] = {"pool_rows": n, "outside_span": int((~miss["cause"].isin(in_span)).sum()),
                         "effective_denominator": covered_plus, "covered": ncov,
                         "genuine_gap": int(len(real)),
                         "genuine_gap_events": int(real["event"].sum()) if len(real) else 0}

    print("\nthe 15 tickers contributing the most missing rows:")
    top = (miss.groupby(["ticker"])
           .agg(rows=("event", "size"), events=("event", "sum"),
                cause=("cause", lambda s: s.value_counts().index[0]))
           .sort_values("rows", ascending=False).head(15))
    out["top_tickers"] = top.reset_index().to_dict("records")
    print(top.to_string())

    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "gate_coverage.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {HERE / 'data' / 'gate_coverage.json'}")


if __name__ == "__main__":
    main()
