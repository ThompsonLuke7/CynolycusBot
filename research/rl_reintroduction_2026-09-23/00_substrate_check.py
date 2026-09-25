"""Stage 0: verify the substrate and CALIBRATE the right-tail event definition.

Nothing is fitted here. Two jobs:

A. Integrity of the two datasets the later stages read, and the house split bounds.
B. Base rates for a menu of candidate "fat right tail" definitions, so the event can be
   frozen BEFORE Stage 1 runs. This is the direct lesson of
   `research/regime_coverage_2026-09-21/recall_precision.py`: +15% MFE in 20 sessions has
   a 25.87% base rate, so a rule firing on 26% of days gets 26% recall for free. An event
   we intend to call rare must be shown to be rare first.

Outputs `data/stage0_calibration.json` and prints every table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

import panel as P  # noqa: E402

MATRIX = REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet"
R_THRESH = [2.0, 3.0, 4.0, 5.0, 6.0]
PCT_THRESH = [0.10, 0.15, 0.25, 0.50]
XS_TOP = [0.005, 0.01, 0.02, 0.05]
CAL_HOLDS = [5, 10, 20]
out: dict = {}


def hdr(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


# --------------------------------------------------------------- A. integrity
hdr("A1. decision panel v2")
raw = P.load_panel(drop_flagged=False)
guarded = P.load_panel(drop_flagged=True)
bars = raw["timestamp"].nunique()
out["panel"] = {
    "rows_raw": int(len(raw)), "rows_guarded": int(len(guarded)),
    "ca_flagged": int(raw["ca_flagged"].sum()), "ca_flagged_share": float(raw["ca_flagged"].mean()),
    "bars": int(bars), "tickers": int(raw["ticker"].nunique()),
    "span": [str(raw["decision_day"].min().date()), str(raw["decision_day"].max().date())],
    "median_xs": float(raw["xs_size"].median()),
}
print(json.dumps(out["panel"], indent=2))
print("\nnull rate, outcome columns (guarded):")
for c in ["entry_session"] + [f"{p}_{h}" for h in P.HOLDS for p in ("fwdret", "mfe", "mae")]:
    print(f"  {c:<14} {guarded[c].isna().mean():.4%}")

hdr("A2. house split (reused from scripts/horizon_thesis/run_horizon_grid.py)")
tr, va, te, bounds = P.split(guarded)
bounds.update(rows=dict(train=int(len(tr)), val=int(len(va)), test=int(len(te))))
out["split"] = bounds
print(json.dumps(bounds, indent=2))

hdr("A3. momentum training matrix (Stage 1 population)")
f = pq.ParquetFile(MATRIX)
gate_cols = ["low_price_flag", "dollar_vol_pctile_252", "dist_to_52w_high_atr",
             "xsec_near_high_rank", "rs_spy_20", "xsec_ret_20_rank", "range_pos_20"]
mcols = ["timestamp", "ticker", "atr_pct_14", "fwd_max_return", "fwd_max_alpha",
         "fwd_max_drawdown", "expansion_survival_score"] + gate_cols
# (timestamp, ticker) is a MultiIndex in this matrix, so reset it into columns.
m = pd.read_parquet(MATRIX, columns=mcols).reset_index()
out["matrix"] = {
    "rows": int(f.metadata.num_rows), "cols": int(f.metadata.num_columns),
    "tickers": int(m["ticker"].nunique()), "bars": int(m["timestamp"].nunique()),
    "span": [str(m["timestamp"].min().date()), str(m["timestamp"].max().date())],
}
print(json.dumps(out["matrix"], indent=2))
print("\nnull rate, gate features:")
out["gate_nulls"] = {c: float(m[c].isna().mean()) for c in gate_cols}
for c, v in out["gate_nulls"].items():
    print(f"  {c:<26} {v:.4%}")

hdr("A4. survivorship exposure of the daily cache")
out["survivorship"] = P.survivorship_note()
print(json.dumps(out["survivorship"], indent=2))

# ------------------------------------------------ B. tail-event base rates
hdr("B. right-tail event base rates (decision panel, tradable convention)")
print("MFE measured from the FIRST EXECUTABLE PRICE (next session's open), per the\n"
      "23_rank_depth_and_options.md section 0 correction. 1R = "
      f"{P.K_RISK} x ATR{P.ATR_WINDOW}%.\n")
pan = P.risk_units(guarded, holds=CAL_HOLDS)
out["atr_coverage"] = float(pan["atr_pct"].notna().mean())
print(f"ATR joined on {out['atr_coverage']:.2%} of guarded rows; "
      f"median ATR% {pan['atr_pct'].median():.4f}\n")

rows = []
for h in CAL_HOLDS:
    rm, mf = pan[f"rmfe_{h}"], pan[f"mfe_{h}"]
    n = int(rm.notna().sum())
    for t in R_THRESH:
        rows.append(dict(kind="R-unit MFE", defn=f"rmfe_{h} >= {t:g}R", hold=h, n=n,
                         base_rate=float((rm >= t).mean())))
    for t in PCT_THRESH:
        rows.append(dict(kind="absolute MFE", defn=f"mfe_{h} >= {t:.0%}", hold=h,
                         n=int(mf.notna().sum()), base_rate=float((mf >= t).mean())))
    g = pan.groupby("timestamp")[f"rmfe_{h}"]
    pct = g.rank(pct=True, ascending=False)
    for q in XS_TOP:
        rows.append(dict(kind="cross-sectional", defn=f"rmfe_{h} top {q:.1%} in bar", hold=h,
                         n=n, base_rate=float((pct <= q).mean())))
cal = pd.DataFrame(rows)
out["base_rates"] = cal.to_dict("records")
for kind, g in cal.groupby("kind", sort=False):
    print(f"\n-- {kind} --")
    piv = g.pivot_table(index="defn", columns="hold", values="base_rate")
    print((piv * 100).round(3).to_string())

hdr("B2. how far out is the tail? quantiles of rmfe")
q = [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]
qt = pd.DataFrame({f"rmfe_{h}": pan[f"rmfe_{h}"].quantile(q) for h in CAL_HOLDS})
out["rmfe_quantiles"] = qt.round(3).to_dict()
print(qt.round(3).to_string())
print("\nfor reference, same quantiles of the R-unit CLOSE return r_h:")
print(pd.DataFrame({f"r_{h}": pan[f"r_{h}"].quantile(q) for h in CAL_HOLDS}).round(3).to_string())

hdr("B3. the regime_coverage reference point, reproduced on this panel")
for h in CAL_HOLDS:
    br = float((pan[f"mfe_{h}"] >= 0.15).mean())
    print(f"  mfe_{h} >= +15%: base rate {br:.2%}")
print("\n(recall_precision.py measured 25.87% for +15% within 20 SESSIONS on the daily\n"
      " cache; hold 20 here is 20 sessions, so these should be comparable.)")

(HERE / "data").mkdir(exist_ok=True)
(HERE / "data" / "stage0_calibration.json").write_text(json.dumps(out, indent=2, default=str))
print(f"\nwrote {HERE / 'data' / 'stage0_calibration.json'}")
