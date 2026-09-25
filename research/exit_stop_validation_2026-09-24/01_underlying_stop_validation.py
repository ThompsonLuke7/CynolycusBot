"""Step 1 -- validate `ExecPolicy.underlying_stop_atr = 1.5` against the live ledger.

The change shipped 2026-08-18 and its own docstring says "NOT yet paper-validated". It
replaced a -39% PREMIUM stop that `research/daily_live_reports/underlying_vs_premium_stop.md`
measured firing at a median -3.1% underlying move (18 of 42 fired with the underlying down
<2%, -$43,944; 13 of those 18 recovered above entry within 40 4H bars).

`core/live_4h_exec.py:312-318` is an if/elif: when an option has a usable underlying basis
the premium stop is NOT evaluated at all. The premium stop survives only for equity and for
options whose basis could not be established (fail-safe).

What this script measures, from `Data/inference/*/closed_trades.jsonl`:
  A  how often each stop flavour fires, by module and by route
  B  the realised premium move at exit -- is the wider stop cutting later and deeper?
  C  the UNDERLYING move at exit vs the -1.5 ATR the rule intends, from the bar cache
  D  basis coverage: how many options still fall back to the premium stop, and why
  E  a before/after-2026-08-18 per-trade comparison, deprecated modules split out

WHAT THIS CANNOT ANSWER, stated up front: the full counterfactual ("would the premium stop
have closed this position earlier, and at what price") needs a historical option PREMIUM path,
which the 2026-07 retraction established we do not have. Every premium number here is the
realised mark AT THE EXIT WE ACTUALLY TOOK.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

LEDGERS = REPO / "Data/inference"
BARS_4H = REPO / "Data/shared/bars/4h"
BARS_1D = REPO / "Data/shared/bars/1d"
SHIP_DATE = pd.Timestamp("2026-08-18", tz="UTC")
# dealer_ranker is DEPRECATED (LIVING_SUMMARY 2026-09-22 module inventory). AGENTS.md forbids
# letting a non-production module anchor a live conclusion, so it is reported separately.
DEPRECATED = {"dealer_ranker"}
STOP_REASONS = ["underlying_stop_-1.5atr", "stop_-39%", "stop_-50%", "sl",
                "trail", "trail_-35%", "restored_unknown_loss_cut"]


def load_ledger() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(LEDGERS / "*/closed_trades.jsonl"))):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    d = pd.DataFrame(rows)
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    d["entry_bar"] = pd.to_datetime(d["entry_bar"], utc=True, errors="coerce")
    for c in ("realized_pnl", "decision_gain", "fill_gain", "u_entry", "u_atr", "runs_held"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["ts"])


def underlying_at(ticker: str, when: pd.Timestamp) -> float | None:
    """Underlying close at or before `when`, 4H cache first then daily."""
    for base, col in ((BARS_4H, "close"), (BARS_1D, "close")):
        f = base / f"{ticker}.parquet"
        if not f.exists():
            continue
        try:
            b = pd.read_parquet(f, columns=["timestamp", col])
        except Exception:
            continue
        b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
        b = b[b["timestamp"] <= when]
        if len(b):
            return float(b[col].iloc[-1])
    return None


def hdr(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def main() -> None:
    out: dict = {}
    d = load_ledger()
    print(f"ledger rows {len(d):,}  span {d['ts'].min().date()} .. {d['ts'].max().date()}")
    print(f"modules: {sorted(d['module'].unique())}")
    print(f"\nDEPRECATED and reported separately: {sorted(DEPRECATED)}")
    print("PREMIUM COUNTERFACTUAL IS NOT AVAILABLE (no historical option marks; see the\n"
          "2026-07 retraction). Premium figures are the mark at the exit actually taken.")

    post = d[d["ts"] >= SHIP_DATE]
    opt = post[post["route"] == "option"].copy()
    opt["live_module"] = ~opt["module"].isin(DEPRECATED)
    out["counts"] = {"ledger": int(len(d)), "post_ship": int(len(post)),
                     "post_ship_options": int(len(opt))}

    # ------------------------------------------------------------------ A
    hdr("A. stop flavours since 2026-08-18 (options only)")
    st = opt[opt["exit_reason"].isin(STOP_REASONS)]
    tab = pd.crosstab(st["exit_reason"], st["module"])
    print(tab.to_string())
    out["A_stop_counts"] = tab.to_dict()
    live = st[st["live_module"]]
    print(f"\nLIVE modules only: {len(live)} stop exits; "
          f"underlying_stop {int((live['exit_reason'] == 'underlying_stop_-1.5atr').sum())}, "
          f"premium fail-safe {int((live['exit_reason'] == 'stop_-39%').sum())}")

    # ------------------------------------------------------------------ B
    hdr("B. realised PREMIUM move at exit -- is the wider stop cutting deeper?")
    print("decision_gain is the option's premium gain at the exit decision.\n")
    for label, frame in (("ALL modules", st), ("LIVE modules only", live)):
        g = (frame.groupby("exit_reason")
             .agg(n=("realized_pnl", "size"), pnl_sum=("realized_pnl", "sum"),
                  pnl_mean=("realized_pnl", "mean"), pnl_median=("realized_pnl", "median"),
                  prem_n=("decision_gain", "count"),
                  prem_mean=("decision_gain", "mean"),
                  prem_median=("decision_gain", "median"))
             .round(3).sort_values("pnl_sum"))
        print(f"-- {label} --")
        print(g.to_string())
        print()
        out[f"B_{label.replace(' ', '_')}"] = g.reset_index().to_dict("records")

    # ------------------------------------------------------------------ C
    hdr("C. UNDERLYING move at exit vs the -1.5 ATR the rule intends")
    us = st[(st["exit_reason"] == "underlying_stop_-1.5atr")
            & st["u_entry"].notna() & st["u_atr"].notna()].copy()
    ps = st[(st["exit_reason"] == "stop_-39%")].copy()
    rows = []
    for label, frame in (("underlying_stop_-1.5atr", us), ("stop_-39% (option fail-safe)", ps)):
        for r in frame.itertuples():
            u_now = underlying_at(r.ticker, r.ts)
            if u_now is None or not np.isfinite(getattr(r, "u_entry", np.nan)):
                continue
            ue, ua = float(r.u_entry), float(r.u_atr)
            if not (np.isfinite(ue) and ue > 0):
                continue
            rows.append({"reason": label, "module": r.module, "ticker": r.ticker,
                         "u_move_pct": (u_now / ue - 1.0) * 100,
                         "u_move_atr": (u_now - ue) / ua if np.isfinite(ua) and ua > 0 else np.nan,
                         "prem_gain_pct": (r.decision_gain * 100
                                           if np.isfinite(r.decision_gain) else np.nan),
                         "pnl": r.realized_pnl, "runs_held": r.runs_held,
                         "deprecated": r.module in DEPRECATED})
    cd = pd.DataFrame(rows)
    out["C_rows"] = cd.to_dict("records")
    if cd.empty:
        print("  no rows with a resolvable underlying basis")
    else:
        for label, frame in (("ALL", cd), ("LIVE only", cd[~cd["deprecated"]])):
            if frame.empty:
                continue
            g = (frame.groupby("reason")
                 .agg(n=("u_move_pct", "size"),
                      u_move_pct_median=("u_move_pct", "median"),
                      u_move_atr_median=("u_move_atr", "median"),
                      prem_gain_median=("prem_gain_pct", "median"),
                      share_u_gt_neg2pct=("u_move_pct", lambda s: float((s > -2).mean())))
                 .round(3))
            print(f"-- {label} --")
            print(g.to_string())
            print()
        print("BASELINE to beat (underlying_vs_premium_stop.md, 42 premium stops "
              "2026-07-17..08-18):\n  median underlying move at the stop = -3.1%; "
              "18 of 42 (42.9%) fired with the underlying down <2%.")

    # ------------------------------------------------------------------ D
    hdr("D. basis coverage -- how often does an option still fall back to premium?")
    allopt = opt[opt["exit_reason"].isin(["underlying_stop_-1.5atr", "stop_-39%"])]
    cov = (allopt.assign(has_basis=allopt["u_entry"].notna() & allopt["u_atr"].notna())
           .groupby(["module", "exit_reason"])["has_basis"]
           .agg(n="size", with_basis="sum").reset_index())
    cov["without_basis"] = cov["n"] - cov["with_basis"]
    print(cov.to_string(index=False))
    out["D_coverage"] = cov.to_dict("records")
    n_fs = int((allopt["exit_reason"] == "stop_-39%").sum())
    n_all = int(len(allopt))
    print(f"\npremium fail-safe share of option hard stops: {n_fs}/{n_all} "
          f"= {n_fs / max(1, n_all):.1%}")
    print("NOTE an `underlying_stop` row with a NULL u_entry in the ledger is an "
          "INSTRUMENTATION\ngap, not a logic error: the basis existed at decision time or the "
          "rule could not have\nfired. Worth fixing so this check is exact.")

    # ------------------------------------------------------------------ E
    hdr("E. before / after 2026-08-18, per-trade option P&L (confounded by regime)")
    o_all = d[d["route"] == "option"].copy()
    o_all["period"] = np.where(o_all["ts"] >= SHIP_DATE, "post", "pre")
    o_all["live_module"] = ~o_all["module"].isin(DEPRECATED)
    for label, frame in (("ALL modules", o_all), ("LIVE modules only", o_all[o_all["live_module"]])):
        g = (frame.groupby("period")
             .agg(n=("realized_pnl", "size"), pnl_sum=("realized_pnl", "sum"),
                  pnl_mean=("realized_pnl", "mean"), pnl_median=("realized_pnl", "median"),
                  win_rate=("realized_pnl", lambda s: float((s > 0).mean())),
                  median_runs=("runs_held", "median")).round(2))
        print(f"-- {label} --")
        print(g.to_string())
        print()
        out[f"E_{label.replace(' ', '_')}"] = g.reset_index().to_dict("records")
    print("This is a BEFORE/AFTER on a live book, not a controlled comparison: the tape,\n"
          "the universe and other config all moved too. Direction only.")

    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "underlying_stop_validation.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\nwrote {HERE / 'data' / 'underlying_stop_validation.json'}")


if __name__ == "__main__":
    main()
