"""Steps 5 and 6 -- score-proportional sizing, and the pullback trigger.

STEP 5. Stage 6 measured Kelly f* rising monotonically 0.02 -> 0.12 across `mom_score`
deciles, with decile 1 showing NEGATIVE log growth at its own optimum. The cheapest action is
"stop trading the bottom decile" -- so first: does live actually trade there? The live audits
carry `rank_pct` per entry, which is the within-bar percentile, so this is directly checkable.

STEP 6. Stage 5 measured "enter on the first pullback, else reject" at -0.5877R against
enter-immediately on 434 candidates. Momentum's live audits show a `trigger_rule` field with
values including `pullback_continuation` -- i.e. the repo has a live pullback trigger. This
compares the realised outcome of each trigger rule, so the Stage 5 result can be checked
against live behaviour instead of assumed to apply.
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
DEPRECATED = {"dealer_ranker"}


def load_entry_audits() -> pd.DataFrame:
    """One row per entry-side signal audit, from every shape the audits use."""
    rows = []
    for p in sorted(glob.glob(str(REPO / "Data/inference/*/live_signal_audit.jsonl"))):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            bar, mod = rec.get("bar"), rec.get("module")

            def take(a: dict, extra: dict | None = None):
                if not isinstance(a, dict):
                    return
                ex = a.get("extra") or {}
                rows.append({"module": a.get("module", mod), "bar": bar,
                             "ticker": a.get("ticker"), "score": a.get("score"),
                             "rank": a.get("rank"), "rank_pct": a.get("rank_pct"),
                             "trigger_rule": ex.get("trigger_rule"),
                             "entry_triggered": ex.get("entry_triggered",
                                                       rec.get("entry_triggered")),
                             **(extra or {})})

            if isinstance(rec.get("signal_audit"), dict):
                take(rec["signal_audit"])
            if isinstance(rec.get("signal_audits"), dict):
                for a in rec["signal_audits"].values():
                    take(a)
            cs = rec.get("contract_selection")
            if isinstance(cs, dict):
                for tkr, sel in cs.items():
                    if isinstance(sel, dict) and isinstance(sel.get("signal_audit"), dict):
                        take(sel["signal_audit"], {"action": sel.get("action")})
    d = pd.DataFrame(rows).dropna(subset=["ticker"])
    for c in ("score", "rank", "rank_pct"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["bar_ts"] = pd.to_datetime(d["bar"], utc=True, errors="coerce")
    d["live_module"] = ~d["module"].isin(DEPRECATED)
    return d


def load_trades() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(REPO / "Data/inference/*/closed_trades.jsonl"))):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    d = pd.DataFrame(rows)
    d["entry_ts"] = pd.to_datetime(d["entry_bar"], utc=True, errors="coerce")
    for c in ("realized_pnl", "decision_gain"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def main() -> None:
    out: dict = {}
    a = load_entry_audits()
    tr = load_trades()
    print(f"entry audits {len(a):,} over {a['module'].nunique()} modules; "
          f"closed trades {len(tr):,}")

    # ------------------------------------------------------------- STEP 5
    print("\n" + "=" * 78 + "\nSTEP 5 -- does live trade the bottom of its own ranking?\n"
          + "=" * 78)
    ent = a[(a["entry_triggered"] == True) & a["rank_pct"].notna()]  # noqa: E712
    print("`rank_pct` is the WITHIN-BAR percentile of the score (1.00 = top of the bar).")
    if len(ent):
        for label, frame in (("ALL modules", ent), ("LIVE only", ent[ent["live_module"]])):
            q = frame["rank_pct"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(4)
            print(f"\n-- {label} -- n={len(frame):,}")
            print(q.to_string())
            below = float((frame["rank_pct"] < 0.90).mean())
            print(f"  share of triggered entries below the top DECILE of their bar: {below:.1%}")
            out[f"step5_{label.replace(' ', '_')}"] = {
                "n": int(len(frame)), "share_below_top_decile": below,
                "rank_pct_median": float(frame["rank_pct"].median()),
                "rank_pct_min": float(frame["rank_pct"].min())}
    else:
        print("  no triggered-entry audits carry rank_pct")

    print("\nrank (integer) of triggered entries -- how deep does live actually go?")
    r = a[(a["entry_triggered"] == True) & a["rank"].notna()]  # noqa: E712
    if len(r):
        print(r.groupby(pd.cut(r["rank"], [0, 1, 2, 3, 5, 10, 20, 1e9]),
                        observed=False)["rank"].size().to_string())
        out["step5_rank_hist"] = {str(k): int(v) for k, v in
                                  r.groupby(pd.cut(r["rank"], [0, 1, 2, 3, 5, 10, 20, 1e9]),
                                            observed=False)["rank"].size().items()}

    # ------------------------------------------------------------- STEP 6
    print("\n" + "=" * 78 + "\nSTEP 6 -- the live pullback trigger vs the alternatives\n"
          + "=" * 78)
    t = a[a["trigger_rule"].notna()]
    if not len(t):
        print("  no trigger_rule field in the audits")
    else:
        print("trigger_rule frequency (all audit rows):")
        print(t["trigger_rule"].value_counts().to_string())
        out["step6_trigger_counts"] = t["trigger_rule"].value_counts().to_dict()

        # join triggered entries to their realised trade on (module, ticker, entry bar)
        te = (t[t["entry_triggered"] == True]  # noqa: E712
              .dropna(subset=["bar_ts"]).sort_values("bar_ts"))
        tt = tr.dropna(subset=["entry_ts"]).sort_values("entry_ts")
        j = pd.merge_asof(tt, te[["module", "ticker", "bar_ts", "trigger_rule", "rank"]],
                          left_on="entry_ts", right_on="bar_ts", by=["module", "ticker"],
                          direction="nearest", tolerance=pd.Timedelta("2D"))
        j = j[j["trigger_rule"].notna()]
        print(f"\ntrades matched to a trigger_rule: {len(j):,}")
        if len(j):
            g = (j.groupby(["trigger_rule", "route"])
                 .agg(n=("realized_pnl", "size"), pnl_sum=("realized_pnl", "sum"),
                      pnl_mean=("realized_pnl", "mean"), pnl_median=("realized_pnl", "median"),
                      win_rate=("realized_pnl", lambda s: float((s > 0).mean())))
                 .round(1))
            print(g.to_string())
            out["step6_by_trigger"] = g.reset_index().to_dict("records")
            gg = (j.groupby("trigger_rule")
                  .agg(n=("realized_pnl", "size"), pnl_sum=("realized_pnl", "sum"),
                       pnl_mean=("realized_pnl", "mean"),
                       win_rate=("realized_pnl", lambda s: float((s > 0).mean())))
                  .round(1).sort_values("pnl_mean"))
            print("\npooled over routes, worst mean first:")
            print(gg.to_string())
            out["step6_pooled"] = gg.reset_index().to_dict("records")
            print("\nSample sizes here are small -- this is a live-ledger CHECK on the "
                  "Stage 5\nresult (-0.5877R for a pullback entry rule on 434 backtested "
                  "candidates),\nnot an independent test. Read direction and consistency, "
                  "not significance.")

    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "sizing_and_triggers.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {HERE / 'data' / 'sizing_and_triggers.json'}")


if __name__ == "__main__":
    main()
