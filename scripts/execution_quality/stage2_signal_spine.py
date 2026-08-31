"""Stage 2 — the signal spine: one row per ranked target per decision, traded or not.

WHY this table and not the trade table. The traded rows are a selected sample:
the order policy chose them, so any "our signals work" claim computed on them is
contaminated by the policy's own selection. Every ranked target in every
`signal_decision` / `order_plan` audit is a forward-return observation the module
committed to at a known timestamp, whether or not capital followed. That is the
correct — and much larger — sample for "do the module signals even make sense".

`was_traded` is carried so the policy's selection can be studied as its own
effect rather than silently conditioned on.

Read-only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "research/execution_quality/data"
MODULES = ["momentum_expansion", "multi_ticker_swing_htf", "meta_ranker", "dealer_ranker"]


def P(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def bar_close(bar_ts):
    """See stage1: 4H bars are left-labelled, so availability != the label."""
    if bar_ts is None:
        return None
    if (bar_ts.hour, bar_ts.minute) == (14, 0):
        return bar_ts.replace(hour=18, minute=0)
    if (bar_ts.hour, bar_ts.minute) == (18, 0):
        return bar_ts.replace(hour=20, minute=0)
    return bar_ts


def flatten(extra: dict) -> dict:
    """Keep scalar model context; drop the nested dealer scope JSON blob."""
    out = {}
    for k, v in (extra or {}).items():
        if k == "scope_scores_json":
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[f"x_{k}"] = v
    return out


def main() -> None:
    traded = defaultdict(set)
    for line in (DATA / "stage1_trade_spine.jsonl").open():
        r = json.loads(line)
        if r.get("module") and r.get("signal_bar"):
            traded[(r["module"], r["ticker"])].add(r["signal_bar"][:19])

    # A single decision is logged twice (signal_decision, then order_plan for
    # the same bar) and only the second carries `plan`. Collect the planned
    # entries per (module, bar) FIRST so the flag survives de-duplication.
    plan_idx: dict[tuple[str, str], set[str]] = defaultdict(set)
    submit_idx: dict[tuple[str, str], bool] = {}
    for m in MODULES:
        for line in (REPO_ROOT / f"Data/inference/{m}/live_signal_audit.jsonl").open():
            if '"order_plan"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "order_plan":
                continue
            b = P(rec.get("bar"))
            if b is None:
                continue
            k = (m, b.isoformat())
            submit_idx[k] = submit_idx.get(k, False) or bool(rec.get("submit"))
            for i in (rec.get("plan") or []):
                if i.get("side") == "buy" and str(i.get("reason", "")).startswith("entry"):
                    plan_idx[k].add(str(i.get("symbol")))

    rows = []
    seen = set()
    for m in MODULES:
        path = REPO_ROOT / f"Data/inference/{m}/live_signal_audit.jsonl"
        for line in path.open():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") not in ("signal_decision", "order_plan"):
                continue
            bar = P(rec.get("bar"))
            if bar is None:
                continue
            close = bar_close(bar)
            planned = plan_idx.get((m, bar.isoformat()), set())
            for tkr, sa in (rec.get("signal_audits") or {}).items():
                tkr = str(tkr).upper()
                # One decision may be logged twice (signal_decision then
                # order_plan for the same bar). Count it once.
                key = (m, tkr, bar.isoformat())
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "module": m,
                    "ticker": tkr,
                    "bar": bar.isoformat(),
                    "available_at": close.isoformat(),
                    "side": sa.get("side"),
                    "score": sa.get("score"),
                    "rank": sa.get("rank"),
                    "rank_pct": sa.get("rank_pct"),
                    "score_bucket": sa.get("score_bucket"),
                    "rank_bucket": sa.get("rank_bucket"),
                    "planned_entry": bool(
                        tkr in planned or any(str(s).startswith(tkr) for s in planned)),
                    "was_traded": bar.isoformat()[:19] in traded.get((m, tkr), set()),
                    "submit": bool(submit_idx.get((m, bar.isoformat()), False)),
                    "bar_kind": ("rth_4h" if (bar.hour, bar.minute) in ((14, 0), (18, 0))
                                 else "wallclock"),
                }
                row.update(flatten(sa.get("extra")))
                rows.append(row)

    out = DATA / "stage2_signal_spine.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"signal rows: {len(rows)}")
    by = Counter(r["module"] for r in rows)
    for m in MODULES:
        sub = [r for r in rows if r["module"] == m]
        if not sub:
            continue
        bars = len({r["bar"] for r in sub})
        act = [r for r in sub if r["submit"]]
        print(f"  {m:24s} rows={by[m]:5d} decisions={bars:4d} "
              f"actionable={len(act):5d} "
              f"tickers={len({r['ticker'] for r in sub}):4d} "
              f"planned={sum(r['planned_entry'] for r in sub):4d} "
              f"traded={sum(r['was_traded'] for r in sub):4d}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
