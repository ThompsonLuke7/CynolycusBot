"""Stage 1 — the trade spine: one row per POSITION LIFECYCLE, with every clock.

Stage 0 established that a ledger row is not a trade: 56 of 183 rows are partial
scale-outs. A lifecycle is reconstructed from the broker order stream instead —
the authoritative record — and then joined back to the ledger and the signal
audit for module attribution and model context.

Lifecycle = for one symbol, a contiguous run of fills that takes the position
from flat, up through any adds, down through any trims, back to flat (or to
"still open" at the end of the sample).

Clocks captured per lifecycle:
  signal_ts       from the module's signal_audit (the model/rule's own stamp)
  first_submit    broker submitted_at of the first entry rung
  first_fill      broker filled_at of the first entry rung   <- T_fill
  last_entry_fill last add
  exit_submit     first exit rung submitted
  exit_fill       final fill that returns the position to flat
Plus rung structure, trim schedule, and the joined signal_audit.

Read-only. Consumes cached artifacts only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "research/execution_quality/data"
MODULES = ["momentum_expansion", "multi_ticker_swing_htf", "meta_ranker", "dealer_ranker"]
OCC = __import__("re").compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def P(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def underlying_of(symbol: str) -> str:
    m = OCC.match(str(symbol))
    return m.group(1) if m else str(symbol)


def occ_meta(symbol: str) -> dict:
    m = OCC.match(str(symbol))
    if not m:
        return {"is_option": False, "expiry": None, "strike": None, "right": None}
    yy, mm, dd = m.group(2)[:2], m.group(2)[2:4], m.group(2)[4:]
    return {
        "is_option": True,
        "expiry": f"20{yy}-{mm}-{dd}",
        "strike": int(m.group(4)) / 1000.0,
        "right": m.group(3),
    }


def build_lifecycles(orders):
    """Walk each symbol's filled orders and cut them into flat-to-flat runs."""
    by_symbol = defaultdict(list)
    for o in orders:
        if o.get("status") != "filled" or not o.get("filled_at"):
            continue
        by_symbol[str(o["symbol"])].append(o)

    cycles = []
    for sym, fills in by_symbol.items():
        fills.sort(key=lambda r: P(r["filled_at"]))
        pos = 0.0
        cur = None
        for o in fills:
            q = float(o.get("filled_qty") or 0)
            if q <= 0:
                continue
            signed = q if o.get("side") == "buy" else -q
            if cur is None:
                if signed <= 0:
                    continue  # a sell with no tracked entry (pre-retention) — skip
                cur = {"symbol": sym, "entries": [], "exits": []}
            (cur["entries"] if signed > 0 else cur["exits"]).append(o)
            pos += signed
            if pos <= 1e-9 and cur["entries"]:
                cur["closed"] = True
                cycles.append(cur)
                cur, pos = None, 0.0
        if cur is not None and cur["entries"]:
            cur["closed"] = False
            cycles.append(cur)
    return cycles


def summarize(cycle):
    e, x = cycle["entries"], cycle["exits"]
    sym = cycle["symbol"]

    def vwap(group):
        q = sum(float(o.get("filled_qty") or 0) for o in group)
        n = sum(float(o.get("filled_qty") or 0) * float(o.get("filled_avg_price") or 0) for o in group)
        return (n / q if q else None), q

    epx, eqty = vwap(e)
    xpx, xqty = vwap(x)
    meta = occ_meta(sym)
    first_fill = min(P(o["filled_at"]) for o in e)
    first_submit = min(P(o["submitted_at"] or o["created_at"]) for o in e)
    last_entry_fill = max(P(o["filled_at"]) for o in e)
    exit_fill = max((P(o["filled_at"]) for o in x), default=None)
    exit_submit = min((P(o["submitted_at"] or o["created_at"]) for o in x), default=None)
    mult = 100.0 if meta["is_option"] else 1.0
    row = {
        "symbol": sym,
        "ticker": underlying_of(sym),
        "route": "option" if meta["is_option"] else "equity",
        "closed": cycle["closed"],
        "entry_rungs": len(e),
        "exit_rungs": len(x),
        "entry_qty": eqty,
        "exit_qty": xqty,
        "entry_vwap": epx,
        "exit_vwap": xpx,
        "first_entry_submit": first_submit.isoformat(),
        "first_entry_fill": first_fill.isoformat(),
        "last_entry_fill": last_entry_fill.isoformat(),
        "exit_first_submit": exit_submit.isoformat() if exit_submit else None,
        "exit_last_fill": exit_fill.isoformat() if exit_fill else None,
        "hold_minutes": ((exit_fill - first_fill).total_seconds() / 60.0) if exit_fill else None,
        "entry_ladder_seconds": (last_entry_fill - first_submit).total_seconds(),
        "gross_pnl": (round((xpx - epx) * mult * xqty, 2)
                      if (xpx is not None and epx is not None and xqty) else None),
        "gross_return": (round(xpx / epx - 1.0, 6) if (xpx and epx) else None),
    }
    row["order_ids"] = [str(o.get("id")) for o in e + x]
    row.update({f"opt_{k}": v for k, v in meta.items() if k != "is_option"})
    if meta["expiry"] and row["first_entry_fill"]:
        row["dte_at_entry"] = (P(meta["expiry"] + "T00:00:00+00:00") - first_fill).days
    return row


def load_swing_order_ids():
    """Broker order ids the 30m swing module submitted — exact attribution.

    The swing module keeps no closed_trades.jsonl; its session audit carries the
    broker order id inside `payload.verification.order.id`, which is a stronger
    join than symbol+time.
    """
    import glob
    ids = set()
    for path in glob.glob(str(REPO_ROOT / "UI/swing_audit/swing_session_*.jsonl")):
        with open(path) as fh:
            for line in fh:
                if '"order_submitted"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("type") or rec.get("event")) != "order_submitted":
                    continue
                pay = rec.get("payload") or {}
                for oid in (
                    ((pay.get("verification") or {}).get("order") or {}).get("id"),
                    (pay.get("verification") or {}).get("order_id"),
                    (pay.get("response") or {}).get("id"),
                ):
                    if oid:
                        ids.add(str(oid))
    return ids


def load_signal_index():
    """(module, ticker) -> [(bar, signal_audit)], plus buy-side order audits and
    the entry orders each module PLANNED (symbol + bar).

    The planned-entry index is the attribution fallback for positions that are
    still open: those have no closed_trades row, so the ledger cannot name their
    module, but the module did record planning the entry.
    """
    idx, order_meta, planned = {}, {}, {}
    for m in MODULES:
        path = REPO_ROOT / f"Data/inference/{m}/live_signal_audit.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bar = rec.get("bar")
            for tkr, sa in (rec.get("signal_audits") or {}).items():
                idx.setdefault((m, str(tkr).upper()), []).append((bar, sa))
            for osym, oa in (rec.get("order_audits") or {}).items():
                if oa.get("side") == "buy":
                    order_meta.setdefault((m, str(osym)), []).append((bar, oa))
            for item in (rec.get("plan") or []):
                if item.get("side") == "buy" and str(item.get("reason", "")).startswith("entry"):
                    planned.setdefault(str(item.get("symbol")), []).append(
                        (m, P(bar), rec.get("signal_audits") or {}))
    return idx, order_meta, planned


def bar_close(bar_ts):
    """Availability time of a 4H decision bar.

    Alpaca 4H bars are LEFT-labelled, so the bar stamped 14:00Z spans 14:00-18:00Z
    and is only complete at 18:00Z; the 18:00Z bar is completed by the 20:00Z
    market close. Using the label as the decision time would credit the modules
    with four hours of latency they do not have — event time is not availability
    time (AGENTS.md). Modules whose `bar` is a wall clock (dealer_ranker) are
    already stamped at availability, so the value passes through.
    """
    if bar_ts is None:
        return None
    hhmm = (bar_ts.hour, bar_ts.minute)
    if hhmm == (14, 0):
        return bar_ts.replace(hour=18, minute=0)
    if hhmm == (18, 0):
        return bar_ts.replace(hour=20, minute=0)
    return bar_ts


def main() -> None:
    orders = [json.loads(l) for l in (DATA / "order_history.jsonl").open() if l.strip()]
    cycles = [summarize(c) for c in build_lifecycles(orders)]
    cycles.sort(key=lambda r: r["first_entry_fill"])

    # Module attribution: the ledger names the module for a symbol; use the
    # ledger row whose exit timestamp falls inside the lifecycle.
    ledger = []
    for m in MODULES:
        for line in (REPO_ROOT / f"Data/inference/{m}/closed_trades.jsonl").open():
            if not line.strip():
                continue
            r = json.loads(line)
            r["_module"] = m
            r["_ts"] = P(r.get("ts"))
            ledger.append(r)

    sig_idx, order_meta, planned = load_signal_index()
    swing_ids = load_swing_order_ids()
    attributed = 0
    for c in cycles:
        lo = P(c["first_entry_fill"])
        hi = P(c["exit_last_fill"])
        hi_pad = (hi + timedelta(hours=12)) if hi else datetime.max.replace(tzinfo=lo.tzinfo)
        hits = [r for r in ledger
                if str(r.get("order_symbol")) == c["symbol"] and r["_ts"]
                and lo - timedelta(minutes=10) <= r["_ts"] <= hi_pad]
        if hits:
            attributed += 1
            c["module"] = hits[0]["_module"]
            c["ledger_exit_reasons"] = [h.get("exit_reason") for h in hits]
            c["ledger_rows"] = len(hits)
            c["ledger_realized_pnl"] = sum(
                float(h["realized_pnl"]) for h in hits if h.get("realized_pnl") is not None) or None
            c["ledger_entry_bar"] = next((h.get("entry_bar") for h in hits if h.get("entry_bar")), None)
            c["decision_gain"] = next((h.get("decision_gain") for h in hits
                                       if h.get("decision_gain") is not None), None)
            c["stop_overshoot"] = next((h.get("stop_overshoot") for h in hits
                                        if h.get("stop_overshoot") is not None), None)
        elif swing_ids.intersection(c["order_ids"]):
            attributed += 1
            c["module"] = "multi_ticker_swing"
            c["ledger_exit_reasons"] = []
            c["ledger_rows"] = 0
        elif planned.get(c["symbol"]):
            # Still-open position: no ledger row exists, but exactly one module
            # planned this symbol as an entry near this fill. Ambiguous symbols
            # (two modules planned the same one) are left unattributed rather
            # than guessed.
            near = [(m, b) for m, b, _ in planned[c["symbol"]]
                    if b and abs((lo - b).total_seconds()) <= 36 * 3600]
            mods = {m for m, _ in near}
            if len(mods) == 1:
                attributed += 1
                c["module"] = mods.pop()
                c["module_source"] = "planned_entry"
                c["ledger_exit_reasons"] = []
                c["ledger_rows"] = 0
            else:
                c["module"] = None
                c["ledger_exit_reasons"] = []
                c["ledger_rows"] = 0
        elif c["ticker"] == "SPY":
            attributed += 1
            c["module"] = "spy_daytrader"
            c["ledger_exit_reasons"] = []
            c["ledger_rows"] = 0
        else:
            c["module"] = None
            c["ledger_exit_reasons"] = []
            c["ledger_rows"] = 0

        # Signal join: the order_plan that actually PLANNED this symbol as an
        # entry — not the nearest bar. The modules run twice a day and the
        # nearest-bar rule silently attached the 18:00Z bar to runs that had
        # decided on the 14:00Z one, inflating measured latency by hours.
        if c["module"]:
            submit = P(c["first_entry_submit"])
            best = None
            for mod, bts, audits in planned.get(c["symbol"], []):
                if mod != c["module"] or bts is None:
                    continue
                close = bar_close(bts)
                if close > submit + timedelta(minutes=15):
                    continue
                if submit - close > timedelta(hours=36):
                    continue
                if best is None or close > best[1]:
                    best = (bts, close, audits)
            if best:
                bts, close, audits = best
                sa = audits.get(c["ticker"], {})
                c["signal_bar"] = bts.isoformat()
                c["signal_available_at"] = close.isoformat()
                c["signal_score"] = sa.get("score")
                c["signal_rank"] = sa.get("rank")
                c["signal_rank_pct"] = sa.get("rank_pct")
                c["signal_side"] = sa.get("side")
                c["signal_extra"] = sa.get("extra") or {}
                c["avail_to_submit_min"] = round((submit - close).total_seconds() / 60.0, 2)
                c["avail_to_fill_min"] = round((lo - close).total_seconds() / 60.0, 2)
                c["bar_to_fill_min"] = round((lo - bts).total_seconds() / 60.0, 2)
            om = order_meta.get((c["module"], c["symbol"]), [])
            if om:
                oa = om[-1][1]
                for k in ("underlying_price", "strike", "premium", "limit_price", "mid_price",
                          "delta", "dte", "breakeven_move_pct"):
                    c[f"oa_{k}"] = oa.get(k)

    out = DATA / "stage1_trade_spine.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for c in cycles:
            fh.write(json.dumps(c) + "\n")

    n = len(cycles)
    withmod = [c for c in cycles if c["module"]]
    closed = [c for c in withmod if c["closed"]]
    withsig = [c for c in withmod if c.get("signal_available_at")]
    print(f"lifecycles reconstructed: {n}")
    print(f"  module-attributed:      {len(withmod)}")
    print(f"  ...of which closed:     {len(closed)}")
    print(f"  ...with a plan join:    {len(withsig)}")
    from collections import Counter
    print("  by module:", dict(Counter(c["module"] for c in withmod)))
    print("  by route :", dict(Counter(c["route"] for c in withmod)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
