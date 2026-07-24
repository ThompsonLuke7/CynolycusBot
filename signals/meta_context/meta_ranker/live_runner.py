"""
Meta Ranker live equity runner — long-only, equal-weight top-K combo portfolio.

Scores the meta-ranker matrix with the confluence (combo) signal, takes the top-K
names on the latest full 4H bar, and reconciles a paper (default) Alpaca account to
an equal-weight long-only target portfolio.

SAFETY (read before running live):
  * DRY-RUN by default. Orders are only submitted with --submit.
  * PAPER account by default (.env#PAPER). --live is required to target the live account.
  * Scoped holdings: this runner ONLY ever sells symbols it previously bought (tracked in
    live_state.json). It never touches positions opened by other strategies sharing the account.
  * Staleness guard: refuses to trade if the matrix's latest bar is older than
    --max-staleness-days (override with --allow-stale). The shipped matrix is label-limited;
    rebuild via build_meta_ranker_matrix.py for current signals.

Run:
  # dry run on paper (no orders), see the plan
  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/live_runner.py
  # actually submit on paper
  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/live_runner.py --submit
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.live_signal_audit import (
    append_jsonl,
    build_equity_order_audit,
    build_option_order_audit,
    build_signal_audit,
)
from signals.meta_context.meta_ranker.score import score_frame
from core.live_4h_exec import (
    ExecPolicy,
    build_mixed_plan,
    defer_entries_if_market_closed,
    drop_failed_entry,
    exit_action as _shared_exit_action,
    record_exit_realized_pnl,
    shares_for_notional,
    submit_pending_open_entries,
)
from core.live_readiness import filter_entry_orders_for_readiness
from signals.meta_context.meta_ranker.options_exec import (
    equity_order_tif,
    route_option_or_shares,
    select_option,
)

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_MATRIX = HERE / "meta_ranker_matrix.parquet"
BARS_4H = REPO / "Data/shared/bars/4h"
STATE_PATH = HERE / "live_state.json"
BLACKLIST_PATH = HERE / "blacklist.txt"  # optional hard exclusions, one ticker per line (# comments ok)
MIN_FULL_BAR = 50  # ignore degenerate edge bars with fewer rows
AUDIT_MODULE = "meta_ranker"
DEFAULT_AUDIT_LOG = REPO / "Data/inference/meta_ranker/live_signal_audit.jsonl"


def _load_blacklist() -> set[str]:
    if not BLACKLIST_PATH.exists():
        return set()
    out = set()
    for line in BLACKLIST_PATH.read_text().splitlines():
        t = line.split("#", 1)[0].strip().upper()
        if t:
            out.add(t)
    return out


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"managed": {}, "history": []}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _ref_price(ticker: str) -> float | None:
    p = BARS_4H / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["close"])
    if b.empty:
        return None
    return float(b["close"].iloc[-1])


def _latest_full_bar(df: pd.DataFrame) -> pd.Timestamp:
    counts = df.groupby("timestamp").size()
    full = counts[counts >= max(MIN_FULL_BAR, int(0.25 * counts.max()))].index
    return full.max()


def _signal_audits(top: pd.DataFrame, *, bar: pd.Timestamp, entry_ok: dict[str, bool]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    total = max(1, len(top))
    for rank, (_idx, row) in enumerate(top.iterrows(), start=1):
        ticker = str(row["ticker"]).upper()
        out[ticker] = build_signal_audit(
            module=AUDIT_MODULE,
            ticker=ticker,
            score=row.get("s_combo"),
            side="long",
            rank=rank,
            rank_pct=1.0 - ((rank - 1) / total),
            signal_ts=bar,
            extra={
                "s_quality": row.get("s_quality"),
                "s_upside": row.get("s_upside"),
                "mom_score": row.get("mom_score"),
                "htf_score": row.get("htf_score"),
                "news_catalyst_score": row.get("news_catalyst_score"),
                "dollar_vol_pctile_252": row.get("dollar_vol_pctile_252"),
                "quality_entry_ok": bool(entry_ok.get(ticker, False)),
            },
        )
    return out


def _option_dte(expiry, bar) -> int | None:
    try:
        exp = pd.Timestamp(expiry).date()
        base = pd.Timestamp(bar).date()
        return max(0, int((exp - base).days))
    except Exception:
        return None


def _exit_action(gain, runs_held, bars_out, trimmed, args) -> tuple[str, str]:
    """Hold-based exit + scale-out (delegates to the shared 4H engine)."""
    policy = ExecPolicy(take_profit=args.take_profit, scale_frac=args.scale_frac,
                        horizon_bars=args.horizon_bars, grace_bars=args.grace_bars,
                        stop_loss=args.stop_loss, trail_stop=args.trail_stop)
    return _shared_exit_action(gain, runs_held, bars_out, trimmed, policy)


def main():
    ap = argparse.ArgumentParser(description="Meta Ranker long-only equity live runner (paper by default).")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--mode", choices=["equity", "options"], default="equity",
                    help="equity = buy shares; options = buy delta-filtered calls on qualifying names.")
    ap.add_argument("--roll-trading-days", type=int, default=5,
                    help="Options: roll to next monthly when nearest is within this many trading days.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--liquidity-floor", type=float, default=0.6,
                    help="Min dollar_vol_pctile_252 to be eligible (generalizable; raises holdout return).")
    ap.add_argument("--combo-floor", type=float, default=0.90,
                    help="Min combo rank-pct to be eligible (confidence gate; generalizable).")
    ap.add_argument("--quality-floor", type=float, default=0.4,
                    help="Min raw s_quality to be eligible for a NEW entry. Backtested (Sep25-May26): "
                         "gating top-5 cross-in entries on s_quality>=0.4 lifts mean forward-close "
                         "from ~8.9%% to ~13.4%% and blocks pyramiding into blow-off tops (e.g. CAR). "
                         "Set to a very negative number to disable.")
    # --- sizing ---
    ap.add_argument("--target-notional", type=float, default=5000.0,
                    help="Dollar size per new entry; shares/contracts are computed from the "
                         "current price/premium so exposure is comparable across tickers.")
    # --- exit policy (hold-based; rebalance-only churns — see backtest_exits.py) ---
    ap.add_argument("--take-profit", type=float, default=0.30, help="Scale out scale_frac at this gain, then ride the rest.")
    ap.add_argument("--scale-frac", type=float, default=0.16, help="Fraction to sell at take-profit.")
    ap.add_argument("--horizon-bars", type=int, default=53, help="Full exit after this many managed bars (~21d).")
    ap.add_argument("--grace-bars", type=int, default=None, help="Rank drop-out backstop: exit after N bars out of top-K. Default None = ride to horizon (backtest-preferred).")
    ap.add_argument("--stop-loss", type=float, default=0.39, help="Hard stop: full exit if gain <= -this from entry (premium for options). 0 disables.")
    ap.add_argument("--trail-stop", type=float, default=None, help="Trailing stop: full exit if value gives back this fraction from its peak. Default None = disabled (2026-07-18 cross-module search: no-trail beat trail on mean return per trade).")
    # Refuse to trade on stale data: if the latest full bar is older than this,
    # abort. Tight (half a day) so a missed nightly refresh can't trade yesterday.
    ap.add_argument("--max-staleness-days", type=float, default=0.5)
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--signal-audit-log", default=str(DEFAULT_AUDIT_LOG),
                    help="Append-only JSONL path for signal/order audit events; set empty to disable.")
    ap.add_argument("--live", action="store_true", help="Target the LIVE account (default: paper).")
    ap.add_argument("--submit", action="store_true", help="Actually place orders (default: dry-run).")
    ap.add_argument("--flush-pending-open", action="store_true", help="Pre-open: submit after-close queued entries still in the top-K (re-rank), then exit.")
    args = ap.parse_args()

    profile = "LIVE" if args.live else "PAPER"
    env_file = f".env#{profile}"
    run_mode = "SUBMIT" if args.submit else "DRY-RUN"
    print(f"=== Meta Ranker {args.mode} runner | account={profile} | mode={run_mode} | top_k={args.top_k} ===")

    # --- score + select ---
    df = pd.read_parquet(args.matrix).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    bar = _latest_full_bar(df)
    staleness = (datetime.now(timezone.utc) - bar.to_pydatetime()).total_seconds() / 86400.0
    print(f"latest full bar: {bar}  ({staleness:.1f} days old)")
    if staleness > args.max_staleness_days and not args.allow_stale:
        raise SystemExit(
            f"ABORT: matrix is {staleness:.1f}d stale (> {args.max_staleness_days}d). "
            f"Rebuild via build_meta_ranker_matrix.py or pass --allow-stale."
        )

    scored = score_frame(df[df["timestamp"] == bar].copy())
    # Eligibility filters (validated to generalize out-of-sample; see analyze_policy notes):
    #   liquidity floor + combo confidence floor + optional manual hard-exclusions.
    n0 = len(scored)
    elig = scored[scored["s_combo"] >= args.combo_floor]
    if "dollar_vol_pctile_252" in elig.columns:
        elig = elig[elig["dollar_vol_pctile_252"].fillna(0) >= args.liquidity_floor]
    blacklist = _load_blacklist()
    if blacklist:
        elig = elig[~elig["ticker"].str.upper().isin(blacklist)]
    top = elig.sort_values("s_combo", ascending=False).head(args.top_k)
    targets = list(top["ticker"])
    # Quality gate is applied to NEW ENTRIES only (held names exit via horizon/grace,
    # not a quality dip). A combo top-K name is bought only if its s_quality clears
    # the floor — this is the backtested "cross-in + quality" entry rule.
    quality_by_ticker = scored.set_index("ticker")["s_quality"].to_dict()
    entry_ok = {t: float(quality_by_ticker.get(t, float("-inf"))) >= args.quality_floor for t in targets}
    signal_audits = _signal_audits(top, bar=bar, entry_ok=entry_ok)
    print(f"\neligible {len(elig)}/{n0} after filters "
          f"(combo>={args.combo_floor}, liq>={args.liquidity_floor}, blacklist={len(blacklist)})")
    print(f"combo top-{args.top_k}: {targets}")
    if signal_audits:
        print("signal audit buckets: " + ", ".join(
            f"{t}:{signal_audits[t].get('score_bucket')}" for t in targets if t in signal_audits
        ))
    audit_log = Path(args.signal_audit_log) if str(args.signal_audit_log or "").strip() else None
    append_jsonl(
        audit_log,
        {
            "event": "signal_decision",
            "module": AUDIT_MODULE,
            "bar": bar,
            "targets": targets,
            "signal_audits": signal_audits,
            "filters": {
                "combo_floor": args.combo_floor,
                "liquidity_floor": args.liquidity_floor,
                "quality_floor": args.quality_floor,
                "blacklist_count": len(blacklist),
            },
        },
    )
    gated_out = [t for t in targets if not entry_ok[t]]
    if gated_out:
        print(f"quality-gated OUT of new entries (s_quality<{args.quality_floor}): {gated_out}")

    # --- account + positions ---
    client = AlpacaOptionsClient(env_file=env_file)
    acct = client.get_account()
    equity = float(acct["equity"])
    pos_info = {
        p["symbol"]: {"qty": int(float(p["qty"])),
                      "avg_entry": float(p.get("avg_entry_price", 0) or 0),
                      "current": float(p.get("current_price", 0) or 0)}
        for p in (client.get_positions() or [])
    }
    state = _load_state()
    managed = state.get("managed", {})  # symbols THIS strategy owns
    print(f"equity=${equity:,.0f}  target_notional/name=${args.target_notional:,.0f}  TP +{int(args.take_profit*100)}% "
          f"(sell {int(args.scale_frac*100)}%)  horizon {args.horizon_bars}b  grace {args.grace_bars}b")
    print(f"managed held: {sorted(managed)}")

    # Pre-open flush: submit after-close-queued entries still in TODAY's top-K
    # (re-rank against the freshly-scored `targets`), then exit — no position mgmt.
    if getattr(args, "flush_pending_open", False):
        if args.submit:
            res = submit_pending_open_entries(client, AUDIT_MODULE, targets,
                                              equity_tif_fn=equity_order_tif, pos_lookup=pos_info)
            managed.update(res["submitted"])
            state["managed"] = managed
            _save_state(state)
            print(f"pending-open flush: submitted {res['count']} / skipped {len(res['skipped'])}")
        else:
            print("pending-open flush (dry-run): add --submit to place queued entries")
        return

    if args.mode == "options":
        return _run_options(args, client, targets, state, managed, pos_info, bar, entry_ok, signal_audits)

    # --- hold-based reconciliation (only ever SELL symbols we manage) ---
    plan: list[tuple[str, str, int, str]] = []  # (symbol, side, qty, reason)
    order_audits: dict[str, dict] = {}
    new_managed: dict[str, dict] = {}
    exit_context: dict[str, tuple[str, dict]] = {}
    dropped: dict[str, dict] = {}
    for sym, st in managed.items():
        info_present = sym in pos_info
        held = pos_info.get(sym, {}).get("qty", 0)
        if held <= 0:
            status = "confirmed_flat" if info_present else "not_found"
            logger.warning("equity reconcile: dropping %s from managed — %s", sym, status)
            dropped[sym] = {"symbol": sym, "route": "equity", "status": status}
            continue  # position gone (closed elsewhere) — drop from state
        in_tgt = sym in targets
        st["runs_held"] = st.get("runs_held", 0) + 1
        st["bars_out"] = 0 if in_tgt else st.get("bars_out", 0) + 1
        info = pos_info.get(sym, {})
        gain = (info["current"] / info["avg_entry"] - 1) if info.get("avg_entry") else None
        action, reason = _exit_action(gain, st["runs_held"], st["bars_out"], st.get("trimmed", False), args)
        if action == "exit":
            plan.append((sym, "sell", held, reason))
            order_audits[sym] = build_equity_order_audit(
                signal_audit=signal_audits.get(sym),
                symbol=sym,
                side="sell",
                qty=held,
                reason=reason,
                reference_price=info.get("current"),
            )
            exit_context[sym] = (sym, dict(st))
            continue
        if action == "trim":
            q = int(math.floor(args.scale_frac * held))
            if q >= 1:
                plan.append((sym, "sell", q, reason))
                order_audits[sym] = build_equity_order_audit(
                    signal_audit=signal_audits.get(sym),
                    symbol=sym,
                    side="sell",
                    qty=q,
                    reason=reason,
                    reference_price=info.get("current"),
                )
                st["trimmed"] = True
        new_managed[sym] = st
    # entries: new top-K names not already held, that clear the quality gate
    for t in targets:
        if t in new_managed or pos_info.get(t, {}).get("qty", 0) > 0:
            continue
        if not entry_ok.get(t, False):
            continue
        qty = shares_for_notional(_ref_price(t), args.target_notional)
        plan.append((t, "buy", qty, "entry"))
        order_audits[t] = build_equity_order_audit(
            signal_audit=signal_audits.get(t),
            symbol=t,
            side="buy",
            qty=qty,
            reason="entry",
            reference_price=_ref_price(t),
        )
        new_managed[t] = {"qty": qty, "runs_held": 0, "bars_out": 0, "trimmed": False, "entry_bar": str(bar)}

    print(f"\n--- order plan ({len(plan)} orders) ---")
    for sym, side, qty, reason in plan:
        px = _ref_price(sym) or 0.0
        print(f"  {side.upper():4} {qty:>4} {sym:<6} (~${qty*px:,.0f} @ {px:.2f})  [{reason}]")
    if not plan:
        print("  (nothing to do — positions within policy)")

    _execute(
        args, client, plan, state, new_managed, bar, targets,
        is_option=False, signal_audits=signal_audits, order_audits=order_audits,
        exit_context=exit_context, dropped=dropped, module="meta_ranker", pos_lookup=pos_info,
    )


def _run_options(args, client, targets, state, managed, pos_info, bar, entry_ok=None, signal_audits=None):
    """Options path: shared 4H route + hold-based exit engine (mixed option/share)."""
    signal_audits = signal_audits or {}
    policy = ExecPolicy(take_profit=args.take_profit, scale_frac=args.scale_frac,
                        horizon_bars=args.horizon_bars, grace_bars=args.grace_bars,
                        stop_loss=args.stop_loss, trail_stop=args.trail_stop,
                        target_notional=args.target_notional,
                        roll_trading_days=args.roll_trading_days)
    res = build_mixed_plan(
        client, targets=targets, managed=managed, pos_info=pos_info, bar=bar,
        signal_audits=signal_audits, policy=policy, route_fn=route_option_or_shares,
        ref_price_fn=_ref_price, entry_ok=entry_ok, gate_reason="quality_gated",
    )
    _execute(
        args, client, res.plan, state, res.new_managed, bar, targets,
        is_option=True, limits=res.limits, signal_audits=signal_audits,
        order_audits=res.order_audits, contract_selection=res.contract_selection,
        exit_context=res.exit_context, dropped=res.dropped, module="meta_ranker", pos_lookup=pos_info,
    )


def _execute(
    args, client, plan, state, new_managed, bar, targets, *, is_option: bool,
    limits=None, signal_audits=None, order_audits=None, contract_selection=None,
    exit_context=None, dropped=None, module="meta_ranker", pos_lookup=None,
):
    """Submit a market/limit order plan (paper/live) and persist managed state. Dry-run by default.

    Managed state (runs_held/bars_out/trimmed) is persisted on every --submit
    pass, not only when the plan has orders — otherwise hold/grace counters
    freeze on quiet passes and horizon exits stall indefinitely. A failed exit
    order's pre-exit state is restored into new_managed (via exit_context) so a
    position that Alpaca never actually closed stays tracked instead of being
    silently orphaned.
    """
    limits = limits or {}
    if args.submit:
        plan, skipped, reason = filter_entry_orders_for_readiness(plan, new_managed=new_managed)
        if skipped:
            print(f"\nreadiness gate: skipped {len(skipped)} entry orders ({reason})")
        # After the close, queue entries for the next open instead of erroring on them.
        plan = defer_entries_if_market_closed(module, bar, plan, new_managed, limits)
        if plan:
            print("\nsubmitting...")
            for item in plan:
                sym, side, qty = item[0], item[1], item[2]
                # Per-order route: explicit 5th element (mixed options run) else the
                # run-wide default from is_option (pure equity/options mode).
                route = item[4] if len(item) > 4 else ("option" if is_option else "equity")
                lim = limits.get(sym)
                try:
                    if route == "option":
                        resp = client.submit_option_order(symbol=sym, qty=qty, side=side,
                                                          order_type="limit" if lim else "market",
                                                          time_in_force="day", limit_price=lim)
                    else:
                        resp = client.submit_order(symbol=sym, qty=qty, side=side,
                                                   order_type="market", time_in_force=equity_order_tif())
                    print(f"  OK {side} {qty} {sym}  id={resp.get('id', '?')}")
                    if str(side).strip().lower() == "sell":
                        es = exit_context.get(sym, (None, None))[1] if exit_context else None
                        record_exit_realized_pnl(client, module=module, item=item, resp=resp,
                                                 entry_state=es, pos_lookup=pos_lookup, bar=bar)
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {side} {qty} {sym}: {exc}")
                    if exit_context and sym in exit_context:
                        tkr, st = exit_context[sym]
                        new_managed[tkr] = st
                        logger.warning(
                            "_execute: exit submit failed for %s (%s) — restoring to managed state", tkr, sym,
                        )
                    else:
                        drop_failed_entry(new_managed, sym)
                finally:
                    # Save after every fill, not just at the end of the plan, so a
                    # sibling module's broker reconcile never finds a fresh
                    # position missing from this module's on-disk managed state
                    # (the 2026-07-23 IOT incident: Dealer Ranker's fresh buy was
                    # adopted and defensively liquidated by Swing because it
                    # wasn't yet persisted here).
                    state["managed"] = new_managed
                    _save_state(state)
        state["managed"] = new_managed
        _append_order_plan_audit(args, bar, targets, plan, signal_audits, order_audits, contract_selection, dropped)
        if plan:
            state.setdefault("history", []).append(
                {
                    "ts": _now(),
                    "bar": str(bar),
                    "mode": args.mode,
                    "targets": targets,
                    "orders": len(plan),
                    "signal_audits": signal_audits or {},
                    "order_audits": order_audits or {},
                }
            )
        _save_state(state)
        print(f"state updated -> {STATE_PATH}")
    else:
        _append_order_plan_audit(args, bar, targets, plan, signal_audits, order_audits, contract_selection, dropped)
        print("\n(dry-run: no orders submitted, state unchanged. Add --submit to execute.)")


def _append_order_plan_audit(args, bar, targets, plan, signal_audits, order_audits,
                             contract_selection=None, dropped=None) -> None:
    audit_log = Path(args.signal_audit_log) if str(args.signal_audit_log or "").strip() else None
    append_jsonl(
        audit_log,
        {
            "event": "order_plan",
            "module": AUDIT_MODULE,
            "bar": bar,
            "mode": args.mode,
            "submit": bool(args.submit),
            "targets": targets,
            "plan": [
                {"symbol": p[0], "side": p[1], "qty": p[2], "reason": p[3],
                 "route": p[4] if len(p) > 4 else args.mode}
                for p in plan
            ],
            "signal_audits": signal_audits or {},
            "order_audits": order_audits or {},
            "contract_selection": contract_selection or {},
            "dropped": dropped or {},
        },
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
