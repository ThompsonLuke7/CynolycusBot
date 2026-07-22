"""Shared 4H execution + exit engine for the technical option/share modules.

Meta Ranker, HTF Swing, Momentum Expansion, and Dealer Ranker are siblings:
same 4H (or near-close, for Dealer Ranker) cadence, same option-or-share
routing, same hold-based exit/scale-out. They differ only in how they SCORE
names (their models/labels) and, optionally, entry gating.

This module is the single source of truth for what those three do AFTER they
have a ranked target list: route each name to options or shares, manage held
positions with one take-profit / horizon / grace machine, and emit the plan +
audit. Callers inject their own price source and routing function so the engine
has no per-module coupling.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from core.live_signal_audit import build_equity_order_audit, build_option_order_audit
from core.live_readiness import filter_entry_orders_for_readiness

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecPolicy:
    """Hold-based exit + scale-out parameters (identical across the three modules).

    Rank rebalance-only exits (the pre-2026-07 baseline, grace<=3) were the
    WEAKEST across every module's OWN top-10 stream (~+1% mean/trade, ~2-bar
    hold) — dumped winners early and rode price losers that stayed top-ranked
    all the way down. A 2026-07-14 backtest (meta_ranker/backtest_exits.py)
    picked stop50/trail35/tp20-scale50/hz25 over that baseline. A follow-up
    2026-07-18 val-selected/test-frozen search (research/capstone/
    exit_policy_cross_module.csv), run independently against Momentum's, HTF
    Swing's, and Meta's own OOF top-10 streams, found a "tail-rider" shape —
    small early trim, no trail, longer horizon — beats that config's mean
    return per trade by 2-2.5x with comparable-or-better win rate in all three
    modules, at the cost of ~2x hold time. That's the current default; the
    trade-off is real (a "harvester" shape — full exit at a small target, no
    trim — wins on win-rate and capital efficiency instead, see the same CSV)
    and this hasn't been paper-validated live yet.
    """
    take_profit: float = 0.30         # scale out scale_frac at +this% gain, then ride the rest
    scale_frac: float = 0.16          # fraction sold at take-profit
    horizon_bars: int = 53            # full exit after this many managed bars (~21d)
    grace_bars: int | None = None     # None = ride to horizon (rank drop-out OFF); int N = exit after N bars out of top-K
    stop_loss: float | None = 0.39    # full exit if gain <= -this from ENTRY (premium for options); None disables
    trail_stop: float | None = None   # full exit if value falls this fraction from its PEAK (ratchet); None disables. "Tail-rider" (id4) config: 2026-07-18 val-selected/test-frozen search across Momentum/HTF/Meta's own OOF top-10 streams (research/capstone/exit_policy_cross_module.csv) found this shape (stop 39%, no trail, take-profit 30%/scale 16%, horizon 53) beats the prior stop50/trail35/tp20/scale50/hz25 default on mean return per trade (2-2.5x, ~10-12% vs ~4-5%) with comparable-or-better win rate in every module, at the cost of ~2x hold time (~52 vs ~25 bars). Prior config's own backtest note (mean +7.04%/61% win/ret-per-bar 0.0088 @ trail 0.35) is superseded by that search; kept here for history. Shares-only backtest — no option-premium path modeled, so real option stop/trail behavior may differ; not yet paper-validated live.
    contracts: int = 10               # option contracts per new entry
    shares: int = 100                 # shares per new entry (share-routed names)
    roll_trading_days: int = 5        # option monthly-roll buffer


@dataclass
class MixedPlan:
    # plan tuples: (symbol, side, qty, reason, route) route in {option, equity}
    plan: list[tuple[str, str, int, str, str]] = field(default_factory=list)
    new_managed: dict[str, dict] = field(default_factory=dict)
    order_audits: dict[str, dict] = field(default_factory=dict)
    contract_selection: dict[str, dict] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=dict)
    # Full-exit orders keyed by order symbol -> (ticker, pre-exit managed state).
    # If the exit order submission fails, execute_plan restores this state into
    # new_managed instead of leaving the position permanently unmanaged.
    exit_context: dict[str, tuple[str, dict]] = field(default_factory=dict)
    # Managed positions dropped this pass because broker qty read <= 0, keyed by
    # ticker -> {symbol, route, status}. status is "confirmed_flat" (broker
    # explicitly reports 0/short) or "not_found" (symbol absent from positions,
    # e.g. a transient read glitch or a never-filled entry).
    dropped: dict[str, dict] = field(default_factory=dict)


def exit_action(gain, runs_held, bars_out, trimmed, policy: ExecPolicy,
                *, peak_gain=None) -> tuple[str, str]:
    """Decide what to do with a held position. Returns (action, reason).

    Priority: (1) hard stop-loss (from entry), (2) trailing stop (ratchet from the
    peak — secures gains on the ride so a winner that rolls over exits before the
    full horizon; backtest: +ret/bar, same win rate), (3) take-profit scale-out,
    (4) time-horizon hard cap, (5) rank drop-out ONLY as an opt-in backstop (off by
    default). Replaced the old "exit the moment it drops out of the top-K" logic that
    both dumped winners and rode rank-sticky price losers to zero.

    `peak_gain` is the highest gain seen since entry (build_mixed_plan tracks it);
    when absent the trailing stop is skipped.
    """
    # 1) hard stop-loss — protects against ride-to-zero, esp. leveraged option premium
    if policy.stop_loss and gain is not None and gain <= -policy.stop_loss:
        return "exit", f"stop_-{int(policy.stop_loss * 100)}%"
    # 2) trailing stop — exit if value gives back trail_stop of its PEAK (ratchet)
    if (policy.trail_stop and gain is not None and peak_gain is not None
            and peak_gain > 0 and (1 + gain) <= (1 + peak_gain) * (1 - policy.trail_stop)):
        return "exit", f"trail_-{int(policy.trail_stop * 100)}%"
    # 3) take-profit scale-out — sell scale_frac, then ride the remainder
    if not trimmed and gain is not None and gain >= policy.take_profit:
        return "trim", f"take_profit_+{int(policy.take_profit * 100)}%"
    # 4) time-horizon hard cap on the hold
    if runs_held >= policy.horizon_bars:
        return "exit", "horizon"
    # 5) rank drop-out backstop — disabled by default (grace_bars=None => ride to horizon)
    if policy.grace_bars is not None and bars_out > policy.grace_bars:
        return "exit", "dropped_out"
    return "hold", ""


def _option_dte(expiry: str | None, bar) -> int | None:
    if not expiry:
        return None
    try:
        return max(0, int((pd.Timestamp(expiry).date() - pd.Timestamp(bar).date()).days))
    except Exception:
        return None


def build_mixed_plan(
    client,
    *,
    targets: list[str],
    managed: dict[str, dict],
    pos_info: dict[str, dict],
    bar,
    signal_audits: dict[str, dict],
    policy: ExecPolicy,
    route_fn: Callable,
    ref_price_fn: Callable[[str], float | None],
    entry_ok: dict[str, bool] | None = None,
    gate_reason: str = "gated",
    verbose: bool = True,
) -> MixedPlan:
    """Manage held positions + route new entries into one mixed option/share plan."""
    out = MixedPlan()
    sa = signal_audits or {}

    def _sell_audit(tkr, sym, qty, route):
        if route == "equity":
            return build_equity_order_audit(signal_audit=sa.get(tkr), symbol=sym, side="sell", qty=qty)
        return build_option_order_audit(signal_audit=sa.get(tkr), option_symbol=sym,
                                        route="call_option", side="sell", qty=qty)

    # 1) manage existing positions (option OR share) with one exit machine.
    for tkr, st in managed.items():
        route = st.get("route", "option")
        sym = st.get("occ") if route == "option" else st.get("symbol", tkr)
        info_present = bool(sym) and sym in pos_info
        held = pos_info.get(sym, {}).get("qty", 0) if sym else 0
        if held <= 0:
            status = "confirmed_flat" if info_present else "not_found"
            logger.warning("build_mixed_plan: dropping %s (%s) from managed — %s", tkr, sym, status)
            out.dropped[tkr] = {"symbol": sym, "route": route, "status": status}
            continue
        in_tgt = tkr in targets
        st["runs_held"] = st.get("runs_held", 0) + 1
        st["bars_out"] = 0 if in_tgt else st.get("bars_out", 0) + 1
        info = pos_info.get(sym, {})
        gain = (info["current"] / info["avg_entry"] - 1) if info.get("avg_entry") else None
        # Persist the broker-authoritative basis and latest mark on every pass.
        # This is attribution-only state: exit decisions still use ``gain`` as
        # before, but the daily briefing can value an open 4H book after the
        # server has stopped without guessing from signal reference prices.
        if info.get("avg_entry"):
            st["entry_avg_price"] = float(info["avg_entry"])
        if info.get("current") is not None:
            st["last_mark_price"] = float(info["current"])
            st["last_mark_bar"] = str(bar)
        if gain is not None:
            st["unrealized_gain"] = float(gain)
        if gain is not None:  # ratchet the peak so the trailing stop has a reference
            st["peak_gain"] = max(st.get("peak_gain", gain), gain)
        action, reason = exit_action(gain, st["runs_held"], st["bars_out"], st.get("trimmed", False),
                                     policy, peak_gain=st.get("peak_gain"))
        if action == "exit":
            out.plan.append((sym, "sell", held, reason, route))
            out.order_audits[sym] = _sell_audit(tkr, sym, held, route)
            out.exit_context[sym] = (tkr, dict(st))
            continue
        if action == "trim":
            q = int(math.floor(policy.scale_frac * held))
            if q >= 1:
                out.plan.append((sym, "sell", q, reason, route))
                out.order_audits[sym] = _sell_audit(tkr, sym, q, route)
                st["trimmed"] = True
        out.new_managed[tkr] = st

    # 2) entries: route each new top-K name to options or shares.
    if verbose:
        print("\n--- order routing (options if optionable, else shares; delta 0.35-0.60) ---")
    entry_ok = entry_ok or {}
    for t in targets:
        if t in out.new_managed:
            continue
        if entry_ok and not entry_ok.get(t, False):
            out.contract_selection[t] = {"action": "skip", "reason": gate_reason, "signal_audit": sa.get(t)}
            continue
        px = ref_price_fn(t)
        if not px or px <= 0:
            out.contract_selection[t] = {"action": "skip", "reason": "no_ref_price", "signal_audit": sa.get(t)}
            if verbose:
                print(f"  ! {t:<6} skip: no price")
            continue
        route, order, reason = route_fn(client, t, px, roll_trading_days=policy.roll_trading_days)
        if route == "skip":
            out.contract_selection[t] = {
                "action": "skip",
                "reason": reason,
                **(order or {}),
                "signal_audit": sa.get(t),
            }
            if verbose:
                print(f"  ! {t:<6} skip: {reason}")
            continue
        if route == "option":
            occ = order["occ"]
            if pos_info.get(occ, {}).get("qty", 0) > 0:
                out.contract_selection[t] = {"action": "skip", "reason": "already_held", "occ": occ,
                                             "signal_audit": sa.get(t)}
                continue
            out.contract_selection[t] = {
                "action": "option", "occ": occ, "ref_price": px,
                "delta": order.get("delta"), "mid": order.get("mid"), "limit": order.get("limit"),
                "strike": order.get("strike"), "expiry": order.get("expiry"),
                "open_interest": order.get("open_interest"), "volume": order.get("volume"),
                "spread": order.get("spread"), "dealer_gate": order.get("dealer_gate"),
                "signal_audit": sa.get(t),
            }
            out.plan.append((occ, "buy", policy.contracts, "entry", "option"))
            out.limits[occ] = order["limit"]
            out.order_audits[occ] = build_option_order_audit(
                signal_audit=sa.get(t), option_symbol=occ, route="call_option", side="buy",
                qty=policy.contracts, underlying_price=px, strike=order.get("strike"),
                premium=order.get("limit") or order.get("mid"), limit_price=order.get("limit"),
                mid_price=order.get("mid"), delta=order.get("delta"),
                dte=_option_dte(order.get("expiry"), bar), expiration=order.get("expiry"),
            )
            out.new_managed[t] = {"route": "option", "occ": occ, "contracts": policy.contracts,
                                  "runs_held": 0, "bars_out": 0, "trimmed": False, "entry_bar": str(bar),
                                  "expiry": order.get("expiry"), "signal_audit": sa.get(t),
                                  "order_audit": out.order_audits[occ]}
            if verbose:
                print(f"  + {t:<6} {occ:<20} x{policy.contracts} exp={order.get('expiry')} "
                      f"delta={order.get('delta'):.2f} mid={order['mid']:.2f}  oi={order.get('open_interest')}")
        else:  # equity — not a good options candidate, trade shares
            if pos_info.get(t, {}).get("qty", 0) > 0:
                out.contract_selection[t] = {"action": "skip", "reason": "already_held_equity",
                                             "signal_audit": sa.get(t)}
                continue
            out.contract_selection[t] = {"action": "equity", "reason": reason, "ref_price": px,
                                         "shares": policy.shares,
                                         "dealer_gate": (order or {}).get("dealer_gate"),
                                         "signal_audit": sa.get(t)}
            out.plan.append((t, "buy", policy.shares, "entry", "equity"))
            out.order_audits[t] = build_equity_order_audit(
                signal_audit=sa.get(t), symbol=t, side="buy", qty=policy.shares,
                reason="entry", reference_price=px,
            )
            out.new_managed[t] = {"route": "equity", "symbol": t, "shares": policy.shares,
                                  "runs_held": 0, "bars_out": 0, "trimmed": False, "entry_bar": str(bar),
                                  "signal_audit": sa.get(t), "order_audit": out.order_audits[t]}
            if verbose:
                print(f"  ~ {t:<6} {policy.shares} shares  [{reason}]")

    if verbose:
        print(f"\n--- order plan ({len(out.plan)} orders) ---")
        for sym, side, qty, reason, route in out.plan:
            tag = ((f" @limit {out.limits[sym]}" if sym in out.limits else " @market")
                   if route == "option" else " shares @market")
            print(f"  {side.upper():4} {qty:>3} {sym}{tag}  [{reason}]")
        if not out.plan:
            print("  (nothing to do — positions within policy)")
    return out


def _reverify_buys_not_held(client, plan: list, new_managed: dict | None) -> list:
    """Drop BUY orders for symbols the broker now shows as already held.

    Two independently-scheduled 4H-family modules (e.g. Dealer Ranker and the
    Swing runner) can rank the same underlying and select the same
    nearest-ATM contract on the same day; each module's own ``pos_info``
    snapshot is taken once near the top of its run, so an entry the OTHER
    module places moments later isn't visible in it yet (observed
    2026-07-17: both modules bought AUR260724C00006000 within ~20s of each
    other, each believing it owned the position outright). Re-checking live
    positions immediately before submit shrinks that race window from the
    length of a full run down to a single API round-trip. Best-effort: any
    failure to fetch positions falls through to the normal submit path
    unchanged rather than blocking real entries on a transient API hiccup.
    """
    if not any(str(item[1]).strip().lower() == "buy" for item in plan):
        return plan
    try:
        held: set[str] = set()
        for p in client.get_positions() or []:
            try:
                if float(p.get("qty", 0) or 0) > 0:
                    held.add(str(p["symbol"]).upper())
            except (TypeError, ValueError, KeyError):
                continue
    except Exception:
        return plan
    out = []
    for item in plan:
        sym, side = item[0], item[1]
        if str(side).strip().lower() == "buy" and str(sym).upper() in held:
            logger.warning(
                "execute_plan: skipping buy %s — already held (cross-module race guard)", sym,
            )
            if new_managed is not None:
                drop_failed_entry(new_managed, sym)
            continue
        out.append(item)
    return out


def execute_plan(
    client, *, plan, limits, submit: bool, equity_tif_fn: Callable[[], str],
    new_managed: dict[str, dict] | None = None,
    exit_context: dict[str, tuple[str, dict]] | None = None,
    module: str | None = None,
    pos_lookup: dict | None = None,
    bar: Any = None,
) -> set[str]:
    """Submit a mixed plan (paper/live). Each order routes per its 5th tuple element.

    If a full-exit order's submission raises, the position was never actually
    closed at the broker; when `new_managed`/`exit_context` are supplied, the
    pre-exit managed state is restored into `new_managed` so the position stays
    tracked and gets re-evaluated next pass instead of being silently orphaned.
    Conversely, a rejected ENTRY order is dropped from `new_managed` so a buy the
    broker refused (e.g. after-hours 403) never becomes a phantom position.
    Filled exits are written to the realized-PnL ledger when `module`/`pos_lookup`
    are supplied. Returns the set of order symbols whose submission failed.
    """
    limits = limits or {}
    failed: set[str] = set()
    if not (submit and plan):
        return failed
    plan, skipped, reason = filter_entry_orders_for_readiness(plan, new_managed=new_managed)
    if skipped:
        print(f"\nreadiness gate: skipped {len(skipped)} entry orders ({reason})")
        failed.update(skipped)
    # After the close, queue entries for the next open instead of erroring on them.
    plan = defer_entries_if_market_closed(module, bar, plan, new_managed, limits)
    if not plan:
        return failed
    plan = _reverify_buys_not_held(client, plan, new_managed)
    if not plan:
        return failed
    print("\nsubmitting...")
    for item in plan:
        sym, side, qty = item[0], item[1], item[2]
        route = item[4] if len(item) > 4 else "option"
        lim = limits.get(sym)
        try:
            if route == "option":
                resp = client.submit_option_order(symbol=sym, qty=qty, side=side,
                                                  order_type="limit" if lim else "market",
                                                  time_in_force="day", limit_price=lim)
            else:
                resp = client.submit_order(symbol=sym, qty=qty, side=side,
                                           order_type="market", time_in_force=equity_tif_fn())
            print(f"  OK {side} {qty} {sym}  id={resp.get('id', '?')}")
            if module and str(side).strip().lower() == "sell":
                es = exit_context.get(sym, (None, None))[1] if exit_context else None
                record_exit_realized_pnl(client, module=module, item=item, resp=resp,
                                         entry_state=es, pos_lookup=pos_lookup, bar=bar)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {side} {qty} {sym}: {exc}")
            failed.add(sym)
            if new_managed is not None and exit_context and sym in exit_context:
                tkr, st = exit_context[sym]
                new_managed[tkr] = st
                logger.warning(
                    "execute_plan: exit submit failed for %s (%s) — restoring to managed state",
                    tkr, sym,
                )
            elif new_managed is not None:
                drop_failed_entry(new_managed, sym)
    return failed


def _resp_fill_price(resp) -> float | None:
    if not isinstance(resp, dict):
        return None
    raw = resp.get("filled_avg_price")
    try:
        return float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None


def poll_exit_fill_price(client, resp, *, timeout_s: float = 4.0, poll_s: float = 0.4) -> float | None:
    """Best-effort fill price for a just-submitted order (paper fills settle fast).

    Returns None rather than blocking indefinitely so the live loop never stalls.
    """
    fp = _resp_fill_price(resp)
    if fp:
        return fp
    order_id = str((resp or {}).get("id", "")).strip()
    if not order_id or not hasattr(client, "get_order"):
        return None
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        try:
            cur = client.get_order(order_id)
        except Exception:
            continue
        fp = _resp_fill_price(cur)
        if fp:
            return fp
        status = str((cur or {}).get("status", "")).strip().lower()
        if status in {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}:
            break
    return fp


def record_exit_realized_pnl(client, *, module, item, resp, entry_state, pos_lookup, bar,
                             ledger_root: str = "Data/inference") -> None:
    """Append a realized-PnL row for a filled SELL (exit or trim). Never raises.

    Cost basis is the broker's average entry (from `pos_lookup`), so this is a
    true realized number rather than a reference-price proxy — the piece the 4H
    modules were missing. One JSON line per exit lands in
    Data/inference/<module>/closed_trades.jsonl for the daily report.
    """
    try:
        import json
        from pathlib import Path
        sym, side, qty = item[0], item[1], item[2]
        if str(side).strip().lower() != "sell":
            return
        reason = item[3] if len(item) > 3 else "exit"
        route = item[4] if len(item) > 4 else "option"
        mult = 100.0 if route == "option" else 1.0
        exit_fill = poll_exit_fill_price(client, resp)
        avg_entry = (pos_lookup or {}).get(sym, {}).get("avg_entry")
        realized = None
        if exit_fill is not None and avg_entry:
            realized = round((exit_fill - float(avg_entry)) * mult * float(qty), 2)
        es = entry_state or {}
        # Underlying ticker: prefer explicit state, else strip an OCC symbol
        # (e.g. SOC260717C00005000 -> SOC) so option rows aren't keyed by the OCC.
        import re
        occ_m = re.match(r"^([A-Z]+)\d{6}[CP]\d{8}$", str(sym))
        ticker = es.get("ticker") or (es.get("symbol") if route != "option" else None) \
            or (occ_m.group(1) if occ_m else sym)
        rec = {
            "ts": now_utc_iso(),
            "module": module,
            "bar": str(bar),
            "ticker": ticker,
            "order_symbol": sym,
            "route": route,
            "side": "sell",
            "qty": float(qty),
            "exit_reason": reason,
            "entry_avg_price": float(avg_entry) if avg_entry else None,
            "exit_fill_price": exit_fill,
            "realized_pnl": realized,
            "entry_bar": es.get("entry_bar"),
            "runs_held": es.get("runs_held"),
            "order_id": str((resp or {}).get("id", "")) or None,
        }
        out = Path(ledger_root) / str(module) / "closed_trades.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        logger.info("realized-PnL logged: %s %s qty=%s reason=%s pnl=%s",
                    module, sym, qty, reason, realized)
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_exit_realized_pnl failed (%s %s): %s", module, item, exc)


def drop_failed_entry(new_managed: dict | None, sym: str) -> None:
    """Remove a position whose ENTRY order was rejected from managed state.

    Without this, a 403/422-rejected buy still leaves a phantom position in
    state that the account never actually opened (the source of the untracked /
    "restored unknown" positions). Equity is keyed by ticker (== order symbol);
    options are keyed by ticker with the OCC under 'occ'/'symbol'.
    """
    if not new_managed:
        return
    if sym in new_managed:
        new_managed.pop(sym, None)
        return
    for tkr, st in list(new_managed.items()):
        if isinstance(st, dict) and sym in (st.get("occ"), st.get("symbol")):
            new_managed.pop(tkr, None)
            return


# --- after-close entry deferral -------------------------------------------------
# The pm 4H bar (18:00 UTC = 2-6pm ET) only finishes AFTER the 16:00 equity close,
# so its entries can never fill same-day and after-hours submission is rejected
# (equity opg 403 / options 422). Instead of erroring, queue those entries and let
# a pre-open flush re-rank them against the fresh top-K and submit at the next open
# (the "next_open" fill validated by backtest_exits.py, ~0.17%/trade erosion).

def pending_open_path(module: str, ledger_root: str = "Data/inference"):
    from pathlib import Path
    return Path(ledger_root) / str(module) / "pending_open_entries.json"


def _managed_key_for_symbol(new_managed: dict | None, sym: str) -> str | None:
    if not new_managed:
        return None
    if sym in new_managed:
        return sym  # equity keyed by ticker
    for tkr, st in new_managed.items():
        if isinstance(st, dict) and sym in (st.get("occ"), st.get("symbol")):
            return tkr  # option keyed by ticker, OCC under 'occ'/'symbol'
    return None


def defer_entries_if_market_closed(module, bar, plan, new_managed, limits, *,
                                   now=None, ledger_root: str = "Data/inference") -> list:
    """When the US equity market is CLOSED, pull ENTRY orders out of the plan into a
    per-module pending-open queue (they'd be rejected after hours) and prune them
    from new_managed (nothing was placed). Exits stay in the plan. Returns the plan
    to submit now — unchanged when the market is open or `module` is unset.
    """
    if not module:
        return plan
    try:
        from core.calendar import is_market_open_now
        if is_market_open_now(now):
            return plan
    except Exception:
        return plan  # fail safe: if we can't tell, behave as before (submit)
    import json
    kept, deferred = [], []
    for item in plan:
        sym = item[0]
        reason = item[3] if len(item) > 3 else ""
        route = item[4] if len(item) > 4 else "option"
        if reason == "entry":
            tkr = _managed_key_for_symbol(new_managed, sym)
            deferred.append({
                "order_symbol": sym, "side": item[1], "qty": item[2], "route": route,
                "limit": (limits or {}).get(sym), "ticker": tkr or sym, "bar": str(bar),
                "managed": (new_managed or {}).get(tkr) if tkr else None,
            })
            if tkr and new_managed is not None:
                new_managed.pop(tkr, None)  # not placed -> don't track yet
        else:
            kept.append(item)
    if deferred:
        out = pending_open_path(module, ledger_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        by_sym = {}
        if out.exists():
            try:
                by_sym = {e["order_symbol"]: e for e in json.loads(out.read_text()).get("entries", [])}
            except Exception:
                by_sym = {}
        for e in deferred:
            by_sym[e["order_symbol"]] = e  # newest wins
        out.write_text(json.dumps({"updated": now_utc_iso(), "entries": list(by_sym.values())},
                                  default=str, indent=1))
        logger.info("%s: market closed — deferred %d entries to next open -> %s",
                    module, len(deferred), out)
    return kept


def submit_pending_open_entries(client, module, targets, *, equity_tif_fn,
                                pos_lookup=None, ledger_root: str = "Data/inference") -> dict:
    """Pre-open flush: submit queued after-close entries that are STILL in the current
    top-K (re-rank) and not already held, as normal day orders. Clears the queue.
    Returns {"submitted": {ticker: managed_dict}, "skipped": [...], "count": n}; never
    raises on an individual order (records it as skipped instead).
    """
    import json
    out = pending_open_path(module, ledger_root)
    if not out.exists():
        return {"submitted": {}, "skipped": [], "count": 0}
    try:
        entries = json.loads(out.read_text()).get("entries", [])
    except Exception:
        entries = []
    tset = set(targets or [])
    pos_lookup = pos_lookup or {}
    submitted, skipped = {}, []
    eligible = []
    for rec in entries:
        tkr, sym = rec.get("ticker"), rec.get("order_symbol")
        if tset and tkr not in tset:
            skipped.append({**rec, "skip": "no_longer_top_k"}); continue
        if pos_lookup.get(sym, {}).get("qty", 0) > 0:
            skipped.append({**rec, "skip": "already_held"}); continue
        eligible.append(rec)

    readiness_plan = [
        (rec.get("order_symbol"), rec.get("side"), rec.get("qty"), "pending_open", rec.get("route", "option"))
        for rec in eligible
    ]
    _kept, readiness_skipped, readiness_reason = filter_entry_orders_for_readiness(readiness_plan)
    if readiness_skipped:
        logger.warning(
            "%s pending-open: readiness gate skipped %d entries (%s)",
            module, len(readiness_skipped), readiness_reason,
        )

    for rec in eligible:
        tkr, sym = rec.get("ticker"), rec.get("order_symbol")
        route = rec.get("route", "option")
        if str(sym) in readiness_skipped:
            skipped.append({**rec, "skip": f"readiness:{readiness_reason}"})
            continue
        try:
            lim = rec.get("limit")
            if route == "option":
                resp = client.submit_option_order(symbol=sym, qty=rec["qty"], side=rec["side"],
                          order_type="limit" if lim else "market", time_in_force="day", limit_price=lim)
            else:
                resp = client.submit_order(symbol=sym, qty=rec["qty"], side=rec["side"],
                          order_type="market", time_in_force=equity_tif_fn())
            logger.info("%s pending-open: submitted %s %s x%s id=%s",
                        module, rec["side"], sym, rec["qty"], (resp or {}).get("id", "?"))
            if tkr and rec.get("managed"):
                submitted[tkr] = dict(rec["managed"])
        except Exception as exc:  # noqa: BLE001
            skipped.append({**rec, "skip": f"submit_failed:{exc}"})
    try:
        out.unlink()
    except Exception:
        pass
    logger.info("%s pending-open flush: submitted=%d skipped=%d", module, len(submitted), len(skipped))
    return {"submitted": submitted, "skipped": skipped, "count": len(submitted)}


def order_plan_audit_record(*, module, bar, mode, submit, targets, plan,
                            signal_audits, order_audits, contract_selection,
                            dropped=None) -> dict[str, Any]:
    return {
        "event": "order_plan",
        "module": module,
        "bar": bar,
        "mode": mode,
        "submit": bool(submit),
        "targets": targets,
        "plan": [
            {"symbol": p[0], "side": p[1], "qty": p[2], "reason": p[3],
             "route": p[4] if len(p) > 4 else mode}
            for p in plan
        ],
        "signal_audits": signal_audits or {},
        "order_audits": order_audits or {},
        "contract_selection": contract_selection or {},
        "dropped": dropped or {},
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
