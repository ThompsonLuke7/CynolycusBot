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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from core.live_signal_audit import build_equity_order_audit, build_option_order_audit
from core.live_readiness import filter_entry_orders_for_readiness

logger = logging.getLogger(__name__)

# Where realized-PnL ledgers and the pending-order queues live. A module-level
# constant rather than a literal default so the test suite can redirect ALL of
# them at once (see conftest.py). Before that, core/tests/test_live_4h_exec_*.py
# called execute_plan(module="dealer_ranker") without a ledger_root and every
# pytest run appended fake "AAA" fills to the REAL
# Data/inference/dealer_ranker/closed_trades.jsonl — 38 of its 41 rows were test
# fixtures by 2026-08-03, against 3 genuine trades.
DEFAULT_LEDGER_ROOT = "Data/inference"


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
    target_notional: float = 5000.0   # target $ per new entry; shares/contracts sized from this
    roll_trading_days: int = 15       # option monthly-roll buffer


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
    # Positions whose mark moved so far in one bar that the move is not credibly
    # a price move (see _implausible_mark_move). Keyed by ticker -> diagnostic.
    # These are HELD and stay managed; no exit is evaluated for them this pass.
    anomalies: dict[str, dict] = field(default_factory=dict)
    # Positions the broker still reports AFTER an accepted exit order failed to
    # fill. Keyed by ticker -> the exit_pending record that went stale. These are
    # re-planned this pass; a name that keeps reappearing here is stuck and needs
    # a human, not another retry.
    stuck_exits: dict[str, dict] = field(default_factory=dict)


# --- corporate-action / bad-mark guard -----------------------------------------
# An equity mark that collapses ~10x between two 4H bars is not a price move. On
# 2026-08-10 Meta held TENX (100sh @ 15.99) and SION (105sh @ 48.00); both marks
# came back on Monday at roughly one tenth of Friday's close (13.30 -> 1.42 and
# 49.61 -> 4.46) with the broker's share counts UNCHANGED, which is the signature
# of a forward split whose cost basis was never adjusted. Both tripped
# `stop_-39%` and were liquidated for -$1,457 and -$4,572. The tell is that no
# other name in a 173-position equity book moved worse than -18.9% that session,
# and dividing Friday's close by exactly 10 turns both into ordinary days
# (+6.8% / -10.1%). See research/daily_live_reports/2026-08-10.md.
#
# Nothing local can separate "10:1 split we did not process" from "the company
# actually collapsed 90% overnight", and the correct action is the same for both:
# do not trade on it. AGENTS.md requires failing fast on misaligned data rather
# than continuing, so the position is held, left managed, and reported loudly for
# a human — never auto-liquidated at a price we cannot vouch for.
#
# Equity only. Option premia legitimately lose 80%+ in a single 4H bar (that is
# the leverage working as designed), so the same threshold there would fire on
# ordinary stops and block the exits that matter most.
IMPLAUSIBLE_EQUITY_BAR_MOVE = 0.70


def _implausible_mark_move(route: str, prev_mark, new_mark,
                           threshold: float = IMPLAUSIBLE_EQUITY_BAR_MOVE) -> dict | None:
    """Diagnostic when an equity mark jumps further in one bar than a price can.

    Returns None when the move is credible, the check does not apply (options,
    no prior mark, non-positive prices), or this is the first time we have seen
    the position.

    `ratio` is reported raw and deliberately NOT rounded to a "suspected N:1
    split". At the magnitudes that trip this guard the integers are only ~10%
    apart, so any tolerance loose enough to match a real split also matches
    everything else — TENX's 9.37 would be labelled a 9:1 split, which is not a
    corporate action anyone performed. A confident-looking wrong diagnosis is
    worse than the raw number, which an operator can check against a real
    corporate-action source.
    """
    if route == "option":
        return None
    try:
        prev, new = float(prev_mark), float(new_mark)
    except (TypeError, ValueError):
        return None
    if prev <= 0 or new <= 0:
        return None
    move = new / prev - 1.0
    if abs(move) < threshold:
        return None
    return {
        "prev_mark": prev,
        "new_mark": new,
        "bar_move": round(move, 6),
        "threshold": threshold,
        "ratio": round(prev / new if new < prev else new / prev, 4),
        "direction": "down" if new < prev else "up",
    }


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


def shares_for_notional(px: float | None, target_notional: float) -> int:
    """Whole shares nearest ``target_notional`` dollars at price ``px``, floored at 1.

    Fixed 100-share entries meant wildly different risk per name (a $5 stock
    was $500 of exposure, a $200 stock was $20,000) and made %/$ gain figures
    incomparable across the book. Sizing off a fixed dollar target instead
    keeps exposure -- and therefore %/$ gain -- consistent name to name.
    """
    if not px or px <= 0:
        return 1
    return max(1, round(target_notional / px))


def contracts_for_notional(premium: float | None, target_notional: float) -> int:
    """Whole option contracts nearest ``target_notional`` dollars, floored at 1.

    One contract controls 100 shares, so notional per contract = premium*100
    (e.g. a $50 premium = $5,000/contract). Mirrors ``shares_for_notional`` so
    equity and option entries carry comparable dollar exposure.
    """
    if not premium or premium <= 0:
        return 1
    return max(1, round(target_notional / (premium * 100.0)))


def _fmt_num(value, places: int = 2, missing: str = "n/a") -> str:
    """Format a number for a log line, tolerating None and non-numerics.

    Diagnostics run inside the order-building path, so they must degrade to a
    placeholder rather than raise — see the call site in build_mixed_plan.
    """
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return missing


# How far the broker's position mark may diverge from a live two-sided mid before
# the mark is treated as stale. Options legitimately move fast, so this is NOT a
# move threshold like IMPLAUSIBLE_EQUITY_BAR_MOVE — it compares two readings of
# the SAME instant and only fires when they disagree about the present.
STALE_OPTION_MARK_DIVERGENCE = 0.25


def _corroborate_option_exit_mark(client, symbol: str, broker_mark, avg_entry):
    """Second opinion on an option mark that is about to trigger an exit.

    Broker `current_price` on an option is a last print or a one-sided quote, and
    after the close it can be neither current nor tradeable. 2026-08-13 is the
    worked example: momentum's AAOI260821C00140000 marked 4.70 against a 9.20
    basis at the 16:30 ET run, tripping `stop_-39%`. The contract traded at 9.90
    the next morning — the position was actually +7.6%, and the realized ledger
    recorded a *positive* `stop_overshoot` of 0.466. A stop fired on a stale
    quote.

    Only called for positions already deemed exit-worthy, so the extra quote
    request costs one call per position actually trading, not one per position
    held (Meta and HTF carry 60-80 managed names apiece).

    Returns ``(corrected_gain, corrected_mark)`` when a live two-sided quote
    materially disagrees with the broker mark, else None. Never returns a
    decision — the caller re-runs the exit policy so one code path owns the rule.
    """
    try:
        entry = float(avg_entry)
        mark = float(broker_mark)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or mark <= 0:
        return None
    _bid, mid = _option_quote(client, symbol)
    try:
        mid = float(mid)
    except (TypeError, ValueError):
        return None
    if mid <= 0:
        return None
    if abs(mark / mid - 1.0) <= STALE_OPTION_MARK_DIVERGENCE:
        return None
    return mid / entry - 1.0, mid


def _size_key(route: str) -> str:
    """Managed-state key holding the position size for this route."""
    return "contracts" if route == "option" else "shares"


def _apply_trim_to_size(st: dict, route: str, sold_qty) -> None:
    """Reduce the persisted size after a partial exit.

    Only attribution/reporting depends on this — `build_mixed_plan` re-reads the
    live quantity from the broker each run — so a missing or unparseable size is
    left alone rather than guessed at. Never goes below zero: a size that would
    go negative means state and broker had already diverged, and inventing a
    negative position would make the next report worse, not better.
    """
    key = _size_key(route)
    current = st.get(key)
    if current is None:
        return
    try:
        remaining = int(current) - int(sold_qty)
    except (TypeError, ValueError):
        return
    st[key] = max(0, remaining)


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
    module: str | None = None,
    ledger_root: str | None = None,
) -> MixedPlan:
    """Manage held positions + route new entries into one mixed option/share plan.

    `module`/`ledger_root` are only needed to settle exits that were accepted but
    never filled on a previous run (see mark_exit_unconfirmed). Without them the
    positions still resolve correctly; only the ledger row is skipped.
    """
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
            # A position carrying exit_pending was left owned on purpose because
            # its exit order was accepted but never filled. The broker no longer
            # reporting it is the settle signal: book the close now, with the
            # basis snapshotted at submit time (the broker can no longer supply
            # it). Without this the loss simply never reaches the ledger.
            settled = False
            if isinstance(st.get("exit_pending"), dict) and module:
                settled = resolve_settled_exit(
                    client, module=module, ticker=tkr, symbol=sym, state=st, bar=bar,
                    ledger_root=ledger_root)
            logger.warning("build_mixed_plan: dropping %s (%s) from managed — %s%s",
                           tkr, sym, status, " (pending exit settled)" if settled else "")
            out.dropped[tkr] = {"symbol": sym, "route": route, "status": status,
                                "exit_settled": settled}
            continue
        # Still held with a resting exit order: the order did not fill and the
        # position is genuinely stuck. Clear the flag so the exit machine
        # re-evaluates and re-submits this pass (broker day orders die at the
        # close, so there is nothing to stack against).
        if isinstance(st.get("exit_pending"), dict):
            stale = st.pop("exit_pending")
            out.stuck_exits[tkr] = {"symbol": sym, "route": route, **stale}
            logger.error(
                "build_mixed_plan: %s (%s) STILL HELD after an accepted exit order "
                "(%s, submitted %s) never filled — re-evaluating the exit this pass. "
                "A contract that repeatedly fails to sell needs a human look.",
                tkr, sym, stale.get("reason"), stale.get("submitted_bar"),
            )
        # The broker confirms the position exists, so an entry flagged
        # unconfirmed by execute_plan has now settled.
        st.pop("pending_fill", None)
        st.pop("entry_order_id", None)
        info = pos_info.get(sym, {})
        # Validate the mark BEFORE it can drive an exit, and before the bar
        # counters advance. `last_mark_price` is the mark this loop persisted on
        # its previous pass, so this compares like with like (broker
        # current_price to broker current_price) and needs no extra data source.
        # A position that fails the check is left exactly as it was — still held,
        # still managed, runs_held/bars_out not advanced (a bar we refused to
        # evaluate must not age it toward the horizon exit) and last_mark_price
        # NOT overwritten, so the next pass re-tests against the same trusted
        # prior mark and either clears or keeps alerting.
        anomaly = _implausible_mark_move(route, st.get("last_mark_price"), info.get("current"))
        if anomaly is not None:
            anomaly.update({"symbol": sym, "route": route,
                            "last_mark_bar": st.get("last_mark_bar"), "bar": str(bar)})
            out.anomalies[tkr] = anomaly
            st["mark_anomaly"] = anomaly
            logger.error(
                "build_mixed_plan: REFUSING to act on %s (%s) — mark moved %.1f%% in one bar "
                "(%.4f -> %.4f, ratio %.2fx %s). Held and left managed; no exit evaluated. "
                "Verify corporate actions for this name before trading it.",
                tkr, sym, anomaly["bar_move"] * 100, anomaly["prev_mark"], anomaly["new_mark"],
                anomaly["ratio"], anomaly["direction"],
            )
            if verbose:
                print(f"  !! {tkr:<6} {sym:<20} MARK ANOMALY {anomaly['prev_mark']} -> "
                      f"{anomaly['new_mark']} ({anomaly['bar_move']*100:+.1f}%) — held, not traded")
            out.new_managed[tkr] = st
            continue
        st.pop("mark_anomaly", None)
        in_tgt = tkr in targets
        st["runs_held"] = st.get("runs_held", 0) + 1
        st["bars_out"] = 0 if in_tgt else st.get("bars_out", 0) + 1
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
        # An option exit is only as good as the mark that triggered it. Get a
        # second opinion before trading on it, and only for names already headed
        # for the door — see _corroborate_option_exit_mark for the AAOI case.
        if action != "hold" and route == "option":
            corrected = _corroborate_option_exit_mark(client, sym, info.get("current"),
                                                      info.get("avg_entry"))
            if corrected is not None:
                new_gain, new_mark = corrected
                new_action, new_reason = exit_action(
                    new_gain, st["runs_held"], st["bars_out"], st.get("trimmed", False),
                    policy, peak_gain=st.get("peak_gain"))
                stale = {"symbol": sym, "broker_mark": float(info["current"]),
                         "quote_mid": new_mark, "broker_gain": gain,
                         "quote_gain": new_gain, "was": reason or action,
                         "now": new_reason or new_action, "bar": str(bar)}
                out.anomalies[tkr] = stale
                logger.warning(
                    "build_mixed_plan: STALE option mark on %s (%s) — broker %.4f vs live mid "
                    "%.4f (gain %+.1f%% vs %+.1f%%). Exit decision '%s' -> '%s'; trading on "
                    "the live quote.",
                    tkr, sym, stale["broker_mark"], new_mark,
                    (gain or 0) * 100, new_gain * 100, stale["was"], stale["now"],
                )
                # Persist the corroborated mark, not the stale one, so the next
                # run's anomaly check compares against a number we trusted.
                gain = new_gain
                st["last_mark_price"] = new_mark
                st["unrealized_gain"] = float(new_gain)
                st["peak_gain"] = max(st.get("peak_gain", new_gain), new_gain)
                action, reason = new_action, new_reason
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
                # Exit sizing re-reads `held` from the broker every run, so a
                # stale size here never mis-sizes an order. It does corrupt every
                # reader that values the open book from state instead of the
                # broker: on 2026-08-13 dealer FIG said 61 contracts against 31
                # actually held, overstating that one position by $6,360 in the
                # daily report. Keep the persisted size in step with what the
                # trim just sold.
                _apply_trim_to_size(st, route, q)
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
            premium = order.get("limit") or order.get("mid")
            contracts = contracts_for_notional(premium, policy.target_notional)
            out.contract_selection[t] = {
                "action": "option", "occ": occ, "ref_price": px,
                "delta": order.get("delta"), "mid": order.get("mid"), "limit": order.get("limit"),
                "strike": order.get("strike"), "expiry": order.get("expiry"),
                "open_interest": order.get("open_interest"), "volume": order.get("volume"),
                "spread": order.get("spread"), "dealer_gate": order.get("dealer_gate"),
                "contracts": contracts,
                "signal_audit": sa.get(t),
                # The validated two-sided mark select_option observed. The
                # governed path builds the option leg from it; without it an
                # entry cannot be priced and is refused.
                "quote": order.get("quote"),
            }
            out.plan.append((occ, "buy", contracts, "entry", "option"))
            out.limits[occ] = order["limit"]
            out.order_audits[occ] = build_option_order_audit(
                signal_audit=sa.get(t), option_symbol=occ, route="call_option", side="buy",
                qty=contracts, underlying_price=px, strike=order.get("strike"),
                premium=order.get("limit") or order.get("mid"), limit_price=order.get("limit"),
                mid_price=order.get("mid"), delta=order.get("delta"),
                dte=_option_dte(order.get("expiry"), bar), expiration=order.get("expiry"),
            )
            out.new_managed[t] = {"route": "option", "occ": occ, "contracts": contracts,
                                  "runs_held": 0, "bars_out": 0, "trimmed": False, "entry_bar": str(bar),
                                  "expiry": order.get("expiry"), "signal_audit": sa.get(t),
                                  "order_audit": out.order_audits[occ]}
            if verbose:
                # Never format a possibly-None field bare. `_select_atm_option`
                # deliberately falls back to the strike band when an expiry has
                # no usable greeks ("delta_pool or same_exp"), so a selected
                # contract legitimately arrives with delta=None — and on
                # 2026-08-10 `delta={...:.2f}` raised TypeError on exactly that,
                # aborting build_mixed_plan and taking the whole Dealer Ranker
                # run down AFTER two contracts had been chosen. Zero orders were
                # submitted for the session because of a diagnostic print.
                # A logging statement must never be able to fail a trading run.
                print(f"  + {t:<6} {occ:<20} x{contracts} exp={order.get('expiry')} "
                      f"delta={_fmt_num(order.get('delta'))} mid={_fmt_num(order.get('mid'))}  "
                      f"oi={order.get('open_interest')}")
        else:  # equity — not a good options candidate, trade shares
            if pos_info.get(t, {}).get("qty", 0) > 0:
                out.contract_selection[t] = {"action": "skip", "reason": "already_held_equity",
                                             "signal_audit": sa.get(t)}
                continue
            shares = shares_for_notional(px, policy.target_notional)
            # Record the contract that was REJECTED, not just the reason string.
            # Without occ/strike/expiry/delta there is no way to tell an
            # genuinely illiquid chain from a bad strike pick after the fact
            # (2026-07-28 CRWV: `illiquid_option(oi=85,vol=8)` was unexplainable
            # from the log alone). `order` is None when selection never got far
            # enough to name a contract (price floor, no contracts, no greeks).
            rejected = order or {}
            out.contract_selection[t] = {"action": "equity", "reason": reason, "ref_price": px,
                                         "shares": shares,
                                         "occ": rejected.get("occ"),
                                         "strike": rejected.get("strike"),
                                         "expiry": rejected.get("expiry"),
                                         "delta": rejected.get("delta"),
                                         "open_interest": rejected.get("open_interest"),
                                         "volume": rejected.get("volume"),
                                         "liquidity_source": rejected.get("liquidity_source"),
                                         "band_size": rejected.get("band_size"),
                                         "dealer_gate": rejected.get("dealer_gate"),
                                         "signal_audit": sa.get(t)}
            out.plan.append((t, "buy", shares, "entry", "equity"))
            out.order_audits[t] = build_equity_order_audit(
                signal_audit=sa.get(t), symbol=t, side="buy", qty=shares,
                reason="entry", reference_price=px,
            )
            out.new_managed[t] = {"route": "equity", "symbol": t, "shares": shares,
                                  "runs_held": 0, "bars_out": 0, "trimmed": False, "entry_bar": str(bar),
                                  "signal_audit": sa.get(t), "order_audit": out.order_audits[t]}
            if verbose:
                print(f"  ~ {t:<6} {shares} shares  [{reason}]")

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
    persist_managed: Callable[[], None] | None = None,
    ledger_root: str | None = None,
    entry_ladder: bool = False,
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

    `persist_managed`, when supplied, is called right after each order fills so
    the on-disk managed-state file reflects the new position immediately rather
    than only after the whole (possibly multi-symbol) plan finishes. Without
    this, a sibling module's broker reconciliation can poll Alpaca directly in
    the gap between "order filled" and "plan-wide state save", see the freshly
    opened position with no owner on disk, and adopt + defensively liquidate it
    as an "unknown restored" position — this is exactly what happened to Dealer
    Ranker's IOT260724C00031500 buy on 2026-07-23, force-sold by Swing 2 minutes
    later for -$4,945 because Swing's reconcile ran before this module's
    end-of-plan `_save_state` had written the fill.
    """
    limits = limits or {}
    failed: set[str] = set()
    if not (submit and plan):
        return failed
    # Order matters. Defer BEFORE gating on readiness: an after-close entry is
    # not being submitted now, it is being *recorded* for the next open, and
    # submit_pending_open_entries re-runs the readiness gate at flush time. Doing
    # readiness first strips the buys (and pops them from new_managed) before
    # defer_entries_if_market_closed can see any reason=="entry" item, so the
    # intent is destroyed rather than queued — the whole 18:00-UTC signal stream
    # is silently discarded whenever the stamp happens to be stale at ~16:20 ET,
    # even if the overnight refresh fixes the data hours later. Observed
    # 2026-07-29: every after-close entry vanished with no pending-open queue
    # written. When the market is open, defer_entries_if_market_closed returns
    # the plan unchanged, so live-session behaviour is identical to before.
    plan = defer_entries_if_market_closed(module, bar, plan, new_managed, limits,
                                         ledger_root=ledger_root)
    # Exits are deferred separately and AFTER entries: the entry queue prunes
    # new_managed (nothing was placed), while a deferred exit must leave the
    # position tracked — it is still held, only the order waits. build_mixed_plan
    # has already removed it from new_managed, so exit_context is passed in to
    # put it back; see defer_exits_if_opg_unavailable for the VSH case this
    # prevents.
    plan = defer_exits_if_opg_unavailable(module, bar, plan, limits, ledger_root=ledger_root,
                                          new_managed=new_managed, exit_context=exit_context)
    plan, skipped, reason = filter_entry_orders_for_readiness(plan, new_managed=new_managed)
    if skipped:
        print(f"\nreadiness gate: skipped {len(skipped)} entry orders ({reason})")
        failed.update(skipped)
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
            if route == "option" and str(side).strip().lower() == "sell" and not lim:
                # Exits must actually get out. A bare market sell is rejected
                # when the contract has no quote, which stranded IOT's -50% stop
                # on 2026-07-24 until the contract expired.
                resp = submit_option_exit_with_ladder(client, symbol=sym, qty=qty)
            elif route == "option" and str(side).strip().lower() == "buy" and lim and entry_ladder:
                # `lim` is the ask (see meta_ranker.options_exec.route_option_or_shares
                # and dealer_positioning._select_atm_option), so the historical
                # path crossed the whole spread in one shot. Opt-in per module:
                # Dealer Ranker's contracts are ~3x wider than the other three,
                # so it is the only one where the crossing cost is large enough
                # to be worth a missed-entry risk.
                resp = submit_option_entry_with_ladder(client, symbol=sym, qty=qty, ask=lim)
            elif route == "option":
                resp = client.submit_option_order(symbol=sym, qty=qty, side=side,
                                                  order_type="limit" if lim else "market",
                                                  time_in_force="day", limit_price=lim)
            else:
                resp = client.submit_order(symbol=sym, qty=qty, side=side,
                                           order_type="market", time_in_force=equity_tif_fn())
            print(f"  OK {side} {qty} {sym}  id={resp.get('id', '?')}")
            if str(side).strip().lower() == "buy":
                mark_entry_unconfirmed(new_managed, sym, resp)
            if module and str(side).strip().lower() == "sell":
                es = exit_context.get(sym, (None, None))[1] if exit_context else None
                # An ACCEPTED sell is not a closed position. The ladder's last
                # rung is $0.01, which is exactly the price an expiring contract
                # with an empty book will never trade at, so the order rests
                # unfilled until it expires. Booking it as a close anyway is how
                # HPE260814C00060000 and ZM260814C00110000 left the dealer's
                # managed state on 2026-08-14 with pnl=None, became unowned
                # broker positions, and were adopted by the swing module — which
                # then failed to sell them (403 uncovered, then 422 expired).
                # 43 of 62 rows in dealer_ranker/closed_trades.jsonl were written
                # this way. Confirm the fill before booking anything.
                exit_fill = poll_exit_fill_price(client, resp)
                is_full_exit = bool(exit_context and sym in exit_context)
                if exit_fill is None and is_full_exit:
                    mark_exit_unconfirmed(new_managed, sym, resp, item=item,
                                          exit_context=exit_context,
                                          pos_lookup=pos_lookup, bar=bar)
                elif exit_fill is None:
                    # A TRIM: the position is legitimately still held either way,
                    # so there is nothing to keep owned and nothing stuck. Just
                    # don't book a partial close that may not have happened.
                    logger.warning(
                        "trim accepted but UNFILLED for %s (qty=%s) — not booked; "
                        "the next broker read reconciles the size", sym, qty)
                else:
                    record_exit_realized_pnl(client, module=module, item=item, resp=resp,
                                             entry_state=es, pos_lookup=pos_lookup, bar=bar,
                                             ledger_root=ledger_root, exit_fill=exit_fill)
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
        finally:
            if persist_managed is not None:
                try:
                    persist_managed()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("execute_plan: persist_managed callback failed after %s %s: %s", side, sym, exc)
    return failed


_EXIT_TICK = 0.01
_EXIT_LADDER_PAUSE_S = 1.5


def _option_quote(client, symbol: str) -> tuple[float | None, float | None]:
    """(bid, mid) for one contract. Either is None when the book can't price it."""
    if not hasattr(client, "get_option_quotes"):
        return None, None
    try:
        resp = client.get_option_quotes(symbols=symbol)
    except Exception:  # noqa: BLE001
        return None, None
    quotes = resp.get("quotes", resp) if isinstance(resp, dict) else None
    if not isinstance(quotes, dict):
        return None, None
    quote = quotes.get(symbol) or (next(iter(quotes.values())) if quotes else None)
    if not isinstance(quote, dict):
        return None, None

    def _num(*keys):
        for key in keys:
            try:
                val = float(quote.get(key))
            except (TypeError, ValueError):
                continue
            if val > 0:
                return val
        return None

    bid = _num("bp", "bid_price", "bid")
    ask = _num("ap", "ask_price", "ask")
    mid = (bid + ask) / 2.0 if (bid and ask) else _num("mark_price", "mark", "lp", "last_price")
    return bid, mid


def _option_bid(client, symbol: str) -> float | None:
    """Current bid for one contract, or None when the book is empty."""
    return _option_quote(client, symbol)[0]


def _exit_limit_ladder(bid: float | None, mid: float | None = None) -> list[float]:
    """Descending sell limits, from the mid down to the bid.

    A market exit is rejected outright when the contract has no quote (Alpaca:
    "order has been rejected due to no available quote for symbol, please
    reenter with a limit"). That is exactly the state an expiring contract is in,
    so the exit has to be priced.

    Anchoring on the MID rather than the bid: the old ladder started at the bid
    and walked to 0.4x the bid, so a collapsed bid priced the whole exit. On
    2026-08-05 the swing module's equivalent bid-anchored ladder sold 159 VALE
    calls at $0.01 into a 0.01 x 0.74 market, realizing $159 on a position the
    broker marked at $3,021. The bid is the floor here, not the starting price.
    One cent stays reachable only when there is no bid at all — a contract with
    an empty book still has to be closable.
    """
    has_bid = bid is not None and bid > _EXIT_TICK
    has_mid = mid is not None and mid > 0 and (not has_bid or mid >= bid)
    top = mid if has_mid else bid
    if top is None or top <= _EXIT_TICK:
        return [_EXIT_TICK]
    if has_bid:
        # A sell limit below the bid still fills AT the bid, so the last rung is
        # marketable rather than a giveaway. When the mid is unknown the bid is
        # all we have: walk a bounded 20% under it to stay marketable, never the
        # old 0.4x collapse.
        floor = bid if has_mid else bid * 0.8
    else:
        floor = _EXIT_TICK
    rungs = [top, top - (top - floor) * 0.5, floor]
    if not has_bid:
        rungs.append(_EXIT_TICK)
    out: list[float] = []
    for rung in rungs:
        rung = max(round(float(rung), 2), _EXIT_TICK)
        if rung not in out:
            out.append(rung)
    return out


def submit_option_exit_with_ladder(client, *, symbol: str, qty, sleep_fn=None,
                                   submit_fn=None, reason: str = "exit",
                                   full_exit: bool = True):
    """Sell-to-close one option, falling back to a priced ladder.

    Tries the plain market exit first (fills best when there IS a book), then
    walks a descending limit ladder with a short pause between rungs. Raises the
    last error only if every rung fails, so a genuinely stuck position still
    surfaces instead of being silently dropped.

    ``submit_fn`` replaces the direct broker call for callers that route through
    the governed path; the ladder shape, quantities, and reasons are unchanged
    either way. Callers that do not pass one keep the existing direct
    behaviour. ``reason``/``full_exit`` are carried through so the governed path
    records what this sell actually was; they do not affect the direct path,
    which has no decision record to label.
    """
    import time as _time

    sleep_fn = sleep_fn or _time.sleep
    if submit_fn is not None:
        # The governed path owns its own laddering and market fallback, so the
        # legacy retry loop below would duplicate orders.
        return submit_fn(symbol=symbol, side="sell", qty=qty, route="option", limit=None,
                         reason=reason, full_exit=full_exit)
    try:
        return client.submit_option_order(
            symbol=symbol, qty=qty, side="sell",
            order_type="market", time_in_force="day")
    except Exception as market_exc:  # noqa: BLE001
        logger.warning("exit ladder: market sell rejected for %s (%s) — repricing as limit",
                       symbol, market_exc)
        last_exc: Exception = market_exc

    ladder = _exit_limit_ladder(*_option_quote(client, symbol))
    for attempt, limit_price in enumerate(ladder, start=1):
        try:
            resp = client.submit_option_order(
                symbol=symbol, qty=qty, side="sell",
                order_type="limit", time_in_force="day", limit_price=limit_price)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("exit ladder: %s limit %.2f rejected (%d/%d): %s",
                           symbol, limit_price, attempt, len(ladder), exc)
            if attempt < len(ladder):
                sleep_fn(_EXIT_LADDER_PAUSE_S)
            continue
        logger.info("exit ladder: %s accepted at limit %.2f (%d/%d)",
                    symbol, limit_price, attempt, len(ladder))
        return resp
    raise last_exc


_ENTRY_LADDER_ATTEMPTS = 3
_ENTRY_LADDER_PAUSE_S = 2.0


def _entry_limit_ladder(mid: float | None, ask: float | None,
                        attempts: int = _ENTRY_LADDER_ATTEMPTS) -> list[float]:
    """Ascending buy limits, from the mid up to the ask.

    Mirror of ``_exit_limit_ladder``. Entries have always been submitted as a
    single limit AT the ask, which crosses the whole spread on every fill: across
    22 filled 4H option entries (2026-08-03..14) the mean fill was +7.3% over
    mid, and for Dealer Ranker — whose contracts run 3x wider — +12.7% against a
    take-profit target of only +20%.

    The ask is still the last rung, because the mid is not a fillable price on
    these contracts: of 24 multi_ticker_swing option entries over 2026-08-12..14,
    which already walk this ladder, ZERO filled at the mid rung, 12 filled at the
    ask rung and 10 never filled at all. Starting at the mid buys roughly half
    the crossing cost when it works and costs a missed entry when it does not.
    """
    if not ask or ask <= 0:
        return []
    if not mid or mid <= 0 or mid > ask:
        mid = ask
    count = max(1, int(attempts))
    if count == 1 or math.isclose(mid, ask, rel_tol=0.0, abs_tol=1e-9):
        return [round(ask, 2)]
    step = (ask - mid) / (count - 1)
    out: list[float] = []
    for idx in range(count):
        rung = round(mid + step * idx, 2)
        if rung > 0 and rung not in out:
            out.append(rung)
    ask_rung = round(ask, 2)
    if ask_rung not in out:
        out.append(ask_rung)
    return out


def submit_option_entry_with_ladder(client, *, symbol: str, qty, ask: float | None,
                                    attempts: int = _ENTRY_LADDER_ATTEMPTS,
                                    sleep_fn=None, poll_fn=None):
    """Buy-to-open one option, walking a limit ladder from the mid up to the ask.

    Each unfilled rung is CANCELLED before the next is submitted — two live buy
    orders for the same contract would double the position. A rung that fills
    returns immediately. If no rung fills the last response is still returned, so
    the caller's `mark_entry_unconfirmed` path treats it exactly as it treats any
    other unconfirmed entry today (flagged `pending_fill`, dropped `not_found` on
    the next run) rather than inventing a new failure mode.
    """
    import time as _time

    sleep_fn = sleep_fn or _time.sleep
    poll_fn = poll_fn or (lambda resp: _poll_order_filled(client, resp))
    bid, mid = _option_quote(client, symbol)
    ladder = _entry_limit_ladder(mid, ask, attempts)
    if not ladder:
        # No usable quote — fall back to the historical behaviour rather than
        # refusing to trade, so a quote outage cannot silently halt entries.
        return client.submit_option_order(symbol=symbol, qty=qty, side="buy",
                                          order_type="limit", time_in_force="day",
                                          limit_price=ask)
    resp = None
    for attempt, limit_price in enumerate(ladder, start=1):
        resp = client.submit_option_order(
            symbol=symbol, qty=qty, side="buy",
            order_type="limit", time_in_force="day", limit_price=limit_price)
        if poll_fn(resp):
            logger.info("entry ladder: %s FILLED at limit %.2f (%d/%d, mid=%s ask=%s)",
                        symbol, limit_price, attempt, len(ladder),
                        _fmt_num(mid), _fmt_num(ask))
            return resp
        if attempt < len(ladder):
            _cancel_order_quietly(client, resp)
            sleep_fn(_ENTRY_LADDER_PAUSE_S)
    logger.info("entry ladder: %s unfilled after %d rungs (mid=%s ask=%s) — left "
                "resting at the ask, entry stays unconfirmed",
                symbol, len(ladder), _fmt_num(mid), _fmt_num(ask))
    return resp


def _poll_order_filled(client, resp, *, timeout_s: float = 3.0, poll_s: float = 0.4) -> bool:
    """True once the broker reports the order filled. Never blocks past timeout."""
    import time as _time

    oid = (resp or {}).get("id") if isinstance(resp, dict) else None
    if not oid:
        return False
    if str((resp or {}).get("status", "")).lower() == "filled":
        return True
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        try:
            order = client.get_order(str(oid))
        except Exception:  # noqa: BLE001 — a poll failure is not a fill
            return False
        if str((order or {}).get("status", "")).lower() == "filled":
            return True
        _time.sleep(poll_s)
    return False


def _cancel_order_quietly(client, resp) -> None:
    """Cancel an unfilled ladder rung. A failure here is logged, never raised."""
    oid = (resp or {}).get("id") if isinstance(resp, dict) else None
    if not oid:
        return
    try:
        client.cancel_order(str(oid))
    except Exception as exc:  # noqa: BLE001
        logger.warning("entry ladder: cancel failed for order %s: %s", oid, exc)


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


_STOP_REASON_RE = re.compile(r"^stop_-(\d+)%$")


def _threshold_from_exit_reason(reason) -> float | None:
    """The stop fraction an exit reason names, e.g. "stop_-39%" -> 0.39.

    Returns None for every non-stop reason, so `stop_overshoot` stays null on
    take-profits, trails and horizon exits rather than being computed against a
    threshold that was never the trigger.
    """
    m = _STOP_REASON_RE.match(str(reason or ""))
    return int(m.group(1)) / 100.0 if m else None


def record_exit_realized_pnl(client, *, module, item, resp, entry_state, pos_lookup, bar,
                             ledger_root: str | None = None,
                             exit_fill: float | None = None) -> None:
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
        if exit_fill is None:
            exit_fill = poll_exit_fill_price(client, resp)
        avg_entry = (pos_lookup or {}).get(sym, {}).get("avg_entry")
        realized = None
        if exit_fill is not None and avg_entry:
            realized = round((exit_fill - float(avg_entry)) * mult * float(qty), 2)
        es = entry_state or {}
        # Underlying ticker: prefer explicit state, else strip an OCC symbol
        # (e.g. SOC260717C00005000 -> SOC) so option rows aren't keyed by the OCC.
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
        # A stop is only TESTED when a runner fires (14:20/16:20 ET and the
        # pre-open flush), so between tests a position can travel arbitrarily far
        # past the threshold. `exit_reason` names the policy ("stop_-39%") and
        # says nothing about where the position actually was, which made the two
        # causes of a bad exit inseparable: gapping past the level before we
        # looked (decision_gain already far below -39%) versus filling badly once
        # we did (fill_gain much worse than decision_gain). Recording both makes
        # that decomposition measurable — the prerequisite for choosing between a
        # tighter threshold, a faster cadence, and a resting protective order.
        # 2026-08-06/07: 9 option stops labelled -39% realized a mean of -57.1%.
        decision_gain = es.get("unrealized_gain")
        fill_gain = (exit_fill / float(avg_entry) - 1.0) if (exit_fill is not None and avg_entry) else None
        threshold = _threshold_from_exit_reason(reason)
        rec["decision_gain"] = float(decision_gain) if decision_gain is not None else None
        rec["fill_gain"] = round(fill_gain, 6) if fill_gain is not None else None
        rec["stop_overshoot"] = (
            round(fill_gain + threshold, 6)
            if fill_gain is not None and threshold is not None else None
        )
        out = Path(ledger_root or DEFAULT_LEDGER_ROOT) / str(module) / "closed_trades.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        logger.info("realized-PnL logged: %s %s qty=%s reason=%s pnl=%s",
                    module, sym, qty, reason, realized)
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_exit_realized_pnl failed (%s %s): %s", module, item, exc)


def mark_entry_unconfirmed(new_managed: dict | None, sym: str, resp) -> None:
    """Record that an ENTRY order was ACCEPTED, not that a position now exists.

    build_mixed_plan writes a new position into managed state when it plans the
    entry, and execute_plan only removes it if the submission raises. An order
    the broker accepts but never fills therefore persists as a position the
    account does not hold. Dealer Ranker showed this on 2026-08-11: it submitted
    nine limit entries at 15:53 ET, seven minutes before the close; four (UMC
    19C, P 110C, GRAB 3.5C, TSCO 36C) never filled, and `live_state.json` claimed
    11 managed positions against the broker's 7.

    The fix is a flag rather than a fill poll on purpose. Dropping an unfilled
    entry is not safe — a limit that fills later would become an unowned
    position, which is how Swing force-sold Dealer Ranker's IOT260724C00031500
    for -$4,945 on 2026-07-23 (see execute_plan). So the entry stays claimed
    (conservative for sibling reconciliation) and is simply labelled unconfirmed
    until a broker read settles it: build_mixed_plan already drops anything with
    qty <= 0 on the next pass, so the flag clears itself either way.
    """
    tkr = _managed_key_for_symbol(new_managed, sym)
    if tkr is None:
        return
    st = new_managed.get(tkr)
    if isinstance(st, dict):
        st["pending_fill"] = True
        st["entry_order_id"] = str((resp or {}).get("id", "")) or None


def mark_exit_unconfirmed(new_managed: dict | None, sym: str, resp, *, item, exit_context,
                          pos_lookup, bar) -> None:
    """Record that an EXIT order was ACCEPTED but has not filled.

    The position is restored to managed state — still ours, still risk-managed —
    rather than being booked as closed. Two things follow from that:

    * No `closed_trades.jsonl` row is written yet, so the realized ledger never
      carries a `realized_pnl=None` phantom close.
    * The position stays claimed, so a sibling module doing broker
      reconciliation cannot adopt it as an unowned position and fire its own
      exit at it. That cross-module fight is what stranded ZM on 2026-08-14.

    The entry basis is snapshotted here because it has to be: once the contract
    settles, the position is gone from the broker and `pos_lookup` can no longer
    answer what it cost. `resolve_settled_exit` reads it back to book the close.
    """
    tkr = _managed_key_for_symbol(new_managed, sym)
    if tkr is None and exit_context and sym in exit_context:
        # Full exits are removed from new_managed when the plan is built; put the
        # pre-exit state back so the position stays owned.
        tkr, st = exit_context[sym]
        if new_managed is not None:
            new_managed[tkr] = st
    if tkr is None or new_managed is None:
        return
    st = new_managed.get(tkr)
    if not isinstance(st, dict):
        return
    basis = (pos_lookup or {}).get(sym, {}).get("avg_entry")
    if basis is None:
        basis = st.get("entry_avg_price")
    st["exit_pending"] = {
        "order_id": str((resp or {}).get("id", "")) or None,
        "reason": item[3] if len(item) > 3 else "exit",
        "route": item[4] if len(item) > 4 else "option",
        "qty": float(item[2]),
        "entry_avg_price": float(basis) if basis else None,
        "submitted_bar": str(bar),
        "submitted_ts": now_utc_iso(),
    }
    logger.warning(
        "exit accepted but UNFILLED for %s (%s) — order resting, position kept in managed "
        "state and NOT booked as closed; will be resolved on the next run",
        tkr, sym,
    )


def resolve_settled_exit(client, *, module, ticker, symbol, state, bar,
                         ledger_root: str | None = None) -> bool:
    """Book a close for a position whose resting exit order has now settled.

    Called when the broker no longer reports the position. The resting order is
    the authority on what happened:

    * `filled` -> book the real fill price.
    * anything else (expired/canceled) with the position gone -> the contract
      expired. An option that expires unexercised is a total loss of premium, so
      realized is -basis. That is the number the ledger was missing.

    Returns True when a row was written, so the caller can clear the flag.
    """
    pending = (state or {}).get("exit_pending")
    if not isinstance(pending, dict):
        return False
    try:
        import json
        from pathlib import Path
        order_id = pending.get("order_id")
        basis = pending.get("entry_avg_price")
        qty = float(pending.get("qty") or 0)
        route = pending.get("route", "option")
        mult = 100.0 if route == "option" else 1.0
        status, fill = "unknown", None
        if order_id and hasattr(client, "get_order"):
            try:
                cur = client.get_order(order_id) or {}
                status = str(cur.get("status", "")).strip().lower() or "unknown"
                fill = _resp_fill_price(cur)
            except Exception:  # noqa: BLE001
                pass
        if fill is not None and basis:
            realized = round((float(fill) - float(basis)) * mult * qty, 2)
            outcome = "exit_filled"
        elif basis:
            # Gone from the broker with no fill: the premium is gone.
            realized, outcome = round(-float(basis) * mult * qty, 2), "expired_worthless"
        else:
            realized, outcome = None, "settled_basis_unknown"
        rec = {
            "ts": now_utc_iso(),
            "module": module,
            "bar": str(bar),
            "ticker": ticker,
            "order_symbol": symbol,
            "route": route,
            "side": "sell",
            "qty": qty,
            "exit_reason": pending.get("reason", "exit"),
            "entry_avg_price": float(basis) if basis else None,
            "exit_fill_price": float(fill) if fill is not None else None,
            "realized_pnl": realized,
            "entry_bar": (state or {}).get("entry_bar"),
            "runs_held": (state or {}).get("runs_held"),
            "order_id": order_id,
            "settle_outcome": outcome,
            "exit_order_status": status,
            "exit_submitted_bar": pending.get("submitted_bar"),
        }
        out = Path(ledger_root or DEFAULT_LEDGER_ROOT) / str(module) / "closed_trades.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        logger.info("pending exit settled: %s %s qty=%s outcome=%s pnl=%s",
                    module, symbol, qty, outcome, realized)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("resolve_settled_exit failed (%s %s): %s", module, symbol, exc)
        return False


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

def pending_open_path(module: str, ledger_root: str | None = None):
    from pathlib import Path
    return Path(ledger_root or DEFAULT_LEDGER_ROOT) / str(module) / "pending_open_entries.json"


def pending_exit_path(module: str, ledger_root: str | None = None):
    from pathlib import Path
    return Path(ledger_root or DEFAULT_LEDGER_ROOT) / str(module) / "pending_exit_orders.json"


# Alpaca only ACCEPTS an 'opg' order between 19:00 and 09:28 ET. equity_order_tif()
# returns 'opg' whenever the market is closed, which covers 16:00-19:00 ET too —
# and the 4H runners fire at ~16:20-16:35 ET, squarely inside that dead zone. So
# every after-close EQUITY exit is submitted with a TIF the broker refuses:
#   HTTP 403 {"code":40310000,"message":"opg orders must be submitted after
#             7:00pm and before 9:28am"}
# Observed 2026-08-03 on the Meta CRWV take_profit_+30% sell. Entries already
# survive this via the pending-open queue; exits had no equivalent path, so they
# just failed and were restored to managed state to fail again the next run.
_OPG_WINDOW_OPEN_HOUR, _OPG_WINDOW_OPEN_MINUTE = 19, 0
_OPG_WINDOW_CLOSE_HOUR, _OPG_WINDOW_CLOSE_MINUTE = 9, 28


def opg_window_is_open(now=None) -> bool:
    """True when the broker will accept an 'opg' order (19:00-09:28 ET)."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(et)
    minutes = local.hour * 60 + local.minute
    return (minutes >= _OPG_WINDOW_OPEN_HOUR * 60 + _OPG_WINDOW_OPEN_MINUTE
            or minutes < _OPG_WINDOW_CLOSE_HOUR * 60 + _OPG_WINDOW_CLOSE_MINUTE)


def defer_exits_if_opg_unavailable(module, bar, plan, limits, *, now=None,
                                   ledger_root: str | None = None,
                                   new_managed: dict | None = None,
                                   exit_context: dict | None = None) -> list:
    """Queue exits the broker would reject after the close, instead of failing them.

    Two different broker restrictions, so two different conditions:

    * EQUITY sells use TIF 'opg' once the market is closed, and Alpaca only
      accepts an OPG order between 19:00 and 09:28 ET. So they are deferred only
      in the 16:00-19:00 dead zone; inside the OPG window the order is accepted
      and queues at the broker for the open, which is strictly better.
    * OPTION sells are market orders on an instrument that only trades in RTH:
      ``422 options market orders are only allowed during market hours``. There
      is no after-hours window that works, so they are deferred whenever the
      market is closed. Observed 2026-08-05 on the HTF CLSK260821C00015000
      stop_-39%, which failed at 16:25 and left the position unstopped overnight.

    Unlike the entry path this never DROPS anything from managed state — the
    position is still held, so it must stay tracked; only the order is postponed.
    It does the opposite: when `new_managed`/`exit_context` are supplied it
    RESTORES the position, because build_mixed_plan already removed it.

    That removal is the subtle part. build_mixed_plan drops a position from
    new_managed the moment it plans a full exit for it, and execute_plan only
    puts it back if the submission raises. A deferred exit is pulled from the
    plan before submission, so neither path runs and the position ends up held
    at the broker but claimed by nobody. Swing's broker reconciliation reads
    siblings' `managed` sets (position_manager._sibling_module_owned_symbols) to
    decide what it may adopt, so an unclaimed position is an adoptable one: on
    2026-08-11 Swing restored HTF's VSH (19x VSH260821C00035000) as its own at
    09:06:28 ET, and HTF's own deferred exit sold those same contracts at
    09:37:04. It self-healed 31 seconds later via broker_position_missing, but
    for half an hour two modules both believed they owned the position, and a
    stop firing in that window would have tried to sell contracts already spoken
    for. See research/daily_live_reports/2026-08-11.md.
    """
    if not module:
        return plan
    try:
        from core.calendar import is_market_open_now
        market_open = is_market_open_now(now)
        if market_open:
            return plan
        opg_open = opg_window_is_open(now)
    except Exception:
        return plan  # fail safe: if we can't tell, behave as before (submit)
    import json
    kept, deferred = [], []
    for item in plan:
        sym = item[0]
        side = str(item[1]).strip().lower()
        reason = item[3] if len(item) > 3 else ""
        route = item[4] if len(item) > 4 else "option"
        is_option = route == "option"
        rejectable = side == "sell" and (is_option or not opg_open)
        if rejectable:
            deferred.append({
                "order_symbol": sym, "side": item[1], "qty": item[2], "route": route,
                "limit": (limits or {}).get(sym), "reason": reason, "bar": str(bar),
                # Whether this sell closes the position or only trims it. Recorded
                # here because exit_context membership is the structural signal and
                # it is only available now — by flush time the plan is gone, and
                # pattern-matching the reason string would be guesswork. A governed
                # submitter needs it to record an EXIT rather than an ADJUSTMENT.
                "full_exit": bool(exit_context and sym in exit_context),
            })
            # Keep claiming it until the exit actually fills.
            if new_managed is not None and exit_context and sym in exit_context:
                tkr, st = exit_context[sym]
                if tkr not in new_managed:
                    new_managed[tkr] = st
                    logger.info(
                        "%s: exit for %s (%s) deferred — restoring to managed state so the "
                        "position stays claimed until the pre-open flush", module, tkr, sym)
        else:
            kept.append(item)
    if deferred:
        out = pending_exit_path(module, ledger_root)
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
        n_opt = sum(1 for e in deferred if e["route"] == "option")
        logger.info("%s: market closed — deferred %d exits to next open (%d option, %d equity) -> %s",
                    module, len(deferred), n_opt, len(deferred) - n_opt, out)
        print(f"\n{module}: deferred {len(deferred)} exit(s) to next open "
              f"({n_opt} option, {len(deferred) - n_opt} equity)")
    return kept


def submit_pending_exit_orders(client, module, *, equity_tif_fn, pos_lookup=None,
                               ledger_root: str | None = None,
                               managed: dict | None = None,
                               submit_fn=None) -> dict:
    """Pre-open flush of queued exits. Skips anything no longer held. Clears the queue.

    Deliberately does NOT re-rank: an exit decision was already made on its own
    bar (take-profit, horizon, dropped_out) and is not conditional on today's
    top-K the way a deferred entry is.

    Every filled exit is written to the realized-PnL ledger, exactly as
    execute_plan does for a same-run exit. This path was missing that call
    entirely: on 2026-08-11 the AMAT260821C00550000 and VSH260821C00035000
    stops both flushed here with broker order ids and neither produced a
    closed_trades.jsonl row, so the day's realized P&L was understated and the
    fills were recorded nowhere. That gap fell precisely on the population the
    stop-overshoot fields exist to measure — a deferred exit sits unpriced from
    the 16:20 ET decision to the 09:35 ET flush, which is the longest a stop can
    travel past its threshold. See research/daily_live_reports/2026-08-11.md.

    `managed` (the module's own managed state) supplies entry_bar / runs_held /
    decision_gain for those rows; without it the row is still written, just with
    those provenance fields null.

    ``submit_fn`` replaces the direct broker call for callers that route through
    the governed path, matching submit_pending_open_entries. A queued exit is
    still an order, so a module that has cut over must not reach the broker
    directly here just because the decision was made on an earlier bar.
    """
    import json
    out = pending_exit_path(module, ledger_root)
    if not out.exists():
        return {"submitted": [], "skipped": [], "count": 0}
    try:
        entries = json.loads(out.read_text()).get("entries", [])
    except Exception:
        entries = []
    pos_lookup = pos_lookup or {}
    submitted, skipped = [], []
    for rec in entries:
        sym = rec.get("order_symbol")
        held = pos_lookup.get(sym, {}).get("qty", 0)
        if held <= 0:
            skipped.append({**rec, "skip": "not_held"})
            continue
        qty = min(int(rec.get("qty") or 0), int(held))
        if qty <= 0:
            skipped.append({**rec, "skip": "zero_qty"})
            continue
        try:
            # Legacy queue entries predate `full_exit`; treat them as full exits.
            # Both readings are only a label on the decision record, and a stop or
            # horizon exit is far and away the common deferred sell.
            full_exit = bool(rec.get("full_exit", True))
            reason = rec.get("reason") or "exit"
            if rec.get("route") == "option":
                # Options are day-only: this is the first moment since the exit
                # decision that a two-sided book exists. Price it off the mid via
                # the shared ladder rather than firing a bare market order.
                resp = submit_option_exit_with_ladder(client, symbol=sym, qty=qty,
                                                      submit_fn=submit_fn,
                                                      reason=reason, full_exit=full_exit)
            elif submit_fn is not None:
                resp = submit_fn(symbol=sym, side=rec.get("side", "sell"), qty=qty,
                                 route="equity", limit=None,
                                 reason=reason, full_exit=full_exit)
            else:
                resp = client.submit_order(symbol=sym, qty=qty, side=rec.get("side", "sell"),
                                           order_type="market", time_in_force=equity_tif_fn())
            submitted.append({**rec, "qty": qty, "order_id": resp.get("id", "")})
            print(f"  OK deferred-exit sell {qty} {sym}  id={resp.get('id', '?')}")
            tkr = _managed_key_for_symbol(managed, sym)
            record_exit_realized_pnl(
                client, module=module,
                item=(sym, rec.get("side", "sell"), qty, rec.get("reason", "exit"),
                      rec.get("route", "option")),
                resp=resp, entry_state=(managed or {}).get(tkr) if tkr else None,
                pos_lookup=pos_lookup, bar=rec.get("bar"), ledger_root=ledger_root,
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({**rec, "skip": f"submit_failed: {exc}"})
            print(f"  FAIL deferred-exit sell {qty} {sym}: {exc}")
    out.write_text(json.dumps({"updated": now_utc_iso(), "entries": [],
                               "last_flush": {"submitted": submitted, "skipped": skipped}},
                              default=str, indent=1))
    return {"submitted": submitted, "skipped": skipped, "count": len(submitted)}


def _managed_key_for_symbol(new_managed: dict | None, sym: str) -> str | None:
    if not isinstance(new_managed, dict) or not new_managed:
        return None
    if sym in new_managed:
        return sym  # equity keyed by ticker
    for tkr, st in new_managed.items():
        if isinstance(st, dict) and sym in (st.get("occ"), st.get("symbol")):
            return tkr  # option keyed by ticker, OCC under 'occ'/'symbol'
    return None


def defer_entries_if_market_closed(module, bar, plan, new_managed, limits, *,
                                   now=None, ledger_root: str | None = None) -> list:
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
    except Exception:  # noqa: BLE001
        # Fail closed for entries. "I could not tell whether the market is
        # open" is not "the market is open"; treating it as open fires
        # after-hours entries that will be rejected, and does so silently.
        # Exits keep going below -- an unreadable calendar must never trap a
        # position.
        logger.warning(
            "%s: market calendar unavailable — deferring entries, exits still go",
            module,
        )
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
                                pos_lookup=None, ledger_root: str | None = None,
                                submit_fn=None) -> dict:
    """Pre-open flush: submit queued after-close entries that are STILL in the current
    top-K (re-rank) and not already held, as normal day orders. Clears the queue.
    Returns {"submitted": {ticker: managed_dict}, "skipped": [...], "count": n}; never
    raises on an individual order (records it as skipped instead).

    ``submit_fn`` replaces the direct broker call for callers that route through
    the governed path. It receives ``(symbol, side, qty, route, limit)`` and
    returns the broker response. Callers that do not pass one keep the existing
    direct behaviour unchanged, so modules still on the legacy path are not
    affected by another module's cutover.
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
    # These records carry their ticker explicitly, so the per-ticker gate does not
    # have to infer it from an OCC root.
    _kept, readiness_skipped, readiness_reason = filter_entry_orders_for_readiness(
        readiness_plan,
        symbol_tickers={
            str(rec.get("order_symbol")): str(rec.get("ticker"))
            for rec in eligible
            if rec.get("order_symbol") and rec.get("ticker")
        },
    )
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
            if submit_fn is not None:
                resp = submit_fn(symbol=sym, side=rec["side"], qty=rec["qty"],
                                 route=route, limit=lim)
            elif route == "option":
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

    # A retained entry is never silently deleted: the queue is the only record
    # that a decision is still waiting. Only entries that were submitted, or
    # that are genuinely finished (the rank they depended on is gone, or the
    # position is already held), leave the queue.
    terminal_skips = ("no_longer_top_k", "already_held")
    retained = [
        {key: value for key, value in rec.items() if key != "skip"}
        for rec in skipped
        if not str(rec.get("skip", "")).startswith(terminal_skips)
    ]
    try:
        if retained:
            out.write_text(
                json.dumps(
                    {"updated": now_utc_iso(), "entries": retained},
                    default=str,
                    indent=1,
                )
            )
        else:
            out.unlink()
    except Exception:  # noqa: BLE001 - the flush result still stands
        logger.warning("%s pending-open: could not rewrite the queue", module)
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
