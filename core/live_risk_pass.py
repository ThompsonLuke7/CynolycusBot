"""Between-bar risk management for the 4H modules.

The 4H modules (Meta Ranker, Momentum, HTF Swing, Dealer Ranker) decide once or
twice a day because that is the bar their models were trained on. Risk is not a
model output, so it does not have to wait for the next bar: whether a held
position has breached its stop, or whether a contract expires today, uses no
model and no training-time feature. Checking those between bars breaks no
research/live parity.

2026-08-14 is the worked example. Dealer Ranker bought HPE (8/12) and ZM (8/13),
both expiring 8/14, and runs once a day at 15:45 ET — so each contract got
exactly ONE stop test in its entire life, seven minutes before it expired, by
which point both were OTM with no bid. They expired worthless for -$9,594.

WHAT THIS PASS DELIBERATELY DOES NOT DO
---------------------------------------
* It never advances ``runs_held`` or ``bars_out``. Those count 4H BARS and feed
  the horizon exit; incrementing them every five minutes would run a position
  through ``policy.horizon_bars`` in a single afternoon and force spurious
  "horizon" exits. Only the 4H runner ages a position.
* It never evaluates ``horizon`` (a bar counter) or ``dropped_out`` (a ranking,
  i.e. model output). Those stay with the 4H runner.
* It does not open positions, and never looks at ``targets``.

CADENCE-SENSITIVE RULES ARE OPT-IN
----------------------------------
A hard stop is cadence-independent: "down 39% from entry" is the same fact
whenever you look, so checking more often is strictly better. A TRAILING stop is
not — it ratchets off the observed peak, so sampling 78 times a day instead of
twice finds higher peaks and therefore triggers earlier. ``trail_stop=0.35`` was
calibrated against the 4H sampling cadence, so enabling it here would silently
change a validated policy rather than enforce it. Same for the take-profit trim.
Both default OFF; turn them on only with a backtest at the new cadence.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from core.live_4h_exec import (
    ExecPolicy,
    _corroborate_option_exit_mark,
    _implausible_mark_move,
    build_equity_order_audit,
    build_option_order_audit,
    # Defined there, not here: live_4h_exec's pending-exit flush needs it too and
    # importing this module from that one would close a cycle. Re-exported so
    # existing callers and tests keep importing it from here.
    parse_occ_expiry,
    resolve_settled_exit,
    take_profit_reason,
    trim_quantity,
    underlying_basis,
    underlying_stop_level,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = REPO_ROOT / "Data/runtime/risk_pass"




@dataclass(frozen=True)
class RiskPassConfig:
    """Which non-model rules this pass is allowed to act on.

    Defaults are the two that are provably cadence-independent and strictly
    risk-reducing. Everything path-dependent is off until it is re-validated at
    this cadence.
    """

    hard_stop: bool = True
    expiry_flatten: bool = True
    settle_pending_exits: bool = True
    # Path-dependent — see the module docstring before enabling either.
    trailing_stop: bool = False
    take_profit_trim: bool = False
    # Flatten an option on its last tradable session once past this ET time.
    # 15:45 matches the swing sleeve's _EXPIRING_ITM_CLOSE_HOUR/MINUTE so the two
    # sleeves do not disagree about when an expiring contract must be gone.
    expiry_cutoff_et: tuple[int, int] = (15, 45)


@dataclass
class RiskPlan:
    """Sell orders this pass wants, plus why it left everything else alone."""

    plan: list[tuple[str, str, int, str, str]] = field(default_factory=list)
    order_audits: dict[str, dict] = field(default_factory=dict)
    exit_context: dict[str, tuple[str, dict]] = field(default_factory=dict)
    new_managed: dict[str, dict] = field(default_factory=dict)
    settled: dict[str, dict] = field(default_factory=dict)
    anomalies: dict[str, dict] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)




def past_expiry_cutoff(now_et: dt.datetime, cutoff: tuple[int, int]) -> bool:
    hour, minute = cutoff
    return (now_et.hour, now_et.minute) >= (hour, minute)


def expiring_before_next_session(symbol: str, now_et: dt.datetime,
                                 cfg: RiskPassConfig) -> bool:
    """True when this contract will not see another tradable session.

    Deliberately calendar-aware rather than "expires today": the Friday before a
    long weekend is the last chance to sell a Monday-expiring contract on screen.
    """
    expiry = parse_occ_expiry(symbol)
    if expiry is None:
        return False
    if not past_expiry_cutoff(now_et, cfg.expiry_cutoff_et):
        return False
    try:
        from core.calendar import next_trading_day
        return next_trading_day(now_et.date()) > expiry
    except Exception:  # noqa: BLE001 - an unreadable calendar must not trap a position
        return expiry <= now_et.date()


def risk_exit_action(gain: float | None, *, trimmed: bool, peak_gain: float | None,
                     policy: ExecPolicy, cfg: RiskPassConfig, route: str = "equity",
                     u_entry=None, u_now=None, u_atr=None) -> tuple[str, str]:
    """The non-model half of ``core.live_4h_exec.exit_action``.

    Same thresholds and the same reason strings, so a stop fired here is
    indistinguishable in the ledger from one the 4H runner fired. Horizon and
    rank drop-out are absent by design — both are bar/model driven.

    The hard stop MUST mirror ``exit_action``'s underlying-referenced form for
    options. This pass runs every few minutes against the 4H runner's twice a
    day, so it is the path that actually fires most stops: leaving a premium
    stop here would keep stopping options out on premium noise no matter what
    the 4H engine does. An underlying stop is just as cadence-independent as a
    premium one — "the underlying is below entry minus 1.5 ATR" is the same fact
    whenever you sample it — so checking it more often stays strictly better.

    Expiry is NOT evaluated here; this pass has its own calendar-driven
    ``expiring_before_next_session`` flatten, which is stricter than the 4H
    engine's ``min_dte_exit`` and already runs ahead of it.
    """
    if cfg.hard_stop:
        u_stop = (underlying_stop_level(policy, u_entry, u_atr)
                  if route == "option" else None)
        if u_stop is not None and u_now is not None:
            if float(u_now) <= u_stop:
                return "exit", f"underlying_stop_-{policy.underlying_stop_atr:g}atr"
        elif policy.stop_loss and gain is not None and gain <= -policy.stop_loss:
            return "exit", f"stop_-{int(policy.stop_loss * 100)}%"
    if (cfg.trailing_stop and policy.trail_stop and gain is not None
            and peak_gain is not None and peak_gain > 0
            and (1 + gain) <= (1 + peak_gain) * (1 - policy.trail_stop)):
        return "exit", f"trail_-{int(policy.trail_stop * 100)}%"
    if (cfg.take_profit_trim and not trimmed and gain is not None
            and gain >= policy.take_profit):
        return "trim", f"take_profit_+{int(policy.take_profit * 100)}%"
    return "hold", ""


def evaluate_risk_exits(
    client,
    *,
    module: str,
    managed: dict[str, dict],
    pos_info: dict[str, dict],
    policy: ExecPolicy,
    now_et: dt.datetime,
    cfg: RiskPassConfig | None = None,
    ledger_root: str | None = None,
    underlying_fn=None,
) -> RiskPlan:
    """Evaluate held positions for non-model exits. Mutates no bar counters."""
    cfg = cfg or RiskPassConfig()
    out = RiskPlan()
    ufn = underlying_fn or underlying_basis

    def _sell_audit(tkr, sym, qty, route):
        if route == "equity":
            return build_equity_order_audit(signal_audit=None, symbol=sym, side="sell", qty=qty)
        return build_option_order_audit(signal_audit=None, option_symbol=sym,
                                        route="call_option", side="sell", qty=qty)

    for tkr, st in managed.items():
        route = st.get("route", "option")
        sym = st.get("occ") if route == "option" else st.get("symbol", tkr)
        held = pos_info.get(sym, {}).get("qty", 0) if sym else 0

        if held <= 0:
            # Gone from the broker. The 4H runner owns dropping it from managed —
            # this pass only settles a resting exit so the loss reaches the
            # ledger promptly instead of waiting for the next bar.
            if cfg.settle_pending_exits and isinstance(st.get("exit_pending"), dict):
                if resolve_settled_exit(client, module=module, ticker=tkr, symbol=sym,
                                        state=st, bar=now_et.isoformat(),
                                        ledger_root=ledger_root):
                    out.settled[tkr] = {"symbol": sym, "route": route}
                    st.pop("exit_pending", None)
            out.new_managed[tkr] = st
            continue

        if isinstance(st.get("exit_pending"), dict):
            # An exit order is already resting against this position. Submitting
            # a second one would double-sell if the first fills.
            out.skipped[tkr] = "exit_order_already_resting"
            out.new_managed[tkr] = st
            continue

        info = pos_info.get(sym, {})
        anomaly = _implausible_mark_move(route, st.get("last_mark_price"), info.get("current"))
        if anomaly is not None:
            # Same refusal the 4H runner makes: a mark that moved 10x in one step
            # is a corporate action, not a price. Never stop out on it.
            anomaly.update({"symbol": sym, "route": route, "bar": now_et.isoformat()})
            out.anomalies[tkr] = anomaly
            out.skipped[tkr] = "mark_anomaly"
            out.new_managed[tkr] = st
            continue

        gain = (info["current"] / info["avg_entry"] - 1) if info.get("avg_entry") else None

        # Underlying basis for the option stop. Read-only here: the 4H runner
        # owns writing u_entry/u_atr into managed state (it is the only pass that
        # sees an entry). If it has not anchored this position yet, there is no
        # basis and the premium stop stands — this pass never invents one.
        u_entry = u_atr = u_now = None
        if route == "option" and policy.underlying_stop_atr:
            u_entry, u_atr = st.get("u_entry"), st.get("u_atr")
            if u_entry is not None and u_atr is not None:
                u_now, _ = ufn(tkr)

        expiring = (route == "option" and cfg.expiry_flatten
                    and expiring_before_next_session(str(sym), now_et, cfg))
        if expiring:
            action, reason = "exit", "expiring_before_closure"
        else:
            action, reason = risk_exit_action(
                gain, trimmed=bool(st.get("trimmed", False)),
                peak_gain=st.get("peak_gain"), policy=policy, cfg=cfg,
                route=route, u_entry=u_entry, u_now=u_now, u_atr=u_atr)

        if action == "hold":
            # Only ratchet the peak when the trailing stop is actually armed
            # here. Recording intraday peaks while the trail is evaluated on the
            # 4H bar would tighten the 4H runner's trail without anyone asking.
            if cfg.trailing_stop and gain is not None:
                st["peak_gain"] = max(st.get("peak_gain", gain), gain)
            out.new_managed[tkr] = st
            continue

        # A stale option mark is the single most common false stop (see
        # _corroborate_option_exit_mark). Re-check before trading on it — but an
        # expiry flatten is driven by the calendar, not the mark, so it stands.
        if route == "option" and not expiring:
            corrected = _corroborate_option_exit_mark(
                client, sym, info.get("current"), info.get("avg_entry"))
            if corrected is not None:
                new_gain, new_mark = corrected
                new_action, new_reason = risk_exit_action(
                    new_gain, trimmed=bool(st.get("trimmed", False)),
                    peak_gain=st.get("peak_gain"), policy=policy, cfg=cfg,
                    route=route, u_entry=u_entry, u_now=u_now, u_atr=u_atr)
                out.anomalies[tkr] = {
                    "symbol": sym, "broker_mark": info.get("current"), "quote_mid": new_mark,
                    "broker_gain": gain, "quote_gain": new_gain,
                    "was": reason or action, "now": new_reason or new_action,
                    "bar": now_et.isoformat(),
                }
                logger.warning(
                    "risk pass: STALE option mark on %s (%s) — broker %s vs live mid %.4f; "
                    "'%s' -> '%s'; trading on the live quote",
                    tkr, sym, info.get("current"), new_mark,
                    reason or action, new_reason or new_action)
                st["last_mark_price"] = new_mark
                st["unrealized_gain"] = float(new_gain)
                gain, action, reason = new_gain, new_action, new_reason
                if action == "hold":
                    out.new_managed[tkr] = st
                    continue

        if info.get("current") is not None and "last_mark_price" not in st:
            st["last_mark_price"] = float(info["current"])
        if gain is not None:
            st["unrealized_gain"] = float(gain)

        if action == "exit":
            out.plan.append((sym, "sell", held, reason, route))
            out.order_audits[sym] = _sell_audit(tkr, sym, held, route)
            out.exit_context[sym] = (tkr, dict(st))
            logger.info("risk pass: %s %s (%s) -> EXIT %s (gain=%s)",
                        module, tkr, sym, reason, f"{gain:+.1%}" if gain is not None else "n/a")
            continue

        if action == "trim":
            qty = trim_quantity(policy.scale_frac, held)
            if qty >= held:
                # Same rule as the 4H engine: an indivisible position takes the
                # profit in full instead of booking nothing. See trim_quantity.
                full_reason = take_profit_reason(policy, full=True)
                out.plan.append((sym, "sell", held, full_reason, route))
                out.order_audits[sym] = _sell_audit(tkr, sym, held, route)
                out.exit_context[sym] = (tkr, dict(st))
                logger.info("risk pass: %s %s (%s) -> FULL take-profit %s x%d",
                            module, tkr, sym, full_reason, held)
                continue
            if qty >= 1:
                out.plan.append((sym, "sell", qty, reason, route))
                out.order_audits[sym] = _sell_audit(tkr, sym, qty, route)
                st["trimmed"] = True
                logger.info("risk pass: %s %s (%s) -> TRIM %s x%d", module, tkr, sym, reason, qty)
        out.new_managed[tkr] = st

    return out


@contextlib.contextmanager
def module_state_lock(module: str, *, timeout_note: str = "") -> Iterator[bool]:
    """Non-blocking exclusive lock on one module's live state.

    The 4H runner and this pass both read-modify-write ``live_state.json``. That
    file has already been corrupted in this repo (see
    ``Data/inference/*/_corrupt_backup``), so a concurrent write is a real
    failure mode rather than a theoretical one. When the runner holds the lock
    this pass yields False and skips the tick: the runner is the authority and
    will evaluate the same stops moments later.
    """
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOCK_ROOT / f"{module}.lock"
    fh = path.open("a+")
    acquired = False
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            logger.info("risk pass: %s state is locked by another writer — skipping tick%s",
                        module, f" ({timeout_note})" if timeout_note else "")
            yield False
            return
        yield True
    finally:
        if acquired:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                logger.debug("risk pass: failed to unlock %s", module, exc_info=True)
        fh.close()


def load_state(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return {"managed": {}, "history": []}


def save_state(path: Path, state: dict) -> None:
    """Write state atomically so a crash mid-write cannot truncate the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(path)
