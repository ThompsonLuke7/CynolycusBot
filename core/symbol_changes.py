"""Follow a held position through a broker symbol change.

A name change retires a symbol overnight: the shares are still in the account,
under the new symbol, and the old one stops resolving (asset lookup 404). To a
module that looks its position up by the symbol it bought, that is identical to
the position having been sold. HTF's 529 PSKY became SKYD on 2026-10-06 (same
CUSIP); the risk pass reported SKYD as an orphan nobody managed, and the next 4H
pass would have dropped the PSKY claim as `not_found` with no ledger row.

A claim is moved only when the broker's own records say so:

  1. the position is an equity the broker does not hold under its recorded
     symbol, with no entry or exit order in flight;
  2. the broker's corporate-action record names exactly one new symbol for it,
     processed on or after the position's entry date;
  3. the broker holds that new symbol, in at least the recorded size; and
  4. nothing else in the book already uses the new symbol.

Anything short of that leaves the claim exactly as it was. The broker is asked
at most once per position per day.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from core.order_reconciliation import number

logger = logging.getLogger(__name__)

#: How far back to ask when a position records no entry bar.
DEFAULT_LOOKBACK_DAYS = 90


def _entry_date(st: dict, today: date) -> date:
    try:
        return pd.Timestamp(st["entry_bar"]).date()
    except (KeyError, TypeError, ValueError):
        return today - timedelta(days=DEFAULT_LOOKBACK_DAYS)


def _held(pos_info: dict[str, dict], symbol: str) -> float:
    return number(pos_info.get(symbol, {}).get("qty")) or 0.0


def broker_symbol_change(client, symbol: str, *, since: date, today: date) -> dict | None:
    """The broker's record of `symbol` being renamed in [since, today], or None.

    None also covers "could not ask" and "more than one answer": neither is
    authority to move a claim.
    """
    read = getattr(client, "get_corporate_actions", None)
    if read is None:
        return None
    try:
        payload = read(symbols=symbol, types="name_change",
                       start=since.isoformat(), end=today.isoformat()) or {}
    except Exception as exc:  # noqa: BLE001 - an unanswered question changes nothing
        logger.warning("symbol change lookup failed for %s (%s)", symbol, exc)
        return None
    changes = [row for row in (payload.get("corporate_actions") or {}).get("name_changes") or []
               if row.get("old_symbol") == symbol and row.get("new_symbol")
               and row.get("new_symbol") != symbol]
    return changes[0] if len(changes) == 1 else None


def follow_symbol_changes(client, managed: dict[str, dict], pos_info: dict[str, dict], *,
                          module: str, today: date) -> dict[str, dict]:
    """Re-key equity positions the broker renamed. Returns {old key: change record}.

    Mutates `managed` in place, before the caller's own pass over it, so the
    position is then managed like any other held position.
    """
    followed: dict[str, dict] = {}
    in_use = {str(st.get("symbol", key)) for key, st in managed.items()
              if isinstance(st, dict) and st.get("route") == "equity"}
    for key, st in list(managed.items()):
        if not isinstance(st, dict) or st.get("route") != "equity":
            continue
        symbol = str(st.get("symbol", key))
        if (_held(pos_info, symbol) > 0 or st.get("pending_fill")
                or isinstance(st.get("exit_pending"), dict)
                or st.get("symbol_change_checked_on") == today.isoformat()):
            continue
        st["symbol_change_checked_on"] = today.isoformat()
        change = broker_symbol_change(client, symbol, since=_entry_date(st, today), today=today)
        if change is None:
            continue
        new = str(change["new_symbol"])
        size = number(st.get("remaining_qty"))
        size = number(st.get("shares")) if size is None else size
        held = _held(pos_info, new)
        if size is None or held < size or new in in_use or new in managed:
            logger.error(
                "%s %s: the broker renamed it to %s, but the claim was NOT moved "
                "(recorded %s, broker holds %s of %s, already in this book: %s)",
                module, symbol, new, size, held, new, new in in_use or new in managed)
            continue
        st["symbol"] = new
        st["symbol_change"] = {
            "old_symbol": symbol, "new_symbol": new,
            "process_date": change.get("process_date"), "corporate_action_id": change.get("id"),
        }
        del managed[key]
        managed[new] = st
        in_use.add(new)
        followed[key] = st["symbol_change"]
        logger.warning("%s: %s is now %s (broker name change, processed %s) — claim on %s "
                       "shares moved to the new symbol", module, symbol, new,
                       change.get("process_date"), f"{size:g}")
    return followed
