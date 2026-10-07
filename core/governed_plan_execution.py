"""Submit one 4H order plan through the governed path, for any module.

Extracted from meta_ranker's ``_submit_via_gateway`` so a second module does
not need a second copy of it.  The bookkeeping here is subtle and every branch
of it was paid for by a real incident -- a duplicated copy is how the two paths
drift, and a drifted exit path loses positions.

The module-specific parts are all parameters: the router, the audit module
name, and the callback that persists managed state.  Nothing here knows what a
score is called or which strategy asked.

``gateway_verdict`` deliberately distinguishes three outcomes rather than two:

* ACCEPTED    - the broker holds it.
* NO_EXPOSURE - refused or rejected; nothing exists at the broker, so a
  would-be exit must be restored and a would-be entry dropped.
* UNCERTAIN   - DUPLICATE / AMBIGUOUS / RECONCILIATION_REQUIRED. Exposure may
  exist. The claim must be KEPT -- releasing it is how a sibling module adopts
  and liquidates a real position -- and no fill may be booked against it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence

from core.live_4h_exec import (
    drop_failed_entry,
    managed_key_for_symbol,
    mark_entry_unconfirmed,
    plan_row_routes,
    track_exit_submission,
)
from core.nervous_system.contracts.enums import PolicyMode


logger = logging.getLogger(__name__)


def gateway_verdict(row) -> tuple[str, str | None, str]:
    """Classify one routed row as ACCEPTED / NO_EXPOSURE / UNCERTAIN.

    ``row.submitted`` alone is not enough: the gateway can refuse before the
    broker is reached, and it can also come back not knowing whether the broker
    took the order. Those two need opposite handling and were once collapsed
    into one silent success.
    """

    from core.nervous_system.execution.gateway import ExecutionOutcome

    result = getattr(row.outcome, "execution_result", None)
    outcome = getattr(result, "outcome", None)
    reason = getattr(result, "reason_code", None)
    text = getattr(result, "detail", None)
    parts = [str(p) for p in (getattr(outcome, "value", outcome), reason, text) if p]
    detail = ": ".join(parts) if parts else "not submitted"

    if row.refusal is not None:
        detail = row.refusal.value
        if getattr(row, "policy_vetoes", ()):
            detail = f"{detail} ({', '.join(row.policy_vetoes)})"
        return "NO_EXPOSURE", None, detail
    if outcome is ExecutionOutcome.SUBMITTED:
        return "ACCEPTED", getattr(result, "broker_order_id", None), detail
    if outcome in {ExecutionOutcome.REFUSED, ExecutionOutcome.REJECTED}:
        return "NO_EXPOSURE", None, detail
    if outcome is None:
        # The gateway was never reached (policy stop, dry run, no instrument).
        return "NO_EXPOSURE", None, detail
    return "UNCERTAIN", getattr(result, "broker_order_id", None), detail


def submit_plan_via_router(
    router,
    plan: Sequence[Sequence[Any]],
    *,
    module: str,
    client,
    bar,
    new_managed: dict[str, dict],
    exit_context: Mapping[str, Any] | None,
    pos_lookup: Mapping[str, Any] | None,
    scores_by_ticker: Mapping[str, Mapping[str, object]] | None,
    reference_prices: Mapping[str, float] | None = None,
    contract_selection: Mapping[str, Mapping[str, Any]] | None = None,
    persist_managed: Callable[[], None] | None = None,
    dispositions: dict[str, str] | None = None,
    is_option: bool = False,
    policy_mode: PolicyMode = PolicyMode.ENFORCE,
) -> None:
    """Route a whole plan, preserving every per-order bookkeeping guarantee."""

    exit_context = dict(exit_context or {})
    contract_selection = dict(contract_selection or {})
    dispositions = dispositions if dispositions is not None else {}

    quotes_by_symbol: dict[str, Any] = {}
    for selection in contract_selection.values():
        occ, quote = selection.get("occ"), selection.get("quote")
        if occ and quote is not None:
            quotes_by_symbol[occ] = quote

    # A mixed plan labels each row; `is_option` only labels rows that carry no
    # route of their own. Booking a share sale under the plan-level flag is what
    # recorded five Meta equity exits at the x100 contract multiplier.
    row_routes = plan_row_routes(plan)

    def _record(row) -> None:
        symbol = row.symbol
        verdict, broker_id, detail = gateway_verdict(row)
        if verdict == "UNCERTAIN":
            # The broker may or may not hold this order, so the one thing we
            # must not do is release the claim.
            print(
                f"  UNCERTAIN {row.side} {row.quantity} {symbol}: {detail}"
                " — claim KEPT, no fill recorded, reconcile before acting"
            )
            logger.error(
                "%s: gateway could not confirm %s %s %s (%s) — the broker may "
                "hold this order; state left unchanged for reconciliation",
                module, row.side, row.quantity, symbol, detail,
            )
            dispositions[symbol] = "uncertain"
            if symbol in exit_context:
                ticker, previous = exit_context[symbol]
                new_managed[ticker] = previous
        elif verdict != "ACCEPTED":
            print(f"  REFUSED {row.side} {row.quantity} {symbol}: {detail}")
            logger.error(
                "%s: governed path refused %s %s %s: %s",
                module, row.side, row.quantity, symbol, detail,
            )
            dispositions[symbol] = "refused"
            if symbol in exit_context:
                ticker, previous = exit_context[symbol]
                new_managed[ticker] = previous
                logger.warning(
                    "%s: exit refused for %s (%s) — restoring to managed state",
                    module, ticker, symbol,
                )
            else:
                drop_failed_entry(new_managed, symbol)
        else:
            broker_id = broker_id or "?"
            print(f"  OK {row.side} {row.quantity} {symbol}  id={broker_id}")
            dispositions[symbol] = "submitted"
            if str(row.side).strip().lower() == "buy":
                # ACCEPTED means the broker took the order, not that it filled.
                # An accepted-but-unfilled entry otherwise persists as a
                # position the account does not hold.
                mark_entry_unconfirmed(
                    new_managed, symbol, {"id": broker_id}, client=client
                )
            if str(row.side).strip().lower() == "sell":
                entry_state = exit_context.get(symbol, (None, None))[1]
                if entry_state is None and new_managed:
                    # A trim never enters exit_context (full exits only), so the
                    # ledger row would lose its entry lineage.
                    key = managed_key_for_symbol(new_managed, symbol)
                    if key is not None and isinstance(new_managed.get(key), dict):
                        entry_state = new_managed[key]
                item = (
                    symbol,
                    row.side,
                    row.quantity,
                    row.intent.reason_codes[0] if row.intent.reason_codes else "exit",
                    row_routes.get(symbol, "option" if is_option else "equity"),
                )
                track_exit_submission(
                    client,
                    module=module,
                    item=item,
                    resp={"id": broker_id},
                    new_managed=new_managed,
                    exit_context=exit_context,
                    pos_lookup=pos_lookup,
                    bar=bar,
                )
        # Saved after every order, not at the end of the plan: a sibling
        # module's broker reconcile must never find a fresh position missing
        # from this module's on-disk state.
        if persist_managed is not None:
            persist_managed()

    router.route(
        plan,
        exit_context=exit_context,
        ticker_by_symbol={
            selection["occ"]: ticker
            for ticker, selection in contract_selection.items()
            if selection.get("occ")
        },
        scores_by_ticker=scores_by_ticker or {},
        decision_bar=bar.to_pydatetime() if hasattr(bar, "to_pydatetime") else bar,
        reference_prices=reference_prices or {},
        position_keys={str(row[0]): f"paper:{row[0]}" for row in plan},
        policy_mode=policy_mode,
        submit=True,
        quotes_by_symbol=quotes_by_symbol,
        quote_failures={},
        on_row=_record,
    )


__all__ = ["gateway_verdict", "submit_plan_via_router"]
