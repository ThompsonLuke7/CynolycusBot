"""Regression test for the expiring-0DTE forced flatten.

Reproduces the 2026-06-25 SPY leak: a sell-to-close that was submitted but never
filled left the policy in pending_broker_reconcile with an optimistically-flat
local count, which gated the 15:40 flatten off so the 0DTE rode into expiry.

The fix must, at/after the cut-off and for an EXPIRING contract:
  * cancel the stale pending close order,
  * clear the pending_broker_reconcile flag,
  * resubmit a MARKET order to actually flatten.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from strategies.spy_intraday.Policy.order_policy import (
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)

ET = ZoneInfo("America/New_York")
EXPIRING_SYMBOL = "SPY260625C00737000"  # expiry embedded = 2026-06-25
STALE_OID = "1f67aef5-stale"


class _FakeClient:
    def __init__(self) -> None:
        self.canceled: list[str] = []
        self.submitted: list[dict] = []

    def cancel_order(self, order_id):
        self.canceled.append(str(order_id))
        return {"id": order_id, "status": "canceled"}

    def submit_option_order(self, **kwargs):
        self.submitted.append(kwargs)
        return {"id": "new-market-1", "status": "accepted",
                "filled_qty": str(kwargs.get("qty", 1)), "filled_avg_price": "0.05"}


def _policy() -> tuple[OptionOrderPolicy, _FakeClient]:
    cfg = OptionOrderPolicyConfig(submit_orders=True, expiring_position_exit_hhmm="15:40")
    pol = OptionOrderPolicy(cfg)
    fake = _FakeClient()
    pol._client = fake
    # The escalation re-syncs from the broker; the broker truth (1 open contract)
    # is the point of the test, so pin it directly and keep sync a no-op.
    pol.sync_from_broker = lambda **_kw: {}  # type: ignore[method-assign]
    return pol, fake


def test_flatten_escalates_past_stale_pending_reconcile():
    pol, fake = _policy()
    # State that defeated the old logic: local thinks flat, broker still holds it,
    # and a stale pending close is parked on the expiring symbol.
    pol._long_contracts = 1
    pol._long_symbol = EXPIRING_SYMBOL
    pol._pending_broker_reconcile = {
        "symbol": EXPIRING_SYMBOL, "intent": "close", "side": "sell",
        "qty": 1, "order_id": STALE_OID,
    }
    local_ts = datetime(2026, 6, 25, 15, 45, tzinfo=ET)  # past the 15:40 cut-off

    result = pol._maybe_force_close_expiring_positions(local_ts=local_ts, logger=lambda *_: None)

    assert result is not None, "expiring position should have been flattened"
    assert STALE_OID in fake.canceled, "stale pending close order must be canceled"
    assert pol._pending_broker_reconcile is None, "pending flag must be cleared"
    assert len(fake.submitted) == 1
    order = fake.submitted[0]
    assert order["symbol"] == EXPIRING_SYMBOL
    assert order["side"] == "sell"
    assert order["order_type"] == "market", "expiring flatten must be a MARKET order"
    assert pol._long_contracts == 0, "local state must reflect the flatten"


def test_no_action_before_cutoff():
    pol, fake = _policy()
    pol._long_contracts = 1
    pol._long_symbol = EXPIRING_SYMBOL
    pol._pending_broker_reconcile = {
        "symbol": EXPIRING_SYMBOL, "intent": "close", "side": "sell",
        "qty": 1, "order_id": STALE_OID,
    }
    local_ts = datetime(2026, 6, 25, 15, 0, tzinfo=ET)  # before 15:40

    result = pol._maybe_force_close_expiring_positions(local_ts=local_ts, logger=lambda *_: None)

    assert result is None
    assert fake.canceled == []
    assert fake.submitted == []
    assert pol._pending_broker_reconcile is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# 2026-09-03: the flatten's market order was rejected for want of a quote and
# the HTTPError escaped handle_bar, stopping the SPY live loop at 15:58 ET for
# the rest of the session.
# ---------------------------------------------------------------------------

NO_QUOTE_403 = (
    'HTTP Error 403: Forbidden: {"code":40310000,"message":"order has been '
    'rejected due to no available quote for symbol. please reenter with a limit"}'
)
NO_BUYING_POWER_403 = (
    'HTTP Error 403: Forbidden: {"code":40310000,"cost_basis":"5306.01",'
    '"message":"insufficient options buying power","options_buying_power":"5283.58"}'
)


class _NoQuoteClient(_FakeClient):
    """Rejects market orders the way Alpaca does on an empty book."""

    def __init__(self, error: str = NO_QUOTE_403) -> None:
        super().__init__()
        self.error = error
        self.market_attempts = 0

    def submit_option_order(self, **kwargs):
        if str(kwargs.get("order_type")) == "market":
            self.market_attempts += 1
            raise RuntimeError(self.error)
        return super().submit_option_order(**kwargs)


def _expiring_policy(client) -> OptionOrderPolicy:
    cfg = OptionOrderPolicyConfig(submit_orders=True, expiring_position_exit_hhmm="15:40")
    pol = OptionOrderPolicy(cfg)
    pol._client = client
    pol.sync_from_broker = lambda **_kw: {}  # type: ignore[method-assign]
    pol._long_contracts = 1
    pol._long_symbol = EXPIRING_SYMBOL
    return pol


def test_a_no_quote_market_rejection_is_repriced_as_a_limit():
    client = _NoQuoteClient()
    pol = _expiring_policy(client)
    # The ladder needs a price; the point here is that we reach it at all.
    pol._get_contract_price = lambda **_kw: 0.05  # type: ignore[method-assign]

    result = pol._maybe_force_close_expiring_positions(
        local_ts=datetime(2026, 6, 25, 15, 45, tzinfo=ET), logger=lambda *_: None
    )

    assert client.market_attempts == 1
    limits = [k for k in client.submitted if str(k.get("order_type")) != "market"]
    assert limits, "the broker asked for a limit and none was sent"
    assert result is not None


def test_a_flatten_that_cannot_submit_never_escapes_to_the_caller():
    """`on_1m_bar` runs inside the live loop; a raise here ends the session."""

    client = _NoQuoteClient()
    pol = _expiring_policy(client)

    def _no_price(**_kw):
        raise RuntimeError("no_quote_for_limit_pricing")

    pol._get_contract_price = _no_price  # type: ignore[method-assign]

    result = pol._maybe_force_close_expiring_positions(
        local_ts=datetime(2026, 6, 25, 15, 45, tzinfo=ET), logger=lambda *_: None
    )

    assert result is not None
    assert any(o["type"].endswith("_failed") for o in result.get("orders", [])), result
    assert pol._long_contracts == 1, (
        "a position we could not close must stay owned, or it becomes an orphan"
    )


def test_an_insufficient_buying_power_rejection_is_not_repriced():
    """Same status and vendor code, but a refusal rather than a reprice request."""

    client = _NoQuoteClient(error=NO_BUYING_POWER_403)
    pol = _expiring_policy(client)
    pol._get_contract_price = lambda **_kw: 0.05  # type: ignore[method-assign]

    result = pol._maybe_force_close_expiring_positions(
        local_ts=datetime(2026, 6, 25, 15, 45, tzinfo=ET), logger=lambda *_: None
    )

    assert client.market_attempts == 1
    assert not [k for k in client.submitted if str(k.get("order_type")) != "market"], (
        "a buying-power refusal must not be retried as a limit order"
    )
    assert result is not None
    assert pol._long_contracts == 1
