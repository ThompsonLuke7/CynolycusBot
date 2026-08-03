"""Alpaca paper adapter translation and safety tests (Task 19).

The low-level transport is always injected.  No test reads process
credentials or contacts Alpaca.
"""

from __future__ import annotations

import urllib.error
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.contracts.enums import (
    AssetClass,
    DebitCredit,
    DecisionKind,
    ExecutionStatus,
    InstrumentFamily,
    OptionType,
    OrderSide,
    PositionIntent,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.orders import OptionLeg, OrderRequest
from core.nervous_system.execution.alpaca_adapter import (
    CLIENT_ORDER_ID_LENGTH,
    AlpacaPaperAdapter,
    client_order_id_for,
    sanitize,
)
from core.nervous_system.execution.broker import (
    BrokerAdapter,
    BrokerAmbiguousSubmission,
    BrokerAuthenticationError,
    BrokerContractError,
    BrokerOrder,
    BrokerRejected,
    BrokerUnavailable,
    OrderReplacement,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 2, 18, 30, tzinfo=UTC)
PAPER_URL = "https://paper-api.alpaca.markets"
IDEMPOTENCY_KEY = "ab" * 32
D = Decimal


class FakeClient:
    """Records calls and returns scripted payloads."""

    def __init__(self, **responses: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responses = responses

    def _answer(self, name: str, **kwargs: Any) -> Any:
        self.calls.append((name, kwargs))
        result = self._responses.get(name)
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result(**kwargs)
        return result

    def get_account(self, **kw: Any) -> Any:
        return self._answer("get_account", **kw)

    def get_positions(self, **kw: Any) -> Any:
        return self._answer("get_positions", **kw)

    def get_orders(self, **kw: Any) -> Any:
        return self._answer("get_orders", **kw)

    def get_order(self, order_id: str) -> Any:
        return self._answer("get_order", order_id=order_id)

    def get_order_by_client_order_id(self, client_order_id: str) -> Any:
        return self._answer(
            "get_order_by_client_order_id", client_order_id=client_order_id
        )

    def submit_order(self, **kw: Any) -> Any:
        return self._answer("submit_order", **kw)

    def submit_multileg_order(self, **kw: Any) -> Any:
        return self._answer("submit_multileg_order", **kw)

    def cancel_order(self, order_id: str) -> Any:
        return self._answer("cancel_order", order_id=order_id)

    def replace_order(self, order_id: str, payload: Any) -> Any:
        return self._answer("replace_order", order_id=order_id, payload=payload)

    def called(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]


def adapter(client: FakeClient, **overrides: Any) -> AlpacaPaperAdapter:
    settings: dict[str, Any] = {
        "environment": RuntimeEnvironment.QA_PAPER,
        "account_alias": "paper",
        "trading_base_url": PAPER_URL,
        "clock": lambda: NOW,
    }
    settings.update(overrides)
    return AlpacaPaperAdapter(client, **settings)


def order_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "id": "brk-1",
        "client_order_id": IDEMPOTENCY_KEY[:CLIENT_ORDER_ID_LENGTH],
        "status": "accepted",
        "submitted_at": "2026-08-02T18:29:00Z",
        "updated_at": "2026-08-02T18:29:01Z",
        "filled_at": None,
        "filled_qty": "0",
        "filled_avg_price": None,
    }
    payload.update(overrides)
    return payload


def equity_request(**overrides: Any) -> OrderRequest:
    settings: dict[str, Any] = {
        "order_request_id": uuid5(NAMESPACE_URL, "adapter-test/order"),
        "decision_id": uuid5(NAMESPACE_URL, "adapter-test/decision"),
        "policy_decision_id": uuid5(NAMESPACE_URL, "adapter-test/policy"),
        "environment": RuntimeEnvironment.QA_PAPER,
        "account_alias": "paper",
        "decision_kind": DecisionKind.ENTRY,
        "risk_reducing": False,
        "instrument_family": InstrumentFamily.EQUITY,
        "equity_symbol": "AMD",
        "equity_side": OrderSide.BUY,
        "parent_quantity": D("25"),
        "debit_credit": DebitCredit.DEBIT,
        "net_limit_price": None,
        "maximum_loss": D("5000"),
        "buying_power_required": D("5000"),
        "time_in_force": "day",
        "order_type": "market",
        "idempotency_key": IDEMPOTENCY_KEY,
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=20),
    }
    settings.update(overrides)
    return OrderRequest.create(**settings)


def option_leg(strike: str, side: OrderSide, intent: PositionIntent) -> OptionLeg:
    value = D(strike)
    return OptionLeg(
        symbol=f"AMD260918C{int(value * 1000):08d}",
        underlying="AMD",
        option_type=OptionType.CALL,
        strike=value,
        expiration=date(2026, 9, 18),
        side=side,
        ratio=1,
        position_intent=intent,
        quote_at=NOW,
        bid=D("5.00"),
        ask=D("5.10"),
    )


def vertical_request(**overrides: Any) -> OrderRequest:
    settings: dict[str, Any] = {
        "instrument_family": InstrumentFamily.VERTICAL,
        "equity_symbol": None,
        "equity_side": None,
        "legs": (
            option_leg("200", OrderSide.BUY, PositionIntent.BUY_TO_OPEN),
            option_leg("210", OrderSide.SELL, PositionIntent.SELL_TO_OPEN),
        ),
        "parent_quantity": D("3"),
        "order_type": "limit",
        "net_limit_price": D("4.70"),
        "debit_credit": DebitCredit.DEBIT,
    }
    settings.update(overrides)
    return equity_request(**settings)


# --------------------------------------------------------------------------
# Paper-only construction
# --------------------------------------------------------------------------


def test_production_live_fails_before_any_transport_exists() -> None:
    class Exploding:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError("no broker call may happen for PRODUCTION_LIVE")

    with pytest.raises(BrokerAuthenticationError, match="paper-only"):
        AlpacaPaperAdapter(
            Exploding(),
            environment=RuntimeEnvironment.PRODUCTION_LIVE,
            account_alias="live",
            trading_base_url="https://api.alpaca.markets",
        )


@pytest.mark.parametrize(
    ("alias", "url", "match"),
    [
        ("live", PAPER_URL, "paper account alias"),
        ("paper", "https://api.alpaca.markets", "paper host"),
    ],
)
def test_qa_paper_requires_paper_identity_and_host(alias, url, match) -> None:
    with pytest.raises(BrokerAuthenticationError, match=match):
        AlpacaPaperAdapter(
            FakeClient(),
            environment=RuntimeEnvironment.QA_PAPER,
            account_alias=alias,
            trading_base_url=url,
        )


def test_development_may_use_an_injected_fake_on_any_host() -> None:
    instance = AlpacaPaperAdapter(
        FakeClient(),
        environment=RuntimeEnvironment.DEVELOPMENT,
        account_alias="dev",
        trading_base_url="http://localhost:9999",
    )
    assert isinstance(instance, BrokerAdapter)


def test_a_production_live_request_is_refused_at_submit() -> None:
    client = FakeClient(submit_order=order_payload())
    with pytest.raises(BrokerAuthenticationError, match="PRODUCTION_LIVE"):
        adapter(client).submit(
            equity_request(environment=RuntimeEnvironment.PRODUCTION_LIVE)
        )
    assert client.calls == []


def test_account_alias_mismatch_is_refused_before_submit() -> None:
    client = FakeClient(submit_order=order_payload())
    with pytest.raises(BrokerAuthenticationError, match="account alias"):
        adapter(client, account_alias="paper").submit(
            equity_request(account_alias="other")
        )
    assert client.calls == []


# --------------------------------------------------------------------------
# Submission payloads
# --------------------------------------------------------------------------


def test_equity_submit_sends_a_48_character_client_order_id() -> None:
    client = FakeClient(submit_order=order_payload())
    order = adapter(client).submit(equity_request())

    sent = client.called("submit_order")[0]
    assert sent["client_order_id"] == IDEMPOTENCY_KEY[:48]
    assert len(sent["client_order_id"]) == CLIENT_ORDER_ID_LENGTH
    assert sent["symbol"] == "AMD"
    assert sent["qty"] == 25
    assert sent["side"] == "buy"
    assert order.status is ExecutionStatus.ACCEPTED


def test_multileg_submit_maps_legs_and_intents() -> None:
    client = FakeClient(submit_multileg_order=order_payload())
    adapter(client).submit(vertical_request())

    sent = client.called("submit_multileg_order")[0]
    assert sent["qty"] == 3
    assert sent["order_type"] == "limit"
    assert sent["limit_price"] == D("4.70")
    assert sent["legs"] == [
        {
            "symbol": "AMD260918C00200000",
            "ratio_qty": 1,
            "side": "buy",
            "position_intent": "buy_to_open",
        },
        {
            "symbol": "AMD260918C00210000",
            "ratio_qty": 1,
            "side": "sell",
            "position_intent": "sell_to_open",
        },
    ]


def test_a_credit_structure_sends_a_negative_limit_price() -> None:
    client = FakeClient(submit_multileg_order=order_payload())
    adapter(client).submit(
        vertical_request(debit_credit=DebitCredit.CREDIT, net_limit_price=D("1.25"))
    )

    assert client.called("submit_multileg_order")[0]["limit_price"] == D("-1.25")


def test_a_submission_response_is_never_treated_as_a_fill() -> None:
    client = FakeClient(submit_order=order_payload(status="accepted"))
    order = adapter(client).submit(equity_request())

    assert order.status is ExecutionStatus.ACCEPTED
    assert order.filled_quantity == D("0")
    assert order.filled_at is None
    assert order.is_terminal is False


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("new", ExecutionStatus.ACCEPTED),
        ("accepted", ExecutionStatus.ACCEPTED),
        ("partially_filled", ExecutionStatus.PARTIALLY_FILLED),
        ("filled", ExecutionStatus.FILLED),
        ("canceled", ExecutionStatus.CANCELED),
        ("expired", ExecutionStatus.EXPIRED),
        ("rejected", ExecutionStatus.REJECTED),
        ("replaced", ExecutionStatus.CANCELED),
        ("something_new_from_alpaca", ExecutionStatus.UNKNOWN),
    ],
)
def test_status_translation_preserves_the_raw_status(raw_status, expected) -> None:
    filled = raw_status in {"filled", "partially_filled"}
    payload = order_payload(
        status=raw_status,
        filled_qty="5" if filled else "0",
        filled_avg_price="10.05" if filled else None,
        filled_at="2026-08-02T18:29:30Z" if raw_status == "filled" else None,
    )
    client = FakeClient(get_order_by_client_order_id=payload)
    order = adapter(client).find_by_client_order_id("cyn")

    assert order.status is expected
    assert order.raw_status == raw_status


def test_filled_order_preserves_quantities_and_timestamps() -> None:
    payload = order_payload(
        status="filled",
        filled_qty="25",
        filled_avg_price="200.13",
        filled_at="2026-08-02T18:29:30Z",
    )
    order = adapter(FakeClient(get_order=payload)).cancel("brk-1")

    assert order.filled_quantity == D("25")
    assert order.average_fill_price == D("200.13")
    assert order.filled_at == datetime(2026, 8, 2, 18, 29, 30, tzinfo=UTC)
    assert order.submitted_at == datetime(2026, 8, 2, 18, 29, tzinfo=UTC)
    assert order.observed_at == NOW


def test_legs_are_translated_with_their_own_identities() -> None:
    payload = order_payload(
        status="partially_filled",
        filled_qty="1",
        filled_avg_price="4.70",
        legs=[
            {
                "id": "leg-1",
                "symbol": "AMD260918C00200000",
                "ratio_qty": 1,
                "side": "buy",
                "position_intent": "buy_to_open",
                "status": "filled",
                "filled_qty": "1",
                "filled_avg_price": "10.05",
            },
            {
                "id": "leg-2",
                "symbol": "AMD260918C00210000",
                "ratio_qty": 1,
                "side": "sell",
                "position_intent": "sell_to_open",
                "status": "new",
                "filled_qty": "0",
            },
        ],
    )
    order = adapter(FakeClient(get_order_by_client_order_id=payload)).find_by_client_order_id("c")

    assert [leg.broker_order_id for leg in order.legs] == ["leg-1", "leg-2"]
    assert order.legs[0].side is OrderSide.BUY
    assert order.legs[0].filled_quantity == D("1")
    assert order.legs[1].raw_status == "new"
    assert order.legs[1].average_fill_price is None


def test_missing_order_returns_none() -> None:
    client = FakeClient(get_order_by_client_order_id=None)
    assert adapter(client).find_by_client_order_id("nope") is None


def test_unknown_response_fields_are_preserved_in_raw() -> None:
    payload = order_payload(some_future_field={"nested": [1, 2]})
    order = adapter(FakeClient(get_order_by_client_order_id=payload)).find_by_client_order_id("c")

    assert order.raw["some_future_field"] == {"nested": (1, 2)}


def test_raw_payload_redacts_credential_like_fields() -> None:
    payload = order_payload(
        api_secret_key="SUPERSECRET", authorization="Bearer abc", client_order_id="keep-me"
    )
    order = adapter(FakeClient(get_order_by_client_order_id=payload)).find_by_client_order_id("c")

    assert order.raw["api_secret_key"] == "***redacted***"
    assert order.raw["authorization"] == "***redacted***"
    # client_order_id is an order identifier, not a credential.
    assert order.raw["client_order_id"] == "keep-me"


def test_sanitize_is_recursive() -> None:
    cleaned = sanitize({"outer": {"secret_key": "x", "safe": [{"token": "y", "n": 1}]}})
    assert cleaned["outer"]["secret_key"] == "***redacted***"
    assert cleaned["outer"]["safe"][0]["token"] == "***redacted***"
    assert cleaned["outer"]["safe"][0]["n"] == 1


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"status": "new"}, "no broker id"),
        ({"id": "brk-1"}, "no status"),
        ({"id": "brk-1", "status": "new", "filled_qty": "abc"}, "not a number"),
        (
            {"id": "brk-1", "status": "new", "submitted_at": "not-a-time"},
            "not an ISO timestamp",
        ),
    ],
)
def test_malformed_payloads_raise_contract_errors(payload, match) -> None:
    client = FakeClient(get_order_by_client_order_id=payload)
    with pytest.raises(BrokerContractError, match=match):
        adapter(client).find_by_client_order_id("c")


def test_contract_validation_failures_stay_inside_the_typed_error_boundary() -> None:
    """Alpaca can report a filled quantity before the average price settles.

    That must surface as BrokerContractError, not a pydantic ValidationError,
    or it slips past every caller handling BrokerError.
    """

    from core.nervous_system.execution.broker import BrokerError

    client = FakeClient(
        get_order_by_client_order_id=order_payload(
            status="partially_filled", filled_qty="5", filled_avg_price=None
        )
    )
    with pytest.raises(BrokerContractError) as caught:
        adapter(client).find_by_client_order_id("c")
    assert isinstance(caught.value, BrokerError)


def test_naive_timestamps_are_refused() -> None:
    client = FakeClient(
        get_order_by_client_order_id=order_payload(submitted_at="2026-08-02T18:29:00")
    )
    with pytest.raises(BrokerContractError, match="timezone-aware"):
        adapter(client).find_by_client_order_id("c")


# --------------------------------------------------------------------------
# Account, positions, cancel, replace
# --------------------------------------------------------------------------


def test_account_and_positions_are_translated() -> None:
    client = FakeClient(
        get_account={
            "id": "acct-1",
            "status": "ACTIVE",
            "equity": "250000.00",
            "cash": "100000.00",
            "buying_power": "200000.00",
        },
        get_positions=[
            {
                "symbol": "AMD",
                "asset_class": "us_equity",
                "qty": "100",
                "avg_entry_price": "200.00",
                "market_value": "20500.00",
            },
            {
                "symbol": "AMD260918C00200000",
                "asset_class": "us_option",
                "qty": "5",
                "avg_entry_price": "10.00",
                "market_value": "5100.00",
            },
        ],
    )
    instance = adapter(client)

    account = instance.account()
    assert account.equity == D("250000.00")
    assert account.account_alias == "paper"

    positions = instance.positions()
    assert positions[0].asset_class is AssetClass.EQUITY
    assert positions[1].asset_class is AssetClass.OPTION
    assert positions[1].quantity == D("5")


def test_cancel_refetches_rather_than_fabricating_a_terminal_state() -> None:
    client = FakeClient(
        cancel_order=None,  # DELETE answers 204 with no body
        get_order=order_payload(status="canceled"),
    )
    order = adapter(client).cancel("brk-1")

    assert client.called("cancel_order") == [{"order_id": "brk-1"}]
    assert client.called("get_order") == [{"order_id": "brk-1"}]
    assert order.status is ExecutionStatus.CANCELED
    assert order.raw_status == "canceled"


def test_cancel_reports_a_fill_that_won_the_race() -> None:
    """A cancel request does not make the order canceled."""

    client = FakeClient(
        cancel_order=None,
        get_order=order_payload(
            status="filled",
            filled_qty="25",
            filled_avg_price="200.10",
            filled_at="2026-08-02T18:29:30Z",
        ),
    )
    order = adapter(client).cancel("brk-1")

    assert order.status is ExecutionStatus.FILLED
    assert order.filled_quantity == D("25")


def test_replace_sends_only_permitted_fields() -> None:
    client = FakeClient(replace_order=order_payload(status="replaced"))
    order = adapter(client).replace(
        "brk-1", OrderReplacement(quantity=D("2"), limit_price=D("4.50"))
    )

    assert client.called("replace_order")[0]["payload"] == {
        "qty": "2",
        "limit_price": "4.50",
    }
    assert order.raw_status == "replaced"


def test_replace_requires_at_least_one_change() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        OrderReplacement()


def test_a_patch_that_lost_the_race_reports_the_broker_status() -> None:
    client = FakeClient(
        replace_order=order_payload(
            status="filled",
            filled_qty="25",
            filled_avg_price="200.10",
            filled_at="2026-08-02T18:29:30Z",
        )
    )
    order = adapter(client).replace("brk-1", OrderReplacement(quantity=D("2")))

    assert order.status is ExecutionStatus.FILLED


# --------------------------------------------------------------------------
# Typed transport errors
# --------------------------------------------------------------------------


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(PAPER_URL, code, "Error", None, None)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (401, BrokerAuthenticationError),
        (403, BrokerAuthenticationError),
        (422, BrokerRejected),
        (429, BrokerUnavailable),
        (503, BrokerAmbiguousSubmission),
    ],
)
def test_submit_failures_map_to_typed_errors(code, expected) -> None:
    client = FakeClient(submit_order=_http(code))
    with pytest.raises(expected):
        adapter(client).submit(equity_request())


def test_a_lost_write_response_is_ambiguous_not_failed() -> None:
    """A 5xx write may already have been accepted; never assume it failed."""

    client = FakeClient(submit_order=urllib.error.URLError("connection reset"))
    with pytest.raises(BrokerAmbiguousSubmission):
        adapter(client).submit(equity_request())


def test_a_failed_read_is_unavailable_not_ambiguous() -> None:
    client = FakeClient(get_account=_http(503))
    with pytest.raises(BrokerUnavailable):
        adapter(client).account()


def test_client_order_id_requires_an_idempotency_key() -> None:
    with pytest.raises(BrokerContractError, match="idempotency key"):
        client_order_id_for(equity_request(idempotency_key=""))
