"""Meta equity plan routed through DecisionCoordinator -> ExecutionGateway.

Task 23, increment 3. This is the cutover itself: the Meta runner stops calling
the broker and starts producing governed decisions. The tests below are mostly
negative — what must *not* reach a broker — because that is the property the
whole nervous system exists to provide.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.contracts.enums import (
    DebitCredit,
    DecisionKind,
    InstrumentFamily,
    OrderSide,
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.orders import OrderRequest
from signals.meta_context.meta_ranker.gateway_execution import (
    GovernedPathUnavailable,
    MetaGatewayRouter,
    RouterRefusal,
    build_router,
    equity_order_request,
)
from signals.meta_context.meta_ranker.nervous_system_adapter import MetaIntentConfig


BAR = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 8, 3, 20, 4, 30, tzinfo=timezone.utc)
ACCOUNT = "paper"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeSnapshot:
    """Minimal stand-in with the fields the router and coordinator read."""

    def __init__(self, ticker: str) -> None:
        self.snapshot_id = uuid5(NAMESPACE_URL, f"meta-router/snapshot/{ticker}")
        self.content_hash = "c" * 64
        self.decision_time = NOW
        self.decision_bar = BAR
        self.valid = True


class FakeSnapshotBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> FakeSnapshot:
        self.calls.append(kwargs)
        return FakeSnapshot(kwargs["entity_id"])


class FakePolicyDecision:
    def __init__(self, action: PolicyAction, budget: Decimal) -> None:
        self.action = action
        self.final_risk_budget = budget
        self.policy_decision_id = uuid5(NAMESPACE_URL, f"meta-router/policy/{action}/{budget}")
        self.config_version = "nervous-system-policy-config@1"


class RecordingCoordinator:
    """Captures what the router hands over, and whether a gateway was built."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.gateway_constructions = 0

    def process_intent(self, intent: Any, **kwargs: Any) -> Any:
        self.calls.append({"intent": intent, **kwargs})
        if kwargs.get("submit"):
            self.gateway_constructions += 1
        return _Outcome(submitted=bool(kwargs.get("submit")))


class _Outcome:
    def __init__(self, submitted: bool) -> None:
        self.submitted = submitted
        self.decision = None
        self.refusal = None
        self.detail = None
        self.execution_result = None
        self.gateway_invoked = submitted


def _config(**updates: Any) -> MetaIntentConfig:
    payload: dict[str, Any] = {
        "requested_notional": Decimal("5000"),
        "model_version": "meta-combo@test",
        "feature_version": "meta-matrix@test",
        "config_version": "meta-runner@test",
        "instrument_preferences": (InstrumentFamily.EQUITY,),
    }
    payload.update(updates)
    return MetaIntentConfig(**payload)


def _router(
    *,
    coordinator: Any = None,
    environment: RuntimeEnvironment = RuntimeEnvironment.QA_PAPER,
    policy_action: PolicyAction = PolicyAction.APPROVE,
    policy_budget: Decimal = Decimal("5000.00"),
    **updates: Any,
) -> MetaGatewayRouter:
    payload: dict[str, Any] = {
        "coordinator": coordinator if coordinator is not None else RecordingCoordinator(),
        "snapshot_builder": FakeSnapshotBuilder(),
        "policy_evaluator": lambda intent, snapshot, config: FakePolicyDecision(
            policy_action, policy_budget
        ),
        "policy_config": object(),
        "freshness_profile": object(),
        "environment": environment,
        "account_alias": ACCOUNT,
        "intent_config": _config(),
        "clock": lambda: NOW,
    }
    payload.update(updates)
    return MetaGatewayRouter(**payload)


def _plan() -> list[tuple]:
    return [
        ("MSFT", "sell", 60, "horizon", "equity"),
        ("AMD", "sell", 9, "take_profit_+30%", "equity"),
        ("NVDA", "buy", 12, "entry", "equity"),
    ]


def _route(router: MetaGatewayRouter, **updates: Any) -> tuple[Any, ...]:
    payload: dict[str, Any] = {
        "exit_context": {"MSFT": ("MSFT", {})},
        "ticker_by_symbol": {},
        "scores_by_ticker": {
            t: {"s_combo": 0.97, "s_quality": 0.6, "s_upside": 0.8}
            for t in ("MSFT", "AMD", "NVDA")
        },
        "decision_bar": BAR,
        "reference_prices": {"NVDA": 100.0},
        "position_keys": {"MSFT": "paper:MSFT", "AMD": "paper:AMD"},
        "policy_mode": PolicyMode.ENFORCE,
        "submit": True,
    }
    payload.update(updates)
    return router.route(_plan(), **payload)


# ---------------------------------------------------------------------------
# The negative guarantees
# ---------------------------------------------------------------------------


def test_production_live_routes_nothing_and_builds_no_gateway() -> None:
    """The environment veto has to happen before any broker object exists, not
    at submission time.
    """

    coordinator = RecordingCoordinator()

    with pytest.raises(ValueError, match="PRODUCTION_LIVE"):
        _router(coordinator=coordinator, environment=RuntimeEnvironment.PRODUCTION_LIVE)

    assert coordinator.calls == []
    assert coordinator.gateway_constructions == 0


def test_off_mode_with_submit_refuses_before_any_planning() -> None:
    """OFF is audit-only. Asking it to submit is a caller error, not a silent
    downgrade to a dry run.
    """

    coordinator = RecordingCoordinator()
    rows = _route(_router(coordinator=coordinator), policy_mode=PolicyMode.OFF, submit=True)

    assert [row.refusal for row in rows] == [RouterRefusal.OFF_MODE_SUBMIT] * 3
    assert coordinator.gateway_constructions == 0


def test_a_dry_run_plans_everything_and_submits_nothing() -> None:
    coordinator = RecordingCoordinator()
    rows = _route(_router(coordinator=coordinator), submit=False)

    assert len(coordinator.calls) == 3
    assert all(call["submit"] is False for call in coordinator.calls)
    assert coordinator.gateway_constructions == 0
    assert all(row.order_request is not None for row in rows)


def test_a_vetoed_intent_never_reaches_an_order_request() -> None:
    rows = _route(_router(policy_action=PolicyAction.REJECT))

    assert [row.refusal for row in rows] == [RouterRefusal.POLICY_VETO] * 3
    assert all(row.order_request is None for row in rows)


def test_every_plan_row_is_accounted_for() -> None:
    """A row that is silently dropped is a position nobody is managing."""

    rows = _route(_router())

    assert len(rows) == 3
    assert [row.symbol for row in rows] == ["MSFT", "AMD", "NVDA"]


# ---------------------------------------------------------------------------
# Order construction
# ---------------------------------------------------------------------------


def test_a_full_exit_orders_the_exact_held_quantity() -> None:
    """Exit parity: the ladder decided 60 shares, so 60 shares is what the
    broker must be asked for.
    """

    row = _route(_router())[0]

    assert row.order_request.decision_kind is DecisionKind.EXIT
    assert row.order_request.risk_reducing is True
    assert row.order_request.parent_quantity == Decimal("60")
    assert row.order_request.equity_side is OrderSide.SELL


def test_a_trim_orders_the_exact_scale_out_quantity() -> None:
    row = _route(_router())[1]

    assert row.order_request.decision_kind is DecisionKind.ADJUSTMENT
    assert row.order_request.parent_quantity == Decimal("9")


def test_a_reduction_carries_the_broker_position_key() -> None:
    """A risk-reducing order without one cannot be tied back to the position it
    is closing, and the contract refuses it.
    """

    row = _route(_router())[0]

    assert row.order_request.broker_position_key == "paper:MSFT"


def test_an_entry_is_sized_from_the_policy_budget_not_the_raw_request() -> None:
    """If the order used the strategy's requested size, the policy engine would
    be decorative: a reduced budget has to produce a smaller order.
    """

    full = _route(_router(policy_budget=Decimal("5000.00")))[2]
    reduced = _route(_router(policy_budget=Decimal("2500.00")))[2]

    assert full.order_request.parent_quantity == Decimal("50")
    assert reduced.order_request.parent_quantity == Decimal("25")


def test_an_unreduced_budget_reproduces_todays_entry_quantity() -> None:
    """Entry parity: when the policy does not downsize, the order must match
    what shares_for_notional would have produced before the cutover.
    """

    from core.live_4h_exec import shares_for_notional

    row = _route(_router(policy_budget=Decimal("5000.00")))[2]

    assert row.order_request.parent_quantity == Decimal(
        str(shares_for_notional(100.0, 5000.0))
    )


def test_an_entry_without_a_reference_price_is_refused_not_guessed() -> None:
    """Sizing needs a price from the exact decision bar. Inventing one would
    put a fabricated quantity on a real order.
    """

    rows = _route(_router(), reference_prices={})

    assert rows[2].refusal is RouterRefusal.NO_REFERENCE_PRICE
    assert rows[2].order_request is None


def test_an_entry_is_never_marked_risk_reducing() -> None:
    row = _route(_router())[2]

    assert row.order_request.risk_reducing is False
    assert row.order_request.decision_kind is DecisionKind.ENTRY
    assert row.order_request.equity_side is OrderSide.BUY


def test_orders_are_equity_market_orders_on_the_paper_account() -> None:
    row = _route(_router())[2]

    assert row.order_request.instrument_family is InstrumentFamily.EQUITY
    assert row.order_request.order_type == "market"
    assert row.order_request.net_limit_price is None
    assert row.order_request.debit_credit is DebitCredit.DEBIT
    assert row.order_request.environment is RuntimeEnvironment.QA_PAPER
    assert row.order_request.account_alias == ACCOUNT


def test_the_order_request_hash_is_self_consistent() -> None:
    for row in _route(_router()):
        assert row.order_request.request_hash == row.order_request.computed_request_hash()


def test_the_same_plan_produces_the_same_order_identity() -> None:
    """A retried 4H pass must converge on the same order, not place a second."""

    first = _route(_router())
    second = _route(_router())

    assert [r.order_request.order_request_id for r in first] == [
        r.order_request.order_request_id for r in second
    ]


def test_a_later_retry_does_not_change_order_identity() -> None:
    """If the order's timestamps moved with each attempt, its content hash
    would move too, and a retried 4H pass would mint a second client order ID
    for the same decision — a duplicate order.
    """

    later = _route(_router(clock=lambda: NOW + timedelta(minutes=6)))

    assert _route(_router())[0].order_request.request_hash == (
        later[0].order_request.request_hash
    )


def test_order_validity_is_anchored_to_the_decision_bar() -> None:
    """The flip side of a stable identity: an order tied to an old bar must go
    stale on its own, so the gateway's expiry check fails closed on a replay
    rather than submitting an hours-old decision.
    """

    row = _route(_router())[0]

    assert row.order_request.created_at == BAR
    assert row.order_request.expires_at == BAR + timedelta(minutes=20)


# ---------------------------------------------------------------------------
# The options path is not yet governed
# ---------------------------------------------------------------------------


def test_an_option_row_is_refused_rather_than_submitted_ungoverned() -> None:
    """options_exec drops the bid/ask it computes and never records a quote
    timestamp, so an OptionLeg cannot be built yet. Falling back to the old
    direct-submit path would defeat the cutover, so the row is refused.
    """

    plan = [("AMD260821C00200000", "buy", 3, "entry", "option")]
    rows = _router().route(
        plan,
        exit_context={},
        ticker_by_symbol={"AMD260821C00200000": "AMD"},
        scores_by_ticker={"AMD": {"s_combo": 0.97}},
        decision_bar=BAR,
        reference_prices={"AMD": 100.0},
        position_keys={},
        policy_mode=PolicyMode.ENFORCE,
        submit=True,
    )

    assert rows[0].refusal is RouterRefusal.OPTION_ROUTE_NOT_GOVERNED
    assert rows[0].order_request is None


# ---------------------------------------------------------------------------
# equity_order_request in isolation
# ---------------------------------------------------------------------------


def test_equity_order_request_rejects_a_non_positive_quantity() -> None:
    with pytest.raises(ValueError, match="quantity"):
        equity_order_request(
            decision_id=uuid5(NAMESPACE_URL, "d"),
            policy_decision_id=uuid5(NAMESPACE_URL, "p"),
            environment=RuntimeEnvironment.QA_PAPER,
            account_alias=ACCOUNT,
            decision_kind=DecisionKind.ENTRY,
            symbol="AMD",
            side=OrderSide.BUY,
            quantity=Decimal("0"),
            risk_reducing=False,
            broker_position_key=None,
            maximum_loss=Decimal("5000"),
            buying_power_required=Decimal("5000"),
            idempotency_key="ab" * 32,
            created_at=NOW,
            expires_at=NOW + timedelta(minutes=20),
        )


def test_equity_order_request_produces_a_valid_contract() -> None:
    request = equity_order_request(
        decision_id=uuid5(NAMESPACE_URL, "d"),
        policy_decision_id=uuid5(NAMESPACE_URL, "p"),
        environment=RuntimeEnvironment.QA_PAPER,
        account_alias=ACCOUNT,
        decision_kind=DecisionKind.EXIT,
        symbol="AMD",
        side=OrderSide.SELL,
        quantity=Decimal("25"),
        risk_reducing=True,
        broker_position_key="paper:AMD",
        maximum_loss=Decimal("0"),
        buying_power_required=Decimal("0"),
        idempotency_key="ab" * 32,
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=20),
    )

    assert isinstance(request, OrderRequest)
    assert request.request_hash == request.computed_request_hash()


# ---------------------------------------------------------------------------
# build_router: misconfiguration must never degrade into a direct broker call
# ---------------------------------------------------------------------------


def test_an_unconfigured_environment_refuses_to_build_the_governed_path() -> None:
    """The caller must stop. Falling back to a direct broker call is exactly
    the bypass this module exists to prevent, so the failure is raised rather
    than returned as None.
    """

    with pytest.raises(GovernedPathUnavailable, match="not configured"):
        build_router(intent_config=_config(), environ={})


def test_a_partially_configured_environment_also_refuses() -> None:
    with pytest.raises(GovernedPathUnavailable):
        build_router(
            intent_config=_config(),
            environ={"CYNOLYCUS_ENVIRONMENT": "QA_PAPER"},
        )


def test_production_live_is_refused_before_any_database_or_broker_is_touched() -> None:
    """The refusal has to precede engine creation: constructing a live-pointed
    engine or adapter is already too far.
    """

    environ = {
        "CYNOLYCUS_ENVIRONMENT": "PRODUCTION_LIVE",
        "CYNOLYCUS_NERVOUS_SYSTEM_MODE": "ENFORCE",
        "CYNOLYCUS_DATABASE_URL": "postgresql+psycopg://u:p@127.0.0.1:1/db",
        "CYNOLYCUS_OPERATIONAL_ROOT": "/tmp/meta-router-test",
        "CYNOLYCUS_EXECUTION_JOURNAL": "local",
        "CYNOLYCUS_ACCOUNT_ALIAS": "live",
    }

    with pytest.raises(GovernedPathUnavailable, match="PRODUCTION_LIVE"):
        build_router(intent_config=_config(), environ=environ)
