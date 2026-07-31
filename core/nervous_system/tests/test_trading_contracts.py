from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
import subprocess
import sys
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.nervous_system.contracts.decisions import (
    DecisionOutcome,
    DecisionRecord,
    HashedDecisionArtifact,
)
from core.nervous_system.contracts.enums import (
    DebitCredit,
    Direction,
    ExecutionStatus,
    InstrumentFamily,
    ModifierOperation,
    OptionType,
    OrderSide,
    PolicyAction,
    PolicyMode,
    PositionIntent,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.execution import ExecutionEvent, ExecutionReport
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.orders import OptionLeg, OrderRequest
from core.nervous_system.contracts.policy import PolicyDecision, PolicyModifier


NOW = datetime(2026, 7, 30, 18, 20, tzinfo=timezone.utc)


def _sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _artifact(stage: str, payload: dict[str, object] | None = None) -> HashedDecisionArtifact:
    body = payload or {"stage": stage, "status": "RUN", "nested": {"safe": True}}
    return HashedDecisionArtifact(
        artifact_type=stage,
        schema_version=1,
        content_hash=_sha(body),
        payload=body,
    )


def _not_run_artifact(stage: str) -> HashedDecisionArtifact:
    body = {"status": "NOT_RUN", "blocking_stage": stage, "reason": "upstream veto"}
    return HashedDecisionArtifact(
        artifact_type=stage,
        schema_version=1,
        content_hash=_sha(body),
        payload=body,
    )


def _intent(**updates: object) -> TradeIntent:
    payload: dict[str, object] = {
        "intent_id": uuid4(),
        "strategy_id": "meta_ranker",
        "ticker": "AMD",
        "direction": Direction.LONG,
        "decision_kind": "ENTRY",
        "raw_score": 0.97,
        "raw_probability": None,
        "expected_return": None,
        "expected_holding_period": "53x4h",
        "entry_window": "current-or-next-open",
        "preferred_entry": None,
        "invalidation": None,
        "target": None,
        "stop": None,
        "position_size_requested": Decimal("5000"),
        "instrument_preferences": (InstrumentFamily.VERTICAL, InstrumentFamily.EQUITY),
        "feature_timestamp": NOW,
        "created_at": NOW,
        "model_version": "meta-combo@current",
        "feature_version": "meta-matrix@current",
        "reason_codes": ("META_TOP_K",),
    }
    payload.update(updates)
    return TradeIntent(**payload)


def _leg(**updates: object) -> OptionLeg:
    payload: dict[str, object] = {
        "symbol": "AMD260821C00200000",
        "underlying": "AMD",
        "option_type": OptionType.CALL,
        "strike": Decimal("200"),
        "expiration": "2026-08-21",
        "side": OrderSide.BUY,
        "ratio": 1,
        "position_intent": PositionIntent.BUY_TO_OPEN,
        "quote_at": NOW,
        "bid": Decimal("4.90"),
        "ask": Decimal("5.10"),
    }
    payload.update(updates)
    return OptionLeg(**payload)


def _order(**updates: object) -> OrderRequest:
    payload: dict[str, object] = {
        "order_request_id": uuid4(),
        "decision_id": uuid4(),
        "policy_decision_id": uuid4(),
        "environment": RuntimeEnvironment.QA_PAPER,
        "account_alias": "paper",
        "instrument_family": InstrumentFamily.VERTICAL,
        "legs": (_leg(),),
        "parent_quantity": 1,
        "debit_credit": DebitCredit.DEBIT,
        "net_limit_price": Decimal("5.00"),
        "maximum_loss": Decimal("500"),
        "buying_power_required": Decimal("500"),
        "time_in_force": "day",
        "order_type": "limit",
        "idempotency_key": "ns-test",
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=40),
    }
    payload.update(updates)
    return OrderRequest.create(**payload)


def _modifier(**updates: object) -> PolicyModifier:
    payload: dict[str, object] = {
        "rule_id": "risk.cap",
        "rule_version": "risk.cap@1",
        "operation": ModifierOperation.CAP,
        "input_value": "1000",
        "configured_condition": "max_position_risk",
        "configured_value": Decimal("1000"),
        "budget_before": Decimal("2000"),
        "budget_after": Decimal("1000"),
        "reason_code": "MAX_POSITION_RISK",
        "source_state_id": None,
        "config_version": "policy@1",
    }
    payload.update(updates)
    return PolicyModifier(**payload)


def _policy(**updates: object) -> PolicyDecision:
    payload: dict[str, object] = {
        "policy_decision_id": uuid4(),
        "intent_id": uuid4(),
        "snapshot_id": uuid4(),
        "environment": RuntimeEnvironment.QA_PAPER,
        "mode": PolicyMode.ENFORCE,
        "action": PolicyAction.APPROVE_REDUCED,
        "approved_direction": Direction.LONG,
        "base_risk_budget": Decimal("2000"),
        "final_risk_budget": Decimal("1000"),
        "allowed_instruments": frozenset({InstrumentFamily.VERTICAL}),
        "hard_vetoes": (),
        "modifiers": (_modifier(),),
        "stop_adjustment": None,
        "target_adjustment": None,
        "holding_period_adjustment": None,
        "hedge_requirement": None,
        "reason_codes": ("RISK_CAPPED",),
        "policy_version": "policy@1",
        "config_version": "config@1",
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=20),
    }
    payload.update(updates)
    return PolicyDecision(**payload)


def _event(
    order_id,
    *,
    status=ExecutionStatus.ACCEPTED,
    observed_at=NOW,
    previous=None,
    client_order_id="cyno-qa-test",
    broker_order_id="broker-1",
    broker_parent_order_id=None,
    broker_event_at=None,
    filled_quantity=Decimal("0"),
    average_fill_price=None,
    leg_reports=None,
    sanitized_response=None,
):
    return ExecutionEvent.create(
        order_request_id=order_id,
        status=status,
        observed_at=observed_at,
        broker_event_at=broker_event_at if broker_event_at is not None else observed_at,
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        broker_parent_order_id=broker_parent_order_id,
        filled_quantity=filled_quantity,
        average_fill_price=average_fill_price,
        leg_reports=leg_reports or ({"symbol": "AMD260821C00200000", "status": status.value},),
        sanitized_response=sanitized_response or {"status": status.value, "nested": {"safe": True}},
        previous_event_hash=previous,
    )


def _decision(**updates: object) -> DecisionRecord:
    payload: dict[str, object] = {
        "decision_record_id": uuid4(),
        "decision_time": NOW,
        "snapshot_id": uuid4(),
        "intent_id": uuid4(),
        "policy_decision_id": uuid4(),
        "order_request_ids": (),
        "source_manifest_hash": "1" * 64,
        "snapshot_hash": "2" * 64,
        "intent_hash": "3" * 64,
        "policy_hash": "4" * 64,
        "raw_strategy_output": _artifact("RAW_STRATEGY_OUTPUT"),
        "exposure_report": _artifact("EXPOSURE_REPORT"),
        "instrument_candidates": _artifact("INSTRUMENT_CANDIDATES"),
        "instrument_selection": _artifact("INSTRUMENT_SELECTION"),
        "order_hashes": (),
        "config_hash": "5" * 64,
        "model_versions": {"meta": "meta@1"},
        "feature_versions": {"matrix": "features@1"},
        "schema_version": 1,
    }
    payload.update(updates)
    return DecisionRecord(**payload)


def test_uncalibrated_meta_intent_keeps_probability_null_and_copy_revalidates():
    intent = _intent()

    assert intent.raw_probability is None
    with pytest.raises(ValidationError):
        intent.model_copy(update={"raw_probability": 1.01})


@pytest.mark.parametrize("field", ["position_size_requested", "preferred_entry", "target", "stop"])
def test_trade_intent_rejects_nonfinite_or_negative_decimal_values(field):
    value = Decimal("NaN") if field == "position_size_requested" else Decimal("-0.01")
    with pytest.raises(ValidationError):
        _intent(**{field: value})


def test_option_leg_requires_valid_quote_expiry_and_side_intent():
    with pytest.raises(ValidationError):
        _leg(bid=Decimal("NaN"))
    with pytest.raises(ValidationError):
        _leg(ask=Decimal("4.89"))
    with pytest.raises(ValidationError):
        _leg(expiration=date(2026, 7, 29))
    with pytest.raises(ValidationError):
        _leg(side=OrderSide.SELL)
    with pytest.raises(ValidationError):
        _leg(position_intent=PositionIntent.SELL_TO_OPEN)


def test_order_rejects_more_than_four_option_legs():
    with pytest.raises(ValidationError):
        _order(legs=(_leg(), _leg(), _leg(), _leg(), _leg()))


def test_order_enforces_equity_option_exclusivity_and_hash_integrity():
    with pytest.raises(ValidationError):
        _order(equity_symbol="AMD", equity_side=OrderSide.BUY)
    with pytest.raises(ValidationError):
        _order(legs=(), equity_symbol=None, equity_side=OrderSide.BUY)

    order = _order()
    invalid_hash = order.model_dump()
    invalid_hash["request_hash"] = "not-a-sha256"
    with pytest.raises(ValidationError):
        OrderRequest(**invalid_hash)

    uppercase_hash = order.model_dump()
    uppercase_hash["request_hash"] = order.request_hash.upper()
    assert OrderRequest(**uppercase_hash).request_hash == order.request_hash


def test_order_rejects_bad_money_bounds_expiry_and_credit_limit():
    with pytest.raises(ValidationError):
        _order(maximum_loss=Decimal("-1"))
    with pytest.raises(ValidationError):
        _order(buying_power_required=Decimal("Infinity"))
    with pytest.raises(ValidationError):
        _order(expires_at=NOW)
    with pytest.raises(ValidationError):
        _order(parent_quantity=0)
    with pytest.raises(ValidationError):
        _order(debit_credit=DebitCredit.CREDIT, net_limit_price=Decimal("0"))


def test_order_accepts_market_equity_without_limit_price_and_hashes_deterministically():
    order = _order(
        instrument_family=InstrumentFamily.EQUITY,
        equity_symbol="AMD",
        equity_side=OrderSide.SELL,
        legs=(),
        order_type="market",
        net_limit_price=None,
    )

    assert order.order_type == "market"
    assert order.net_limit_price is None
    assert order.request_hash == order.computed_request_hash()
    same_content = OrderRequest.create(
        order_request_id=uuid4(),
        **order.request_hash_material().model_dump(),
    )
    assert same_content.request_hash == order.request_hash


def test_order_rejects_limit_without_price():
    with pytest.raises(
        ValidationError,
        match="limit orders require a positive non-null net_limit_price",
    ):
        _order(net_limit_price=None)


def test_order_rejects_market_with_price():
    with pytest.raises(
        ValidationError,
        match="market orders require net_limit_price to be null",
    ):
        _order(order_type="market", net_limit_price=Decimal("5.00"))


@pytest.mark.parametrize("order_type", ["stop", "LIMIT", "MARKET"])
def test_order_rejects_unsupported_or_uppercase_order_types(order_type):
    with pytest.raises(ValidationError, match="Input should be 'limit' or 'market'"):
        _order(order_type=order_type)


def test_policy_rejects_vetoed_approval_and_nonzero_rejection_budget():
    with pytest.raises(ValidationError):
        _policy(action=PolicyAction.APPROVE, hard_vetoes=("LIVE_DISABLED",))
    with pytest.raises(ValidationError):
        _policy(action=PolicyAction.REJECT, final_risk_budget=Decimal("1"), modifiers=())
    with pytest.raises(ValidationError):
        _policy(expires_at=NOW)
    with pytest.raises(ValidationError):
        _policy(action=PolicyAction.APPROVE_REDUCED, hard_vetoes=("LIVE_DISABLED",))


@pytest.mark.parametrize("hard_vetoes", [(), ("ENV_DISABLED",)])
def test_policy_reject_allows_positive_base_and_zero_final_without_modifier(hard_vetoes):
    decision = _policy(
        action=PolicyAction.REJECT,
        base_risk_budget=Decimal("2000"),
        final_risk_budget=Decimal("0"),
        hard_vetoes=hard_vetoes,
        modifiers=(),
    )

    assert decision.base_risk_budget == Decimal("2000")
    assert decision.final_risk_budget == Decimal("0")


def test_policy_reject_validates_modifier_chain_without_requiring_zeroing_modifier():
    decision = _policy(
        action=PolicyAction.REJECT,
        final_risk_budget=Decimal("0"),
        modifiers=(_modifier(),),
    )
    assert decision.modifiers[-1].budget_after == Decimal("1000")

    misaligned = _modifier(
        configured_value=Decimal("500"),
        budget_before=Decimal("1000"),
        budget_after=Decimal("500"),
    )
    with pytest.raises(ValidationError):
        _policy(
            action=PolicyAction.REJECT,
            final_risk_budget=Decimal("0"),
            modifiers=(misaligned,),
        )


def test_policy_modifiers_form_a_budget_chain():
    second = _modifier(
        rule_id="risk.multiply",
        operation=ModifierOperation.MULTIPLY,
        configured_value=Decimal("0.5"),
        budget_before=Decimal("1000"),
        budget_after=Decimal("500"),
    )
    decision = _policy(modifiers=(_modifier(), second), final_risk_budget=Decimal("500"))
    assert decision.modifiers[1].budget_before == Decimal("1000")
    with pytest.raises(ValidationError):
        _policy(modifiers=(_modifier(),), final_risk_budget=Decimal("500"))
    with pytest.raises(ValidationError):
        _modifier(
            operation=ModifierOperation.MULTIPLY,
            configured_value=Decimal("1.01"),
            budget_before=Decimal("1000"),
            budget_after=Decimal("1010"),
        )


def test_option_parent_quantity_is_integral_but_equity_quantity_may_be_fractional():
    with pytest.raises(ValidationError):
        _order(parent_quantity=Decimal("1.5"))
    assert _order(
        instrument_family=InstrumentFamily.EQUITY,
        equity_symbol="AMD",
        equity_side=OrderSide.BUY,
        legs=(),
        parent_quantity=Decimal("1.5"),
    ).parent_quantity == Decimal("1.5")


def test_single_option_side_determines_debit_or_credit():
    sell_leg = _leg(side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN)
    with pytest.raises(ValidationError):
        _order(legs=(sell_leg,), debit_credit=DebitCredit.DEBIT)
    assert _order(legs=(sell_leg,), debit_credit=DebitCredit.CREDIT).debit_credit is DebitCredit.CREDIT


def test_order_hash_authenticates_content_across_construction_copy_and_json():
    order = _order()
    assert order.request_hash == order.computed_request_hash()

    same_content = OrderRequest.create(
        order_request_id=uuid4(),
        **order.request_hash_material().model_dump(),
    )
    assert same_content.request_hash == order.request_hash
    assert order.model_copy(update={"order_request_id": uuid4()}).request_hash == order.request_hash
    assert OrderRequest.model_validate_json(order.model_dump_json()) == order

    tampered = order.model_dump()
    tampered["maximum_loss"] = Decimal("501")
    with pytest.raises(ValidationError):
        OrderRequest(**tampered)
    with pytest.raises(ValidationError):
        order.model_copy(update={"maximum_loss": Decimal("501")})
    encoded = json.loads(order.model_dump_json())
    encoded["maximum_loss"] = "501"
    with pytest.raises(ValidationError):
        OrderRequest.model_validate_json(json.dumps(encoded))


def test_order_hash_preserves_option_leg_order():
    long_leg = _leg()
    short_leg = _leg(
        symbol="AMD260821C00210000",
        strike=Decimal("210"),
        side=OrderSide.SELL,
        position_intent=PositionIntent.SELL_TO_OPEN,
        bid=Decimal("2.40"),
        ask=Decimal("2.60"),
    )
    order = _order(legs=(long_leg, short_leg))
    material = order.request_hash_material().model_dump()
    material["legs"] = (short_leg, long_leg)
    reversed_order = OrderRequest.create(order_request_id=uuid4(), **material)

    assert reversed_order.request_hash != order.request_hash


def test_order_hash_is_stable_across_python_hash_seeds():
    script = (
        "from datetime import datetime, timezone; from decimal import Decimal; "
        "from uuid import UUID; "
        "from core.nervous_system.contracts.enums import "
        "DebitCredit, InstrumentFamily, OptionType, OrderSide, PositionIntent, RuntimeEnvironment; "
        "from core.nervous_system.contracts.orders import OptionLeg, OrderRequest; "
        "now=datetime(2026,7,30,18,20,tzinfo=timezone.utc); "
        "leg=OptionLeg(symbol='AMD260821C00200000',underlying='AMD',option_type=OptionType.CALL,"
        "strike=Decimal('200'),expiration='2026-08-21',side=OrderSide.BUY,ratio=1,"
        "position_intent=PositionIntent.BUY_TO_OPEN,quote_at=now,bid=Decimal('4.90'),"
        "ask=Decimal('5.10')); "
        "order=OrderRequest.create(order_request_id=UUID('00000000-0000-0000-0000-000000000001'),"
        "decision_id=UUID('00000000-0000-0000-0000-000000000002'),"
        "policy_decision_id=UUID('00000000-0000-0000-0000-000000000003'),"
        "environment=RuntimeEnvironment.QA_PAPER,account_alias='paper',"
        "instrument_family=InstrumentFamily.VERTICAL,legs=(leg,),parent_quantity=1,"
        "debit_credit=DebitCredit.DEBIT,net_limit_price=Decimal('5.00'),"
        "maximum_loss=Decimal('500'),buying_power_required=Decimal('500'),"
        "time_in_force='day',order_type='limit',idempotency_key='ns-test',"
        "created_at=now,expires_at=now.replace(hour=19)); print(order.request_hash)"
    )
    outputs = []
    for seed in ("1", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]


def test_special_payload_values_are_frozen_canonical_and_json_round_trip():
    artifact = HashedDecisionArtifact.from_payload(
        "SPECIAL",
        1,
        {
            "symbols": frozenset({"NVDA", "AMD"}),
            "tags": {"z", "a"},
            "created_at": NOW,
            "identifier": uuid4(),
            "amount": Decimal("1.20"),
            "direction": Direction.LONG,
        },
    )
    assert artifact.payload["symbols"] == ("AMD", "NVDA")
    assert artifact.payload["tags"] == ("a", "z")
    with pytest.raises(TypeError):
        artifact.payload["symbols"] += ("TSLA",)
    assert HashedDecisionArtifact.model_validate_json(artifact.model_dump_json()) == artifact


def test_nonfinite_decimal_is_rejected_in_arbitrary_artifact_payload():
    with pytest.raises(ValueError):
        HashedDecisionArtifact.from_payload("BAD", 1, {"bad": Decimal("NaN")})
    with pytest.raises(ValidationError):
        HashedDecisionArtifact(
            artifact_type="BAD",
            schema_version=1,
            content_hash="0" * 64,
            payload={"bad": Decimal("Infinity")},
        )
    with pytest.raises(ValidationError):
        HashedDecisionArtifact(
            artifact_type="BAD",
            schema_version=1,
            content_hash=_sha({"bad": "Infinity"}),
            payload={"bad": Decimal("Infinity")},
        )


def test_special_artifact_hash_is_stable_across_python_hash_seeds():
    script = (
        "from datetime import datetime, timezone; from decimal import Decimal; "
        "from core.nervous_system.contracts.decisions import HashedDecisionArtifact; "
        "from core.nervous_system.contracts.enums import Direction; "
        "p={'symbols': frozenset({'NVDA','AMD'}), 'tags': {'z','a'}, "
        "'created_at': datetime(2026,7,30,18,20,tzinfo=timezone.utc), "
        "'amount': Decimal('1.20'), 'direction': Direction.LONG}; "
        "print(HashedDecisionArtifact.from_payload('SPECIAL',1,p).content_hash)"
    )
    outputs = []
    for seed in ("1", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]


def test_hashed_artifacts_and_decision_maps_are_deeply_immutable_and_round_trip():
    record = _decision()

    with pytest.raises(TypeError):
        record.raw_strategy_output.payload["new"] = True
    with pytest.raises(TypeError):
        record.raw_strategy_output.payload["nested"]["new"] = True
    with pytest.raises(TypeError):
        record.model_versions["new"] = "bad"

    restored = DecisionRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.instrument_selection.content_hash == record.instrument_selection.content_hash


def test_hashed_artifact_rejects_payload_hash_mismatch_and_invalid_hash_case():
    with pytest.raises(ValidationError):
        HashedDecisionArtifact(
            artifact_type="RAW",
            schema_version=1,
            content_hash="a" * 64,
            payload={"value": 1},
        )
    with pytest.raises(ValidationError):
        HashedDecisionArtifact(
            artifact_type="RAW",
            schema_version=1,
            content_hash="g" * 64,
            payload={},
        )


def test_execution_report_requires_ordered_hash_chain_and_last_status():
    order = _order()
    first = _event(order.order_request_id)
    second = _event(
        order.order_request_id,
        status=ExecutionStatus.FILLED,
        observed_at=NOW + timedelta(seconds=1),
        previous=first.event_hash,
    )
    report = ExecutionReport(
        order_request_id=order.order_request_id,
        events=(first, second),
        current_status=ExecutionStatus.FILLED,
    )
    assert report.events[-1].status is ExecutionStatus.FILLED
    with pytest.raises(ValidationError):
        ExecutionReport(
            order_request_id=order.order_request_id,
            events=(second, first),
            current_status=ExecutionStatus.FILLED,
        )
    with pytest.raises(ValidationError):
        ExecutionReport(
            order_request_id=order.order_request_id,
            events=(first, second),
            current_status=ExecutionStatus.ACCEPTED,
        )


def test_execution_payloads_and_leg_reports_are_immutable():
    event = _event(uuid4())

    with pytest.raises(TypeError):
        event.sanitized_response["status"] = "MUTATED"
    with pytest.raises(TypeError):
        event.sanitized_response["nested"]["safe"] = False
    with pytest.raises(TypeError):
        event.leg_reports[0]["status"] = "MUTATED"
    assert ExecutionEvent.model_validate_json(event.model_dump_json()) == event


def test_execution_event_hash_authenticates_content_but_excludes_event_identity():
    event = _event(uuid4())
    assert event.event_hash == event.computed_event_hash()
    assert _event(event.order_request_id).event_hash == event.event_hash
    assert _event(uuid4()).event_hash != event.event_hash

    tampered = event.model_dump()
    tampered["sanitized_response"]["status"] = "TAMPERED"
    with pytest.raises(ValidationError):
        ExecutionEvent(**tampered)
    with pytest.raises(ValidationError):
        event.model_copy(update={"sanitized_response": {"status": "TAMPERED"}})
    encoded = json.loads(event.model_dump_json())
    encoded["sanitized_response"]["status"] = "TAMPERED"
    with pytest.raises(ValidationError):
        ExecutionEvent.model_validate_json(json.dumps(encoded))


def test_execution_event_rejects_future_broker_time_and_report_identity_drift():
    with pytest.raises(ValidationError):
        _event(uuid4(), broker_event_at=NOW + timedelta(seconds=1))
    order_id = uuid4()
    first = _event(order_id, broker_order_id=None, broker_parent_order_id=None)
    second = _event(
        order_id,
        observed_at=NOW + timedelta(seconds=1),
        previous=first.event_hash,
        client_order_id="different-client",
        broker_order_id="broker-1",
    )
    with pytest.raises(ValidationError):
        ExecutionReport(order_request_id=order_id, events=(first, second), current_status=second.status)
    third = _event(
        order_id,
        observed_at=NOW + timedelta(seconds=1),
        previous=first.event_hash,
        broker_order_id="broker-1",
    )
    fourth = _event(
        order_id,
        observed_at=NOW + timedelta(seconds=2),
        previous=third.event_hash,
        broker_order_id=None,
    )
    with pytest.raises(ValidationError):
        ExecutionReport(order_request_id=order_id, events=(first, third, fourth), current_status=fourth.status)


def test_decision_record_requires_explicit_not_run_artifacts_for_upstream_veto():
    record = _decision(
        raw_strategy_output=_not_run_artifact("STRATEGY"),
        exposure_report=_not_run_artifact("STRATEGY"),
        instrument_candidates=_not_run_artifact("STRATEGY"),
        instrument_selection=_not_run_artifact("STRATEGY"),
    )
    assert record.instrument_selection.payload["status"] == "NOT_RUN"
    with pytest.raises(ValidationError):
        _decision(instrument_selection=None)
    with pytest.raises(ValidationError):
        _decision(
            raw_strategy_output=_not_run_artifact("STRATEGY"),
            exposure_report=_artifact("EXPOSURE_REPORT"),
        )


def test_decision_record_rejects_duplicate_order_ids():
    duplicate = uuid4()
    with pytest.raises(ValidationError):
        _decision(
            order_request_ids=(duplicate, duplicate),
            order_hashes=("6" * 64, "7" * 64),
        )


def test_outcome_factory_rejects_evaluation_before_linked_decision():
    record = _decision()
    with pytest.raises(ValueError):
        DecisionOutcome.for_decision(
            record,
            outcome_id=uuid4(),
            evaluated_at=NOW - timedelta(seconds=1),
            horizon="5d",
            underlying_return=0.02,
            instrument_return=0.03,
            source_fitness_report_id=None,
            metrics={"pnl": 0.03},
        )

    outcome = DecisionOutcome.for_decision(
        record,
        outcome_id=uuid4(),
        evaluated_at=NOW + timedelta(days=5),
        horizon="5d",
        underlying_return=0.02,
        instrument_return=0.03,
        source_fitness_report_id=None,
        metrics={"pnl": 0.03},
    )
    assert outcome.decision_record_id == record.decision_record_id


def test_standalone_outcome_cannot_claim_cross_record_validation():
    outcome = DecisionOutcome(
        outcome_id=uuid4(),
        decision_record_id=uuid4(),
        evaluated_at=NOW - timedelta(days=1),
        horizon="5d",
        underlying_return=None,
        instrument_return=None,
        source_fitness_report_id=None,
        metrics={},
    )
    assert outcome.evaluated_at < NOW
