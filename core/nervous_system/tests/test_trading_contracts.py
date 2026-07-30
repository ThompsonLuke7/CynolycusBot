from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
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
        "request_hash": "a" * 64,
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=40),
    }
    payload.update(updates)
    return OrderRequest(**payload)


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


def _event(order_id, *, status=ExecutionStatus.ACCEPTED, observed_at=NOW, previous=None, event_hash="b" * 64):
    return ExecutionEvent(
        execution_event_id=uuid4(),
        order_request_id=order_id,
        status=status,
        observed_at=observed_at,
        broker_event_at=observed_at,
        client_order_id="cyno-qa-test",
        broker_order_id="broker-1",
        broker_parent_order_id=None,
        filled_quantity=Decimal("0"),
        average_fill_price=None,
        leg_reports=({"symbol": "AMD260821C00200000", "status": status.value},),
        sanitized_response={"status": status.value, "nested": {"safe": True}},
        previous_event_hash=previous,
        event_hash=event_hash,
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


def test_order_enforces_equity_option_exclusivity_and_hash_shape():
    with pytest.raises(ValidationError):
        _order(equity_symbol="AMD", equity_side=OrderSide.BUY)
    with pytest.raises(ValidationError):
        _order(legs=(), equity_symbol=None, equity_side=OrderSide.BUY)
    with pytest.raises(ValidationError):
        _order(request_hash="not-a-sha256")
    assert _order(request_hash="A" * 64).request_hash == "a" * 64


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


def test_policy_rejects_vetoed_approval_and_nonzero_rejection_budget():
    with pytest.raises(ValidationError):
        _policy(action=PolicyAction.APPROVE, hard_vetoes=("LIVE_DISABLED",))
    with pytest.raises(ValidationError):
        _policy(action=PolicyAction.REJECT, final_risk_budget=Decimal("1"), modifiers=())
    with pytest.raises(ValidationError):
        _policy(expires_at=NOW)


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
    first = _event(order.order_request_id, event_hash="b" * 64)
    second = _event(
        order.order_request_id,
        status=ExecutionStatus.FILLED,
        observed_at=NOW + timedelta(seconds=1),
        previous="b" * 64,
        event_hash="c" * 64,
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
