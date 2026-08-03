from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.nervous_system.config.policy import PolicyConfig
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import (
    Direction,
    InstrumentFamily,
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.policy.engine import evaluate_policy


NOW = datetime(2026, 7, 30, 18, 20, tzinfo=timezone.utc)


def make_intent(**updates):
    values = {
        "intent_id": uuid4(),
        "strategy_id": "policy_test",
        "ticker": "AMD",
        "direction": Direction.LONG,
        "decision_kind": "ENTRY",
        "raw_score": 0.9,
        "raw_probability": None,
        "expected_return": None,
        "expected_holding_period": "10x4h",
        "entry_window": "current-or-next-open",
        "preferred_entry": None,
        "invalidation": None,
        "target": None,
        "stop": None,
        "position_size_requested": Decimal("1000"),
        "instrument_preferences": (InstrumentFamily.EQUITY,),
        "feature_timestamp": NOW,
        "created_at": NOW,
        "model_version": "policy-test@1",
        "feature_version": "policy-test@1",
        "reason_codes": ("TEST_INTENT",),
    }
    values.update(updates)
    return TradeIntent(**values)


def make_snapshot(**updates):
    values = {
        "snapshot_id": uuid4(),
        "decision_time": NOW,
        "strategy_id": "policy_test",
        "ticker": "AMD",
        "freshness_profile": "policy-test@1",
        "market_state": None,
        "sector_states": (),
        "theme_memberships": (),
        "theme_states": (),
        "ticker_state": None,
        "catalyst_events": (),
        "catalyst_pressures": (),
        "dealer_state": None,
        "portfolio_state": SimpleNamespace(
            account_alias="paper",
            equity=Decimal("10000"),
            buying_power=Decimal("10000"),
            positions=(),
            open_order_ids=(),
        ),
        "readiness_state": SimpleNamespace(ready=True, reason_codes=()),
        "state_ids": (),
        "state_hashes": (),
        "stale_inputs": (),
        "missing_inputs": (),
        "data_quality": SimpleNamespace(is_usable=True),
        "config_version": "snapshot-test@1",
        "model_versions": (),
        "feature_versions": (),
        "schema_version": 1,
        "content_hash": "snapshot-hash",
    }
    values.update(updates)
    return ContextSnapshot.model_construct(**values)


def make_config(**updates):
    values = {
        "policy_version": "policy@15.1",
        "config_version": "policy-config@15.1",
        "mode": PolicyMode.ENFORCE,
        "environment": RuntimeEnvironment.QA_PAPER,
        "allowed_instruments": frozenset({InstrumentFamily.EQUITY}),
        "allowed_structures": frozenset({"LONG_EQUITY"}),
        "required_snapshot_profile": "policy-test@1",
        "minimum_order_notional": Decimal("1"),
        "max_portfolio_notional": Decimal("300"),
        "regime_multiplier": Decimal("0.8"),
        "theme_multiplier": Decimal("0.5"),
        "liquidity_thresholds": {},
        "context_modifier_thresholds": {},
    }
    values.update(updates)
    return PolicyConfig(**values)


@pytest.mark.parametrize(
    ("environment", "expected_action", "reason"),
    [
        (RuntimeEnvironment.PRODUCTION_LIVE, PolicyAction.REJECT, "PRODUCTION_LIVE_DISABLED"),
        (RuntimeEnvironment.DEVELOPMENT, PolicyAction.REJECT, "POLICY_OFF_AUDIT_ONLY"),
    ],
)
def test_environment_vetoes_are_stable_and_human_readable(environment, expected_action, reason):
    config = make_config(environment=environment, mode=PolicyMode.OFF)
    decision = evaluate_policy(make_intent(), make_snapshot(), config)

    assert decision.action is expected_action
    assert reason in decision.reason_codes
    assert decision.hard_vetoes
    assert all(isinstance(code, str) and code for code in decision.reason_codes)


def test_invalid_or_stale_snapshot_vetoes_entry():
    snapshot = make_snapshot(
        stale_inputs=("ticker_state",),
        missing_inputs=("readiness_state",),
    )

    decision = evaluate_policy(make_intent(), snapshot, make_config())

    assert decision.action is PolicyAction.REJECT
    assert {"SNAPSHOT_STALE_REQUIRED", "SNAPSHOT_MISSING_REQUIRED"} <= set(decision.reason_codes)


def test_qa_paper_requires_paper_identity_and_credentials():
    snapshot = make_snapshot(
        portfolio_state=SimpleNamespace(
            account_alias=None,
            equity=Decimal("10000"),
            buying_power=Decimal("10000"),
            positions=(),
            open_order_ids=(),
        )
    )

    decision = evaluate_policy(make_intent(), snapshot, make_config())

    assert decision.action is PolicyAction.REJECT
    assert "PAPER_ACCOUNT_UNAVAILABLE" in decision.reason_codes


@pytest.mark.parametrize(
    "intent_updates",
    [
        {"instrument_preferences": (InstrumentFamily.OPTION,), "maximum_loss": None},
        {"instrument_preferences": (InstrumentFamily.NAKED_SHORT_OPTION,)},
        {"instrument_preferences": (InstrumentFamily.UNCOVERED_RATIO,)},
        {"idempotency_key": "already-used"},
    ],
)
def test_option_structure_and_duplicate_entry_hard_vetoes(intent_updates):
    snapshot_updates = {}
    if "idempotency_key" in intent_updates:
        snapshot_updates["portfolio_state"] = SimpleNamespace(
            account_alias="paper",
            equity=Decimal("10000"),
            buying_power=Decimal("10000"),
            positions=(),
            open_order_ids=("already-used",),
        )

    decision = evaluate_policy(
        make_intent(**intent_updates),
        make_snapshot(**snapshot_updates),
        make_config(
            allowed_instruments=frozenset(
                {
                    InstrumentFamily.EQUITY,
                    InstrumentFamily.OPTION,
                    InstrumentFamily.NAKED_SHORT_OPTION,
                    InstrumentFamily.UNCOVERED_RATIO,
                }
            )
        ),
    )

    assert decision.action is PolicyAction.REJECT
    assert decision.hard_vetoes


def test_readiness_gate_failure_vetoes_entry():
    decision = evaluate_policy(
        make_intent(),
        make_snapshot(readiness_state=SimpleNamespace(ready=False, reason_codes=("NOT_READY",))),
        make_config(),
    )

    assert decision.action is PolicyAction.REJECT
    assert "READINESS_NOT_READY" in decision.reason_codes


def test_risk_reducing_intent_is_exit_not_entry():
    decision = evaluate_policy(
        make_intent(decision_kind="EXIT"), make_snapshot(), make_config()
    )

    assert decision.action is PolicyAction.EXIT


def test_modifiers_apply_in_order_and_cap_using_decimals():
    decision = evaluate_policy(make_intent(), make_snapshot(), make_config())

    assert decision.base_risk_budget == Decimal("1000")
    assert decision.final_risk_budget == Decimal("300")
    assert [modifier.operation.value for modifier in decision.modifiers] == [
        "MULTIPLY",
        "MULTIPLY",
        "CAP",
    ]
    assert [modifier.budget_before for modifier in decision.modifiers] == [
        Decimal("1000"),
        Decimal("800"),
        Decimal("400"),
    ]
    assert [modifier.budget_after for modifier in decision.modifiers] == [
        Decimal("800"),
        Decimal("400"),
        Decimal("300"),
    ]
    for modifier in decision.modifiers:
        assert modifier.rule_id
        assert modifier.config_version == "policy-config@15.1"


def test_modifier_multipliers_cannot_increase_risk():
    with pytest.raises(ValueError):
        make_config(regime_multiplier=Decimal("1.01"))
