"""Property tests over a deterministic grid of valid policy inputs (Task 15).

``hypothesis`` is not a project dependency, so the "generated" inputs are an
exhaustive deterministic product of the dimensions that actually drive the
policy engine.  That keeps the properties reproducible and seed-free.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from itertools import product
from typing import Iterator

import pytest

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import (
    DealerRegime,
    DecisionKind,
    MarketRegime,
    ModifierOperation,
    PolicyAction,
    PolicyMode,
    ThemeRegime,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.policy.engine import evaluate_policy
from core.nervous_system.policy.reason_codes import ReasonCode
from core.nervous_system.contracts.enums import StateType
from core.nervous_system.tests.test_policy_engine import (
    build_config,
    build_intent,
    build_snapshot,
    dealer_state,
    degraded_requirement,
    market_state,
    portfolio_state,
    readiness_state,
    theme_state,
    ticker_state,
)


# Ordered from most permissive to most hostile.  Default multipliers must be
# monotone non-increasing along each sequence.
MARKET_SEVERITY = (
    MarketRegime.STRONG_RISK_ON,
    MarketRegime.RISK_ON,
    MarketRegime.NEUTRAL,
    MarketRegime.DETERIORATING,
    MarketRegime.RISK_OFF,
    MarketRegime.CRISIS,
)
THEME_SEVERITY = (
    ThemeRegime.LEADERSHIP,
    ThemeRegime.ACCUMULATION,
    ThemeRegime.HEALTHY,
    ThemeRegime.NEUTRAL,
    ThemeRegime.DETERIORATING,
    ThemeRegime.DISTRIBUTION,
    ThemeRegime.LIQUIDATION,
)
DEALER_SEVERITY = (
    DealerRegime.POSITIVE_GAMMA,
    DealerRegime.UPSIDE_ACCELERATION,
    DealerRegime.NEUTRAL_GAMMA,
    DealerRegime.PINNING,
    DealerRegime.SHORT_GAMMA,
    DealerRegime.DOWNSIDE_ACCELERATION,
)
SIZES = (Decimal("0.00"), Decimal("100.00"), Decimal("1000.01"), Decimal("50000.00"))


def _snapshot_for(
    market: MarketRegime,
    theme: ThemeRegime,
    dealer: DealerRegime,
) -> ContextSnapshot:
    return build_snapshot(
        states=(
            market_state(regime=market),
            theme_state(theme_regime=theme),
            dealer_state(dealer_regime=dealer),
            ticker_state(),
            portfolio_state(),
            readiness_state(),
        )
    )


def _grid() -> Iterator[tuple[TradeIntent, ContextSnapshot]]:
    for market, theme, dealer, size in product(
        MARKET_SEVERITY, THEME_SEVERITY, DEALER_SEVERITY, SIZES
    ):
        snapshot = _snapshot_for(market, theme, dealer)
        yield build_intent(snapshot=snapshot, position_size_requested=size), snapshot


GRID = tuple(_grid())


def test_grid_is_non_trivial() -> None:
    assert len(GRID) == len(MARKET_SEVERITY) * len(THEME_SEVERITY) * len(
        DEALER_SEVERITY
    ) * len(SIZES)


def test_identical_inputs_produce_identical_canonical_decisions() -> None:
    config = build_config()
    for intent, snapshot in GRID:
        first = evaluate_policy(intent, snapshot, config)
        second = evaluate_policy(intent, snapshot, config)
        assert first.policy_decision_id == second.policy_decision_id
        assert content_hash(first) == content_hash(second)


def test_final_size_is_bounded_by_zero_and_the_request() -> None:
    config = build_config()
    for intent, snapshot in GRID:
        decision = evaluate_policy(intent, snapshot, config)
        assert decision.final_risk_budget >= Decimal("0")
        assert decision.final_risk_budget <= intent.position_size_requested
        assert decision.base_risk_budget == intent.position_size_requested


def test_adding_a_hard_veto_cannot_increase_final_size() -> None:
    config = build_config()
    for intent, snapshot in GRID:
        baseline = evaluate_policy(intent, snapshot, config)
        vetoed_snapshot = build_snapshot(
            states=(
                market_state(),
                theme_state(),
                dealer_state(),
                ticker_state(),
                portfolio_state(),
                readiness_state(),
            ),
            stale_inputs=("MARKET",),
            requirement_results=(
                degraded_requirement(StateType.MARKET, required=True, status="STALE"),
            ),
        )
        vetoed = evaluate_policy(
            build_intent(
                snapshot=vetoed_snapshot,
                position_size_requested=intent.position_size_requested,
            ),
            vetoed_snapshot,
            config,
        )
        assert vetoed.action is PolicyAction.REJECT
        assert vetoed.final_risk_budget == Decimal("0")
        assert vetoed.final_risk_budget <= baseline.final_risk_budget


@pytest.mark.parametrize(
    ("severity", "sequence"),
    [
        ("market", MARKET_SEVERITY),
        ("theme", THEME_SEVERITY),
        ("dealer", DEALER_SEVERITY),
    ],
)
def test_worsening_one_risk_dimension_cannot_increase_final_size(
    severity: str, sequence: tuple
) -> None:
    config = build_config()
    previous: Decimal | None = None
    for value in sequence:
        market = value if severity == "market" else MarketRegime.NEUTRAL
        theme = value if severity == "theme" else ThemeRegime.NEUTRAL
        dealer = value if severity == "dealer" else DealerRegime.NEUTRAL_GAMMA
        snapshot = _snapshot_for(market, theme, dealer)
        # Below the position cap so the multiplier, not the cap, is under test.
        decision = evaluate_policy(
            build_intent(snapshot=snapshot, position_size_requested=Decimal("1000.00")),
            snapshot,
            config,
        )
        if previous is not None:
            assert decision.final_risk_budget <= previous
        previous = decision.final_risk_budget


def test_no_modifier_is_omitted_from_the_audit_trail() -> None:
    config = build_config()
    expected_rule_ids = [
        "policy.modifier.market_regime",
        "policy.modifier.theme_regime",
        "policy.modifier.dealer_regime",
        "policy.cap.max_position_notional",
        "policy.cap.money_quantum",
    ]
    for intent, snapshot in GRID:
        decision = evaluate_policy(intent, snapshot, config)
        assert [m.rule_id for m in decision.modifiers] == expected_rule_ids

        budget = decision.base_risk_budget
        for modifier in decision.modifiers:
            assert modifier.budget_before == budget
            if modifier.operation is ModifierOperation.MULTIPLY:
                assert Decimal("0") <= modifier.configured_value <= Decimal("1")
                assert modifier.budget_after == budget * modifier.configured_value
            else:
                assert modifier.budget_after == min(budget, modifier.configured_value)
            assert modifier.config_version == config.config_version
            assert modifier.reason_code == ReasonCode(modifier.reason_code).value
            budget = modifier.budget_after

        if decision.action is PolicyAction.REJECT:
            assert decision.final_risk_budget == Decimal("0")
        else:
            assert decision.final_risk_budget == budget


def test_multipliers_never_increase_risk_across_the_grid() -> None:
    config = build_config()
    for intent, snapshot in GRID:
        decision = evaluate_policy(intent, snapshot, config)
        for modifier in decision.modifiers:
            assert modifier.budget_after <= modifier.budget_before


def test_shadow_and_enforce_agree_on_every_field_except_mode() -> None:
    for intent, snapshot in GRID:
        enforce = evaluate_policy(
            intent, snapshot, build_config(mode=PolicyMode.ENFORCE)
        )
        shadow = evaluate_policy(intent, snapshot, build_config(mode=PolicyMode.SHADOW))
        assert content_hash(
            enforce, exclude={"policy_decision_id", "mode"}
        ) == content_hash(shadow, exclude={"policy_decision_id", "mode"})


def test_off_mode_always_records_the_unmodified_baseline() -> None:
    config = build_config(mode=PolicyMode.OFF)
    for intent, snapshot in GRID:
        decision = evaluate_policy(intent, snapshot, config)
        assert decision.action is PolicyAction.DEFER
        assert decision.modifiers == ()
        assert decision.final_risk_budget == intent.position_size_requested


def test_risk_reducing_exits_are_never_downsized() -> None:
    config = build_config()
    for intent, snapshot in GRID:
        exit_intent = build_intent(
            snapshot=snapshot,
            decision_kind=DecisionKind.EXIT,
            position_size_requested=intent.position_size_requested,
        )
        decision = evaluate_policy(exit_intent, snapshot, config)
        assert decision.action is PolicyAction.EXIT
        assert decision.final_risk_budget == exit_intent.position_size_requested


def test_config_hash_changes_with_any_material_field() -> None:
    base = build_config()
    variants = (
        replace(base, max_position_notional=Decimal("4321.00")),
        replace(base, minimum_order_notional=Decimal("1.00")),
        replace(base, policy_version="nervous-system-policy@2"),
        replace(base, required_snapshot_profile="meta_4h_1620@1"),
    )
    hashes = {base.content_hash} | {variant.content_hash for variant in variants}
    assert len(hashes) == len(variants) + 1
