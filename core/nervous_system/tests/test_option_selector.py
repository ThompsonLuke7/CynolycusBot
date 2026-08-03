"""Deterministic instrument selection (Task 18)."""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.config.options import (
    MVP_OPTION_SELECTION_CONFIG,
    OptionSelectionConfig,
)
from core.nervous_system.contracts.enums import (
    DecisionKind,
    Direction,
    InstrumentFamily,
    OptionType,
    OrderSide,
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.policy import PolicyDecision
from core.nervous_system.execution.options.selector import (
    FitnessReason,
    SelectionOutcome,
    select_instrument,
)
from core.nervous_system.tests.fixtures.option_chains import (
    DECISION_TIME,
    FAR_EXPIRY,
    NEAR_EXPIRY,
    base_chain,
    foreign_underlying_quote,
    future_quote,
    illiquid_quote,
    mismatched_multiplier_quote,
    short_dated_quote,
    stale_quote,
    two_expiry_chain,
    wide_spread_quote,
    zero_bid_quote,
    quote,
)
from core.nervous_system.tests.test_policy_engine import (
    build_intent,
    build_snapshot,
    market_state,
    portfolio_state,
    readiness_state,
    ticker_state,
)


D = Decimal


def snapshot_with_chain_time():
    return build_snapshot(
        states=(market_state(), ticker_state(), portfolio_state(), readiness_state())
    )


def build_policy(
    *,
    allowed: frozenset[InstrumentFamily] | None = None,
    budget: Decimal = D("5000.00"),
    action: PolicyAction = PolicyAction.APPROVE,
    intent=None,
    snapshot=None,
) -> PolicyDecision:
    return PolicyDecision(
        policy_decision_id=uuid5(NAMESPACE_URL, "selector-test/policy"),
        intent_id=intent.intent_id,
        snapshot_id=snapshot.snapshot_id,
        environment=RuntimeEnvironment.QA_PAPER,
        mode=PolicyMode.ENFORCE,
        action=action,
        approved_direction=Direction.LONG,
        base_risk_budget=budget,
        final_risk_budget=budget if action is not PolicyAction.REJECT else D("0"),
        allowed_instruments=allowed
        if allowed is not None
        else frozenset(
            {
                InstrumentFamily.EQUITY,
                InstrumentFamily.SINGLE_OPTION,
                InstrumentFamily.VERTICAL,
            }
        ),
        hard_vetoes=() if action is not PolicyAction.REJECT else ("TEST_VETO",),
        modifiers=(),
        stop_adjustment=None,
        target_adjustment=None,
        holding_period_adjustment=None,
        hedge_requirement=None,
        reason_codes=("TEST",),
        policy_version="policy@1",
        config_version="policy-config@1",
        created_at=DECISION_TIME,
        expires_at=DECISION_TIME + timedelta(minutes=20),
    )


def run(
    *,
    chain=None,
    preferences: tuple[InstrumentFamily, ...] = (InstrumentFamily.SINGLE_OPTION,),
    allowed: frozenset[InstrumentFamily] | None = None,
    budget: Decimal = D("5000.00"),
    action: PolicyAction = PolicyAction.APPROVE,
    direction: Direction = Direction.LONG,
    config: OptionSelectionConfig | None = None,
    portfolio=None,
    snapshot=None,
):
    snap = snapshot if snapshot is not None else snapshot_with_chain_time()
    intent = build_intent(
        snapshot=snap, instrument_preferences=preferences, direction=direction
    )
    policy = build_policy(
        allowed=allowed, budget=budget, action=action, intent=intent, snapshot=snap
    )
    return select_instrument(
        intent,
        policy,
        snap,
        base_chain() if chain is None else chain,
        portfolio if portfolio is not None else portfolio_state(),
        config=config or MVP_OPTION_SELECTION_CONFIG,
    )


def reasons_for(selection, symbol: str) -> tuple[str, ...]:
    for item in selection.rejected:
        if symbol in item.leg_symbols:
            return item.reason_codes
    return ()


# --------------------------------------------------------------------------
# Step 1: chain fitness
# --------------------------------------------------------------------------


def test_a_fit_chain_selects_a_single_option() -> None:
    selection = run()

    assert selection.outcome is SelectionOutcome.SELECTED_OPTION
    assert selection.structure is InstrumentFamily.SINGLE_OPTION
    assert len(selection.legs) == 1
    assert selection.legs[0].side is OrderSide.BUY
    assert selection.quantity >= 1
    assert selection.quote_snapshot_at is not None
    assert selection.content_hash == selection.computed_content_hash()


def test_stale_quote_is_rejected_despite_a_recent_trade_print() -> None:
    stale = stale_quote()
    selection = run(
        chain=(stale,), allowed=frozenset({InstrumentFamily.SINGLE_OPTION})
    )

    assert selection.outcome is SelectionOutcome.NO_ELIGIBLE_INSTRUMENT
    assert FitnessReason.QUOTE_STALE.value in reasons_for(selection, stale.symbol)


def test_quote_after_the_decision_time_is_not_available() -> None:
    ahead = future_quote()
    selection = run(chain=(ahead,))

    assert (
        FitnessReason.QUOTE_NOT_AVAILABLE_AT_DECISION.value
        in reasons_for(selection, ahead.symbol)
    )


@pytest.mark.parametrize(
    ("factory", "reason"),
    [
        (wide_spread_quote, FitnessReason.QUOTE_SPREAD_TOO_WIDE),
        (zero_bid_quote, FitnessReason.QUOTE_ZERO_BID),
        (short_dated_quote, FitnessReason.QUOTE_DTE_OUT_OF_RANGE),
        (foreign_underlying_quote, FitnessReason.QUOTE_WRONG_UNDERLYING),
    ],
)
def test_unfit_quotes_are_rejected_with_their_reason(factory, reason) -> None:
    contract = factory()
    selection = run(chain=(contract,))

    assert reason.value in reasons_for(selection, contract.symbol)


def test_illiquid_quote_reports_both_liquidity_reasons() -> None:
    contract = illiquid_quote()
    codes = reasons_for(run(chain=(contract,)), contract.symbol)

    assert FitnessReason.QUOTE_OPEN_INTEREST_TOO_LOW.value in codes
    assert FitnessReason.QUOTE_VOLUME_TOO_LOW.value in codes


def test_mixed_multipliers_cannot_form_one_order() -> None:
    odd = mismatched_multiplier_quote()
    selection = run(
        chain=base_chain() + (odd,),
        preferences=(InstrumentFamily.SINGLE_OPTION,),
    )

    assert (
        FitnessReason.QUOTE_MULTIPLIER_MISMATCH.value
        in reasons_for(selection, odd.symbol)
    )
    assert selection.outcome is SelectionOutcome.SELECTED_OPTION


def test_every_rejection_is_recorded_not_silently_dropped() -> None:
    chain = base_chain() + (
        wide_spread_quote(),
        zero_bid_quote(),
        illiquid_quote(),
    )
    selection = run(chain=chain)

    recorded = {symbol for item in selection.rejected for symbol in item.leg_symbols}
    for contract in (wide_spread_quote(), zero_bid_quote(), illiquid_quote()):
        assert contract.symbol in recorded
    assert all(item.reason_codes for item in selection.rejected)


# --------------------------------------------------------------------------
# Step 2: full-suite construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "chain_factory", "leg_count"),
    [
        (InstrumentFamily.SINGLE_OPTION, base_chain, 1),
        (InstrumentFamily.VERTICAL, base_chain, 2),
        (InstrumentFamily.STRADDLE, base_chain, 2),
        (InstrumentFamily.STRANGLE, base_chain, 2),
        (InstrumentFamily.BUTTERFLY, base_chain, 3),
        (InstrumentFamily.CONDOR, base_chain, 4),
        (InstrumentFamily.IRON_BUTTERFLY, base_chain, 4),
        (InstrumentFamily.IRON_CONDOR, base_chain, 4),
        (InstrumentFamily.CALENDAR, two_expiry_chain, 2),
        (InstrumentFamily.DIAGONAL, two_expiry_chain, 2),
    ],
)
def test_each_approved_structure_can_be_selected_when_requested(
    family, chain_factory, leg_count
) -> None:
    selection = run(
        chain=chain_factory(),
        preferences=(family,),
        allowed=frozenset({family}),
        budget=D("50000.00"),
    )

    assert selection.outcome is SelectionOutcome.SELECTED_OPTION, selection.rejected
    assert selection.structure is family
    assert len(selection.legs) == leg_count
    assert selection.max_loss > 0


def test_cash_secured_put_requires_the_cash() -> None:
    rich = run(
        preferences=(InstrumentFamily.CASH_SECURED_PUT,),
        allowed=frozenset({InstrumentFamily.CASH_SECURED_PUT}),
        budget=D("50000.00"),
        portfolio=portfolio_state(),
    )
    assert rich.outcome is SelectionOutcome.SELECTED_OPTION

    poor_portfolio = portfolio_state()
    poor_portfolio = poor_portfolio.model_copy(update={"cash": 100.0})
    poor = run(
        preferences=(InstrumentFamily.CASH_SECURED_PUT,),
        allowed=frozenset({InstrumentFamily.CASH_SECURED_PUT}),
        budget=D("50000.00"),
        portfolio=poor_portfolio,
    )
    assert poor.outcome is SelectionOutcome.NO_ELIGIBLE_INSTRUMENT
    assert any(
        "NAKED_SHORT_PUT" in item.reason_codes for item in poor.rejected
    )


def test_covered_call_requires_share_coverage() -> None:
    selection = run(
        preferences=(InstrumentFamily.COVERED_CALL,),
        allowed=frozenset({InstrumentFamily.COVERED_CALL}),
        budget=D("50000.00"),
    )

    assert selection.outcome is SelectionOutcome.NO_ELIGIBLE_INSTRUMENT
    assert any(
        "NAKED_SHORT_CALL" in item.reason_codes for item in selection.rejected
    )


def test_short_direction_selects_puts() -> None:
    selection = run(direction=Direction.SHORT)

    assert selection.outcome is SelectionOutcome.SELECTED_OPTION
    assert selection.legs[0].option_type is OptionType.PUT


# --------------------------------------------------------------------------
# Step 3: deterministic scoring and ordering
# --------------------------------------------------------------------------


def test_selection_is_invariant_to_chain_input_order() -> None:
    chain = list(base_chain())
    baseline = run(chain=tuple(chain))

    for seed in range(8):
        shuffled = list(chain)
        random.Random(seed).shuffle(shuffled)
        candidate = run(chain=tuple(shuffled))
        assert candidate.content_hash == baseline.content_hash
        assert candidate.legs == baseline.legs
        assert candidate.selection_id == baseline.selection_id


def test_score_components_are_named_measures_not_probabilities() -> None:
    selection = run()

    assert selection.score is not None
    assert set(selection.score.components) == {
        "spread",
        "liquidity",
        "dte",
        "delta",
        "budget",
    }
    for value in selection.score.components.values():
        assert D("0") <= value <= D("1")
    assert not hasattr(selection.score, "probability")


def test_delta_target_drives_the_single_option_choice() -> None:
    near_target = run(config=replace(MVP_OPTION_SELECTION_CONFIG, target_delta=D("0.55")))
    far_target = run(config=replace(MVP_OPTION_SELECTION_CONFIG, target_delta=D("0.20")))

    assert near_target.legs[0].strike == D("200")
    assert far_target.legs[0].strike == D("220")


def test_unknown_delta_scores_zero_rather_than_on_target() -> None:
    no_delta = tuple(
        contract.model_copy(update={"delta": None}) for contract in base_chain()
    )
    selection = run(chain=no_delta)

    assert selection.outcome is SelectionOutcome.SELECTED_OPTION
    assert selection.score.components["delta"] == D("0")


def test_identical_candidates_break_ties_on_symbol() -> None:
    twin_a = quote("200", OptionType.CALL, bid="9.90", ask="10.10", delta="0.55")
    twin_b = quote(
        "205", OptionType.CALL, bid="9.90", ask="10.10", delta="0.55"
    ).model_copy(update={"strike": D("205")})
    selection = run(chain=(twin_b, twin_a))
    reverse = run(chain=(twin_a, twin_b))

    assert selection.legs == reverse.legs
    assert selection.content_hash == reverse.content_hash


# --------------------------------------------------------------------------
# Steps 5-6: policy filtering and explicit fallback
# --------------------------------------------------------------------------


def test_structure_outside_policy_permission_is_refused() -> None:
    selection = run(
        preferences=(InstrumentFamily.IRON_CONDOR,),
        allowed=frozenset({InstrumentFamily.EQUITY}),
    )

    assert any(
        FitnessReason.STRUCTURE_NOT_PERMITTED.value in item.reason_codes
        for item in selection.rejected
    )
    # Recording the refusal is not enough: the forbidden structure must never
    # be built or selected, whatever the chain would have supported.
    assert selection.structure is not InstrumentFamily.IRON_CONDOR
    assert all(
        leg.symbol == "" for leg in selection.legs
    ) or selection.outcome is not SelectionOutcome.SELECTED_OPTION


@pytest.mark.parametrize(
    "allowed",
    [
        frozenset({InstrumentFamily.EQUITY}),
        frozenset({InstrumentFamily.SINGLE_OPTION}),
        frozenset({InstrumentFamily.VERTICAL}),
    ],
)
def test_selected_structure_always_lies_inside_policy_permission(allowed) -> None:
    """Policy permission is a hard boundary, not a preference hint."""

    every_family = (
        InstrumentFamily.IRON_CONDOR,
        InstrumentFamily.STRADDLE,
        InstrumentFamily.SINGLE_OPTION,
        InstrumentFamily.VERTICAL,
        InstrumentFamily.EQUITY,
    )
    selection = run(
        chain=base_chain(),
        preferences=every_family,
        allowed=allowed,
        budget=D("50000.00"),
    )

    if selection.outcome is SelectionOutcome.SELECTED_OPTION:
        assert selection.structure in allowed
    elif selection.outcome is SelectionOutcome.SELECTED_EQUITY_FALLBACK:
        assert InstrumentFamily.EQUITY in allowed


def test_equity_fallback_when_no_option_is_eligible() -> None:
    selection = run(
        chain=(),
        preferences=(InstrumentFamily.SINGLE_OPTION, InstrumentFamily.EQUITY),
        budget=D("5000.00"),
    )

    assert selection.outcome is SelectionOutcome.SELECTED_EQUITY_FALLBACK
    assert selection.equity_symbol == "AMD"
    assert selection.equity_side is OrderSide.BUY
    # 5000 budget at the snapshot's 100 reference price.
    assert selection.quantity == 50


def test_no_fallback_when_equity_is_not_permitted() -> None:
    selection = run(
        chain=(),
        preferences=(InstrumentFamily.SINGLE_OPTION,),
        allowed=frozenset({InstrumentFamily.SINGLE_OPTION}),
    )

    assert selection.outcome is SelectionOutcome.NO_ELIGIBLE_INSTRUMENT
    assert selection.legs == ()
    assert selection.quantity == 0


def test_fallback_disabled_by_config_is_recorded() -> None:
    selection = run(
        chain=(),
        preferences=(InstrumentFamily.EQUITY,),
        config=replace(MVP_OPTION_SELECTION_CONFIG, equity_fallback_allowed=False),
    )

    assert selection.outcome is SelectionOutcome.NO_ELIGIBLE_INSTRUMENT
    assert any(
        FitnessReason.EQUITY_FALLBACK_NOT_PERMITTED.value in item.reason_codes
        for item in selection.rejected
    )


def test_a_non_executable_policy_selects_nothing() -> None:
    selection = run(action=PolicyAction.REJECT)

    assert selection.outcome is SelectionOutcome.NO_ELIGIBLE_INSTRUMENT
    assert any(
        FitnessReason.POLICY_NOT_EXECUTABLE.value in item.reason_codes
        for item in selection.rejected
    )


def test_budget_too_small_for_one_contract_is_refused() -> None:
    selection = run(
        budget=D("50.00"),
        preferences=(InstrumentFamily.SINGLE_OPTION,),
        allowed=frozenset({InstrumentFamily.SINGLE_OPTION}),
    )

    assert selection.outcome is SelectionOutcome.NO_ELIGIBLE_INSTRUMENT
    assert any(
        FitnessReason.RISK_BUDGET_EXCEEDED.value in item.reason_codes
        for item in selection.rejected
    )


def test_quantity_never_exceeds_the_policy_budget() -> None:
    selection = run(budget=D("2500.00"))

    assert selection.outcome is SelectionOutcome.SELECTED_OPTION
    assert selection.max_loss <= D("2500.00")


def test_selection_records_price_loss_collateral_and_config_hash() -> None:
    selection = run()

    assert selection.estimated_net_price is not None
    assert selection.max_loss > 0
    assert selection.collateral >= 0
    assert selection.config_hash == MVP_OPTION_SELECTION_CONFIG.content_hash


def test_config_hash_changes_with_any_material_field() -> None:
    base = MVP_OPTION_SELECTION_CONFIG
    variants = (
        replace(base, target_delta=D("0.30")),
        replace(base, min_dte=7),
        replace(base, equity_fallback_allowed=False),
    )
    hashes = {base.content_hash} | {variant.content_hash for variant in variants}
    assert len(hashes) == len(variants) + 1


def test_score_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to exactly 1"):
        replace(
            MVP_OPTION_SELECTION_CONFIG,
            score_weights={
                "spread": D("0.5"),
                "liquidity": D("0.5"),
                "dte": D("0.5"),
                "delta": D("0"),
                "budget": D("0"),
            },
        )
