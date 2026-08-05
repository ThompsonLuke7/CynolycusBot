"""Meta Ranker plan -> TradeIntent mapping (Task 23, increment 2).

The mapping is the seam where a strategy's private DataFrame vocabulary becomes
the governed contract vocabulary. Everything the policy engine, the audit trail,
and the replay harness will ever know about a Meta decision has to survive this
translation, so these tests pin the translation itself rather than the runner
that calls it.

``build_trade_intents`` (Task 14) already covered ENTRY from a ranking frame.
Task 23 adds the reductions — trims and full exits — and the whole-plan mapping
that routes every row of an order plan through the same contracts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from core.nervous_system.contracts.enums import (
    DecisionKind,
    Direction,
    InstrumentFamily,
    SizeUnit,
)
from core.nervous_system.contracts.intent import TradeIntent
from signals.meta_context.meta_ranker.nervous_system_adapter import (
    UNSCORED_REASON_CODE,
    MetaIntentConfig,
    build_plan_intents,
    build_reduction_intent,
    underlying_for,
)


BAR = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 8, 3, 20, 4, 30, tzinfo=timezone.utc)
OCC = "AMD260821C00200000"
SNAPSHOT = UUID("11111111-2222-5333-8444-555555555555")


def _config(**updates: object) -> MetaIntentConfig:
    payload: dict[str, object] = {
        "requested_notional": Decimal("5000"),
        "model_version": "meta-combo@2026-06-20",
        "feature_version": "meta-matrix@v4",
        "config_version": "meta-runner@1",
        "instrument_preferences": (InstrumentFamily.EQUITY,),
        "expected_holding_period": "53x4h",
    }
    payload.update(updates)
    return MetaIntentConfig(**payload)  # type: ignore[arg-type]


def _scores(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "s_combo": 0.9731,
        "s_quality": 0.612,
        "s_upside": 0.884,
    }
    payload.update(updates)
    return payload


def _reduction(**updates: object) -> TradeIntent:
    payload: dict[str, object] = {
        "decision_kind": DecisionKind.EXIT,
        "ticker": "AMD",
        "quantity": Decimal("41"),
        "unit": SizeUnit.SHARES,
        "reason_codes": ("horizon",),
        "scores": _scores(),
        "decision_time": NOW,
        "decision_bar": BAR,
        "snapshot_id": SNAPSHOT,
        "config": _config(),
    }
    payload.update(updates)
    return build_reduction_intent(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Required Meta mapping
# ---------------------------------------------------------------------------


def test_a_reduction_carries_the_meta_strategy_and_underlying_ticker() -> None:
    intent = _reduction()

    assert intent.strategy_id == "meta_ranker"
    assert intent.ticker == "AMD"
    assert intent.direction is Direction.LONG


def test_raw_score_is_s_combo() -> None:
    intent = _reduction(scores=_scores(s_combo=0.9412))

    assert intent.raw_score == pytest.approx(0.9412)


def test_an_uncalibrated_score_never_becomes_a_probability() -> None:
    """s_combo is a ranking statistic, not P(win). Storing it as a probability
    would let every downstream consumer read a calibrated number that does not
    exist.
    """

    assert _reduction().raw_probability is None


def test_the_scoring_context_survives_the_translation() -> None:
    components = dict(_reduction().score_components)

    assert components["s_combo"] == pytest.approx(0.9731)
    assert components["s_quality"] == pytest.approx(0.612)
    assert components["s_upside"] == pytest.approx(0.884)


def test_the_selected_bar_and_feature_time_are_the_decision_bar() -> None:
    intent = _reduction()

    assert intent.selected_bar == BAR
    assert intent.feature_timestamp == BAR
    assert intent.created_at == NOW


def test_versions_and_reasons_are_preserved() -> None:
    intent = _reduction(reason_codes=("horizon", "managed"))

    assert intent.model_version == "meta-combo@2026-06-20"
    assert intent.feature_version == "meta-matrix@v4"
    assert intent.config_version == "meta-runner@1"
    assert intent.reason_codes == ("horizon", "managed")


def test_an_occ_symbol_is_never_used_as_the_ticker() -> None:
    assert underlying_for(OCC) == "AMD"
    intent = _reduction(ticker=OCC, unit=SizeUnit.CONTRACTS, quantity=Decimal("3"))

    assert intent.ticker == "AMD"


# ---------------------------------------------------------------------------
# Decision kinds and the requested-size unit
# ---------------------------------------------------------------------------


def test_a_full_exit_is_an_exit_with_a_typed_share_quantity() -> None:
    intent = _reduction()

    assert intent.decision_kind is DecisionKind.EXIT
    assert intent.position_size_requested == Decimal("41")
    assert intent.position_size_unit is SizeUnit.SHARES


def test_a_trim_is_an_adjustment_with_a_typed_contract_quantity() -> None:
    intent = _reduction(
        decision_kind=DecisionKind.ADJUSTMENT,
        quantity=Decimal("2"),
        unit=SizeUnit.CONTRACTS,
        reason_codes=("take_profit_+30%",),
    )

    assert intent.decision_kind is DecisionKind.ADJUSTMENT
    assert intent.position_size_requested == Decimal("2")
    assert intent.position_size_unit is SizeUnit.CONTRACTS


@pytest.mark.parametrize(
    "decision_kind", [DecisionKind.EXIT, DecisionKind.ADJUSTMENT]
)
def test_a_reduction_may_not_be_requested_in_dollars(decision_kind: DecisionKind) -> None:
    """A dollar figure cannot reduce a position exactly; only a typed quantity
    can. Allowing NOTIONAL_USD here would silently reopen the rounding hole the
    unit exists to close.

    Both kinds are covered on purpose: the contract independently refuses a
    dollar-denominated EXIT, so only the ADJUSTMENT case actually exercises the
    builder's own guard.
    """

    with pytest.raises(ValueError, match="typed quantity unit"):
        _reduction(decision_kind=decision_kind, unit=SizeUnit.NOTIONAL_USD)


def test_an_entry_is_not_a_reduction() -> None:
    with pytest.raises(ValueError, match="EXIT or an ADJUSTMENT"):
        _reduction(decision_kind=DecisionKind.ENTRY)


@pytest.mark.parametrize("quantity", [Decimal("0"), Decimal("-5")])
def test_a_reduction_quantity_must_be_positive(quantity: Decimal) -> None:
    with pytest.raises(ValueError, match="positive"):
        _reduction(quantity=quantity)


# ---------------------------------------------------------------------------
# Identity: a replayed bar must converge, not duplicate
# ---------------------------------------------------------------------------


def test_the_same_decision_maps_to_the_same_identity() -> None:
    assert _reduction().intent_id == _reduction().intent_id
    assert _reduction().idempotency_key == _reduction().idempotency_key


def test_rerunning_the_same_bar_later_does_not_mint_a_new_intent() -> None:
    """Identity must not depend on the wall clock. If it did, a retried 4H pass
    would look like a brand-new decision and close the same position twice.
    """

    later = _reduction(decision_time=NOW + timedelta(minutes=7))

    assert _reduction().intent_id == later.intent_id


def test_rebuilding_the_context_snapshot_does_not_mint_a_new_intent() -> None:
    """The snapshot is lineage, not identity. The same name, on the same bar,
    for the same reason is the same decision even if the snapshot was rebuilt.
    """

    rebuilt = _reduction(snapshot_id=UUID("99999999-2222-5333-8444-555555555555"))

    assert _reduction().intent_id == rebuilt.intent_id
    assert _reduction().snapshot_id != rebuilt.snapshot_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("ticker", "NVDA"),
        ("quantity", Decimal("40")),
        ("unit", SizeUnit.CONTRACTS),
        ("reason_codes", ("stop_-39%",)),
        ("decision_kind", DecisionKind.ADJUSTMENT),
        ("decision_bar", BAR - timedelta(hours=4)),
    ],
)
def test_a_materially_different_decision_gets_a_different_identity(
    field: str, value: object
) -> None:
    assert _reduction().intent_id != _reduction(**{field: value}).intent_id


def test_a_different_config_version_gets_a_different_identity() -> None:
    other = _reduction(config=_config(config_version="meta-runner@2"))

    assert _reduction().intent_id != other.intent_id


# ---------------------------------------------------------------------------
# Data integrity at the boundary
# ---------------------------------------------------------------------------


def test_a_reduction_survives_the_name_leaving_the_scored_universe() -> None:
    """A held name can be delisted or dropped from the pool. Refusing to build
    its exit would strand a real position; a risk-reducing action must never
    depend on having a fresh opinion.
    """

    intent = _reduction(scores=None)

    assert intent.raw_score is None
    assert dict(intent.score_components) == {}
    assert UNSCORED_REASON_CODE in intent.reason_codes


def test_an_unscored_reduction_is_still_a_stable_identity() -> None:
    assert _reduction(scores=None).intent_id == _reduction(scores=None).intent_id


def test_an_infinite_score_is_refused_rather_than_coerced() -> None:
    with pytest.raises(ValueError, match="s_quality"):
        _reduction(scores=_scores(s_quality=float("inf")))


def test_a_nan_score_is_treated_as_missing_not_as_a_value() -> None:
    """NaN is how the matrix spells "no value here". Carrying it onto the
    contract would fail the non-finite guard; imputing it would invent a score.
    """

    components = dict(_reduction(scores=_scores(s_quality=float("nan"))).score_components)

    assert "s_quality" not in components
    assert components["s_combo"] == pytest.approx(0.9731)


def test_a_nan_combo_score_makes_the_reduction_unscored() -> None:
    intent = _reduction(scores=_scores(s_combo=float("nan")))

    assert intent.raw_score is None
    assert UNSCORED_REASON_CODE in intent.reason_codes


def test_a_missing_component_is_omitted_not_imputed() -> None:
    scores = _scores()
    del scores["s_upside"]

    components = dict(_reduction(scores=scores).score_components)

    assert "s_upside" not in components
    assert components["s_combo"] == pytest.approx(0.9731)


def test_a_decision_bar_after_the_decision_time_is_refused() -> None:
    """A bar stamped after the moment we acted is look-ahead, and the boundary
    must reject it rather than let it reach the audit record.
    """

    with pytest.raises(ValueError, match="decision_bar"):
        _reduction(decision_time=BAR - timedelta(minutes=1))


def test_a_naive_decision_bar_is_refused() -> None:
    with pytest.raises(ValueError):
        _reduction(decision_bar=datetime(2026, 8, 3, 20, 0))


def test_a_non_uuid_snapshot_is_refused() -> None:
    with pytest.raises(TypeError):
        _reduction(snapshot_id="11111111-2222-5333-8444-555555555555")


# ---------------------------------------------------------------------------
# Whole-plan mapping: nothing is silently dropped or reordered
# ---------------------------------------------------------------------------


def _plan_fixture() -> tuple[list[tuple], dict[str, tuple[str, dict]], dict[str, str]]:
    plan = [
        ("MSFT", "sell", 60, "horizon", "equity"),
        ("AMD", "sell", 9, "take_profit_+30%", "equity"),
        (OCC, "sell", 3, "stop_-39%", "option"),
        ("NVDA", "buy", 12, "entry", "equity"),
    ]
    # build_mixed_plan records exit_context only for FULL exits, never trims.
    exit_context = {"MSFT": ("MSFT", {}), OCC: ("AMD", {})}
    ticker_by_symbol = {OCC: "AMD"}
    return plan, exit_context, ticker_by_symbol


def _map_plan(**updates: object) -> tuple[TradeIntent, ...]:
    plan, exit_context, ticker_by_symbol = _plan_fixture()
    payload: dict[str, object] = {
        "exit_context": exit_context,
        "ticker_by_symbol": ticker_by_symbol,
        "scores_by_ticker": {t: _scores() for t in ("MSFT", "AMD", "NVDA")},
        "snapshot_id_by_ticker": {t: SNAPSHOT for t in ("MSFT", "AMD", "NVDA")},
        "decision_time": NOW,
        "decision_bar": BAR,
        "config": _config(),
    }
    payload.update(updates)
    return build_plan_intents(plan, **payload)  # type: ignore[arg-type]


def test_every_plan_row_becomes_exactly_one_intent_in_order() -> None:
    intents = _map_plan()

    assert len(intents) == 4
    assert [i.ticker for i in intents] == ["MSFT", "AMD", "AMD", "NVDA"]


def test_the_plan_classifier_separates_full_exits_from_trims() -> None:
    assert [i.decision_kind for i in _map_plan()] == [
        DecisionKind.EXIT,
        DecisionKind.ADJUSTMENT,
        DecisionKind.EXIT,
        DecisionKind.ENTRY,
    ]


def test_the_plan_mapping_preserves_ladder_quantities_and_reasons() -> None:
    intents = _map_plan()

    assert [i.position_size_requested for i in intents] == [
        Decimal("60"),
        Decimal("9"),
        Decimal("3"),
        Decimal("5000"),
    ]
    assert [i.position_size_unit for i in intents] == [
        SizeUnit.SHARES,
        SizeUnit.SHARES,
        SizeUnit.CONTRACTS,
        SizeUnit.NOTIONAL_USD,
    ]
    assert [i.reason_codes[0] for i in intents] == [
        "horizon",
        "take_profit_+30%",
        "stop_-39%",
        "entry",
    ]


def test_an_entry_without_scores_fails_loudly() -> None:
    """Opening risk we cannot explain is never acceptable."""

    with pytest.raises(ValueError, match="NVDA"):
        _map_plan(scores_by_ticker={t: _scores() for t in ("MSFT", "AMD")})


def test_an_exit_survives_the_name_dropping_out_of_the_matrix() -> None:
    intents = _map_plan(scores_by_ticker={t: _scores() for t in ("AMD", "NVDA")})

    assert intents[0].ticker == "MSFT"
    assert intents[0].decision_kind is DecisionKind.EXIT
    assert intents[0].position_size_requested == Decimal("60")
    assert intents[0].raw_score is None
    assert UNSCORED_REASON_CODE in intents[0].reason_codes


def test_an_occ_row_without_a_known_ticker_falls_back_to_the_occ_root() -> None:
    assert _map_plan(ticker_by_symbol={})[2].ticker == "AMD"


def test_a_plan_row_without_a_snapshot_fails_loudly() -> None:
    """An intent with no context lineage cannot be replayed or audited."""

    with pytest.raises(KeyError, match="NVDA"):
        _map_plan(snapshot_id_by_ticker={t: SNAPSHOT for t in ("MSFT", "AMD")})


def test_a_malformed_plan_row_is_refused() -> None:
    with pytest.raises(ValueError, match="plan rows"):
        build_plan_intents(
            [("MSFT", "sell", 60)],
            exit_context={},
            ticker_by_symbol={},
            scores_by_ticker={"MSFT": _scores()},
            snapshot_id_by_ticker={"MSFT": SNAPSHOT},
            decision_time=NOW,
            decision_bar=BAR,
            config=_config(),
        )


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------


def test_the_contract_itself_refuses_a_dollar_denominated_exit() -> None:
    """The builder guards this too, but the guarantee has to live in the
    contract: any other producer that builds an EXIT must hit the same wall.
    """

    payload = _reduction().model_dump()
    payload["position_size_unit"] = SizeUnit.NOTIONAL_USD

    with pytest.raises(ValidationError, match="NOTIONAL_USD"):
        TradeIntent(**payload)


def test_an_entry_contract_requires_a_score() -> None:
    """The permission to omit a score belongs to reductions only."""

    payload = _map_plan()[3].model_dump()
    payload["raw_score"] = None

    with pytest.raises(ValidationError, match="raw_score"):
        TradeIntent(**payload)


def test_a_new_format_intent_must_declare_its_size_unit() -> None:
    """Without an explicit unit, ``position_size_requested`` is an unlabelled
    number that means dollars in one place and shares in another.
    """

    payload = _reduction().model_dump()
    payload["position_size_unit"] = SizeUnit.UNKNOWN

    with pytest.raises(ValidationError, match="position_size_unit"):
        TradeIntent(**payload)
