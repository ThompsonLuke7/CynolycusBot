"""Portfolio-scoped peer memberships and the concentration veto.

Before this, ``calculate_exposure`` was reachable only from its own tests and
the symbol/sector/theme/factor limits were configured but enforced by nothing.
Theme was additionally uncomputable: ``_allocate_theme`` bucketed every
position whose underlying was not the snapshot's ticker into ``UNALLOCATED``,
so a theme limit could only ever see one position.

These tests pin both halves: peers make portfolio-wide theme exposure
computable, and the policy engine vetoes on it.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.config.portfolio import MVP_PORTFOLIO_CONFIG
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import StateType
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import ThemeMembership
from core.nervous_system.policy.reason_codes import ReasonCode
from core.nervous_system.policy.rules import concentration_vetoes
from core.nervous_system.portfolio.exposure import UNALLOCATED, calculate_exposure

from core.nervous_system.tests.test_portfolio_exposure import (
    DECISION_BAR,
    DECISION_TIME,
    REPORT_ID,
    TICKER,
    build_config,
    equity_position,
    portfolio_state,
    sector_state,
    theme_membership,
    ticker_state,
)


UTC = timezone.utc
THEME = "ai_infrastructure"


def peer_membership(
    *,
    ticker: str,
    theme_id: str = THEME,
    weight: float = 1.0,
    available_at: datetime | None = None,
) -> ThemeMembership:
    as_of = datetime(2026, 7, 29, 21, 0, tzinfo=UTC)
    resolved_available = available_at or datetime(2026, 7, 30, 2, 0, tzinfo=UTC)
    return ThemeMembership(
        state_id=uuid5(NAMESPACE_URL, f"peer-test/membership/{ticker}/{theme_id}"),
        state_type=StateType.THEME_MEMBERSHIP,
        entity_id=theme_id,
        as_of=as_of,
        available_at=resolved_available,
        generated_at=as_of,
        valid_until=resolved_available + timedelta(days=2),
        source_window_start=as_of - timedelta(minutes=5),
        source_window_end=as_of,
        schema_version=1,
        producer="peer-test@1",
        model_version="peer-test-model@1",
        feature_version="peer-test-features@1",
        config_version="peer-test-config@1",
        lineage_ids=(f"peer-test:{ticker}:{theme_id}",),
        data_quality=DataQualitySummary(),
        ticker=ticker,
        theme_id=theme_id,
        weight=weight,
        membership_version="themes@4",
        effective_from=as_of,
        effective_until=None,
    )


def build_snapshot(*, states=(), peers=()) -> ContextSnapshot:
    return ContextSnapshot.from_states(
        snapshot_id=uuid5(NAMESPACE_URL, "peer-test/snapshot"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker=TICKER,
        states=states or (sector_state(), ticker_state()),
        peer_theme_memberships=peers,
        freshness_profile="meta_4h_1420@1",
        freshness_profile_hash="c" * 64,
        decision_bar=DECISION_BAR,
        decision_session="2026-07-30",
    )


# ---------------------------------------------------------------------------
# The snapshot contract
# ---------------------------------------------------------------------------


def test_empty_peers_leave_the_snapshot_hash_byte_identical() -> None:
    """Meta's replay parity depends on this: peers must be purely additive."""

    without = build_snapshot()
    with_empty = build_snapshot(peers=())
    assert without.content_hash == with_empty.content_hash
    assert without.state_hashes == with_empty.state_hashes
    assert without.peer_theme_memberships == ()


def test_adding_a_peer_changes_the_snapshot_hash() -> None:
    baseline = build_snapshot()
    with_peer = build_snapshot(peers=(peer_membership(ticker="NVDA"),))
    assert with_peer.content_hash != baseline.content_hash
    # The peer is hashed as a real embedded state, not as loose metadata.
    assert len(with_peer.state_hashes) == len(baseline.state_hashes) + 1
    peer_ids = {m.state_id for m in with_peer.peer_theme_memberships}
    assert peer_ids <= set(with_peer.state_ids)


def test_a_peer_naming_the_context_ticker_is_rejected() -> None:
    """Admitting it twice would double-count the decision ticker's notional."""

    with pytest.raises(ValueError, match="duplicates the snapshot ticker"):
        build_snapshot(peers=(peer_membership(ticker=TICKER),))


def test_a_peer_unavailable_at_decision_time_is_rejected() -> None:
    """Peers are held to the same causal bar as the context ticker's own state."""

    future = DECISION_TIME + timedelta(hours=1)
    with pytest.raises(ValueError, match="unavailable at decision time"):
        build_snapshot(peers=(peer_membership(ticker="NVDA", available_at=future),))


def test_duplicate_peer_ticker_theme_pairs_are_rejected() -> None:
    peer = peer_membership(ticker="NVDA")
    with pytest.raises(ValueError, match="duplicate"):
        build_snapshot(peers=(peer, peer))


def test_peer_quality_issues_do_not_enter_the_decision_quality_summary() -> None:
    """A stale holding elsewhere must not veto THIS ticker via data quality."""

    snapshot = build_snapshot(peers=(peer_membership(ticker="NVDA"),))
    # The peer carries a distinct config_version; if peers fed the version
    # accounting the snapshot would report MIXED.
    assert not snapshot.config_version.startswith("MIXED")


# ---------------------------------------------------------------------------
# Exposure: the bucketing that was structurally impossible before
# ---------------------------------------------------------------------------


def test_peer_positions_bucket_to_their_own_theme() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=100, market_value=20_000.0),
            equity_position(symbol="NVDA", quantity=100, market_value=30_000.0),
        )
    )
    snapshot = build_snapshot(
        states=(sector_state(), ticker_state(), theme_membership(theme_id=THEME, weight=1.0)),
        peers=(peer_membership(ticker="NVDA"),),
    )
    report = calculate_exposure(
        portfolio, snapshot, config=build_config(), report_id=REPORT_ID
    )
    # Both names now land in the theme; previously NVDA was UNALLOCATED.
    assert report.theme_notional[THEME] == Decimal("50000.00")
    assert UNALLOCATED not in report.theme_notional


def test_a_position_with_no_carried_membership_stays_unallocated() -> None:
    """Unknown theme is reported as unknown, never guessed."""

    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=100, market_value=20_000.0),
            equity_position(symbol="JPM", quantity=100, market_value=30_000.0),
        )
    )
    snapshot = build_snapshot(
        states=(sector_state(), ticker_state(), theme_membership(theme_id=THEME, weight=1.0)),
    )
    report = calculate_exposure(
        portfolio, snapshot, config=build_config(), report_id=REPORT_ID
    )
    assert report.theme_notional[THEME] == Decimal("20000.00")
    assert report.theme_notional[UNALLOCATED] == Decimal("30000.00")


# ---------------------------------------------------------------------------
# The policy veto
# ---------------------------------------------------------------------------


def build_policy_config(**overrides):
    from core.nervous_system.tests.test_policy_engine import build_config as policy_config

    return policy_config(**overrides)


def build_policy_intent(snapshot, **overrides):
    from core.nervous_system.tests.test_policy_engine import build_intent

    return build_intent(snapshot=snapshot, **overrides)


def test_theme_limit_vetoes_only_once_peers_reveal_the_concentration() -> None:
    """The whole point: the same portfolio passes blind and fails sighted."""

    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="NVDA", quantity=100, market_value=30_000.0),
            equity_position(symbol="AVGO", quantity=100, market_value=25_000.0),
        )
    )
    states = (sector_state(), ticker_state(), theme_membership(theme_id=THEME, weight=1.0))
    config = build_policy_config(portfolio_exposure=MVP_PORTFOLIO_CONFIG)

    # Sized so only the THEME bucket differs between the two snapshots; the
    # symbol/sector/factor limits are asserted elsewhere and are not the claim.
    blind = build_snapshot(states=(*states, portfolio))
    blind_intent = build_policy_intent(blind, position_size_requested=Decimal("10000.00"))
    assert ReasonCode.PORTFOLIO_MAX_THEME_NOTIONAL_BREACH not in concentration_vetoes(
        blind_intent, blind, config
    )

    sighted = build_snapshot(
        states=(*states, portfolio),
        peers=(peer_membership(ticker="NVDA"), peer_membership(ticker="AVGO")),
    )
    sighted_intent = build_policy_intent(
        sighted, position_size_requested=Decimal("10000.00")
    )
    # 30k + 25k held in the theme, plus a 10k proposal, against a 50k limit.
    assert ReasonCode.PORTFOLIO_MAX_THEME_NOTIONAL_BREACH in concentration_vetoes(
        sighted_intent, sighted, config
    )


def test_concentration_gating_is_off_by_default() -> None:
    """An existing policy version keeps its exact behaviour and identity."""

    portfolio = portfolio_state(
        positions=(equity_position(symbol="NVDA", quantity=100, market_value=90_000.0),)
    )
    snapshot = build_snapshot(
        states=(sector_state(), ticker_state(), portfolio),
        peers=(peer_membership(ticker="NVDA"),),
    )
    config = build_policy_config()
    assert config.portfolio_exposure is None
    assert concentration_vetoes(build_policy_intent(snapshot), snapshot, config) == ()


def test_unallocated_exposure_never_vetoes() -> None:
    """A mapping gap is a data-quality problem, not concentration."""

    portfolio = portfolio_state(
        positions=(equity_position(symbol="JPM", quantity=100, market_value=90_000.0),)
    )
    snapshot = build_snapshot(states=(sector_state(), ticker_state(), portfolio))
    config = build_policy_config(portfolio_exposure=MVP_PORTFOLIO_CONFIG)
    vetoes = concentration_vetoes(build_policy_intent(snapshot), snapshot, config)
    assert ReasonCode.PORTFOLIO_MAX_THEME_NOTIONAL_BREACH not in vetoes


def test_sector_limit_counts_every_held_position() -> None:
    """Sector always worked portfolio-wide via the config map. Guard it."""

    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="NVDA", quantity=100, market_value=40_000.0),
            equity_position(symbol="AVGO", quantity=100, market_value=30_000.0),
        )
    )
    snapshot = build_snapshot(states=(sector_state(), ticker_state(), portfolio))
    config = build_policy_config(portfolio_exposure=MVP_PORTFOLIO_CONFIG)
    vetoes = concentration_vetoes(
        build_policy_intent(snapshot, position_size_requested=Decimal("5000.00")),
        snapshot,
        config,
    )
    assert ReasonCode.PORTFOLIO_MAX_SECTOR_NOTIONAL_BREACH in vetoes


def test_concentration_never_blocks_a_risk_reducing_exit() -> None:
    """Trapping an open position is worse than the concentration it avoids."""

    from core.nervous_system.policy.rules import HARD_RULES

    rule = next(r for r in HARD_RULES if r.rule_id == "policy.rule.concentration")
    assert rule.applies_to_risk_reducing is False


def test_concentration_reaches_a_real_policy_decision() -> None:
    """End-to-end: the rule must be REACHABLE, not merely registered.

    A rule can sit in HARD_RULES and still never fire if the evaluator returns
    before reaching it. This drives the whole evaluator.
    """

    from core.nervous_system.policy.engine import evaluate_policy, is_executable
    from core.nervous_system.tests.test_policy_engine import (
        market_state,
        position,
        readiness_state,
        ticker_state as policy_ticker,
    )

    held = (
        position(symbol="NVDA", market_value=30_000.0),
        position(symbol="AVGO", market_value=25_000.0),
    )
    snapshot = build_snapshot(
        states=(
            market_state(),
            policy_ticker(),
            readiness_state(),
            portfolio_state(positions=held),
            theme_membership(theme_id=THEME, weight=1.0),
        ),
        peers=(peer_membership(ticker="NVDA"), peer_membership(ticker="AVGO")),
    )
    intent = build_policy_intent(snapshot, position_size_requested=Decimal("10000.00"))

    ungated = evaluate_policy(intent, snapshot, build_policy_config())
    gated = evaluate_policy(
        intent, snapshot, build_policy_config(portfolio_exposure=MVP_PORTFOLIO_CONFIG)
    )

    assert ReasonCode.PORTFOLIO_MAX_THEME_NOTIONAL_BREACH.value not in ungated.hard_vetoes
    assert ReasonCode.PORTFOLIO_MAX_THEME_NOTIONAL_BREACH.value in gated.hard_vetoes
    assert not is_executable(gated)
    # Enabling the gate changes the policy version's identity, as it must.
    assert gated.policy_decision_id != ungated.policy_decision_id


# ---------------------------------------------------------------------------
# Observe mode
# ---------------------------------------------------------------------------


def test_observe_mode_records_the_breach_without_blocking() -> None:
    """The dry run PolicyMode.SHADOW cannot give you.

    Shadow selects the sizing basis and hard vetoes bind in every mode, so it
    answers "what would this gate refuse?" only by actually refusing it.
    """

    from core.nervous_system.policy.engine import evaluate_policy, is_executable
    from core.nervous_system.tests.test_policy_engine import (
        market_state,
        position,
        readiness_state,
        ticker_state as policy_ticker,
    )

    held = (
        position(symbol="NVDA", market_value=30_000.0),
        position(symbol="AVGO", market_value=25_000.0),
    )
    snapshot = build_snapshot(
        states=(
            market_state(),
            policy_ticker(),
            readiness_state(),
            portfolio_state(positions=held),
            theme_membership(theme_id=THEME, weight=1.0),
        ),
        peers=(peer_membership(ticker="NVDA"), peer_membership(ticker="AVGO")),
    )
    intent = build_policy_intent(snapshot, position_size_requested=Decimal("10000.00"))

    enforced = evaluate_policy(
        intent, snapshot, build_policy_config(portfolio_exposure=MVP_PORTFOLIO_CONFIG)
    )
    observed = evaluate_policy(
        intent,
        snapshot,
        build_policy_config(
            portfolio_exposure=MVP_PORTFOLIO_CONFIG, concentration_observe_only=True
        ),
    )

    # Enforce blocks.
    assert ReasonCode.PORTFOLIO_MAX_THEME_NOTIONAL_BREACH.value in enforced.hard_vetoes
    assert not is_executable(enforced)

    # Observe records and lets it through.
    assert observed.hard_vetoes == ()
    assert ReasonCode.PORTFOLIO_CONCENTRATION_OBSERVED.value in observed.reason_codes
    assert is_executable(observed)


def test_observe_mode_stays_silent_when_nothing_would_breach() -> None:
    from core.nervous_system.policy.engine import evaluate_policy
    from core.nervous_system.tests.test_policy_engine import (
        market_state,
        readiness_state,
        ticker_state as policy_ticker,
    )

    snapshot = build_snapshot(
        states=(
            market_state(),
            policy_ticker(),
            readiness_state(),
            portfolio_state(positions=()),
        ),
    )
    intent = build_policy_intent(snapshot, position_size_requested=Decimal("100.00"))
    decision = evaluate_policy(
        intent,
        snapshot,
        build_policy_config(
            portfolio_exposure=MVP_PORTFOLIO_CONFIG, concentration_observe_only=True
        ),
    )
    assert ReasonCode.PORTFOLIO_CONCENTRATION_OBSERVED.value not in decision.reason_codes


def test_observe_and_enforce_never_both_report() -> None:
    """A breach is a veto or a note, never counted twice."""

    from core.nervous_system.policy.rules import concentration_notes, concentration_vetoes
    from core.nervous_system.tests.test_policy_engine import (
        market_state,
        position,
        readiness_state,
        ticker_state as policy_ticker,
    )

    snapshot = build_snapshot(
        states=(
            market_state(),
            policy_ticker(),
            readiness_state(),
            portfolio_state(positions=(position(symbol="NVDA", market_value=90_000.0),)),
            theme_membership(theme_id=THEME, weight=1.0),
        ),
        peers=(peer_membership(ticker="NVDA"),),
    )
    intent = build_policy_intent(snapshot, position_size_requested=Decimal("10000.00"))

    enforce = build_policy_config(portfolio_exposure=MVP_PORTFOLIO_CONFIG)
    observe = build_policy_config(
        portfolio_exposure=MVP_PORTFOLIO_CONFIG, concentration_observe_only=True
    )
    assert concentration_vetoes(intent, snapshot, enforce)
    assert concentration_notes(intent, snapshot, enforce) == ()
    assert concentration_vetoes(intent, snapshot, observe) == ()
    assert concentration_notes(intent, snapshot, observe)


def test_observe_only_without_limits_is_refused() -> None:
    """There is nothing to observe without limits to evaluate."""

    with pytest.raises(ValueError, match="requires portfolio_exposure"):
        build_policy_config(concentration_observe_only=True, portfolio_exposure=None)
