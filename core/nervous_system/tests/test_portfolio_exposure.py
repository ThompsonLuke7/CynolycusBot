"""Canonical portfolio exposure tests (Task 16)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from core.nervous_system.config.portfolio import (
    MVP_PORTFOLIO_CONFIG,
    PortfolioConfig,
)
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import (
    AssetClass,
    DataQualitySeverity,
    Direction,
    StateType,
    ThemeRegime,
    TickerSetup,
)
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import (
    PortfolioPosition,
    PortfolioState,
    SectorState,
    StateContract,
    ThemeMembership,
    ThemeState,
    TickerState,
)
from core.nervous_system.portfolio.exposure import (
    UNALLOCATED,
    calculate_exposure,
)


UTC = timezone.utc
TICKER = "AMD"
DECISION_TIME = datetime(2026, 7, 30, 18, 20, tzinfo=UTC)
DECISION_BAR = datetime(2026, 7, 30, 18, 0, tzinfo=UTC)
REPORT_ID = uuid5(NAMESPACE_URL, "portfolio-test/report")


def _envelope(
    *,
    state_id: UUID,
    state_type: StateType,
    entity_id: str,
    as_of: datetime,
    available_at: datetime,
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "state_type": state_type,
        "entity_id": entity_id,
        "as_of": as_of,
        "available_at": available_at,
        "generated_at": available_at,
        "valid_until": available_at + timedelta(days=2),
        "source_window_start": as_of - timedelta(minutes=5),
        "source_window_end": as_of,
        "schema_version": 1,
        "producer": "portfolio-test@1",
        "model_version": "portfolio-test-model@1",
        "feature_version": "portfolio-test-features@1",
        "config_version": "portfolio-test-config@1",
        "lineage_ids": (f"portfolio-test:{state_id}",),
        "data_quality": DataQualitySummary(),
    }


def equity_position(
    *,
    symbol: str,
    quantity: float,
    market_value: float,
    strategy_id: str | None = "meta_ranker",
) -> PortfolioPosition:
    return PortfolioPosition(
        broker_position_id=f"pos-{symbol}",
        symbol=symbol,
        underlying=symbol,
        asset_class=AssetClass.EQUITY,
        quantity=quantity,
        current_price=abs(market_value / quantity) if quantity else None,
        market_value=market_value,
        strategy_id=strategy_id,
        ownership_status="ASSIGNED" if strategy_id else "UNASSIGNED",
    )


def option_position(
    *,
    symbol: str = "AMD261218C00200000",
    underlying: str = "AMD",
    quantity: float = 5.0,
    market_value: float = 4_000.0,
    greeks: dict[str, float] | None = None,
) -> PortfolioPosition:
    return PortfolioPosition(
        broker_position_id=f"pos-{symbol}",
        symbol=symbol,
        underlying=underlying,
        asset_class=AssetClass.OPTION,
        quantity=quantity,
        market_value=market_value,
        strategy_id="meta_ranker",
        ownership_status="ASSIGNED",
        greeks=greeks if greeks is not None else {
            "delta": 0.55,
            "gamma": 0.01,
            "vega": 0.20,
            "theta": -0.08,
        },
    )


def portfolio_state(
    *,
    positions: tuple[PortfolioPosition, ...],
    account_alias: str = "paper",
    state_id: UUID | None = None,
) -> PortfolioState:
    payload = _envelope(
        state_id=state_id or uuid5(NAMESPACE_URL, "portfolio-test/portfolio"),
        state_type=StateType.PORTFOLIO,
        entity_id=account_alias,
        as_of=datetime(2026, 7, 30, 18, 15, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 18, 16, tzinfo=UTC),
    )
    payload.update(
        {
            "account_alias": account_alias,
            "equity": 250_000.0,
            "cash": 100_000.0,
            "buying_power": 200_000.0,
            "positions": positions,
            "broker_observed_at": datetime(2026, 7, 30, 18, 15, tzinfo=UTC),
        }
    )
    return PortfolioState(**payload)


def sector_state(*, sector_id: str = "XLK") -> SectorState:
    payload = _envelope(
        state_id=uuid5(NAMESPACE_URL, f"portfolio-test/sector/{sector_id}"),
        state_type=StateType.SECTOR,
        entity_id=sector_id,
        as_of=datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        available_at=datetime(2026, 7, 29, 20, 30, tzinfo=UTC),
    )
    payload.update({"sector_id": sector_id, "capital_flow_direction": Direction.UNKNOWN})
    return SectorState(**payload)


def theme_membership(*, theme_id: str, weight: float) -> ThemeMembership:
    as_of = datetime(2026, 7, 29, 21, 0, tzinfo=UTC)
    payload = _envelope(
        state_id=uuid5(NAMESPACE_URL, f"portfolio-test/membership/{theme_id}"),
        state_type=StateType.THEME_MEMBERSHIP,
        entity_id=theme_id,
        as_of=as_of,
        available_at=datetime(2026, 7, 30, 2, 0, tzinfo=UTC),
    )
    payload["generated_at"] = as_of
    payload.update(
        {
            "ticker": TICKER,
            "theme_id": theme_id,
            "weight": weight,
            "membership_version": "themes@4",
            "effective_from": as_of,
            "effective_until": None,
        }
    )
    return ThemeMembership(**payload)


def theme_state(*, theme_id: str) -> ThemeState:
    as_of = datetime(2026, 7, 29, 21, 0, tzinfo=UTC)
    payload = _envelope(
        state_id=uuid5(NAMESPACE_URL, f"portfolio-test/theme/{theme_id}"),
        state_type=StateType.THEME,
        entity_id=theme_id,
        as_of=as_of,
        available_at=datetime(2026, 7, 30, 2, 0, tzinfo=UTC),
    )
    payload["generated_at"] = as_of
    payload.update({"theme_id": theme_id, "theme_regime": ThemeRegime.HEALTHY})
    return ThemeState(**payload)


def ticker_state() -> TickerState:
    payload = _envelope(
        state_id=uuid5(NAMESPACE_URL, "portfolio-test/ticker"),
        state_type=StateType.TICKER,
        entity_id=TICKER,
        as_of=DECISION_BAR,
        available_at=DECISION_BAR + timedelta(minutes=5),
    )
    payload.update(
        {
            "ticker": TICKER,
            "selected_bar": DECISION_BAR,
            "reference_price": 200.0,
            "ticker_setup": TickerSetup.BREAKOUT,
        }
    )
    return TickerState(**payload)


def build_snapshot(*, states: tuple[StateContract, ...] | None = None) -> ContextSnapshot:
    if states is None:
        states = (sector_state(), ticker_state())
    return ContextSnapshot.from_states(
        snapshot_id=uuid5(NAMESPACE_URL, "portfolio-test/snapshot"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker=TICKER,
        states=states,
        freshness_profile="meta_4h_1420@1",
        freshness_profile_hash="c" * 64,
        decision_bar=DECISION_BAR,
        decision_session="2026-07-30",
    )


def build_config(**overrides: Any) -> PortfolioConfig:
    return replace(MVP_PORTFOLIO_CONFIG, **overrides) if overrides else MVP_PORTFOLIO_CONFIG


def run(
    portfolio: PortfolioState,
    snapshot: ContextSnapshot | None = None,
    *,
    config: PortfolioConfig | None = None,
    proposed_position: PortfolioPosition | None = None,
):
    return calculate_exposure(
        portfolio,
        snapshot if snapshot is not None else build_snapshot(),
        config=config or build_config(),
        report_id=REPORT_ID,
        proposed_position=proposed_position,
    )


# --------------------------------------------------------------------------
# Step 1: notional aggregates
# --------------------------------------------------------------------------


def test_gross_net_long_short_and_symbol_notional() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),
            equity_position(symbol="NVDA", quantity=-50.0, market_value=-8_000.0),
            equity_position(symbol="JPM", quantity=10.0, market_value=2_000.0),
        )
    )

    report = run(portfolio)

    assert report.long_notional == Decimal("22000.00")
    assert report.short_notional == Decimal("8000.00")
    assert report.gross_notional == Decimal("30000.00")
    assert report.net_notional == Decimal("14000.00")
    assert report.symbol_notional["AMD"] == Decimal("20000.00")
    assert report.symbol_notional["NVDA"] == Decimal("-8000.00")
    assert report.symbol_notional["JPM"] == Decimal("2000.00")
    assert report.portfolio_state_id == portfolio.state_id
    assert report.calculated_at == portfolio.as_of
    assert report.content_hash == report.computed_content_hash()


def test_sector_notional_uses_the_canonical_sector_state() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),
            equity_position(symbol="JPM", quantity=10.0, market_value=2_000.0),
        )
    )
    snapshot = build_snapshot(states=(sector_state(sector_id="XLK"), ticker_state()))

    report = run(portfolio, snapshot)

    # XLK is a canonical sector state; XLF is mapped but has no state, so it is
    # reported separately rather than silently folded into XLK.
    assert report.sector_notional["XLK"] == Decimal("20000.00")
    assert report.sector_notional["XLF"] == Decimal("2000.00")


def test_unmapped_ticker_goes_to_the_unallocated_sector_bucket() -> None:
    portfolio = portfolio_state(
        positions=(equity_position(symbol="ZZZZ", quantity=10.0, market_value=5_000.0),)
    )

    report = run(portfolio)

    assert report.sector_notional[UNALLOCATED] == Decimal("5000.00")
    assert "ZZZZ" not in report.sector_notional
    assert any(
        issue.code == "SECTOR_UNMAPPED" for issue in report.quality.issues
    )


def test_theme_weights_are_normalised_and_never_double_count() -> None:
    portfolio = portfolio_state(
        positions=(equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),)
    )
    snapshot = build_snapshot(
        states=(
            sector_state(),
            ticker_state(),
            theme_membership(theme_id="ai_infra", weight=0.75),
            theme_membership(theme_id="semis", weight=0.25),
            theme_state(theme_id="ai_infra"),
            theme_state(theme_id="semis"),
        )
    )

    report = run(portfolio, snapshot)

    assert report.theme_notional["ai_infra"] == Decimal("15000.00")
    assert report.theme_notional["semis"] == Decimal("5000.00")
    total = sum(report.theme_notional.values())
    assert total == Decimal("20000.00"), "full notional must not land in every theme"


def test_theme_allocation_ignores_non_positive_weights() -> None:
    portfolio = portfolio_state(
        positions=(equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),)
    )
    snapshot = build_snapshot(
        states=(
            sector_state(),
            ticker_state(),
            theme_membership(theme_id="ai_infra", weight=1.0),
            theme_membership(theme_id="dead_theme", weight=0.0),
            theme_state(theme_id="ai_infra"),
        )
    )

    report = run(portfolio, snapshot)

    assert report.theme_notional["ai_infra"] == Decimal("20000.00")
    assert "dead_theme" not in report.theme_notional


def test_positions_without_memberships_stay_in_the_unallocated_theme_bucket() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),
            equity_position(symbol="JPM", quantity=10.0, market_value=2_000.0),
        )
    )
    snapshot = build_snapshot(
        states=(
            sector_state(),
            ticker_state(),
            theme_membership(theme_id="ai_infra", weight=1.0),
            theme_state(theme_id="ai_infra"),
        )
    )

    report = run(portfolio, snapshot)

    assert report.theme_notional["ai_infra"] == Decimal("20000.00")
    assert report.theme_notional[UNALLOCATED] == Decimal("2000.00")


def test_correlated_factor_tags_expose_overlap() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),
            equity_position(symbol="SOXL", quantity=100.0, market_value=5_000.0),
            equity_position(symbol="TQQQ", quantity=100.0, market_value=7_000.0),
            equity_position(symbol="NBIS", quantity=100.0, market_value=3_000.0),
            equity_position(symbol="APLD", quantity=100.0, market_value=1_000.0),
            equity_position(symbol="IREN", quantity=100.0, market_value=2_000.0),
        )
    )

    report = run(portfolio)

    # AMD and SOXL are both semiconductor; SOXL and TQQQ are both leveraged
    # index; AMD/NBIS/APLD/IREN are all AI infrastructure.  The same dollar is
    # deliberately visible in more than one factor, because that is the risk.
    assert report.factor_notional["semiconductor"] == Decimal("25000.00")
    assert report.factor_notional["leveraged_index"] == Decimal("12000.00")
    assert report.factor_notional["ai_infrastructure"] == Decimal("26000.00")


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


def test_option_greeks_aggregate_when_available() -> None:
    portfolio = portfolio_state(
        positions=(
            option_position(quantity=5.0, market_value=4_000.0),
            equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),
        )
    )

    report = run(portfolio)

    # 5 contracts x 100 multiplier: delta 275, gamma 5, vega 100, theta -40.
    assert report.option_greeks["delta"] == Decimal("275.00")
    assert report.option_greeks["gamma"] == Decimal("5.00")
    assert report.option_greeks["vega"] == Decimal("100.00")
    assert report.option_greeks["theta"] == Decimal("-40.00")


def test_underlying_equivalent_combines_shares_and_delta_shares() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),
            option_position(quantity=5.0, market_value=4_000.0),
        )
    )

    report = run(portfolio)

    # 100 shares + (0.55 x 100 x 5) delta-equivalent shares.
    assert report.underlying_equivalent["AMD"] == Decimal("375.00")


def test_missing_greeks_flag_unknown_exposure_conservatively() -> None:
    portfolio = portfolio_state(
        positions=(option_position(quantity=5.0, market_value=4_000.0, greeks={}),)
    )

    report = run(portfolio)

    assert report.has_unknown_exposure
    issues = {issue.code: issue for issue in report.quality.issues}
    assert "OPTION_GREEKS_UNKNOWN" in issues
    assert issues["OPTION_GREEKS_UNKNOWN"].severity is DataQualitySeverity.ERROR
    # The option must not silently contribute zero delta-equivalent exposure.
    assert "AMD" not in report.underlying_equivalent
    breached = {result.limit_id for result in report.limit_results if result.breached}
    assert "exposure.known" in breached


def test_partial_greeks_are_not_imputed() -> None:
    portfolio = portfolio_state(
        positions=(
            option_position(quantity=5.0, market_value=4_000.0, greeks={"gamma": 0.01}),
        )
    )

    report = run(portfolio)

    assert report.has_unknown_exposure
    assert "delta" not in report.option_greeks
    assert report.option_greeks["gamma"] == Decimal("5.00")


def test_missing_market_value_flags_unknown_notional() -> None:
    position = PortfolioPosition(
        broker_position_id="pos-AMD",
        symbol="AMD",
        underlying="AMD",
        asset_class=AssetClass.EQUITY,
        quantity=100.0,
        market_value=None,
    )
    report = run(portfolio_state(positions=(position,)))

    assert report.has_unknown_exposure
    assert any(
        issue.code == "POSITION_NOTIONAL_UNKNOWN" for issue in report.quality.issues
    )
    assert "AMD" not in report.symbol_notional


def test_option_contract_multiplier_override_is_respected() -> None:
    position = option_position(quantity=5.0, market_value=4_000.0)
    position = position.model_copy(update={"contract_multiplier": 10.0})

    report = run(portfolio_state(positions=(position,)))

    assert report.option_greeks["delta"] == Decimal("27.50")


# --------------------------------------------------------------------------
# Proposed order and limits
# --------------------------------------------------------------------------


def test_proposed_order_incremental_exposure_and_post_trade_limits() -> None:
    portfolio = portfolio_state(
        positions=(equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),)
    )
    proposed = equity_position(symbol="AMD", quantity=40.0, market_value=8_000.0)

    report = run(portfolio, proposed_position=proposed)

    assert report.proposed_incremental_exposure["GROSS"] == Decimal("8000.00")
    assert report.proposed_incremental_exposure["NET"] == Decimal("8000.00")
    assert report.proposed_incremental_exposure["SYMBOL:AMD"] == Decimal("8000.00")
    assert report.proposed_incremental_exposure["SECTOR:XLK"] == Decimal("8000.00")
    assert report.proposed_incremental_exposure["FACTOR:semiconductor"] == Decimal("8000.00")

    symbol_limit = next(
        result
        for result in report.limit_results
        if result.scope == "SYMBOL" and result.scope_id == "AMD"
    )
    # Post-trade 28,000 exceeds the 25,000 symbol limit even though the
    # existing 20,000 position does not.
    assert symbol_limit.observed == Decimal("28000.00")
    assert symbol_limit.breached is True


def test_limits_are_not_breached_by_a_compliant_portfolio() -> None:
    portfolio = portfolio_state(
        positions=(equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),)
    )

    report = run(portfolio)

    assert all(result.breached is False for result in report.limit_results)
    assert report.proposed_incremental_exposure == {}


def test_gross_limit_breach_is_reported() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=1000.0, market_value=100_000.0),
            equity_position(symbol="JPM", quantity=1000.0, market_value=60_000.0),
        )
    )

    report = run(portfolio)

    gross = next(result for result in report.limit_results if result.scope == "GROSS")
    assert gross.observed == Decimal("160000.00")
    assert gross.breached is True


def test_short_positions_consume_gross_limit_by_magnitude() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=-1000.0, market_value=-100_000.0),
            equity_position(symbol="JPM", quantity=1000.0, market_value=60_000.0),
        )
    )

    report = run(portfolio)

    gross = next(result for result in report.limit_results if result.scope == "GROSS")
    assert gross.observed == Decimal("160000.00")
    assert gross.breached is True
    assert report.net_notional == Decimal("-40000.00")


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_exposure_is_deterministic_and_content_addressed() -> None:
    portfolio = portfolio_state(
        positions=(
            equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0),
            option_position(),
        )
    )
    snapshot = build_snapshot()

    first = run(portfolio, snapshot)
    second = run(portfolio, snapshot)

    assert first.content_hash == second.content_hash
    assert first == second


def test_position_order_does_not_change_the_report() -> None:
    a = equity_position(symbol="AMD", quantity=100.0, market_value=20_000.0)
    b = equity_position(symbol="JPM", quantity=10.0, market_value=2_000.0)

    forward = run(portfolio_state(positions=(a, b)))
    reverse = run(portfolio_state(positions=(b, a)))

    assert forward.content_hash == reverse.content_hash


def test_empty_portfolio_produces_a_zero_report() -> None:
    report = run(portfolio_state(positions=()))

    assert report.gross_notional == Decimal("0.00")
    assert report.net_notional == Decimal("0.00")
    assert report.symbol_notional == {}
    assert report.has_unknown_exposure is False


def test_config_hash_changes_with_limits_and_tags() -> None:
    base = build_config()
    variants = (
        build_config(max_gross_notional=Decimal("1.00")),
        build_config(sector_map={"AMD": "XLY"}),
        build_config(factor_tags={"solo": frozenset({"AMD"})}),
    )
    hashes = {base.content_hash} | {variant.content_hash for variant in variants}
    assert len(hashes) == len(variants) + 1


def test_config_rejects_silent_sector_fallback() -> None:
    config = build_config()
    assert config.sector_for("AMD") == "XLK"
    assert config.sector_for("NOT_A_TICKER") is None
