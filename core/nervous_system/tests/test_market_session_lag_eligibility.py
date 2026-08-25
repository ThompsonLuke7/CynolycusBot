"""A session-lagged MARKET state must be usable by a pre-open flush.

The pre-open flush carries every deferred entry, and every deferred entry
carries the after-close 18:00Z bar. The profile's market_session_lag=1 then
demands that session's MARKET state — which is stamped at the 16:00 ET close,
20:00Z, two hours AFTER the bar.

Applying the bar bound to it as well left no row able to satisfy both gates:

    MARKET   MARKET_SESSION_MISMATCH x16     (every other session)
    MARKET   FUTURE_BAR              x2      (the right session, stamped later)

so no pre-open flush could ever succeed. Meta's queue did not drain once between
2026-08-18 and 2026-08-24.

The bar bound still applies to everything built FROM the decision bar, and to
MARKET when the profile has no session lag.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.config.freshness import get_snapshot_profile
from core.nervous_system.context.requirements import evaluate_requirements
from core.nervous_system.contracts.enums import MarketRegime, StateType
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import MarketState, TickerState
from core.nervous_system.contracts.enums import TickerSetup

UTC = timezone.utc
PROFILE = get_snapshot_profile("meta_4h_1420@1")

# Friday's after-close 4H bar — what every deferred entry carries.
FRIDAY_BAR = datetime(2026, 8, 21, 18, 0, tzinfo=UTC)
# Friday's 16:00 ET session close, when the regime tables are stamped.
FRIDAY_CLOSE = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
# Monday's 09:35 ET pre-open flush.
MONDAY_FLUSH = datetime(2026, 8, 24, 13, 35, tzinfo=UTC)


def _market(as_of: datetime) -> MarketState:
    return MarketState(
        state_id=uuid5(NAMESPACE_URL, f"market/{as_of.isoformat()}"),
        state_type=StateType.MARKET, entity_id="US", as_of=as_of,
        available_at=as_of + timedelta(minutes=30),
        generated_at=as_of + timedelta(minutes=30),
        valid_until=as_of + timedelta(hours=96),
        source_window_start=as_of, source_window_end=as_of,
        schema_version=1, producer="signals.market_regime", model_version="rules@1",
        feature_version="mr@1", config_version="mr@1",
        lineage_ids=("mr:US",), data_quality=DataQualitySummary(),
        regime=MarketRegime.NEUTRAL, metrics={"risk_appetite_z": 0.68},
        reason_codes=("TEST",),
    )


def _ticker(as_of: datetime) -> TickerState:
    return TickerState(
        state_id=uuid5(NAMESPACE_URL, f"ticker/{as_of.isoformat()}"),
        state_type=StateType.TICKER, entity_id="PSIG", as_of=as_of,
        available_at=as_of, generated_at=as_of,
        valid_until=as_of + timedelta(hours=6),
        source_window_start=as_of, source_window_end=as_of,
        schema_version=1, producer="meta", model_version="m@1",
        feature_version="f@1", config_version="c@1",
        lineage_ids=("m:PSIG",), data_quality=DataQualitySummary(),
        ticker="PSIG", selected_bar=as_of, reference_price=2.71,
        ticker_setup=TickerSetup.UNKNOWN, metrics={}, transition_probabilities={},
    )


def _verdict(candidates, state_type, *, bar, now):
    result = evaluate_requirements(
        candidates, entity_id="PSIG", decision_time=now, decision_bar=bar,
        profile=PROFILE,
    )
    return next(r for r in result.requirement_results if r.state_type is state_type)


def test_the_flush_can_use_the_lagged_sessions_market_state():
    verdict = _verdict([_market(FRIDAY_CLOSE)], StateType.MARKET,
                       bar=FRIDAY_BAR, now=MONDAY_FLUSH)
    assert verdict.status == "FRESH", verdict.reason_code


def test_a_wrong_session_is_still_refused():
    """Relaxing the bar bound must not relax the session gate."""
    thursday_close = datetime(2026, 8, 20, 20, 0, tzinfo=UTC)
    verdict = _verdict([_market(thursday_close)], StateType.MARKET,
                       bar=FRIDAY_BAR, now=MONDAY_FLUSH)
    assert verdict.status == "MISSING"
    assert verdict.reason_code == "MARKET_SESSION_MISMATCH"


def test_a_state_not_yet_available_at_the_decision_time_is_still_refused():
    """The causal bound that remains is available_at <= decision_time."""
    verdict = _verdict([_market(FRIDAY_CLOSE)], StateType.MARKET,
                       bar=FRIDAY_BAR,
                       now=datetime(2026, 8, 21, 20, 10, tzinfo=UTC))  # before it published
    assert verdict.status == "MISSING"


def test_the_bar_bound_still_binds_a_ticker_state():
    """TICKER is built FROM the decision bar; a later stamp saw future bars."""
    verdict = _verdict([_ticker(datetime(2026, 8, 21, 22, 0, tzinfo=UTC))],
                       StateType.TICKER, bar=FRIDAY_BAR, now=MONDAY_FLUSH)
    assert verdict.status == "MISSING"
    assert verdict.reason_code == "FUTURE_BAR"


def test_market_keeps_the_bar_bound_when_the_profile_has_no_session_lag():
    """Without a lag the state is same-session, so the bar bound is the only
    thing standing between it and a genuine look-ahead."""
    same_session = replace(PROFILE, market_session_lag=0)
    result = evaluate_requirements(
        [_market(FRIDAY_CLOSE)], entity_id="PSIG",
        decision_time=MONDAY_FLUSH, decision_bar=FRIDAY_BAR, profile=same_session,
    )
    verdict = next(r for r in result.requirement_results if r.state_type is StateType.MARKET)
    assert verdict.status == "MISSING"
    assert verdict.reason_code == "FUTURE_BAR"


def test_the_intraday_case_is_unchanged():
    """Friday 14:20 ET run: expects Thursday's state, stamped before the bar.
    This path always worked and must keep working identically."""
    friday_intraday_bar = datetime(2026, 8, 21, 14, 0, tzinfo=UTC)
    friday_1420 = datetime(2026, 8, 21, 18, 20, tzinfo=UTC)
    thursday_close = datetime(2026, 8, 20, 20, 0, tzinfo=UTC)
    verdict = _verdict([_market(thursday_close)], StateType.MARKET,
                       bar=friday_intraday_bar, now=friday_1420)
    assert verdict.status == "FRESH"
