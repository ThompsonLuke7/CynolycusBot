"""Regression: a TTL expiry must price the setup off its OWN tape.

The live log for 2026-08-24T13:07:00Z contains GS (a ~$1,040 stock), APTV
(~$50) and FIVE (~$250) all closing at ``spot=133.45`` -- the price of whichever
symbol's bar happened to arrive and trigger the global TTL sweep. Harmless while
nothing prices off that field; corrupting the moment a ledger reads it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction, SetupState


NOW = datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc)


def _engine() -> IntradayStructureEngine:
    config = IntradayStructureConfig(
        enabled=True, min_average_dollar_volume=0.0, candidate_ttl_minutes=60,
    )
    return IntradayStructureEngine(config)


def test_an_expiring_setup_is_priced_off_its_own_last_bar_not_the_arriving_one() -> None:
    engine = _engine()
    # GS is stale: its last observed print was 1040.00, an hour ago.
    engine.register_candidate(Candidate("GS", NOW, Direction.LONG, ("dealer_ranker",), score=0.7))
    engine.on_bar(Bar("GS", NOW, 1040.0, 1041.0, 1039.0, 1040.0, 5_000))

    # IREN keeps printing, so ITS bar is what drives the global TTL sweep.
    engine.register_candidate(
        Candidate("IREN", NOW + timedelta(minutes=61), Direction.LONG, ("momentum",), score=0.7)
    )
    driver = Bar("IREN", NOW + timedelta(minutes=61), 133.0, 133.9, 132.8, 133.45, 9_000)
    engine.on_bar(driver)

    closes = [t for t in engine.transitions if t.ticker == "GS" and t.to_state == SetupState.CLOSED]
    assert closes, "the stale GS candidate should have expired"
    for transition in closes:
        assert transition.spot == 1040.0, "GS must not be priced at IREN's 133.45"
        # Decision time is now; the price is an hour old, and the record says so.
        assert transition.timestamp == driver.timestamp
        assert transition.spot_as_of == NOW


def test_a_ticker_that_never_printed_gets_no_invented_price() -> None:
    engine = _engine()
    engine.register_candidate(Candidate("QUIET", NOW, Direction.LONG, ("manual",), score=0.7))
    engine.register_candidate(
        Candidate("IREN", NOW + timedelta(minutes=61), Direction.LONG, ("momentum",), score=0.7)
    )
    engine.on_bar(Bar("IREN", NOW + timedelta(minutes=61), 133.0, 133.9, 132.8, 133.45, 9_000))

    closes = [t for t in engine.transitions if t.ticker == "QUIET" and t.to_state == SetupState.CLOSED]
    assert closes
    for transition in closes:
        assert transition.spot is None
        assert transition.spot_as_of is None


def test_a_price_driven_transition_still_reports_a_fresh_price() -> None:
    engine = _engine()
    engine.register_candidate(Candidate("XYZ", NOW, Direction.LONG, ("manual",), score=0.7))
    bar = Bar("XYZ", NOW, 10.0, 10.2, 9.9, 10.1, 1_000)
    engine.on_bar(bar)
    engine.on_bar(Bar("XYZ", NOW + timedelta(minutes=1), 0.5, 0.6, 0.4, 0.5, 1_000))

    closes = [t for t in engine.transitions if t.to_state == SetupState.CLOSED]
    assert closes, "the sub-minimum-price bar should close the setup"
    for transition in closes:
        assert transition.spot == 0.5
        assert transition.spot_as_of == transition.timestamp
