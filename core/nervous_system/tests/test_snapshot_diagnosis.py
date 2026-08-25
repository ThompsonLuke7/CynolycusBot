"""A snapshot must be able to say why it is invalid.

Every snapshot already carried `requirement_results` and `rejected_candidates`.
Nothing read them, so a blocked order surfaced only as

    POLICY_VETO (SNAPSHOT_INVALID, SNAPSHOT_REQUIRED_STATE_MISSING)

a category true of five states for four different reasons. Meta's pre-open flush
was blocked 2026-08-18..24 and every failed snapshot in that window held the
answer — MARKET: MARKET_SESSION_MISMATCH x16, FUTURE_BAR x2 — in memory, unread.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid4, uuid5

from core.nervous_system.context.diagnosis import diagnose_snapshot
from core.nervous_system.contracts.context import (
    ContextSnapshot, FreshnessResult, RejectedCandidate,
)
from core.nervous_system.contracts.enums import StateType

UTC = timezone.utc
BAR = datetime(2026, 8, 21, 18, 0, tzinfo=UTC)
NOW = datetime(2026, 8, 24, 13, 35, 17, tzinfo=UTC)


def _result(state_type, *, required, status, reason="X"):
    return FreshnessResult(
        state_type=state_type, entity_id="PSIG", required=required, status=status,
        selected_state_id=None, age_seconds=None, max_age_seconds=3600.0,
        reason_code=reason,
    )


def _rejected(state_type, reason, n):
    return tuple(
        RejectedCandidate(state_id=uuid4(), state_type=state_type,
                          entity_id="US", reason_code=reason)
        for _ in range(n)
    )


def _snapshot(*, results, rejected=(), valid=False):
    return ContextSnapshot.from_states(
        snapshot_id=uuid5(NAMESPACE_URL, "diagnosis-test"),
        decision_time=NOW, strategy_id="meta_ranker", ticker="PSIG", states=(),
        freshness_profile="meta_4h_1420@1", freshness_profile_hash="c" * 64,
        decision_bar=BAR, decision_session="2026-08-24",
        requirement_results=results, rejected_candidates=rejected, valid=valid,
    )


def _the_real_failure():
    return _snapshot(
        results=(
            _result(StateType.MARKET, required=True, status="MISSING"),
            _result(StateType.TICKER, required=True, status="FRESH"),
            _result(StateType.SECTOR, required=True, status="FRESH"),
            _result(StateType.THEME, required=False, status="MISSING"),
        ),
        rejected=(
            _rejected(StateType.MARKET, "MARKET_SESSION_MISMATCH", 16)
            + _rejected(StateType.MARKET, "FUTURE_BAR", 2)
            + _rejected(StateType.SECTOR, "EXPIRED", 77)
        ),
    )


def test_it_names_the_blocking_state_and_why_its_candidates_were_refused():
    diagnosis = diagnose_snapshot(_the_real_failure())

    assert [item.state_type for item in diagnosis.blocking] == ["MARKET"]
    market = diagnosis.blocking[0]
    assert market.rejections == (("MARKET_SESSION_MISMATCH", 16), ("FUTURE_BAR", 2))
    assert "MARKET_SESSION_MISMATCH x16" in market.describe()
    assert "FUTURE_BAR x2" in market.describe()


def test_a_resolved_rule_shows_no_evidence():
    """SECTOR resolved. Its 77 aged-out rows are normal and printing them reads
    as a fault where there is none."""
    text = diagnose_snapshot(_the_real_failure()).describe()
    sector_line = next(l for l in text.splitlines() if "SECTOR" in l)
    assert "EXPIRED" not in sector_line
    assert sector_line.strip().endswith("FRESH")


def test_an_optional_state_is_never_blocking():
    diagnosis = diagnose_snapshot(_the_real_failure())
    assert "THEME" not in [item.state_type for item in diagnosis.blocking]


def test_nothing_published_reads_differently_from_candidates_refused():
    """A producer that never ran is a different problem from a selector that
    said no, and the line must not conflate them."""
    diagnosis = diagnose_snapshot(_snapshot(results=(
        _result(StateType.TICKER, required=True, status="MISSING"),
    )))
    assert "nothing published" in diagnosis.blocking[0].describe()


def test_the_header_carries_the_decision_parameters():
    """Which bar and which decision time is the first thing to check: the same
    states resolve differently for an intraday run and a pre-open flush."""
    text = diagnose_snapshot(_the_real_failure()).describe()
    header = text.splitlines()[0]
    assert "INVALID" in header
    assert "meta_ranker/PSIG" in header
    assert "bar=2026-08-21T18:00:00Z" in header
    assert "time=2026-08-24T13:35:17Z" in header


def test_a_valid_snapshot_says_so_and_blocks_nothing():
    diagnosis = diagnose_snapshot(_snapshot(
        results=(_result(StateType.MARKET, required=True, status="FRESH"),), valid=True,
    ))
    assert diagnosis.blocking == ()
    assert diagnosis.describe().startswith("snapshot VALID")


def test_rejection_ordering_is_stable():
    """Same failure renders identically every run, so it stays greppable."""
    first = diagnose_snapshot(_the_real_failure()).describe()
    second = diagnose_snapshot(_the_real_failure()).describe()
    assert first == second
