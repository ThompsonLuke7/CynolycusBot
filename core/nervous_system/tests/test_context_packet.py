"""The structured decision packet, and the peer-group state it can show.

The packet changes how evidence READS, never what it says: it is derived from
one snapshot with no IO, no clock, and no model, so an archived snapshot
renders identically years later.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.context.packet import (
    format_context_packet,
    render_context_packet,
)
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import StateType
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import PeerGroupState

from core.nervous_system.tests.test_peer_exposure_concentration import (
    THEME,
    build_snapshot,
    peer_membership,
)
from core.nervous_system.tests.test_portfolio_exposure import (
    DECISION_BAR,
    DECISION_TIME,
    TICKER,
    sector_state,
    theme_membership,
    theme_state,
    ticker_state,
)


UTC = timezone.utc


def peer_group(
    *,
    group_id: str = "corr20260729g001",
    members: tuple[str, ...] = ("AMD", "AVGO", "NVDA"),
    cohesion: float | None = 0.41,
) -> PeerGroupState:
    as_of = datetime(2026, 7, 29, 20, 0, tzinfo=UTC)
    return PeerGroupState(
        state_id=uuid5(NAMESPACE_URL, f"packet-test/group/{group_id}"),
        state_type=StateType.PEER_GROUP,
        entity_id=group_id,
        as_of=as_of,
        available_at=as_of,
        generated_at=as_of,
        valid_until=as_of + timedelta(days=9),
        source_window_start=as_of - timedelta(days=90),
        source_window_end=as_of,
        schema_version=1,
        producer="packet-test@1",
        model_version="peer-corr@1",
        feature_version="peer-corr-features@1",
        config_version="peer-corr-config@1",
        lineage_ids=("packet-test:bars",),
        data_quality=DataQualitySummary(),
        group_id=group_id,
        method="trailing_correlation_kmeans",
        members=members,
        lookback_sessions=60,
        trailing_cohesion=cohesion,
        universe_version="snap@1",
    )


# ---------------------------------------------------------------------------
# PeerGroupState contract
# ---------------------------------------------------------------------------


def test_members_must_be_sorted_for_a_stable_hash() -> None:
    with pytest.raises(ValueError, match="sorted"):
        peer_group(members=("NVDA", "AMD"))


def test_members_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        peer_group(members=("AMD", "AMD"))


def test_member_weights_may_not_name_a_non_member() -> None:
    payload = peer_group().model_dump()
    payload["member_weights"] = {"TSLA": 1.0}
    with pytest.raises(ValueError, match="non-members"):
        PeerGroupState.model_validate(payload)


def test_an_empty_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        peer_group(members=())


# ---------------------------------------------------------------------------
# The packet
# ---------------------------------------------------------------------------


def full_snapshot() -> ContextSnapshot:
    return ContextSnapshot.from_states(
        snapshot_id=uuid5(NAMESPACE_URL, "packet-test/snapshot"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker=TICKER,
        states=(
            sector_state(),
            ticker_state(),
            theme_state(theme_id=THEME),
            theme_membership(theme_id=THEME, weight=1.0),
            peer_group(),
        ),
        peer_theme_memberships=(
            peer_membership(ticker="NVDA"),
            peer_membership(ticker="AVGO"),
        ),
        freshness_profile="meta_4h_1420@1",
        freshness_profile_hash="c" * 64,
        decision_bar=DECISION_BAR,
        decision_session="2026-07-30",
    )


def test_packet_is_deterministic() -> None:
    """Same snapshot in, byte-identical packet out -- no clock, no IO."""

    snapshot = full_snapshot()
    assert render_context_packet(snapshot) == render_context_packet(snapshot)
    assert format_context_packet(snapshot) == format_context_packet(snapshot)


def test_packet_surfaces_the_lateral_axis() -> None:
    packet = render_context_packet(full_snapshot())
    assert sorted(packet["peers"][THEME]) == ["AVGO", "NVDA"]
    assert packet["peer_groups"][0]["method"] == "trailing_correlation_kmeans"
    assert packet["peer_groups"][0]["member_count"] == 3


def test_packet_carries_the_evidence_a_reviewer_needs() -> None:
    packet = render_context_packet(full_snapshot())
    evidence = packet["evidence"]
    assert evidence["freshness_profile"] == "meta_4h_1420@1"
    assert "stale_inputs" in evidence and "missing_inputs" in evidence
    assert packet["decision"]["content_hash"]


def test_text_rendering_names_every_level_present() -> None:
    text = format_context_packet(full_snapshot())
    for heading in (
        "MARKET REGIME",
        "SECTOR",
        "THEME",
        "PEER STRUCTURE",
        "TICKER",
        "EVIDENCE",
    ):
        assert heading in text, heading
    assert TICKER in text


def test_packet_survives_a_sparse_snapshot() -> None:
    """A snapshot missing optional states must render, not explode."""

    sparse = build_snapshot()
    packet = render_context_packet(sparse)
    assert packet["dealer"] is None
    assert packet["themes"] == []
    assert "EVIDENCE" in format_context_packet(sparse)
