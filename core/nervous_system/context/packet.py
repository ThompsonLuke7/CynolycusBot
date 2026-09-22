"""Render a context snapshot as a structured decision packet.

The snapshot already holds everything a reader needs to reconstruct a
decision, but it holds it as a flat bag of contract objects.  A person asking
"why did we buy this?" has to join ten fields by hand.

This module is the cheap half of representation work: it does not change what
is measured or how anything is scored, it changes how the same evidence READS.
Everything here is derived from the snapshot alone -- no IO, no clock, no
model -- so a packet rendered today from an archived snapshot is identical to
the one rendered the day the decision was made.

Deliberately NOT a feature source.  Nothing in the scoring path should import
this; it exists for audit, review, and the daily report.
"""

from __future__ import annotations

from typing import Any

from core.nervous_system.contracts.context import ContextSnapshot


_UNKNOWN = "UNKNOWN"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_context_packet(snapshot: ContextSnapshot) -> dict[str, Any]:
    """Structure one snapshot as nested, JSON-safe evidence."""

    market = snapshot.market_state
    ticker_state = snapshot.ticker_state
    dealer = snapshot.dealer_state

    themes = [
        {
            "theme_id": state.theme_id,
            "regime": state.theme_regime.value,
            "relative_strength": state.relative_strength,
            "breadth": state.breadth,
            "momentum": state.momentum,
            "leadership_score": state.leadership_score,
            "crowding": state.crowding,
        }
        for state in snapshot.theme_states
    ]
    memberships = [
        {"theme_id": m.theme_id, "weight": m.weight} for m in snapshot.theme_memberships
    ]
    # Peers are the lateral axis: which OTHER holdings share this ticker's
    # grouping. Present for exposure, and worth showing for the same reason.
    peers: dict[str, list[str]] = {}
    for membership in snapshot.peer_theme_memberships:
        peers.setdefault(membership.theme_id, []).append(membership.ticker)

    groups = [
        {
            "group_id": group.group_id,
            "method": group.method,
            "member_count": len(group.members),
            "trailing_cohesion": group.trailing_cohesion,
            # Only the members that matter to this decision are listed; the
            # full group can run to dozens and belongs in the state, not here.
            "members_sample": list(group.members[:12]),
        }
        for group in snapshot.peer_groups
    ]

    return {
        "decision": {
            "ticker": snapshot.ticker,
            "strategy_id": snapshot.strategy_id,
            "decision_time": snapshot.decision_time.isoformat(),
            "decision_bar": (
                None if snapshot.decision_bar is None else snapshot.decision_bar.isoformat()
            ),
            "decision_session": snapshot.decision_session,
            "snapshot_id": str(snapshot.snapshot_id),
            "content_hash": snapshot.content_hash,
            "valid": snapshot.valid,
        },
        "market": None
        if market is None
        else {
            "regime": market.regime.value,
            "risk_on_probability": market.risk_on_probability,
            "risk_off_probability": market.risk_off_probability,
            "reason_codes": list(market.reason_codes),
        },
        "sectors": [
            {
                "sector_id": state.sector_id,
                "regime": state.sector_regime,
                "relative_strength": state.relative_strength,
                "breadth": state.breadth,
                "rotation_rank": state.rotation_rank,
                "capital_flow": state.capital_flow_direction.value,
            }
            for state in snapshot.sector_states
        ],
        "themes": themes,
        "memberships": memberships,
        "peers": {theme: sorted(names) for theme, names in sorted(peers.items())},
        "peer_groups": groups,
        "ticker": None
        if ticker_state is None
        else {
            "setup": ticker_state.ticker_setup.value,
            "reference_price": ticker_state.reference_price,
            "selected_bar": ticker_state.selected_bar.isoformat(),
            "trend_state": ticker_state.trend_state,
            "relative_strength_state": ticker_state.relative_strength_state,
            "volume_state": ticker_state.volume_state,
            "theme_alignment": ticker_state.theme_alignment,
            "market_alignment": ticker_state.market_alignment,
            "dealer_alignment": ticker_state.dealer_alignment,
        },
        "dealer": None
        if dealer is None
        else {
            "regime": dealer.dealer_regime.value,
            "spot": dealer.spot,
            "total_gex": dealer.total_gex,
            "call_wall": dealer.call_wall,
            "put_wall": dealer.put_wall,
            "gamma_flip": dealer.gamma_flip,
            "nearest_magnet": dealer.nearest_magnet,
            "pinning_score": dealer.pinning_score,
        },
        "catalysts": [
            {
                "event_type": event.event_type,
                "channel": event.channel,
                "source": event.source,
                "headline": event.headline,
                "observed_at": event.observed_at.isoformat(),
                "is_direct": event.is_direct,
                "relation_confidence": event.relation_confidence,
            }
            for event in snapshot.catalyst_events
        ],
        "catalyst_pressure": [
            {
                "scope": f"{pressure.scope_type}:{pressure.scope_id}",
                "aggregate_score": pressure.aggregate_score,
            }
            for pressure in snapshot.catalyst_pressures
        ],
        "evidence": {
            "freshness_profile": snapshot.freshness_profile,
            "stale_inputs": list(snapshot.stale_inputs),
            "missing_inputs": list(snapshot.missing_inputs),
            "rejected_reasons": sorted(set(snapshot.rejected_reasons)),
            "warnings": sorted(set(snapshot.warning_reasons)),
            "data_quality_usable": snapshot.data_quality.is_usable,
            "model_versions": list(snapshot.model_versions),
            "feature_versions": list(snapshot.feature_versions),
            "config_version": snapshot.config_version,
        },
    }


def format_context_packet(snapshot: ContextSnapshot) -> str:
    """Render the packet as the flat text an incident reviewer can skim."""

    packet = render_context_packet(snapshot)
    decision = packet["decision"]
    lines: list[str] = [
        f"Ticker: {decision['ticker']}   ({decision['strategy_id']})",
        f"Decision: {decision['decision_time']}  bar={decision['decision_bar']}"
        f"  session={decision['decision_session']}",
        "",
    ]

    market = packet["market"]
    lines.append("MARKET REGIME")
    lines.append(f"  {market['regime'] if market else _UNKNOWN}")
    if market and market["reason_codes"]:
        lines.append(f"  because: {', '.join(market['reason_codes'])}")

    if packet["sectors"]:
        lines.append("")
        lines.append("SECTOR")
        for sector in packet["sectors"]:
            lines.append(
                f"  {sector['sector_id']}: {sector['regime']}"
                f"  rs={_num(sector['relative_strength'])}"
                f"  breadth={_num(sector['breadth'])}"
            )

    if packet["themes"] or packet["memberships"]:
        lines.append("")
        lines.append("THEME")
        for theme in packet["themes"]:
            lines.append(
                f"  {theme['theme_id']}: {theme['regime']}"
                f"  rs={_num(theme['relative_strength'])}"
                f"  breadth={_num(theme['breadth'])}"
            )
        for membership in packet["memberships"]:
            lines.append(
                f"  membership {membership['theme_id']} w={_num(membership['weight'])}"
            )

    if packet["peers"] or packet["peer_groups"]:
        lines.append("")
        lines.append("PEER STRUCTURE")
        for theme_id, names in packet["peers"].items():
            lines.append(f"  held in {theme_id}: {', '.join(names)}")
        for group in packet["peer_groups"]:
            lines.append(
                f"  {group['group_id']} ({group['method']}, n={group['member_count']},"
                f" cohesion={_num(group['trailing_cohesion'])})"
            )

    ticker = packet["ticker"]
    if ticker:
        lines.append("")
        lines.append("TICKER")
        lines.append(
            f"  setup={ticker['setup']}  ref={_num(ticker['reference_price'])}"
            f"  trend={ticker['trend_state']}  volume={ticker['volume_state']}"
        )
        lines.append(
            f"  alignment: theme={_num(ticker['theme_alignment'])}"
            f"  market={_num(ticker['market_alignment'])}"
            f"  dealer={_num(ticker['dealer_alignment'])}"
        )

    dealer = packet["dealer"]
    if dealer:
        lines.append("")
        lines.append("DEALER STRUCTURE")
        lines.append(f"  regime={dealer['regime']}  spot={_num(dealer['spot'])}")
        lines.append(
            f"  call_wall={_num(dealer['call_wall'])}"
            f"  put_wall={_num(dealer['put_wall'])}"
            f"  gamma_flip={_num(dealer['gamma_flip'])}"
        )

    if packet["catalysts"]:
        lines.append("")
        lines.append("CATALYSTS")
        for event in packet["catalysts"][:8]:
            direct = "direct" if event["is_direct"] else "indirect"
            lines.append(
                f"  [{event['channel']}/{direct}] {event['event_type']}"
                f" - {event['headline'] or ''}".rstrip(" -")
            )

    evidence = packet["evidence"]
    lines.append("")
    lines.append("EVIDENCE")
    lines.append(f"  profile={evidence['freshness_profile']}  valid={decision['valid']}")
    if evidence["stale_inputs"]:
        lines.append(f"  stale: {', '.join(evidence['stale_inputs'])}")
    if evidence["missing_inputs"]:
        lines.append(f"  missing: {', '.join(evidence['missing_inputs'])}")
    if evidence["warnings"]:
        lines.append(f"  warnings: {', '.join(evidence['warnings'])}")
    lines.append(f"  snapshot={decision['snapshot_id']}")
    return "\n".join(lines)


__all__ = ["format_context_packet", "render_context_packet"]
