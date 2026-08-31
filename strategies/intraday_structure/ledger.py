"""Immutable closed-setup ledger — the module's measurement instrument.

WHY THIS EXISTS. Before it, a setup that reached a terminal state left behind
only a ``transitions.jsonl`` breadcrumb (``state``, ``spot``, ``reason``) and
whatever ``state.json`` happened to hold at the next save.  ``state.json`` is a
snapshot: a closed setup's entry price, targets, invalidation and excursions
were overwritten and gone.  So the engine could run for a month, confirm 1,159
setups, and still not answer "did any of that work?".  Every field below already
existed in memory at close time and was simply never written down.

This is a MODELLED paper record, not a fill record.  ``entry_price`` is the open
of the bar after confirmation (the engine's existing one-bar entry delay);
``exit_price`` is the engine's own recorded exit.  Costs are the configured
``ReplayPolicy`` assumptions and are stamped into every row so a row is
self-describing and a later cost change cannot silently re-price history.

The same builder feeds the live sink and ``replay._trade_frame`` so research and
live cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from strategies.intraday_structure.config import ReplayPolicy
from strategies.intraday_structure.models import SetupRecord, SetupState
from strategies.intraday_structure.state_store import append_jsonl


LEDGER_SCHEMA_VERSION = "intraday_structure_closed_setup_v1"

# A setup is ledgered on the FIRST of ``models.TERMINAL_STATES`` it reaches.
# INVALIDATED and EXHAUSTED are later archived to CLOSED by the engine, which
# must not produce a second row.

#: Marker written into ``SetupRecord.metadata`` once a row has been emitted.  It
#: rides along in ``state.json``, so a restart mid-session cannot duplicate a row.
LEDGERED_FLAG = "ledgered_at"


@dataclass(frozen=True)
class ClosedSetupRecord:
    """One closed, modelled paper setup.  Append-only; never rewritten."""

    schema_version: str
    engine_version: str
    setup_id: str
    ticker: str
    direction: str
    setup_type: str

    # Time correctness: these are five distinct clocks and must stay distinct.
    candidate_timestamp: str | None       # when the upstream signal was for
    candidate_available_at: str | None    # when it became readable here
    confirmed_at: str | None              # signal generation
    entry_time: str | None                # modelled fill
    exit_time: str | None                 # evaluation

    entry_price: float
    exit_price: float
    exit_reason: str
    terminal_state: str

    initial_invalidation: float | None
    final_invalidation: float | None
    targets: list[float]
    target_index_reached: int
    extensions: int

    gross_points: float
    net_points: float
    gross_return: float
    net_return: float
    risk_points: float | None
    realized_r_after_costs: float | None
    mfe_points: float
    mae_points: float
    bars_held: int

    # Decision inputs, kept so the ablation in step D can slice on them.
    candidate_sources: list[str]
    candidate_score: float
    confidence: float
    runway_score: float
    expected_reward_risk: float | None
    market_alignment_score: float
    context_regime: str
    pivot: float | None
    target_level_type: str | None
    target_level_sources: list[str]
    dealer_plate_qualified: bool
    dealer_plate_score: float | None
    options_source: str
    evidence: list[str]
    warnings: list[str]

    # Cost assumptions this row was priced under.
    cost_spread_bps: float
    cost_slippage_bps: float
    cost_commission_per_share: float

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_closed_setup_record(
    setup: SetupRecord,
    *,
    replay_policy: ReplayPolicy,
    engine_version: str,
    terminal_state: SetupState | None = None,
) -> ClosedSetupRecord | None:
    """Build a ledger row, or ``None`` if the setup never actually entered.

    A setup that was confirmed but closed before its entry bar arrived has no
    ``entry_price``.  That is a real outcome, but it is not a trade, and giving
    it a synthetic entry would fabricate P&L.  It is left to the abstention log.
    """
    if setup.entry_price is None or not math.isfinite(float(setup.entry_price)) or setup.entry_price <= 0:
        return None

    entry = float(setup.entry_price)
    exit_price = _exit_price(setup)
    if exit_price is None:
        return None

    sign = 1.0 if setup.direction.value == "long" else -1.0
    initial_stop = _finite(setup.metadata.get("initial_invalidation"))
    if initial_stop is None:
        initial_stop = _finite(setup.invalidation)
    risk = abs(entry - initial_stop) if initial_stop is not None else None

    # Round-trip cost: the spread/slippage assumption is a fraction of notional,
    # charged once against the entry price, plus commission on both legs.
    cost_fraction = (replay_policy.spread_bps + replay_policy.slippage_bps) / 10_000.0
    gross = sign * (exit_price - entry)
    net = gross - entry * cost_fraction - 2.0 * replay_policy.commission_per_share

    plate = setup.metadata.get("dealer_plate") or {}
    state = terminal_state or setup.state

    return ClosedSetupRecord(
        schema_version=LEDGER_SCHEMA_VERSION,
        engine_version=engine_version,
        setup_id=setup.setup_id,
        ticker=setup.ticker,
        direction=setup.direction.value,
        setup_type=setup.setup_type.value,
        candidate_timestamp=_iso(setup.candidate.timestamp),
        candidate_available_at=_iso(setup.candidate.available_at),
        confirmed_at=_iso(setup.metadata.get("confirmed_at")),
        entry_time=_iso(setup.entry_time),
        exit_time=_iso(setup.updated_at),
        entry_price=entry,
        exit_price=exit_price,
        exit_reason=str(setup.metadata.get("exit_reason") or "unspecified"),
        terminal_state=state.value if isinstance(state, SetupState) else str(state),
        initial_invalidation=initial_stop,
        final_invalidation=_finite(setup.invalidation),
        targets=[float(x) for x in setup.targets if _finite(x) is not None],
        target_index_reached=int(setup.active_target_index),
        extensions=int(setup.extensions),
        gross_points=gross,
        net_points=net,
        gross_return=gross / entry,
        net_return=net / entry,
        risk_points=risk,
        realized_r_after_costs=(net / risk) if risk and risk > 0 else None,
        mfe_points=float(setup.max_favorable_excursion),
        mae_points=float(setup.max_adverse_excursion),
        bars_held=int(setup.bars_alive),
        candidate_sources=list(setup.candidate.sources),
        candidate_score=float(setup.candidate.score),
        confidence=float(setup.confidence),
        runway_score=float(setup.runway_score),
        expected_reward_risk=_finite(setup.expected_reward_risk),
        market_alignment_score=float(setup.market_alignment_score),
        context_regime=str(setup.metadata.get("context_regime") or "unknown"),
        pivot=_finite(setup.pivot),
        target_level_type=setup.metadata.get("target_level_type"),
        target_level_sources=list(setup.metadata.get("target_level_sources") or []),
        dealer_plate_qualified=bool(plate.get("qualified")),
        dealer_plate_score=_finite(plate.get("score")),
        options_source=str((setup.options_context or {}).get("source") or "none"),
        evidence=list(setup.evidence),
        warnings=list(setup.warnings),
        cost_spread_bps=float(replay_policy.spread_bps),
        cost_slippage_bps=float(replay_policy.slippage_bps),
        cost_commission_per_share=float(replay_policy.commission_per_share),
        metadata={
            "runway_components": setup.metadata.get("runway_components") or {},
            "intermediate_obstacle_count": len(setup.metadata.get("intermediate_obstacles") or []),
        },
    )


def ledger_sink(path: str | Path) -> Callable[[ClosedSetupRecord], None]:
    """Append-only JSONL sink, matching ``transition_log_sink``'s shape."""
    return lambda record: append_jsonl(path, record.to_dict())


def _exit_price(setup: SetupRecord) -> float | None:
    for value in (setup.metadata.get("exit_price"), setup.spot):
        price = _finite(value)
        if price is not None and price > 0:
            return price
    return None


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
