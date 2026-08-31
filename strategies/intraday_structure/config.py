from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_ROOT / "config" / "intraday_structure_v1.json"


@dataclass(frozen=True)
class DetectorThresholds:
    min_history_bars: int = 20
    min_relative_volume: float = 1.20
    capitulation_volume: float = 2.20
    range_expansion: float = 1.50
    selloff_return_3: float = -0.012
    long_wick_ratio: float = 0.42
    pivot_break_buffer_atr: float = 0.05
    retest_tolerance_atr: float = 0.25
    hold_bars: int = 2
    vwap_hold_bars: int = 2
    trend_strength: float = 0.35
    pullback_max_atr: float = 0.50
    rejection_distance_atr: float = 0.30
    exhaustion_momentum: float = 0.05
    max_failed_breaks: int = 2


@dataclass(frozen=True)
class LevelPolicy:
    cluster_atr: float = 0.20
    cluster_pct: float = 0.0015
    swing_lookback_bars: int = 60
    opening_range_minutes: int = 30
    premarket_start_hour_et: int = 4
    round_number_steps: tuple[float, ...] = (1.0, 5.0, 10.0)
    volume_profile_bins: int = 16
    options_max_age_minutes: int = 24 * 60


@dataclass(frozen=True)
class TargetPolicy:
    min_runway_score: float = 0.45
    min_reward_risk: float = 1.25
    max_invalidation_atr: float = 2.0
    #: A destination further than this is not a same-session target. Matches
    #: DealerPlatePolicy.max_target_distance_atr, which already draws this line
    #: for dealer destinations.
    max_target_distance_atr: float = 8.0
    max_extensions: int = 2
    extension_min_runway: float = 0.58
    partial_profit_fraction: float = 0.50
    move_stop_to_entry_after_target: bool = True
    trailing_structure_atr: float = 0.35
    max_setup_bars: int = 180
    time_exit_bars: int = 120
    close_at_session_end: bool = True


@dataclass(frozen=True)
class ReplayPolicy:
    entry_delay_bars: int = 1
    spread_bps: float = 8.0
    slippage_bps: float = 4.0
    commission_per_share: float = 0.0
    min_bar_dollar_volume: float = 25_000.0
    label_forward_bars: int = 60
    overlap_cooldown_bars: int = 15


@dataclass(frozen=True)
class ExecutionPolicy:
    """Real paper-broker execution for confirmed setups. OFF by default.

    Until now this engine had no broker code at all: every closed setup was a
    MODELLED fill (entry at the next bar's open, costs from ``ReplayPolicy``).
    That is honest, but it means the engine's numbers are not comparable with
    the other modules' and its upgrades can be neither credited nor blamed.

    DTE is deliberately short here, and that is not a contradiction of the
    2026-07-28 finding that pushed the 4H modules to a 21-day floor. That floor
    exists because those modules hold for days-to-weeks and a 2-DTE contract
    burns its whole life in theta before the thesis resolves. This engine's
    median hold is minutes: a near-dated contract is the right expression, and
    a monthly would be paying for time it never uses. 0DTE is still excluded —
    pin and assignment risk into the close is a different problem from theta.
    """

    enabled: bool = True
    option_type: str = "auto"          # auto -> call for long, put for short

    # Expiry selection follows the SPY daytrader's proven convention
    # (`OptionOrderPolicy.dte_cutoff_hhmm` / `expiring_position_exit_hhmm`):
    # take same-day expiry early in the session, roll to the next session after
    # the cutoff, and force-flatten anything expiring today before the close.
    #
    # 0DTE is the right expression HERE and a hazard elsewhere, and the
    # difference is holding period. This engine's median hold is minutes; a
    # news-driven +10% burst is exactly what a same-day contract captures and
    # what a monthly dilutes. The 4H modules hold for days and were moved to a
    # 21-day floor for the opposite reason (2026-07-28). Same-day expiry is only
    # listed daily on the index ETFs and the largest names — everything else
    # simply resolves to its nearest listed expiry, which is the desired
    # behaviour rather than a fallback.
    allow_0dte: bool = True
    dte_cutoff_hhmm: str = "13:00"     # before -> 0DTE allowed; after -> next session
    expiring_exit_hhmm: str = "15:40"  # force-flatten same-day expiries; never assigned
    min_dte: int = 0
    max_dte: int = 7

    target_notional: float = 1_000.0
    max_contracts: int = 20
    max_concurrent_positions: int = 5
    max_new_positions_per_session: int = 20
    state_path: str = "Data/inference/intraday_structure/open_option_positions.json"


@dataclass(frozen=True)
class RegimePolicy:
    """Thresholds for the rule-based LONG/SHORT/SIDEWAYS context label.

    ``veto_enabled`` is deliberately OFF by default.  Recording the regime
    changes nothing about what the engine trades, so it can ship immediately and
    accumulate evidence; letting the regime BLOCK a setup is a trading change
    and must be switched on as a measured ablation, not as a side effect.
    """

    #: 14-bar ATR over its 60-bar baseline; at or below this the tape is coiling.
    compression_atr_ratio: float = 0.72
    #: Support and resistance both inside this many ATR means price is boxed in.
    #: ``None`` disables the test. Set it to None whenever the levels and the
    #: ATR come from the same timeframe as the levels themselves -- on daily
    #: bars, prior-day high and low bracket spot by construction at about one
    #: daily ATR, so "inside 0.75 ATR of both" is nearly always true and the
    #: test measures the timeframe rather than the tape.
    trapped_room_atr: float | None = 0.75
    #: ...but only levels this strong count as walls. Without it the test fires
    #: on every name: cluster_levels() emits a dozen families, so the NEAREST
    #: level on each side is always close, and "trapped" degenerates into a
    #: measure of level density rather than of the tape. Strength is a noisy-OR
    #: over independent mechanisms, so this asks for a level several of them
    #: agree on, not a lone round number.
    trapped_min_strength: float = 0.55
    #: Rejections on the surrounding levels before the location is called hostile.
    failed_break_count: int = 3
    #: |trend_strength| needed before a side is called trending rather than balanced.
    trend_strength: float = 0.35
    veto_enabled: bool = False


@dataclass(frozen=True)
class DealerPlatePolicy:
    """Paper-only policy for broad dealer-map candidates and qualified alerts.

    The score is an interpretable hypothesis, not a statement that option OI
    reveals dealer inventory or causes a move.  It is kept separate from the
    price detectors so replay can ablate it cleanly.
    """

    enabled: bool = True
    snapshot_root: str = "Data/dealer_positioning/historical_snapshots"
    ranking_path: str = "Data/dealer_positioning/rankings/dealer_swing_rankings_latest.parquet"
    candidate_top_structural: int = 40
    candidate_top_change: int = 40
    candidate_max_age_hours: float = 30.0
    min_score: float = 0.70
    min_target_strength: float = 0.60
    min_target_distance_atr: float = 0.75
    max_target_distance_atr: float = 8.0


@dataclass(frozen=True)
class LiquidityUniversePolicy:
    """Bounded broad-universe seed for price-structure discovery.

    This selects liquid names from the locally maintained shared universe.  It
    is a watchlist source only: price confirmation and, for a qualified plate,
    an as-of dealer-map destination remain mandatory.
    """

    enabled: bool = True
    universe_path: str = "Data/shared/universe/shared_universe.csv"
    top_n: int = 50


@dataclass(frozen=True)
class OpeningDiscoveryPolicy:
    """Paper-only opening-mover discovery over a bounded liquid universe.

    These thresholds promote a symbol into the existing structure engine; they
    never authorize an order.  ``top_n`` also defines the additional symbols
    the combined shared stream must cover.
    """

    enabled: bool = False
    universe_path: str = "Data/shared/universe/shared_universe.csv"
    top_n: int = 750
    scan_start_hour_et: int = 4
    scan_end_hour_et: int = 11
    min_gap_pct: float = 0.02
    min_rth_return_pct: float = 0.008
    min_three_bar_return_pct: float = 0.003
    min_session_range_pct: float = 0.012
    min_volume_pace: float = 1.25
    min_relative_strength_pct: float = 0.004
    max_candidates_per_session: int = 60


@dataclass(frozen=True)
class CatalystDiscoveryPolicy:
    """Strict promotion policy for the already-scored live catalyst ledger."""

    enabled: bool = False
    ledger_path: str = "signals/news/data/processed/live_catalyst_records.parquet"
    universe_path: str = "Data/shared/universe/shared_universe.csv"
    max_age_minutes: int = 180
    min_bullish_score: float = 0.62
    max_bearish_score: float = 0.38
    refresh_seconds: float = 30.0
    max_candidates_per_poll: int = 25
    strict_ticker_relevance: bool = True


@dataclass(frozen=True)
class CandidateCapacityPolicy:
    """Retention policy when the structure watchlist reaches its hard cap."""

    opening_reserve: int = 25
    catalyst_reserve: int = 15
    source_priority: tuple[str, ...] = (
        "manual_watchlist",
        "manual_dashboard",
        "manual",
        "validated_catalyst",
        "opening_momentum",
        "30m_swing",
        "meta_ranker",
        "momentum_expansion",
        "4h_swing",
        "dealer_ranker",
        "dealer_level_map",
        "high_liquidity_universe",
    )


@dataclass(frozen=True)
class EvidencePolicy:
    """Append-only funnel ledger and candidate-level fixed-horizon outcomes."""

    enabled: bool = False
    event_path: str = "Data/inference/intraday_structure/decision_events.jsonl"
    outcome_horizons_minutes: tuple[int, ...] = (5, 15, 30, 60)


@dataclass(frozen=True)
class IntradayStructureConfig:
    version: str = "intraday_structure_v1"
    enabled: bool = False
    paper_only: bool = True
    candidate_limit: int = 25
    candidate_ttl_minutes: int = 24 * 60
    min_price: float = 1.0
    min_average_dollar_volume: float = 2_000_000.0
    supported_tickers: tuple[str, ...] = ()
    market_alignment_penalty: float = 0.20
    exceptional_relative_strength: float = 0.75
    max_market_conflict: float = -0.70
    failure_cooldown_bars: int = 20
    alert_states: tuple[str, ...] = (
        "SETUP_DETECTED",
        "ARMED",
        "CONFIRMED",
        "TARGET_REACHED",
        "EXTENDED",
        "EXHAUSTED",
        "INVALIDATED",
        "CLOSED",
    )
    context_symbols: tuple[str, ...] = ("SPY", "QQQ", "VIXY")
    manual_watchlist: tuple[str, ...] = ()
    state_path: str = "Data/inference/intraday_structure/state.json"
    transition_log_path: str = "Data/inference/intraday_structure/transitions.jsonl"
    signal_path: str = "Data/inference/intraday_structure/active_signals.json"
    ledger_path: str = "Data/inference/intraday_structure/closed_setups.jsonl"
    abstention_path: str = "Data/inference/intraday_structure/abstentions.jsonl"
    #: Append-only 1-minute bar archive. Forward-only and the sole input for the
    #: D2 counterfactual, so a session not collected is a session D2 loses.
    #: Size scales with the combined streamed universe, not candidate_limit.
    archive_bars: bool = True
    bar_archive_root: str = "Data/archive/intraday_1m"
    detector: DetectorThresholds = field(default_factory=DetectorThresholds)
    levels: LevelPolicy = field(default_factory=LevelPolicy)
    target: TargetPolicy = field(default_factory=TargetPolicy)
    replay: ReplayPolicy = field(default_factory=ReplayPolicy)
    regime: RegimePolicy = field(default_factory=RegimePolicy)
    dealer_plate: DealerPlatePolicy = field(default_factory=DealerPlatePolicy)
    liquidity_universe: LiquidityUniversePolicy = field(default_factory=LiquidityUniversePolicy)
    opening_discovery: OpeningDiscoveryPolicy = field(default_factory=OpeningDiscoveryPolicy)
    catalyst_discovery: CatalystDiscoveryPolicy = field(default_factory=CatalystDiscoveryPolicy)
    candidate_capacity: CandidateCapacityPolicy = field(default_factory=CandidateCapacityPolicy)
    evidence: EvidencePolicy = field(default_factory=EvidencePolicy)
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "IntradayStructureConfig":
        known = dict(raw)
        known["detector"] = DetectorThresholds(**known.get("detector", {}))
        levels = dict(known.get("levels", {}))
        if "round_number_steps" in levels:
            levels["round_number_steps"] = tuple(levels["round_number_steps"])
        known["levels"] = LevelPolicy(**levels)
        known["target"] = TargetPolicy(**known.get("target", {}))
        known["replay"] = ReplayPolicy(**known.get("replay", {}))
        known["regime"] = RegimePolicy(**known.get("regime", {}))
        known["dealer_plate"] = DealerPlatePolicy(**known.get("dealer_plate", {}))
        known["liquidity_universe"] = LiquidityUniversePolicy(**known.get("liquidity_universe", {}))
        known["opening_discovery"] = OpeningDiscoveryPolicy(**known.get("opening_discovery", {}))
        known["catalyst_discovery"] = CatalystDiscoveryPolicy(**known.get("catalyst_discovery", {}))
        capacity = dict(known.get("candidate_capacity", {}))
        if "source_priority" in capacity:
            capacity["source_priority"] = tuple(capacity["source_priority"])
        known["candidate_capacity"] = CandidateCapacityPolicy(**capacity)
        evidence = dict(known.get("evidence", {}))
        if "outcome_horizons_minutes" in evidence:
            evidence["outcome_horizons_minutes"] = tuple(evidence["outcome_horizons_minutes"])
        known["evidence"] = EvidencePolicy(**evidence)
        known["execution"] = ExecutionPolicy(**known.get("execution", {}))
        for key in ("alert_states", "context_symbols", "manual_watchlist", "supported_tickers"):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> IntradayStructureConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = IntradayStructureConfig.from_mapping(raw)
    if config.version != "intraday_structure_v1":
        raise ValueError(f"unsupported intraday structure config version: {config.version}")
    if not config.paper_only:
        raise ValueError("intraday structure v1 must remain paper_only")
    if config.execution.enabled and not config.paper_only:
        # Belt and braces: the check above already forbids it, but execution is
        # the one setting here that can move money, so it states its own
        # precondition rather than relying on a neighbour.
        raise ValueError("intraday structure execution requires paper_only")
    if config.execution.enabled and int(config.execution.min_dte) < 0:
        raise ValueError("intraday structure execution min_dte must be >= 0")
    if config.execution.allow_0dte and not str(config.execution.expiring_exit_hhmm or "").strip():
        # 0DTE without a flatten time is how a same-day contract rides into
        # expiry and gets assigned. The two settings are one decision.
        raise ValueError("intraday structure execution: allow_0dte requires expiring_exit_hhmm")
    return config
