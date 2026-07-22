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
    detector: DetectorThresholds = field(default_factory=DetectorThresholds)
    levels: LevelPolicy = field(default_factory=LevelPolicy)
    target: TargetPolicy = field(default_factory=TargetPolicy)
    replay: ReplayPolicy = field(default_factory=ReplayPolicy)
    dealer_plate: DealerPlatePolicy = field(default_factory=DealerPlatePolicy)
    liquidity_universe: LiquidityUniversePolicy = field(default_factory=LiquidityUniversePolicy)

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
        known["dealer_plate"] = DealerPlatePolicy(**known.get("dealer_plate", {}))
        known["liquidity_universe"] = LiquidityUniversePolicy(**known.get("liquidity_universe", {}))
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
    return config
