from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_datetime(value: datetime | str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class SetupType(str, Enum):
    V_REVERSAL = "v_shaped_capitulation_reversal"
    BREAKOUT = "breakout_continuation"
    VWAP_RECLAIM = "vwap_reclaim_continuation"
    STRUCTURAL_REJECTION = "gamma_structural_level_rejection"
    TREND_PULLBACK = "trend_pullback_continuation"
    EXHAUSTION = "exhaustion_failed_breakout"


class SetupState(str, Enum):
    WATCHING = "WATCHING"
    SETUP_DETECTED = "SETUP_DETECTED"
    ARMED = "ARMED"
    CONFIRMED = "CONFIRMED"
    RUNNING = "RUNNING"
    TARGET_REACHED = "TARGET_REACHED"
    EXTENDED = "EXTENDED"
    EXHAUSTED = "EXHAUSTED"
    INVALIDATED = "INVALIDATED"
    CLOSED = "CLOSED"


TERMINAL_STATES = {SetupState.EXHAUSTED, SetupState.INVALIDATED, SetupState.CLOSED}


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int | None = None
    vwap: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("bar OHLCV must be finite")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError("bar prices must be positive and volume non-negative")
        if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close, self.high):
            raise ValueError("invalid OHLC ordering")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Bar":
        return cls(
            symbol=str(raw.get("symbol") or raw.get("ticker") or ""),
            timestamp=utc_datetime(raw["timestamp"]),
            open=float(raw["open"]), high=float(raw["high"]), low=float(raw["low"]),
            close=float(raw["close"]), volume=float(raw.get("volume", 0.0)),
            trade_count=int(raw["trade_count"]) if raw.get("trade_count") is not None else None,
            vwap=float(raw["vwap"]) if raw.get("vwap") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["timestamp"] = self.timestamp.isoformat()
        return out


@dataclass(frozen=True)
class PriceUpdate:
    symbol: str
    timestamp: datetime
    price: float
    event_type: str = "trade"
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        if not self.symbol or not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("price update requires a symbol and positive finite price")
        if self.event_type not in {"trade", "quote"}:
            raise ValueError("event_type must be trade or quote")


@dataclass(frozen=True)
class Candidate:
    ticker: str
    timestamp: datetime
    direction: Direction
    sources: tuple[str, ...]
    score: float = 0.5
    pivot: float | None = None
    sector_etf: str | None = None
    average_dollar_volume: float | None = None
    available_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", self.ticker.strip().upper())
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        object.__setattr__(self, "available_at", utc_datetime(self.available_at or self.timestamp))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(self, "sources", tuple(sorted({str(x) for x in self.sources if str(x)})))
        if not self.ticker or not self.sources:
            raise ValueError("candidate requires ticker and at least one source")

    @property
    def candidate_id(self) -> str:
        return f"{self.ticker}:{self.direction.value}:{self.timestamp.isoformat()}"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["timestamp"] = self.timestamp.isoformat()
        out["available_at"] = self.available_at.isoformat() if self.available_at else out["timestamp"]
        out["direction"] = self.direction.value
        out["sources"] = list(self.sources)
        return out

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Candidate":
        sources = raw.get("sources") or raw.get("candidate_source") or raw.get("source") or ("manual",)
        if isinstance(sources, str):
            sources = (sources,)
        return cls(
            ticker=str(raw.get("ticker") or raw.get("symbol") or ""),
            timestamp=utc_datetime(raw.get("timestamp") or raw.get("bar")),
            direction=Direction(str(raw.get("direction") or raw.get("side") or "long").lower()),
            sources=tuple(sources), score=float(raw.get("score", raw.get("confidence", 0.5))),
            pivot=_optional_float(raw.get("pivot")), sector_etf=raw.get("sector_etf"),
            average_dollar_volume=_optional_float(raw.get("average_dollar_volume")),
            available_at=utc_datetime(raw.get("available_at") or raw.get("timestamp") or raw.get("bar")),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass
class StructuralLevel:
    price: float
    level_type: str
    strength: float
    freshness_bars: int = 0
    directionality: str = "both"
    touch_count: int = 0
    rejection_count: int = 0
    break_status: str = "unbroken"
    hold_status: str = "untested"
    distance_from_spot: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "StructuralLevel":
        return cls(**raw)


@dataclass(frozen=True)
class OptionsContext:
    timestamp: datetime | None = None
    source: str = "none"
    gamma_regime: str = "unknown"
    gamma_flip: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    local_gamma_exposure: float | None = None
    dealer_imbalance: float | None = None
    dealer_strength_score: float = 0.0
    strike_congestion_score: float = 0.0
    live_call_flow_acceleration: float = 0.0
    live_put_flow_acceleration: float = 0.0
    net_delta_premium: float = 0.0
    short_dated_flow_ratio: float = 0.0
    levels: tuple[StructuralLevel, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return out


@dataclass(frozen=True)
class MarketContext:
    timestamp: datetime
    spy_direction: float = 0.0
    qqq_direction: float = 0.0
    sector_relative_strength: float = 0.0
    breadth: float | None = None
    volatility_regime: float = 0.0
    opening_range_state: str = "unknown"
    index_vwap_alignment: float = 0.0
    market_alignment_score: float = 0.5
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["timestamp"] = self.timestamp.isoformat()
        return out


@dataclass
class SetupRecord:
    setup_id: str
    ticker: str
    setup_type: SetupType
    direction: Direction
    candidate: Candidate
    state: SetupState = SetupState.WATCHING
    phase: str = "WATCHING"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    state_entered_at: datetime | None = None
    pivot: float | None = None
    spot: float | None = None
    invalidation: float | None = None
    targets: list[float] = field(default_factory=list)
    active_target_index: int = 0
    confidence: float = 0.0
    runway_score: float = 0.0
    expected_reward_risk: float | None = None
    market_alignment_score: float = 0.5
    options_context: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions: int = 0
    bars_alive: int = 0
    bars_in_state: int = 0
    entry_price: float | None = None
    entry_time: datetime | None = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    @property
    def active_target(self) -> float | None:
        if not self.targets:
            return None
        return self.targets[min(self.active_target_index, len(self.targets) - 1)]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("created_at", "updated_at", "state_entered_at", "entry_time"):
            value = getattr(self, key)
            out[key] = value.isoformat() if value else None
        out["setup_type"] = self.setup_type.value
        out["direction"] = self.direction.value
        out["state"] = self.state.value
        out["candidate"] = self.candidate.to_dict()
        out["active_target"] = self.active_target
        return out

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SetupRecord":
        data = dict(raw)
        data.pop("active_target", None)
        data["setup_type"] = SetupType(data["setup_type"])
        data["direction"] = Direction(data["direction"])
        data["state"] = SetupState(data["state"])
        data["candidate"] = Candidate.from_mapping(data["candidate"])
        for key in ("created_at", "updated_at", "state_entered_at", "entry_time"):
            data[key] = utc_datetime(data[key]) if data.get(key) else None
        return cls(**data)


@dataclass(frozen=True)
class StateTransition:
    setup_id: str
    ticker: str
    setup_type: str
    timestamp: datetime
    from_state: SetupState
    to_state: SetupState
    phase: str
    spot: float | None
    reason: str
    evidence: tuple[str, ...] = ()
    #: When ``spot`` was observed on THIS setup's own tape.  Equal to
    #: ``timestamp`` for a price-driven transition.  Earlier when the transition
    #: was driven by something other than this ticker printing a bar (a TTL
    #: expiry, say), in which case the price is the last one actually seen for
    #: this ticker rather than whichever symbol's bar happened to arrive.
    #: ``None`` means no price for this ticker had been observed at all.
    spot_as_of: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["timestamp"] = self.timestamp.isoformat()
        out["from_state"] = self.from_state.value
        out["to_state"] = self.to_state.value
        out["spot_as_of"] = self.spot_as_of.isoformat() if self.spot_as_of else None
        return out


@dataclass(frozen=True)
class StructureSignal:
    ticker: str
    timestamp: datetime
    setup_type: str
    state: str
    direction: str
    candidate_source: tuple[str, ...]
    confidence: float
    edge_score: int
    pivot: float | None
    spot: float | None
    invalidation: float | None
    target_1: float | None
    target_2: float | None
    active_target: float | None
    runway_score: float
    expected_reward_risk: float | None
    market_alignment_score: float
    options_context: dict[str, Any]
    dealer_plate: dict[str, Any]
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    #: Rule-based tape label at the last confirmation decision, and the reason
    #: the engine declined if it did. ``no_trade_reason`` is None on a setup
    #: that was actually taken.
    context_regime: str = "unknown"
    no_trade_reason: str | None = None
    version: str = "intraday_structure_v1"

    @classmethod
    def from_setup(cls, setup: SetupRecord, version: str) -> "StructureSignal":
        return cls(
            ticker=setup.ticker, timestamp=setup.updated_at or setup.candidate.timestamp,
            setup_type=setup.setup_type.value, state=setup.state.value,
            direction=setup.direction.value, candidate_source=setup.candidate.sources,
            confidence=round(setup.confidence, 4), edge_score=int(round(100 * setup.confidence)),
            pivot=setup.pivot, spot=setup.spot, invalidation=setup.invalidation,
            target_1=setup.targets[0] if setup.targets else None,
            target_2=setup.targets[1] if len(setup.targets) > 1 else None,
            active_target=setup.active_target, runway_score=round(setup.runway_score, 4),
            expected_reward_risk=setup.expected_reward_risk,
            market_alignment_score=round(setup.market_alignment_score, 4),
            options_context=dict(setup.options_context), dealer_plate=dict(setup.metadata.get("dealer_plate") or {}),
            evidence=tuple(setup.evidence),
            warnings=tuple(setup.warnings),
            context_regime=str(setup.metadata.get("context_regime") or "unknown"),
            no_trade_reason=setup.metadata.get("no_trade_reason"),
            version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["timestamp"] = self.timestamp.isoformat()
        out["candidate_source"] = list(self.candidate_source)
        out["evidence"] = list(self.evidence)
        out["warnings"] = list(self.warnings)
        return out


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None
