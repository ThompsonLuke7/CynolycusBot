from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class OptionContractRow:
    timestamp: datetime
    symbol: str
    expiration: str
    dte: int | None
    strike: float
    option_type: str
    open_interest: float
    volume: float
    gamma: float
    delta: float | None = None
    vega: float | None = None
    iv: float | None = None
    expiry_bucket: int | None = None


@dataclass(frozen=True)
class GammaLevels:
    timestamp: str
    symbol: str
    spot: float
    total_gex: float
    call_wall: float | None
    put_wall: float | None
    nearest_magnet: float | None
    next_magnet_above: float | None
    next_magnet_below: float | None
    vega_wall: float | None
    next_vega_wall_above: float | None
    next_vega_wall_below: float | None
    gamma_flip: float | None
    air_gap_above_score: float
    air_gap_below_score: float
    magnet_threshold_abs_net_gex: float
    vega_threshold_total_vex: float
    expirations: list[str] = field(default_factory=list)
    per_dte_levels: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Levels recomputed inside each expiry bucket. A 0DTE gamma pile and a
    # 60DTE gamma pile are different structures; the aggregate hides that.
    per_bucket_levels: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Share of absolute gamma exposure by bucket, plus the derived slope.
    term_structure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GammaStructure:
    """Everything one chain snapshot supports, with reliability attached.

    Deliberately separates what is close to observed (``topology``: where gamma
    sits) from what is inferred (``signed``: which side dealers hold), so a
    consumer can weight them differently instead of receiving one blended
    number that hides which half it should doubt.
    """

    ladder: Any
    levels: GammaLevels
    topology: Any
    signed: Any
    confidence: Any
    stability: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe summary. The ladder frame is deliberately excluded."""
        out: dict[str, Any] = {"levels": self.levels.to_dict()}
        for name in ("topology", "signed", "confidence", "stability"):
            value = getattr(self, name)
            out[name] = value.to_dict() if hasattr(value, "to_dict") else value
        return out


@dataclass(frozen=True)
class DealerSignal:
    timestamp: str
    symbol: str
    signal_type: str
    direction: str
    spot: float
    level: float
    target: float | None
    stop: float | None
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimTrade:
    trade_id: str
    opened_at: str
    symbol: str
    signal_type: str
    direction: str
    entry: float
    target: float | None
    stop: float | None
    status: str = "open"
    closed_at: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_points: float | None = None
    pnl_dollars: float | None = None
    order_mode: str = "simulated"
    contract_symbol: str | None = None
    contract_expiration: str | None = None
    contract_strike: float | None = None
    contract_side: str | None = None
    entry_contract_price: float | None = None
    exit_contract_price: float | None = None
    entry_contract_bid: float | None = None
    entry_contract_ask: float | None = None
    entry_contract_spread_pct: float | None = None
    exit_contract_bid: float | None = None
    exit_contract_ask: float | None = None
    exit_contract_spread_pct: float | None = None
    open_order_id: str | None = None
    close_order_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
