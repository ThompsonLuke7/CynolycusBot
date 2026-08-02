"""Versioned, immutable configuration for the deterministic policy engine.

The configuration is content-hashed and stored with every policy decision, so
a replayed decision can prove which thresholds produced it.  Nothing here
reads the environment, the clock, or the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import timedelta
from decimal import Decimal
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from core.nervous_system.contracts.enums import (
    DataQualitySeverity,
    DealerRegime,
    InstrumentFamily,
    MarketRegime,
    PolicyMode,
    RuntimeEnvironment,
    ThemeRegime,
)


class StructureRisk(str, Enum):
    """Maximum-loss classification of an instrument family at intent time.

    Intents carry an instrument *family*, not legs, so this classification is
    necessarily family-level.  Leg-level naked/ratio detection belongs to
    instrument construction, where option legs and orientations exist.
    Anything not classified is treated as ``UNKNOWN`` and vetoed.
    """

    DEFINED_LOSS = "DEFINED_LOSS"
    COLLATERALIZED = "COLLATERALIZED"
    NAKED_SHORT = "NAKED_SHORT"
    UNCOVERED_RATIO = "UNCOVERED_RATIO"
    UNKNOWN = "UNKNOWN"


BOUNDED_STRUCTURE_RISKS = frozenset(
    {StructureRisk.DEFINED_LOSS, StructureRisk.COLLATERALIZED}
)

_ONE = Decimal("1")
_ZERO = Decimal("0")


def _decimal_text(value: Decimal) -> str:
    """Render a Decimal so numerically equal values hash identically."""

    return format(value.normalize(), "f")


def _require_decimal(name: str, value: object, *, allow_zero: bool = True) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    if value < _ZERO or (not allow_zero and value == _ZERO):
        raise ValueError(f"{name} must be positive")
    return value


def _require_multipliers(
    name: str,
    value: object,
    key_type: type[Enum],
) -> Mapping[Enum, Decimal]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[Enum, Decimal] = {}
    for key, multiplier in value.items():
        if not isinstance(key, key_type):
            raise TypeError(f"{name} keys must be {key_type.__name__}")
        _require_decimal(f"{name}[{key.value}]", multiplier)
        if multiplier > _ONE:
            raise ValueError(
                f"{name}[{key.value}] must not increase risk during the MVP"
            )
        normalized[key] = multiplier
    missing = tuple(member.value for member in key_type if member not in normalized)
    if missing:
        raise ValueError(f"{name} is missing multipliers for {missing}")
    return MappingProxyType(normalized)


def _require_frozenset(name: str, value: object, item_type: type) -> frozenset:
    if not isinstance(value, frozenset):
        raise TypeError(f"{name} must be a frozenset")
    if any(not isinstance(item, item_type) for item in value):
        raise TypeError(f"{name} items must be {item_type.__name__}")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class PolicyConfig:
    """One immutable, content-hashed policy version."""

    policy_version: str
    config_version: str
    mode: PolicyMode
    environment: RuntimeEnvironment
    account_alias: str
    paper_account_aliases: frozenset[str]
    permitted_strategies: frozenset[str]
    allowed_instruments: frozenset[InstrumentFamily]
    structure_risk: Mapping[InstrumentFamily, StructureRisk]
    required_snapshot_profile: str
    required_readiness_jobs: frozenset[str]
    max_daily_loss: Decimal
    max_gross_notional: Decimal
    max_position_notional: Decimal
    minimum_order_notional: Decimal
    money_quantum: Decimal
    entry_window: timedelta
    liquidity_metric: str
    min_liquidity_value: Decimal
    blocking_data_quality_severities: frozenset[DataQualitySeverity]
    market_regime_multipliers: Mapping[MarketRegime, Decimal]
    theme_regime_multipliers: Mapping[ThemeRegime, Decimal]
    dealer_regime_multipliers: Mapping[DealerRegime, Decimal]
    data_quality_multipliers: Mapping[DataQualitySeverity, Decimal]

    def __post_init__(self) -> None:
        for name in ("policy_version", "config_version", "account_alias",
                     "required_snapshot_profile", "liquidity_metric"):
            _require_text(name, getattr(self, name))
        for name in ("policy_version", "config_version"):
            if "@" not in getattr(self, name):
                raise ValueError(f"{name} must be versioned with '@'")
        if not isinstance(self.mode, PolicyMode):
            raise TypeError("mode must be a PolicyMode")
        if not isinstance(self.environment, RuntimeEnvironment):
            raise TypeError("environment must be a RuntimeEnvironment")
        if not isinstance(self.entry_window, timedelta) or self.entry_window <= timedelta(0):
            raise ValueError("entry_window must be a positive timedelta")

        object.__setattr__(
            self,
            "paper_account_aliases",
            _require_frozenset("paper_account_aliases", self.paper_account_aliases, str),
        )
        object.__setattr__(
            self,
            "permitted_strategies",
            _require_frozenset("permitted_strategies", self.permitted_strategies, str),
        )
        object.__setattr__(
            self,
            "allowed_instruments",
            _require_frozenset(
                "allowed_instruments", self.allowed_instruments, InstrumentFamily
            ),
        )
        object.__setattr__(
            self,
            "required_readiness_jobs",
            _require_frozenset(
                "required_readiness_jobs", self.required_readiness_jobs, str
            ),
        )
        object.__setattr__(
            self,
            "blocking_data_quality_severities",
            _require_frozenset(
                "blocking_data_quality_severities",
                self.blocking_data_quality_severities,
                DataQualitySeverity,
            ),
        )
        if not self.permitted_strategies:
            raise ValueError("permitted_strategies must not be empty")
        if not self.required_readiness_jobs:
            raise ValueError("required_readiness_jobs must not be empty")

        if not isinstance(self.structure_risk, Mapping):
            raise TypeError("structure_risk must be a mapping")
        classification: dict[InstrumentFamily, StructureRisk] = {}
        for family, risk in self.structure_risk.items():
            if not isinstance(family, InstrumentFamily):
                raise TypeError("structure_risk keys must be InstrumentFamily")
            if not isinstance(risk, StructureRisk):
                raise TypeError("structure_risk values must be StructureRisk")
            classification[family] = risk
        object.__setattr__(self, "structure_risk", MappingProxyType(classification))

        _require_decimal("max_daily_loss", self.max_daily_loss, allow_zero=False)
        _require_decimal("max_gross_notional", self.max_gross_notional, allow_zero=False)
        _require_decimal(
            "max_position_notional", self.max_position_notional, allow_zero=False
        )
        _require_decimal("minimum_order_notional", self.minimum_order_notional)
        _require_decimal("money_quantum", self.money_quantum, allow_zero=False)
        _require_decimal("min_liquidity_value", self.min_liquidity_value)
        if self.minimum_order_notional > self.max_position_notional:
            raise ValueError(
                "minimum_order_notional must not exceed max_position_notional"
            )

        object.__setattr__(
            self,
            "market_regime_multipliers",
            _require_multipliers(
                "market_regime_multipliers", self.market_regime_multipliers, MarketRegime
            ),
        )
        object.__setattr__(
            self,
            "theme_regime_multipliers",
            _require_multipliers(
                "theme_regime_multipliers", self.theme_regime_multipliers, ThemeRegime
            ),
        )
        object.__setattr__(
            self,
            "dealer_regime_multipliers",
            _require_multipliers(
                "dealer_regime_multipliers", self.dealer_regime_multipliers, DealerRegime
            ),
        )
        object.__setattr__(
            self,
            "data_quality_multipliers",
            _require_multipliers(
                "data_quality_multipliers",
                self.data_quality_multipliers,
                DataQualitySeverity,
            ),
        )

    def structure_risk_for(self, family: InstrumentFamily) -> StructureRisk:
        """Classify a family, failing closed on anything unclassified."""

        return self.structure_risk.get(family, StructureRisk.UNKNOWN)

    def canonical_payload(self) -> dict[str, object]:
        """Return the stable JSON-compatible representation of this version."""

        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            payload[field.name] = _canonical_value(value)
        return payload

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_value(value: object) -> object:
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, timedelta):
        return _decimal_text(Decimal(str(value.total_seconds())))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, frozenset):
        return sorted(str(_canonical_value(item)) for item in value)
    if isinstance(value, Mapping):
        return {
            str(_canonical_value(key)): _canonical_value(item)
            for key, item in value.items()
        }
    return value


_MVP_STRUCTURE_RISK: Mapping[InstrumentFamily, StructureRisk] = MappingProxyType(
    {
        InstrumentFamily.EQUITY: StructureRisk.DEFINED_LOSS,
        InstrumentFamily.SINGLE_OPTION: StructureRisk.DEFINED_LOSS,
        InstrumentFamily.VERTICAL: StructureRisk.DEFINED_LOSS,
        InstrumentFamily.BUTTERFLY: StructureRisk.DEFINED_LOSS,
        InstrumentFamily.IRON_BUTTERFLY: StructureRisk.DEFINED_LOSS,
        InstrumentFamily.CONDOR: StructureRisk.DEFINED_LOSS,
        InstrumentFamily.IRON_CONDOR: StructureRisk.DEFINED_LOSS,
        InstrumentFamily.COVERED_CALL: StructureRisk.COLLATERALIZED,
        InstrumentFamily.CASH_SECURED_PUT: StructureRisk.COLLATERALIZED,
        InstrumentFamily.PROTECTIVE_PUT: StructureRisk.COLLATERALIZED,
        InstrumentFamily.COLLAR: StructureRisk.COLLATERALIZED,
        # Family alone cannot prove a bounded loss for these, so they fail closed
        # until instrument construction can inspect legs.
        InstrumentFamily.CALENDAR: StructureRisk.UNKNOWN,
        InstrumentFamily.DIAGONAL: StructureRisk.UNKNOWN,
        InstrumentFamily.STRADDLE: StructureRisk.UNKNOWN,
        InstrumentFamily.STRANGLE: StructureRisk.UNKNOWN,
        InstrumentFamily.ROLL: StructureRisk.UNKNOWN,
    }
)

_MVP_MARKET_MULTIPLIERS: Mapping[MarketRegime, Decimal] = MappingProxyType(
    {
        MarketRegime.STRONG_RISK_ON: Decimal("1.0"),
        MarketRegime.RISK_ON: Decimal("1.0"),
        MarketRegime.NEUTRAL: Decimal("1.0"),
        MarketRegime.DETERIORATING: Decimal("0.8"),
        MarketRegime.RISK_OFF: Decimal("0.5"),
        MarketRegime.CRISIS: Decimal("0.25"),
        MarketRegime.UNKNOWN: Decimal("0.5"),
    }
)

_MVP_THEME_MULTIPLIERS: Mapping[ThemeRegime, Decimal] = MappingProxyType(
    {
        ThemeRegime.LEADERSHIP: Decimal("1.0"),
        ThemeRegime.ACCUMULATION: Decimal("1.0"),
        ThemeRegime.HEALTHY: Decimal("1.0"),
        ThemeRegime.NEUTRAL: Decimal("1.0"),
        ThemeRegime.DETERIORATING: Decimal("0.75"),
        ThemeRegime.DISTRIBUTION: Decimal("0.5"),
        ThemeRegime.LIQUIDATION: Decimal("0.25"),
        ThemeRegime.UNKNOWN: Decimal("0.5"),
    }
)

_MVP_DEALER_MULTIPLIERS: Mapping[DealerRegime, Decimal] = MappingProxyType(
    {
        DealerRegime.POSITIVE_GAMMA: Decimal("1.0"),
        DealerRegime.UPSIDE_ACCELERATION: Decimal("1.0"),
        DealerRegime.NEUTRAL_GAMMA: Decimal("1.0"),
        DealerRegime.PINNING: Decimal("0.9"),
        DealerRegime.SHORT_GAMMA: Decimal("0.75"),
        DealerRegime.DOWNSIDE_ACCELERATION: Decimal("0.5"),
        DealerRegime.UNKNOWN: Decimal("0.75"),
    }
)

_MVP_DATA_QUALITY_MULTIPLIERS: Mapping[DataQualitySeverity, Decimal] = MappingProxyType(
    {
        DataQualitySeverity.INFO: Decimal("1.0"),
        DataQualitySeverity.WARNING: Decimal("0.75"),
        # ERROR and CRITICAL are hard vetoes; the multiplier is unreachable but
        # must exist so the mapping is total.
        DataQualitySeverity.ERROR: Decimal("0"),
        DataQualitySeverity.CRITICAL: Decimal("0"),
    }
)


MVP_POLICY_CONFIG = PolicyConfig(
    policy_version="nervous-system-policy@1",
    config_version="nervous-system-policy-config@1",
    mode=PolicyMode.SHADOW,
    environment=RuntimeEnvironment.DEVELOPMENT,
    account_alias="paper",
    paper_account_aliases=frozenset({"paper"}),
    permitted_strategies=frozenset({"meta_ranker"}),
    allowed_instruments=frozenset(
        {
            InstrumentFamily.EQUITY,
            InstrumentFamily.SINGLE_OPTION,
            InstrumentFamily.VERTICAL,
        }
    ),
    structure_risk=_MVP_STRUCTURE_RISK,
    required_snapshot_profile="meta_4h_1420@1",
    required_readiness_jobs=frozenset({"nightly_data_readiness"}),
    max_daily_loss=Decimal("2000.00"),
    max_gross_notional=Decimal("150000.00"),
    max_position_notional=Decimal("5000.00"),
    minimum_order_notional=Decimal("100.00"),
    money_quantum=Decimal("0.01"),
    entry_window=timedelta(minutes=20),
    liquidity_metric="dollar_volume_20d",
    min_liquidity_value=Decimal("5000000.00"),
    blocking_data_quality_severities=frozenset(
        {DataQualitySeverity.ERROR, DataQualitySeverity.CRITICAL}
    ),
    market_regime_multipliers=_MVP_MARKET_MULTIPLIERS,
    theme_regime_multipliers=_MVP_THEME_MULTIPLIERS,
    dealer_regime_multipliers=_MVP_DEALER_MULTIPLIERS,
    data_quality_multipliers=_MVP_DATA_QUALITY_MULTIPLIERS,
)


__all__ = [
    "BOUNDED_STRUCTURE_RISKS",
    "MVP_POLICY_CONFIG",
    "PolicyConfig",
    "StructureRisk",
]
