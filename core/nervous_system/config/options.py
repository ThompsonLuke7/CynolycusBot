"""Versioned configuration for option chain fitness and candidate scoring.

Score components are transparent normalised measures of the chain, not
probabilities.  Nothing here estimates a chance of profit; a probability would
require a separately calibrated and versioned model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from core.nervous_system.contracts.enums import InstrumentFamily


_ZERO = Decimal("0")
_ONE = Decimal("1")

SCORE_COMPONENTS = ("spread", "liquidity", "dte", "delta", "budget")


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _require(name: str, value: object, *, allow_zero: bool = True) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    if value < _ZERO or (not allow_zero and value == _ZERO):
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class OptionSelectionConfig:
    """One immutable, content-hashed selection version."""

    config_version: str
    max_quote_age_seconds: Decimal
    max_spread_absolute: Decimal
    max_spread_fraction: Decimal
    min_open_interest: int
    min_volume: int
    min_dte: int
    max_dte: int
    target_delta: Decimal
    default_preferences: tuple[InstrumentFamily, ...]
    equity_fallback_allowed: bool
    score_weights: Mapping[str, Decimal]
    max_candidates_per_structure: int
    money_quantum: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.config_version, str) or "@" not in self.config_version:
            raise ValueError("config_version must be versioned with '@'")
        _require("max_quote_age_seconds", self.max_quote_age_seconds, allow_zero=False)
        _require("max_spread_absolute", self.max_spread_absolute, allow_zero=False)
        _require("max_spread_fraction", self.max_spread_fraction, allow_zero=False)
        _require("money_quantum", self.money_quantum, allow_zero=False)
        target = _require("target_delta", self.target_delta, allow_zero=False)
        if target > _ONE:
            raise ValueError("target_delta must be a magnitude in (0, 1]")

        for name in ("min_open_interest", "min_volume", "min_dte", "max_dte"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.min_dte > self.max_dte:
            raise ValueError("min_dte must not exceed max_dte")
        if (
            not isinstance(self.max_candidates_per_structure, int)
            or self.max_candidates_per_structure <= 0
        ):
            raise ValueError("max_candidates_per_structure must be positive")
        if not isinstance(self.equity_fallback_allowed, bool):
            raise TypeError("equity_fallback_allowed must be a bool")

        preferences = tuple(self.default_preferences)
        if not preferences:
            raise ValueError("default_preferences must not be empty")
        if any(not isinstance(item, InstrumentFamily) for item in preferences):
            raise TypeError("default_preferences must contain InstrumentFamily values")
        if len(set(preferences)) != len(preferences):
            raise ValueError("default_preferences must not repeat a family")
        object.__setattr__(self, "default_preferences", preferences)

        if not isinstance(self.score_weights, Mapping):
            raise TypeError("score_weights must be a mapping")
        weights: dict[str, Decimal] = {}
        for name in SCORE_COMPONENTS:
            if name not in self.score_weights:
                raise ValueError(f"score_weights is missing {name!r}")
            weights[name] = _require(f"score_weights[{name}]", self.score_weights[name])
        extra = set(self.score_weights) - set(SCORE_COMPONENTS)
        if extra:
            raise ValueError(f"score_weights has unknown components: {sorted(extra)}")
        total = sum(weights.values(), _ZERO)
        if total != _ONE:
            raise ValueError("score_weights must sum to exactly 1")
        object.__setattr__(self, "score_weights", MappingProxyType(weights))

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Decimal):
                payload[field.name] = _decimal_text(value)
            elif isinstance(value, Mapping):
                payload[field.name] = {
                    key: _decimal_text(item) for key, item in value.items()
                }
            elif isinstance(value, tuple):
                payload[field.name] = [
                    item.value if isinstance(item, Enum) else item for item in value
                ]
            elif isinstance(value, Enum):
                payload[field.name] = value.value
            else:
                payload[field.name] = value
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


# Meta starts conservatively.  The selector supports every approved structure,
# but only when an intent explicitly asks for one; Meta does not begin choosing
# the full suite without a separately versioned strategy rule.
MVP_OPTION_SELECTION_CONFIG = OptionSelectionConfig(
    config_version="nervous-system-option-selection@1",
    max_quote_age_seconds=Decimal("300"),
    max_spread_absolute=Decimal("0.50"),
    max_spread_fraction=Decimal("0.10"),
    min_open_interest=100,
    min_volume=10,
    min_dte=14,
    max_dte=120,
    target_delta=Decimal("0.55"),
    default_preferences=(
        InstrumentFamily.EQUITY,
        InstrumentFamily.SINGLE_OPTION,
        InstrumentFamily.VERTICAL,
    ),
    equity_fallback_allowed=True,
    score_weights=MappingProxyType(
        {
            "spread": Decimal("0.25"),
            "liquidity": Decimal("0.20"),
            "dte": Decimal("0.20"),
            "delta": Decimal("0.20"),
            "budget": Decimal("0.15"),
        }
    ),
    max_candidates_per_structure=200,
    money_quantum=Decimal("0.01"),
)


__all__ = [
    "MVP_OPTION_SELECTION_CONFIG",
    "SCORE_COMPONENTS",
    "OptionSelectionConfig",
]
