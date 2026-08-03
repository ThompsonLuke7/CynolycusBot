"""Versioned configuration for exposure math and portfolio limits.

Factor tags are hand-curated risk metadata, not learned correlations.  They
record that two names move together for a structural reason so that overlap is
visible to policy; they never claim a measured correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping


_ZERO = Decimal("0")


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _require_positive(name: str, value: object) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite() or value <= _ZERO:
        raise ValueError(f"{name} must be a positive finite Decimal")
    return value


@dataclass(frozen=True)
class PortfolioConfig:
    """One immutable, content-hashed exposure configuration."""

    config_version: str
    sector_map: Mapping[str, str]
    factor_tags: Mapping[str, frozenset[str]]
    default_contract_multiplier: Decimal
    money_quantum: Decimal
    max_gross_notional: Decimal
    max_symbol_notional: Decimal
    max_sector_notional: Decimal
    max_theme_notional: Decimal
    max_factor_notional: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.config_version, str) or "@" not in self.config_version:
            raise ValueError("config_version must be versioned with '@'")
        if not isinstance(self.sector_map, Mapping):
            raise TypeError("sector_map must be a mapping")
        sectors: dict[str, str] = {}
        for ticker, sector_id in self.sector_map.items():
            if not isinstance(ticker, str) or not ticker.strip():
                raise ValueError("sector_map tickers must be non-empty strings")
            if not isinstance(sector_id, str) or not sector_id.strip():
                raise ValueError(f"sector_map[{ticker}] must be a non-empty sector id")
            sectors[ticker] = sector_id
        object.__setattr__(self, "sector_map", MappingProxyType(sectors))

        if not isinstance(self.factor_tags, Mapping):
            raise TypeError("factor_tags must be a mapping")
        tags: dict[str, frozenset[str]] = {}
        for factor_id, tickers in self.factor_tags.items():
            if not isinstance(factor_id, str) or not factor_id.strip():
                raise ValueError("factor ids must be non-empty strings")
            if not isinstance(tickers, frozenset):
                raise TypeError(f"factor_tags[{factor_id}] must be a frozenset")
            if not tickers:
                raise ValueError(f"factor_tags[{factor_id}] must not be empty")
            if any(not isinstance(item, str) or not item.strip() for item in tickers):
                raise ValueError(f"factor_tags[{factor_id}] must contain tickers")
            tags[factor_id] = tickers
        object.__setattr__(self, "factor_tags", MappingProxyType(tags))

        for name in (
            "default_contract_multiplier",
            "money_quantum",
            "max_gross_notional",
            "max_symbol_notional",
            "max_sector_notional",
            "max_theme_notional",
            "max_factor_notional",
        ):
            _require_positive(name, getattr(self, name))

    def sector_for(self, ticker: str) -> str | None:
        """Resolve a sector, returning None rather than guessing.

        Mirrors ``signals.market_regime.sector_map``: an unmapped ticker is an
        explicit unknown, never a silent fallback to one sector.
        """

        return self.sector_map.get(ticker)

    def factors_for(self, ticker: str) -> tuple[str, ...]:
        """Return every factor a ticker belongs to, in stable order."""

        return tuple(
            sorted(
                factor_id
                for factor_id, tickers in self.factor_tags.items()
                if ticker in tickers
            )
        )

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Decimal):
                payload[field.name] = _decimal_text(value)
            elif isinstance(value, Mapping):
                payload[field.name] = {
                    key: sorted(item) if isinstance(item, frozenset) else item
                    for key, item in value.items()
                }
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


# Structural overlap among names the acceptance fixture trades together.
# AMD sits in both semiconductor and AI infrastructure; SOXL sits in both
# semiconductor and leveraged index, so the buckets deliberately intersect.
_MVP_FACTOR_TAGS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "semiconductor": frozenset({"AMD", "NVDA", "AVGO", "SOXL"}),
        "leveraged_index": frozenset({"SOXL", "TQQQ"}),
        "ai_infrastructure": frozenset({"NBIS", "APLD", "IREN", "AMD", "NVDA"}),
    }
)


# Sector ids follow the canonical curated map in
# ``signals/market_regime/sector_map.py`` (sector-ETF symbols).  That module
# imports pandas and bar loaders, so the pure exposure path takes the mapping
# as versioned configuration instead of importing it; runtime assembly is
# responsible for keeping the two in sync.
_MVP_SECTOR_MAP: Mapping[str, str] = MappingProxyType(
    {
        "AMD": "XLK",
        "NVDA": "XLK",
        "AVGO": "XLK",
        "SOXL": "XLK",
        "TQQQ": "XLK",
        "NBIS": "XLK",
        "APLD": "XLK",
        "IREN": "XLK",
        "JPM": "XLF",
    }
)


MVP_PORTFOLIO_CONFIG = PortfolioConfig(
    config_version="nervous-system-portfolio@1",
    sector_map=_MVP_SECTOR_MAP,
    factor_tags=_MVP_FACTOR_TAGS,
    default_contract_multiplier=Decimal("100"),
    money_quantum=Decimal("0.01"),
    max_gross_notional=Decimal("150000.00"),
    max_symbol_notional=Decimal("25000.00"),
    max_sector_notional=Decimal("60000.00"),
    max_theme_notional=Decimal("50000.00"),
    max_factor_notional=Decimal("60000.00"),
)


__all__ = ["MVP_PORTFOLIO_CONFIG", "PortfolioConfig"]
