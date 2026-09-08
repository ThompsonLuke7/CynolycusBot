"""Versioned, paper-only configuration for the momentum scalper v1."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("momentum_scalper_v1.json")


@dataclass(frozen=True)
class ScannerConfigV1:
    min_price: float = 1.0
    max_price: float = 20.0
    min_gap_pct: float = 10.0
    min_premarket_volume: int = 250_000
    min_rvol: float = 5.0
    max_float: float = 10_000_000.0
    rvol_lookback_sessions: int = 20
    min_rvol_history_sessions: int = 5
    max_spread_pct: float = 1.5
    max_quote_age_seconds: float = 2.0
    required_quote_feed: str = "SIP"
    min_daily_resistance_distance_pct: float = 5.0
    require_material_catalyst: bool = True
    require_quote: bool = True


@dataclass(frozen=True)
class PatternConfigV1:
    min_breakout_volume_ratio: float = 1.5
    max_chase_pct: float = 2.0
    flat_top_tolerance_pct: float = 0.5
    min_flag_bars: int = 3
    max_flag_retracement_pct: float = 20.0
    max_pullback_retracement_pct: float = 25.0


@dataclass(frozen=True)
class RiskConfigV1:
    max_concurrent_positions: int = 1
    max_trades_per_day: int = 3
    max_daily_loss_r: float = 3.0
    max_risk_per_trade_r: float = 1.0
    max_consecutive_losses: int = 2
    max_position_notional: float = 10_000.0
    max_quote_participation: float = 0.10
    partial_at_r: float = 2.0
    partial_fraction: float = 0.50
    max_hold_minutes: int = 30
    entry_expiry_seconds: int = 15


@dataclass(frozen=True)
class MomentumScalperConfigV1:
    version: str = "momentum_scalper_v1"
    paper_only: bool = True
    session_start_et: str = "04:00"
    new_entry_cutoff_et: str = "11:00"
    scanner: ScannerConfigV1 = ScannerConfigV1()
    patterns: PatternConfigV1 = PatternConfigV1()
    risk: RiskConfigV1 = RiskConfigV1()

    def validate(self) -> MomentumScalperConfigV1:
        if self.version != "momentum_scalper_v1":
            raise ValueError("unsupported momentum scalper configuration version")
        if not self.paper_only:
            raise ValueError("momentum scalper v1 must remain paper_only")
        if self.scanner.min_price <= 0 or self.scanner.max_price <= self.scanner.min_price:
            raise ValueError("scanner price range is invalid")
        if self.scanner.rvol_lookback_sessions < self.scanner.min_rvol_history_sessions:
            raise ValueError("RVOL lookback must cover the required history")
        if not 0 < self.risk.partial_fraction <= 1:
            raise ValueError("partial_fraction must be in (0, 1]")
        if self.risk.max_concurrent_positions < 1 or self.risk.max_trades_per_day < 1:
            raise ValueError("portfolio limits must be positive")
        return self


def _section(payload: dict, name: str, cls):
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cls(**value)


def load_config(path: Path = CONFIG_PATH) -> MomentumScalperConfigV1:
    """Load a strict, versioned configuration without any live-mode escape hatch."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("momentum scalper config must be an object")
    known = {
        "version", "paper_only", "session_start_et", "new_entry_cutoff_et",
        "scanner", "patterns", "risk",
    }
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown momentum scalper config fields: {unknown}")
    return MomentumScalperConfigV1(
        version=str(payload.get("version", "momentum_scalper_v1")),
        paper_only=bool(payload.get("paper_only", True)),
        session_start_et=str(payload.get("session_start_et", "04:00")),
        new_entry_cutoff_et=str(payload.get("new_entry_cutoff_et", "11:00")),
        scanner=_section(payload, "scanner", ScannerConfigV1),
        patterns=_section(payload, "patterns", PatternConfigV1),
        risk=_section(payload, "risk", RiskConfigV1),
    ).validate()


__all__ = [
    "CONFIG_PATH", "MomentumScalperConfigV1", "PatternConfigV1", "RiskConfigV1",
    "ScannerConfigV1", "load_config",
]
