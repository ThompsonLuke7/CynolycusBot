from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from strategies.intraday_structure.models import OptionsContext, StructuralLevel, utc_datetime


class OptionsProvider(Protocol):
    def context(self, symbol: str, timestamp: datetime, spot: float) -> OptionsContext: ...


class NullOptionsProvider:
    def context(self, symbol: str, timestamp: datetime, spot: float) -> OptionsContext:
        return OptionsContext(timestamp=timestamp, source="none", warnings=("options_unavailable",))


class DealerSnapshotOptionsProvider:
    """Adapter over existing dealer-positioning level JSON files.

    Snapshot time must be at or before the decision time. This makes accidental
    use of a current snapshot in historical replay fail closed instead of leak.
    """

    def __init__(self, root: str | Path = "Data/dealer_positioning", max_age_minutes: int = 1440) -> None:
        self.root = Path(root)
        self.max_age = timedelta(minutes=max_age_minutes)

    def context(self, symbol: str, timestamp: datetime, spot: float) -> OptionsContext:
        path = self.root / symbol.upper() / "live_gamma_levels.json"
        if not path.exists():
            return OptionsContext(timestamp=timestamp, source="none", warnings=("dealer_snapshot_missing",))
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            snapshot_ts = utc_datetime(raw["timestamp"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return OptionsContext(timestamp=timestamp, source="none", warnings=("dealer_snapshot_invalid",))
        decision_ts = utc_datetime(timestamp)
        if snapshot_ts > decision_ts:
            return OptionsContext(timestamp=decision_ts, source="none", warnings=("future_dealer_snapshot_rejected",))
        age = decision_ts - snapshot_ts
        warnings: list[str] = []
        if age > self.max_age:
            warnings.append("dealer_snapshot_stale")
        total_gex = _optional_float(raw.get("total_gex"))
        gamma_regime = "positive" if total_gex is not None and total_gex >= 0 else "negative" if total_gex is not None else "unknown"
        level_specs = [
            ("put_wall", "options_put_wall", "support", 0.90),
            ("call_wall", "options_call_wall", "resistance", 0.90),
            ("gamma_flip", "options_gamma_flip", "both", 0.82),
            ("nearest_magnet", "options_gamma_magnet", "both", 0.78),
            ("next_magnet_above", "options_gamma_magnet_above", "resistance", 0.70),
            ("next_magnet_below", "options_gamma_magnet_below", "support", 0.70),
        ]
        levels = tuple(
            StructuralLevel(price=value, level_type=kind, strength=strength, directionality=direction,
                            metadata={"snapshot_timestamp": snapshot_ts.isoformat(), "estimated_positioning": True})
            for key, kind, direction, strength in level_specs
            if (value := _optional_float(raw.get(key))) is not None and value > 0
        )
        return OptionsContext(
            timestamp=snapshot_ts, source="dealer_positioning_static",
            gamma_regime=gamma_regime, gamma_flip=_optional_float(raw.get("gamma_flip")),
            call_wall=_optional_float(raw.get("call_wall")), put_wall=_optional_float(raw.get("put_wall")),
            local_gamma_exposure=total_gex,
            strike_congestion_score=float(np.clip(len(levels) / 6.0, 0.0, 1.0)),
            levels=levels, warnings=tuple(warnings),
        )


class CboeStrikeSnapshotOptionsProvider:
    """Causal adapter for stored CBOE strike snapshots (OI/volume/IV/delta).

    This is static positioning/aggregated activity, not a reconstruction of
    dealer inventory and not a substitute for OPRA trade-by-trade flow.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"ticker", "snapshot_timestamp", "strike", "side", "open_interest", "volume"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"CBOE strike snapshot missing columns: {sorted(missing)}")
        self.frame = frame.copy()
        self.frame["snapshot_timestamp"] = pd.to_datetime(self.frame["snapshot_timestamp"], utc=True, errors="coerce")
        self.frame["ticker"] = self.frame["ticker"].astype(str).str.upper()

    @classmethod
    def from_parquet(cls, path: str | Path) -> "CboeStrikeSnapshotOptionsProvider":
        return cls(pd.read_parquet(path))

    def context(self, symbol: str, timestamp: datetime, spot: float) -> OptionsContext:
        decision = pd.Timestamp(utc_datetime(timestamp))
        rows = self.frame[(self.frame["ticker"] == symbol.upper()) & (self.frame["snapshot_timestamp"] <= decision)]
        if rows.empty:
            return OptionsContext(timestamp=timestamp, source="none", warnings=("cboe_snapshot_unavailable_asof",))
        latest_ts = rows["snapshot_timestamp"].max()
        rows = rows[rows["snapshot_timestamp"] == latest_ts].copy()
        rows["strike"] = pd.to_numeric(rows["strike"], errors="coerce")
        rows["open_interest"] = pd.to_numeric(rows["open_interest"], errors="coerce").fillna(0.0)
        rows["volume"] = pd.to_numeric(rows["volume"], errors="coerce").fillna(0.0)
        calls = rows[rows["side"].astype(str).str.upper().str.startswith("C")]
        puts = rows[rows["side"].astype(str).str.upper().str.startswith("P")]
        call_wall = _weighted_wall(calls, above=spot, mode="max")
        put_wall = _weighted_wall(puts, above=spot, mode="min")
        concentrations = rows.groupby("strike", as_index=False)[["open_interest", "volume"]].sum()
        total_oi = float(concentrations["open_interest"].sum())
        top_share = float(concentrations["open_interest"].max() / total_oi) if total_oi > 0 and not concentrations.empty else 0.0
        levels = []
        if call_wall is not None:
            levels.append(StructuralLevel(call_wall, "options_oi_call_wall", 0.82, directionality="resistance"))
        if put_wall is not None:
            levels.append(StructuralLevel(put_wall, "options_oi_put_wall", 0.82, directionality="support"))
        return OptionsContext(
            timestamp=latest_ts.to_pydatetime(), source="cboe_static_snapshot",
            call_wall=call_wall, put_wall=put_wall,
            strike_congestion_score=float(np.clip(top_share * 4.0, 0.0, 1.0)),
            levels=tuple(levels), warnings=("dealer_inventory_not_observable",),
        )


@dataclass(frozen=True)
class OptionFlowPrint:
    timestamp: datetime
    ticker: str
    strike: float
    expiration: str
    option_type: str
    size: int
    premium: float
    price: float
    bid: float | None = None
    ask: float | None = None
    delta: float | None = None
    gamma: float | None = None
    classification: str | None = None
    is_sweep: bool | None = None
    is_block: bool | None = None

    @property
    def signed_delta_premium(self) -> float:
        execution = self.classification or _execution_classification(self.price, self.bid, self.ask)
        sign = 1.0 if execution == "ask" else -1.0 if execution == "bid" else 0.0
        cp_sign = 1.0 if self.option_type.upper().startswith("C") else -1.0
        delta = abs(self.delta) if self.delta is not None else 0.5
        return sign * cp_sign * self.premium * delta


class LiveOptionsFlowProvider:
    """Provider-neutral in-memory flow aggregator for future OPRA adapters."""

    def __init__(self, window_minutes: int = 15, max_prints: int = 20_000) -> None:
        self.window = timedelta(minutes=window_minutes)
        self._prints: dict[str, deque[OptionFlowPrint]] = defaultdict(lambda: deque(maxlen=max_prints))

    def ingest(self, print_: OptionFlowPrint) -> None:
        symbol = print_.ticker.upper()
        if self._prints[symbol] and print_.timestamp < self._prints[symbol][-1].timestamp:
            raise ValueError("live options flow must be ingested chronologically")
        self._prints[symbol].append(print_)

    def context(self, symbol: str, timestamp: datetime, spot: float) -> OptionsContext:
        decision = utc_datetime(timestamp)
        rows = [p for p in self._prints[symbol.upper()] if utc_datetime(p.timestamp) <= decision and decision - utc_datetime(p.timestamp) <= self.window]
        if not rows:
            return OptionsContext(timestamp=decision, source="none", warnings=("live_flow_unavailable",))
        midpoint = decision - self.window / 2
        early = [p for p in rows if utc_datetime(p.timestamp) < midpoint]
        late = [p for p in rows if utc_datetime(p.timestamp) >= midpoint]
        call_early = sum(p.premium for p in early if p.option_type.upper().startswith("C"))
        call_late = sum(p.premium for p in late if p.option_type.upper().startswith("C"))
        put_early = sum(p.premium for p in early if p.option_type.upper().startswith("P"))
        put_late = sum(p.premium for p in late if p.option_type.upper().startswith("P"))
        total = sum(p.premium for p in rows)
        short = sum(p.premium for p in rows if _days_to_expiry(p.expiration, decision) <= 7)
        gamma_exposure = sum((p.gamma or 0.0) * p.size * 100.0 * spot for p in rows)
        return OptionsContext(
            timestamp=max(utc_datetime(p.timestamp) for p in rows), source="live_options_flow",
            local_gamma_exposure=gamma_exposure,
            live_call_flow_acceleration=(call_late - call_early) / max(total, 1.0),
            live_put_flow_acceleration=(put_late - put_early) / max(total, 1.0),
            net_delta_premium=sum(p.signed_delta_premium for p in rows),
            short_dated_flow_ratio=short / max(total, 1.0),
            warnings=("flow_direction_is_execution_inference", "dealer_inventory_not_observable"),
        )


class CompositeOptionsProvider:
    def __init__(self, *providers: OptionsProvider) -> None:
        self.providers = providers

    def context(self, symbol: str, timestamp: datetime, spot: float) -> OptionsContext:
        contexts = [provider.context(symbol, timestamp, spot) for provider in self.providers]
        available = [ctx for ctx in contexts if ctx.source != "none"]
        if not available:
            warnings = tuple(dict.fromkeys(w for ctx in contexts for w in ctx.warnings))
            return OptionsContext(timestamp=timestamp, source="none", warnings=warnings or ("options_unavailable",))
        static = next((ctx for ctx in available if "static" in ctx.source or "dealer" in ctx.source), available[0])
        flow = next((ctx for ctx in available if "flow" in ctx.source), None)
        return OptionsContext(
            timestamp=max((ctx.timestamp for ctx in available if ctx.timestamp), default=utc_datetime(timestamp)),
            source="+".join(ctx.source for ctx in available), gamma_regime=static.gamma_regime,
            gamma_flip=static.gamma_flip, call_wall=static.call_wall, put_wall=static.put_wall,
            local_gamma_exposure=(flow.local_gamma_exposure if flow and flow.local_gamma_exposure is not None else static.local_gamma_exposure),
            strike_congestion_score=static.strike_congestion_score,
            live_call_flow_acceleration=flow.live_call_flow_acceleration if flow else 0.0,
            live_put_flow_acceleration=flow.live_put_flow_acceleration if flow else 0.0,
            net_delta_premium=flow.net_delta_premium if flow else 0.0,
            short_dated_flow_ratio=flow.short_dated_flow_ratio if flow else 0.0,
            levels=tuple(level for ctx in available for level in ctx.levels),
            warnings=tuple(dict.fromkeys(w for ctx in available for w in ctx.warnings)),
        )


def _weighted_wall(rows: pd.DataFrame, *, above: float, mode: str) -> float | None:
    if rows.empty:
        return None
    valid = rows[rows["strike"] > above] if mode == "max" else rows[rows["strike"] < above]
    if valid.empty:
        return None
    idx = valid["open_interest"].idxmax()
    value = float(valid.loc[idx, "strike"])
    return value if math.isfinite(value) and value > 0 else None


def _execution_classification(price: float, bid: float | None, ask: float | None) -> str:
    if ask is not None and price >= ask:
        return "ask"
    if bid is not None and price <= bid:
        return "bid"
    return "mid"


def _days_to_expiry(expiration: str, timestamp: datetime) -> int:
    try:
        return max(0, (pd.Timestamp(expiration).date() - timestamp.date()).days)
    except (TypeError, ValueError):
        return 999


def _optional_float(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None
