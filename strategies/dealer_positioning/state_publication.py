"""Publish DEALER states so the governed policy can actually see gamma structure.

``nervous_system_adapter.adapt_dealer_state`` has been written, tested, and
uncalled since it was built. The consequence is quiet but total: the policy's
``policy.modifier.dealer_regime`` rule at ``core/nervous_system/policy/rules.py``
has never fired, and every governed decision has recorded
``CONTEXT_DEALER_UNAVAILABLE`` instead. This module is the missing caller.

What regime gets published is the design decision worth stating. Gamma carries
little information about direction and possibly some about dispersion, so this
module maps structure onto the *volatility* regimes only -- ``SHORT_GAMMA``,
``PINNING``, ``POSITIVE_GAMMA`` -- and never onto ``UPSIDE_ACCELERATION`` or
``DOWNSIDE_ACCELERATION``. Those two are directional claims that an
OI-sign-assumption proxy has no standing to make.

Confidence gating is the second. Dealer sign is inferred, not observed, and the
inference is far weaker on a single name than on SPY. Below the structure floor
the state is still published, but with no regime asserted, which resolves to
``UNKNOWN`` and its 0.75 multiplier -- the same conservative sizing the policy
already applies when no dealer state exists at all. Publishing a weak snapshot
as ``NEUTRAL_GAMMA`` would be worse than publishing nothing: neutral multiplies
by 1.0, so a low-confidence read would *raise* size relative to today.

Publication never raises. An unreachable state store degrades the decision to
the no-dealer-context branch, which is exactly where it already is.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core.nervous_system.contracts.quality import LineageRef
from strategies.dealer_positioning.confidence import (
    assess_chain_quality,
    liquidity_tier,
    sign_confidence,
    structure_confidence,
)
from strategies.dealer_positioning.nervous_system_adapter import adapt_dealer_state

logger = logging.getLogger(__name__)

# Scope precedence when a ticker has rows in several. The adapter refuses to
# choose among same-ticker scopes, so the caller must; monthly is the scope the
# 4-hour modules actually buy.
SCOPE_PRECEDENCE = ("through_month", "daily_week", "two_months")

# Below this structure confidence no regime is asserted. Not tuned -- set at the
# point where fewer than half the strikes carried usable exposure.
MIN_STRUCTURE_CONFIDENCE = 0.50

# Deadband on the scale-free imbalance (estimated net exposure / total absolute
# exposure) before a gamma regime is called either way. A near-zero imbalance is
# not a positive-gamma reading, it is an absence of one.
IMBALANCE_DEADBAND = 0.05

# `pinning_score` is |spot - exposure-weighted strike| in units of average strike
# spacing, so a small value means price is sitting on the weighted centre of the
# gamma distribution.
PINNING_SCORE_MAX = 0.50

# DEALER freshness in the snapshot profile is 4 hours; the state stays valid for
# a trading day so an overnight decision can still read the evening capture.
DEFAULT_VALID_FOR = timedelta(hours=24)

_STATE_STORE_UNAVAILABLE = "STATE_STORE_UNAVAILABLE"


def select_scope_row(rows: pd.DataFrame) -> pd.Series | None:
    """Pick one canonical scope row for a ticker, by declared precedence."""
    if rows.empty:
        return None
    for scope in SCOPE_PRECEDENCE:
        match = rows[rows["scope"].astype(str) == scope]
        if not match.empty:
            return match.iloc[0]
    return None


def snapshot_confidence(row: Mapping[str, Any]) -> tuple[float, float, str]:
    """Structure and sign confidence for one captured summary row.

    The summary row is post-aggregation, so chain quality is reconstructed from
    the counts the capture recorded rather than from contract rows.
    """
    ticker = str(row.get("ticker") or row.get("symbol") or "").upper()
    tier = liquidity_tier(
        ticker,
        avg_dollar_volume_20d=_float(row.get("avg_dollar_volume_20d")),
        market_cap=_float(row.get("market_cap")),
    )
    quality = assess_chain_quality(
        pd.DataFrame(
            {
                "strike": range(int(_float(row.get("row_count")) or 0)),
                "gamma": [1.0] * int(_float(row.get("row_count")) or 0),
                "open_interest": [1.0] * int(_float(row.get("row_count")) or 0),
                "iv": [1.0] * int(_float(row.get("row_count")) or 0),
            }
        )
    )
    structure = structure_confidence(quality)
    return structure, sign_confidence(tier), tier


def dealer_regime_from_structure(
    row: Mapping[str, Any],
    *,
    structure: float,
    min_structure_confidence: float = MIN_STRUCTURE_CONFIDENCE,
) -> str | None:
    """Map captured structure onto a volatility regime, or onto nothing.

    Returns ``None`` when no regime should be asserted -- the adapter then
    resolves the state to ``UNKNOWN``, which the policy sizes conservatively.

    Deliberately cannot return a directional regime. Gamma structure is evidence
    about dispersion, not about which way price goes, and the enum's
    acceleration members would smuggle a directional claim into a sizing rule.
    """
    if structure < float(min_structure_confidence):
        return None
    imbalance = _float(row.get("dealer_imbalance"))
    if imbalance is None:
        return None
    if imbalance < -IMBALANCE_DEADBAND:
        # Estimated dealer-short gamma: hedging amplifies moves, so expect wider
        # dispersion and size down.
        return "SHORT_GAMMA"
    if imbalance > IMBALANCE_DEADBAND:
        pinning = _float(row.get("pinning_score"))
        if pinning is not None and pinning <= PINNING_SCORE_MAX:
            return "PINNING"
        return "POSITIVE_GAMMA"
    return "NEUTRAL_GAMMA"


def _artifact_sha256(path: Path) -> str:
    """Hash the snapshot file so a published state names the bytes it came from.

    An unreadable artifact hashes its path instead of failing: the state is
    still worth publishing, and the locator records which file it claimed.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return hashlib.sha256(str(path).encode()).hexdigest()
    return digest.hexdigest()


def _row_captured_at(row: Mapping[str, Any]) -> datetime | None:
    raw = row.get("captured_at")
    if raw is None:
        return None
    stamp = pd.to_datetime(raw, errors="coerce", utc=True)
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    return stamp.to_pydatetime()


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


def build_dealer_states(
    summary: pd.DataFrame,
    *,
    captured_at: datetime,
    snapshot_path: Path,
    valid_for: timedelta = DEFAULT_VALID_FOR,
    min_structure_confidence: float = MIN_STRUCTURE_CONFIDENCE,
) -> tuple[list[Any], dict[str, int]]:
    """Adapt one captured summary frame into DEALER states.

    Returns the states plus a count of why rows were dropped, so a caller can
    tell "nothing to publish" apart from "everything failed validation".
    """
    skipped: dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    if summary is None or summary.empty:
        return [], {"empty_summary": 1}

    work = summary.copy()
    if "ticker" not in work.columns and "symbol" in work.columns:
        work["ticker"] = work["symbol"]
    if "ticker" not in work.columns or "scope" not in work.columns:
        return [], {"missing_key_columns": 1}

    captured_at_utc = captured_at.astimezone(timezone.utc)
    lineage = (
        LineageRef(
            source_id="dealer-capture",
            content_hash=_artifact_sha256(snapshot_path),
            record_locator=snapshot_path.name,
        ),
    )

    states: list[Any] = []
    for ticker, rows in work.groupby(work["ticker"].astype(str).str.upper(), sort=True):
        row = select_scope_row(rows)
        if row is None:
            _skip("no_canonical_scope")
            continue
        payload = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
        payload["ticker"] = ticker

        structure, sign, tier = snapshot_confidence(payload)
        payload["structure_confidence"] = structure
        payload["sign_confidence"] = sign
        payload["liquidity_tier"] = tier
        # The capture carries no direction field, and must not acquire one here.
        payload.pop("dealer_direction", None)
        payload.pop("regime", None)
        regime = dealer_regime_from_structure(
            payload, structure=structure, min_structure_confidence=min_structure_confidence
        )
        if regime is None:
            payload.pop("dealer_regime", None)
        else:
            payload["dealer_regime"] = regime

        # Capture runs over minutes, so each row carries its own capture time.
        # Passing a frame-wide timestamp makes the adapter reject every row whose
        # real capture differed from it.
        row_captured_at = _row_captured_at(payload) or captured_at_utc
        try:
            states.append(
                adapt_dealer_state(
                    payload,
                    None,
                    captured_at=row_captured_at,
                    valid_until=row_captured_at + valid_for,
                    lineage=lineage,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the rest
            _skip(type(exc).__name__)
            logger.debug("dealer state rejected for %s: %s", ticker, exc)
    return states, skipped


def publish_dealer_states(
    summary: pd.DataFrame,
    *,
    captured_at: datetime,
    snapshot_path: Path,
    valid_for: timedelta = DEFAULT_VALID_FOR,
    tickers: Sequence[str] | None = None,
) -> Mapping[str, object]:
    """Build and persist DEALER states. Never raises."""
    frame = summary
    if tickers is not None and summary is not None and not summary.empty:
        wanted = {str(t).upper() for t in tickers}
        key = "ticker" if "ticker" in summary.columns else "symbol"
        frame = summary[summary[key].astype(str).str.upper().isin(wanted)]

    states, skipped = build_dealer_states(
        frame, captured_at=captured_at, snapshot_path=Path(snapshot_path), valid_for=valid_for
    )
    result: dict[str, object] = {"published": 0, "skipped": skipped, "status": "PUBLISHED"}
    if not states:
        result["status"] = "NO_STATES"
        return result

    try:
        from core.nervous_system.config.runtime import NervousSystemSettings
        from core.nervous_system.persistence.database import (
            create_database_engine,
            create_session_factory,
        )
        from core.nervous_system.persistence.uow import UnitOfWork

        settings = NervousSystemSettings.from_env()
        session_factory = create_session_factory(create_database_engine(settings))
        with UnitOfWork(session_factory) as uow:
            uow.states.insert_states_idempotently(states)
            uow.commit()
    except Exception as exc:  # noqa: BLE001 - an unreachable store is a veto, not a crash
        logger.warning(
            "dealer-state publication failed (%s): %s", _STATE_STORE_UNAVAILABLE, type(exc).__name__
        )
        result["status"] = "FAILED"
        return result

    result["published"] = len(states)
    logger.info("dealer states published: %d (skipped %s)", len(states), skipped or "none")
    return result
