"""Publish one TICKER state per name the Meta runner may act on this pass.

``adapt_scored_ticker_state`` has existed and been tested since the nervous-system
MVP landed, with no production caller — so ``state_records`` never held a TICKER
row and ``SnapshotBuilder`` resolved ``ticker_state=None`` for every name. That
is one of the required states whose absence kept Meta from submitting a single
order through the governed path on 2026-08-20 and 2026-08-21.

Three things this module is careful about, because each was a live defect:

*   **The liquidity metric is dollars, not a percentile.** ``policy.rule.liquidity``
    reads ``metrics["dollar_volume_20d"]`` and compares it to $5,000,000. The Meta
    matrix has no such column; its only liquidity field is
    ``dollar_vol_pctile_252``, a rank in [0, 1]. Publishing the percentile under
    that name would compare 0.6 against 5e6 and veto every name. The metric is
    computed here from daily bars instead.

*   **The reference price comes from the bar, not the matrix.** The matrix carries
    no OHLCV at all. ``adapt_scored_ticker_state`` wants an explicitly selected
    4H bar whose timestamp equals the decision bar, and refuses any other.

*   **Publication never blocks trading.** A state store that is down must degrade
    to "the governed path will veto for want of state", which is a refusal the
    runner already handles, not an exception that takes the process down mid-plan.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import logging
from pathlib import Path

import pandas as pd

from core.nervous_system.contracts.quality import LineageRef

from .nervous_system_adapter import adapt_scored_ticker_state

logger = logging.getLogger(__name__)

_BARS_4H_ROOT = Path("Data/shared/bars/4h")
_BARS_1D_ROOT = Path("Data/shared/bars/1d")

# `policy.rule.liquidity` reads exactly this key.
LIQUIDITY_METRIC = "dollar_volume_20d"
_LIQUIDITY_WINDOW = 20

# Match the freshness rule that consumes TICKER states
# (config/freshness.py MVP_POLICY_DEFAULTS[StateType.TICKER] is 6h). A 4H bar's
# state should stop being valid around the time the next bar's would replace it.
_STATE_VALIDITY = timedelta(hours=6)

_STATE_STORE_UNAVAILABLE = "TICKER_STATE_STORE_UNAVAILABLE"


def _artifact_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dollar_volume_20d(ticker: str, *, root: Path = _BARS_1D_ROOT) -> float | None:
    """Mean close*volume over the last 20 completed daily bars, or None.

    None rather than a default: an unknown liquidity is not a liquid name, and
    `policy.rule.liquidity` already has a reason code for a metric it cannot
    read. Inventing a number here would launder a data gap into a permission.
    """

    path = root / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, columns=["close", "volume"])
    except Exception:  # noqa: BLE001 - a corrupt cache file is a missing metric
        return None
    if len(frame) < _LIQUIDITY_WINDOW:
        return None
    tail = frame.tail(_LIQUIDITY_WINDOW)
    value = float((tail["close"] * tail["volume"]).mean())
    if not pd.notna(value) or value <= 0.0:
        return None
    return value


def _selected_4h_bar(
    ticker: str,
    *,
    decision_bar: pd.Timestamp,
    root: Path = _BARS_4H_ROOT,
) -> tuple[dict[str, object], str] | None:
    """The one 4H bar stamped exactly at the decision bar, plus its lineage hash."""

    path = root / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    if "timestamp" not in frame.columns or frame.empty:
        return None
    stamps = pd.to_datetime(frame["timestamp"], utc=True)
    match = frame[stamps == decision_bar]
    if match.empty:
        # No exact bar: the cache has not caught up to this decision bar for this
        # name. Publishing the nearest bar instead would silently misdate the
        # state, so this name simply has no TICKER state this pass.
        return None
    row = match.iloc[-1].to_dict()
    row["timestamp"] = decision_bar
    row["ticker"] = ticker
    return row, _artifact_sha256(path)


def build_ticker_states(
    scored: pd.DataFrame,
    *,
    tickers: Iterable[str],
    decision_bar: datetime,
    available_at: datetime,
    matrix_path: Path,
    bars_4h_root: Path = _BARS_4H_ROOT,
    bars_1d_root: Path = _BARS_1D_ROOT,
) -> tuple[list[object], dict[str, str]]:
    """Adapt one TICKER state per resolvable name.

    Returns the states and a per-ticker reason for every name skipped, so the
    caller can log which names will have no state rather than discovering it as
    a policy veto later.
    """

    skipped: dict[str, str] = {}
    states: list[object] = []
    if scored is None or scored.empty:
        return states, {str(t): "no_scored_frame" for t in tickers}

    bar_utc = pd.Timestamp(decision_bar).tz_convert("UTC") if pd.Timestamp(decision_bar).tzinfo else pd.Timestamp(decision_bar).tz_localize("UTC")
    matrix_hash = _artifact_sha256(matrix_path)
    valid_until = available_at + _STATE_VALIDITY

    frame = scored.reset_index() if "ticker" not in scored.columns else scored
    by_ticker = {str(row["ticker"]).upper(): row for _idx, row in frame.iterrows()}

    for raw in tickers:
        ticker = str(raw).upper()
        scored_row = by_ticker.get(ticker)
        if scored_row is None:
            skipped[ticker] = "not_in_scored_frame"
            continue
        selected = _selected_4h_bar(ticker, decision_bar=bar_utc, root=bars_4h_root)
        if selected is None:
            skipped[ticker] = "no_exact_4h_bar"
            continue
        bar_row, bar_hash = selected

        liquidity = dollar_volume_20d(ticker, root=bars_1d_root)
        row = dict(scored_row)
        row["timestamp"] = bar_utc
        if liquidity is not None:
            row[LIQUIDITY_METRIC] = liquidity

        try:
            state = adapt_scored_ticker_state(
                row,
                bar_row,
                bar_ticker=ticker,
                decision_bar=bar_utc.to_pydatetime(),
                available_at=available_at,
                valid_until=valid_until,
                matrix_lineage=(
                    LineageRef(
                        source_id=str(matrix_path),
                        content_hash=matrix_hash,
                        record_locator=f"meta_ranker_matrix:{ticker}:{bar_utc.isoformat()}",
                    ),
                ),
                bar_lineage=(
                    LineageRef(
                        source_id=str(bars_4h_root / f"{ticker}.parquet"),
                        content_hash=bar_hash,
                        record_locator=f"bars_4h:{ticker}:{bar_utc.isoformat()}",
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the rest
            skipped[ticker] = f"adapt_failed:{type(exc).__name__}"
            continue
        states.append(state)
    return states, skipped


def publish_ticker_states(
    scored: pd.DataFrame,
    *,
    tickers: Sequence[str],
    decision_bar: datetime,
    matrix_path: Path,
    available_at: datetime | None = None,
) -> Mapping[str, object]:
    """Publish TICKER states, owning the unit of work. Never raises.

    ``available_at`` defaults to now, which is correct for a live pass: the
    matrix row became readable when this process read it. A backfill must pass
    the real availability instead, or replay will not reproduce the decision.
    """

    observed_at = available_at or datetime.now(timezone.utc)
    states, skipped = build_ticker_states(
        scored,
        tickers=tickers,
        decision_bar=decision_bar,
        available_at=observed_at,
        matrix_path=Path(matrix_path),
    )
    result: dict[str, object] = {
        "published": 0,
        "skipped": skipped,
        "status": "PUBLISHED",
    }
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
            "ticker-state publication failed (%s): %s",
            _STATE_STORE_UNAVAILABLE,
            type(exc).__name__,
        )
        result["status"] = "FAILED"
        return result

    result["published"] = len(states)
    return result


__all__ = [
    "LIQUIDITY_METRIC",
    "build_ticker_states",
    "dollar_volume_20d",
    "publish_ticker_states",
]
