"""Publish momentum TICKER states into the nervous-system registry.

Momentum's snapshot cannot resolve without a TICKER state for the name being
decided, and Meta publishes states only for its own top-K plus what it manages.
Any momentum name outside that set has no state, and the governed path refuses
a decision it cannot evidence.

This reuses Meta's builder rather than copying it: the adapter reads generic
columns (close, setup, the state strings) and takes its producer identifiers as
parameters, so the only momentum-specific part is which matrix the row and
lineage came from.

Two producers publishing TICKER state for the same name is intended and safe.
The state describes the TICKER at a bar, not the strategy that looked at it,
and ``_stable_state_id`` keys on ticker/bar/lineage/content rather than on the
producer -- so identical content converges on one row, and genuinely different
feature matrices produce two candidates the selector orders deterministically.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from signals.meta_context.meta_ranker.ticker_state_publication import (
    build_ticker_states as _build_ticker_states,
)


logger = logging.getLogger(__name__)

PRODUCER = "strategies.momentum_expansion"
MODEL_VERSION = "momentum-expansion-adapter@1"
FEATURE_VERSION = "momentum-features-4h@1"
CONFIG_VERSION = "momentum-expansion-ticker@1"

_STATE_STORE_UNAVAILABLE = "TICKER_STATE_STORE_UNAVAILABLE"


def build_momentum_ticker_states(
    scored: pd.DataFrame,
    *,
    tickers: Sequence[str],
    decision_bar: datetime,
    available_at: datetime,
    matrix_path: Path,
) -> tuple[list[object], dict[str, str]]:
    """Adapt one TICKER state per resolvable momentum name."""

    return _build_ticker_states(
        scored,
        tickers=tickers,
        decision_bar=decision_bar,
        available_at=available_at,
        matrix_path=matrix_path,
        producer_identity={
            "producer": PRODUCER,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "config_version": CONFIG_VERSION,
        },
        record_prefix="momentum_features_4h",
    )


def publish_momentum_ticker_states(
    scored: pd.DataFrame,
    *,
    tickers: Sequence[str],
    decision_bar: datetime,
    matrix_path: Path,
    available_at: datetime | None = None,
) -> Mapping[str, object]:
    """Publish momentum TICKER states, owning the unit of work. Never raises.

    Publication failure degrades the governed path to "cannot decide", never
    the 4H pass to "cannot trade": momentum's direct execution path does not
    read these states.
    """

    observed_at = available_at or datetime.now(timezone.utc)
    states, skipped = build_momentum_ticker_states(
        scored,
        tickers=tickers,
        decision_bar=decision_bar,
        available_at=observed_at,
        matrix_path=Path(matrix_path),
    )
    result: dict[str, object] = {
        "published": 0,
        "skipped": dict(skipped),
        "status": "PUBLISHED",
    }
    if not states:
        result["status"] = "NOTHING_TO_PUBLISH"
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
            uow.states.insert_states_idempotently(tuple(states))
            uow.commit()
    except Exception as exc:  # noqa: BLE001 - publication must never stop trading
        logger.warning(
            "momentum ticker-state publication failed (%s): %s",
            _STATE_STORE_UNAVAILABLE,
            type(exc).__name__,
        )
        result["status"] = _STATE_STORE_UNAVAILABLE
        return result

    result["published"] = len(states)
    return result


__all__ = ["build_momentum_ticker_states", "publish_momentum_ticker_states"]
