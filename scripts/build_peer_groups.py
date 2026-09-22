#!/usr/bin/env python
"""Nightly: build trailing-correlation peer groups and publish them as state.

The lateral axis of the state registry.  Run after the 1d bar catch-up so the
as-of session's bars are on disk.

    .venv/bin/python scripts/build_peer_groups.py --as-of 2026-09-11

Publication failure never fails the build: the JSON artifact on disk is the
durable output, matching how market-regime and dealer publication behave.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.shared_universe.universe import load_universe_as_of  # noqa: E402
from signals.peer_structure.correlation_groups import (  # noqa: E402
    PeerGroupConfig,
    build_peer_groups,
    load_screened_returns,
)


logger = logging.getLogger("peer_groups")

OUTPUT_DIR = REPO_ROOT / "Data/shared/peer_groups"
_STATE_STORE_UNAVAILABLE = "state store unavailable"


def publish(states) -> str:
    """Publish PEER_GROUP states, owning the unit of work here."""

    if not states:
        return "NOTHING_TO_PUBLISH"
    try:
        from core.nervous_system.config.runtime import NervousSystemSettings
        from core.nervous_system.persistence.database import (
            create_database_engine,
            create_session_factory,
        )
        from core.nervous_system.persistence.uow import UnitOfWork

        # No argument, so from_env() falls back to .env for names the process
        # environment does not define. The server does not export CYNOLYCUS_*.
        settings = NervousSystemSettings.from_env()
        session_factory = create_session_factory(create_database_engine(settings))
    except Exception as exc:  # noqa: BLE001 - the JSON artifact must survive
        logger.warning(
            "peer-group publication skipped (%s): %s",
            _STATE_STORE_UNAVAILABLE,
            type(exc).__name__,
        )
        return "SKIPPED"

    try:
        with UnitOfWork(session_factory) as uow:
            uow.states.insert_states_idempotently(tuple(states))
            uow.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "peer-group publication failed (%s): %s",
            _STATE_STORE_UNAVAILABLE,
            type(exc).__name__,
        )
        return "FAILED"
    return "PUBLISHED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        default=None,
        help="Session to build from (YYYY-MM-DD). Defaults to the latest weekday.",
    )
    parser.add_argument("--lookback", type=int, default=60, help="Trailing sessions.")
    parser.add_argument("--groups", type=int, default=60, help="Target group count.")
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=10_000_000.0,
        help="Liquidity floor; 87%% of the 2026-09 lead/lag events failed this.",
    )
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()
    config = PeerGroupConfig(
        lookback_sessions=args.lookback,
        n_groups=args.groups,
        min_dollar_volume=args.min_dollar_volume,
    )

    # Point-in-time, and it raises rather than falling back to today's list:
    # that fallback is the survivorship bias the snapshot exists to remove.
    universe = load_universe_as_of(
        as_of.tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59)
    )
    tickers = sorted(universe["ticker"].astype(str).unique())
    logger.info("universe as of %s: %d tickers", as_of.date(), len(tickers))

    returns = load_screened_returns(tickers, as_of_session=as_of, config=config)
    if returns.empty:
        logger.error("no screened returns for %s; nothing built", as_of.date())
        return 1
    logger.info(
        "screened: %d names survive the $%.0fM/day and $%.0f floors",
        returns.shape[1],
        config.min_dollar_volume / 1e6,
        config.min_price,
    )

    states = build_peer_groups(
        returns,
        as_of_session=as_of,
        config=config,
        universe_version=str(universe.attrs.get("snapshot_utc", "unknown")),
    )
    cohesion = [s.trailing_cohesion for s in states if s.trailing_cohesion is not None]
    logger.info(
        "built %d groups covering %d names (mean trailing cohesion %.3f)",
        len(states),
        sum(len(s.members) for s in states),
        sum(cohesion) / len(cohesion) if cohesion else float("nan"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT_DIR / f"peer_groups_{as_of.strftime('%Y%m%d')}.json"
    artifact.write_text(
        json.dumps(
            {
                "as_of_session": as_of.date().isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "config_version": config.config_version,
                "groups": [
                    {
                        "group_id": s.group_id,
                        "method": s.method,
                        "members": list(s.members),
                        "trailing_cohesion": s.trailing_cohesion,
                    }
                    for s in states
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", artifact)

    status = "SKIPPED" if args.no_publish else publish(states)
    logger.info("state publication: %s", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
