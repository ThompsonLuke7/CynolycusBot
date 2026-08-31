"""SQLAlchemy connection and session boundaries."""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from core.nervous_system.config.runtime import NervousSystemSettings


# Nothing in a trading decision path may wait on a database lock indefinitely.
#
# On 2026-08-26 a meta_ranker pre-open flush blocked on `Lock/transactionid`
# inside `save_context_snapshot_idempotently` and was still waiting 13 hours
# later, having placed no order and written no state. The self-deadlock that
# caused it is fixed in SnapshotBuilder, but the reason it cost a whole session
# rather than one logged failure is that the wait had no ceiling.
#
# `lock_timeout` bounds waiting for a lock; `statement_timeout` bounds the whole
# statement and catches a slow query that is not lock-related. Both raise a
# normal SQLAlchemy error, which the callers already treat as an infrastructure
# refusal (GovernedPathUnavailable / a recorded submit failure) rather than a
# crash — so the failure mode becomes "could not submit now, queued, logged"
# instead of a silent hang.
#
# Override with NERVOUS_SYSTEM_DB_LOCK_TIMEOUT_MS / _STATEMENT_TIMEOUT_MS for a
# backfill or migration that legitimately needs longer.
DEFAULT_LOCK_TIMEOUT_MS = 10_000
DEFAULT_STATEMENT_TIMEOUT_MS = 60_000


def _timeout_options() -> str:
    """libpq `options` string carrying the per-connection timeouts."""

    lock_ms = int(os.environ.get("NERVOUS_SYSTEM_DB_LOCK_TIMEOUT_MS",
                                DEFAULT_LOCK_TIMEOUT_MS))
    stmt_ms = int(os.environ.get("NERVOUS_SYSTEM_DB_STATEMENT_TIMEOUT_MS",
                                 DEFAULT_STATEMENT_TIMEOUT_MS))
    return f"-c lock_timeout={lock_ms} -c statement_timeout={stmt_ms}"


def create_database_engine(settings: NervousSystemSettings) -> Engine:
    """Create the synchronous SQLAlchemy engine for validated settings."""

    # The timeouts are server-side GUCs. NervousSystemSettings already refuses
    # any url that is not `postgresql+psycopg`, so there is no other backend to
    # guard against here — the offline SQLite tests build their engines
    # directly rather than through this function.
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        future=True,
        connect_args={"options": _timeout_options()},
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create non-expiring, non-autoflush synchronous sessions."""

    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def database_healthcheck(engine: Engine) -> bool:
    """Return false for database errors while surfacing unexpected failures."""

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False
    return True
