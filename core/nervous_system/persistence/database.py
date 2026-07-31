"""SQLAlchemy connection and session boundaries."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from core.nervous_system.config.runtime import NervousSystemSettings


def create_database_engine(settings: NervousSystemSettings) -> Engine:
    """Create the synchronous SQLAlchemy engine for validated settings."""

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        future=True,
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
