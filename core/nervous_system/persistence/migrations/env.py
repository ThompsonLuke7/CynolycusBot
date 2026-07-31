"""Alembic environment with repository-root and URL-override support."""

from __future__ import annotations

from pathlib import Path
import sys

from alembic import context
from sqlalchemy import engine_from_config, pool


config = context.config


def _repository_root() -> Path:
    """Find the repository root from the absolute Alembic config path."""

    config_name = config.config_file_name
    if config_name:
        config_path = Path(config_name).resolve()
        for candidate in (config_path.parent, *config_path.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    return Path(__file__).resolve().parents[5]


REPOSITORY_ROOT = _repository_root()
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.nervous_system.persistence.models import Base  # noqa: E402


target_metadata = Base.metadata


def _database_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("Alembic requires sqlalchemy.url or a test override")
    return url


def run_migrations_offline() -> None:
    """Render PostgreSQL DDL without opening a database connection."""

    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema=None,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations using the configured synchronous SQLAlchemy engine."""

    configuration = dict(config.get_section(config.config_ini_section) or {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema=None,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
