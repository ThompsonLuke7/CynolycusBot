"""Non-destructive database operations for the nervous system.

Every command here is additive or read-only. There is deliberately no
downgrade, drop, truncate, or reset: this tool is pointed at the database that
holds the audit ledger, and a destructive command that exists is a destructive
command that eventually gets run against the wrong target at the wrong hour.
Rolling back a migration is a deliberate, manual act performed with the
knowledge of what is in the table — not a flag on an operations script.

Passwords are never accepted as arguments. A DSN on a command line lands in
shell history, in `ps` output, and in any CI log that echoes its commands, so
the connection always comes from the environment.

    python -m scripts.cloud.nervous_system_db schema-status
    python -m scripts.cloud.nervous_system_db upgrade-schema
    python -m scripts.cloud.nervous_system_db import-history --dry-run
    python -m scripts.cloud.nervous_system_db verify-counts
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from core.nervous_system.config.runtime import NervousSystemSettings


SCHEMA = "nervous_system"

# Refused outright rather than validated: a tool that can express a destructive
# intent will eventually be handed one.
FORBIDDEN_COMMANDS = ("downgrade", "drop", "reset", "truncate", "delete")


def _redacted(dsn: str) -> str:
    """A DSN safe to print: host and database, never credentials."""

    try:
        url = make_url(dsn)
    except Exception:  # noqa: BLE001
        return "<unparseable DSN>"
    return f"{url.drivername}://{url.host or 'socket'}/{url.database or '?'}"


def _settings() -> NervousSystemSettings:
    return NervousSystemSettings.from_env()


def _engine(settings: NervousSystemSettings):
    return create_engine(settings.database_url)


def schema_status(settings: NervousSystemSettings, _args: Any) -> dict[str, Any]:
    """Report the revision and table count without changing anything."""

    with _engine(settings).connect() as connection:
        revision = connection.execute(
            text("select version_num from public.alembic_version")
        ).scalar()
        tables = connection.execute(
            text(
                "select count(*) from information_schema.tables "
                "where table_schema = :schema"
            ),
            {"schema": SCHEMA},
        ).scalar()
    return {
        "target": _redacted(settings.database_url),
        "environment": settings.environment.value,
        "revision": revision,
        "tables": tables,
    }


def create_database(settings: NervousSystemSettings, args: Any) -> dict[str, Any]:
    """Create the database if it is absent. Never touches an existing one."""

    url = make_url(settings.database_url)
    target = url.database
    admin = url.set(database="postgres")
    engine = create_engine(admin, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        exists = connection.execute(
            text("select 1 from pg_database where datname = :name"), {"name": target}
        ).scalar()
        if exists:
            return {"created": False, "reason": "already_exists", "database": target}
        if args.dry_run:
            return {"created": False, "reason": "dry_run", "database": target}
        connection.execute(text(f'create database "{target}"'))
    return {"created": True, "database": target}


def upgrade_schema(settings: NervousSystemSettings, args: Any) -> dict[str, Any]:
    """Upgrade to head. Upgrade only — there is no downgrade path here."""

    from alembic import command
    from alembic.config import Config

    before = schema_status(settings, args)
    if args.dry_run:
        return {"upgraded": False, "reason": "dry_run", "revision": before["revision"]}
    config = Config("core/nervous_system/persistence/alembic.ini")
    config.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(config, "head")
    after = schema_status(settings, args)
    return {
        "upgraded": before["revision"] != after["revision"],
        "from": before["revision"],
        "to": after["revision"],
    }


def verify_counts(settings: NervousSystemSettings, _args: Any) -> dict[str, Any]:
    """Row counts per table, for comparing two databases after a move."""

    with _engine(settings).connect() as connection:
        names = [
            row[0]
            for row in connection.execute(
                text(
                    "select table_name from information_schema.tables "
                    "where table_schema = :schema order by table_name"
                ),
                {"schema": SCHEMA},
            )
        ]
        counts = {
            name: connection.execute(
                text(f'select count(*) from {SCHEMA}."{name}"')
            ).scalar()
            for name in names
        }
    return {"target": _redacted(settings.database_url), "counts": counts}


def import_history(settings: NervousSystemSettings, args: Any) -> dict[str, Any]:
    """Import historical artifacts. Dry-run is the default, not a flag.

    An import that writes by default is one keystroke from being run against a
    populated database by someone who wanted to see what it would do.
    """

    if not args.write:
        return {
            "wrote": False,
            "reason": "dry_run_is_the_default",
            "hint": "pass --write to import",
            "target": _redacted(settings.database_url),
        }
    from core.nervous_system.data_registry.import_legacy import (  # noqa: PLC0415
        run_import,
    )

    return {"wrote": True, "result": run_import(settings)}


def verify_backup(settings: NervousSystemSettings, _args: Any) -> dict[str, Any]:
    """Report backup evidence, or say plainly that there is none.

    Backup authority is the Cloud SQL API. This command deliberately does not
    infer a backup from anything else — an unverified "probably backed up" is
    worse than a clear "unknown", because only one of them makes somebody go
    and look.
    """

    return {
        "status": "UNVERIFIED",
        "authority": "cloud_sql_api",
        "detail": (
            "backup verification requires the Cloud SQL Admin API; this command "
            "reports rather than infers"
        ),
        "instance": settings.cloud_sql_instance,
    }


COMMANDS: dict[str, Callable[[NervousSystemSettings, Any], dict[str, Any]]] = {
    "schema-status": schema_status,
    "create-database": create_database,
    "upgrade-schema": upgrade_schema,
    "import-history": import_history,
    "verify-counts": verify_counts,
    "verify-backup": verify_backup,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-destructive nervous-system database operations.",
        epilog="There is no downgrade, drop, or reset command, by design.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change"
    )
    parser.add_argument(
        "--write", action="store_true", help="import-history only: actually write"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in FORBIDDEN_COMMANDS:  # pragma: no cover - unreachable by choices
        print(f"ERROR: {args.command} is not available by design", file=sys.stderr)
        return 2
    try:
        result = COMMANDS[args.command](_settings(), args)
    except Exception as exc:  # noqa: BLE001
        # Never the raw exception: a driver error carries the DSN.
        print(
            json.dumps({"ok": False, "command": args.command,
                        "error": type(exc).__name__}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"ok": True, "command": args.command, **result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
