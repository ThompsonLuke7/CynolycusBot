"""The operations CLI is non-destructive by construction (Task 26).

This tool is pointed at the database holding the audit ledger. A destructive
command that exists is one that eventually gets run against the wrong target at
the wrong hour, so the safety here is structural: the commands simply are not
there, and the parser rejects them before any connection is opened.
"""

from __future__ import annotations

import json

import pytest

from scripts.cloud.nervous_system_db import (
    COMMANDS,
    FORBIDDEN_COMMANDS,
    _redacted,
    build_parser,
    main,
    verify_backup,
)


# ---------------------------------------------------------------------------
# What the tool cannot do
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", sorted(FORBIDDEN_COMMANDS))
def test_a_destructive_command_does_not_exist(command: str) -> None:
    """Not validated and rejected — absent. There is nothing to bypass."""

    assert command not in COMMANDS


@pytest.mark.parametrize("command", ["downgrade", "drop", "reset", "truncate"])
def test_the_parser_refuses_a_destructive_command_before_connecting(command: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_the_command_set_is_exactly_the_approved_one() -> None:
    """A new command should be a deliberate decision, not an accident."""

    assert set(COMMANDS) == {
        "schema-status", "create-database", "upgrade-schema",
        "import-history", "verify-counts", "verify-backup",
    }


def test_no_command_accepts_a_password_argument() -> None:
    """A DSN on a command line lands in shell history, ps output, and CI logs."""

    actions = {action.dest for action in build_parser()._actions}

    assert not {name for name in actions if "password" in name or "dsn" in name}
    assert not {name for name in actions if "url" in name or "secret" in name}


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_a_printed_dsn_keeps_the_host_and_drops_the_credentials() -> None:
    """Useful enough to confirm the target, useless to an onlooker."""

    rendered = _redacted("postgresql+psycopg://user:hunter2@10.0.0.1:5432/cynolycus")

    assert "hunter2" not in rendered
    assert "user" not in rendered
    assert "10.0.0.1" in rendered
    assert "cynolycus" in rendered


def test_a_socket_dsn_renders_without_credentials() -> None:
    rendered = _redacted(
        "postgresql+psycopg://cynolycus:s3cret@/cynolycus"
        "?host=/cloudsql/p:us-east5:i"
    )

    assert "s3cret" not in rendered


def test_an_unparseable_dsn_does_not_leak_its_contents() -> None:
    assert "hunter2" not in _redacted("not a dsn hunter2")


# ---------------------------------------------------------------------------
# Defaults that fail safe
# ---------------------------------------------------------------------------


def test_import_history_is_a_dry_run_unless_told_otherwise() -> None:
    """An import that writes by default is one keystroke from being run against
    a populated database by somebody who wanted to see what it would do.
    """

    from scripts.cloud.nervous_system_db import import_history

    class _Settings:
        database_url = "postgresql+psycopg://u:p@h/cynolycus"

    args = build_parser().parse_args(["import-history"])
    result = import_history(_Settings(), args)

    assert result["wrote"] is False
    assert result["reason"] == "dry_run_is_the_default"


def test_verify_backup_reports_unknown_rather_than_inferring() -> None:
    """An unverified "probably backed up" is worse than a clear "unknown":
    only one of them makes somebody go and look.
    """

    class _Settings:
        cloud_sql_instance = "p:us-east5:i"

    result = verify_backup(_Settings(), None)

    assert result["status"] == "UNVERIFIED"
    assert result["authority"] == "cloud_sql_api"


# ---------------------------------------------------------------------------
# Failures stay quiet about credentials
# ---------------------------------------------------------------------------


def test_a_failure_prints_the_exception_type_not_its_text(capsys, monkeypatch) -> None:
    """A driver error routinely carries the whole DSN in its message."""

    import scripts.cloud.nervous_system_db as cli

    def _explode() -> None:
        raise ConnectionError("could not connect to postgresql://u:hunter2@h/db")

    monkeypatch.setattr(cli, "_settings", _explode)

    code = main(["schema-status"])
    captured = capsys.readouterr()

    assert code == 1
    assert "hunter2" not in captured.err
    assert json.loads(captured.err)["error"] == "ConnectionError"
