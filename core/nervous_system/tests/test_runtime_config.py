from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from core.nervous_system.config.runtime import NervousSystemSettings
from core.nervous_system.contracts.enums import PolicyMode, RuntimeEnvironment
from core.nervous_system.persistence import database


def _env(**overrides: str) -> dict[str, str]:
    values = {
        "CYNOLYCUS_ENVIRONMENT": "development",
        "CYNOLYCUS_NERVOUS_SYSTEM_MODE": "off",
        "CYNOLYCUS_DATABASE_URL": "postgresql+psycopg://u:p@db/cynolycus",
        "CYNOLYCUS_DB_POOL_SIZE": "5",
        "CYNOLYCUS_DB_MAX_OVERFLOW": "5",
        "CYNOLYCUS_OPERATIONAL_ROOT": "Data/operational/nervous_system",
        "CYNOLYCUS_EXECUTION_JOURNAL": "local",
        "CYNOLYCUS_EXECUTION_JOURNAL_BUCKET": "",
        "CYNOLYCUS_ACCOUNT_ALIAS": "paper",
    }
    values.update(overrides)
    return values


def test_from_env_normalizes_case_and_hyphenated_enum_values() -> None:
    settings = NervousSystemSettings.from_env(
        _env(
            CYNOLYCUS_ENVIRONMENT="Qa-PaPeR",
            CYNOLYCUS_NERVOUS_SYSTEM_MODE="EnFoRcE",
            CYNOLYCUS_EXECUTION_JOURNAL="GCS",
            CYNOLYCUS_EXECUTION_JOURNAL_BUCKET="qa-journal",
            CYNOLYCUS_ACCOUNT_ALIAS="PAPER",
        )
    )

    assert settings.environment is RuntimeEnvironment.QA_PAPER
    assert settings.policy_mode is PolicyMode.ENFORCE
    assert settings.journal_backend == "gcs"
    assert settings.gcs_bucket == "qa-journal"
    assert settings.account_alias == "paper"


def test_from_env_uses_supplied_mapping_without_process_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CYNOLYCUS_DATABASE_URL", "postgresql+psycopg://leak:me@live/db")

    supplied = _env()
    del supplied["CYNOLYCUS_DATABASE_URL"]

    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.from_env(supplied)

    assert "CYNOLYCUS_DATABASE_URL" in str(exc_info.value)


def test_from_env_aggregates_missing_required_names() -> None:
    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.from_env({})

    message = str(exc_info.value)
    for name in (
        "CYNOLYCUS_ENVIRONMENT",
        "CYNOLYCUS_NERVOUS_SYSTEM_MODE",
        "CYNOLYCUS_DATABASE_URL",
        "CYNOLYCUS_OPERATIONAL_ROOT",
        "CYNOLYCUS_EXECUTION_JOURNAL",
        "CYNOLYCUS_ACCOUNT_ALIAS",
    ):
        assert name in message


def test_from_env_defaults_pool_overrides_and_treats_blank_bucket_as_absent() -> None:
    supplied = _env()
    del supplied["CYNOLYCUS_DB_POOL_SIZE"]
    del supplied["CYNOLYCUS_DB_MAX_OVERFLOW"]

    settings = NervousSystemSettings.from_env(supplied)

    assert settings.db_pool_size == 5
    assert settings.db_max_overflow == 5
    assert settings.gcs_bucket is None


def test_qa_paper_requires_paper_alias_and_gcs_journal() -> None:
    with pytest.raises(ValidationError, match="paper account alias"):
        NervousSystemSettings.from_env(
            _env(
                CYNOLYCUS_ENVIRONMENT="qa-paper",
                CYNOLYCUS_ACCOUNT_ALIAS="live",
            )
        )

    with pytest.raises(ValidationError, match="durable GCS journal"):
        NervousSystemSettings.from_env(
            _env(
                CYNOLYCUS_ENVIRONMENT="qa-paper",
                CYNOLYCUS_ACCOUNT_ALIAS="paper",
            )
        )


def test_gcs_journal_requires_nonblank_bucket() -> None:
    with pytest.raises(ValidationError, match="gcs journal requires"):
        NervousSystemSettings.from_env(
            _env(
                CYNOLYCUS_EXECUTION_JOURNAL="gcs",
                CYNOLYCUS_EXECUTION_JOURNAL_BUCKET="  ",
            )
        )


def test_qa_paper_model_copy_revalidates_account_safety_boundary() -> None:
    settings = NervousSystemSettings.from_env(
        _env(
            CYNOLYCUS_ENVIRONMENT="qa-paper",
            CYNOLYCUS_EXECUTION_JOURNAL="gcs",
            CYNOLYCUS_EXECUTION_JOURNAL_BUCKET="qa-journal",
        )
    )

    with pytest.raises(ValidationError, match="paper account alias"):
        settings.model_copy(update={"account_alias": "live"})


@pytest.mark.parametrize(
    "key,value",
    [
        ("CYNOLYCUS_DB_POOL_SIZE", "0"),
        ("CYNOLYCUS_DB_MAX_OVERFLOW", "-1"),
        ("CYNOLYCUS_DB_POOL_SIZE", "not-an-int"),
    ],
)
def test_from_env_rejects_nonpositive_pool_values(key: str, value: str) -> None:
    with pytest.raises(ValidationError):
        NervousSystemSettings.from_env(_env(**{key: value}))


@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite:///not-postgres",
        "postgresql://u:p@db/cynolycus",
        "postgresql+psycopg://",
    ],
)
def test_from_env_requires_valid_postgresql_psycopg_url(database_url: str) -> None:
    with pytest.raises(ValidationError, match="postgresql\\+psycopg"):
        NervousSystemSettings.from_env(_env(CYNOLYCUS_DATABASE_URL=database_url))


def test_production_live_is_parseable_without_a_live_execution_path() -> None:
    settings = NervousSystemSettings.from_env(
        _env(
            CYNOLYCUS_ENVIRONMENT="production-live",
            CYNOLYCUS_ACCOUNT_ALIAS="live",
        )
    )

    assert settings.environment is RuntimeEnvironment.PRODUCTION_LIVE
    assert settings.account_alias == "live"


def test_create_database_engine_forwards_validated_pool_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = NervousSystemSettings.from_env(
        _env(CYNOLYCUS_DB_POOL_SIZE="7", CYNOLYCUS_DB_MAX_OVERFLOW="2")
    )
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_create_engine(url: str, **kwargs: Any) -> object:
        captured["url"] = url
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(database, "create_engine", fake_create_engine)

    assert database.create_database_engine(settings) is sentinel
    assert captured == {
        "url": settings.database_url,
        "pool_pre_ping": True,
        "pool_size": 7,
        "max_overflow": 2,
        "future": True,
    }


def test_create_session_factory_uses_nonexpiring_nonautoflush_sessions() -> None:
    engine = object()

    factory = database.create_session_factory(engine)

    assert factory.kw["bind"] is engine
    assert factory.kw["autoflush"] is False
    assert factory.kw["expire_on_commit"] is False


class _Connection(AbstractContextManager["_Connection"]):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.executed: list[Any] = []

    def __enter__(self) -> "_Connection":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, statement: Any) -> None:
        self.executed.append(statement)
        if self.error is not None:
            raise self.error


class _Engine:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def connect(self) -> _Connection:
        return self.connection


def test_database_healthcheck_returns_true_after_select_one() -> None:
    connection = _Connection()

    assert database.database_healthcheck(_Engine(connection)) is True
    assert [str(statement) for statement in connection.executed] == ["SELECT 1"]


def test_database_healthcheck_returns_false_for_sqlalchemy_error() -> None:
    assert database.database_healthcheck(
        _Engine(_Connection(SQLAlchemyError("database unavailable")))
    ) is False


def test_database_healthcheck_propagates_non_sqlalchemy_errors() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        database.database_healthcheck(_Engine(_Connection(RuntimeError("unexpected"))))
