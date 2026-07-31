from __future__ import annotations

from contextlib import AbstractContextManager
import json
from typing import Any
from urllib.parse import quote

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


_SECRET_USERNAME = "UNMISTAKABLE_DB_USERNAME_9f3a"
_SECRET_PASSWORD = "UNMISTAKABLE_DB_PASSWORD_4c7b"
_SECRET_DATABASE_URL = (
    f"postgresql+psycopg://{_SECRET_USERNAME}:{_SECRET_PASSWORD}@db/cynolycus"
)
_STRUCTURED_USERNAME = "structured.user+qa@example.com"
_STRUCTURED_PASSWORD = "structured:password/with@symbols"
_STRUCTURED_USERNAME_ENCODED = quote(_STRUCTURED_USERNAME, safe="")
_STRUCTURED_PASSWORD_ENCODED = quote(_STRUCTURED_PASSWORD, safe="")
_STRUCTURED_DATABASE_URL = (
    "postgresql+psycopg://"
    f"{_STRUCTURED_USERNAME_ENCODED}:{_STRUCTURED_PASSWORD_ENCODED}"
    "@db/cynolycus"
)


def _structured_qa_payload() -> dict[str, str]:
    return {
        "environment": "QA_PAPER",
        "policy_mode": "SHADOW",
        "database_url": _STRUCTURED_DATABASE_URL,
        "operational_root": "Data/operational/nervous_system",
        "journal_backend": "gcs",
        "gcs_bucket": "qa-journal",
        "account_alias": "live",
    }


def _assert_validation_error_is_secret_free(error: ValidationError) -> None:
    secrets = (
        _STRUCTURED_USERNAME,
        _STRUCTURED_PASSWORD,
        _STRUCTURED_USERNAME_ENCODED,
        _STRUCTURED_PASSWORD_ENCODED,
    )

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(key)
                visit(nested)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for nested in value:
                visit(nested)
            return
        rendered = f"{value!r} {value}"
        assert all(secret not in rendered for secret in secrets)

    details = error.errors()
    assert details
    assert details[0]["loc"] == ()
    assert "QA_PAPER requires the paper account alias" in details[0]["msg"]
    visit(details)
    rendered_json = error.json()
    assert all(secret not in rendered_json for secret in secrets)


def _builtin_error_payload() -> dict[str, Any]:
    return {
        "environment": "DEVELOPMENT",
        "policy_mode": "OFF",
        "database_url": _STRUCTURED_DATABASE_URL,
        "operational_root": "Data/operational/nervous_system",
        "journal_backend": "local",
        "account_alias": "paper",
    }


def _assert_builtin_error_is_secret_free(
    error: ValidationError,
    *,
    expected_type: str,
    expected_loc: tuple[str, ...],
    expected_message: str,
) -> None:
    details = error.errors()
    assert len(details) == 1
    assert details[0]["type"] == expected_type
    assert details[0]["loc"] == expected_loc
    assert details[0]["msg"] == expected_message

    secrets = (
        _STRUCTURED_USERNAME,
        _STRUCTURED_PASSWORD,
        _STRUCTURED_USERNAME_ENCODED,
        _STRUCTURED_PASSWORD_ENCODED,
    )

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(key)
                visit(nested)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for nested in value:
                visit(nested)
            return
        rendered = f"{value!r} {value}"
        assert all(secret not in rendered for secret in secrets)

    visit(details)
    rendered_json = error.json()
    assert all(secret not in rendered_json for secret in secrets)


@pytest.mark.parametrize(
    ("field", "value", "remove", "expected_type", "expected_message"),
    [
        (
            "operational_root",
            object(),
            False,
            "path_type",
            "Input is not a valid path for <class 'pathlib.Path'>",
        ),
        (
            "db_pool_size",
            "not-an-int",
            False,
            "int_parsing",
            "Input should be a valid integer, unable to parse string as an integer",
        ),
        (
            "db_pool_size",
            0,
            False,
            "greater_than",
            "Input should be greater than 0",
        ),
        (
            "journal_backend",
            "not-a-backend",
            False,
            "literal_error",
            "Input should be 'local' or 'gcs'",
        ),
        (
            "environment",
            "NOT_AN_ENVIRONMENT",
            False,
            "enum",
            "Input should be 'DEVELOPMENT', 'QA_PAPER' or 'PRODUCTION_LIVE'",
        ),
        (
            "surprise",
            True,
            False,
            "extra_forbidden",
            "Extra inputs are not permitted",
        ),
        (
            "account_alias",
            None,
            True,
            "missing",
            "Field required",
        ),
    ],
)
def test_builtin_validation_errors_preserve_details_without_secret_leaks(
    field: str,
    value: Any,
    remove: bool,
    expected_type: str,
    expected_message: str,
) -> None:
    payload = _builtin_error_payload()
    if remove:
        payload.pop(field)
    else:
        payload[field] = value

    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.model_validate(payload)

    _assert_builtin_error_is_secret_free(
        exc_info.value,
        expected_type=expected_type,
        expected_loc=(field,),
        expected_message=expected_message,
    )


def test_mixed_validation_errors_redact_database_input_without_secondary_errors() -> None:
    payload = _builtin_error_payload()
    payload["operational_root"] = object()
    payload.pop("account_alias")

    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.model_validate(payload)

    details = exc_info.value.errors()
    assert [(detail["type"], detail["loc"], detail["msg"]) for detail in details] == [
        (
            "path_type",
            ("operational_root",),
            "Input is not a valid path for <class 'pathlib.Path'>",
        ),
        ("missing", ("account_alias",), "Field required"),
    ]
    rendered = f"{str(exc_info.value)}\n{details!r}\n{exc_info.value.json()}"
    for secret in (
        _STRUCTURED_USERNAME,
        _STRUCTURED_PASSWORD,
        _STRUCTURED_USERNAME_ENCODED,
        _STRUCTURED_PASSWORD_ENCODED,
    ):
        assert secret not in rendered


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


def test_database_credentials_are_redacted_from_public_representations() -> None:
    settings = NervousSystemSettings.from_env(
        _env(CYNOLYCUS_DATABASE_URL=_SECRET_DATABASE_URL)
    )

    rendered = "\n".join(
        (
            repr(settings),
            str(settings),
            json.dumps(settings.model_dump(mode="json")),
            settings.model_dump_json(),
        )
    )

    assert _SECRET_USERNAME not in rendered
    assert _SECRET_PASSWORD not in rendered
    assert isinstance(settings.database_url, str)
    assert settings.database_url == _SECRET_DATABASE_URL
    assert settings.model_dump()["database_url"] != _SECRET_DATABASE_URL


def test_database_credentials_are_hidden_from_url_validation_errors() -> None:
    invalid_url = (
        f"postgresql+psycopg://{_SECRET_USERNAME}:{_SECRET_PASSWORD}@db"
    )

    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.from_env(
            _env(CYNOLYCUS_DATABASE_URL=invalid_url)
        )

    message = str(exc_info.value)
    assert _SECRET_USERNAME not in message
    assert _SECRET_PASSWORD not in message


def test_from_env_structured_errors_redact_percent_encoded_credentials() -> None:
    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.from_env(
            _env(
                CYNOLYCUS_ENVIRONMENT="qa-paper",
                CYNOLYCUS_DATABASE_URL=_STRUCTURED_DATABASE_URL,
                CYNOLYCUS_EXECUTION_JOURNAL="gcs",
                CYNOLYCUS_EXECUTION_JOURNAL_BUCKET="qa-journal",
                CYNOLYCUS_ACCOUNT_ALIAS="live",
            )
        )

    assert "QA_PAPER requires the paper account alias" in str(exc_info.value)
    _assert_validation_error_is_secret_free(exc_info.value)


def test_direct_model_validate_structured_errors_are_secret_free() -> None:
    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.model_validate(_structured_qa_payload())

    assert "QA_PAPER requires the paper account alias" in str(exc_info.value)
    _assert_validation_error_is_secret_free(exc_info.value)


def test_model_validate_json_structured_errors_are_secret_free() -> None:
    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.model_validate_json(
            json.dumps(_structured_qa_payload())
        )

    assert "QA_PAPER requires the paper account alias" in str(exc_info.value)
    _assert_validation_error_is_secret_free(exc_info.value)


def test_model_validate_strings_structured_errors_are_secret_free() -> None:
    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings.model_validate_strings(_structured_qa_payload())

    assert "QA_PAPER requires the paper account alias" in str(exc_info.value)
    _assert_validation_error_is_secret_free(exc_info.value)


def test_direct_construction_structured_errors_are_secret_free() -> None:
    with pytest.raises(ValidationError) as exc_info:
        NervousSystemSettings(**_structured_qa_payload())

    assert "QA_PAPER requires the paper account alias" in str(exc_info.value)
    _assert_validation_error_is_secret_free(exc_info.value)


def test_model_copy_structured_errors_are_secret_free() -> None:
    settings = NervousSystemSettings.from_env(
        _env(
            CYNOLYCUS_ENVIRONMENT="qa-paper",
            CYNOLYCUS_DATABASE_URL=_STRUCTURED_DATABASE_URL,
            CYNOLYCUS_EXECUTION_JOURNAL="gcs",
            CYNOLYCUS_EXECUTION_JOURNAL_BUCKET="qa-journal",
        )
    )

    with pytest.raises(ValidationError) as exc_info:
        settings.model_copy(update={"account_alias": "live"})

    assert "QA_PAPER requires the paper account alias" in str(exc_info.value)
    _assert_validation_error_is_secret_free(exc_info.value)


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
