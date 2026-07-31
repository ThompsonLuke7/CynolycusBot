"""Validated runtime boundaries for the nervous-system process."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    model_validator,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from core.nervous_system.contracts.base import ContractModel
from core.nervous_system.contracts.enums import PolicyMode, RuntimeEnvironment


_ENVIRONMENT_NAME = "CYNOLYCUS_ENVIRONMENT"
_MODE_NAME = "CYNOLYCUS_NERVOUS_SYSTEM_MODE"
_DATABASE_URL_NAME = "CYNOLYCUS_DATABASE_URL"
_POOL_SIZE_NAME = "CYNOLYCUS_DB_POOL_SIZE"
_MAX_OVERFLOW_NAME = "CYNOLYCUS_DB_MAX_OVERFLOW"
_OPERATIONAL_ROOT_NAME = "CYNOLYCUS_OPERATIONAL_ROOT"
_JOURNAL_NAME = "CYNOLYCUS_EXECUTION_JOURNAL"
_JOURNAL_BUCKET_NAME = "CYNOLYCUS_EXECUTION_JOURNAL_BUCKET"
_ACCOUNT_ALIAS_NAME = "CYNOLYCUS_ACCOUNT_ALIAS"

_REQUIRED_ENV_NAMES = (
    _ENVIRONMENT_NAME,
    _MODE_NAME,
    _DATABASE_URL_NAME,
    _OPERATIONAL_ROOT_NAME,
    _JOURNAL_NAME,
    _ACCOUNT_ALIAS_NAME,
)


def _missing_environment_error(names: list[str]) -> ValidationError:
    return ValidationError.from_exception_data(
        "NervousSystemSettings",
        [
            {"type": "missing", "loc": (name,), "input": None}
            for name in names
        ],
    )


def _normalise_enum(value: str) -> str:
    return value.strip().replace("-", "_").upper()


def _redact_database_url(value: str) -> str:
    """Render a valid SQLAlchemy URL without its username or password."""

    try:
        parsed_url = make_url(value)
    except (ArgumentError, ValueError):
        return "<redacted database URL>"

    if parsed_url.username is not None:
        parsed_url = parsed_url.set(username="redacted")
    if parsed_url.password is not None:
        parsed_url = parsed_url.set(password="redacted")
    return parsed_url.render_as_string(hide_password=True)


class NervousSystemSettings(ContractModel):
    """Immutable, validated configuration for one nervous-system process."""

    model_config = ConfigDict(hide_input_in_errors=True)

    environment: RuntimeEnvironment
    policy_mode: PolicyMode
    database_url: str = Field(repr=False)
    db_pool_size: int = Field(default=5, gt=0)
    db_max_overflow: int = Field(default=5, gt=0)
    operational_root: Path
    journal_backend: Literal["local", "gcs"]
    gcs_bucket: str | None = None
    account_alias: str

    @field_serializer("database_url")
    def serialize_database_url(self, value: str) -> str:
        """Keep public model dumps safe while retaining the raw field internally."""

        return _redact_database_url(value)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Revalidate updates without round-tripping the redacted URL."""

        if update is None:
            return super().model_copy(deep=deep)

        payload = {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
        }
        payload.update(update)
        validated = type(self).model_validate(payload, by_alias=False, by_name=True)
        normalized_update = {
            field_name: getattr(validated, field_name) for field_name in update
        }
        return BaseModel.model_copy(self, update=normalized_update, deep=deep)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> Self:
        """Build settings from exactly the documented CYNOLYCUS variables."""

        source: Mapping[str, str] = os.environ if environ is None else environ
        missing = [
            name
            for name in _REQUIRED_ENV_NAMES
            if name not in source or not str(source[name]).strip()
        ]
        if missing:
            raise _missing_environment_error(missing)

        data: dict[str, object] = {
            "environment": _normalise_enum(
                str(source[_ENVIRONMENT_NAME])
            ),
            "policy_mode": _normalise_enum(
                str(source[_MODE_NAME])
            ),
            "database_url": str(source[_DATABASE_URL_NAME]).strip(),
            "operational_root": str(source[_OPERATIONAL_ROOT_NAME]).strip(),
            "journal_backend": str(source[_JOURNAL_NAME]).strip().lower(),
            "account_alias": str(source[_ACCOUNT_ALIAS_NAME]).strip().lower(),
        }

        if _POOL_SIZE_NAME in source:
            data["db_pool_size"] = source[_POOL_SIZE_NAME]
        if _MAX_OVERFLOW_NAME in source:
            data["db_max_overflow"] = source[_MAX_OVERFLOW_NAME]

        if _JOURNAL_BUCKET_NAME in source:
            bucket = str(source[_JOURNAL_BUCKET_NAME]).strip()
            data["gcs_bucket"] = bucket or None

        return cls.model_validate(data)

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        try:
            parsed_url = make_url(self.database_url)
        except (ArgumentError, ValueError) as exc:
            raise ValueError(
                "database_url must use postgresql+psycopg and include a database name"
            ) from exc

        if parsed_url.drivername != "postgresql+psycopg" or not parsed_url.database:
            raise ValueError(
                "database_url must use postgresql+psycopg and include a database name"
            )
        if not self.account_alias.strip():
            raise ValueError("account_alias must not be blank")
        if (
            self.environment is RuntimeEnvironment.QA_PAPER
            and self.account_alias.lower() != "paper"
        ):
            raise ValueError("QA_PAPER requires the paper account alias")
        if (
            self.environment is RuntimeEnvironment.QA_PAPER
            and self.journal_backend != "gcs"
        ):
            raise ValueError("QA_PAPER requires the durable GCS journal")
        if self.journal_backend == "gcs" and not self.gcs_bucket:
            raise ValueError(
                "gcs journal requires CYNOLYCUS_EXECUTION_JOURNAL_BUCKET"
            )
        return self
