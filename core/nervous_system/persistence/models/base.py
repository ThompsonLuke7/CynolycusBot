"""Shared SQLAlchemy conventions for the nervous-system schema."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import BigInteger, Boolean, DateTime, Integer, MetaData, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


SCHEMA = "nervous_system"
DECIMAL_PRECISION = 20
DECIMAL_SCALE = 8


class Base(DeclarativeBase):
    """Declarative base whose metadata is explicitly scoped to our schema."""

    metadata = MetaData(schema=SCHEMA)


UUIDType = PostgreSQLUUID(as_uuid=True)
TimestampType = DateTime(timezone=True)
DecimalType = Numeric(DECIMAL_PRECISION, DECIMAL_SCALE)


def jsonb_default(value: str = "{}") -> sa.TextClause:
    """Return a PostgreSQL JSONB server default without a mutable Python default."""

    if value not in {"{}", "[]"}:
        raise ValueError("only empty JSON object and array defaults are supported")
    return sa.text(f"'{value}'::jsonb")


def uuid_primary_key() -> Mapped[UUID]:
    """Return the common native PostgreSQL UUID primary-key mapping."""

    return mapped_column(UUIDType, primary_key=True, nullable=False)


def utc_timestamp(*, nullable: bool = False) -> Mapped[datetime | None]:
    """Return a timezone-aware timestamp mapping."""

    return mapped_column(TimestampType, nullable=nullable)


def jsonb_column(
    *,
    nullable: bool = False,
    default: str = "{}",
) -> Mapped[dict[str, Any]]:
    """Return a JSONB payload column with a server-side empty value."""

    return mapped_column(
        JSONB,
        nullable=nullable,
        server_default=jsonb_default(default),
    )


__all__ = [
    "Base",
    "BigInteger",
    "Boolean",
    "Decimal",
    "DecimalType",
    "DateTime",
    "Integer",
    "JSONB",
    "Numeric",
    "PostgreSQLUUID",
    "SCHEMA",
    "String",
    "Text",
    "TimestampType",
    "UUID",
    "UUIDType",
    "jsonb_column",
    "jsonb_default",
    "mapped_column",
    "utc_timestamp",
    "uuid_primary_key",
]
