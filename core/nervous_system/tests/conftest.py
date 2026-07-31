"""Shared fixtures for nervous-system tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """Return the disposable migration-test database URL or skip.

    The migration integration test upgrades, downgrades, and recreates the
    schema.  ``NERVOUS_SYSTEM_TEST_DATABASE_URL`` must therefore point to a
    dedicated disposable test database, never a development, QA, or
    production database.
    """

    value = os.environ.get("NERVOUS_SYSTEM_TEST_DATABASE_URL")
    if not value:
        pytest.skip("set NERVOUS_SYSTEM_TEST_DATABASE_URL for postgres tests")
    return value
