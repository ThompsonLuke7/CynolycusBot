"""Tests for the Schwab refresh-token expiry check.

The 2026-07-30 failure this guards against: the refresh token was minted
2026-07-22 23:30 ET, silently died 7 days later, and the first anyone knew of it
was 535 ``invalid_grant`` failures during the session.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core.schwab_token_status import (
    REFRESH_TOKEN_LIFETIME,
    banner_line,
    schwab_token_status,
)


def _write_token(tmp_path, issued: dt.datetime, **extra):
    path = tmp_path / "schwab_token.json"
    payload = {"creation_timestamp": issued.timestamp(), "token": {"refresh_token": "x"}}
    payload.update(extra)
    path.write_text(json.dumps(payload))
    return path


def test_fresh_token_is_ok(tmp_path):
    now = dt.datetime(2026, 7, 30, 12, 0, tzinfo=dt.timezone.utc)
    path = _write_token(tmp_path, now - dt.timedelta(days=1))

    status = schwab_token_status(path, now=now)

    assert status.found and not status.expired and not status.expiring_soon
    assert status.severity == "info"
    assert status.remaining == REFRESH_TOKEN_LIFETIME - dt.timedelta(days=1)


def test_token_expiring_inside_warning_window(tmp_path):
    now = dt.datetime(2026, 7, 30, 12, 0, tzinfo=dt.timezone.utc)
    # 5.5 days old -> 1.5 days left, inside the 2-day warning window.
    path = _write_token(tmp_path, now - dt.timedelta(days=5, hours=12))

    status = schwab_token_status(path, now=now)

    assert not status.expired
    assert status.expiring_soon
    assert status.severity == "warning"
    assert "RENEW SOON" in banner_line(status)


def test_expired_token_is_reported_as_error(tmp_path):
    """The real 2026-07-30 case: minted 07-22 23:30 ET, dead by the 07-30 open."""
    issued = dt.datetime(2026, 7, 23, 3, 30, 3, tzinfo=dt.timezone.utc)  # 07-22 23:30 ET
    now = dt.datetime(2026, 7, 30, 13, 31, 50, tzinfo=dt.timezone.utc)  # first invalid_grant
    path = _write_token(tmp_path, issued)

    status = schwab_token_status(path, now=now)

    assert status.expired
    assert status.severity == "error"
    assert status.expires_at == issued + REFRESH_TOKEN_LIFETIME
    assert "EXPIRED" in status.message
    assert "--reauth" in status.message


def test_boundary_exactly_at_expiry_counts_as_expired(tmp_path):
    now = dt.datetime(2026, 7, 30, 12, 0, tzinfo=dt.timezone.utc)
    path = _write_token(tmp_path, now - REFRESH_TOKEN_LIFETIME)

    assert schwab_token_status(path, now=now).expired


def test_missing_file_is_fatal_not_silent(tmp_path):
    status = schwab_token_status(tmp_path / "nope.json")

    assert not status.found
    assert status.expired
    assert status.severity == "error"


def test_malformed_token_is_treated_as_expired(tmp_path):
    path = tmp_path / "schwab_token.json"
    path.write_text("{not json")

    status = schwab_token_status(path)

    assert status.found and status.expired
    assert status.severity == "error"


def test_mtime_is_not_used_as_the_deadline(tmp_path):
    """An access-token refresh rewrites the file without extending the refresh token."""
    now = dt.datetime(2026, 7, 30, 12, 0, tzinfo=dt.timezone.utc)
    path = _write_token(tmp_path, now - dt.timedelta(days=8))
    path.touch()  # fresh mtime, stale refresh token

    status = schwab_token_status(path, now=now)

    assert status.expired, "expiry must come from creation_timestamp, not mtime"


@pytest.mark.parametrize("missing", ["creation_timestamp"])
def test_token_without_creation_timestamp_is_expired(tmp_path, missing):
    path = tmp_path / "schwab_token.json"
    path.write_text(json.dumps({"token": {"refresh_token": "x"}}))

    assert schwab_token_status(path).expired
