"""Health must actually check the journal (Task 26 follow-up).

`AuditStore.health()` shipped with a probe that defaulted to true — a check
that always passes is worse than no check, because it occupies the slot where a
real one would go and reports green while the thing it names is broken.

The journal is the outage-time evidence. If it is unwritable, the system can
still trade and will simply have no record of having done so, which is the one
failure mode the whole design exists to prevent. So health has to know.

The probe is bounded and never raises: a health endpoint that throws turns a
degraded system into an unreachable one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.nervous_system.execution.journal_probe import (
    JournalProbeResult,
    probe_journal,
)


class _Bucket:
    def __init__(self, exists: bool = True, raises: BaseException | None = None):
        self._exists = exists
        self._raises = raises

    def exists(self) -> bool:
        if self._raises is not None:
            raise self._raises
        return self._exists


class _Client:
    def __init__(self, bucket: _Bucket):
        self._bucket = bucket

    def bucket(self, _name: str) -> _Bucket:
        return self._bucket


# ---------------------------------------------------------------------------
# Local journal
# ---------------------------------------------------------------------------


def test_a_writable_local_root_is_healthy(tmp_path: Path) -> None:
    result = probe_journal(backend="local", root=tmp_path)

    assert result.ok is True
    assert result.detail == "writable"


def test_a_missing_local_root_is_created_and_healthy(tmp_path: Path) -> None:
    """The journal directory not existing yet is normal on a fresh host, and
    is not a reason to report the system unhealthy.
    """

    target = tmp_path / "does" / "not" / "exist"

    assert probe_journal(backend="local", root=target).ok is True
    assert target.exists()


def test_an_unwritable_local_root_is_unhealthy(tmp_path: Path) -> None:
    """This is the case that matters. Without it the system trades and keeps
    no record of having done so.
    """

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        result = probe_journal(backend="local", root=blocked / "journal")

        assert result.ok is False
        assert "PermissionError" in result.detail
    finally:
        blocked.chmod(0o700)


def test_a_local_probe_leaves_nothing_behind(tmp_path: Path) -> None:
    """A probe that litters the journal with its own files corrupts the
    evidence it exists to protect.
    """

    probe_journal(backend="local", root=tmp_path)

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# GCS journal
# ---------------------------------------------------------------------------


def test_a_reachable_bucket_is_healthy() -> None:
    result = probe_journal(backend="gcs", client=_Client(_Bucket(exists=True)), bucket="b")

    assert result.ok is True


def test_a_missing_bucket_is_unhealthy() -> None:
    result = probe_journal(backend="gcs", client=_Client(_Bucket(exists=False)), bucket="b")

    assert result.ok is False
    assert result.detail == "bucket_not_found"


def test_a_gcs_error_is_unhealthy_without_leaking_its_text() -> None:
    """A cloud client error routinely carries the project, the bucket, and a
    signed URL. Health output is read in shared places.
    """

    boom = RuntimeError("403 on projects/_/buckets/b?key=AIzaSyRealLookingKey")
    result = probe_journal(backend="gcs", client=_Client(_Bucket(raises=boom)), bucket="b")

    assert result.ok is False
    assert "AIzaSyRealLookingKey" not in result.detail
    assert result.detail == "RuntimeError"


def test_a_gcs_probe_without_a_client_is_unhealthy_not_assumed_healthy() -> None:
    """Absent wiring is the exact condition that produced the always-true
    probe in the first place.
    """

    result = probe_journal(backend="gcs", client=None, bucket="b")

    assert result.ok is False
    assert result.detail == "no_client"


# ---------------------------------------------------------------------------
# The probe never becomes the outage
# ---------------------------------------------------------------------------


def test_an_unknown_backend_is_unhealthy_rather_than_assumed_fine() -> None:
    assert probe_journal(backend="carrier-pigeon").ok is False


def test_the_probe_never_raises() -> None:
    """A health endpoint that throws turns a degraded system into an
    unreachable one.
    """

    class _Exploding:
        def bucket(self, _name):
            raise KeyboardInterrupt("not even this")

    result = probe_journal(backend="gcs", client=_Exploding(), bucket="b")

    assert isinstance(result, JournalProbeResult)
    assert result.ok is False
