"""Bounded health probe for the execution journal.

The journal is the outage-time evidence. If it is unwritable the system can
still trade and will simply have no record of having done so — the one failure
mode the whole design exists to prevent — so health has to know rather than
assume.

Two rules shape this. The probe never raises: a health endpoint that throws
turns a degraded system into an unreachable one. And it never reports an
exception's text: a cloud client error routinely carries the project, bucket,
and a signed URL, and health output is read in shared places.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any


@dataclass(frozen=True)
class JournalProbeResult:
    ok: bool
    detail: str


def _probe_local(root: Path | str | None) -> JournalProbeResult:
    if root is None:
        return JournalProbeResult(False, "no_root")
    target = Path(root)
    try:
        # Creating the directory is part of the check: not existing yet is
        # normal on a fresh host and is not a reason to report unhealthy.
        target.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=str(target), prefix=".probe-")
        try:
            import os

            os.close(handle)
        finally:
            Path(name).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        return JournalProbeResult(False, type(exc).__name__)
    return JournalProbeResult(True, "writable")


def _probe_gcs(client: Any, bucket: str | None) -> JournalProbeResult:
    if client is None:
        # Absent wiring is exactly the condition that produced an always-true
        # probe in the first place, so it reads as unhealthy.
        return JournalProbeResult(False, "no_client")
    if not bucket:
        return JournalProbeResult(False, "no_bucket")
    try:
        if not client.bucket(bucket).exists():
            return JournalProbeResult(False, "bucket_not_found")
    except BaseException as exc:  # noqa: BLE001 - health must never propagate
        return JournalProbeResult(False, type(exc).__name__)
    return JournalProbeResult(True, "reachable")


def probe_journal(
    *,
    backend: str,
    root: Path | str | None = None,
    client: Any = None,
    bucket: str | None = None,
) -> JournalProbeResult:
    """Check the configured journal sink. Never raises."""

    try:
        if backend == "local":
            return _probe_local(root)
        if backend == "gcs":
            return _probe_gcs(client, bucket)
        # An unrecognised backend is unhealthy rather than assumed fine: a new
        # sink should have to prove itself.
        return JournalProbeResult(False, "unknown_backend")
    except BaseException as exc:  # noqa: BLE001
        return JournalProbeResult(False, type(exc).__name__)


__all__ = ["JournalProbeResult", "probe_journal"]
