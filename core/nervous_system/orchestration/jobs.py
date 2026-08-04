"""Durable job leases and postcondition-verified stages.

A file lock is not a job lease: it does not survive a host change, it cannot
be observed by another process, and it says nothing about whether the work
actually produced anything. Leases here live in PostgreSQL, carry a fencing
token, and a stage is only successful when its *output* is verified, never
because a subprocess exited zero.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class StageStatus(str, Enum):
    OK = "OK"
    FAILED = "FAILED"
    NOT_RUN = "NOT_RUN"
    SKIPPED_CERTIFIED = "SKIPPED_CERTIFIED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class StageResult:
    name: str
    status: StageStatus
    exit_code: int | None = None
    reason: str | None = None
    counts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {StageStatus.OK, StageStatus.SKIPPED_CERTIFIED}


@dataclass
class JobResult:
    job_type: str
    scheduled_for: datetime
    claimed: bool
    status: str
    stages: list[StageResult] = field(default_factory=list)
    job_run_id: UUID | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "SUCCEEDED"

    def stage(self, name: str) -> StageResult | None:
        for item in self.stages:
            if item.name == name:
                return item
        return None


@dataclass(frozen=True)
class Stage:
    """One pipeline step with an explicit postcondition.

    ``verify`` is what makes the stage trustworthy: a zero exit code only
    means the process ended, not that it produced fresh, complete output.
    """

    name: str
    run: Callable[[], int]
    verify: Callable[[], tuple[bool, str, Mapping[str, Any]]] | None = None
    required: bool = True
    skip: bool = False
    skip_certificate: Callable[[], tuple[bool, str]] | None = None


def redact_exception(exc: BaseException) -> dict[str, str]:
    """Summarise a failure without leaking a credential in the message."""

    text = str(exc)
    for marker in ("key=", "secret=", "token=", "password="):
        if marker in text.lower():
            text = "<redacted: message contained a credential marker>"
            break
    return {"type": type(exc).__name__, "message": text[:400]}


def run_stages(stages: Sequence[Stage]) -> list[StageResult]:
    """Run stages in order, marking everything after a failure NOT_RUN.

    A failed or stale required stage prevents every downstream stage. The
    downstream steps are recorded as NOT_RUN rather than omitted, so the audit
    shows what was prevented and why.
    """

    results: list[StageResult] = []
    blocked_by: str | None = None

    for stage in stages:
        if blocked_by is not None:
            results.append(
                StageResult(
                    name=stage.name,
                    status=StageStatus.NOT_RUN,
                    reason=f"blocked by {blocked_by}",
                )
            )
            continue

        if stage.skip:
            certified, detail = (
                stage.skip_certificate() if stage.skip_certificate else (False, "no certificate")
            )
            if not certified:
                # Skipping without proof that the data is already fresh is a
                # silent staleness risk, so it blocks the pipeline.
                results.append(
                    StageResult(
                        name=stage.name,
                        status=StageStatus.FAILED,
                        reason=f"skip refused: {detail}",
                    )
                )
                if stage.required:
                    blocked_by = stage.name
                continue
            results.append(
                StageResult(
                    name=stage.name,
                    status=StageStatus.SKIPPED_CERTIFIED,
                    reason=detail,
                )
            )
            continue

        try:
            exit_code = stage.run()
        except Exception as exc:
            results.append(
                StageResult(
                    name=stage.name,
                    status=StageStatus.FAILED,
                    reason=str(redact_exception(exc)),
                )
            )
            if stage.required:
                blocked_by = stage.name
            continue

        if exit_code != 0:
            results.append(
                StageResult(
                    name=stage.name,
                    status=StageStatus.FAILED,
                    exit_code=exit_code,
                    reason=f"exit code {exit_code}",
                )
            )
            if stage.required:
                blocked_by = stage.name
            continue

        counts: Mapping[str, Any] = {}
        if stage.verify is not None:
            verified, detail, counts = stage.verify()
            if not verified:
                # Exit zero is not enough: the output must actually be there.
                results.append(
                    StageResult(
                        name=stage.name,
                        status=StageStatus.FAILED,
                        exit_code=exit_code,
                        reason=f"postcondition failed: {detail}",
                        counts=counts,
                    )
                )
                if stage.required:
                    blocked_by = stage.name
                continue

        results.append(
            StageResult(
                name=stage.name,
                status=StageStatus.OK,
                exit_code=exit_code,
                counts=counts,
            )
        )

    return results


class JobRunner:
    """Claim a scheduled slot, run its stages, and record the outcome."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], Any],
        worker_id: str,
        host: str,
        revision: str,
        clock: Callable[[], datetime],
        lease_seconds: int = 300,
    ) -> None:
        self._uow_factory = unit_of_work_factory
        self._worker_id = worker_id
        self._host = host
        self._revision = revision
        self._clock = clock
        self._lease_seconds = lease_seconds

    def run_once(
        self,
        *,
        job_type: str,
        scheduled_for: datetime,
        config_hash: str,
        stages: Sequence[Stage],
    ) -> JobResult:
        token = uuid4().hex
        now = self._clock()

        # 1. Short transaction: take the lease and record the claim.
        with self._uow_factory() as uow:
            record, claimed = uow.operations.claim_job(
                job_type=job_type,
                scheduled_for=scheduled_for,
                config_hash=config_hash,
                owner=self._worker_id,
                lease_token=token,
                now=now,
                lease_seconds=self._lease_seconds,
                host=self._host,
                revision=self._revision,
            )
            uow.commit()

        if not claimed:
            return JobResult(
                job_type=job_type,
                scheduled_for=scheduled_for,
                claimed=False,
                status=record.status,
                job_run_id=record.job_run_id,
                reason="another worker owns this scheduled slot",
            )

        # 2. Stages run outside any database transaction.
        results = run_stages(stages)
        failed = [item for item in results if item.status is StageStatus.FAILED]
        status = "FAILED" if failed else "SUCCEEDED"

        # 3. Finalise, fenced by the lease token.
        with self._uow_factory() as uow:
            uow.operations.finish_job(
                record.job_run_id,
                lease_token=token,
                status=status,
                finished_at=self._clock(),
                counts={item.name: dict(item.counts) for item in results},
                error=failed[0].reason if failed else None,
            )
            uow.commit()

        return JobResult(
            job_type=job_type,
            scheduled_for=scheduled_for,
            claimed=True,
            status=status,
            stages=results,
            job_run_id=record.job_run_id,
        )

    def heartbeat(self, job_run_id: UUID, *, lease_token: str) -> bool:
        with self._uow_factory() as uow:
            renewed = uow.operations.heartbeat_job(
                job_run_id,
                lease_token=lease_token,
                now=self._clock(),
                lease_seconds=self._lease_seconds,
            )
            uow.commit()
        return renewed


__all__ = [
    "JobResult",
    "JobRunner",
    "Stage",
    "StageResult",
    "StageStatus",
    "redact_exception",
    "run_stages",
]
