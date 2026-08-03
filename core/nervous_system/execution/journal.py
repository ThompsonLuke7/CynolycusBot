"""Always-on execution journal.

The journal is the outage backup for execution evidence. PostgreSQL is the
operational authority and is *not* a journal sink; if the database is
unavailable the journal must still hold enough evidence to reconstruct what
was sent to the broker.

Every record is immutable. A local file is installed with a create-exclusive
link rather than a rename, because ``os.replace`` overwrites; a GCS object is
uploaded with ``if_generation_match=0``. Writing the same event twice
converges instead of duplicating; writing *different* bytes under the same
identity is a ``JournalConflict``, never a silent overwrite.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import AfterValidator, Field, model_validator

from core.nervous_system.contracts.base import (
    ContractModel,
    PositiveSchemaVersion,
    Sha256Hex,
    UtcDatetime,
    _canonicalize,
    _freeze_mapping,
)
from core.nervous_system.contracts.enums import RuntimeEnvironment


JOURNAL_PREFIX = "execution_journal"
JOURNAL_VERSION = "v1"

ImmutableObjectMap = Annotated[dict[str, object], AfterValidator(_freeze_mapping)]

# Anything whose key looks like one of these never reaches a hash or a sink.
_SECRET_HINTS = (
    "password",
    "passwd",
    "api_key",
    "apikey",
    "apca",
    "secret",
    "access_token",
    "refresh_token",
    "token",
    "private_key",
    "authorization",
    "auth_header",
    "credential",
    "account_number",
    "account_id",
)
# Order identifiers legitimately contain "order" and are not credentials.
_SECRET_EXEMPT = ("order",)
REDACTED = "***redacted***"


class JournalError(Exception):
    """Base class for journal failures."""


class JournalConflict(JournalError):
    """The same identity already holds different bytes."""


class JournalUnavailable(JournalError):
    """A sink could not be written and the event is not durable there."""


class JournalBackend(str, Enum):
    LOCAL = "LOCAL"
    GCS = "GCS"


class JournalWriteStatus(str, Enum):
    WRITTEN = "WRITTEN"
    IDEMPOTENT = "IDEMPOTENT"


class CompositeStatus(str, Enum):
    DURABLE = "DURABLE"
    IDEMPOTENT = "IDEMPOTENT"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    CONFLICT = "CONFLICT"


class PostgresPersistenceStatus(str, Enum):
    PENDING = "PENDING"
    PERSISTED = "PERSISTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


def redact(value: Any) -> Any:
    """Recursively remove credentials and private account identifiers."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            secret = any(hint in lowered for hint in _SECRET_HINTS) and not any(
                exempt in lowered for exempt in _SECRET_EXEMPT
            )
            cleaned[name] = REDACTED if secret else redact(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class JournalLocator(ContractModel):
    """Exactly where an event lives; reads never rediscover it by scanning."""

    backend: JournalBackend
    object_name: str
    uri: str


class JournalReceipt(ContractModel):
    backend: JournalBackend
    locator: JournalLocator
    content_hash: Sha256Hex
    status: JournalWriteStatus


class ExecutionJournalEvent(ContractModel):
    event_id: UUID
    event_time: UtcDatetime
    observed_at: UtcDatetime
    account_id: str
    environment: RuntimeEnvironment
    event_type: str
    decision_id: UUID | None
    order_request_id: UUID
    sequence_no: Annotated[int, Field(ge=1)]
    client_order_id: str
    broker_order_id: str | None
    payload: ImmutableObjectMap = Field(default_factory=dict)
    previous_event_id: UUID | None = None
    previous_event_hash: Sha256Hex | None = None
    postgres_persistence_status: PostgresPersistenceStatus = (
        PostgresPersistenceStatus.PENDING
    )
    schema_version: PositiveSchemaVersion = 1
    event_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_event(self) -> ExecutionJournalEvent:
        if self.sequence_no == 1:
            if self.previous_event_id is not None or self.previous_event_hash is not None:
                raise ValueError("sequence 1 has no predecessor")
        elif self.previous_event_id is None or self.previous_event_hash is None:
            raise ValueError(
                "events after sequence 1 require both predecessor id and hash"
            )
        if self.observed_at < self.event_time:
            raise ValueError("observed_at must not precede event_time")
        if not self.account_id.strip():
            raise ValueError("account_id (the account alias) is required")
        if self.environment is RuntimeEnvironment.PRODUCTION_LIVE:
            raise ValueError("PRODUCTION_LIVE journal events are refused")
        if self.event_hash != self.computed_event_hash():
            raise ValueError("event_hash does not match event content")
        return self

    def computed_event_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes(with_hash=False)).hexdigest()

    def canonical_bytes(self, *, with_hash: bool = True) -> bytes:
        exclude = None if with_hash else {"event_hash"}
        payload = _canonicalize(self.model_dump(mode="python", exclude=exclude))
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @property
    def object_name(self) -> str:
        """The one canonical identity used by every sink."""

        stamp = self.event_time.strftime("%Y%m%dT%H%M%S%fZ")
        return (
            f"{JOURNAL_PREFIX}/{JOURNAL_VERSION}/"
            f"{self.event_time:%Y/%m/%d}/{self.account_id}/"
            f"{stamp}_{self.event_id}.json"
        )

    @classmethod
    def create(cls, **fields: Any) -> ExecutionJournalEvent:
        """Build an event, redacting the payload before it is ever hashed."""

        fields = dict(fields)
        fields["payload"] = redact(fields.get("payload") or {})
        probe = cls.model_construct(**fields, event_hash="0" * 64)
        return cls(**fields, event_hash=probe.computed_event_hash())


@runtime_checkable
class ExecutionJournal(Protocol):
    def write(self, event: ExecutionJournalEvent) -> JournalReceipt: ...

    def read(self, locator: JournalLocator) -> ExecutionJournalEvent: ...

    def iter_events(
        self,
        *,
        account_id: str,
        after: datetime | None = None,
    ) -> Iterator[ExecutionJournalEvent]: ...


class LocalAtomicJournal:
    """Append-only local sink installed without overwrite."""

    backend = JournalBackend.LOCAL

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def path_for(self, event: ExecutionJournalEvent) -> Path:
        return self._root / event.object_name

    def write(self, event: ExecutionJournalEvent) -> JournalReceipt:
        final = self.path_for(event)
        final.parent.mkdir(parents=True, exist_ok=True)
        data = event.canonical_bytes()

        handle, temp_name = tempfile.mkstemp(
            dir=str(final.parent), prefix=".journal-", suffix=".tmp"
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # link() refuses to clobber an existing name, so an installed
                # record can never be silently replaced. os.replace would.
                os.link(temp, final)
            except FileExistsError:
                return self._reconcile_existing(final, data, event)
            self._fsync_directory(final.parent)
        finally:
            # A crash before link leaves only this temp file, which recovery
            # ignores; the final name never appears half-written.
            temp.unlink(missing_ok=True)

        return JournalReceipt(
            backend=self.backend,
            locator=self._locator(final, event),
            content_hash=hashlib.sha256(data).hexdigest(),
            status=JournalWriteStatus.WRITTEN,
        )

    def _reconcile_existing(
        self,
        final: Path,
        data: bytes,
        event: ExecutionJournalEvent,
    ) -> JournalReceipt:
        existing = final.read_bytes()
        if existing != data:
            raise JournalConflict(
                f"{final} already holds different bytes for event {event.event_id}"
            )
        return JournalReceipt(
            backend=self.backend,
            locator=self._locator(final, event),
            content_hash=hashlib.sha256(data).hexdigest(),
            status=JournalWriteStatus.IDEMPOTENT,
        )

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _locator(self, path: Path, event: ExecutionJournalEvent) -> JournalLocator:
        return JournalLocator(
            backend=self.backend,
            object_name=event.object_name,
            uri=str(path),
        )

    def read(self, locator: JournalLocator) -> ExecutionJournalEvent:
        if locator.backend is not self.backend:
            raise JournalError(f"locator is not local: {locator.backend.value}")
        return _decode(Path(locator.uri).read_bytes())

    def iter_events(
        self,
        *,
        account_id: str,
        after: datetime | None = None,
    ) -> Iterator[ExecutionJournalEvent]:
        base = self._root / JOURNAL_PREFIX / JOURNAL_VERSION
        if not base.exists():
            return
        for path in sorted(base.glob(f"*/*/*/{account_id}/*.json")):
            event = _decode(path.read_bytes())
            if after is not None and event.event_time <= after:
                continue
            yield event


def _is_precondition_failure(exc: BaseException) -> bool:
    if getattr(exc, "code", None) == 412:
        return True
    return type(exc).__name__ in {"PreconditionFailed", "AlreadyExists"}


class GCSImmutableJournal:
    """Immutable object sink using a generation precondition."""

    backend = JournalBackend.GCS

    def __init__(
        self,
        client: Any,
        bucket_name: str,
        *,
        precondition_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        if not bucket_name:
            raise ValueError("bucket_name is required")
        self._client = client
        self._bucket_name = bucket_name
        self._precondition_errors = precondition_errors

    def write(self, event: ExecutionJournalEvent) -> JournalReceipt:
        data = event.canonical_bytes()
        blob = self._client.bucket(self._bucket_name).blob(event.object_name)
        try:
            # if_generation_match=0 means "only if this object does not exist",
            # so the write is immutable without ever listing the bucket.
            blob.upload_from_string(
                data, content_type="application/json", if_generation_match=0
            )
        except self._precondition_errors as exc:  # type: ignore[misc]
            return self._reconcile_existing(blob, data, event, exc)
        except Exception as exc:
            if _is_precondition_failure(exc):
                return self._reconcile_existing(blob, data, event, exc)
            raise JournalUnavailable(
                f"GCS upload failed for {event.object_name}: {type(exc).__name__}"
            ) from exc
        return JournalReceipt(
            backend=self.backend,
            locator=self._locator(event),
            content_hash=hashlib.sha256(data).hexdigest(),
            status=JournalWriteStatus.WRITTEN,
        )

    def _reconcile_existing(
        self,
        blob: Any,
        data: bytes,
        event: ExecutionJournalEvent,
        cause: BaseException,
    ) -> JournalReceipt:
        # Fetch this exact object rather than listing: identical bytes mean the
        # earlier attempt already succeeded.
        existing = blob.download_as_bytes()
        if existing != data:
            raise JournalConflict(
                f"gs://{self._bucket_name}/{event.object_name} already holds "
                f"different bytes for event {event.event_id}"
            ) from cause
        return JournalReceipt(
            backend=self.backend,
            locator=self._locator(event),
            content_hash=hashlib.sha256(data).hexdigest(),
            status=JournalWriteStatus.IDEMPOTENT,
        )

    def _locator(self, event: ExecutionJournalEvent) -> JournalLocator:
        return JournalLocator(
            backend=self.backend,
            object_name=event.object_name,
            uri=f"gs://{self._bucket_name}/{event.object_name}",
        )

    def read(self, locator: JournalLocator) -> ExecutionJournalEvent:
        if locator.backend is not self.backend:
            raise JournalError(f"locator is not GCS: {locator.backend.value}")
        blob = self._client.bucket(self._bucket_name).blob(locator.object_name)
        return _decode(blob.download_as_bytes())

    def iter_events(
        self,
        *,
        account_id: str,
        after: datetime | None = None,
    ) -> Iterator[ExecutionJournalEvent]:
        prefix = f"{JOURNAL_PREFIX}/{JOURNAL_VERSION}/"
        for blob in sorted(
            self._client.list_blobs(self._bucket_name, prefix=prefix),
            key=lambda item: item.name,
        ):
            if f"/{account_id}/" not in blob.name:
                continue
            event = _decode(blob.download_as_bytes())
            if after is not None and event.event_time <= after:
                continue
            yield event


class CompositeJournalResult(ContractModel):
    status: CompositeStatus
    receipts: tuple[JournalReceipt, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def is_durable(self) -> bool:
        return self.status in {CompositeStatus.DURABLE, CompositeStatus.IDEMPOTENT}


class CompositeExecutionJournal:
    """Fan out to required and optional sinks without ever rolling one back."""

    def __init__(
        self,
        *,
        required: Sequence[Any] = (),
        optional: Sequence[Any] = (),
    ) -> None:
        if not required:
            raise ValueError("at least one required journal sink is needed")
        self._required = tuple(required)
        self._optional = tuple(optional)

    def write(self, event: ExecutionJournalEvent) -> CompositeJournalResult:
        receipts: list[JournalReceipt] = []
        failures: list[str] = []
        conflict = False
        required_ok = True

        for sink, is_required in (
            *((sink, True) for sink in self._required),
            *((sink, False) for sink in self._optional),
        ):
            label = _name(sink) if is_required else f"optional {_name(sink)}"
            try:
                receipts.append(sink.write(event))
            except JournalConflict as exc:
                conflict = True
                failures.append(f"{label}: {exc}")
                if is_required:
                    required_ok = False
            except Exception as exc:
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                if is_required:
                    required_ok = False

        if conflict:
            status = CompositeStatus.CONFLICT
        elif not required_ok:
            status = CompositeStatus.FAILED
        elif failures:
            # A required sink succeeded but an optional one did not: durable,
            # yet explicitly degraded rather than quietly clean.
            status = CompositeStatus.DEGRADED
        elif receipts and all(
            receipt.status is JournalWriteStatus.IDEMPOTENT for receipt in receipts
        ):
            status = CompositeStatus.IDEMPOTENT
        else:
            status = CompositeStatus.DURABLE

        return CompositeJournalResult(
            status=status,
            receipts=tuple(receipts),
            failures=tuple(failures),
        )


def _name(sink: Any) -> str:
    return type(sink).__name__


def _decode(data: bytes) -> ExecutionJournalEvent:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JournalError(f"journal record is not valid JSON: {exc}") from exc
    return ExecutionJournalEvent.model_validate(payload)


def verify_chain(events: Sequence[ExecutionJournalEvent]) -> None:
    """Validate one order's chain, raising on any tamper signature.

    Detects mutation, deletion between checkpoints, reordering, duplicate
    sequence numbers, and events borrowed from another order's chain.
    """

    if not events:
        return
    order_ids = {event.order_request_id for event in events}
    if len(order_ids) != 1:
        raise JournalConflict("a chain must cover exactly one order_request_id")

    previous: ExecutionJournalEvent | None = None
    seen: set[int] = set()
    for event in events:
        if event.event_hash != event.computed_event_hash():
            raise JournalConflict(f"event {event.event_id} has been mutated")
        if event.sequence_no in seen:
            raise JournalConflict(f"duplicate sequence {event.sequence_no}")
        seen.add(event.sequence_no)
        expected = 1 if previous is None else previous.sequence_no + 1
        if event.sequence_no != expected:
            raise JournalConflict(
                f"expected sequence {expected}, found {event.sequence_no}"
            )
        if previous is None:
            if event.previous_event_id is not None:
                raise JournalConflict("the first event cannot cite a predecessor")
        else:
            if event.previous_event_id != previous.event_id:
                raise JournalConflict(
                    f"event {event.event_id} cites the wrong predecessor"
                )
            if event.previous_event_hash != previous.event_hash:
                raise JournalConflict(
                    f"event {event.event_id} breaks the hash chain"
                )
        previous = event


def link_event(
    previous: ExecutionJournalEvent | None,
    **fields: Any,
) -> ExecutionJournalEvent:
    """Create the next event in a chain, carrying the predecessor forward."""

    if previous is None:
        return ExecutionJournalEvent.create(sequence_no=1, **fields)
    if "order_request_id" in fields and fields["order_request_id"] != previous.order_request_id:
        raise JournalConflict("a chain cannot cross order_request_id boundaries")
    fields.setdefault("order_request_id", previous.order_request_id)
    return ExecutionJournalEvent.create(
        sequence_no=previous.sequence_no + 1,
        previous_event_id=previous.event_id,
        previous_event_hash=previous.event_hash,
        **fields,
    )


__all__ = [
    "JOURNAL_PREFIX",
    "JOURNAL_VERSION",
    "REDACTED",
    "CompositeExecutionJournal",
    "CompositeJournalResult",
    "CompositeStatus",
    "ExecutionJournal",
    "ExecutionJournalEvent",
    "GCSImmutableJournal",
    "JournalBackend",
    "JournalConflict",
    "JournalError",
    "JournalLocator",
    "JournalReceipt",
    "JournalUnavailable",
    "JournalWriteStatus",
    "LocalAtomicJournal",
    "PostgresPersistenceStatus",
    "link_event",
    "redact",
    "verify_chain",
]
