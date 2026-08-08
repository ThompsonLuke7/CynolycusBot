"""The read-only audit surface.

This exists so a human can see what the system decided. It must never become a
way to make it decide something, which drives two properties:

*It cannot act.* Only GET is served. No route reaches a gateway, a broker, or a
credential, and a GET never changes state as a side effect — not even to
acknowledge an alert. Reading a problem is not the same as accepting it.

*It cannot leak.* Audit output is the most likely thing in the system to end up
in a ticket or a screenshot, so DSNs, keys, headers, broker payloads, and raw
exception text never reach it. Redaction is recursive because secrets do not
stay at the top level.

Deliberately transport-agnostic: it takes a request value and returns a
response value, so the same handler serves the local compatibility route today
and a request-driven audit service later without the rules being reimplemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.nervous_system.contracts.enums import PolicyMode, RuntimeEnvironment


DEFAULT_LIMIT = 50
MAX_LIMIT = 200

REDACTED = "[REDACTED]"

# Substrings that mark a value as never-displayable. Matched on the key rather
# than the value: a key called "password" is a secret regardless of what it
# currently holds, including when it is empty or a placeholder.
_SECRET_KEY_MARKERS = (
    "password", "secret", "token", "api_key", "apikey", "authorization",
    "credential", "database_url", "dsn", "private", "profile_path", "key_id",
    "account_id", "broker_payload", "raw",
)

_ROUTE_PREFIX = "/api/nervous-system"


@dataclass(frozen=True)
class AuditRequest:
    method: str
    path: str
    query: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditResponse:
    status: int
    body: Any


@dataclass(frozen=True)
class HealthReport:
    schema_revision: str
    database_ok: bool
    journal_ok: bool
    latest_job_heartbeat: datetime | None
    latest_reconciliation: datetime | None
    open_critical_alerts: int
    stale_states: Sequence[str]
    checked_at: datetime


def _is_secret(key: str) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def redact(value: Any) -> Any:
    """Recursively replace secret-bearing values, keeping the shape intact.

    The shape matters: dropping keys entirely would make the response schema
    depend on whether a secret happened to be present, and a consumer cannot
    code against that.
    """

    if isinstance(value, Mapping):
        return {
            key: REDACTED if _is_secret(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def _utc(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


class AuditRouter:
    """Serves GET reads over the audit store, and nothing else."""

    def __init__(
        self,
        *,
        store: Any,
        environment: RuntimeEnvironment,
        policy_mode: PolicyMode,
        account_alias: str,
    ) -> None:
        self.store = store
        self._environment = environment
        self._policy_mode = policy_mode
        self._account_alias = account_alias

    # -- public surface -----------------------------------------------------

    def handle(self, request: AuditRequest) -> AuditResponse:
        # Compared case-insensitively so a lowercase verb is not a loophole.
        if str(request.method).upper() != "GET":
            return AuditResponse(405, {"error": "this surface is read-only"})

        path = request.path.rstrip("/")
        if path == f"{_ROUTE_PREFIX}/live":
            # Liveness deliberately touches nothing: the process being alive
            # and the system being ready are different questions, and
            # conflating them turns a database blip into a restart loop.
            return AuditResponse(200, {"alive": True})

        try:
            return self._route(path, request.query)
        except _BadRequest as exc:
            return AuditResponse(400, {"error": str(exc)})
        except _NotFound:
            return AuditResponse(404, {"error": "not found"})
        except Exception:  # noqa: BLE001
            # Never the exception text: a driver error routinely carries the
            # DSN, and this response is the most likely thing to be pasted
            # into a ticket.
            return AuditResponse(
                503, {"ready": False, "error": "audit store unavailable"}
            )

    # -- routing ------------------------------------------------------------

    def _route(self, path: str, query: Mapping[str, str]) -> AuditResponse:
        if path == f"{_ROUTE_PREFIX}/health":
            return self._health()
        if path == f"{_ROUTE_PREFIX}/decisions":
            return self._list(self.store.decisions, query)
        if path == f"{_ROUTE_PREFIX}/decisions/detail":
            return self._detail(query)
        if path == f"{_ROUTE_PREFIX}/alerts":
            return self._list(self.store.alerts, query)
        if path == f"{_ROUTE_PREFIX}/reconciliations":
            return self._list(self.store.reconciliations, query)
        raise _NotFound()

    def _list(self, reader: Any, query: Mapping[str, str]) -> AuditResponse:
        rows = reader(
            limit=_limit(query),
            **{
                name: query[name]
                for name in ("strategy_id", "ticker", "environment", "account_alias")
                if name in query
            },
        )
        return AuditResponse(200, {"items": redact(list(rows))})

    def _detail(self, query: Mapping[str, str]) -> AuditResponse:
        identifier = query.get("id")
        if not identifier:
            raise _BadRequest("id is required")
        row = self.store.decision(identifier)
        if row is None:
            # An empty success would read as "this decision had no content"
            # rather than "this decision does not exist".
            raise _NotFound()
        return AuditResponse(200, redact(row))

    def _health(self) -> AuditResponse:
        report = self.store.health()
        ready = bool(report.database_ok and report.journal_ok)
        body = {
            "ready": ready,
            "environment": self._environment.value,
            "policy_mode": self._policy_mode.value,
            "account_alias": self._account_alias,
            # Stated as a status rather than a capability: production-live is
            # not something this build can be configured into.
            "production_live": "BLOCKED BY MVP POLICY",
            "schema_revision": report.schema_revision,
            "database_ok": report.database_ok,
            "journal_ok": report.journal_ok,
            "open_critical_alerts": report.open_critical_alerts,
            "stale_states": list(report.stale_states),
            "latest_job_heartbeat": _utc(report.latest_job_heartbeat),
            "latest_reconciliation": _utc(report.latest_reconciliation),
            "checked_at": _utc(report.checked_at),
        }
        return AuditResponse(200 if ready else 503, redact(body))


class _BadRequest(ValueError):
    pass


class _NotFound(LookupError):
    pass


def _limit(query: Mapping[str, str]) -> int:
    """Bounded by default and capped on request.

    An unbounded list over a table that only grows is an outage waiting to
    happen, so a caller may narrow the page but never widen it past the cap.
    """

    raw = query.get("limit")
    if raw is None:
        return DEFAULT_LIMIT
    try:
        value = int(str(raw))
    except (TypeError, ValueError):
        raise _BadRequest("limit must be a positive integer") from None
    if value <= 0:
        raise _BadRequest("limit must be a positive integer")
    return min(value, MAX_LIMIT)


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "AuditRequest",
    "AuditResponse",
    "AuditRouter",
    "HealthReport",
    "redact",
]
