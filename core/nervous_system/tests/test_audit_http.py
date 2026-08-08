"""The read-only audit surface (Task 25).

This handler exists to let a human see what the system decided. It must never
become a way to make it decide something. Every test below is about one of two
properties:

*It cannot act.* No mutating verb is served, no route reaches a gateway or a
broker, and a GET never changes state as a side effect — not even to
acknowledge an alert.

*It cannot leak.* Audit output is the most likely thing to be pasted into a
ticket or a screenshot, so DSNs, keys, headers, broker payloads, and raw
exception text must never reach it. Redaction is recursive because secrets do
not stay at the top level.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from core.nervous_system.contracts.enums import PolicyMode, RuntimeEnvironment
from core.nervous_system.orchestration.http import (
    AuditRequest,
    AuditRouter,
    HealthReport,
    redact,
)


NOW = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)


class _Store:
    """Stands in for the repositories; records what the handler asked for."""

    def __init__(self, **overrides):
        self.calls: list[tuple[str, dict]] = []
        self._overrides = overrides
        self.healthy = overrides.get("healthy", True)

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if not self.healthy:
            raise ConnectionError(
                "could not connect to postgresql://user:hunter2@10.0.0.1/db"
            )
        return self._overrides.get(name, [])

    def decisions(self, **kwargs):
        return self._record("decisions", **kwargs)

    def decision(self, decision_record_id, **kwargs):
        rows = self._overrides.get("decision")
        self.calls.append(("decision", {"id": decision_record_id}))
        return rows

    def alerts(self, **kwargs):
        return self._record("alerts", **kwargs)

    def reconciliations(self, **kwargs):
        return self._record("reconciliations", **kwargs)

    def health(self):
        self.calls.append(("health", {}))
        if not self.healthy:
            raise ConnectionError("postgres down")
        return HealthReport(
            schema_revision="0004_audit_observability",
            database_ok=True,
            journal_ok=True,
            latest_job_heartbeat=NOW,
            latest_reconciliation=NOW,
            open_critical_alerts=0,
            stale_states=(),
            checked_at=NOW,
        )


def _router(**overrides) -> AuditRouter:
    return AuditRouter(
        store=_Store(**overrides),
        environment=RuntimeEnvironment.QA_PAPER,
        policy_mode=PolicyMode.SHADOW,
        account_alias="paper",
    )


def _get(router, path="/api/nervous-system/decisions", **query):
    return router.handle(AuditRequest(method="GET", path=path, query=query))


# ---------------------------------------------------------------------------
# It cannot act
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_no_mutating_verb_is_served(method: str) -> None:
    """A read-only surface that answers one write verb is not read-only."""

    router = _router()
    response = router.handle(
        AuditRequest(method=method, path="/api/nervous-system/decisions", query={})
    )

    assert response.status == 405
    assert router.store.calls == [], "a rejected verb must not touch the store"


def test_a_lowercase_mutating_verb_is_not_a_loophole() -> None:
    response = _router().handle(
        AuditRequest(method="post", path="/api/nervous-system/decisions", query={})
    )

    assert response.status == 405


def test_verb_matching_is_case_insensitive_for_reads_too() -> None:
    """The comparison is normalised in both directions. Rejecting a lowercase
    "get" would be safe but wrong — it would make the surface depend on how a
    proxy happened to spell the verb.
    """

    response = _router().handle(
        AuditRequest(method="get", path="/api/nervous-system/decisions", query={})
    )

    assert response.status == 200


def test_the_router_exposes_no_gateway_or_broker() -> None:
    """Asserted on the object: a route that could reach execution is a route
    somebody will eventually call.
    """

    router = _router()
    names = {name.lower() for name in dir(router) if not name.startswith("_")}

    assert not {name for name in names if "submit" in name or "broker" in name}
    assert not {name for name in names if "gateway" in name or "order" in name}


def test_reading_alerts_does_not_acknowledge_them() -> None:
    """A GET that changes state is a write wearing a read's clothes."""

    router = _router(alerts=[{"code": "ORDER_STUCK", "occurrence_count": 2}])

    _get(router, "/api/nervous-system/alerts")

    assert [name for name, _ in router.store.calls] == ["alerts"]


def test_an_unknown_route_is_a_404() -> None:
    assert _get(_router(), "/api/nervous-system/nope").status == 404


# ---------------------------------------------------------------------------
# It cannot leak
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"database_url": "postgresql://user:hunter2@10.0.0.1/db"},
        {"nested": {"api_key": "sk-live-abcdef"}},
        {"list": [{"authorization": "Bearer abcdef"}]},
        {"deep": {"deeper": {"password": "hunter2"}}},
        {"profile_path": "/home/luket/.config/alpaca/creds.json"},
        {"broker_payload": {"account_id": "PA3XYZ", "raw": "..."}},
    ],
)
def test_secrets_are_redacted_at_any_depth(payload: dict) -> None:
    """Secrets do not stay at the top level, so redaction cannot either."""

    rendered = json.dumps(redact(payload))

    for leaked in ("hunter2", "sk-live-abcdef", "Bearer abcdef", "PA3XYZ", "creds.json"):
        assert leaked not in rendered


def test_redaction_keeps_the_shape_so_the_view_still_works() -> None:
    """Dropping the key entirely would make the response schema unstable."""

    redacted = redact({"database_url": "postgresql://u:p@h/db", "environment": "QA_PAPER"})

    assert set(redacted) == {"database_url", "environment"}
    assert redacted["environment"] == "QA_PAPER"
    assert redacted["database_url"] == "[REDACTED]"


def test_safe_operational_fields_survive_redaction() -> None:
    payload = {
        "environment": "QA_PAPER", "policy_mode": "SHADOW", "account_alias": "paper",
        "schema_revision": "0004_audit_observability", "config_hash": "a" * 64,
    }

    assert redact(payload) == payload


def test_an_exception_never_reaches_the_response_verbatim() -> None:
    """A stack trace or a driver error string routinely contains the DSN."""

    response = _get(_router(healthy=False))

    assert response.status == 503
    body = json.dumps(response.body)
    assert "hunter2" not in body
    assert "10.0.0.1" not in body


# ---------------------------------------------------------------------------
# Bounded, indexed reads
# ---------------------------------------------------------------------------


def test_a_list_applies_a_default_limit() -> None:
    """An unbounded list over a growing audit table is an outage waiting to
    happen.
    """

    router = _router()
    _get(router)

    assert router.store.calls[0][1]["limit"] == 50


def test_a_caller_may_narrow_but_not_widen_the_limit() -> None:
    router = _router()
    _get(router, limit="500")

    assert router.store.calls[0][1]["limit"] == 200


def test_a_smaller_limit_is_honoured() -> None:
    router = _router()
    _get(router, limit="10")

    assert router.store.calls[0][1]["limit"] == 10


@pytest.mark.parametrize("value", ["0", "-5", "abc", ""])
def test_a_nonsense_limit_is_a_400(value: str) -> None:
    assert _get(_router(), limit=value).status == 400


def test_filters_are_passed_through_to_the_indexed_query() -> None:
    router = _router()
    _get(router, strategy_id="meta_ranker", ticker="AMD")

    assert router.store.calls[0][1]["strategy_id"] == "meta_ranker"
    assert router.store.calls[0][1]["ticker"] == "AMD"


def test_a_missing_detail_is_a_404_not_an_empty_success() -> None:
    router = _router(decision=None)

    response = _get(router, "/api/nervous-system/decisions/detail", id="abc")

    assert response.status == 404


# ---------------------------------------------------------------------------
# Health separates liveness from readiness
# ---------------------------------------------------------------------------


def test_liveness_answers_without_a_database() -> None:
    """The process being alive and the system being ready are different
    questions; conflating them makes a restart loop out of a DB blip.
    """

    response = _router(healthy=False).handle(
        AuditRequest(method="GET", path="/api/nervous-system/live", query={})
    )

    assert response.status == 200
    assert response.body["alive"] is True


def test_readiness_reports_the_environment_mode_and_veto() -> None:
    response = _get(_router(), "/api/nervous-system/health")

    assert response.status == 200
    assert response.body["environment"] == "QA_PAPER"
    assert response.body["policy_mode"] == "SHADOW"
    assert response.body["account_alias"] == "paper"
    assert response.body["production_live"] == "BLOCKED BY MVP POLICY"


def test_readiness_is_503_when_the_database_is_unreachable() -> None:
    response = _get(_router(healthy=False), "/api/nervous-system/health")

    assert response.status == 503
    assert response.body["ready"] is False


def test_a_reachable_but_unready_system_is_still_503() -> None:
    """Reachable is not ready. A store that answers while its journal is gone
    would otherwise report a healthy 200 with a failure buried in the body,
    which no load balancer or operator reads.
    """

    class _Degraded(_Store):
        def health(self):
            self.calls.append(("health", {}))
            return HealthReport(
                schema_revision="0004_audit_observability",
                database_ok=True,
                journal_ok=False,
                latest_job_heartbeat=NOW,
                latest_reconciliation=NOW,
                open_critical_alerts=1,
                stale_states=("readiness",),
                checked_at=NOW,
            )

    router = AuditRouter(
        store=_Degraded(),
        environment=RuntimeEnvironment.QA_PAPER,
        policy_mode=PolicyMode.SHADOW,
        account_alias="paper",
    )
    response = _get(router, "/api/nervous-system/health")

    assert response.status == 503
    assert response.body["ready"] is False
    assert response.body["journal_ok"] is False


def test_readiness_reports_the_schema_revision() -> None:
    response = _get(_router(), "/api/nervous-system/health")

    assert response.body["schema_revision"] == "0004_audit_observability"


def test_every_timestamp_is_utc() -> None:
    response = _get(_router(), "/api/nervous-system/health")

    assert response.body["checked_at"].endswith("+00:00")
