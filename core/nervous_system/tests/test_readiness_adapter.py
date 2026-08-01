from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from core.live_readiness import readiness_status
from core.nervous_system.context.readiness_adapter import adapt_readiness_status


UTC = timezone.utc
CHECKED_AT = datetime(2026, 7, 30, 21, 0, tzinfo=UTC)
COMPLETED_AT = datetime(2026, 7, 30, 20, 30, tzinfo=UTC)


def _payload(completed_at: object = COMPLETED_AT) -> dict[str, object]:
    return {
        "job": "nightly_data_readiness",
        "status": "success",
        "completed_at_utc": completed_at.isoformat() if isinstance(completed_at, datetime) else completed_at,
        "latest_required_session": "2026-07-29",
        "version": 1,
    }


def test_current_success_maps_completion_to_availability_and_checked_to_observation() -> None:
    state = adapt_readiness_status(
        ready=True,
        reason="readiness stamp OK (0.5h old)",
        payload=_payload(),
        checked_at=CHECKED_AT,
        max_age_hours=96.0,
    )

    assert state.ready is True
    assert state.status == "CURRENT"
    assert state.completed_at == COMPLETED_AT
    assert state.available_at == COMPLETED_AT
    assert state.checked_at == CHECKED_AT
    assert state.generated_at == CHECKED_AT
    assert state.latest_required_session == "2026-07-29"
    assert state.reason_codes == ()


def test_stale_success_remains_observed_with_completed_availability() -> None:
    completed = CHECKED_AT - timedelta(hours=120)
    state = adapt_readiness_status(
        ready=False,
        reason="readiness stamp is 120.0h old (> 96.0h)",
        payload=_payload(completed),
        checked_at=CHECKED_AT,
        max_age_hours=96.0,
    )

    assert state.ready is False
    assert state.status == "STALE"
    assert state.available_at == completed
    assert state.completed_at == completed
    assert state.reason_codes == ("READINESS_STALE",)


def test_missing_stamp_is_a_non_ready_observation_at_check_time() -> None:
    state = adapt_readiness_status(
        ready=False,
        reason="missing readiness stamp /tmp/latest_success.json",
        payload={},
        checked_at=CHECKED_AT,
        max_age_hours=96.0,
    )

    assert state.ready is False
    assert state.status == "MISSING"
    assert state.completed_at is None
    assert state.available_at == CHECKED_AT
    assert "READINESS_MISSING" in state.reason_codes


def test_invalid_or_naive_completion_timestamp_is_not_silently_treated_as_utc() -> None:
    for value in ("not-a-timestamp", "2026-07-30T20:30:00"):
        state = adapt_readiness_status(
            ready=False,
            reason=f"invalid readiness timestamp {value!r}",
            payload=_payload(value),
            checked_at=CHECKED_AT,
            max_age_hours=96.0,
        )
        assert state.ready is False
        assert state.status == "INVALID"
        assert state.completed_at is None
        assert state.available_at == CHECKED_AT
        assert "READINESS_INVALID_TIMESTAMP" in state.reason_codes


def test_naive_observation_timestamp_is_rejected_without_inventing_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        adapt_readiness_status(
            ready=True,
            reason="readiness stamp OK",
            payload=_payload(),
            checked_at=CHECKED_AT.replace(tzinfo=None),
            max_age_hours=96.0,
        )


def test_disabled_gate_is_not_accepted_in_qa_paper(monkeypatch) -> None:
    monkeypatch.setenv("CYNOLYCUS_READINESS_REQUIRED", "0")
    monkeypatch.setenv("CYNOLYCUS_ENVIRONMENT", "QA_PAPER")
    ready, reason, payload = readiness_status(now=CHECKED_AT)

    assert (ready, reason, payload) == (
        True,
        "readiness gate disabled by CYNOLYCUS_READINESS_REQUIRED=0",
        {},
    )
    state = adapt_readiness_status(
        ready=ready,
        reason=reason,
        payload=payload,
        checked_at=CHECKED_AT,
        max_age_hours=96.0,
    )

    assert state.ready is False
    assert state.status == "DISABLED"
    assert state.completed_at is None
    assert state.available_at == CHECKED_AT
    assert state.reason_codes == ("READINESS_DISABLED_NOT_ACCEPTED_IN_QA",)


def test_disabled_gate_is_warning_only_in_development(monkeypatch) -> None:
    monkeypatch.setenv("CYNOLYCUS_READINESS_REQUIRED", "0")
    monkeypatch.setenv("CYNOLYCUS_ENVIRONMENT", "DEVELOPMENT")
    ready, reason, payload = readiness_status(now=CHECKED_AT)

    assert (ready, reason, payload) == (
        True,
        "readiness gate disabled by CYNOLYCUS_READINESS_REQUIRED=0",
        {},
    )
    state = adapt_readiness_status(
        ready=ready,
        reason=reason,
        payload=payload,
        checked_at=CHECKED_AT,
        max_age_hours=96.0,
    )

    assert state.ready is True
    assert state.status == "DISABLED"
    assert state.completed_at is None
    assert state.available_at == CHECKED_AT
    assert state.reason_codes == ("READINESS_DISABLED_WARNING",)
    assert tuple(issue.code for issue in state.data_quality.issues) == (
        "READINESS_DISABLED_WARNING",
    )
    assert state.data_quality.is_usable is True


def test_source_hash_identity_versions_and_quality_are_deterministic() -> None:
    first = adapt_readiness_status(
        ready=True,
        reason="readiness stamp OK (0.5h old)",
        payload=_payload(),
        checked_at=CHECKED_AT,
        max_age_hours=96.0,
    )
    second = adapt_readiness_status(
        ready=True,
        reason="readiness stamp OK (0.5h old)",
        payload=_payload(),
        checked_at=CHECKED_AT,
        max_age_hours=96.0,
    )

    expected_source_hash = hashlib.sha256(
        json.dumps(_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert first.state_id == second.state_id
    assert first.lineage_ids == second.lineage_ids == (expected_source_hash,)
    assert first.producer == second.producer == "core.live_readiness"
    assert first.model_version == second.model_version == "readiness-adapter@1"
    assert first.feature_version == second.feature_version == "readiness@1"
    assert first.config_version == second.config_version == "readiness-policy@1"
    assert first.data_quality == second.data_quality

    changed_payload = _payload()
    changed_payload["version"] = 2
    changed = adapt_readiness_status(
        ready=True,
        reason="readiness stamp OK (0.5h old)",
        payload=changed_payload,
        checked_at=CHECKED_AT,
        max_age_hours=96.0,
    )
    assert changed.lineage_ids != first.lineage_ids


def test_existing_latest_completed_session_check_remains_authoritative(tmp_path) -> None:
    stamp = tmp_path / "latest_success.json"
    stamp_payload = {
        "job": "nightly_data_readiness",
        "status": "success",
        # Sunday stamp authorizes Monday but not Tuesday after Monday traded.
        "completed_at_utc": "2026-07-12T17:26:44+00:00",
    }
    stamp.write_text(json.dumps(stamp_payload))
    monday_result = readiness_status(
        path=stamp,
        now=datetime(2026, 7, 13, 13, 0, tzinfo=UTC),
    )
    tuesday_result = readiness_status(
        path=stamp,
        now=datetime(2026, 7, 14, 13, 0, tzinfo=UTC),
    )

    assert monday_result == (
        True,
        "readiness stamp OK (19.6h old)",
        stamp_payload,
    )
    assert tuesday_result == (
        False,
        "readiness stamp predates latest completed trading session "
        "(2026-07-13 16:00 ET)",
        stamp_payload,
    )
    monday_ok, monday_reason, monday_payload = monday_result
    tuesday_ok, tuesday_reason, tuesday_payload = tuesday_result

    monday = adapt_readiness_status(
        ready=monday_ok,
        reason=monday_reason,
        payload=monday_payload,
        checked_at=datetime(2026, 7, 13, 13, 0, tzinfo=UTC),
        max_age_hours=96.0,
    )
    tuesday = adapt_readiness_status(
        ready=tuesday_ok,
        reason=tuesday_reason,
        payload=tuesday_payload,
        checked_at=datetime(2026, 7, 14, 13, 0, tzinfo=UTC),
        max_age_hours=96.0,
    )
    assert monday.ready is True
    assert tuesday.ready is False
    assert tuesday.status == "STALE"
    assert tuesday.reason_codes == (
        "READINESS_PREDATES_LATEST_COMPLETED_SESSION",
    )
    assert tuple(issue.code for issue in tuesday.data_quality.issues) == (
        "READINESS_PREDATES_LATEST_COMPLETED_SESSION",
    )
