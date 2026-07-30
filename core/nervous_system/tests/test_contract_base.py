from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.nervous_system.contracts.base import (
    ContractModel,
    UtcDatetime,
    content_hash,
)
from core.nervous_system.contracts.enums import (
    AssetClass,
    DataQualitySeverity,
    Direction,
    ExecutionStatus,
    MarketRegime,
    OptionType,
    PolicyAction,
)
from core.nervous_system.contracts.quality import (
    DataQualityIssue,
    DataQualitySummary,
    LineageRef,
)


class Example(ContractModel):
    observed_at: UtcDatetime
    value: float


class Nested(ContractModel):
    values: list[int]


class Container(ContractModel):
    nested: Nested


def test_contract_normalizes_aware_time_to_utc_and_hashes_stably():
    model = Example(
        observed_at=datetime.fromisoformat("2026-07-30T10:00:00-04:00"),
        value=1.25,
    )
    assert model.observed_at == datetime(2026, 7, 30, 14, tzinfo=timezone.utc)
    assert content_hash(model) == content_hash(
        Example.model_validate_json(model.model_dump_json())
    )


def test_contract_rejects_naive_time_unknown_fields_and_nonfinite_number():
    with pytest.raises(ValidationError):
        Example(observed_at=datetime(2026, 7, 30, 10), value=1.0)
    with pytest.raises(ValidationError):
        Example(
            observed_at=datetime(2026, 7, 30, 14, tzinfo=timezone.utc),
            value=float("nan"),
            surprise=True,
        )


def test_contract_model_copy_revalidates_updates():
    model = Example(
        observed_at=datetime(2026, 7, 30, 14, tzinfo=timezone.utc),
        value=1.0,
    )

    with pytest.raises(ValidationError):
        model.model_copy(update={"value": float("nan")})


def test_contract_model_copy_preserves_shallow_and_deep_copy_behavior():
    model = Container(nested={"values": [1]})

    shallow = model.model_copy()
    deep = model.model_copy(deep=True)

    assert shallow == model
    assert shallow.nested is model.nested
    assert deep == model
    assert deep.nested is not model.nested
    assert deep.nested.values is not model.nested.values


def test_required_enums_expose_their_unknown_or_explicit_values():
    assert Direction.UNKNOWN.value == "UNKNOWN"
    assert MarketRegime.UNKNOWN.value == "UNKNOWN"
    assert PolicyAction.APPROVE.value == "APPROVE"
    assert AssetClass.EQUITY.value == "EQUITY"
    assert OptionType.CALL.value == "CALL"
    assert ExecutionStatus.RECONCILIATION_REQUIRED.value == "RECONCILIATION_REQUIRED"
    assert DataQualitySeverity.CRITICAL.value == "CRITICAL"


def test_empty_quality_summary_is_explicitly_healthy():
    quality = DataQualitySummary()
    assert quality.is_usable is True
    assert quality.issues == ()


def test_quality_summary_is_unusable_for_error_or_critical_issues():
    issue = DataQualityIssue(
        code="missing_bar",
        severity=DataQualitySeverity.ERROR,
        component="bars",
        message="A required bar is missing.",
    )
    quality = DataQualitySummary(issues=(issue,))

    assert quality.is_usable is False


def test_lineage_ref_accepts_optional_record_locator():
    lineage = LineageRef(source_id="bars", content_hash="abc123")

    assert lineage.record_locator is None
