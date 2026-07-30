from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import Field, ValidationError, model_validator

from core.nervous_system.contracts import (
    ContractModel as ExportedContractModel,
    FiniteFloat as ExportedFiniteFloat,
    PositiveSchemaVersion as ExportedPositiveSchemaVersion,
    Probability as ExportedProbability,
    UtcDatetime as ExportedUtcDatetime,
    canonical_json as exported_canonical_json,
    content_hash as exported_content_hash,
)

from core.nervous_system.contracts.base import (
    ContractModel,
    FiniteFloat,
    PositiveSchemaVersion,
    Probability,
    UtcDatetime,
    canonical_json,
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
    marker: int = 0


class AliasedContainer(ContractModel):
    value: float = Field(alias="valueAlias")
    nested: Nested = Field(alias="nestedAlias")


class CanonicalExample(ContractModel):
    tags: set[str]
    frozen_tags: frozenset[str]
    ordered: tuple[str, ...]
    observed_at: datetime
    amount: Decimal
    identifier: UUID
    direction: Direction


class SetNumbers(ContractModel):
    values: set[float]


class FrozenSetNumbers(ContractModel):
    values: frozenset[float]


class NumericBoundaries(ContractModel):
    finite: FiniteFloat
    probability: Probability
    schema_version: PositiveSchemaVersion


class OrderedRange(ContractModel):
    start: int
    end: int

    @model_validator(mode="after")
    def require_ordered_bounds(self):
        if self.end < self.start:
            raise ValueError("end must not precede start")
        return self


def test_contract_normalizes_aware_time_to_utc_and_hashes_stably():
    model = Example(
        observed_at=datetime.fromisoformat("2026-07-30T10:00:00-04:00"),
        value=1.25,
    )
    assert model.observed_at == datetime(2026, 7, 30, 14, tzinfo=timezone.utc)
    assert content_hash(model) == content_hash(
        Example.model_validate_json(model.model_dump_json())
    )


def test_canonical_json_sorts_unordered_collections_and_preserves_ordered_values():
    model = CanonicalExample(
        tags={"alpha", "mike", "zulu"},
        frozen_tags=frozenset({"beta", "eta", "zeta"}),
        ordered=("second", "first"),
        observed_at=datetime(2026, 7, 30, 14, tzinfo=timezone.utc),
        amount=Decimal("1.20"),
        identifier=UUID("12345678-1234-5678-1234-567812345678"),
        direction=Direction.LONG,
    )

    assert canonical_json(model) == (
        '{"amount":"1.20","direction":"LONG",'
        '"frozen_tags":["beta","eta","zeta"],'
        '"identifier":"12345678-1234-5678-1234-567812345678",'
        '"observed_at":"2026-07-30T14:00:00Z",'
        '"ordered":["second","first"],'
        '"tags":["alpha","mike","zulu"]}'
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


@pytest.mark.parametrize("model_type", [SetNumbers, FrozenSetNumbers])
def test_contract_rejects_nonfinite_set_members(model_type):
    with pytest.raises(ValidationError):
        model_type(values={1.0, float("nan")})


def test_contract_model_copy_revalidates_updates():
    model = Example(
        observed_at=datetime(2026, 7, 30, 14, tzinfo=timezone.utc),
        value=1.0,
    )

    with pytest.raises(ValidationError):
        model.model_copy(update={"value": float("nan")})


def test_contract_model_copy_accepts_field_name_updates_on_aliased_models():
    model = AliasedContainer(valueAlias=1.0, nestedAlias={"values": [1]})

    updated = model.model_copy(update={"value": 2.0})

    assert updated.value == 2.0


def test_contract_model_copy_normalizes_nested_dict_updates():
    model = Container(nested={"values": [1]})

    updated = model.model_copy(update={"nested": {"values": [2]}})

    assert isinstance(updated.nested, Nested)
    assert updated.nested.values == [2]


def test_contract_model_copy_normalizes_updated_utc_datetime():
    model = Example(
        observed_at=datetime(2026, 7, 30, 14, tzinfo=timezone.utc),
        value=1.0,
    )

    updated = model.model_copy(
        update={
            "observed_at": datetime.fromisoformat("2026-07-30T10:00:00-04:00")
        }
    )

    assert updated.observed_at == datetime(2026, 7, 30, 14, tzinfo=timezone.utc)
    assert updated.observed_at.tzinfo is timezone.utc


def test_contract_model_copy_does_not_bypass_cross_field_validators():
    model = OrderedRange(start=1, end=2)

    with pytest.raises(ValidationError):
        model.model_copy(update={"end": 0})


def test_contract_model_copy_updates_preserve_shallow_and_deep_identity():
    model = Container(nested={"values": [1]})

    shallow = model.model_copy(update={"marker": 1})
    deep = model.model_copy(update={"marker": 1}, deep=True)

    assert shallow.nested is model.nested
    assert deep.nested is not model.nested


def test_contract_model_copy_rejects_unknown_updates():
    model = Example(
        observed_at=datetime(2026, 7, 30, 14, tzinfo=timezone.utc),
        value=1.0,
    )

    with pytest.raises(ValidationError):
        model.model_copy(update={"surprise": True})


def test_contract_model_copy_preserves_shallow_and_deep_copy_behavior():
    model = Container(nested={"values": [1]})

    shallow = model.model_copy()
    deep = model.model_copy(deep=True)

    assert shallow == model
    assert shallow.nested is model.nested
    assert deep == model
    assert deep.nested is not model.nested
    assert deep.nested.values is not model.nested.values


def test_numeric_contract_annotations_enforce_their_boundaries():
    model = NumericBoundaries(finite=1.5, probability=0.0, schema_version=1)

    assert model.finite == 1.5
    assert model.probability == 0.0
    assert model.schema_version == 1

    with pytest.raises(ValidationError):
        NumericBoundaries(finite=float("inf"), probability=0.5, schema_version=1)
    with pytest.raises(ValidationError):
        NumericBoundaries(finite=1.5, probability=-0.01, schema_version=1)
    with pytest.raises(ValidationError):
        NumericBoundaries(finite=1.5, probability=1.01, schema_version=1)
    with pytest.raises(ValidationError):
        NumericBoundaries(finite=1.5, probability=0.5, schema_version=0)


def test_contract_package_exports_foundation_types_and_helpers():
    assert ExportedContractModel is ContractModel
    assert ExportedFiniteFloat is FiniteFloat
    assert ExportedProbability is Probability
    assert ExportedPositiveSchemaVersion is PositiveSchemaVersion
    assert ExportedUtcDatetime is UtcDatetime
    assert exported_canonical_json is canonical_json
    assert exported_content_hash is content_hash


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


def test_canonical_json_and_content_hash_support_backward_compatible_exclusions():
    first = CanonicalExample(
        tags={"alpha"},
        frozen_tags=frozenset({"beta"}),
        ordered=("first",),
        observed_at=datetime(2026, 7, 30, 14, tzinfo=timezone.utc),
        amount=Decimal("1.20"),
        identifier=UUID("12345678-1234-5678-1234-567812345678"),
        direction=Direction.LONG,
    )
    second = first.model_copy(
        update={"identifier": UUID("87654321-4321-8765-4321-876543218765")}
    )

    assert content_hash(first) != content_hash(second)
    assert content_hash(first, exclude={"identifier"}) == content_hash(
        second, exclude={"identifier"}
    )
    assert canonical_json(first, exclude={"identifier"}) == canonical_json(
        second, exclude={"identifier"}
    )
