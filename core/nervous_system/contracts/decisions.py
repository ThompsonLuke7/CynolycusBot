from __future__ import annotations

import hashlib
import json
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, model_validator
from pydantic_core import to_jsonable_python

from .base import (
    ContractModel,
    FiniteFloat,
    ImmutableFloatMap,
    PositiveSchemaVersion,
    Sha256Hex,
    UtcDatetime,
    _canonicalize,
    _freeze_mapping,
    _reject_nonfinite,
)


ImmutableObjectMap = Annotated[dict[str, object], AfterValidator(_freeze_mapping)]
ImmutableStringMap = Annotated[dict[str, str], AfterValidator(_freeze_mapping)]


def _payload_hash(payload: dict[str, object]) -> str:
    _reject_nonfinite(payload)
    encoded = json.dumps(
        _canonicalize(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class HashedDecisionArtifact(ContractModel):
    artifact_type: str
    schema_version: PositiveSchemaVersion
    content_hash: Sha256Hex
    payload: ImmutableObjectMap

    @model_validator(mode="after")
    def validate_payload_hash(self) -> HashedDecisionArtifact:
        if self.content_hash != _payload_hash(self.payload):
            raise ValueError("content_hash does not match artifact payload")
        if self.artifact_type == "NOT_RUN" or self.payload.get("status") == "NOT_RUN":
            if self.payload.get("status") != "NOT_RUN":
                raise ValueError("NOT_RUN artifacts require status=NOT_RUN")
            if not self.payload.get("blocking_stage") or not self.payload.get("reason"):
                raise ValueError("NOT_RUN artifacts require blocking_stage and reason")
        return self

    @classmethod
    def from_payload(
        cls,
        artifact_type: str,
        schema_version: int,
        payload: dict[str, object],
    ) -> HashedDecisionArtifact:
        return cls(
            artifact_type=artifact_type,
            schema_version=schema_version,
            content_hash=_payload_hash(payload),
            payload=payload,
        )

    @classmethod
    def not_run(cls, blocking_stage: str, reason: str) -> HashedDecisionArtifact:
        return cls.from_payload(
            "NOT_RUN",
            1,
            {"status": "NOT_RUN", "blocking_stage": blocking_stage, "reason": reason},
        )


class DecisionRecord(ContractModel):
    decision_record_id: UUID
    decision_time: UtcDatetime
    snapshot_id: UUID
    intent_id: UUID
    policy_decision_id: UUID
    order_request_ids: tuple[UUID, ...]
    source_manifest_hash: Sha256Hex
    snapshot_hash: Sha256Hex
    intent_hash: Sha256Hex
    policy_hash: Sha256Hex
    raw_strategy_output: HashedDecisionArtifact
    exposure_report: HashedDecisionArtifact
    instrument_candidates: HashedDecisionArtifact
    instrument_selection: HashedDecisionArtifact
    order_hashes: tuple[Sha256Hex, ...]
    config_hash: Sha256Hex
    model_versions: ImmutableStringMap
    feature_versions: ImmutableStringMap
    schema_version: PositiveSchemaVersion

    @model_validator(mode="after")
    def validate_decision_links(self) -> DecisionRecord:
        if len(self.order_request_ids) != len(self.order_hashes):
            raise ValueError("order request IDs and hashes must be one-to-one")
        if len(set(self.order_request_ids)) != len(self.order_request_ids):
            raise ValueError("order request IDs must be unique")
        artifacts = (
            self.raw_strategy_output,
            self.exposure_report,
            self.instrument_candidates,
            self.instrument_selection,
        )
        seen_not_run = False
        for artifact in artifacts:
            if not artifact.payload:
                raise ValueError("decision stages must contain explicit artifact payloads")
            is_not_run = artifact.payload.get("status") == "NOT_RUN"
            if seen_not_run and not is_not_run:
                raise ValueError("downstream decision stages must remain NOT_RUN")
            seen_not_run = seen_not_run or is_not_run
        return self


class DecisionOutcome(ContractModel):
    outcome_id: UUID
    decision_record_id: UUID
    evaluated_at: UtcDatetime
    horizon: str
    underlying_return: FiniteFloat | None
    instrument_return: FiniteFloat | None
    source_fitness_report_id: UUID | None
    metrics: ImmutableFloatMap

    @classmethod
    def for_decision(
        cls,
        decision_record: DecisionRecord,
        *,
        outcome_id: UUID,
        evaluated_at: UtcDatetime,
        horizon: str,
        underlying_return: float | None,
        instrument_return: float | None,
        source_fitness_report_id: UUID | None,
        metrics: dict[str, float],
    ) -> DecisionOutcome:
        candidate = cls(
            outcome_id=outcome_id,
            decision_record_id=decision_record.decision_record_id,
            evaluated_at=evaluated_at,
            horizon=horizon,
            underlying_return=underlying_return,
            instrument_return=instrument_return,
            source_fitness_report_id=source_fitness_report_id,
            metrics=metrics,
        )
        if candidate.evaluated_at < decision_record.decision_time:
            raise ValueError("outcome evaluated_at must not precede decision_time")
        return candidate

    def validate_against(self, decision_record: DecisionRecord) -> DecisionOutcome:
        if self.decision_record_id != decision_record.decision_record_id:
            raise ValueError("outcome does not reference the supplied decision record")
        if self.evaluated_at < decision_record.decision_time:
            raise ValueError("outcome evaluated_at must not precede decision_time")
        return self
