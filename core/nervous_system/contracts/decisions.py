from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, Field, model_validator
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
    status: Literal["COMPLETE", "FAILED"] = "COMPLETE"
    snapshot_id: UUID | None = None
    intent_id: UUID | None = None
    policy_decision_id: UUID | None = None
    order_request_ids: tuple[UUID, ...] = ()
    source_manifest_hash: Sha256Hex | None = None
    snapshot_hash: Sha256Hex | None = None
    intent_hash: Sha256Hex | None = None
    policy_hash: Sha256Hex | None = None
    raw_strategy_output: HashedDecisionArtifact | None = None
    exposure_report: HashedDecisionArtifact | None = None
    instrument_candidates: HashedDecisionArtifact | None = None
    instrument_selection: HashedDecisionArtifact | None = None
    order_hashes: tuple[Sha256Hex, ...] = ()
    config_hash: Sha256Hex | None = None
    model_versions: ImmutableStringMap = Field(default_factory=dict)
    feature_versions: ImmutableStringMap = Field(default_factory=dict)
    schema_version: PositiveSchemaVersion = 1
    failure_stage: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    @model_validator(mode="after")
    def validate_decision_links(self) -> DecisionRecord:
        if len(self.order_request_ids) != len(self.order_hashes):
            raise ValueError("order request IDs and hashes must be one-to-one")
        if len(set(self.order_request_ids)) != len(self.order_request_ids):
            raise ValueError("order request IDs must be unique")
        links = (
            ("snapshot", self.snapshot_id, self.snapshot_hash),
            ("intent", self.intent_id, self.intent_hash),
            ("policy", self.policy_decision_id, self.policy_hash),
        )
        for link_name, link_id, link_hash in links:
            if (link_id is None) != (link_hash is None):
                raise ValueError(f"{link_name} link and hash must be supplied together")

        if self.status == "FAILED":
            if not self.failure_stage:
                raise ValueError("FAILED decision records require failure_stage")
            if not self.failure_code:
                raise ValueError("FAILED decision records require failure_code")
            if not self.failure_message:
                raise ValueError("FAILED decision records require failure_message")
            return self

        if any((self.failure_stage, self.failure_code, self.failure_message)):
            raise ValueError("COMPLETE decision records cannot contain failure fields")
        if any(link_id is None for _, link_id, _ in links):
            raise ValueError(
                "COMPLETE decision records require snapshot, intent, and policy links"
            )
        if any(link_hash is None for _, _, link_hash in links):
            raise ValueError(
                "COMPLETE decision records require snapshot, intent, and policy hashes"
            )
        artifacts = (
            self.raw_strategy_output,
            self.exposure_report,
            self.instrument_candidates,
            self.instrument_selection,
        )
        if any(artifact is None for artifact in artifacts):
            raise ValueError("COMPLETE decision records require all decision artifacts")
        seen_not_run = False
        for artifact in artifacts:
            assert artifact is not None
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
        if decision_record.status != "COMPLETE":
            raise ValueError("outcomes require complete decision records")
        if self.evaluated_at < decision_record.decision_time:
            raise ValueError("outcome evaluated_at must not precede decision_time")
        return self
