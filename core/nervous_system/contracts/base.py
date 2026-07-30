from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("number must be finite")
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_utc)]
FiniteFloat = Annotated[float, AfterValidator(_finite)]
Probability = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
PositiveSchemaVersion = Annotated[int, Field(ge=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)

        payload = self.model_dump()
        payload.update(update)
        return type(self).model_validate(payload)

    @model_validator(mode="after")
    def reject_nonfinite_recursively(self) -> Self:
        def visit(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("contract contains a non-finite number")
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(self.model_dump())
        return self


def canonical_json(model: ContractModel) -> str:
    payload = model.model_dump(mode="json", exclude_none=False)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(model: ContractModel) -> str:
    return hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()
