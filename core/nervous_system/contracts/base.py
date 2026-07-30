from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Set
from datetime import datetime, timezone
from decimal import Decimal
import re
from typing import AbstractSet, Annotated, Any, Generic, Self, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("number must be finite")
    return value


def _finite_decimal(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ValueError("decimal must be finite")
    return value


def _sha256_hex(value: str) -> str:
    if len(value) != 64 or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("value must be a 64-character SHA-256 hex string")
    return value.lower()


UtcDatetime = Annotated[datetime, AfterValidator(_utc)]
FiniteFloat = Annotated[float, AfterValidator(_finite)]
Probability = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
PositiveSchemaVersion = Annotated[int, Field(ge=1)]
FiniteDecimal = Annotated[Decimal, AfterValidator(_finite_decimal)]
NonNegativeDecimal = Annotated[FiniteDecimal, Field(ge=Decimal("0"))]
PositiveDecimal = Annotated[FiniteDecimal, Field(gt=Decimal("0"))]
Sha256Hex = Annotated[str, AfterValidator(_sha256_hex)]


_FrozenKey = TypeVar("_FrozenKey")
_FrozenValue = TypeVar("_FrozenValue")


class FrozenDict(dict[_FrozenKey, _FrozenValue], Generic[_FrozenKey, _FrozenValue]):
    """A JSON-compatible dict that rejects all mutation operations."""

    _ERROR = "FrozenDict is immutable"

    def _reject(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(self._ERROR)

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _reject

    def __ior__(self, other: Mapping[_FrozenKey, _FrozenValue]):
        self._reject(other)
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict[_FrozenKey, _FrozenValue]:
        return FrozenDict(self)


def _freeze_mapping(value: Mapping[_FrozenKey, _FrozenValue]) -> FrozenDict[_FrozenKey, _FrozenValue]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return FrozenDict({key: freeze(nested) for key, nested in item.items()})
        if isinstance(item, list):
            return tuple(freeze(nested) for nested in item)
        if isinstance(item, tuple):
            return tuple(freeze(nested) for nested in item)
        return item

    return FrozenDict({key: freeze(item) for key, item in value.items()})


ImmutableFloatMap = Annotated[
    dict[str, FiniteFloat], AfterValidator(_freeze_mapping)
]
ImmutableProbabilityMap = Annotated[
    dict[str, Probability], AfterValidator(_freeze_mapping)
]


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
        validated = type(self).model_validate(payload, by_alias=False, by_name=True)
        normalized_update = {
            field_name: getattr(validated, field_name) for field_name in update
        }
        return super().model_copy(update=normalized_update, deep=deep)

    @model_validator(mode="after")
    def reject_nonfinite_recursively(self) -> Self:
        def visit(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("contract contains a non-finite number")
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple, Set)):
                for item in value:
                    visit(item)

        visit(self.model_dump())
        return self


def canonical_json(
    model: ContractModel,
    exclude: AbstractSet[str] | None = None,
) -> str:
    payload = _canonicalize(
        model.model_dump(mode="python", exclude=exclude, exclude_none=False)
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return to_jsonable_python(
            {key: _canonicalize(item) for key, item in value.items()}
        )
    if isinstance(value, Set):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=_canonical_sort_key)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return to_jsonable_python(value)


def content_hash(
    model: ContractModel,
    exclude: AbstractSet[str] | None = None,
) -> str:
    return hashlib.sha256(canonical_json(model, exclude=exclude).encode("utf-8")).hexdigest()
