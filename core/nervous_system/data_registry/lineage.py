"""Canonical identities and lineage values for historical import rows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic_core import to_jsonable_python


IMPORTER_VERSION = "legacy-operational-evidence@1"


def _canonical_json(value: Any) -> str:
    normalized = to_jsonable_python(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash normalized payload content independently of source identity."""

    return hashlib.sha256(_canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def adapter_version(adapter: str) -> str:
    return f"{adapter}@1"


def import_item_identity(
    *,
    source_sha256: str,
    record_locator: str,
    adapter: str,
    importer_version: str = IMPORTER_VERSION,
    normalized_payload: Mapping[str, Any],
) -> str:
    """Return the immutable row identity used as ``import_items.normalized_hash``.

    The source digest is intentionally part of the identity.  Replacing a file
    at the same path therefore creates new evidence instead of making a
    revised row look like a duplicate of the old evidence.
    """

    material = {
        "adapter_version": adapter_version(adapter),
        "importer_version": importer_version,
        "normalized_payload": dict(normalized_payload),
        "record_locator": record_locator,
        "source_sha256": source_sha256,
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def target_id_for_identity(identity: str) -> str:
    if len(identity) != 64:
        raise ValueError("identity must be a SHA-256 hex string")
    return f"legacy:{identity}"


@dataclass(frozen=True)
class ImportIdentity:
    source_sha256: str
    record_locator: str
    adapter_version: str
    importer_version: str
    normalized_hash: str

    @classmethod
    def build(
        cls,
        *,
        source_sha256: str,
        record_locator: str,
        adapter: str,
        normalized_payload: Mapping[str, Any],
        importer_version: str = IMPORTER_VERSION,
    ) -> "ImportIdentity":
        return cls(
            source_sha256=source_sha256,
            record_locator=record_locator,
            adapter_version=adapter_version(adapter),
            importer_version=importer_version,
            normalized_hash=import_item_identity(
                source_sha256=source_sha256,
                record_locator=record_locator,
                adapter=adapter,
                importer_version=importer_version,
                normalized_payload=normalized_payload,
            ),
        )


__all__ = [
    "IMPORTER_VERSION",
    "ImportIdentity",
    "adapter_version",
    "canonical_payload_hash",
    "import_item_identity",
    "target_id_for_identity",
]
