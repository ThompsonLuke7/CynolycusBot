"""Idempotent historical operational-evidence importer and CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.decisions import DecisionRecord
from core.nervous_system.contracts.states import StateEnvelope
from core.nervous_system.data_registry.artifacts import SourceArtifact, register_artifact
from core.nervous_system.data_registry.legacy_adapters import (
    LegacyAdapterResult,
    adapt_legacy_record,
)
from core.nervous_system.data_registry.lineage import (
    IMPORTER_VERSION,
    ImportIdentity,
    adapter_version,
    target_id_for_identity,
)
from core.nervous_system.data_registry.parsers import ParseIssue, RawImportItem, iter_source_events
from core.nervous_system.persistence.repositories.registry import (
    ImportItemRecord,
    ImportQuarantineRecord,
    ImportRunRecord,
    LineageEdgeRecord,
    SourceArtifactRecord,
)
from core.nervous_system.persistence.uow import UnitOfWork


DISPOSABLE_DATABASE_NAME = "cynolycus_nervous_system_test"


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    glob: str
    adapter: str


@dataclass(frozen=True)
class DiscoveredSource:
    path: Path
    spec: SourceSpec


@dataclass(frozen=True)
class DiscoveryComparison:
    matched: tuple[Path, ...]
    explicitly_excluded: tuple[Path, ...]
    unmatched: tuple[Path, ...]

    @property
    def complete(self) -> bool:
        return not self.unmatched


@dataclass(frozen=True)
class ImportSummary:
    discovered_artifacts: int
    parsed: int
    imported: int
    duplicates: int
    skipped: int
    quarantined: int
    source_hashes: tuple[str, ...]
    import_run_id: UUID

    def counts(self) -> dict[str, int]:
        return {
            "discovered_artifacts": self.discovered_artifacts,
            "parsed": self.parsed,
            "imported": self.imported,
            "duplicates": self.duplicates,
            "skipped": self.skipped,
            "quarantined": self.quarantined,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.counts(),
            "source_hashes": list(self.source_hashes),
            "import_run_id": str(self.import_run_id),
        }


OPERATIONAL_ROOTS = (
    "Data/inference/account_snapshots",
    "Data/inference/dealer_ranker",
    "Data/inference/intraday_structure",
    "Data/inference/live_runs",
    "Data/inference/meta_ranker",
    "Data/inference/momentum_expansion",
    "Data/inference/multi_ticker_swing",
    "Data/inference/multi_ticker_swing_htf",
    "Data/inference/shadow_two_sleeve",
    "Data/readiness",
    "Data/runtime",
    "UI/swing_audit/swing_session_*.jsonl",
    "UI/swing_audit/paper/swing_session_*.jsonl",
    "signals/meta_context/meta_ranker/live_state.json",
    "strategies/momentum_expansion/live/momentum_live_state.json",
    "strategies/momentum_expansion/live/alerts.jsonl",
    "strategies/multi_ticker_swing_htf/live/htf_live_state.json",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_manifest(path: Path) -> tuple[SourceSpec, ...]:
    manifest_path = Path(path)
    try:
        with manifest_path.open("rb") as manifest_file:
            payload = tomllib.load(manifest_file)
    except OSError as exc:
        raise FileNotFoundError(f"unable to read legacy source manifest: {manifest_path}") from exc
    if payload.get("version") != 1:
        raise ValueError("legacy source manifest version must be 1")
    entries = payload.get("source")
    if not isinstance(entries, list) or not entries:
        raise ValueError("legacy source manifest must contain at least one [[source]]")
    specs: list[SourceSpec] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each legacy source manifest entry must be a table")
        try:
            spec = SourceSpec(
                kind=str(entry["kind"]),
                glob=str(entry["glob"]),
                adapter=str(entry["adapter"]),
            )
        except KeyError as exc:
            raise ValueError(f"legacy source entry is missing {exc.args[0]}") from exc
        if not spec.kind or not spec.glob or not spec.adapter:
            raise ValueError("legacy source kind, glob, and adapter must be non-empty")
        specs.append(spec)
    return tuple(specs)


def _manifest_root(manifest_path: Path, specs: Iterable[SourceSpec]) -> Path:
    """Find the nearest ancestor that owns at least one manifest pattern."""

    path = Path(manifest_path)
    candidates = (path.parent, *path.parent.parents)
    for candidate in candidates:
        for spec in specs:
            try:
                if any(candidate.glob(spec.glob)):
                    return candidate
            except (NotImplementedError, ValueError):
                continue
            first_component = spec.glob.split("/", 1)[0]
            if "*" not in first_component and (candidate / first_component).exists():
                return candidate
    # The repository manifest has a stable root even when ignored artifacts
    # are absent from an isolated worktree.
    if len(path.parents) >= 4:
        return path.parents[3]
    return path.parent


def discover_manifest_sources(
    manifest_path: Path,
    *,
    root: Path | None = None,
    source_kind: str | None = None,
) -> tuple[DiscoveredSource, ...]:
    specs = load_manifest(manifest_path)
    discovery_root = Path(root) if root is not None else _manifest_root(Path(manifest_path), specs)
    discovered: list[DiscoveredSource] = []
    seen: set[tuple[Path, str, str]] = set()
    for spec in specs:
        if source_kind is not None and spec.kind != source_kind:
            continue
        pattern = Path(spec.glob)
        matches = (
            pattern.parent.glob(pattern.name)
            if pattern.is_absolute()
            else discovery_root.glob(spec.glob)
        )
        for candidate in sorted(matches):
            if not candidate.is_file():
                continue
            key = (candidate, spec.kind, spec.adapter)
            if key in seen:
                continue
            seen.add(key)
            discovered.append(DiscoveredSource(candidate, spec))
    return tuple(discovered)


def _is_explicitly_excluded(relative_path: Path) -> bool:
    # Option-chain metadata is a JSON artifact but is intentionally only
    # registered by reference in the analytical data layer, not row-imported.
    return relative_path.name.endswith(".meta.json")


def _operational_candidates(root: Path) -> set[Path]:
    candidates: set[Path] = set()
    for root_spec in OPERATIONAL_ROOTS:
        pattern = Path(root_spec)
        matches = root.glob(root_spec) if "*" in root_spec else (root / root_spec,)
        for match in matches:
            if match.is_file():
                if match.suffix.lower() in {".json", ".jsonl"} or (
                    match.suffix.lower() == ".csv"
                    and match.name.startswith("unique_broker_orders_")
                ):
                    candidates.add(match)
                continue
            if not match.is_dir():
                continue
            for candidate in match.rglob("*"):
                if not candidate.is_file():
                    continue
                if candidate.suffix.lower() in {".json", ".jsonl"} or (
                    candidate.suffix.lower() == ".csv"
                    and candidate.name.startswith("unique_broker_orders_")
                ):
                    candidates.add(candidate)
    return candidates


def compare_operational_discovery(
    root: Path,
    manifest_path: Path | None = None,
) -> DiscoveryComparison:
    """Compare audited operational candidates with manifest matches.

    The comparison is read-only and intentionally takes an explicit root so it
    can be run against the source checkout without making that checkout the
    import target.
    """

    source_root = Path(root)
    manifest = manifest_path or source_root / "core/nervous_system/config/legacy_sources.toml"
    matched = {
        item.path
        for item in discover_manifest_sources(manifest, root=source_root)
        if item.path.is_file()
    }
    candidates = _operational_candidates(source_root)
    excluded = {
        candidate
        for candidate in candidates
        if _is_explicitly_excluded(candidate.relative_to(source_root))
    }
    return DiscoveryComparison(
        matched=tuple(sorted(candidates & matched)),
        explicitly_excluded=tuple(sorted(excluded - matched)),
        unmatched=tuple(sorted(candidates - matched - excluded)),
    )


def _raw_identity_payload(raw_item: RawImportItem) -> Mapping[str, Any]:
    return raw_item.raw_payload


def _parse_issue_identity_payload(issue: ParseIssue) -> Mapping[str, Any]:
    return {
        "error_code": issue.error_code,
        "raw_text": issue.raw_text,
    }


def _warnings(
    *,
    raw_payload: Mapping[str, Any],
    artifact: SourceArtifact,
    spec: SourceSpec,
    adapter_result: LegacyAdapterResult | None,
    retain_raw_payload: bool = False,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "adapter": spec.adapter,
        "adapter_version": adapter_version(spec.adapter),
        "source_sha256": artifact.sha256,
    }
    if retain_raw_payload:
        values["raw_payload"] = dict(raw_payload)
    if adapter_result is not None:
        values["warnings"] = list(adapter_result.warnings)
        if adapter_result.contract is not None:
            values["normalized_contract"] = adapter_result.contract.model_dump(mode="json")
    return values


def _persist_contract(
    uow: UnitOfWork,
    contract: Any,
    *,
    identity: str,
) -> str:
    if isinstance(contract, StateEnvelope):
        state_hash = content_hash(contract, exclude={"state_id"})
        existing = uow.states.get_state_by_content_hash(state_hash)
        if existing is None:
            uow.states.save_state(contract)
            return str(contract.state_id)
        return str(existing.state_id)
    if isinstance(contract, DecisionRecord):
        # This branch is deliberately narrow: a future adapter may return a
        # fully validated current DecisionRecord.  Partial legacy decisions
        # remain raw operational evidence instead of entering the decision
        # authority with fabricated links.
        uow.decisions.save_decision_record(contract)
        return str(contract.decision_record_id)
    return target_id_for_identity(identity)


def _save_quarantined_item(
    uow: UnitOfWork,
    *,
    run_id: UUID,
    source_id: UUID,
    importer_version: str,
    identity: ImportIdentity,
    target_type: str,
    raw_payload: Mapping[str, Any] | None,
    raw_text: str | None,
    error_code: str,
    error_message: str,
    warning_values: Mapping[str, Any],
) -> None:
    uow.registry.save_import_item(
        ImportItemRecord(
            import_run_id=run_id,
            source_id=source_id,
            importer_version=importer_version,
            record_locator=identity.record_locator,
            normalized_hash=identity.normalized_hash,
            target_type=target_type,
            target_id=None,
            status="QUARANTINED",
            warnings=warning_values,
        )
    )
    uow.registry.save_import_quarantine(
        ImportQuarantineRecord(
            import_run_id=run_id,
            source_id=source_id,
            record_locator=identity.record_locator,
            raw_payload=None if raw_payload is None else dict(raw_payload),
            raw_text=raw_text,
            error_code=error_code,
            error_message=error_message,
            created_at=_now(),
        )
    )


def _save_imported_item(
    uow: UnitOfWork,
    *,
    run_id: UUID,
    source_id: UUID,
    importer_version: str,
    identity: ImportIdentity,
    target_type: str,
    target_id: str,
    warning_values: Mapping[str, Any],
) -> None:
    uow.registry.save_import_item(
        ImportItemRecord(
            import_run_id=run_id,
            source_id=source_id,
            importer_version=importer_version,
            record_locator=identity.record_locator,
            normalized_hash=identity.normalized_hash,
            target_type=target_type,
            target_id=target_id,
            status="IMPORTED",
            warnings=warning_values,
        )
    )
    uow.registry.save_lineage_edge(
        LineageEdgeRecord(
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relationship="IMPORTED_AS",
            created_at=_now(),
        )
    )


def import_manifest(
    path: Path,
    uow_factory: Callable[[], UnitOfWork],
    dry_run: bool = False,
    *,
    limit: int | None = None,
    source_kind: str | None = None,
) -> ImportSummary:
    """Import every manifest record in one caller-owned transaction."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")
    manifest_path = Path(path)
    discovered = discover_manifest_sources(manifest_path, source_kind=source_kind)
    run_id = uuid4()
    counts = {
        "discovered_artifacts": len({item.path for item in discovered}),
        "parsed": 0,
        "imported": 0,
        "duplicates": 0,
        "skipped": 0,
        "quarantined": 0,
    }
    registered: list[tuple[DiscoveredSource, SourceArtifact, SourceArtifactRecord]] = []
    source_hashes: list[str] = []
    with uow_factory() as uow:
        started_at = _now()
        uow.registry.save_import_run(
            ImportRunRecord(
                import_run_id=run_id,
                importer_version=IMPORTER_VERSION,
                started_at=started_at,
                finished_at=None,
                status="RUNNING",
                counts=counts,
            )
        )
        for item in discovered:
            artifact = register_artifact(item.path)
            source_hashes.append(artifact.sha256)
            source_record = uow.registry.get_source_artifact(artifact.uri, artifact.sha256)
            if source_record is None:
                source_record = uow.registry.save_source_artifact(
                    SourceArtifactRecord(
                        uri=artifact.uri,
                        sha256=artifact.sha256,
                        byte_size=artifact.byte_size,
                        source_kind=item.spec.kind,
                        discovered_at=started_at,
                        metadata={
                            "manifest_glob": item.spec.glob,
                            "adapter": item.spec.adapter,
                            "format": item.path.suffix.lower().lstrip("."),
                        },
                    )
                )
            registered.append((item, artifact, source_record))

        records_seen = 0
        stop = False
        for item, artifact, source_record in registered:
            if stop:
                break
            for event in iter_source_events(item.path):
                if limit is not None and records_seen >= limit:
                    stop = True
                    break
                records_seen += 1
                if isinstance(event, ParseIssue):
                    if event.skippable:
                        counts["skipped"] += 1
                        continue
                    identity = ImportIdentity.build(
                        source_sha256=artifact.sha256,
                        record_locator=event.record_locator,
                        adapter=item.spec.adapter,
                        normalized_payload=_parse_issue_identity_payload(event),
                    )
                    if (
                        uow.registry.get_import_item(
                            source_id=source_record.source_id,
                            record_locator=identity.record_locator,
                            importer_version=identity.importer_version,
                            normalized_hash=identity.normalized_hash,
                        )
                        is not None
                    ):
                        counts["duplicates"] += 1
                        continue
                    _save_quarantined_item(
                        uow,
                        run_id=run_id,
                        source_id=source_record.source_id,
                        importer_version=identity.importer_version,
                        identity=identity,
                        target_type=item.spec.kind.upper(),
                        raw_payload=None,
                        raw_text=event.raw_text,
                        error_code=event.error_code,
                        error_message=event.error_message,
                        warning_values=_warnings(
                            raw_payload=_parse_issue_identity_payload(event),
                            artifact=artifact,
                            spec=item.spec,
                            adapter_result=None,
                            retain_raw_payload=True,
                        ),
                    )
                    counts["quarantined"] += 1
                    continue

                counts["parsed"] += 1
                identity = ImportIdentity.build(
                    source_sha256=artifact.sha256,
                    record_locator=event.record_locator,
                    adapter=item.spec.adapter,
                    normalized_payload=_raw_identity_payload(event),
                )
                if (
                    uow.registry.get_import_item(
                        source_id=source_record.source_id,
                        record_locator=identity.record_locator,
                        importer_version=identity.importer_version,
                        normalized_hash=identity.normalized_hash,
                    )
                    is not None
                ):
                    counts["duplicates"] += 1
                    continue
                result = adapt_legacy_record(item.spec.kind, item.spec.adapter, event.raw_payload)
                warning_values = _warnings(
                    raw_payload=event.raw_payload,
                    artifact=artifact,
                    spec=item.spec,
                    adapter_result=result,
                    retain_raw_payload=result.quarantine_code is not None,
                )
                if result.quarantine_code is not None or result.contract is None:
                    _save_quarantined_item(
                        uow,
                        run_id=run_id,
                        source_id=source_record.source_id,
                        importer_version=identity.importer_version,
                        identity=identity,
                        target_type=result.target_type,
                        raw_payload=event.raw_payload,
                        raw_text=event.raw_text,
                        error_code=result.quarantine_code or "NO_NORMALIZED_CONTRACT",
                        error_message=result.quarantine_message or "legacy adapter returned no contract",
                        warning_values=warning_values,
                    )
                    counts["quarantined"] += 1
                    continue
                target_id = _persist_contract(uow, result.contract, identity=identity.normalized_hash)
                _save_imported_item(
                    uow,
                    run_id=run_id,
                    source_id=source_record.source_id,
                    importer_version=identity.importer_version,
                    identity=identity,
                    target_type=result.target_type,
                    target_id=target_id,
                    warning_values=warning_values,
                )
                counts["imported"] += 1

        summary = ImportSummary(
            discovered_artifacts=counts["discovered_artifacts"],
            parsed=counts["parsed"],
            imported=counts["imported"],
            duplicates=counts["duplicates"],
            skipped=counts["skipped"],
            quarantined=counts["quarantined"],
            source_hashes=tuple(sorted(set(source_hashes))),
            import_run_id=run_id,
        )
        uow.registry.finish_import_run(
            run_id,
            finished_at=_now(),
            status="DRY_RUN" if dry_run else "COMPLETED",
            counts=summary.counts(),
        )
        if dry_run:
            uow.rollback()
        else:
            uow.commit()
    return summary


def _uow_factory_for_database(database_url: str) -> tuple[Callable[[], UnitOfWork], Any]:
    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return lambda: UnitOfWork(sessions), engine


def _validate_cli_database_url(database_url: str) -> None:
    parsed = make_url(database_url)
    if parsed.get_backend_name() != "postgresql":
        raise ValueError("historical operational import requires PostgreSQL")
    if parsed.database != DISPOSABLE_DATABASE_NAME:
        raise ValueError(
            "historical operational import is restricted to the disposable "
            f"database {DISPOSABLE_DATABASE_NAME!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source-kind")
    args = parser.parse_args(argv)
    try:
        _validate_cli_database_url(args.database_url)
        uow_factory, engine = _uow_factory_for_database(args.database_url)
        try:
            summary = import_manifest(
                args.manifest,
                uow_factory,
                dry_run=args.dry_run,
                limit=args.limit,
                source_kind=args.source_kind,
            )
        finally:
            engine.dispose()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DISPOSABLE_DATABASE_NAME",
    "DiscoveryComparison",
    "ImportSummary",
    "OPERATIONAL_ROOTS",
    "SourceSpec",
    "compare_operational_discovery",
    "discover_manifest_sources",
    "import_manifest",
    "load_manifest",
    "main",
]
