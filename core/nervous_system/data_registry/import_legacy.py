"""Idempotent historical operational-evidence importer and CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.decisions import DecisionRecord
from core.nervous_system.contracts.states import StateEnvelope
from core.nervous_system.data_registry.artifacts import (
    SourceArtifact,
    register_artifact,
    snapshot_artifact,
)
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


IMPORT_BATCH_SIZE = 1000


class SourceMutationError(ValueError):
    """Raised when the path no longer represents the registered bytes."""


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

    def require_complete(self) -> "DiscoveryComparison":
        if not self.complete:
            preview = ", ".join(path.as_posix() for path in self.unmatched[:5])
            suffix = "" if len(self.unmatched) <= 5 else " ..."
            raise ValueError(
                f"unmatched operational evidence ({len(self.unmatched)}): {preview}{suffix}"
            )
        return self


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


@dataclass(frozen=True)
class _PendingImport:
    item: ImportItemRecord
    quarantine: ImportQuarantineRecord | None = None
    lineage: LineageEdgeRecord | None = None
    contract: Any | None = None


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


_EXPLICIT_OPTION_CHAIN_METADATA_PATHS = frozenset(
    {
        "Data/inference/meta_ranker/options.meta.json",
        "Data/inference/live_runs/option_chain.meta.json",
    }
)


def _is_explicitly_excluded(relative_path: Path) -> bool:
    """Return true only for named option-chain metadata artifacts."""

    return relative_path.as_posix() in _EXPLICIT_OPTION_CHAIN_METADATA_PATHS


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
    *,
    require_complete: bool = False,
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
    comparison = DiscoveryComparison(
        matched=tuple(sorted(candidates & matched)),
        explicitly_excluded=tuple(sorted(excluded - matched)),
        unmatched=tuple(sorted(candidates - matched - excluded)),
    )
    if require_complete:
        comparison.require_complete()
    return comparison


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


def _identity_key(item: ImportItemRecord) -> tuple[UUID, str, str, str]:
    return (
        item.source_id,
        item.record_locator,
        item.importer_version,
        item.normalized_hash,
    )


def _flush_batch(
    uow: UnitOfWork,
    pending: list[_PendingImport],
    counts: dict[str, int],
) -> None:
    if not pending:
        return
    typed_states = [
        entry.contract
        for entry in pending
        if isinstance(entry.contract, StateEnvelope)
    ]
    state_ids = uow.states.insert_states_if_absent(typed_states)
    if state_ids:
        for index, entry in enumerate(pending):
            if not isinstance(entry.contract, StateEnvelope):
                continue
            state_hash = content_hash(entry.contract, exclude={"state_id"})
            target_id = str(state_ids[state_hash])
            pending[index] = replace(
                entry,
                item=replace(entry.item, target_id=target_id),
                lineage=replace(entry.lineage, target_id=target_id)
                if entry.lineage is not None
                else None,
            )
    inserted = uow.registry.insert_import_items_if_absent(
        [entry.item for entry in pending]
    )
    counts["duplicates"] += len(pending) - len(inserted)
    quarantines = [
        entry.quarantine
        for entry in pending
        if entry.quarantine is not None and _identity_key(entry.item) in inserted
    ]
    edges = [
        entry.lineage
        for entry in pending
        if entry.lineage is not None and _identity_key(entry.item) in inserted
    ]
    uow.registry.save_import_quarantines(
        [quarantine for quarantine in quarantines if quarantine is not None]
    )
    uow.registry.save_lineage_edges([edge for edge in edges if edge is not None])
    counts["quarantined"] += len(quarantines)
    counts["imported"] += len(edges)
    pending.clear()
    # Bulk Core statements do not populate the ORM identity map.  Discard any
    # small number of state objects created by a typed adapter before the next
    # batch so memory remains bounded for large JSONL sources.
    uow.session.expunge_all()


def _verify_source_unchanged(path: Path, expected: SourceArtifact) -> None:
    try:
        current = register_artifact(path)
    except FileNotFoundError as exc:
        raise SourceMutationError(
            f"source artifact changed during import: {path} is no longer readable"
        ) from exc
    if current.sha256 != expected.sha256 or current.byte_size != expected.byte_size:
        raise SourceMutationError(
            f"source artifact changed during import: {path} no longer matches "
            f"registered SHA-256 {expected.sha256}"
        )


def _pending_from_event(
    event: RawImportItem | ParseIssue,
    *,
    item: DiscoveredSource,
    artifact: SourceArtifact,
    source_record: SourceArtifactRecord,
    run_id: UUID,
    uow: UnitOfWork,
    counts: dict[str, int],
) -> _PendingImport:
    if isinstance(event, ParseIssue):
        if event.skippable:
            counts["skipped"] += 1
            raise StopIteration
        raw_payload = _parse_issue_identity_payload(event)
        identity = ImportIdentity.build(
            source_sha256=artifact.sha256,
            record_locator=event.record_locator,
            adapter=item.spec.adapter,
            normalized_payload=raw_payload,
        )
        return _PendingImport(
            item=ImportItemRecord(
                import_run_id=run_id,
                source_id=source_record.source_id,
                importer_version=identity.importer_version,
                record_locator=identity.record_locator,
                normalized_hash=identity.normalized_hash,
                target_type=item.spec.kind.upper(),
                target_id=None,
                status="QUARANTINED",
                warnings=_warnings(
                    raw_payload=raw_payload,
                    artifact=artifact,
                    spec=item.spec,
                    adapter_result=None,
                    retain_raw_payload=True,
                ),
            ),
            quarantine=ImportQuarantineRecord(
                run_id,
                source_record.source_id,
                event.record_locator,
                None,
                event.raw_text,
                event.error_code,
                event.error_message,
                _now(),
            ),
        )

    counts["parsed"] += 1
    identity = ImportIdentity.build(
        source_sha256=artifact.sha256,
        record_locator=event.record_locator,
        adapter=item.spec.adapter,
        normalized_payload=_raw_identity_payload(event),
    )
    result = adapt_legacy_record(item.spec.kind, item.spec.adapter, event.raw_payload)
    warning_values = _warnings(
        raw_payload=event.raw_payload,
        artifact=artifact,
        spec=item.spec,
        adapter_result=result,
        retain_raw_payload=result.quarantine_code is not None,
    )
    if result.quarantine_code is not None or result.contract is None:
        return _PendingImport(
            item=ImportItemRecord(
                import_run_id=run_id,
                source_id=source_record.source_id,
                importer_version=identity.importer_version,
                record_locator=identity.record_locator,
                normalized_hash=identity.normalized_hash,
                target_type=result.target_type,
                target_id=None,
                status="QUARANTINED",
                warnings=warning_values,
            ),
            quarantine=ImportQuarantineRecord(
                run_id,
                source_record.source_id,
                event.record_locator,
                event.raw_payload,
                event.raw_text,
                result.quarantine_code or "NO_NORMALIZED_CONTRACT",
                result.quarantine_message or "legacy adapter returned no contract",
                _now(),
            ),
        )

    target_id = None
    if not isinstance(result.contract, StateEnvelope):
        target_id = _persist_contract(uow, result.contract, identity=identity.normalized_hash)
    imported_item = ImportItemRecord(
        import_run_id=run_id,
        source_id=source_record.source_id,
        importer_version=identity.importer_version,
        record_locator=identity.record_locator,
        normalized_hash=identity.normalized_hash,
        target_type=result.target_type,
        target_id=target_id,
        status="IMPORTED",
        warnings=warning_values,
    )
    return _PendingImport(
        item=imported_item,
        lineage=LineageEdgeRecord(
            source_id=source_record.source_id,
            target_type=result.target_type,
            target_id=target_id,
            relationship="IMPORTED_AS",
            created_at=_now(),
        ),
        contract=result.contract,
    )


def _validate_event_for_dry_run(
    event: RawImportItem | ParseIssue,
    *,
    item: DiscoveredSource,
    counts: dict[str, int],
) -> None:
    if isinstance(event, ParseIssue):
        if event.skippable:
            counts["skipped"] += 1
        else:
            counts["quarantined"] += 1
        return
    counts["parsed"] += 1
    result = adapt_legacy_record(item.spec.kind, item.spec.adapter, event.raw_payload)
    if result.quarantine_code is not None or result.contract is None:
        counts["quarantined"] += 1
    else:
        counts["imported"] += 1


def import_manifest(
    path: Path,
    uow_factory: Callable[[], UnitOfWork],
    dry_run: bool = False,
    *,
    limit: int | None = None,
    source_kind: str | None = None,
) -> ImportSummary:
    """Parse all records and import write-mode batches with resumable commits."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")
    manifest_path = Path(path)
    specs = load_manifest(manifest_path)
    discovery_root = _manifest_root(manifest_path, specs)
    if source_kind is None:
        compare_operational_discovery(
            discovery_root,
            manifest_path,
            require_complete=True,
        )
    discovered = discover_manifest_sources(
        manifest_path,
        root=discovery_root,
        source_kind=source_kind,
    )
    run_id = uuid4()
    counts = {
        "discovered_artifacts": len({item.path for item in discovered}),
        "parsed": 0,
        "imported": 0,
        "duplicates": 0,
        "skipped": 0,
        "quarantined": 0,
    }
    registered = [(item, register_artifact(item.path)) for item in discovered]
    source_hashes = [artifact.sha256 for _, artifact in registered]

    if dry_run:
        records_seen = 0
        stop = False
        for item, expected_artifact in registered:
            if stop:
                break
            with snapshot_artifact(item.path) as (artifact, source_file):
                if artifact.sha256 != expected_artifact.sha256 or artifact.byte_size != expected_artifact.byte_size:
                    raise SourceMutationError(
                        f"source artifact changed during import: {item.path} changed before parsing"
                    )
                for event in iter_source_events(item.path, source_file=source_file):
                    if limit is not None and records_seen >= limit:
                        stop = True
                        break
                    records_seen += 1
                    _validate_event_for_dry_run(event, item=item, counts=counts)
                _verify_source_unchanged(item.path, expected_artifact)
        return ImportSummary(
            discovered_artifacts=counts["discovered_artifacts"],
            parsed=counts["parsed"],
            imported=counts["imported"],
            duplicates=counts["duplicates"],
            skipped=counts["skipped"],
            quarantined=counts["quarantined"],
            source_hashes=tuple(sorted(set(source_hashes))),
            import_run_id=run_id,
        )

    started_at = _now()
    run_saved = False
    with uow_factory() as uow:
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
        uow.commit()
        run_saved = True
        committed_counts = dict(counts)
        records_seen = 0
        stop = False
        try:
            for item, expected_artifact in registered:
                if stop:
                    break
                with snapshot_artifact(item.path) as (artifact, source_file):
                    if (
                        artifact.sha256 != expected_artifact.sha256
                        or artifact.byte_size != expected_artifact.byte_size
                    ):
                        raise SourceMutationError(
                            f"source artifact changed during import: {item.path} changed before parsing"
                        )
                    source_record = uow.registry.insert_source_artifact_if_absent(
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
                    pending: list[_PendingImport] = []
                    for event in iter_source_events(item.path, source_file=source_file):
                        if limit is not None and records_seen >= limit:
                            stop = True
                            break
                        records_seen += 1
                        try:
                            pending.append(
                                _pending_from_event(
                                    event,
                                    item=item,
                                    artifact=artifact,
                                    source_record=source_record,
                                    run_id=run_id,
                                    uow=uow,
                                    counts=counts,
                                )
                            )
                        except StopIteration:
                            continue
                        if len(pending) >= IMPORT_BATCH_SIZE:
                            _flush_batch(uow, pending, counts)
                            uow.registry.update_import_run_progress(run_id, counts)
                            uow.commit()
                            committed_counts = dict(counts)
                    _flush_batch(uow, pending, counts)
                    _verify_source_unchanged(item.path, expected_artifact)
                uow.registry.update_import_run_progress(run_id, counts)
                uow.commit()
                committed_counts = dict(counts)

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
                status="COMPLETED",
                counts=summary.counts(),
            )
            uow.commit()
            return summary
        except Exception as exc:
            if run_saved:
                uow.rollback()
                failure_counts = {**committed_counts, "error": str(exc)}
                uow.registry.finish_import_run(
                    run_id,
                    finished_at=_now(),
                    status="FAILED",
                    counts=failure_counts,
                )
                uow.commit()
            raise


def _uow_factory_for_database(database_url: str) -> tuple[Callable[[], UnitOfWork], Any]:
    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return lambda: UnitOfWork(sessions), engine


def redact_database_url(database_url: str) -> str:
    """Render a database URL without exposing its password."""

    try:
        return make_url(database_url).render_as_string(hide_password=True)
    except (TypeError, ValueError):
        return "<redacted database URL>"


def _validate_cli_database_url(database_url: str) -> None:
    parsed = make_url(database_url)
    if parsed.get_backend_name() != "postgresql":
        raise ValueError("historical operational import requires PostgreSQL")


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
    except (OSError, ValueError, SQLAlchemyError) as exc:
        message = str(exc)
        redacted_url = redact_database_url(args.database_url)
        message = message.replace(args.database_url, redacted_url)
        parser.error(message)
    print(json.dumps(summary.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DiscoveryComparison",
    "IMPORT_BATCH_SIZE",
    "ImportSummary",
    "OPERATIONAL_ROOTS",
    "SourceMutationError",
    "SourceSpec",
    "compare_operational_discovery",
    "discover_manifest_sources",
    "import_manifest",
    "load_manifest",
    "main",
    "redact_database_url",
]
