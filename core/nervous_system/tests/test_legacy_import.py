from __future__ import annotations

from hashlib import sha256
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from core.nervous_system.contracts.states import PortfolioState
from core.nervous_system.data_registry.artifacts import register_artifact
from core.nervous_system.data_registry.legacy_adapters import (
    LegacyOperationalEvidence,
    OwnershipCandidateEvidence,
    adapt_legacy_record,
)
from core.nervous_system.data_registry.import_legacy import import_manifest
from core.nervous_system.data_registry.import_legacy import (
    compare_operational_discovery,
    load_manifest,
    main,
)
from core.nervous_system.data_registry.parsers import parse_csv, parse_json, parse_jsonl
from core.nervous_system.persistence.models import (
    ImportItem,
    ImportQuarantine,
    SourceArtifact,
    StateRecord,
)
from core.nervous_system.persistence.uow import UnitOfWork
from core.nervous_system.tests.fixtures.legacy_records import (
    account_snapshot_payload,
    managed_state_payload,
)


@pytest.fixture
def pg_uow_factory(session_factory):
    return lambda: UnitOfWork(session_factory)


def write_manifest(
    root: Path,
    source: Path,
    *,
    kind: str,
    adapter: str = "live_signal_audit",
) -> Path:
    manifest = root / "manifest.toml"
    manifest.write_text(
        "version = 1\n\n"
        "[[source]]\n"
        f'kind = "{kind}"\n'
        f'glob = "{source.name}"\n'
        f'adapter = "{adapter}"\n',
        encoding="utf-8",
    )
    return manifest


@pytest.mark.postgres
def test_import_is_idempotent_and_preserves_bad_row(pg_uow_factory, tmp_path):
    source = tmp_path / "audit.jsonl"
    source.write_text(
        json.dumps(
            {
                "event": "signal_decision",
                "module": "meta_ranker",
                "bar": "2026-07-29T18:00:00Z",
                "observed_at": "2026-07-29T18:00:01Z",
            },
            separators=(",", ":"),
        )
        + "\n"
        + json.dumps(
            {
                "event": "signal_decision",
                "module": "meta_ranker",
                "bar": "not-a-time",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path, source, kind="meta_signal_audit")

    first = import_manifest(manifest, pg_uow_factory, dry_run=False)
    second = import_manifest(manifest, pg_uow_factory, dry_run=False)

    assert first.imported == 1
    assert first.quarantined == 1
    assert second.imported == 0
    assert second.duplicates == 2
    assert source.read_bytes().startswith(b'{"event":"signal_decision"')

    with pg_uow_factory() as uow:
        source_row = uow.session.scalar(
            select(SourceArtifact).where(SourceArtifact.uri == source.as_posix())
        )
        assert source_row is not None
        quarantine_rows = uow.session.scalars(
            select(ImportQuarantine).where(ImportQuarantine.source_id == source_row.source_id)
        ).all()
        assert len(quarantine_rows) == 1
        assert quarantine_rows[0].raw_payload["bar"] == "not-a-time"
        assert quarantine_rows[0].raw_text == (
            '{"event":"signal_decision","module":"meta_ranker","bar":"not-a-time"}\n'
        )


def test_register_artifact_streams_bytes_and_does_not_use_mtime(tmp_path, monkeypatch):
    source = tmp_path / "large.jsonl"
    source_bytes = (b'{"event":"x"}\n' * 20000) + b"tail"
    source.write_bytes(source_bytes)

    def fail_stat(_self):
        raise AssertionError("artifact registration must not inspect filesystem mtime")

    monkeypatch.setattr(Path, "stat", fail_stat)
    artifact = register_artifact(source)

    assert artifact.byte_size == len(source_bytes)
    assert artifact.sha256 == sha256(source_bytes).hexdigest()
    assert source.read_bytes() == source_bytes


def test_jsonl_json_and_csv_locators_are_exact_and_streamable(tmp_path, monkeypatch):
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_bytes(b'{"row":1}\n{"row":2}\n')
    document = tmp_path / "record.json"
    document.write_text('{"row":3}', encoding="utf-8")
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("ticker,score\nSPY,1\nAMD,2\n", encoding="utf-8")

    assert list(parse_json(document))[0].record_locator == "document:1"
    monkeypatch.setattr(Path, "read_bytes", lambda _self: pytest.fail("JSONL parser read whole file"))
    assert [item.record_locator for item in parse_jsonl(jsonl)] == ["line:1", "line:2"]
    assert [item.record_locator for item in parse_csv(csv_path)] == ["row:1", "row:2"]


def test_signal_adapter_separates_bar_from_availability_and_keeps_score_raw():
    result = adapt_legacy_record(
        "meta_signal_audit",
        "live_signal_audit",
        {
            "event": "signal_decision",
            "module": "meta_ranker",
            "bar": "2026-07-29T18:00:00Z",
            "observed_at": "2026-07-29T18:00:01-04:00",
            "score": 0.97,
        },
    )

    assert result.quarantine_code is None
    assert isinstance(result.contract, LegacyOperationalEvidence)
    assert result.contract.as_of == datetime(2026, 7, 29, 18, tzinfo=timezone.utc)
    assert result.contract.available_at == datetime(2026, 7, 29, 22, 0, 1, tzinfo=timezone.utc)
    assert result.contract.payload["score"] == 0.97
    assert "probability" not in result.contract.model_dump()


def test_missing_or_naive_availability_is_quarantined_without_mtime_fallback():
    missing = adapt_legacy_record(
        "meta_signal_audit",
        "live_signal_audit",
        {"event": "signal_decision", "bar": "2026-07-29T18:00:00Z"},
    )
    naive = adapt_legacy_record(
        "meta_signal_audit",
        "live_signal_audit",
        {
            "event": "signal_decision",
            "bar": "2026-07-29T18:00:00Z",
            "observed_at": "2026-07-29T18:00:01",
        },
    )

    assert missing.contract is None
    assert missing.quarantine_code == "MISSING_AVAILABLE_AT"
    assert naive.contract is None
    assert naive.quarantine_code == "NAIVE_AVAILABLE_AT"


def test_account_and_managed_state_adapters_preserve_causal_and_ownership_rules():
    account = adapt_legacy_record(
        "account_snapshot", "broker_equity_snapshot", account_snapshot_payload()
    )
    managed = adapt_legacy_record(
        "strategy_live_state", "managed_state", managed_state_payload()
    )

    assert isinstance(account.contract, PortfolioState)
    assert account.contract.available_at == datetime(2026, 7, 29, 20, tzinfo=timezone.utc)
    assert account.contract.positions[0].strategy_id is None
    assert account.contract.positions[0].ownership_status == "UNASSIGNED"
    assert isinstance(managed.contract, OwnershipCandidateEvidence)
    assert managed.contract.ownership_status == "UNASSIGNED"
    assert managed.contract.payload["positions"][0]["strategy_id"] is None
    assert managed.contract.payload["positions"][0]["confirmed_ownership"] is False


def test_manifest_contains_the_audited_34_globs():
    manifest = Path("core/nervous_system/config/legacy_sources.toml")
    specs = load_manifest(manifest)
    assert len(specs) == 34
    assert specs[0].glob == "Data/inference/account_snapshots/*.jsonl"
    assert specs[-1].glob == "Data/readiness/*.json"


def test_discovery_comparison_exposes_unmatched_evidence_and_exact_exclusions(tmp_path):
    manifest = tmp_path / "legacy_sources.toml"
    canonical_manifest = Path("core/nervous_system/config/legacy_sources.toml").read_text(
        encoding="utf-8"
    )
    manifest.write_text(canonical_manifest, encoding="utf-8")
    matched = tmp_path / "Data/inference/meta_ranker/live_signal_audit.jsonl"
    unmatched = tmp_path / "Data/inference/meta_ranker/new_operational.json"
    excluded = tmp_path / "Data/inference/meta_ranker/options.meta.json"
    matched.parent.mkdir(parents=True)
    matched.write_text("{}\n", encoding="utf-8")
    unmatched.write_text("{}", encoding="utf-8")
    excluded.write_text("{}", encoding="utf-8")

    comparison = compare_operational_discovery(tmp_path, manifest)

    assert matched in comparison.matched
    assert unmatched in comparison.unmatched
    assert excluded in comparison.explicitly_excluded
    assert not comparison.complete


@pytest.mark.postgres
def test_dry_run_parses_and_validates_but_rolls_back_all_rows(pg_uow_factory, session_factory, tmp_path):
    source = tmp_path / "audit.jsonl"
    source.write_text(
        '{"event":"signal_decision","bar":"2026-07-29T18:00:00Z","observed_at":"2026-07-29T18:00:01Z"}\n',
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path, source, kind="meta_signal_audit")

    summary = import_manifest(manifest, pg_uow_factory, dry_run=True)

    assert summary.parsed == 1
    assert summary.imported == 1
    with session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(SourceArtifact).where(
                SourceArtifact.uri == source.as_posix()
            )
        ) == 0


@pytest.mark.postgres
def test_revised_source_hash_is_new_evidence_not_a_duplicate(pg_uow_factory, session_factory, tmp_path):
    source = tmp_path / "audit.jsonl"
    source.write_text(
        '{"event":"signal_decision","bar":"2026-07-29T18:00:00Z","observed_at":"2026-07-29T18:00:01Z","score":0.1}\n',
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path, source, kind="meta_signal_audit")
    first = import_manifest(manifest, pg_uow_factory, dry_run=False)
    source.write_text(
        '{"event":"signal_decision","bar":"2026-07-29T18:00:00Z","observed_at":"2026-07-29T18:00:01Z","score":0.2}\n',
        encoding="utf-8",
    )
    second = import_manifest(manifest, pg_uow_factory, dry_run=False)

    assert first.imported == second.imported == 1
    assert second.duplicates == 0
    assert first.source_hashes != second.source_hashes
    with session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(SourceArtifact).where(
                SourceArtifact.uri == source.as_posix()
            )
        ) == 2


@pytest.mark.postgres
def test_account_snapshot_is_persisted_as_typed_portfolio_state(
    pg_uow_factory, session_factory, tmp_path
):
    source = tmp_path / "account.jsonl"
    source.write_text(json.dumps(account_snapshot_payload()) + "\n", encoding="utf-8")
    manifest = write_manifest(
        tmp_path,
        source,
        kind="account_snapshot",
        adapter="broker_equity_snapshot",
    )

    summary = import_manifest(manifest, pg_uow_factory, dry_run=False)

    assert summary.imported == 1
    with session_factory() as session:
        row = session.scalar(
            select(StateRecord).where(StateRecord.entity_id == "qa-paper")
        )
        assert row is not None
        assert row.state_type == "PORTFOLIO"
        assert row.available_at.isoformat().startswith("2026-07-29T20:00:00")
        assert row.payload["positions"][0]["ownership_status"] == "UNASSIGNED"


@pytest.mark.postgres
def test_managed_state_persists_only_an_unassigned_candidate_projection(
    pg_uow_factory, session_factory, tmp_path
):
    source = tmp_path / "managed.json"
    source.write_text(json.dumps(managed_state_payload()), encoding="utf-8")
    manifest = write_manifest(
        tmp_path,
        source,
        kind="strategy_live_state",
        adapter="managed_state",
    )

    summary = import_manifest(manifest, pg_uow_factory, dry_run=False)

    assert summary.imported == 1
    with session_factory() as session:
        item = session.scalar(select(ImportItem).where(ImportItem.record_locator == "document:1"))
        assert item is not None
        normalized = item.warnings["normalized_contract"]
        assert normalized["ownership_status"] == "UNASSIGNED"
        assert normalized["payload"]["positions"][0]["strategy_id"] is None
        assert normalized["payload"]["positions"][0]["confirmed_ownership"] is False


@pytest.mark.postgres
def test_cli_dry_run_prints_one_json_summary(postgres_url, tmp_path, capsys):
    source = tmp_path / "audit.jsonl"
    source.write_text(
        '{"event":"status","observed_at":"2026-07-29T18:00:01Z"}\n',
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path, source, kind="spy_status")

    assert main(
        [
            "--manifest",
            str(manifest),
            "--database-url",
            postgres_url,
            "--dry-run",
        ]
    ) == 0
    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert set(payload) == {
        "discovered_artifacts",
        "parsed",
        "imported",
        "duplicates",
        "skipped",
        "quarantined",
        "source_hashes",
        "import_run_id",
    }
    assert payload["imported"] == 1
