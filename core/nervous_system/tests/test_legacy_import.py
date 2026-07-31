from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select

from core.nervous_system.contracts.states import PortfolioState
from core.nervous_system.data_registry.artifacts import register_artifact
from core.nervous_system.data_registry import import_legacy as import_legacy_module
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
    redact_database_url,
)
from core.nervous_system.data_registry import parsers as parser_module
from core.nervous_system.data_registry.parsers import (
    ParseIssue,
    iter_source_events,
    parse_csv,
    parse_json,
    parse_jsonl,
    raw_text_to_bytes,
)
from core.nervous_system.persistence.models import (
    ImportItem,
    ImportRun,
    ImportQuarantine,
    SourceArtifact,
    StateRecord,
)
from core.nervous_system.persistence.repositories.registry import (
    ImportItemRecord,
    ImportRunRecord,
    RegistryRepository,
    SourceArtifactRecord,
)
from core.nervous_system.persistence.repositories.state import StateRepository
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


def test_cli_database_url_accepts_arbitrary_postgresql_and_redacts_credentials():
    database_url = "postgresql+psycopg://alice:secret@db.example:5432/arbitrary_db"

    import_legacy_module._validate_cli_database_url(database_url)

    redacted = redact_database_url(database_url)
    assert "secret" not in redacted
    assert "arbitrary_db" in redacted


def test_cli_rejects_non_postgresql_database_url(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--manifest",
                "manifest.toml",
                "--database-url",
                "sqlite:///not-allowed.db",
                "--dry-run",
            ]
        )

    assert exc_info.value.code == 2
    assert "requires PostgreSQL" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("source_kind", "adapter", "payload", "expected_available_at"),
    (
        (
            "meta_closed_trade",
            "closed_trade",
            {
                "ts": "2026-07-31T01:53:37.174089+00:00",
                "module": "meta_ranker",
                "bar": "2026-07-29 18:00:00+00:00",
                "ticker": "OLD",
                "side": "sell",
            },
            datetime(2026, 7, 31, 1, 53, 37, 174089, tzinfo=timezone.utc),
        ),
        (
            "swing_session",
            "swing_session",
            {
                "type": "position_opened",
                "ts": "2026-05-28T12:05:28.955193+00:00",
                "payload": {"ticker": "AMD", "qty": 1},
            },
            datetime(2026, 5, 28, 12, 5, 28, 955193, tzinfo=timezone.utc),
        ),
    ),
)
def test_actual_closed_trade_and_swing_session_ts_is_event_availability(
    source_kind, adapter, payload, expected_available_at
):
    result = adapt_legacy_record(source_kind, adapter, payload)

    assert result.quarantine_code is None
    assert isinstance(result.contract, LegacyOperationalEvidence)
    assert result.contract.available_at == expected_available_at


@pytest.mark.parametrize(
    "payload",
    (
        {"type": "position_opened"},
        {"type": "position_opened", "ts": "2026-05-28T12:05:28.955193"},
    ),
)
def test_swing_session_missing_or_naive_ts_is_quarantined(payload):
    result = adapt_legacy_record("swing_session", "swing_session", payload)

    assert result.contract is None
    assert result.quarantine_code in {"MISSING_AVAILABLE_AT", "NAIVE_AVAILABLE_AT"}


def test_ts_is_not_a_generic_availability_fallback():
    result = adapt_legacy_record(
        "spy_log",
        "raw_operational_event",
        {"event": "log_line", "ts": "2026-07-31T01:53:37Z"},
    )

    assert result.contract is None
    assert result.quarantine_code == "MISSING_AVAILABLE_AT"


def test_future_as_of_is_quarantined_instead_of_persisted():
    result = adapt_legacy_record(
        "meta_signal_audit",
        "live_signal_audit",
        {
            "event": "signal_decision",
            "bar": "2026-07-31T02:00:00Z",
            "observed_at": "2026-07-31T01:00:00Z",
        },
    )

    assert result.contract is None
    assert result.quarantine_code == "AS_OF_AFTER_AVAILABLE_AT"


def test_typed_state_rejects_as_of_after_available_at():
    result = adapt_legacy_record(
        "account_snapshot", "broker_equity_snapshot", account_snapshot_payload()
    )
    assert isinstance(result.contract, PortfolioState)
    payload = result.contract.model_dump(mode="json")
    payload["as_of"] = "2026-07-29T21:00:00Z"

    with pytest.raises(ValueError, match="as_of cannot follow available_at"):
        PortfolioState.model_validate(payload)


@pytest.mark.parametrize(
    ("position_update", "expected_code"),
    (
        ({"asset_class": "mystery", "quantity": 1, "symbol": "AMD"}, "UNKNOWN_ASSET_CLASS"),
        ({"asset_class": "EQUITY", "symbol": "AMD"}, "MISSING_POSITION_QUANTITY"),
        ({"asset_class": "EQUITY", "quantity": 1, "symbol": "AMD"}, None),
    ),
)
def test_portfolio_position_conversion_is_fail_closed(position_update, expected_code):
    payload = {
        "account_alias": "qa-paper",
        "captured_at_utc": "2026-07-29T20:00:00Z",
        "equity": 100000.0,
        "cash": 90000.0,
        "buying_power": 90000.0,
        "positions": [position_update],
    }
    result = adapt_legacy_record("account_snapshot", "broker_equity_snapshot", payload)

    if expected_code is None:
        assert isinstance(result.contract, PortfolioState)
    else:
        assert result.contract is None
        assert result.quarantine_code == expected_code


def test_portfolio_invalid_position_is_not_silently_dropped():
    payload = account_snapshot_payload()
    payload["positions"] = [account_snapshot_payload()["positions"][0], "not-a-position"]

    result = adapt_legacy_record("account_snapshot", "broker_equity_snapshot", payload)

    assert result.contract is None
    assert result.quarantine_code == "INVALID_PORTFOLIO_POSITION"


def test_portfolio_default_validity_uses_named_versioned_import_rule():
    result = adapt_legacy_record(
        "account_snapshot", "broker_equity_snapshot", account_snapshot_payload()
    )

    assert isinstance(result.contract, PortfolioState)
    assert result.contract.config_version == "legacy-portfolio-validity@1d@1"
    assert any("legacy-portfolio-validity@1d@1" in warning for warning in result.warnings)


def test_discovery_strict_mode_fails_on_unmatched_and_only_exact_option_metadata_is_excluded(
    tmp_path,
):
    manifest = tmp_path / "legacy_sources.toml"
    manifest.write_text(
        Path("core/nervous_system/config/legacy_sources.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    matched = tmp_path / "Data/inference/meta_ranker/live_signal_audit.jsonl"
    unmatched = tmp_path / "Data/inference/meta_ranker/new_operational.meta.json"
    excluded = tmp_path / "Data/inference/meta_ranker/options.meta.json"
    matched.parent.mkdir(parents=True)
    matched.write_text("{}\n", encoding="utf-8")
    unmatched.write_text("{}", encoding="utf-8")
    excluded.write_text("{}", encoding="utf-8")

    comparison = compare_operational_discovery(tmp_path, manifest)

    assert unmatched in comparison.unmatched
    assert excluded in comparison.explicitly_excluded
    with pytest.raises(ValueError, match="unmatched"):
        compare_operational_discovery(tmp_path, manifest, require_complete=True)


def test_json_document_size_guard_quarantines_before_json_loading(tmp_path, monkeypatch):
    document = tmp_path / "oversized.json"
    document.write_bytes(b'{"payload":"0123456789"}')
    monkeypatch.setattr(parser_module, "MAX_JSON_DOCUMENT_BYTES", 8, raising=False)

    event = next(iter_source_events(document))

    assert isinstance(event, ParseIssue)
    assert event.error_code == "JSON_DOCUMENT_TOO_LARGE"


def test_invalid_utf8_quarantine_text_round_trips_raw_bytes(tmp_path):
    source = tmp_path / "invalid.jsonl"
    raw_bytes = b'{"payload":\xff}\n'
    source.write_bytes(raw_bytes)

    event = next(iter_source_events(source))

    assert isinstance(event, ParseIssue)
    assert event.raw_text is not None
    assert raw_text_to_bytes(event.raw_text) == raw_bytes


def test_dry_run_validates_all_rows_without_opening_a_write_uow(tmp_path):
    source = tmp_path / "audit.jsonl"
    rows = [
        json.dumps(
            {
                "event": "signal_decision",
                "bar": "2026-07-29T18:00:00Z",
                "observed_at": "2026-07-29T18:00:01Z",
                "row": index,
            },
            separators=(",", ":"),
        )
        for index in range(2050)
    ]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = write_manifest(tmp_path, source, kind="meta_signal_audit")

    def fail_if_write_uow_is_requested():
        pytest.fail("dry-run must not open a write transaction")

    summary = import_manifest(manifest, fail_if_write_uow_is_requested, dry_run=True)

    assert summary.parsed == 2050
    assert summary.imported == 2050
    assert summary.quarantined == 0


@pytest.mark.postgres
def test_write_import_does_not_query_identity_once_per_row(
    pg_uow_factory, tmp_path, monkeypatch
):
    source = tmp_path / "audit.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "signal_decision",
                    "bar": "2026-07-29T18:00:00Z",
                    "observed_at": "2026-07-29T18:00:01Z",
                    "row": index,
                },
                separators=(",", ":"),
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path, source, kind="meta_signal_audit")

    def fail_if_row_identity_is_queried(*args, **kwargs):
        pytest.fail("write importer must use atomic/bulk identity insertion")

    monkeypatch.setattr(RegistryRepository, "get_import_item", fail_if_row_identity_is_queried)

    summary = import_manifest(manifest, pg_uow_factory, dry_run=False)

    assert summary.imported == 3


@pytest.mark.postgres
def test_write_import_batches_typed_state_persistence(
    pg_uow_factory, tmp_path, monkeypatch
):
    source = tmp_path / "accounts.jsonl"
    rows = []
    for index in range(3):
        payload = account_snapshot_payload()
        payload["captured_at_utc"] = f"2026-07-29T20:0{index}:00Z"
        rows.append(json.dumps(payload, separators=(",", ":")))
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = write_manifest(
        tmp_path,
        source,
        kind="account_snapshot",
        adapter="broker_equity_snapshot",
    )

    def fail_if_row_state_lookup(*args, **kwargs):
        pytest.fail("typed state persistence must use bounded bulk operations")

    monkeypatch.setattr(
        StateRepository, "get_state_by_content_hash", fail_if_row_state_lookup
    )
    monkeypatch.setattr(StateRepository, "save_state", fail_if_row_state_lookup)

    summary = import_manifest(manifest, pg_uow_factory, dry_run=False)

    assert summary.imported == 3


@pytest.mark.postgres
def test_source_mutation_after_registration_aborts_import(
    pg_uow_factory, session_factory, tmp_path, monkeypatch
):
    source = tmp_path / "audit.jsonl"
    source.write_text(
        '{"event":"signal_decision","bar":"2026-07-29T18:00:00Z",'
        '"observed_at":"2026-07-29T18:00:01Z"}\n',
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path, source, kind="meta_signal_audit")
    original_iter_source_events = import_legacy_module.iter_source_events

    def mutate_after_first_event(path, *args, **kwargs):
        events = original_iter_source_events(path, *args, **kwargs)
        first = next(events)
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        yield first
        yield from events

    monkeypatch.setattr(import_legacy_module, "iter_source_events", mutate_after_first_event)

    with pytest.raises(ValueError, match="changed during import"):
        import_manifest(manifest, pg_uow_factory, dry_run=False)

    with session_factory() as session:
        failed_run = session.scalar(
            select(ImportRun)
            .where(ImportRun.status == "FAILED")
            .order_by(ImportRun.started_at.desc())
            .limit(1)
        )
        assert failed_run is not None
        assert failed_run.counts["imported"] == 0


@pytest.mark.postgres
def test_concurrent_import_item_identity_insert_converges_atomically(
    session_factory, postgres_engine
):
    source_id = uuid4()
    run_id = uuid4()
    with session_factory() as session:
        repo = RegistryRepository(session)
        source_record = repo.save_source_artifact(
            SourceArtifactRecord(
                source_id=source_id,
                uri=f"task8-concurrent-{source_id}.jsonl",
                sha256="a" * 64,
                byte_size=1,
                source_kind="test",
                discovered_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
            )
        )
        source_id = source_record.source_id
        repo.save_import_run(
            ImportRunRecord(
                import_run_id=run_id,
                importer_version="task8-test@1",
                started_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
                finished_at=None,
                status="RUNNING",
                counts={},
            )
        )
        session.commit()

    barrier = Barrier(2)

    def synchronize_import_item_insert(_conn, _cursor, statement, _parameters, _context, _many):
        if "insert into nervous_system.import_items" in statement.lower():
            barrier.wait(timeout=10)

    event.listen(postgres_engine, "before_cursor_execute", synchronize_import_item_insert)
    item_payload = dict(
        import_run_id=run_id,
        source_id=source_id,
        importer_version="task8-test@1",
        record_locator="line:1",
        normalized_hash="b" * 64,
        target_type="SIGNAL_AUDIT",
        target_id="target-1",
        status="IMPORTED",
        warnings={},
    )

    def insert_once(_index):
        with session_factory() as session:
            result = RegistryRepository(session).insert_import_item_if_absent(
                ImportItemRecord(import_item_id=uuid4(), **item_payload)
            )
            session.commit()
            return result

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(insert_once, (1, 2)))
    finally:
        event.remove(postgres_engine, "before_cursor_execute", synchronize_import_item_insert)

    records = tuple(result[0] for result in results)
    inserted = tuple(result[1] for result in results)
    assert sorted(inserted) == [False, True]
    assert records[0].import_item_id == records[1].import_item_id

    with session_factory() as session:
        revised, was_inserted = RegistryRepository(session).insert_import_item_if_absent(
            ImportItemRecord(
                import_run_id=run_id,
                source_id=source_id,
                importer_version="task8-test@1",
                record_locator="line:1",
                normalized_hash="c" * 64,
                target_type="SIGNAL_AUDIT",
                target_id="target-2",
                status="IMPORTED",
                warnings={},
            )
        )
        session.commit()
        assert was_inserted is True
        assert revised.normalized_hash == "c" * 64
        assert session.scalar(
            select(func.count()).select_from(ImportItem).where(ImportItem.source_id == source_id)
        ) == 2


@pytest.mark.postgres
def test_partial_write_failure_marks_import_run_failed_after_committed_batch(
    pg_uow_factory, session_factory, tmp_path, monkeypatch
):
    source = tmp_path / "audit.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "signal_decision",
                    "bar": "2026-07-29T18:00:00Z",
                    "observed_at": "2026-07-29T18:00:01Z",
                    "row": index,
                },
                separators=(",", ":"),
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path, source, kind="meta_signal_audit")
    monkeypatch.setattr(import_legacy_module, "IMPORT_BATCH_SIZE", 1, raising=False)
    original_persist_contract = import_legacy_module._persist_contract
    calls = 0

    def fail_on_second_contract(uow, contract, *, identity):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("intentional task8 batch failure")
        return original_persist_contract(uow, contract, identity=identity)

    monkeypatch.setattr(import_legacy_module, "_persist_contract", fail_on_second_contract)

    with pytest.raises(RuntimeError, match="intentional task8 batch failure"):
        import_manifest(manifest, pg_uow_factory, dry_run=False)

    with session_factory() as session:
        failed_run = session.scalar(
            select(ImportRun)
            .where(ImportRun.status == "FAILED")
            .order_by(ImportRun.started_at.desc())
            .limit(1)
        )
        assert failed_run is not None
        assert failed_run.counts["imported"] == 1
