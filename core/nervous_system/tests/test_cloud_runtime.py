"""Cloud/QA-paper runtime boundaries (Task 26).

Two jobs. First, make the settings describe a real deployment: Cloud SQL, a
GCS journal, Secret Manager credentials, an explicit paper account identity.
Second, and more important, make it impossible to be casually wrong — QA-paper
cannot borrow a development shortcut, production-live parses for read-only
inspection but can never execute, and a secret never appears anywhere a human
or a log will see it.

The DSN accepts two shapes on purpose. Cloud Run reaches Cloud SQL over a Unix
socket; the VM that runs the schedulers and the persistent Alpaca WebSocket
cannot lift into Cloud Run and will reach the same instance over private-IP
TCP. Supporting one shape now and the other later is a rewrite; supporting both
now is a config decision that costs nothing.
"""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from core.nervous_system.config.runtime import NervousSystemSettings


SOCKET_DSN = (
    "postgresql+psycopg://cynolycus:s3cret@/cynolycus"
    "?host=/cloudsql/cynolycusbot-dev:us-east5:cynolycus-qa"
)
TCP_DSN = "postgresql+psycopg://cynolycus:s3cret@10.20.30.40:5432/cynolycus"


def _env(**updates: str) -> dict[str, str]:
    payload = {
        "CYNOLYCUS_ENVIRONMENT": "QA_PAPER",
        "CYNOLYCUS_NERVOUS_SYSTEM_MODE": "SHADOW",
        "CYNOLYCUS_DATABASE_URL": SOCKET_DSN,
        "CYNOLYCUS_OPERATIONAL_ROOT": "/var/cynolycus",
        "CYNOLYCUS_EXECUTION_JOURNAL": "gcs",
        "CYNOLYCUS_EXECUTION_JOURNAL_BUCKET": "cynolycusbot-execution-journal",
        "CYNOLYCUS_ACCOUNT_ALIAS": "paper",
        "CYNOLYCUS_GCP_PROJECT": "cynolycusbot-dev",
        "CYNOLYCUS_CLOUD_SQL_INSTANCE": "cynolycusbot-dev:us-east5:cynolycus-qa",
        "CYNOLYCUS_ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "CYNOLYCUS_ALPACA_ACCOUNT_ID": "PA123456",
        "CYNOLYCUS_SECRET_BINDING": "projects/cynolycusbot-dev/secrets/alpaca-paper",
    }
    payload.update(updates)
    return payload


# ---------------------------------------------------------------------------
# Both DSN shapes
# ---------------------------------------------------------------------------


def test_a_cloud_sql_unix_socket_dsn_is_accepted() -> None:
    """How Cloud Run reaches Cloud SQL."""

    assert NervousSystemSettings.from_env(_env()).database_url == SOCKET_DSN


def test_a_private_ip_tcp_dsn_is_accepted() -> None:
    """How the scheduler VM reaches the same instance. It cannot lift into
    Cloud Run — persistent WebSocket, in-process schedulers — so it will be a
    primary writer over TCP, and rejecting that shape would be a rewrite later.
    """

    settings = NervousSystemSettings.from_env(_env(CYNOLYCUS_DATABASE_URL=TCP_DSN))

    assert settings.database_url == TCP_DSN


def test_a_dsn_without_a_database_name_is_refused() -> None:
    with pytest.raises(ValidationError):
        NervousSystemSettings.from_env(
            _env(CYNOLYCUS_DATABASE_URL="postgresql+psycopg://u:p@10.0.0.1:5432/")
        )


def test_a_non_postgres_driver_is_refused() -> None:
    with pytest.raises(ValidationError):
        NervousSystemSettings.from_env(
            _env(CYNOLYCUS_DATABASE_URL="mysql://u:p@10.0.0.1/cynolycus")
        )


# ---------------------------------------------------------------------------
# QA-paper cannot borrow a development shortcut
# ---------------------------------------------------------------------------


def test_qa_paper_requires_the_durable_gcs_journal() -> None:
    """A local journal on ephemeral Cloud Run storage is no journal at all."""

    with pytest.raises(ValidationError, match="journal"):
        NervousSystemSettings.from_env(_env(CYNOLYCUS_EXECUTION_JOURNAL="local"))


def test_qa_paper_requires_a_cloud_sql_instance() -> None:
    with pytest.raises(ValidationError, match="[Cc]loud SQL"):
        NervousSystemSettings.from_env(_env(CYNOLYCUS_CLOUD_SQL_INSTANCE=""))


def test_qa_paper_requires_an_explicit_paper_base_url() -> None:
    """Identity is not inferred from the alias. A paper alias pointed at the
    live endpoint would look correct in every log line.
    """

    with pytest.raises(ValidationError, match="paper"):
        NervousSystemSettings.from_env(
            _env(CYNOLYCUS_ALPACA_BASE_URL="https://api.alpaca.markets")
        )


def test_qa_paper_requires_an_account_identity() -> None:
    with pytest.raises(ValidationError, match="account"):
        NervousSystemSettings.from_env(_env(CYNOLYCUS_ALPACA_ACCOUNT_ID=""))


def test_qa_paper_requires_a_secret_binding() -> None:
    with pytest.raises(ValidationError, match="[Ss]ecret"):
        NervousSystemSettings.from_env(_env(CYNOLYCUS_SECRET_BINDING=""))


def test_qa_paper_defaults_to_not_submitting() -> None:
    """Deployment is not authorisation. Submission is a separate, explicit act.
    """

    assert NervousSystemSettings.from_env(_env()).submit_enabled is False


def test_submission_must_be_enabled_explicitly() -> None:
    settings = NervousSystemSettings.from_env(
        _env(CYNOLYCUS_SUBMIT_ENABLED="true")
    )

    assert settings.submit_enabled is True


@pytest.mark.parametrize("value", ["yes", "1", "on", "TRUE ", "maybe", ""])
def test_only_an_unambiguous_true_enables_submission(value: str) -> None:
    """Anything fuzzy resolves to off. A typo must fail closed."""

    settings = NervousSystemSettings.from_env(_env(CYNOLYCUS_SUBMIT_ENABLED=value))

    assert settings.submit_enabled is (value.strip().lower() == "true")


# ---------------------------------------------------------------------------
# Development keeps its shortcuts
# ---------------------------------------------------------------------------


def _dev_env(**updates: str) -> dict[str, str]:
    payload = {
        "CYNOLYCUS_ENVIRONMENT": "DEVELOPMENT",
        "CYNOLYCUS_NERVOUS_SYSTEM_MODE": "SHADOW",
        "CYNOLYCUS_DATABASE_URL": "postgresql+psycopg://u:p@127.0.0.1:55432/cynolycus",
        "CYNOLYCUS_OPERATIONAL_ROOT": "/tmp/cynolycus",
        "CYNOLYCUS_EXECUTION_JOURNAL": "local",
        "CYNOLYCUS_ACCOUNT_ALIAS": "paper",
    }
    payload.update(updates)
    return payload


def test_development_permits_tcp_and_a_local_journal() -> None:
    """Otherwise nobody can run the thing on a laptop, and a rule people cannot
    follow is a rule they route around.
    """

    settings = NervousSystemSettings.from_env(_dev_env())

    assert settings.journal_backend == "local"
    assert settings.submit_enabled is False


def test_development_does_not_require_cloud_bindings() -> None:
    settings = NervousSystemSettings.from_env(_dev_env())

    assert settings.cloud_sql_instance is None
    assert settings.secret_binding is None


# ---------------------------------------------------------------------------
# Production-live parses, and can never execute
# ---------------------------------------------------------------------------


def test_production_live_parses_for_read_only_inspection() -> None:
    """Reconciling a live account's history is a legitimate read. Refusing to
    parse would mean the only way to inspect it is to disable the guard.
    """

    settings = NervousSystemSettings.from_env(
        _env(CYNOLYCUS_ENVIRONMENT="PRODUCTION_LIVE", CYNOLYCUS_ACCOUNT_ALIAS="live")
    )

    assert settings.environment.value == "PRODUCTION_LIVE"


def test_production_live_never_reports_submission_as_enabled() -> None:
    settings = NervousSystemSettings.from_env(
        _env(
            CYNOLYCUS_ENVIRONMENT="PRODUCTION_LIVE",
            CYNOLYCUS_ACCOUNT_ALIAS="live",
            CYNOLYCUS_SUBMIT_ENABLED="true",
        )
    )

    assert settings.submit_enabled is False


def test_production_live_returns_the_stable_veto_code() -> None:
    settings = NervousSystemSettings.from_env(
        _env(CYNOLYCUS_ENVIRONMENT="PRODUCTION_LIVE", CYNOLYCUS_ACCOUNT_ALIAS="live")
    )

    assert settings.execution_veto() == "ENV_PRODUCTION_LIVE_DISABLED_MVP"


def test_a_permitted_environment_has_no_veto() -> None:
    assert NervousSystemSettings.from_env(_env()).execution_veto() is None


# ---------------------------------------------------------------------------
# Secrets never surface
# ---------------------------------------------------------------------------


def test_the_password_is_absent_from_repr() -> None:
    assert "s3cret" not in repr(NervousSystemSettings.from_env(_env()))


def test_the_password_is_absent_from_a_json_dump() -> None:
    dumped = NervousSystemSettings.from_env(_env()).model_dump_json()

    assert "s3cret" not in dumped


def test_the_password_is_absent_from_a_validation_error() -> None:
    """A validation error is the most likely thing to be pasted into a chat."""

    with pytest.raises(ValidationError) as failure:
        NervousSystemSettings.from_env(
            _env(CYNOLYCUS_DATABASE_URL=SOCKET_DSN.replace("/cynolycus?", "/?"))
        )

    assert "s3cret" not in str(failure.value)


def test_the_password_is_absent_from_health_output() -> None:
    body = json.dumps(NervousSystemSettings.from_env(_env()).health_summary())

    assert "s3cret" not in body


def test_health_output_still_says_what_it_is_connected_to() -> None:
    """Redaction that removes the useful part just gets bypassed."""

    summary = NervousSystemSettings.from_env(_env()).health_summary()

    assert summary["environment"] == "QA_PAPER"
    assert summary["account_alias"] == "paper"
    assert summary["cloud_sql_instance"] == "cynolycusbot-dev:us-east5:cynolycus-qa"
    assert summary["journal_backend"] == "gcs"


def test_the_password_is_absent_from_logs(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        settings = NervousSystemSettings.from_env(_env())
        logging.getLogger(__name__).info("settings: %s", settings)

    assert "s3cret" not in caplog.text
