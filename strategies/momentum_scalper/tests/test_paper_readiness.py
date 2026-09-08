from __future__ import annotations

from dataclasses import dataclass

from alpaca.data.enums import DataFeed

from strategies.momentum_scalper.configs.v1 import MomentumScalperConfigV1
from strategies.momentum_scalper.live.readiness import (
    SIP_ENTITLEMENT_CODE,
    check_paper_readiness,
)


@dataclass
class _Settings:
    environment: object
    cloud_sql_instance: str | None = "project:region:instance"
    gcs_bucket: str | None = "journal-bucket"
    alpaca_account_id: str | None = "paper-account"
    secret_binding: str | None = "projects/p/secrets/alpaca"


@dataclass
class _Environment:
    value: str


def _report(*, sip_error: Exception | None = None, environment: str = "QA_PAPER"):
    def quotes(feed: DataFeed) -> bool:
        if feed is DataFeed.SIP and sip_error is not None:
            raise sip_error
        return True

    return check_paper_readiness(
        config=MomentumScalperConfigV1(),
        settings=_Settings(_Environment(environment)),  # type: ignore[arg-type]
        trading_host="https://paper-api.alpaca.markets",
        account_probe=lambda: object(),
        quote_probe=quotes,
    )


def test_paper_readiness_requires_sip_and_qa_durable_configuration() -> None:
    report = _report()

    assert report.ready_for_paper_submission is True
    assert report.sip_reason is None


def test_paper_readiness_reports_missing_sip_entitlement() -> None:
    report = _report(sip_error=RuntimeError("subscription does not permit querying recent SIP data"))

    assert report.sip_quote_readable is False
    assert report.sip_reason == SIP_ENTITLEMENT_CODE
    assert report.ready_for_paper_submission is False


def test_paper_readiness_rejects_development_even_with_working_broker_data() -> None:
    report = _report(environment="DEVELOPMENT")

    assert report.nervous_system_is_qa_paper is False
    assert report.ready_for_paper_submission is False
