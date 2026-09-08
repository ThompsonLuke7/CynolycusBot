"""Read-only preflight for a momentum-scalper paper session.

This module never submits, replaces, or cancels an order.  It verifies the
minimum broker/data conditions before an operator enables a paper session.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient

from core.API.Alpaca_API.core.config import AlpacaConfig
from core.API.Alpaca_API.options.options_api import OptionsClientConfig
from core.nervous_system.config.runtime import NervousSystemSettings
from strategies.momentum_scalper.configs.v1 import MomentumScalperConfigV1, load_config


PAPER_HOST = "paper-api.alpaca.markets"
SIP_ENTITLEMENT_CODE = "SIP_ENTITLEMENT_MISSING"


@dataclass(frozen=True)
class PaperReadinessReport:
    config_paper_only: bool
    trading_host_is_paper: bool
    nervous_system_is_qa_paper: bool
    durable_execution_prerequisites_present: bool
    paper_account_readable: bool
    paper_account_reason: str | None
    iex_quote_readable: bool
    iex_reason: str | None
    sip_quote_readable: bool
    sip_reason: str | None

    @property
    def ready_for_paper_submission(self) -> bool:
        return all((
            self.config_paper_only,
            self.trading_host_is_paper,
            self.nervous_system_is_qa_paper,
            self.durable_execution_prerequisites_present,
            self.paper_account_readable,
            self.sip_quote_readable,
        ))

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ready_for_paper_submission"] = self.ready_for_paper_submission
        return payload


def _sip_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "subscription does not permit querying recent sip data" in text:
        return SIP_ENTITLEMENT_CODE
    return type(exc).__name__


def _read_quote(
    client: Any,
    feed: DataFeed,
) -> bool:
    quotes = client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols="SPY", feed=feed)
    )
    return bool(quotes.get("SPY"))


def _probe_with_retry(probe: Callable[[], object]) -> tuple[bool, Exception | None]:
    """Retry a transient transport failure; never retry an entitlement refusal."""

    last_error: Exception | None = None
    for delay in (0.0, 0.25, 0.75):
        if delay:
            time.sleep(delay)
        try:
            probe()
            return True, None
        except Exception as exc:  # The result is a preflight report, not a crash.
            last_error = exc
            if type(exc).__name__ != "ConnectionError":
                break
    return False, last_error


def check_paper_readiness(
    *,
    config: MomentumScalperConfigV1,
    settings: NervousSystemSettings,
    trading_host: str,
    account_probe: Callable[[], object],
    quote_probe: Callable[[DataFeed], bool],
) -> PaperReadinessReport:
    """Evaluate preflight results with no broker mutation capability."""

    account_ok, account_error = _probe_with_retry(account_probe)
    account_reason = type(account_error).__name__ if account_error is not None else None

    iex_ok, iex_error = _probe_with_retry(lambda: quote_probe(DataFeed.IEX))
    iex_reason = type(iex_error).__name__ if iex_error is not None else None

    sip_ok, sip_error = _probe_with_retry(lambda: quote_probe(DataFeed.SIP))
    sip_reason = _sip_reason(sip_error) if sip_error is not None else None

    durable = bool(
        settings.cloud_sql_instance
        and settings.gcs_bucket
        and settings.alpaca_account_id
        and settings.secret_binding
    )
    return PaperReadinessReport(
        config_paper_only=config.paper_only,
        trading_host_is_paper=(urlparse(trading_host).hostname or "").lower() == PAPER_HOST,
        nervous_system_is_qa_paper=settings.environment.value == "QA_PAPER",
        durable_execution_prerequisites_present=durable,
        paper_account_readable=account_ok,
        paper_account_reason=account_reason,
        iex_quote_readable=iex_ok,
        iex_reason=iex_reason,
        sip_quote_readable=sip_ok,
        sip_reason=sip_reason,
    )


def probe_paper_readiness(
    *,
    env_file: str | None = ".env",
    config: MomentumScalperConfigV1 | None = None,
) -> PaperReadinessReport:
    """Run authenticated, read-only Alpaca account and quote checks."""

    config = config or load_config()
    settings = NervousSystemSettings.from_env()
    credentials = AlpacaConfig.from_env(env_file)
    trading_host = OptionsClientConfig.from_env(env_file).trading_base_url
    account_client = TradingClient(credentials.key_id, credentials.secret_key, paper=True)
    data_client = StockHistoricalDataClient(credentials.key_id, credentials.secret_key)
    return check_paper_readiness(
        config=config,
        settings=settings,
        trading_host=trading_host,
        account_probe=account_client.get_account,
        quote_probe=lambda feed: _read_quote(data_client, feed),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only momentum-scalper paper preflight")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = probe_paper_readiness(env_file=args.env_file)
    payload = report.as_dict()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}={value}")
    return 0 if report.ready_for_paper_submission else 2


__all__ = [
    "PAPER_HOST",
    "SIP_ENTITLEMENT_CODE",
    "PaperReadinessReport",
    "check_paper_readiness",
    "probe_paper_readiness",
]


if __name__ == "__main__":
    raise SystemExit(main())
