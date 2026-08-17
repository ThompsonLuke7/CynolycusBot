"""The governed path can actually build its broker gateway.

`_build_gateway` is handed to DecisionCoordinator as a lazy factory and is only
called at the moment of submission, so nothing exercised it: planning, policy
and persistence all pass with it broken. It was broken --
`AlpacaPaperAdapter(environment=...)` against a constructor that also requires
`client`, `account_alias` and `trading_base_url`, i.e. a TypeError at the exact
instant an order was due to leave.

The rule these tests hold down: **the URL the adapter validates must be the URL
the client will actually post to.** Validating a URL from settings while the
client resolved a different one from its own env file would let the paper-host
check pass on a client pointed somewhere else.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.nervous_system.contracts.enums import RuntimeEnvironment
from core.nervous_system.execution.alpaca_adapter import BrokerAuthenticationError
from signals.meta_context.meta_ranker import gateway_execution as ge


PAPER = "https://paper-api.alpaca.markets"
LIVE = "https://api.alpaca.markets"


class _Settings:
    """Only the attributes _build_gateway reads."""

    def __init__(self, *, environment=RuntimeEnvironment.DEVELOPMENT,
                 account_alias="paper", alpaca_base_url=None, tmp_path=None):
        self.environment = environment
        self.account_alias = account_alias
        self.alpaca_base_url = alpaca_base_url
        self.operational_root = tmp_path


class _Client:
    def __init__(self, trading_base=PAPER):
        self._trading_base = trading_base


@pytest.fixture
def paper_client(monkeypatch):
    """Never construct a real client: that reads .env and would bind to
    whatever account the developer happens to have configured."""

    made: list[dict] = []

    def _factory(**kwargs):
        made.append(kwargs)
        return _Client()

    monkeypatch.setattr(ge, "AlpacaOptionsClient", _factory, raising=False)
    return made


def _build(tmp_path, **settings_kwargs):
    return ge._build_gateway(
        _Settings(tmp_path=tmp_path, **settings_kwargs),
        None,
        lambda: datetime(2026, 8, 17, 14, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# It builds at all
# ---------------------------------------------------------------------------


def test_the_gateway_constructs(tmp_path, paper_client) -> None:
    """The regression: this raised TypeError at submission time."""

    assert _build(tmp_path) is not None


def test_the_broker_client_is_built_from_the_paper_profile(tmp_path, paper_client) -> None:
    """Same profile the runner uses, so the gateway and the runner's own reads
    are talking to one account rather than two."""

    _build(tmp_path)

    assert paper_client[0]["env_file"] == ".env#PAPER"


# ---------------------------------------------------------------------------
# The validated URL is the URL in use
# ---------------------------------------------------------------------------


def test_a_settings_url_that_contradicts_the_client_is_refused(tmp_path, monkeypatch) -> None:
    """Two sources of truth for where orders go is a misconfiguration, and the
    safe-looking one must not be the one that gets checked."""

    monkeypatch.setattr(ge, "AlpacaOptionsClient", lambda **_: _Client(LIVE), raising=False)

    with pytest.raises(BrokerAuthenticationError, match="disagree"):
        _build(tmp_path, alpaca_base_url=PAPER, environment=RuntimeEnvironment.QA_PAPER)


def test_a_live_client_url_is_refused_under_qa_paper(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ge, "AlpacaOptionsClient", lambda **_: _Client(LIVE), raising=False)

    with pytest.raises(BrokerAuthenticationError):
        _build(tmp_path, environment=RuntimeEnvironment.QA_PAPER)


def test_a_matching_paper_url_is_accepted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ge, "AlpacaOptionsClient", lambda **_: _Client(PAPER), raising=False)

    assert _build(tmp_path, alpaca_base_url=PAPER,
                  environment=RuntimeEnvironment.QA_PAPER) is not None


def test_development_still_refuses_a_live_client_url(tmp_path, monkeypatch) -> None:
    """DEVELOPMENT skips the adapter's own QA_PAPER host check, so this is the
    only thing standing between a mistyped base URL and the live account."""

    monkeypatch.setattr(ge, "AlpacaOptionsClient", lambda **_: _Client(LIVE), raising=False)

    with pytest.raises(BrokerAuthenticationError):
        _build(tmp_path, environment=RuntimeEnvironment.DEVELOPMENT)


def test_production_live_is_refused_before_a_client_exists(tmp_path, monkeypatch) -> None:
    def _explode(**_):
        raise AssertionError("no broker client may be built for PRODUCTION_LIVE")

    monkeypatch.setattr(ge, "AlpacaOptionsClient", _explode, raising=False)

    with pytest.raises(BrokerAuthenticationError):
        _build(tmp_path, environment=RuntimeEnvironment.PRODUCTION_LIVE)
