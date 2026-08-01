from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.broker_equity_snapshot import adapt_broker_portfolio_snapshot, capture_snapshot
from core.nervous_system.contracts.enums import AssetClass
from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.states import PortfolioState


UTC = timezone.utc
CAPTURED_AT = datetime(2026, 7, 30, 20, 5, tzinfo=UTC)
_SECRET_API_KEY = "AKIA_PORTFOLIO_DO_NOT_LEAK_7f3d"
_SECRET_TOKEN = "broker-token-round3-secret-91ac"
_SECRET_PASSWORD = "database-password-round3-secret-45be"
_SECRET_DSN = (
    "postgresql+psycopg://portfolio_user:"
    f"{_SECRET_PASSWORD}@db.internal/nervous_system"
)
_SECRET_URL = (
    "https://paper-api.alpaca.markets/v2/orders?"
    f"api_key={_SECRET_API_KEY}&token={_SECRET_TOKEN}"
)
_SECRET_FRAGMENTS = (
    _SECRET_API_KEY,
    _SECRET_TOKEN,
    _SECRET_PASSWORD,
    _SECRET_DSN,
    _SECRET_URL,
)


class _CredentialBearingError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            {
                "api_key": _SECRET_API_KEY,
                "nested": [
                    {"dsn": _SECRET_DSN},
                    {"url": _SECRET_URL, "token": _SECRET_TOKEN},
                ],
            }
        )
        self.render_attempts = 0

    def __str__(self) -> str:
        self.render_attempts += 1
        return f"api_key={_SECRET_API_KEY} url={_SECRET_URL}"

    def __repr__(self) -> str:
        self.render_attempts += 1
        return f"_CredentialBearingError({_SECRET_DSN!r})"


def _assert_secret_free(*values: object) -> None:
    rendered = json.dumps(values, sort_keys=True)
    for secret in _SECRET_FRAGMENTS:
        assert secret not in rendered


def _raw_snapshot() -> dict[str, object]:
    return {
        "schema_version": 2,
        "captured_at_utc": CAPTURED_AT.isoformat(),
        "captured_at_et": "2026-07-30T16:05:00-04:00",
        "session_date_et": "2026-07-30",
        "session_phase": "extended",
        "account_label": "paper",
        "account": {
            "equity": "100123.45",
            "last_equity": "100000.00",
            "portfolio_value": "100123.45",
            "cash": "50000.00",
            "buying_power": "200000.00",
        },
        "day_pl": 123.45,
        "positions": [
            {
                "symbol": "AAPL",
                "asset_class": "us_equity",
                "qty": "100",
                "side": "long",
                "avg_entry_price": "190.00",
                "current_price": "195.00",
                "market_value": "19500.00",
                "cost_basis": "19000.00",
            },
            {
                "symbol": "AAPL260821C00195000",
                "asset_class": "us_option",
                "qty": "2",
                "side": "long",
                "avg_entry_price": "3.10",
                "current_price": "3.50",
                "market_value": "700.00",
                "cost_basis": "620.00",
            },
            {
                "symbol": "MSFT",
                "asset_class": "us_equity",
                "qty": "10",
                "side": "long",
                "avg_entry_price": "400.00",
                "current_price": "401.00",
                "market_value": "4010.00",
                "cost_basis": "4000.00",
            },
        ],
        "open_orders": [{"id": "order-1"}, {"id": "order-2"}],
    }


def test_portfolio_adapter_preserves_broker_facts_occ_identity_and_fill_ownership() -> None:
    state = adapt_broker_portfolio_snapshot(
        _raw_snapshot(),
        strategy_ownership={
            "AAPL": "meta_ranker",
            "AAPL260821C00195000": "dealer_ranker",
        },
    )

    assert isinstance(state, PortfolioState)
    assert state.account_alias == "paper"
    assert state.available_at == CAPTURED_AT
    assert state.broker_observed_at == CAPTURED_AT
    assert state.equity == 100123.45
    assert state.cash == 50000.0
    assert state.buying_power == 200000.0
    assert state.day_pl == 123.45
    assert state.open_order_ids == ("order-1", "order-2")

    positions = {position.symbol: position for position in state.positions}
    assert positions["AAPL"].asset_class is AssetClass.EQUITY
    assert positions["AAPL"].quantity == 100.0
    assert positions["AAPL"].strategy_id == "meta_ranker"
    assert positions["AAPL"].ownership_status == "ASSIGNED"
    assert positions["AAPL260821C00195000"].symbol == "AAPL260821C00195000"
    assert positions["AAPL260821C00195000"].underlying == "AAPL"
    assert positions["AAPL260821C00195000"].strategy_id == "dealer_ranker"
    assert positions["MSFT"].strategy_id is None
    assert positions["MSFT"].ownership_status == "UNASSIGNED"
    assert state.data_quality.is_usable
    assert state.lineage_ids and "core.broker_equity_snapshot" in state.lineage_ids[0]
    assert state.producer == "core.broker_equity_snapshot"
    assert state.model_version == "broker-portfolio-adapter@1"
    assert state.feature_version == "broker-portfolio@1"
    assert state.config_version == "broker-portfolio@1:validity-24h"
    lineage = json.loads(state.lineage_ids[0])
    assert lineage["source_id"] == "core.broker_equity_snapshot"
    assert len(lineage["content_hash"]) == 64
    assert lineage["record_locator"] == "account_snapshot:paper:2026-07-30T20:05:00+00:00"
    assert content_hash(state, exclude={"state_id"}) == content_hash(
        adapt_broker_portfolio_snapshot(
            _raw_snapshot(),
            strategy_ownership={
                "AAPL": "meta_ranker",
                "AAPL260821C00195000": "dealer_ranker",
            },
        ),
        exclude={"state_id"},
    )


def test_portfolio_adapter_does_not_call_a_broker_or_infer_underlying_ownership() -> None:
    class ExplodingBroker:
        def __getattr__(self, name: str):
            raise AssertionError(f"adapter made broker call: {name}")

    raw = _raw_snapshot()
    raw["broker"] = ExplodingBroker()
    state = adapt_broker_portfolio_snapshot(raw, strategy_ownership={"AAPL": "meta_ranker"})
    positions = {position.symbol: position for position in state.positions}
    assert positions["AAPL260821C00195000"].strategy_id is None
    assert positions["AAPL260821C00195000"].ownership_status == "UNASSIGNED"


def test_portfolio_adapter_does_not_republish_legacy_open_order_error_text() -> None:
    raw = _raw_snapshot()
    raw.pop("open_orders")
    raw["open_orders_observation"] = {
        "status": "FAILED",
        "error": _SECRET_URL,
        "context": {"api_key": _SECRET_API_KEY, "nested": [_SECRET_DSN]},
    }

    state = adapt_broker_portfolio_snapshot(raw, strategy_ownership={})

    assert state.open_order_ids == ()
    issues = {issue.code: issue for issue in state.data_quality.issues}
    assert issues["OPEN_ORDERS_NOT_OBSERVED"].severity.value == "WARNING"
    assert "FAILED" in issues["OPEN_ORDERS_NOT_OBSERVED"].message
    assert "OPEN_ORDERS_READ_FAILED" in issues["OPEN_ORDERS_NOT_OBSERVED"].message
    _assert_secret_free(state.model_dump(mode="json"))


@pytest.mark.parametrize(
    "captured_at",
    [None, "2026-07-30T20:05:00", "not-a-timestamp"],
)
def test_portfolio_adapter_fails_closed_for_unaware_or_invalid_capture_time(captured_at) -> None:
    raw = _raw_snapshot()
    if captured_at is None:
        raw.pop("captured_at_utc")
    else:
        raw["captured_at_utc"] = captured_at

    with pytest.raises(ValueError, match="captured_at_utc"):
        adapt_broker_portfolio_snapshot(raw, strategy_ownership={})


@pytest.mark.parametrize(
    ("field", "value"),
    [("equity", "NaN"), ("cash", float("inf"))],
)
def test_portfolio_adapter_fails_closed_for_non_finite_account_values(field, value) -> None:
    raw = _raw_snapshot()
    raw["account"][field] = value

    with pytest.raises(ValueError, match=field):
        adapt_broker_portfolio_snapshot(raw, strategy_ownership={})


def test_portfolio_adapter_fails_closed_for_non_finite_position_quantity() -> None:
    raw = _raw_snapshot()
    raw["positions"][0]["qty"] = float("nan")

    with pytest.raises(ValueError, match="qty"):
        adapt_broker_portfolio_snapshot(raw, strategy_ownership={})


class _Client:
    def __init__(self, *, orders=None, orders_error: Exception | None = None):
        self.calls = []
        self.orders = (
            [{"id": "order-2", "symbol": "MSFT"}, {"id": "order-1", "symbol": "AAPL"}]
            if orders is None
            else orders
        )
        self.orders_error = orders_error

    def get_account(self):
        self.calls.append("get_account")
        return _raw_snapshot()["account"]

    def get_positions(self):
        self.calls.append("get_positions")
        return _raw_snapshot()["positions"]

    def get_orders(self, **params):
        self.calls.append(("get_orders", params))
        if self.orders_error is not None:
            raise self.orders_error
        return self.orders


class _LegacyClient:
    def __init__(self):
        self.calls = []

    def get_account(self):
        self.calls.append("get_account")
        return _raw_snapshot()["account"]

    def get_positions(self):
        self.calls.append("get_positions")
        return _raw_snapshot()["positions"]


class _RecordingStates:
    def __init__(self, events: list[tuple[str, object]], path):
        self.events = events
        self.path = path
        self.staged_states = []

    def insert_states_idempotently(self, states):
        staged = tuple(states)
        self.staged_states.extend(staged)
        self.events.append(("stage", self.path.read_text()))
        return {state.state_id: state.state_id for state in staged}


class _RecordingUow:
    def __init__(self, events: list[tuple[str, object]], path):
        self.events = events
        self.states = _RecordingStates(events, path)

    def commit(self):  # pragma: no cover - caller owns the transaction
        raise AssertionError("snapshot writer must not commit caller-owned UOW")

    def rollback(self):  # pragma: no cover - caller owns the transaction
        raise AssertionError("snapshot writer must not rollback caller-owned UOW")


def _record_fsync(monkeypatch, events: list[tuple[str, object]]) -> None:
    original_fsync = __import__("os").fsync

    def recording_fsync(fd):
        events.append(("fsync", fd))
        return original_fsync(fd)

    monkeypatch.setattr("core.broker_equity_snapshot.os.fsync", recording_fsync)


def test_alpaca_client_open_order_method_uses_read_only_get_and_status_param() -> None:
    calls = []
    client = object.__new__(AlpacaOptionsClient)
    client._trading_base = "https://paper-api.alpaca.markets"  # type: ignore[attr-defined]

    def record_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return [{"id": "order-1"}]

    client._request = record_request  # type: ignore[method-assign]

    assert client.get_orders(status="open") == [{"id": "order-1"}]
    assert calls == [
        (
            "GET",
            "https://paper-api.alpaca.markets/v2/orders",
            {"params": {"status": "open"}},
        )
    ]


def test_snapshot_stages_only_after_fsync_and_leaves_commit_to_caller(
    tmp_path, monkeypatch
) -> None:
    events: list[tuple[str, object]] = []
    path = tmp_path / "broker_equity_20260730_paper.jsonl"
    _record_fsync(monkeypatch, events)
    uow = _RecordingUow(events, path)
    client = _Client()
    result = capture_snapshot(
        client=client,
        account_label="paper",
        root=tmp_path,
        now=CAPTURED_AT,
        unit_of_work=uow,
        strategy_ownership={"AAPL": "meta_ranker"},
    )

    assert result["publication_status"] == "STAGED"
    assert result["publication_status"] != "PUBLISHED"
    assert [event[0] for event in events] == ["fsync", "stage"]
    assert len(uow.states.staged_states) == 1
    assert client.calls == [
        "get_account",
        "get_positions",
        ("get_orders", {"status": "open"}),
    ]
    row = json.loads(path.read_text())
    assert row["open_orders_observation"] == {"status": "OBSERVED"}
    assert row["open_orders"] == [{"id": "order-1"}, {"id": "order-2"}]
    state = uow.states.staged_states[0]
    assert state.open_order_ids == ("order-1", "order-2")
    assert "OPEN_ORDERS_NOT_OBSERVED" not in {
        issue.code for issue in state.data_quality.issues
    }


def test_snapshot_distinguishes_observed_zero_open_orders_from_unavailable(
    tmp_path,
) -> None:
    path = tmp_path / "broker_equity_20260730_paper.jsonl"
    events: list[tuple[str, object]] = []
    uow = _RecordingUow(events, path)
    client = _Client(orders=[])

    result = capture_snapshot(
        client=client,
        account_label="paper",
        root=tmp_path,
        now=CAPTURED_AT,
        unit_of_work=uow,
    )

    row = json.loads(path.read_text())
    assert result["publication_status"] == "STAGED"
    assert row["open_orders_observation"] == {"status": "OBSERVED"}
    assert row["open_orders"] == []
    assert uow.states.staged_states[0].open_order_ids == ()
    assert "OPEN_ORDERS_NOT_OBSERVED" not in {
        issue.code for issue in uow.states.staged_states[0].data_quality.issues
    }
    assert client.calls == [
        "get_account",
        "get_positions",
        ("get_orders", {"status": "open"}),
    ]


def test_snapshot_unsupported_open_orders_remain_durable_but_are_not_staged(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "broker_equity_20260730_paper.jsonl"
    events: list[tuple[str, object]] = []
    _record_fsync(monkeypatch, events)
    uow = _RecordingUow(events, path)
    client = _LegacyClient()

    result = capture_snapshot(
        client=client,
        account_label="paper",
        root=tmp_path,
        now=CAPTURED_AT,
        unit_of_work=uow,
    )

    row = json.loads(path.read_text())
    assert result["publication_status"] == "FAILED"
    assert result["publication_error"] == "OPEN_ORDERS_READ_UNSUPPORTED"
    assert row["open_orders_observation"] == {
        "status": "UNSUPPORTED",
        "error_code": "OPEN_ORDERS_READ_UNSUPPORTED",
    }
    assert "open_orders" not in row
    assert client.calls == ["get_account", "get_positions"]
    assert [event[0] for event in events] == ["fsync"]
    assert uow.states.staged_states == []
    local_state = adapt_broker_portfolio_snapshot(row, strategy_ownership={})
    assert local_state.open_order_ids == ()
    assert "OPEN_ORDERS_NOT_OBSERVED" in {
        issue.code for issue in local_state.data_quality.issues
    }


def test_snapshot_failed_open_order_read_remains_durable_but_is_not_staged(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "broker_equity_20260730_paper.jsonl"
    events: list[tuple[str, object]] = []
    _record_fsync(monkeypatch, events)
    uow = _RecordingUow(events, path)
    orders_error = _CredentialBearingError()
    client = _Client(orders_error=orders_error)

    result = capture_snapshot(
        client=client,
        account_label="paper",
        root=tmp_path,
        now=CAPTURED_AT,
        unit_of_work=uow,
    )

    row = json.loads(path.read_text())
    assert result["publication_status"] == "FAILED"
    assert result["publication_error"] == "OPEN_ORDERS_READ_FAILED"
    assert result["publication_exception_type"] == "RuntimeError"
    assert row["open_orders_observation"] == {
        "status": "FAILED",
        "error_code": "OPEN_ORDERS_READ_FAILED",
        "exception_type": "RuntimeError",
    }
    assert "open_orders" not in row
    assert client.calls == [
        "get_account",
        "get_positions",
        ("get_orders", {"status": "open"}),
    ]
    assert [event[0] for event in events] == ["fsync"]
    assert uow.states.staged_states == []
    local_state = adapt_broker_portfolio_snapshot(row, strategy_ownership={})
    assert "OPEN_ORDERS_NOT_OBSERVED" in {
        issue.code for issue in local_state.data_quality.issues
    }
    _assert_secret_free(
        result,
        row,
        local_state.model_dump(mode="json"),
        path.read_text(),
    )
    assert orders_error.render_attempts == 0


def test_snapshot_rejects_naive_now_before_broker_file_or_uow_side_effects(tmp_path) -> None:
    events: list[tuple[str, object]] = []
    path = tmp_path / "broker_equity_20260730_paper.jsonl"
    client = _Client()
    uow = _RecordingUow(events, path)
    error = None

    try:
        capture_snapshot(
            client=client,
            account_label="paper",
            root=tmp_path,
            now=datetime(2026, 7, 30, 20, 5),
            unit_of_work=uow,
        )
    except ValueError as exc:
        error = exc

    assert client.calls == []
    assert list(tmp_path.iterdir()) == []
    assert events == []
    assert uow.states.staged_states == []
    assert error is not None
    assert "now must be timezone-aware" in str(error)


def test_snapshot_default_now_is_utc_and_observes_open_orders(tmp_path) -> None:
    client = _Client()

    result = capture_snapshot(client=client, account_label="paper", root=tmp_path)

    captured_at = datetime.fromisoformat(result["captured_at_utc"])
    assert captured_at.utcoffset() == timezone.utc.utcoffset(captured_at)
    assert client.calls == [
        "get_account",
        "get_positions",
        ("get_orders", {"status": "open"}),
    ]
    assert list(tmp_path.glob("broker_equity_*_paper.jsonl"))


def test_snapshot_returns_publication_failure_after_local_snapshot_is_durable(tmp_path) -> None:
    publication_error = _CredentialBearingError()

    class FailingStates:
        def insert_states_idempotently(self, states):
            raise publication_error

    class FailingUow:
        states = FailingStates()

    result = capture_snapshot(
        client=_Client(),
        account_label="paper",
        root=tmp_path,
        now=CAPTURED_AT,
        unit_of_work=FailingUow(),
    )

    assert result["publication_status"] == "FAILED"
    assert result["publication_error"] == "PORTFOLIO_STATE_PUBLICATION_FAILED"
    assert result["publication_exception_type"] == "RuntimeError"
    assert result["publication_status"] != "STALE_SUCCESS"
    path = tmp_path / "broker_equity_20260730_paper.jsonl"
    assert path.exists()
    _assert_secret_free(result, json.loads(path.read_text()), path.read_text())
    assert publication_error.render_attempts == 0
