from datetime import datetime, timezone
import json

from core.broker_equity_snapshot import capture_snapshot


class _Client:
    def get_account(self):
        return {
            "equity": "100123.45", "last_equity": "100000", "portfolio_value": "100123.45",
            "cash": "50000", "buying_power": "200000", "account_number": "must_not_persist",
        }

    def get_positions(self):
        return [{
            "symbol": "ABC", "asset_class": "us_equity", "qty": "100", "side": "long",
            "avg_entry_price": "10", "current_price": "11", "market_value": "1100",
            "cost_basis": "1000", "unrealized_pl": "100", "unrealized_plpc": "0.1",
            "unrealized_intraday_pl": "25", "unrealized_intraday_plpc": "0.023",
        }]


def test_capture_snapshot_writes_account_and_position_marks_without_account_id(tmp_path):
    out = capture_snapshot(
        client=_Client(), account_label="paper", root=tmp_path,
        now=datetime(2026, 7, 14, 20, 5, tzinfo=timezone.utc),
    )
    path = tmp_path / "broker_equity_20260714_paper.jsonl"
    assert out["path"] == str(path)
    row = json.loads(path.read_text())
    assert row["session_date_et"] == "2026-07-14"
    assert row["account"]["equity"] == "100123.45"
    assert row["positions"][0]["symbol"] == "ABC"
    assert "account_number" not in json.dumps(row)


# --- schema v2: session phase, account day P&L, derived unrealized ----------
#
# 2026-07-24 showed why these are needed: the 20:05 ET capture reported
# `unrealized_intraday_pl` of -$2,240/-$3,010/-$1,560 for three positions opened
# THAT day whose true unrealized P&L was -$1,280/$0/+$520 -- Alpaca computes
# intraday against a previous close that does not exist for a same-day open.
# The account delta is always right; the derived field makes a disagreement
# visible instead of silent.

from datetime import datetime as _dt  # noqa: E402
from zoneinfo import ZoneInfo as _ZI  # noqa: E402

from core.broker_equity_snapshot import capture_snapshot as _capture  # noqa: E402
from core.broker_equity_snapshot import session_phase as _phase  # noqa: E402

_ET_TZ = _ZI("America/New_York")


class _StubClient:
    def __init__(self, account, positions):
        self._account, self._positions = account, positions

    def get_account(self):
        return self._account

    def get_positions(self):
        return self._positions


def test_session_phase_boundaries():
    def at(h, m=0, day=24):
        return _phase(_dt(2026, 7, day, h, m, tzinfo=_ET_TZ))

    assert at(3, 59) == "closed"
    assert at(4, 0) == "premarket"
    assert at(9, 29) == "premarket"
    assert at(9, 30) == "regular"
    assert at(15, 59) == "regular"
    assert at(16, 0) == "extended"
    assert at(19, 59) == "extended"
    assert at(20, 0) == "closed"
    assert at(11, 0, day=25) == "closed"  # Saturday


def test_day_pl_and_derived_unrealized(tmp_path):
    record = _capture(
        client=_StubClient(
            {"equity": "926680.18", "last_equity": "973068.72", "cash": "1",
             "buying_power": "2", "portfolio_value": "926680.18"},
            [
                {"symbol": "SNDK", "asset_class": "us_equity", "qty": "100", "side": "long",
                 "market_value": "143301", "cost_basis": "167495", "unrealized_pl": "-24194"},
                # Shorts carry negative qty/market_value/cost_basis, so the
                # subtraction must NOT be sign-flipped again.
                {"symbol": "DIA", "asset_class": "us_equity", "qty": "-40", "side": "short",
                 "market_value": "-20640", "cost_basis": "-20800", "unrealized_pl": "160"},
                {"symbol": "NOBASIS", "asset_class": "us_equity", "qty": "1", "side": "long",
                 "market_value": None, "cost_basis": None},
            ],
        ),
        account_label="paper",
        root=tmp_path,
        now=_dt(2026, 7, 24, 16, 5, tzinfo=_ET_TZ),
    )

    assert record["schema_version"] == 2
    assert record["session_phase"] == "extended"
    assert record["day_pl"] == -46388.54
    by_symbol = {p["symbol"]: p for p in record["positions"]}
    assert by_symbol["SNDK"]["unrealized_pl_derived"] == -24194.0
    assert by_symbol["DIA"]["unrealized_pl_derived"] == 160.0
    assert by_symbol["NOBASIS"]["unrealized_pl_derived"] is None


def test_day_pl_is_none_without_a_prior_equity(tmp_path):
    """The first snapshot of a brand-new account reports last_equity 0; a
    900,000-dollar "gain" must not be invented from it."""
    record = _capture(
        client=_StubClient(
            {"equity": "926680.18", "last_equity": "0", "cash": "1",
             "buying_power": "2", "portfolio_value": "926680.18"},
            [],
        ),
        account_label="paper",
        root=tmp_path,
        now=_dt(2026, 7, 24, 16, 5, tzinfo=_ET_TZ),
    )
    assert record["day_pl"] is None


# --------------------------------------------------------------------------
# Regression: the portfolio state must actually be published.
#
# capture_snapshot only stages a PortfolioState when it is given a unit of
# work, and capture_from_env -- the sole production caller -- passed none. So
# nothing ever wrote the state the policy engine reads, SnapshotBuilder always
# resolved portfolio_state=None, and policy.rule.broker vetoed every Meta order
# with BROKER_PORTFOLIO_STATE_MISSING. That rule applies to risk-reducing
# orders, so it trapped exits too: AMLX and MRVL260821C00210000 sat queued
# across the 2026-08-18 and 2026-08-19 pre-open flushes.
# --------------------------------------------------------------------------
import core.broker_equity_snapshot as _bes  # noqa: E402


class _RecordingStates:
    def __init__(self):
        self.inserted = []

    def insert_states_idempotently(self, states):
        self.inserted.extend(states)


class _RecordingUow:
    def __init__(self, *_args, **_kwargs):
        self.states = _RecordingStates()
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def commit(self):
        self.committed = True


class _ObservableClient(_Client):
    """`_Client` plus an observable open-order book.

    Publication fails closed when open orders cannot be read, because an
    unobserved book cannot rule out a duplicate — so a stub without
    `get_orders` would test the refusal path, not the publication path.
    """

    def get_orders(self, **_kwargs):
        return []


def _patch_store(monkeypatch, uow):
    monkeypatch.setattr(_bes, "AlpacaOptionsClient", lambda **_k: _ObservableClient())
    import sys
    import types

    runtime = types.ModuleType("core.nervous_system.config.runtime")

    def _from_env(*args, **kwargs):
        # from_env() with no argument falls back to .env for names the process
        # does not export; passing a mapping opts OUT of that file read. The
        # combined server does not export the CYNOLYCUS_* names, so calling it
        # with os.environ produced six "Field required" errors in production
        # while every .env-sourced shell worked. Recorded so the call shape is
        # asserted, not assumed.
        _from_env.calls.append((args, kwargs))
        return object()

    _from_env.calls = []
    runtime.NervousSystemSettings = type("S", (), {"from_env": staticmethod(_from_env)})
    runtime._from_env = _from_env
    database = types.ModuleType("core.nervous_system.persistence.database")
    database.create_database_engine = lambda _s: object()
    database.create_session_factory = lambda _e: object()
    uow_mod = types.ModuleType("core.nervous_system.persistence.uow")
    uow_mod.UnitOfWork = lambda _factory: uow
    for name, mod in (
        ("core.nervous_system.config.runtime", runtime),
        ("core.nervous_system.persistence.database", database),
        ("core.nervous_system.persistence.uow", uow_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def test_capture_from_env_publishes_and_commits_the_portfolio_state(tmp_path, monkeypatch):
    uow = _RecordingUow()
    _patch_store(monkeypatch, uow)

    result = _bes.capture_from_env(env_file=".env", account_label="paper", root=tmp_path)

    import sys

    runtime = sys.modules["core.nervous_system.config.runtime"]
    assert runtime._from_env.calls == [((), {})], (
        "from_env must be called with no argument so .env backs the process "
        "environment; passing a mapping disables that fallback"
    )
    assert result["publication_status"] == "PUBLISHED"
    assert uow.committed is True
    assert len(uow.states.inserted) == 1
    # A state nobody can find is the same as no state: the policy engine looks
    # the portfolio up by account alias.
    assert uow.states.inserted[0].account_alias == "paper"


def test_capture_from_env_keeps_the_local_record_when_publication_fails(tmp_path, monkeypatch):
    class _Exploding(_RecordingUow):
        def __enter__(self):
            raise RuntimeError("state store unreachable")

    _patch_store(monkeypatch, _Exploding())

    result = _bes.capture_from_env(env_file=".env", account_label="paper", root=tmp_path)

    # The mark record is the durable artifact and must survive a dead store.
    assert result["position_count"] == 1
    assert json.loads(open(result["path"]).readline())["account"]["equity"] == "100123.45"
    assert result["publication_status"] == "FAILED"
