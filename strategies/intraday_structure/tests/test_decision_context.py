"""The engine records what it had in hand at each decision, as of the decision time.

Regression context (2026-11-24):
* Nothing kept the level map, the dealer snapshot or the regime a decision was made
  on, so there was nothing to train on and the replay could only rebuild TA levels.
* The live dealer snapshot was compared with the bar's OPEN. The poller writes ~48 s
  into each minute, so during RTH nearly every snapshot was "from the future" and
  41 of 102 index-ETF setups (10-01..11-23) decided with no dealer levels at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import strategies.intraday_structure.engine as engine_module
from strategies.intraday_structure.config import DecisionContextPolicy, IntradayStructureConfig
from strategies.intraday_structure.decision_context import (
    DecisionContextRecorder, instrument_class, nearest_before, read_session,
)
from strategies.intraday_structure.detectors.base import DetectionDecision
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction, OptionsContext, SetupState, SetupType
from strategies.intraday_structure.options import (
    ArchivedLadderOptionsProvider, CompositeOptionsProvider, DealerSnapshotOptionsProvider,
)
from strategies.intraday_structure.runway import RunwayResult
from strategies.intraday_structure.target_manager import INVALIDATION_TOO_WIDE, TargetPlan, TargetPlanOutcome

T0 = datetime(2026, 11, 23, 16, 0, tzinfo=timezone.utc)  # 11:00 ET
SETUP_ID = "QQQ:long:gamma_structural_level_rejection"


class _Idle:
    def evaluate(self, setup, ctx):  # noqa: ARG002
        return DetectionDecision()


class _ConfirmWhenArmed:
    def evaluate(self, setup, ctx):  # noqa: ARG002
        if setup.state == SetupState.ARMED:
            return DetectionDecision(SetupState.CONFIRMED, "CONFIRM", "stub confirmation")
        return DetectionDecision()


def _engine(records, **overrides):
    engine = IntradayStructureEngine(
        IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0, **overrides),
        context_sink=records.append,
    )
    engine._detectors = {t: _Idle() for t in engine._detectors}
    engine._exhaustion = _Idle()
    return engine


def _bar(i: int, price: float = 650.0) -> Bar:
    return Bar("QQQ", T0 + timedelta(minutes=i), price, price + 0.3, price - 0.3, price, 50_000.0)


def _warm(engine, n: int = 25) -> int:
    engine.register_candidate(Candidate("QQQ", T0, Direction.LONG, ("manual_watchlist",), score=0.6))
    for i in range(n):
        engine.on_bar(_bar(i, 650.0 + 0.1 * (i % 5)))
    return n


def _plan(spot=650.0):
    runway = RunwayResult(0.8, 652.0, (), ("stub",), {"distance": 1.0}, "options_call_wall", 0.9, ("options_call_wall",))
    return TargetPlan(spot - 0.5, (652.0,), runway, 3.0)


def test_confirm_entry_and_manage_are_recorded_with_the_level_map(monkeypatch):
    records = []
    engine = _engine(records)
    monkeypatch.setattr(engine_module, "build_target_plan", lambda setup, ctx: TargetPlanOutcome(_plan(ctx.bar.close)))
    i = _warm(engine)
    engine.setups[SETUP_ID].state = SetupState.ARMED
    engine._detectors[SetupType.STRUCTURAL_REJECTION] = _ConfirmWhenArmed()
    for k in range(3):
        engine.on_bar(_bar(i + k))
    kinds = [r["kind"] for r in records if r["setup_id"] == SETUP_ID]
    assert kinds[:3] == ["confirm", "entry", "manage"]
    confirm = next(r for r in records if r["kind"] == "confirm")
    assert confirm["plan"]["target_level_type"] == "options_call_wall"
    assert confirm["decided_at"] == (datetime.fromisoformat(confirm["bar_ts"]) + timedelta(minutes=1)).isoformat()
    assert len(confirm["features"]) > 20 and confirm["intraday_regime"]["regime"]
    assert confirm["levels"] and confirm["raw_levels"] and len(confirm["raw_levels"]) >= len(confirm["levels"])
    zone = next(x for x in confirm["levels"] if "low" in x)
    assert zone["low"] <= zone["price"] <= zone["high"]
    manage = next(r for r in records if r["kind"] == "manage")
    assert "raw_levels" not in manage, "managed bars keep the clustered map only"


def test_refusals_are_throttled_per_reason(monkeypatch):
    records = []
    engine = _engine(records)
    monkeypatch.setattr(engine_module, "build_target_plan", lambda setup, ctx: TargetPlanOutcome(
        reason=INVALIDATION_TOO_WIDE, proposed_invalidation=640.0, proposed_risk_atr=3.0, max_invalidation_atr=2.0))
    i = _warm(engine)
    engine.setups[SETUP_ID].state = SetupState.ARMED
    engine._detectors[SetupType.STRUCTURAL_REJECTION] = _ConfirmWhenArmed()
    for k in range(31):
        engine.on_bar(_bar(i + k))
    refusals = [r for r in records if r["kind"] == "refuse" and r["setup_id"] == SETUP_ID]
    assert len(refusals) == 3, "first refusal, then one every 15 bars"
    assert refusals[0]["action"] == INVALIDATION_TOO_WIDE and refusals[0]["refusal"]["proposed_risk_atr"] == 3.0


def test_no_sink_builds_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(engine_module, "build_decision_context", lambda **k: called.append(1))
    engine = IntradayStructureEngine(IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0))
    _warm(engine)
    assert not called


# --- dealer snapshot timing -------------------------------------------------------

def _gamma_json(root, symbol, stamp, **levels):
    d = root / symbol
    d.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": stamp.isoformat(), "symbol": symbol, "spot": 650.0, "total_gex": 1.0e8,
               "call_wall": 652.0, "put_wall": 650.0, "gamma_flip": 648.45, "nearest_magnet": 650.0,
               "air_gap_above_score": 0.4, "air_gap_below_score": 0.7, "vega_wall": 650.0,
               "per_dte_levels": {"D0": {"call_wall": 652.0}}, **levels}
    (d / "live_gamma_levels.json").write_text(json.dumps(payload))


def test_a_snapshot_taken_during_the_bar_is_used_by_that_bar(tmp_path):
    records = []
    engine = _engine(records)
    engine.options_provider = DealerSnapshotOptionsProvider(root=tmp_path)
    i = _warm(engine)
    _gamma_json(tmp_path, "QQQ", T0 + timedelta(minutes=i, seconds=48))  # mid-bar, as the poller writes
    engine.on_bar(_bar(i))
    ctx = engine.setups[SETUP_ID].options_context
    assert ctx["source"] == "dealer_positioning_static", ctx["warnings"]
    assert ctx["structure"]["air_gap_below_score"] == 0.7 and "per_dte_levels" in ctx["structure"]


def test_a_snapshot_after_the_bar_closed_is_still_rejected(tmp_path):
    provider = DealerSnapshotOptionsProvider(root=tmp_path)
    close = T0 + timedelta(minutes=1)
    _gamma_json(tmp_path, "QQQ", close + timedelta(seconds=5))
    assert provider.context("QQQ", close, 650.0).warnings == ("future_dealer_snapshot_rejected",)


def test_the_freshest_dealer_snapshot_supplies_the_scalars():
    class _Fixed:
        def __init__(self, ctx):
            self.ctx = ctx

        def context(self, *a):
            return self.ctx

    nightly = OptionsContext(timestamp=T0 - timedelta(hours=18), source="dealer_level_summary_static",
                             gamma_flip=640.0, call_wall=660.0)
    live = OptionsContext(timestamp=T0, source="dealer_positioning_static", gamma_flip=648.45, call_wall=652.0,
                          structure={"air_gap_above_score": 0.4})
    got = CompositeOptionsProvider(_Fixed(nightly), _Fixed(live)).context("QQQ", T0, 650.0)
    assert (got.gamma_flip, got.call_wall) == (648.45, 652.0)
    assert got.structure == {"air_gap_above_score": 0.4}


def test_archived_ladders_are_read_as_of_the_decision(tmp_path):
    snaps = tmp_path / "QQQ" / "snapshots"
    snaps.mkdir(parents=True)
    for minute, wall in ((0, 651.0), (5, 652.0)):
        stamp = T0 + timedelta(minutes=minute, seconds=48)
        strikes = [645.0, 648.0, 650.0, wall, 655.0]
        pd.DataFrame({
            "timestamp": stamp.isoformat(), "symbol": "QQQ", "strike": strikes, "spot": 650.0,
            "call_gex": [1, 2, 3, 50, 4], "put_gex": [-40, -3, -2, -1, -1],
            "net_gex": [-39, -1, 1, 49, 3], "abs_net_gex": [39, 1, 1, 49, 3], "total_vex": [1, 1, 5, 1, 1],
        }).to_csv(snaps / f"gamma_ladder_{stamp:%Y%m%dT%H%M%S.%f}_0000.csv", index=False)
    provider = ArchivedLadderOptionsProvider(root=tmp_path)
    assert provider.context("QQQ", T0 + timedelta(minutes=3), 650.0).call_wall == 651.0
    assert provider.context("QQQ", T0 + timedelta(minutes=6), 650.0).call_wall == 652.0
    assert provider.context("QQQ", T0, 650.0).source == "none", "nothing before the first ladder"


# --- recorder enrichment ------------------------------------------------------------

def test_recorder_adds_regime_and_news_only_from_before_the_decision(tmp_path):
    pd.DataFrame({
        "date": pd.to_datetime(["2026-11-20", "2026-11-23"]),
        "available_at": pd.to_datetime(["2026-11-20T21:30Z", "2026-11-23T21:30Z"]),
        "spy_trend_state": [-1.0, 1.0], "breadth_z": [-2.0, 1.0], "risk_appetite_z": [-0.9, 1.0],
        "credit_risk_z": [-2.6, 1.0], "spy_rv20_z": [3.7, 0.0],
    }).to_parquet(tmp_path / "regime.parquet")
    pd.DataFrame({
        "ticker": ["NVDA", "NVDA", "QQQ"],
        "timestamp": pd.to_datetime(["2026-11-23T12:00Z", "2026-11-23T16:30Z", "2026-11-23T13:00Z"]),
        "scored_at": pd.to_datetime(["2026-11-23T12:05Z", "2026-11-23T16:35Z", "2026-11-23T13:05Z"]),
        "headline": ["Nvidia CEO reinstated", "later news", "QQQ gaps up"],
        "catalyst_family": "company_news", "catalyst_subtype": "management_change",
        "catalyst_score": [0.92, 0.10, 0.70], "information_direction": "other",
    }).to_parquet(tmp_path / "news.parquet")
    stamps = pd.bdate_range(end="2026-11-20", periods=60).tz_localize("America/New_York").tz_convert("UTC")
    bars = tmp_path / "bars"
    bars.mkdir()
    pd.DataFrame({"timestamp": stamps, "close": [700.0] * 59 + [608.45]}).to_parquet(bars / "QQQ.parquet")
    policy = DecisionContextPolicy(root=str(tmp_path / "ctx"), regime_path=str(tmp_path / "regime.parquet"),
                                   news_path=str(tmp_path / "news.parquet"), universe_path=str(tmp_path / "none.csv"),
                                   daily_bars_dir=str(bars))
    recorder = DecisionContextRecorder(policy)
    bar_ts = datetime(2026, 11, 23, 16, 0, tzinfo=timezone.utc)
    recorder({"kind": "confirm", "ticker": "QQQ", "setup_id": SETUP_ID, "bar_ts": bar_ts.isoformat(),
              "decided_at": (bar_ts + timedelta(minutes=1)).isoformat(), "features": {"atr": float("nan")}})
    rows = read_session(tmp_path / "ctx", "2026-11-23", ticker="QQQ")
    assert len(rows) == 1
    r = rows[0]
    assert r["instrument_class"] == "index_etf"
    assert r["daily_regime"]["date"] == "2026-11-20" and r["daily_regime"]["bear_votes_of_5"] == 5
    assert r["daily_regime"]["qqq_below_sma50"] is True and r["daily_regime"]["qqq_prior_close"] == 608.45
    assert r["news"]["count"] == 1 and r["news"]["latest"]["headline"] == "QQQ gaps up"
    assert [n["headline"] for n in r["market_news"]] == ["Nvidia CEO reinstated", "QQQ gaps up"], \
        "the 16:35 headline was scored after the 16:01 decision"
    assert r["features"]["atr"] is None, "non-finite values are written as null"
    assert nearest_before(rows, bar_ts + timedelta(minutes=5))["setup_id"] == SETUP_ID


@pytest.mark.parametrize("ticker, etfs, expected", [
    ("SPY", frozenset(), "index_etf"), ("XLK", frozenset({"XLK"}), "etf"), ("NVDA", frozenset({"XLK"}), "single_stock"),
])
def test_instrument_class(ticker, etfs, expected):
    assert instrument_class(ticker, etfs) == expected


def test_a_snapshot_older_than_the_limit_is_not_used(tmp_path):
    """Friday's 15:59 snapshot is 2.7 days old at Monday's open; the 24h limit used to be a warning only."""
    provider = DealerSnapshotOptionsProvider(root=tmp_path, max_age_minutes=1440)
    friday_close = datetime(2026, 11, 20, 20, 59, 43, tzinfo=timezone.utc)
    _gamma_json(tmp_path, "QQQ", friday_close, call_wall=610.0, put_wall=600.0)
    got = provider.context("QQQ", datetime(2026, 11, 23, 14, 31, tzinfo=timezone.utc), 650.0)
    assert got.source == "none" and got.levels == () and got.warnings == ("dealer_snapshot_stale",)
    fresh = provider.context("QQQ", friday_close + timedelta(hours=17), 609.0)
    assert fresh.source == "dealer_positioning_static", "an overnight snapshot is inside the limit"
