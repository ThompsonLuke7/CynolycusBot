from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from uuid import NAMESPACE_URL, UUID, uuid5

import pandas as pd
import pytest

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.enums import DecisionKind, Direction, InstrumentFamily
from signals.meta_context.meta_ranker.live_runner import (
    MetaRankingConfig,
    _ref_price,
    rank_meta_candidates,
)
from signals.meta_context.meta_ranker.nervous_system_adapter import (
    MetaIntentConfig,
    build_trade_intents,
)
from signals.meta_context.meta_ranker.tests.fixtures.meta_rows import (
    DECISION_BAR,
    DECISION_TIME,
    fixture_booster_loader,
    meta_rows,
)


EXPECTED_TICKERS = ["HELD", "AAA", "BBB", "MISS"]
EXPECTED_COMPONENTS = {
    "HELD": (0.75, 1.0, 0.35),
    "AAA": (0.5178571428571428, 0.8, 0.2),
    "BBB": (0.5178571428571428, 0.8, 0.2),
    "MISS": (0.47619047619047616, 0.4, 0.6),
}


def _ranking_config(**updates: object) -> MetaRankingConfig:
    payload: dict[str, object] = {
        "top_k": 5,
        "liquidity_floor": 0.6,
        "combo_floor": 0.45,
        "blacklist": frozenset({"BLACK"}),
        "booster_loader": fixture_booster_loader,
    }
    payload.update(updates)
    return MetaRankingConfig(**payload)


def _intent_config(**updates: object) -> MetaIntentConfig:
    payload: dict[str, object] = {
        "quality_floor": 0.4,
        "held_tickers": frozenset({"HELD"}),
        "requested_notional": Decimal("5000"),
        "model_version": "meta-combo@task14",
        "feature_version": "meta-matrix@task14",
        "config_version": "meta-intent@task14",
        "instrument_preferences": (
            InstrumentFamily.EQUITY,
            InstrumentFamily.SINGLE_OPTION,
            InstrumentFamily.VERTICAL,
            InstrumentFamily.CALENDAR,
            InstrumentFamily.DIAGONAL,
        ),
    }
    payload.update(updates)
    return MetaIntentConfig(**payload)


def _ranked() -> pd.DataFrame:
    return rank_meta_candidates(
        meta_rows(),
        bar=DECISION_BAR,
        config=_ranking_config(),
    )


def test_rank_meta_candidates_matches_frozen_exact_bar_selection_and_ties() -> None:
    ranked = _ranked()

    assert ranked["ticker"].tolist() == EXPECTED_TICKERS
    assert ranked["timestamp"].eq(DECISION_BAR).all()
    assert ranked["close"].tolist() == [303.0, 101.0, 202.0, 606.0]
    for ticker, (combo, upside, quality) in EXPECTED_COMPONENTS.items():
        row = ranked.loc[ranked["ticker"] == ticker].iloc[0]
        assert row["s_combo"] == pytest.approx(combo)
        assert row["s_upside"] == pytest.approx(upside)
        assert row["s_quality"] == pytest.approx(quality)


def test_rank_meta_candidates_requires_the_supplied_bar_and_never_uses_latest() -> None:
    later = rank_meta_candidates(
        meta_rows(),
        bar=DECISION_BAR + pd.Timedelta(hours=4),
        config=_ranking_config(),
    )
    assert later["ticker"].tolist() == ["AAA"]
    assert later["close"].tolist() == [999.0]

    with pytest.raises(ValueError, match="bar|timestamp"):
        rank_meta_candidates(
            meta_rows(),
            bar=DECISION_BAR + pd.Timedelta(hours=8),
            config=_ranking_config(),
        )


def test_build_trade_intents_preserves_quality_gate_for_new_names_and_held_names() -> None:
    ranked = _ranked()
    snapshots = {
        ticker: UUID(int=index + 1)
        for index, ticker in enumerate(EXPECTED_TICKERS)
    }

    intents = build_trade_intents(
        ranked,
        decision_time=DECISION_TIME,
        decision_bar=DECISION_BAR.to_pydatetime(),
        snapshot_id_by_ticker=snapshots,
        config=_intent_config(),
    )

    assert len(intents) == 1
    intent = intents[0]
    assert intent.strategy_id == "meta_ranker"
    assert intent.ticker == "MISS"
    assert intent.direction is Direction.LONG
    assert intent.decision_kind is DecisionKind.ENTRY
    assert intent.raw_score == pytest.approx(EXPECTED_COMPONENTS["MISS"][0])
    assert intent.raw_probability is None
    assert intent.snapshot_id == snapshots["MISS"]
    assert intent.selected_bar == DECISION_BAR.to_pydatetime()
    assert intent.feature_timestamp == DECISION_BAR.to_pydatetime()
    assert intent.created_at == DECISION_TIME
    assert intent.preferred_entry == Decimal("606.0")
    assert intent.position_size_requested == Decimal("5000")
    assert intent.model_version == "meta-combo@task14"
    assert intent.feature_version == "meta-matrix@task14"
    assert intent.config_version == "meta-intent@task14"
    assert intent.instrument_preferences == _intent_config().instrument_preferences
    assert intent.score_components == {
        "s_combo": pytest.approx(EXPECTED_COMPONENTS["MISS"][0]),
        "s_upside": pytest.approx(EXPECTED_COMPONENTS["MISS"][1]),
        "s_quality": pytest.approx(EXPECTED_COMPONENTS["MISS"][2]),
    }
    assert intent.reason_codes
    assert intent.idempotency_key
    assert intent.intent_id == uuid5(NAMESPACE_URL, intent.idempotency_key)


def test_build_trade_intents_is_byte_identical_on_rebuild_and_uses_canonical_uuidv5_key() -> None:
    ranked = _ranked()
    snapshots = {ticker: uuid5(NAMESPACE_URL, f"snapshot:{ticker}") for ticker in EXPECTED_TICKERS}
    config = _intent_config()

    first = build_trade_intents(
        ranked,
        decision_time=DECISION_TIME,
        decision_bar=DECISION_BAR.to_pydatetime(),
        snapshot_id_by_ticker=snapshots,
        config=config,
    )
    second = build_trade_intents(
        ranked.copy(deep=True),
        decision_time=DECISION_TIME,
        decision_bar=DECISION_BAR.to_pydatetime(),
        snapshot_id_by_ticker=snapshots,
        config=config,
    )

    assert first[0].model_dump_json() == second[0].model_dump_json()
    assert content_hash(first[0]) == content_hash(second[0])

    material = {
        "strategy_id": "meta_ranker",
        "decision_bar": DECISION_BAR.isoformat(),
        "ticker": "MISS",
        "side": "LONG",
        "config_version": "meta-intent@task14",
        "intent_ordinal": 1,
    }
    expected_key = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert first[0].idempotency_key == expected_key
    assert first[0].intent_id == uuid5(NAMESPACE_URL, expected_key)


def test_missing_model_feature_keeps_existing_fail_closed_manifest_behavior() -> None:
    frame = meta_rows().drop(columns=["required_feature"])
    with pytest.raises(KeyError, match="missing|features"):
        rank_meta_candidates(frame, bar=DECISION_BAR, config=_ranking_config())


def test_nonfinite_combo_is_omitted_by_existing_selection_comparison() -> None:
    ranked = _ranked()
    assert "NFIN" not in ranked["ticker"].tolist()


def test_compatibility_reference_price_requires_exact_selected_bar(tmp_path, monkeypatch) -> None:
    bars = tmp_path / "AAA.parquet"
    pd.DataFrame(
        {
            "timestamp": [DECISION_BAR, DECISION_BAR + pd.Timedelta(hours=4)],
            "close": [101.0, 999.0],
        }
    ).to_parquet(bars)
    monkeypatch.setattr("signals.meta_context.meta_ranker.live_runner.BARS_4H", tmp_path)

    assert _ref_price("AAA", decision_bar=DECISION_BAR) == 101.0

    duplicate = pd.DataFrame(
        {
            "timestamp": [DECISION_BAR, DECISION_BAR],
            "close": [101.0, 102.0],
        }
    )
    duplicate.to_parquet(bars)
    with pytest.raises(ValueError, match="duplicate|exact"):
        _ref_price("AAA", decision_bar=DECISION_BAR)

    absent = pd.DataFrame(
        {"timestamp": [DECISION_BAR + pd.Timedelta(hours=4)], "close": [999.0]}
    )
    absent.to_parquet(bars)
    with pytest.raises(ValueError, match="absent|exact|match"):
        _ref_price("AAA", decision_bar=DECISION_BAR)

    nonfinite = pd.DataFrame({"timestamp": [DECISION_BAR], "close": [float("nan")]})
    nonfinite.to_parquet(bars)
    with pytest.raises(ValueError, match="finite"):
        _ref_price("AAA", decision_bar=DECISION_BAR)
