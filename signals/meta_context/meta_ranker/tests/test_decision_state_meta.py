"""Meta archives every scored name, with gates that reproduce its own selection."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from core.decision_state_archive import load_decision_state
from signals.meta_context.meta_ranker import live_runner as lr
from signals.meta_context.meta_ranker.live_runner import (
    MetaRankingConfig,
    archive_meta_decision,
    meta_cross_section_gates,
    rank_meta_candidates_with_diagnostics,
)
from signals.meta_context.meta_ranker.tests.fixtures.meta_rows import (
    DECISION_BAR,
    fixture_booster_loader,
    meta_rows,
)


@pytest.mark.parametrize("top_k,combo_floor", [(5, 0.45), (2, 0.0), (1, 0.9), (10, 0.3)])
def test_top_k_gate_is_exactly_the_ranked_set(top_k, combo_floor):
    config = MetaRankingConfig(top_k=top_k, liquidity_floor=0.6, combo_floor=combo_floor,
                               blacklist=frozenset({"BLACK"}), booster_loader=fixture_booster_loader)
    result = rank_meta_candidates_with_diagnostics(meta_rows(), bar=DECISION_BAR, config=config)
    gates, rank = meta_cross_section_gates(result, config, entry_ok={})
    passed_all_but_quality = (gates["combo_floor"] & gates["liquidity_floor"]
                              & gates["not_blacklisted"] & gates["top_k"])
    picked = result.scored.loc[passed_all_but_quality].sort_values(by="ticker")["ticker"].tolist()
    assert picked == sorted(result.ranked["ticker"].tolist())
    ranked_order = result.scored.loc[rank.dropna().sort_values().index, "ticker"]
    assert ranked_order.head(top_k).tolist() == result.ranked["ticker"].tolist()
    assert int((gates["combo_floor"] & gates["liquidity_floor"] & gates["not_blacklisted"]).sum()) \
        == result.eligible_count


def test_archive_meta_decision_writes_the_scored_bar(monkeypatch, tmp_path):
    config = MetaRankingConfig(top_k=5, liquidity_floor=0.6, combo_floor=0.45,
                               blacklist=frozenset({"BLACK"}), booster_loader=fixture_booster_loader)
    result = rank_meta_candidates_with_diagnostics(meta_rows(), bar=DECISION_BAR, config=config)
    targets = result.ranked["ticker"].tolist()
    entry_ok = {t: i == 0 for i, t in enumerate(targets)}
    monkeypatch.setattr(lr, "meta_feature_names", lambda models_dir=None: ["required_feature", "up", "qual"])
    args = SimpleNamespace(matrix=str(tmp_path / "m.parquet"), flush_pending_open=False, submit=False,
                           mode="equity", quality_floor=0.4)
    archive_meta_decision(args, result=result, config=config, entry_ok=entry_ok, run_id="r" * 32,
                          inference_start=pd.Timestamp.now(tz="UTC"))
    df = load_decision_state(module="meta_ranker")
    assert len(df) == len(result.scored) and set(df["run_id"]) == {"r" * 32}
    assert df.loc[df["selected"], "ticker"].tolist() == [targets[0]]
    assert df.loc[df["ticker"] == "BLACK", "gate_reason"].iloc[0] == "not_blacklisted"
    assert df.loc[df["ticker"] == "LOWL", "g__liquidity_floor"].iloc[0] == False  # noqa: E712
    assert {"f__required_feature", "f__up", "f__qual", "x__s_upside", "x__s_quality"} <= set(df.columns)
    if len(targets) > 1:
        assert df.loc[df["ticker"] == targets[1], "action"].iloc[0] == "quality_gated"
    assert (df["bar_ts"] == DECISION_BAR).all()
