"""HTF archives every row on its decision bar with gates matching its own selection."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import strategies.multi_ticker_swing_htf.live.runner as hr
from core.decision_state_archive import load_decision_state

BAR = pd.Timestamp("2026-11-12 19:00", tz="UTC")


def _bar_rows(seed: int, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    score = rng.random(n)
    score[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame({
        "timestamp": BAR,
        "ticker": [f"T{i:02d}" for i in range(n)],
        "htf_score": score,
        "dollar_vol_pctile_252": rng.random(n),
        "mom_score": rng.random(n),
        "feat_a": rng.normal(size=n),
    })


def _runner_selection(rows, *, floor, liq, blacklist, k):
    """The selection exactly as runner.main writes it."""
    cur = rows.dropna(subset=["htf_score"]).copy()
    cur["htf_rank_pct"] = cur["htf_score"].rank(pct=True)
    elig = cur[cur["htf_rank_pct"] >= floor]
    elig = elig[elig["dollar_vol_pctile_252"].fillna(0) >= liq]
    elig = elig[~elig["ticker"].str.upper().isin(blacklist)]
    return list(elig.sort_values("htf_score", ascending=False).head(k)["ticker"])


@pytest.mark.parametrize("seed", range(5))
def test_gates_reproduce_the_runner_selection(seed):
    rows = _bar_rows(seed)
    blacklist = {"T03", "T07"}
    targets = _runner_selection(rows, floor=0.5, liq=0.3, blacklist=blacklist, k=5)
    cand, gates = hr.htf_cross_section(rows, htf_rank_floor=0.5, liquidity_floor=0.3,
                                       blacklist=blacklist, targets=targets)
    pre_top = gates["has_score"] & gates["htf_rank_floor"] & gates["liquidity_floor"] & gates["not_blacklisted"]
    ranked = cand.loc[pre_top].sort_values("htf_live_rank")["ticker"].tolist()
    assert ranked[:5] == targets
    assert sorted(cand.loc[gates["top_k"], "ticker"]) == sorted(targets)


def test_archive_htf_decision_writes_every_bar_row(tmp_path):
    rows = _bar_rows(1)
    targets = _runner_selection(rows, floor=0.85, liq=0.6, blacklist=set(), k=10)
    args = SimpleNamespace(matrix=str(tmp_path / "m.parquet"), htf_rank_floor=0.85, liquidity_floor=0.6,
                           top_k=10, flush_pending_open=False, submit=False, mode="equity")
    hr.archive_htf_decision(args, bar_rows=rows, bar=BAR, blacklist=set(), targets=targets,
                            run_id="h" * 32, inference_start=pd.Timestamp.now(tz="UTC"))
    df = load_decision_state(module="multi_ticker_swing_htf")
    assert len(df) == len(rows)
    assert sorted(df.loc[df["selected"], "ticker"]) == sorted(targets)
    assert df.loc[df["score"].isna(), "gate_reason"].eq("has_score").all()
    assert {"f__feat_a", "f__dollar_vol_pctile_252", "x__mom_score", "x__htf_rank_pct"} <= set(df.columns)
    assert "f__htf_score" not in df.columns
