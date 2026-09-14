"""build_mixed_plan refuses NEW entries on names with a recent corporate action.

Why the screen exists: the momentum ranker selects for extreme recent price
action, and an unadjusted split or recapitalisation is indistinguishable from
it. In the 2022-2026 OOF study such names were 186x over-represented at rank 1.
The screen is injected here so these tests never depend on the real bar cache.
"""
from __future__ import annotations

import pandas as pd

from core import live_4h_exec
from core.live_4h_exec import ExecPolicy, build_mixed_plan, corporate_action_screen

_FLAG = {"session": "2026-09-01", "sessions_ago": 3, "gap_ratio": 132.05,
         "direction": "up", "volume_ratio": 0.04, "lookback_sessions": 20}


def _plan(targets, *, flagged=(), managed=None, pos_info=None):
    return build_mixed_plan(
        None, targets=list(targets), managed=managed or {}, pos_info=pos_info or {},
        bar="2026-09-10 14:00:00+00:00", signal_audits={}, policy=ExecPolicy(),
        route_fn=lambda *a, **k: ("equity", None, "ok"),
        ref_price_fn=lambda _t: 10.0, verbose=False,
        underlying_fn=lambda _t, _at=None: (None, None),
        corporate_action_fn=lambda t: dict(_FLAG) if t in flagged else None,
    )


def _buys(res):
    return [p[0] for p in res.plan if p[1] == "buy"]


def test_flagged_new_target_is_not_bought():
    res = _plan(["LGCL"], flagged={"LGCL"})
    assert _buys(res) == []
    sel = res.contract_selection["LGCL"]
    assert sel["action"] == "skip"
    assert sel["reason"] == "corporate_action_suspect"
    assert sel["corporate_action"]["gap_ratio"] == 132.05


def test_clean_targets_still_route_alongside_a_flagged_one():
    res = _plan(["LGCL", "AAA"], flagged={"LGCL"})
    assert _buys(res) == ["AAA"]


def test_a_held_position_is_not_touched_by_the_entry_screen():
    """Held names are the mark guard's job; the entry screen must neither sell
    nor drop them."""
    managed = {"LGCL": {"route": "equity", "symbol": "LGCL", "runs_held": 2,
                        "last_mark_price": 10.0}}
    res = _plan(["LGCL"], flagged={"LGCL"}, managed=managed,
                pos_info={"LGCL": {"qty": 50, "avg_entry": 10.0, "current": 10.1}})
    assert res.plan == []
    assert "LGCL" in res.new_managed
    assert "LGCL" not in res.contract_selection


def test_default_screen_is_used_when_none_is_injected(monkeypatch):
    seen = []
    monkeypatch.setattr(live_4h_exec, "corporate_action_screen",
                        lambda t: seen.append(t) or None)
    build_mixed_plan(
        None, targets=["AAA"], managed={}, pos_info={}, bar="2026-09-10",
        signal_audits={}, policy=ExecPolicy(),
        route_fn=lambda *a, **k: ("skip", None, "n/a"), ref_price_fn=lambda _t: 10.0,
        verbose=False, underlying_fn=lambda _t, _at=None: (None, None),
    )
    assert seen == ["AAA"]


def test_default_screen_reads_the_4h_cache(tmp_path):
    """End to end on a 4H file: a reverse split 3 sessions ago is reported, and a
    name with no cache file is clear (no data is not evidence of a split)."""
    days = pd.bdate_range("2026-07-01", periods=40, tz="America/New_York")
    ts, rows = [], []
    for i, d in enumerate(days):
        px, vol = (2.0, 1_000_000) if i < 37 else (30.0, 60_000)
        for h in (10, 14):
            ts.append((d + pd.Timedelta(hours=h)).tz_convert("UTC"))
            rows.append({"open": px, "high": px, "low": px, "close": px, "volume": vol})
    frame = pd.DataFrame(rows)
    frame.insert(0, "timestamp", ts)
    frame.to_parquet(tmp_path / "XYZ.parquet")
    hit = corporate_action_screen("XYZ", bars_dir=tmp_path)
    assert hit is not None and hit["sessions_ago"] == 2 and hit["direction"] == "up"
    assert corporate_action_screen("NOPE", bars_dir=tmp_path) is None
