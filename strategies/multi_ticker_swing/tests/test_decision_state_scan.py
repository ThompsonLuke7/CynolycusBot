"""The 30m scanner archives every scored name; its returned signals are unchanged."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.decision_state_archive import load_decision_state
from strategies.multi_ticker_swing.live.ranker_scanner import RankerSwingScanner
from strategies.multi_ticker_swing.live.universe import TickerConfig

FEATS = ["fa", "fb", "fc"]
BAR = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=40)


class _FB:
    def __init__(self, rows):
        self.rows = rows

    def get_last_bar(self, t):
        # naive on purpose: the live builder can hand back naive UTC bars
        return {"timestamp": BAR.replace(tzinfo=None), "high": 10.0, "low": 9.0}

    def compute_latest_full(self, t):
        return self.rows[t]

    def get_atr(self, t):
        return 0.5


def _scanner(probs: dict[str, tuple[float, float]], max_entries=2) -> RankerSwingScanner:
    sc = RankerSwingScanner.__new__(RankerSwingScanner)
    rows = {t: pd.Series({"fa": i, "fb": -i, "extra_col": 1.0}) for i, t in enumerate(probs)}
    sc._fb = _FB(rows)
    sc._feats = FEATS
    sc._warned_missing = True
    sc._max_entries = max_entries
    sc._min_conviction = 0.30
    sc._catalyst = None
    sc._cat_alpha = 0.5
    cfg = dict(tier=1, entry_threshold=0.6, sl_atr=4.0, np_n_bars=None, np_mfe_atr=None,
               avg_win_pct=0.02, avg_loss_pct=-0.01, profit_factor=1.5, sharpe=1.0)
    sc._universe = {t: TickerConfig(ticker=t, **cfg) for t in probs}
    order = list(probs)
    sc._raw_p = lambda df: (np.array([probs[t][0] for t in df.index]),
                            np.array([probs[t][1] for t in df.index]))
    return sc


def test_scan_archives_every_scored_name():
    probs = {"AAA": (0.8, 0.1), "BBB": (0.1, 0.7), "CCC": (0.2, 0.25),  # CCC: no conviction
             "DDD": (0.31, 0.05), "EEE": (0.32, 0.9)}
    sc = _scanner(probs, max_entries=2)
    top = sc.scan(list(probs))
    df = load_decision_state(module="multi_ticker_swing_30m")
    assert len(df) == 5 and df["run_id"].nunique() == 1
    picked = sorted(s.ticker for s in top)
    assert sorted(df.loc[df["selected"], "ticker"]) == picked == ["AAA", "EEE"]
    row = df.set_index("ticker")
    assert row.loc["CCC", "gate_reason"] == "conviction"
    assert row.loc["BBB", "gate_reason"] == "top_k" and row.loc["BBB", "side"] == "short"
    assert row.loc["AAA", "x__p_long_win"] == 0.8
    # Model features archived (missing one recorded as NaN), builder extras are not.
    assert {"f__fa", "f__fb", "f__fc"} <= set(df.columns) and row["f__fc"].isna().all()
    assert "f__extra_col" not in df.columns
    assert (df["bar_ts"] == pd.Timestamp(BAR)).all()
    assert row.loc["AAA", "rank"] == 2 and row.loc["EEE", "rank"] == 1  # by ev_score


def test_archive_failure_does_not_change_the_scan(monkeypatch):
    import strategies.multi_ticker_swing.live.ranker_scanner as rs

    def boom(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(rs, "archive_decision_state", boom)
    probs = {"AAA": (0.8, 0.1), "EEE": (0.32, 0.9)}
    top = _scanner(probs).scan(list(probs))
    assert [s.ticker for s in top] == ["EEE", "AAA"]
