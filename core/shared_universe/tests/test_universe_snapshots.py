"""Point-in-time universe snapshots: written on every build, never overwritten,
and read back without look-ahead."""
from __future__ import annotations

import pandas as pd
import pytest

from core.shared_universe import universe as uni


def _df(tickers, eligible=True):
    return pd.DataFrame({"ticker": tickers, "is_eligible": [eligible] * len(tickers)})


def test_every_build_writes_a_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(uni, "PENDING_TICKERS_CSV", tmp_path / "missing.csv")
    out_path = tmp_path / "shared_universe.csv"
    built = uni.build_shared_universe(out_path=out_path)
    snaps = list((tmp_path / "snapshots").glob("shared_universe_*.csv.gz"))
    assert len(snaps) == 1
    back = pd.read_csv(snaps[0])
    assert list(back["ticker"]) == list(built["ticker"])
    assert "snapshot_utc" in back.columns


def test_a_snapshot_is_never_overwritten(tmp_path):
    uni.write_universe_snapshot(_df(["AAA"]), snapshot_dir=tmp_path, now="2026-09-10T22:00:00Z")
    with pytest.raises(FileExistsError):
        uni.write_universe_snapshot(_df(["BBB"]), snapshot_dir=tmp_path,
                                    now="2026-09-10T22:00:00Z")


def test_as_of_returns_the_latest_snapshot_not_after_the_date(tmp_path):
    uni.write_universe_snapshot(_df(["OLD"]), snapshot_dir=tmp_path, now="2026-09-01T22:00:00Z")
    uni.write_universe_snapshot(_df(["MID"]), snapshot_dir=tmp_path, now="2026-09-05T22:00:00Z")
    uni.write_universe_snapshot(_df(["NEW"]), snapshot_dir=tmp_path, now="2026-09-09T22:00:00Z")
    got = uni.load_universe_as_of("2026-09-07", snapshot_dir=tmp_path)
    assert list(got["ticker"]) == ["MID"]
    assert list(uni.load_universe_as_of("2026-09-30", snapshot_dir=tmp_path)["ticker"]) == ["NEW"]


def test_as_of_before_the_first_snapshot_refuses_rather_than_using_today(tmp_path):
    uni.write_universe_snapshot(_df(["AAA"]), snapshot_dir=tmp_path, now="2026-09-10T22:00:00Z")
    with pytest.raises(LookupError):
        uni.load_universe_as_of("2025-01-01", snapshot_dir=tmp_path)


def test_eligible_only_filters(tmp_path):
    df = pd.DataFrame({"ticker": ["IN", "OUT"], "is_eligible": [True, False]})
    uni.write_universe_snapshot(df, snapshot_dir=tmp_path, now="2026-09-10T22:00:00Z")
    assert list(uni.load_universe_as_of("2026-09-11", snapshot_dir=tmp_path)["ticker"]) == ["IN"]
    assert len(uni.load_universe_as_of("2026-09-11", snapshot_dir=tmp_path,
                                       eligible_only=False)) == 2
