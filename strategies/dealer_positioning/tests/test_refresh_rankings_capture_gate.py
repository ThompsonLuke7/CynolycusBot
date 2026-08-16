"""The rankings artifact must not be replaced from a degraded chain capture.

2026-08-07: a mid-capture DNS outage failed 1,820 of 2,790 tasks (65%). The 747
surviving summaries produced a 266-row ranking — against 737 the day before —
which was written straight over `dealer_swing_rankings_latest.parquet`. Nothing
downstream could distinguish it from a real ranking, so the module then ran its
target scan against a universe missing two thirds of its names.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from strategies.dealer_positioning import live_ranked_options as lro
from strategies.dealer_positioning.live_ranked_options import (
    DegradedCaptureError,
    refresh_rankings,
)


class _Result:
    def __init__(self, *, symbols: int, scopes: int, errors: int, summary_rows: int, output_dir: Path):
        self.symbols = symbols
        self.scopes = scopes
        self.errors = errors
        self.summary_rows = summary_rows
        self.strike_rows = summary_rows * 40
        self.output_dir = output_dir


def _install(monkeypatch, tmp_path, *, errors: int, ranking_rows: int,
             symbols: int = 1395, scopes: int = 2) -> Path:
    out_dir = tmp_path / "capture"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dealer_level_summary.parquet").write_text("")

    monkeypatch.setattr(lro, "load_symbols", lambda **_kw: ["AAA"] * symbols)
    monkeypatch.setattr(lro, "capture_snapshots", lambda **_kw: _Result(
        symbols=symbols, scopes=scopes, errors=errors,
        summary_rows=max(0, symbols * scopes - errors), output_dir=out_dir,
    ))
    # Only the capture summary is faked; real parquet reads (the previous
    # `latest`) must still go through pandas.
    real_read_parquet = pd.read_parquet

    def _read_parquet(path, *a, **k):
        if Path(path).name == "dealer_level_summary.parquet":
            return pd.DataFrame({"symbol": ["X"]})
        return real_read_parquet(path, *a, **k)

    monkeypatch.setattr(lro.pd, "read_parquet", _read_parquet)
    monkeypatch.setattr(lro, "build_rankings", lambda _frame: pd.DataFrame({
        "symbol": [f"S{i}" for i in range(ranking_rows)],
        "snapshot_date": ["2026-08-07"] * ranking_rows,
        "dealer_swing_rank": range(1, ranking_rows + 1),
    }))
    ranking_root = tmp_path / "rankings"
    ranking_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(lro, "RANKING_ROOT", ranking_root)
    return ranking_root


def _kwargs():
    return dict(workers=1, limit=None, sleep_seconds=0.0, snapshot_date=None, ref_date=None)


def test_bulk_capture_failure_refuses_to_publish(monkeypatch, tmp_path):
    # The 2026-08-07 shape: 65% of tasks failed.
    _install(monkeypatch, tmp_path, errors=1820, ranking_rows=266)
    with pytest.raises(DegradedCaptureError, match="degraded"):
        refresh_rankings(**_kwargs())


def test_previous_rankings_survive_a_degraded_capture(monkeypatch, tmp_path):
    ranking_root = _install(monkeypatch, tmp_path, errors=1820, ranking_rows=266)
    latest = ranking_root / "dealer_swing_rankings_latest.parquet"
    good = pd.DataFrame({"symbol": [f"G{i}" for i in range(737)]})
    good.to_parquet(latest, index=False)

    with pytest.raises(DegradedCaptureError):
        refresh_rankings(**_kwargs())

    # Untouched: a stale-but-complete ranking beats a fresh truncated one.
    assert list(pd.read_parquet(latest)["symbol"]) == list(good["symbol"])


def test_normal_error_rate_publishes(monkeypatch, tmp_path):
    # 2026-08-06 shape: 98/2790 errors (3.5%) is routine and must not block.
    ranking_root = _install(monkeypatch, tmp_path, errors=98, ranking_rows=737)
    latest = refresh_rankings(**_kwargs())
    assert latest.exists()
    assert len(pd.read_parquet(ranking_root / "dealer_swing_rankings_latest.parquet")) == 737


def test_row_collapse_refuses_even_when_error_rate_is_tolerable(monkeypatch, tmp_path):
    # Independent second gate: few reported errors, but the universe still
    # halved against the ranking already on disk.
    ranking_root = _install(monkeypatch, tmp_path, errors=50, ranking_rows=200)
    latest = ranking_root / "dealer_swing_rankings_latest.parquet"
    pd.DataFrame({"symbol": [f"G{i}" for i in range(737)]}).to_parquet(latest, index=False)

    with pytest.raises(DegradedCaptureError, match="collapsed"):
        refresh_rankings(**_kwargs())


def test_first_ever_run_publishes_without_a_previous_file(monkeypatch, tmp_path):
    ranking_root = _install(monkeypatch, tmp_path, errors=10, ranking_rows=300)
    assert not (ranking_root / "dealer_swing_rankings_latest.parquet").exists()
    refresh_rankings(**_kwargs())
    assert (ranking_root / "dealer_swing_rankings_latest.parquet").exists()
