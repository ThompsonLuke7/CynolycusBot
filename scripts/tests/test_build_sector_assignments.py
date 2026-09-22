"""The sector-assignment builder's recurring-job safety properties.

This runs weekly and unattended, so the failure modes that matter are the
quiet ones: silently destroying the point-in-time history, or writing a
snapshot for a month that has not happened.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build_sector_assignments.py"
PYTHON = REPO / ".venv" / "bin" / "python"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PYTHON), "-u", str(SCRIPT), *args],
        cwd=REPO, capture_output=True, text=True, timeout=1800,
    )


def seed(path: Path, month: str, tickers: list[str]) -> None:
    pd.DataFrame([
        {
            "ticker": t,
            "snapshot_month": pd.Timestamp(month),
            "sector_etf": "XLK",
            "correlation": 0.5,
            "n_obs": 120,
            "resolver_sector_etf": "XLK",
            "curated": False,
        }
        for t in tickers
    ]).to_parquet(path)


@pytest.mark.slow
def test_a_future_as_of_is_refused_and_writes_nothing(tmp_path) -> None:
    """A premature snapshot would be served instead of a real one later.

    `empirical_sector_for` keys on (ticker, snapshot_month), so a row labelled
    for next month, built from this month's bars, silently wins the lookup when
    that month arrives.
    """

    out = tmp_path / "sa.parquet"
    future = (pd.Timestamp.today().normalize() + pd.DateOffset(months=2)).date()
    result = run("--as-of", str(future), "--out", str(out))
    assert "in the future" in (result.stdout + result.stderr)
    assert not out.exists(), "a refused run must not write a file"


@pytest.mark.slow
def test_an_already_cached_month_is_a_no_op(tmp_path) -> None:
    """The weekly caller must not pay two minutes to reproduce the same answer."""

    out = tmp_path / "sa.parquet"
    this_month = pd.Timestamp.today().normalize().replace(day=1)
    seed(out, str(this_month.date()), ["AAA", "BBB"])
    before = out.read_bytes()

    result = run("--out", str(out))
    assert result.returncode == 0
    assert "nothing to do" in (result.stdout + result.stderr)
    assert out.read_bytes() == before, "a skipped run must not rewrite the file"


@pytest.mark.slow
def test_earlier_months_survive_a_rebuild(tmp_path) -> None:
    """Earlier months ARE the point-in-time record; overwriting destroys it."""

    out = tmp_path / "sa.parquet"
    seed(out, "2020-01-01", ["OLDONE", "OLDTWO"])

    result = run("--out", str(out), "--force")
    assert result.returncode == 0, result.stderr[-2000:]

    frame = pd.read_parquet(out)
    months = set(frame["snapshot_month"].astype(str).str[:7])
    assert "2020-01" in months, "the earlier month was destroyed"
    assert set(frame[frame["snapshot_month"] == pd.Timestamp("2020-01-01")]["ticker"]) == {
        "OLDONE", "OLDTWO"
    }
    # And this month was actually written.
    this_month = pd.Timestamp.today().normalize().replace(day=1)
    assert (frame["snapshot_month"] == this_month).any()


@pytest.mark.slow
def test_force_replaces_only_the_current_month(tmp_path) -> None:
    out = tmp_path / "sa.parquet"
    this_month = pd.Timestamp.today().normalize().replace(day=1)
    seed(out, str(this_month.date()), ["STALE_PLACEHOLDER"])
    seed_prior = pd.read_parquet(out)

    result = run("--out", str(out), "--force")
    assert result.returncode == 0, result.stderr[-2000:]

    frame = pd.read_parquet(out)
    current = frame[frame["snapshot_month"] == this_month]
    assert "STALE_PLACEHOLDER" not in set(current["ticker"]), (
        "the current month must be replaced, not appended to"
    )
    assert len(current) > len(seed_prior)
