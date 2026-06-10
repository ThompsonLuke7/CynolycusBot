"""Robustness tests for the unified catalyst layer."""

from __future__ import annotations

import pandas as pd

from signals.catalysts.pipeline import build_catalyst_records, news_to_catalysts


def test_news_to_catalysts_preserves_sec_form() -> None:
    """SEC 10-Q / 10-K / 13D filings must not all collapse to event_type=sec_8k."""
    news = pd.DataFrame(
        [
            {"record_id": "a", "ticker": "RKLB", "timestamp": "2026-01-01T14:00:00Z", "source": "sec_8-k", "headline": "8-K"},
            {"record_id": "b", "ticker": "RKLB", "timestamp": "2026-01-02T14:00:00Z", "source": "sec_10-q", "headline": "10-Q"},
            {"record_id": "c", "ticker": "RKLB", "timestamp": "2026-01-03T14:00:00Z", "source": "sec_10-k", "headline": "10-K"},
            {"record_id": "d", "ticker": "RKLB", "timestamp": "2026-01-04T14:00:00Z", "source": "sec_sc_13d", "headline": "13D"},
            {"record_id": "e", "ticker": "RKLB", "timestamp": "2026-01-05T14:00:00Z", "source": "finnhub", "headline": "News"},
        ]
    )
    out = news_to_catalysts(news)
    by_record = {row["record_id"]: row["event_type"] for _, row in out.iterrows()}
    assert by_record["a"] == "sec_8-k"
    assert by_record["b"] == "sec_10-q"
    assert by_record["c"] == "sec_10-k"
    assert by_record["d"] == "sec_sc_13d"
    assert by_record["e"] == "company_news"
    # catalyst_kind should still bucket SEC vs press wire.
    kind_by_record = {row["record_id"]: row["catalyst_kind"] for _, row in out.iterrows()}
    assert kind_by_record["a"] == "sec_filing"
    assert kind_by_record["e"] == "news"


def test_build_catalyst_records_handles_empty_inputs(tmp_path) -> None:
    """Empty news / events should not raise; should produce an empty parquet."""
    empty_news_path = tmp_path / "news.parquet"
    empty_macro_path = tmp_path / "macro.parquet"
    empty_earnings_path = tmp_path / "earnings.parquet"
    out_path = tmp_path / "out.parquet"
    pd.DataFrame(columns=["record_id", "ticker", "timestamp", "source", "headline"]).to_parquet(empty_news_path, index=False)
    pd.DataFrame(columns=["event_type", "timestamp", "title", "source", "ticker"]).to_parquet(empty_macro_path, index=False)
    pd.DataFrame(columns=["event_type", "timestamp", "title", "source", "ticker"]).to_parquet(empty_earnings_path, index=False)
    out = build_catalyst_records(
        news_path=empty_news_path,
        macro_path=empty_macro_path,
        earnings_path=empty_earnings_path,
        output_path=out_path,
    )
    assert out.empty


def run_all() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    test_news_to_catalysts_preserves_sec_form()
    with TemporaryDirectory() as directory:
        test_build_catalyst_records_handles_empty_inputs(Path(directory))
    print("catalysts robustness tests passed")


if __name__ == "__main__":
    run_all()
