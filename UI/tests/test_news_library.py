"""Index build + query behaviour for the Library's news/catalyst search index.

Builds a miniature version of every source store into a tmp tree and repoints
the module's paths at it, so the tests exercise the real dedupe/join/sort logic
without touching the 425MB production record store.
"""
from __future__ import annotations

import pandas as pd
import pytest

from UI import news_library


@pytest.fixture()
def library(tmp_path, monkeypatch):
    """Miniature source tree wired into the module's module-level paths."""
    proc = tmp_path / "processed"
    proc.mkdir()

    ts = pd.Timestamp("2026-07-20 14:00", tz="UTC")

    news = pd.DataFrame({
        "record_id": ["r1", "r2", "r3"],
        "ticker": ["AAPL", "aapl", "NVDA"],
        "timestamp": [ts, ts + pd.Timedelta(days=1), ts],
        "headline": ["Apple ships a widget", "Apple beats on EPS", "Nvidia FDA nonsense"],
        "source": ["google_news_rss", "yfinance", "google_news_rss"],
        "url": ["u1", "u2", "u3"],
        "catalyst_family": ["company_news", "earnings_guidance", "company_news"],
        "catalyst_subtype": ["product", "eps_beat", None],
        "relation_type": ["direct_mention", "direct_mention", "ambiguous"],
        "impact_role": ["company_specific", "earnings_beat", "unknown"],
        "source_quality": ["high_alpha", "high_alpha", "aggregator"],
        "is_direct_catalyst": [1.0, 1.0, 0.0],
        "relation_confidence": [0.9, 0.95, 0.3],
        "content_hash": ["h1", "h2", "h3"],
    })
    news.to_parquet(proc / "news_records.parquet", index=False)

    # One live row duplicates the canonical h1; the other is genuinely new.
    live = pd.DataFrame({
        "record_id": ["L1", "L2"],
        "ticker": ["AAPL", "AAPL"],
        "timestamp": [ts, ts + pd.Timedelta(days=2)],
        "headline": ["Apple ships a widget", "Apple breaking intraday"],
        "source": ["finnhub", "finnhub"],
        "catalyst_family": ["company_news", "company_news"],
        "catalyst_subtype": ["product", "product"],
        "catalyst_score": [0.4, 0.77],
        "scored_at": [ts, ts],
        "content_hash": ["h1", "hL2"],
    })
    live.to_parquet(proc / "live_catalyst_records.parquet", index=False)

    cat_proc = tmp_path / "catalyst_processed"
    cat_proc.mkdir()
    # c1 shadows news r1 (join-only); c2 exists nowhere else (appended as a row).
    catalysts = pd.DataFrame({
        "catalyst_id": ["c1", "c2"],
        "record_id": ["r1", "cOnly"],
        "ticker": ["AAPL", "AAPL"],
        "timestamp": [ts, ts + pd.Timedelta(days=3)],
        "headline": ["Apple ships a widget", "Scheduled Apple earnings"],
        "summary": ["s", "s"],
        "source": ["google_news_rss", "calendar"],
        "url": ["u1", "u4"],
        "catalyst_kind": ["news", "scheduled"],
        "event_type": ["product_launch", "earnings_date"],
        "relation_type": ["direct_mention", "direct_mention"],
        "impact_role": ["company_specific", "earnings_result"],
        "relation_confidence": [0.9, 1.0],
        "is_direct_catalyst": [1.0, 1.0],
    })
    catalysts.to_parquet(cat_proc / "catalyst_records.parquet", index=False)
    pd.DataFrame({
        "record_id": ["r1", "cOnly"], "catalyst_score": [0.61, 0.82],
    }).to_parquet(cat_proc / "catalyst_scores.parquet", index=False)
    # Per-record output of the DEPLOYED catalyst model. r1 reads bullish,
    # r2 bearish, r3 neutral.
    pd.DataFrame({
        "record_id": ["r1", "r2", "r3"],
        "catalyst_score": [0.81, 0.44, 0.52],
        "p_bull_steady": [0.40, 0.05, 0.10],
        "p_bull_volatile": [0.25, 0.05, 0.15],
        "p_v_bounce": [0.10, 0.10, 0.30],
        "p_crash_stayed": [0.05, 0.55, 0.20],
        "p_flat": [0.20, 0.25, 0.25],
    }).to_parquet(proc / "news_catalyst_per_record.parquet", index=False)

    # FinBERT tone rides with the embeddings, not the feature matrix.
    pd.DataFrame({
        "record_id": ["r1", "r2", "r3"],
        "finbert_positive_score": [0.80, 0.05, 0.20],
        "finbert_negative_score": [0.05, 0.85, 0.25],
        "finbert_neutral_score": [0.15, 0.10, 0.55],
    }).to_parquet(proc / "news_embeddings.parquet", index=False)

    # Forward returns (hindsight). r1 rips, r2 dumps, r3 is flat; L2 unlabelled.
    pd.DataFrame({
        "record_id": ["r1", "r2", "r3"],
        "forward_1d_return": [0.085, -0.043, 0.004],
        "forward_5d_return": [0.120, -0.061, 0.009],
        "max_forward_return": [0.150, 0.011, 0.020],
        "max_drawdown": [-0.010, -0.090, -0.015],
    }).to_parquet(proc / "news_labels.parquet", index=False)

    # The per-(ticker, day) score the live meta ranker trades on.
    sig_dir = tmp_path / "meta_processed"
    sig_dir.mkdir()
    pd.DataFrame({
        "timestamp": [ts.normalize(), (ts + pd.Timedelta(days=1)).normalize()],
        "ticker": ["AAPL", "AAPL"],
        "news_catalyst_score": [0.71, 0.34],
    }).to_parquet(sig_dir / "news_catalyst_signal.parquet", index=False)

    bars_dir = tmp_path / "bars_1d"
    bars_dir.mkdir()
    days = pd.date_range("2026-07-14", periods=10, freq="D", tz="UTC")
    pd.DataFrame({
        "timestamp": days,
        "open": range(100, 110), "high": range(101, 111),
        "low": range(99, 109), "close": range(100, 110),
        "volume": [1_000.0] * 10,
    }).to_parquet(bars_dir / "AAPL.parquet", index=False)

    paths = {
        "NEWS_RECORDS": proc / "news_records.parquet",
        "LIVE_RECORDS": proc / "live_catalyst_records.parquet",
        "CATALYST_RECORDS": cat_proc / "catalyst_records.parquet",
        "NEWS_LABELS": proc / "news_labels.parquet",
        "NEWS_CATALYST_PER_RECORD": proc / "news_catalyst_per_record.parquet",
        "NEWS_EMBEDDINGS": proc / "news_embeddings.parquet",
        "NEWS_CATALYST_SIGNAL": sig_dir / "news_catalyst_signal.parquet",
        "INDEX_PATH": tmp_path / "index.parquet",
        "STAMP_PATH": tmp_path / "index.json",
        "BARS_1D_DIR": bars_dir,
    }
    for name, value in paths.items():
        monkeypatch.setattr(news_library, name, value)
    monkeypatch.setattr(news_library, "_SOURCES", {
        "news_records": paths["NEWS_RECORDS"],
        "live_catalyst_records": paths["LIVE_RECORDS"],
        "catalyst_records": paths["CATALYST_RECORDS"],
        "news_labels": paths["NEWS_LABELS"],
        "news_catalyst_signal": paths["NEWS_CATALYST_SIGNAL"],
        "news_catalyst_per_record": paths["NEWS_CATALYST_PER_RECORD"],
        "news_embeddings": paths["NEWS_EMBEDDINGS"],
    })
    return news_library


def test_build_dedupes_live_against_canonical(library):
    stamp = library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH)

    # 3 news + 1 non-duplicate live + 1 catalyst-only = 5. The live row sharing
    # content_hash h1 with canonical r1 must NOT produce a second row.
    assert stamp["rows"] == 5
    assert len(df) == 5
    assert (df["headline"] == "Apple ships a widget").sum() == 1
    assert sorted(df["origin"].unique()) == ["catalyst", "live", "news"]
    assert set(df["record_id"]) == {"r1", "r2", "r3", "L2", "cOnly"}


def test_build_joins_scores_and_catalyst_fields(library):
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH).set_index("record_id")

    # catalyst_kind/event_type joined onto the canonical news row, not appended.
    assert df.loc["r1", "catalyst_kind"] == "news"
    assert df.loc["r1", "event_type"] == "product_launch"
    # The deployed model's per-record score, not the retired similarity score.
    assert df.loc["r1", "record_catalyst_score"] == pytest.approx(0.81)
    # Unscored rows stay null — the index never imputes a score.
    assert pd.isna(df.loc["L2", "record_catalyst_score"])


def test_retired_similarity_branch_is_not_read(library):
    """news_scores/catalyst_scores come from build_news_similarity_scores(), which
    froze on 2026-05-22 and which NOTHING consumes — the deployed booster has no
    similarity feature. They must not reappear as index columns or as sources
    (being a source would also make the index rebuild whenever they are touched)."""
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH)
    for gone in ("news_similarity_score", "realized_news_score", "catalyst_score",
                 "news_score"):
        assert gone not in df.columns
    assert "news_scores" not in library._SOURCES
    assert "catalyst_scores" not in library._SOURCES


def test_build_normalizes_ticker_case(library):
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH)
    assert set(df["ticker"]) == {"AAPL", "NVDA"}
    assert library.search(ticker="aapl")["total"] == 4


def test_search_orders_most_recent_first(library):
    rows = library.search(ticker="AAPL")["rows"]
    stamps = [r["timestamp"] for r in rows]
    assert stamps == sorted(stamps, reverse=True)
    assert rows[0]["headline"] == "Scheduled Apple earnings"


def test_search_filters(library):
    assert library.search(query="fda")["total"] == 1
    assert library.search(query="FDA")["rows"][0]["ticker"] == "NVDA"
    assert library.search(ticker="AAPL", family="earnings_guidance")["total"] == 1
    assert library.search(ticker="AAPL", origin="live")["total"] == 1
    assert library.search(relation="ambiguous")["total"] == 1
    assert library.search(ticker="AAPL", source="finnhub")["total"] == 1


def test_search_date_window_is_inclusive(library):
    out = library.search(ticker="AAPL", start="2026-07-21", end="2026-07-22T23:59:59Z")
    assert [r["headline"] for r in out["rows"]] == [
        "Apple breaking intraday", "Apple beats on EPS",
    ]


def test_search_paginates(library):
    first = library.search(ticker="AAPL", limit=2, offset=0)
    second = library.search(ticker="AAPL", limit=2, offset=2)
    assert first["total"] == second["total"] == 4
    assert len(first["rows"]) == 2 and len(second["rows"]) == 2
    assert {r["record_id"] for r in first["rows"]} & {r["record_id"] for r in second["rows"]} == set()


def test_stale_index_rebuilds(library):
    library.build_index(force=True)
    assert library.index_is_current()

    df = pd.read_parquet(library.NEWS_RECORDS)
    df.loc[len(df)] = {
        **df.iloc[0].to_dict(), "record_id": "r9", "content_hash": "h9",
        "headline": "Apple does something new",
    }
    df.to_parquet(library.NEWS_RECORDS, index=False)

    assert not library.index_is_current()
    assert library.search(query="something new")["total"] == 1


def test_facets_are_ticker_scoped(library):
    library.build_index(force=True)
    assert "ambiguous" not in library.facets("AAPL")["relation_type"]
    assert "ambiguous" in library.facets()["relation_type"]


def test_tickers_suggestions(library):
    library.build_index(force=True)
    assert library.tickers() == ["AAPL", "NVDA"]
    assert library.tickers("NV") == ["NVDA"]


def test_predicted_direction_uses_the_pipeline_thresholds(library):
    """bullish/bearish must use build_news_signal.py's own thresholds, so the
    library's call matches the meta signal's bull/bear alignment counts."""
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH).set_index("record_id")
    # r1: bull 0.40+0.25 = 0.65 >= 0.50 -> bullish
    assert df.loc["r1", "predicted_direction"] == "bullish"
    assert df.loc["r1", "p_bullish"] == pytest.approx(0.65)
    # r2: p_crash_stayed 0.55 >= 0.30 -> bearish (bear check wins)
    assert df.loc["r2", "predicted_direction"] == "bearish"
    # r3: bull 0.25 < 0.50 and crash 0.20 < 0.30 -> neutral
    assert df.loc["r3", "predicted_direction"] == "neutral"
    # Unscored record gets no direction rather than a default.
    assert pd.isna(df.loc["L2", "predicted_direction"])
    assert pd.isna(df.loc["L2", "p_bullish"])


def test_predicted_direction_is_not_hindsight(library):
    """Direction comes from the trajectory MODEL, so it must stay out of the
    hindsight zone — it is knowable at publication time."""
    assert "predicted_direction" not in library.HINDSIGHT_COLUMNS
    assert "tone" not in library.HINDSIGHT_COLUMNS
    assert "record_catalyst_score" not in library.HINDSIGHT_COLUMNS


def test_tone_is_the_finbert_argmax(library):
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH).set_index("record_id")
    assert df.loc["r1", "tone"] == "positive"   # .80 / .05 / .15
    assert df.loc["r2", "tone"] == "negative"   # .05 / .85 / .10
    assert df.loc["r3", "tone"] == "neutral"    # .20 / .25 / .55
    # No FinBERT row -> null, never a defaulted "neutral" (an all-NA argmax also
    # raises on newer pandas, so this pins the guard too).
    assert pd.isna(df.loc["L2", "tone"])


def test_forward_move_columns_join_per_record(library):
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH).set_index("record_id")
    assert df.loc["r1", "move_1d_pct"] == pytest.approx(8.5)      # stored as %
    assert df.loc["r2", "move_1d_pct"] == pytest.approx(-4.3)
    assert df.loc["r1", "max_favorable_pct"] == pytest.approx(15.0)
    assert df.loc["r2", "max_adverse_pct"] == pytest.approx(-9.0)
    # An unlabelled record stays null rather than being imputed to zero.
    assert pd.isna(df.loc["L2", "move_1d_pct"])


def test_outcome_classifies_on_the_realized_move(library):
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH).set_index("record_id")
    assert df.loc["r1", "outcome"] == "good"      # +8.5% > +2% band
    assert df.loc["r2", "outcome"] == "bad"       # -4.3% < -2% band
    assert df.loc["r3", "outcome"] == "neutral"   # +0.4% inside the band
    # No move -> no verdict. Defaulting to "neutral" would invent a claim.
    assert pd.isna(df.loc["L2", "outcome"])


def test_outcome_band_boundaries(library, monkeypatch):
    """A move exactly on the band edge counts as good/bad, not neutral."""
    lab = pd.read_parquet(library.NEWS_LABELS)
    lab.loc[lab.record_id == "r1", "forward_1d_return"] = 0.02    # exactly +2%
    lab.loc[lab.record_id == "r2", "forward_1d_return"] = -0.02   # exactly -2%
    lab.to_parquet(library.NEWS_LABELS, index=False)
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH).set_index("record_id")
    assert df.loc["r1", "outcome"] == "good"
    assert df.loc["r2", "outcome"] == "bad"


def test_move_rank_is_percentile_within_the_ticker(library):
    """Ranking per ticker, not globally: a 3% move means different things for a
    biotech and a mega-cap, so a global rank would mostly encode volatility."""
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH).set_index("record_id")
    # AAPL's labelled moves are r1 (+8.5) and r2 (-4.3) -> r1 ranks top.
    assert df.loc["r1", "move_rank_pct"] == pytest.approx(100.0)
    assert df.loc["r2", "move_rank_pct"] == pytest.approx(50.0)
    # NVDA's only labelled move is r3, so it is the top of its own ticker.
    assert df.loc["r3", "ticker"] == "NVDA"
    assert df.loc["r3", "move_rank_pct"] == pytest.approx(100.0)


def test_day_catalyst_score_joins_on_the_et_session(library):
    """The live meta-ranker score is per (ticker, DAY) and must land on every
    record of that ET session — including a late-evening UTC headline."""
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH).set_index("record_id")
    assert df.loc["r1", "day_catalyst_score"] == pytest.approx(0.71)
    assert df.loc["r2", "day_catalyst_score"] == pytest.approx(0.34)
    # NVDA has no signal row -> null, never back-filled from another ticker.
    assert pd.isna(df.loc["r3", "day_catalyst_score"])


def test_hindsight_columns_are_declared(library):
    """Every after-the-fact column must be in HINDSIGHT_COLUMNS so the UI can
    fence them off; a new one added without declaring it is a leakage risk."""
    library.build_index(force=True)
    df = pd.read_parquet(library.INDEX_PATH)
    for col in library.HINDSIGHT_COLUMNS:
        assert col in df.columns
    assert "day_catalyst_score" not in library.HINDSIGHT_COLUMNS


def test_event_days_buckets_by_et_session(library):
    library.build_index(force=True)
    out = library.event_days(ticker="AAPL")
    days = out["days"]
    assert [d["day"] for d in days] == ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"]
    assert all(d["count"] == 1 for d in days)
    assert days[0]["headlines"][0]["headline"] == "Apple ships a widget"
    assert out["total"] == 4


def test_event_days_caps_coloured_classes_and_folds_the_rest(library):
    """Marks are dots, compared all-pairs: only 3 hues clear the CVD/normal
    floors on this surface, so the 4th+ class must fold into 'Other'."""
    library.build_index(force=True)
    out = library.event_days(ticker="AAPL", color_by="source", max_classes=2)
    names = [c["name"] for c in out["classes"]]
    assert len(names) == 3 and names[-1] == library.OTHER_CLASS
    # Counts still add up to every matching record — folding must not drop any.
    assert sum(c["count"] for c in out["classes"]) == out["total"] == 4
    assert all(d["klass"] in names for d in out["days"])


def test_event_days_reports_a_mixed_day_breakdown(library):
    """A day's marker takes its dominant class, but the breakdown keeps the mix
    so a mixed day is never misread as pure."""
    library.build_index(force=True)
    out = library.event_days(ticker="AAPL", color_by="catalyst_family", max_classes=3)
    by_day = {d["day"]: d for d in out["days"]}
    assert by_day["2026-07-21"]["breakdown"] == {"earnings_guidance": 1}
    assert sum(sum(d["breakdown"].values()) for d in out["days"]) == out["total"]


def test_event_days_rejects_unknown_color_by(library):
    library.build_index(force=True)
    out = library.event_days(ticker="AAPL", color_by="not_a_column")
    assert out["color_by"] == "catalyst_family"


def test_event_days_covers_every_match_not_just_a_page(library, monkeypatch):
    """The overlay must aggregate the whole match set.

    Counting a capped page instead would drop older news days off the chart, so
    a long-range plot of a heavily-covered name would read as "no news before
    <recent date>". Build a set larger than MAX_LIMIT and assert the oldest day
    still gets a marker.
    """
    ts = pd.Timestamp("2026-01-05 15:00", tz="UTC")
    n = library.MAX_LIMIT + 25
    pd.DataFrame({
        "record_id": [f"b{i}" for i in range(n)],
        "ticker": ["BULK"] * n,
        "timestamp": [ts + pd.Timedelta(hours=i) for i in range(n)],
        "headline": [f"bulk story {i}" for i in range(n)],
        "source": ["google_news_rss"] * n,
        "url": [""] * n,
        "catalyst_family": ["company_news"] * n,
        "catalyst_subtype": [None] * n,
        "relation_type": ["direct_mention"] * n,
        "impact_role": ["company_specific"] * n,
        "source_quality": ["high_alpha"] * n,
        "is_direct_catalyst": [1.0] * n,
        "relation_confidence": [0.9] * n,
        "content_hash": [f"bh{i}" for i in range(n)],
    }).to_parquet(library.NEWS_RECORDS, index=False)

    out = library.event_days(ticker="BULK")
    days = out["days"]
    assert sum(d["count"] for d in days) == n == out["total"]
    # A page-based overlay would return only the most recent MAX_LIMIT records,
    # losing the earliest session entirely.
    assert days[0]["day"] == "2026-01-05"
    assert library.search(ticker="BULK")["total"] == n


def test_price_series_reads_shared_bars(library):
    out = library.price_series("AAPL", days=0)
    assert len(out["bars"]) == 10
    assert out["bars"][0]["c"] == 100.0
    assert out["bars"][-1]["c"] == 109.0
    assert out["bars"][0]["t"] < out["bars"][-1]["t"]


def test_price_series_missing_ticker_is_explicit(library):
    out = library.price_series("ZZZZ")
    assert out["bars"] == []
    assert "no 1d bar cache" in out["error"]
