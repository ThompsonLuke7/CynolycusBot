"""The incremental embedding append must survive the two on-disk timestamp forms.

``collect_company_news`` writes ``news_records.parquet`` through
``schema.parquet_safe_causal_metadata``, which serializes causal columns to ISO
strings. ``news_embeddings.parquet`` holds real datetimes. On 2026-08-17 the
weekly refresh concatenated the two for the first time on main (the stringifying
write reached main in the 08-16 merge) and pyarrow rejected the resulting mixed
str/Timestamp object column, so no new embeddings were persisted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.news import pipeline

pytestmark = pytest.mark.safe


def _records_frame_with_string_timestamps() -> pd.DataFrame:
    """Mirrors what parquet_safe_causal_metadata leaves on disk."""
    return pd.DataFrame(
        {
            "record_id": ["r1", "r2"],
            "ticker": ["AAA", "BBB"],
            "timestamp": ["2026-08-16T13:30:00+00:00", "2026-08-16T14:00:00+00:00"],
            "text": ["first headline", "second headline"],
        }
    )


def _prior_embeddings_frame() -> pd.DataFrame:
    """Mirrors the existing embeddings artifact: real tz-aware datetimes."""
    return pd.DataFrame(
        {
            "record_id": ["r0"],
            "ticker": ["ZZZ"],
            "timestamp": pd.to_datetime(["2026-08-15T12:00:00+00:00"], utc=True),
            "text": ["older headline"],
            "embedding": [np.zeros(3, dtype=np.float32)],
            "embedding_available": [1.0],
            "finbert_available": [0.0],
            "_text_hash": ["deadbeefdeadbeef"],
        }
    )


def _run_incremental(tmp_path, monkeypatch):
    news_path = tmp_path / "news_records.parquet"
    out_path = tmp_path / "news_embeddings.parquet"

    # Write records the way production does, then confirm the fixture really is
    # the string form — otherwise this test would pass without exercising it.
    from signals.news.schema import parquet_safe_causal_metadata

    parquet_safe_causal_metadata(_records_frame_with_string_timestamps()).to_parquet(
        news_path, index=False
    )
    assert pd.read_parquet(news_path)["timestamp"].dtype == object

    _prior_embeddings_frame().to_parquet(out_path, index=False)

    # Keep the test off the model download path; only the persistence layer is
    # under test here.
    monkeypatch.setattr(pipeline, "ensure_data_dirs", lambda: None)

    return pipeline.build_news_embeddings(
        news_path,
        output_path=out_path,
        generate_embeddings=False,
        generate_finbert=False,
        incremental=True,
    ), out_path


def test_incremental_append_writes_despite_string_timestamps(tmp_path, monkeypatch):
    out, out_path = _run_incremental(tmp_path, monkeypatch)

    # The regression was a write failure, so the artifact on disk is the assertion.
    saved = pd.read_parquet(out_path)
    assert set(saved["record_id"]) == {"r0", "r1", "r2"}
    assert len(out) == 3


def test_appended_timestamps_land_as_utc_datetimes(tmp_path, monkeypatch):
    _, out_path = _run_incremental(tmp_path, monkeypatch)
    saved = pd.read_parquet(out_path)

    assert str(saved["timestamp"].dtype) == "datetime64[ns, UTC]"
    by_id = dict(zip(saved["record_id"], saved["timestamp"]))
    assert by_id["r1"] == pd.Timestamp("2026-08-16T13:30:00+00:00")
    # The pre-existing row must not be shifted by the alignment.
    assert by_id["r0"] == pd.Timestamp("2026-08-15T12:00:00+00:00")


def test_unparseable_timestamp_raises_rather_than_nulling():
    values = pd.Series(["2026-08-16T13:30:00+00:00", "not-a-timestamp"])

    with pytest.raises(ValueError, match="unparseable"):
        pipeline._as_utc_timestamps(values, label="news_records")


def test_existing_nulls_are_preserved_not_counted_as_failures():
    values = pd.Series(["2026-08-16T13:30:00+00:00", None])

    converted = pipeline._as_utc_timestamps(values, label="news_records")

    assert converted.isna().tolist() == [False, True]
