"""The batched per-ticker news-embedding average must equal the whole-file one.

The whole-file read held every ~8.5 KB JSON vector plus its text in pandas; the
batched read keeps four columns and one batch at a time. Same rows, same order,
same float32 mean, so the theme vectors are bit-identical.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import themes.dynamic_theme.stages.step02_embed as step02

pytestmark = pytest.mark.safe


def test_batched_average_is_bit_identical_to_the_whole_file_path(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    rows = []
    for i in range(60):
        ticker = ["aaa", "BBB", "CCC", "DDD"][i % 4]
        text = "Form 4 statement of changes in beneficial ownership" if i % 7 == 0 else f"news item {i}"
        rows.append({
            "record_id": f"r{i}", "ticker": ticker, "text": text,
            "embedding": json.dumps(rng.standard_normal(8).tolist()),
            "embedding_available": 0.0 if i == 3 else 1.0,
        })
    path = tmp_path / "news_embeddings.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    monkeypatch.setattr(step02, "_NEWS_EMBEDDINGS_PATH", path)

    tickers = ["AAA", "BBB", "CCC", "EEE"]
    streamed = step02._load_article_embeddings_by_ticker(tickers)
    monkeypatch.setattr(step02, "_article_embeddings_by_ticker_streaming", lambda _t: None)
    whole = step02._load_article_embeddings_by_ticker(tickers)

    assert list(streamed) == list(whole) == ["AAA", "BBB", "CCC"]
    for ticker in whole:
        assert streamed[ticker].dtype == whole[ticker].dtype
        assert np.array_equal(streamed[ticker], whole[ticker])
