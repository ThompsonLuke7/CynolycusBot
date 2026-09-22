"""Guards for reusable model-selection and temporal-split invariants."""
from __future__ import annotations

import pandas as pd

from strategies.model_training.colab_competition import choose_best, time_split


def _frame(n=40):
    rows = []
    for i, timestamp in enumerate(pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")):
        for ticker in ("AAA", "BBB"):
            rows.append({
                "timestamp": timestamp,
                "ticker": ticker,
                # A five-bar forward label cannot be known at its decision time.
                "label_end": timestamp + pd.Timedelta(hours=20),
                "feature": i,
            })
    return pd.DataFrame(rows)


def test_time_split_purges_labels_that_cross_each_boundary():
    train, validation, test = time_split(
        _frame(), .60, .20, timestamp_column="timestamp", label_end_column="label_end")

    first_validation = validation["timestamp"].min()
    first_test = test["timestamp"].min()
    assert (train["label_end"] < first_validation).all()
    assert (validation["label_end"] < first_test).all()
    # Split boundaries keep whole decision timestamps together across tickers.
    assert train.groupby("timestamp").size().eq(2).all()
    assert validation.groupby("timestamp").size().eq(2).all()


def test_model_winner_uses_validation_not_opposite_test_result():
    rows = pd.DataFrame([
        {"family": "validation_winner", "seed": 1, "val_ndcg_at_10": .8,
         "val_precision_at_10": .7, "test_ndcg_at_10": .1},
        {"family": "test_winner", "seed": 2, "val_ndcg_at_10": .2,
         "val_precision_at_10": .1, "test_ndcg_at_10": .9},
    ])
    assert choose_best(rows)["family"] == "validation_winner"
