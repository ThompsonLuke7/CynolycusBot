"""Deterministic, model-artifact-free Meta Ranker parity rows."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


DECISION_BAR = pd.Timestamp("2026-07-30T18:00:00Z")
LATER_BAR = DECISION_BAR + pd.Timedelta(hours=4)


def meta_rows() -> pd.DataFrame:
    """Return the synthetic matrix used by the ranking/intents parity tests.

    ``required_feature`` is present in the model manifest but missing for
    ``MISS``. ``NFIN`` receives a non-finite booster output in the injected
    prediction boundary. ``AAA`` also has a later row whose close must never
    be used when the decision bar is ``DECISION_BAR``.
    """

    return pd.DataFrame(
        [
            {
                "timestamp": DECISION_BAR,
                "ticker": "AAA",
                "close": 101.0,
                "dollar_vol_pctile_252": 0.95,
                "up": 0.8,
                "qual": 0.2,
                "required_feature": 1.0,
            },
            {
                "timestamp": DECISION_BAR,
                "ticker": "BBB",
                "close": 202.0,
                "dollar_vol_pctile_252": 0.92,
                "up": 0.8,
                "qual": 0.2,
                "required_feature": 2.0,
            },
            {
                "timestamp": DECISION_BAR,
                "ticker": "HELD",
                "close": 303.0,
                "dollar_vol_pctile_252": 0.90,
                "up": 1.0,
                "qual": 0.35,
                "required_feature": 3.0,
            },
            {
                "timestamp": DECISION_BAR,
                "ticker": "LOWL",
                "close": 404.0,
                "dollar_vol_pctile_252": 0.10,
                "up": 0.6,
                "qual": 0.95,
                "required_feature": 4.0,
            },
            {
                "timestamp": DECISION_BAR,
                "ticker": "BLACK",
                "close": 505.0,
                "dollar_vol_pctile_252": 0.95,
                "up": 0.5,
                "qual": 0.9,
                "required_feature": 5.0,
            },
            {
                "timestamp": DECISION_BAR,
                "ticker": "MISS",
                "close": 606.0,
                "dollar_vol_pctile_252": 0.95,
                "up": 0.4,
                "qual": 0.6,
                "required_feature": np.nan,
            },
            {
                "timestamp": DECISION_BAR,
                "ticker": "NFIN",
                "close": 707.0,
                "dollar_vol_pctile_252": 0.10,
                "up": 0.3,
                "qual": 0.1,
                "required_feature": 7.0,
            },
            {
                "timestamp": DECISION_BAR,
                "ticker": "PINF",
                "close": 808.0,
                "dollar_vol_pctile_252": 0.95,
                "up": 0.2,
                "qual": 0.05,
                "required_feature": 8.0,
            },
            {
                "timestamp": LATER_BAR,
                "ticker": "AAA",
                "close": 999.0,
                "dollar_vol_pctile_252": 0.95,
                "up": 9.0,
                "qual": 9.0,
                "required_feature": 9.0,
            },
        ]
    )


class FixtureBooster:
    """Tiny booster double used only at ``score_frame``'s prediction seam."""

    def __init__(self, label: str):
        self.label = label

    def predict(self, dmatrix: object) -> np.ndarray:
        values = dmatrix.get_data()  # type: ignore[attr-defined]
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values)[:, 0 if self.label == "upside" else 1].astype(float)
        if len(values) > 6:
            # These invalid outputs must not change the percentile baseline of
            # the valid rows.  They are deliberately at the end of the fixture.
            if self.label == "quality":
                values[6] = np.nan
            if len(values) > 7:
                values[7] = np.inf
        if self.label == "upside" and len(values) > 7:
            values[7] = np.inf
        return values


def fixture_booster_loader(label: str) -> tuple[FixtureBooster, list[str]]:
    return FixtureBooster(label), ["up", "qual", "required_feature"]


DECISION_TIME = datetime(2026, 7, 30, 18, 20, tzinfo=timezone.utc)
