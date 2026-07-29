"""Parabolic-likelihood scoring for share entries.

Label definition (what "parabolic" means here):
    favorable excursion >= +25% of entry price within 20 4H bars,
    measured from the underlying's own bars, independent of the module's exit.

Why percentage and not ATR: an ATR-normalised label counted a 3.4% drift on a
quiet stock as "parabolic" and trained the model to select LOW-volatility names --
the opposite of a squeeze candidate. Relabelled on percentage move, the filter
selects volatile names and lifts share returns significantly.

Time correctness: every feature must be observable at the decision bar. The scorer
never sees the label, and `predict_proba` refuses input whose feature timestamp is
after the decision timestamp.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD_PCT = 0.25   # >= +25% favorable excursion
DEFAULT_HORIZON_BARS = 20      # within 20 4H bars

# Validated per-module recommendation (09_shares_parabolic_filter.md)
_MODEL_MODULES = {"momentum_expansion"}
_ATR_RULE_MODULES = {"multi_ticker_swing_htf"}

# Operating points with significant out-of-sample lift. momentum's top-10% cell was
# NOT significant (n=153), so 0.10 is deliberately absent.
RECOMMENDED_TOP_FRACTION = {
    "momentum_expansion": 0.20,        # +3.61pp, CI [+0.79, +5.84]
    "multi_ticker_swing_htf": 0.30,    # +2.92pp, CI [+1.18, +4.79]
}


def recommended_selector(module: str) -> str:
    """'model' or 'atr_rule' -- which selector is justified for this module."""
    if module in _MODEL_MODULES:
        return "model"
    if module in _ATR_RULE_MODULES:
        return "atr_rule"
    raise ValueError(
        f"no validated parabolic-filter recommendation for module {module!r}; "
        f"validated modules: {sorted(_MODEL_MODULES | _ATR_RULE_MODULES)}"
    )


def atr_rule_rank(df: pd.DataFrame, *, atr_col: str = "atr_at_entry",
                  price_col: str = "entry_px_underlying") -> pd.Series:
    """Rank candidates by ATR as a fraction of price (higher = more squeeze-capable).

    This is the whole selector for multi_ticker_swing_htf: it matched the model
    within noise there (+2.43pp vs +2.92pp), so shipping ML is not justified.
    Returns a 0-1 score (percentile rank), NaN where inputs are unusable.
    """
    for c in (atr_col, price_col):
        if c not in df.columns:
            raise KeyError(f"atr_rule_rank requires column {c!r}")
    atr_pct = pd.to_numeric(df[atr_col], errors="coerce") / pd.to_numeric(df[price_col], errors="coerce")
    atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan)
    return atr_pct.rank(pct=True)


@dataclass
class ParabolicFilter:
    """Trained model scorer. Use for momentum_expansion.

    Load with `ParabolicFilter.load(dir)`; score with `predict_proba(features, asof)`.
    """

    booster: object
    feature_names: list[str]
    module: str
    threshold_pct: float = DEFAULT_THRESHOLD_PCT
    horizon_bars: int = DEFAULT_HORIZON_BARS
    trained_through: Optional[str] = None

    # ---------------------------------------------------------------- persistence
    @classmethod
    def load(cls, directory: str | Path) -> "ParabolicFilter":
        import xgboost as xgb
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        b = xgb.XGBClassifier()
        b.load_model(str(d / "model.json"))
        return cls(booster=b, feature_names=meta["feature_names"], module=meta["module"],
                   threshold_pct=meta.get("threshold_pct", DEFAULT_THRESHOLD_PCT),
                   horizon_bars=meta.get("horizon_bars", DEFAULT_HORIZON_BARS),
                   trained_through=meta.get("trained_through"))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / "model.json"))
        (d / "meta.json").write_text(json.dumps({
            "module": self.module,
            "feature_names": self.feature_names,
            "threshold_pct": self.threshold_pct,
            "horizon_bars": self.horizon_bars,
            "trained_through": self.trained_through,
        }, indent=1))

    # ---------------------------------------------------------------- scoring
    def predict_proba(self, features: pd.DataFrame, *, asof=None,
                      feature_ts_col: str = "timestamp") -> np.ndarray:
        """Probability of a parabolic move. Fails loudly rather than guessing.

        Raises if a required feature column is missing, or if `asof` is supplied and
        any feature row is timestamped after it (lookahead).
        """
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise KeyError(
                f"parabolic filter missing {len(missing)} feature(s), e.g. {missing[:5]}. "
                "Scoring with absent features would silently change the model's meaning."
            )
        if asof is not None and feature_ts_col in features.columns:
            ts = pd.to_datetime(features[feature_ts_col], utc=True)
            # accept naive or tz-aware asof; normalise both to UTC
            a = pd.Timestamp(asof)
            a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
            bad = int((ts > a).sum())
            if bad:
                raise ValueError(
                    f"lookahead: {bad} feature row(s) timestamped after asof={asof}"
                )
        X = features[self.feature_names].replace([np.inf, -np.inf], np.nan)
        return self.booster.predict_proba(X)[:, 1]

    def select(self, features: pd.DataFrame, *, top_fraction: Optional[float] = None,
               asof=None) -> pd.Series:
        """Boolean mask of the top `top_fraction` candidates by parabolic probability."""
        frac = top_fraction if top_fraction is not None else RECOMMENDED_TOP_FRACTION.get(self.module)
        if frac is None:
            raise ValueError(f"no recommended top_fraction for module {self.module!r}")
        p = self.predict_proba(features, asof=asof)
        if len(p) == 0:
            return pd.Series([], dtype=bool, index=features.index)
        cutoff = np.quantile(p, 1.0 - frac)
        return pd.Series(p >= cutoff, index=features.index)
