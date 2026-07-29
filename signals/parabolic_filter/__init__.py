"""Parabolic-likelihood filter.

Scores a signal at decision time with the probability that the underlying makes a
large PERCENTAGE move (default: >= +25% favorable excursion within 20 4H bars).

Two entry points, one per module, because the validated answer differs:

  * ``momentum_expansion``  -> use the model (`predict_proba`). A plain volatility
    sort is worthless there (n.s. at every cut-off) while the model is worth
    +2.6 to +3.6pp per trade.
  * ``multi_ticker_swing_htf`` -> use `atr_rule_rank`. A one-line ATR% sort matches
    the model within noise (+2.43pp vs +2.92pp, overlapping CIs), so the ML
    dependency is not justified.

See research/options_experiment/09_shares_parabolic_filter.md for the evidence.
This is a SHARES selector. It says nothing about options — that branch was retracted
(see 10_RETRACTION_option_pnl_invalid.md).
"""
from .filter import (  # noqa: F401
    ParabolicFilter,
    atr_rule_rank,
    recommended_selector,
    DEFAULT_THRESHOLD_PCT,
    DEFAULT_HORIZON_BARS,
)
