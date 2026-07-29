"""Regime-conditional POLICY study (research only).

See ``HYPOTHESES.md`` in this package for the pre-registered rules,
thresholds, statistic, and FDR family. This package does not retrain any
model and does not add features to any live/deployed pipeline; it replays
the existing deployed model's real walk-forward OOF signal stream under
alternate admission/sizing/stop policies, reusing the unchanged
``family_backtest`` exit engine and ``portfolio_backtest`` trade/snapshot
types.
"""
