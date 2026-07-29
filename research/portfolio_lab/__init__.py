"""Portfolio construction / sizing research lab (WS-D).

Research-only portfolio-risk layer: Ledoit-Wolf shrunk covariance, per-position
/ per-sector / per-theme exposure caps, volatility targeting, correlation-aware
top-N sizing, turnover/liquidity constraints, and a whole-share rebalance plan.
Not wired into any live execution path -- see docs/superpowers/plans/
2026-07-26-market-regime-and-sector-context.md section 4 (WS-D).
"""
