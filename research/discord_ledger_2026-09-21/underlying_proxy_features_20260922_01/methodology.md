# What the favorable-cohort feature comparison did—and did not—show

The comparison was **not a grid search, model fit, or causal analysis**. Before looking at the management-proxy outcome labels, the analysis reused five existing, interpretable intraday measures: direction-adjusted 5-minute return, direction-adjusted 20-minute return, direction-adjusted distance from session VWAP, last-5-minute versus prior-20-minute volume, and directional distance from the opening-range high. Each is calculated only from completed one-minute bars preceding the Discord alert.

It then compared medians for the 41 favorable versus 10 unfavorable reported-management underlying proxies. That descriptive screen found higher short-term directional momentum and volume among favorable rows; the lower directional VWAP distance motivated a **hypothesis** that ACE may enter nearer a retest rather than chase a stretch. It does not establish that any feature caused the outcome. The sample is small, labels are based on selectively posted management, and the interpretation was formed after seeing this cohort.

## Correct recreation test

1. Freeze a small candidate rule before evaluating more data: liquid index/mega-cap universe; call/put direction aligned with 5-/20-minute momentum; a volume-confirmation threshold; and a bounded, direction-aware VWAP-distance filter.
2. Generate signals from underlying bars at every eligible minute, including intervals when ACE did not alert. Do not train on a post-profit claim or on the future management time.
3. Use a date-based split: development period, untouched validation period, then a final untouched test period. Fit thresholds/model only on development; lock them before each next period.
4. Evaluate the same entry/exit horizon and costs for every signal, compare against direction-only and existing-system baselines, and report precision, recall, trade count, expectancy, drawdown, and results by regime/symbol.
5. Only after the underlying signal survives that test, run a separately specified option-execution paper study using real contemporaneous contract quotes. Do not assume an underlying winner is an option winner.

The current system already exposes most needed ingredients: `strategies/intraday_structure/features.py` has VWAP-distance and relative-volume features; `strategies/multi_ticker_swing/features/build_features.py` has VWAP/relative-volume features; and the Discord SPY pilot has point-in-time 5-/20-minute return snapshots. The remaining work is a frozen, leakage-safe evaluation layer—not another indicator search.
