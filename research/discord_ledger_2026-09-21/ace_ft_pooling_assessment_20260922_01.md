# ACE and FT: pooling assessment

ACE and FT share a broad instrument family—primarily long calls/puts in liquid index ETFs and large-cap names—but the current export does **not** support treating them as one managed strategy.

| Characteristic | ACE | FT | Pooling implication |
| --- | ---: | ---: | --- |
| Entry lifecycles | 222 | 147 | Enough for separate descriptive studies, not enough for a high-capacity pooled model. |
| Calls / puts | 151 / 69 | 82 / 65 | Both trade both directions, but FT is materially more put-heavy. |
| Explicit 0DTE alerts | 65 (29.3%) | 59 (40.1%) | Both use short-dated options; FT does so more often. |
| Complete option identity | 147 (66.2%) | 108 (73.5%) | Both can support contract-level research. |
| Top three symbols’ share | 35.6% | 29.3% | ACE is more concentrated in SPY/IWM/SPX; FT is led by IWM/SPY then individual megacaps. |
| Management events per entry | 0.93 | 0.12 | The decisive difference: ACE reports frequent trims, exits, adds, and stop changes; FT’s management trail is sparse. |
| Conservatively linked managed lifecycles | 35 | 3 | A pooled exit/management model would mostly learn ACE behavior and mislabel FT. |
| Management-time underlying proxy outcomes | 41 favorable / 10 unfavorable | 3 / 1 | The tiny FT outcome subset is not sufficient to establish comparable management quality. |

## Decision

Do **not** pool ACE and FT for managed-trade outcome labels, exit rules, contract-management policy, or profitability estimates. Their apparent similarity is insufficient: their reporting/management processes differ by almost an order of magnitude, and unavailable management is not a neutral outcome.

It is reasonable to share a common **feature vocabulary**: point-in-time underlying momentum, VWAP distance, relative volume, opening-range structure, daily trend, catalyst flags, liquidity, and option tenor/moneyness. Train and evaluate a separate behavioral/entry model for each caller, then test whether a shared model beats two separate baselines on an untouched date range. Pool only if that test demonstrates an out-of-sample improvement for both callers.

## Appropriate ML experiment

Reinforcement learning is not appropriate for this source: it needs a reliable reward for every action and a market simulator or live interaction. Here, reported result posts are selectively observed, and option fills/management actions are incomplete. RL would optimize the simulator assumptions, not reconstruct either trader.

The first valid experiment is two-stage and low-capacity:

1. **Behavioral imitation:** label each ACE or FT alert time as positive and matched same-symbol/time-of-day non-alert intervals as negative. Use only data known at the candidate time. A regularized logistic regression or shallow tree can reveal whether the caller’s alert timing is reproducible beyond a ticker/time-of-day shortcut.
2. **Tradeability:** independently evaluate the frozen signal on underlying forward returns and, later, real option bid/ask execution. This asks whether the recreated behavior has edge; it must not use post-profit claims as labels.

Use date-based development/validation/test splits, retain ACE/FT as a feature only in a deliberately pooled challenger, and benchmark against existing intraday-structure and momentum modules. The model must beat separate caller baselines after costs and across symbols/regimes before it informs paper trading.
