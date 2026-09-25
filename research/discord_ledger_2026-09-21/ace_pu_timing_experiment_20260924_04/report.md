# ACE positive-unlabeled timing experiment

Research-only. This measures whether a compact, fixed selector reproduces the timing of *observed* ACE alerts among frozen same-symbol/session quiet-window candidates. It is not a profitability backtest, option model, full-universe discovery test, or live-trading recommendation.

## Design

- Positives: source-cited ACE entry alerts. Unlabeled candidates: frozen same-symbol/session, same-direction no-nearby-alert windows. An unlabeled candidate is not a verified bad trade.
- Inputs: completed-bar directional price/volume structure, circular time-of-session, and strictly pre-decision exported Discord watchlist flags. No outcomes, options data, claimed P&L, news, or universe membership are model inputs.
- Split: chronological by ET session date, fixed before fitting. All variants are fixed ablations; none was selected on the test partition.
- Scores are observed-alert propensities for this sampled panel—not probabilities of profitability or execution.

## Held-out observed-label retrieval

| Model | Test AP | Test ROC AUC | Group top-1 | Mean reciprocal rank |
| --- | ---: | ---: | ---: | ---: |
| price_structure_logistic | 0.129 | 0.588 | 0.160 | 0.372 |
| price_time_watchlist_logistic | 0.172 | 0.665 | 0.120 | 0.373 |
| price_time_watchlist_hgb | 0.293 | 0.774 | 0.360 | 0.577 |
| price_time_watchlist_xgb | 0.158 | 0.695 | 0.200 | 0.420 |
| permuted-train-label null (price_time_watchlist_xgb) | 0.112 | 0.532 | 0.040 | 0.286 |

## Separate held-out underlying outcome description

These are not trained targets and are not option P&L. Each value is the alert's directional underlying return less the mean of its frozen matched candidates. The MDE is the approximate absolute mean difference this sample could detect with 80% power; if it is large, a null is ‘not measurable here,’ not evidence of no edge.

| Horizon | Paired groups | Alert mean (%) | Control-group mean (%) | Difference (%) | 95% bootstrap CI (%) | MDE (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15m | 15 | -0.076 | 0.001 | -0.077 | [-0.153, -0.009] | 0.104 |
| 60m | 8 | -0.105 | -0.017 | -0.088 | [-0.209, 0.020] | 0.179 |

## Interpretation boundary

A model can be useful only if it exceeds the permuted-label diagnostic in a later untouched candidate tape and its selected candidates then have a practically meaningful underlying outcome after realistic delay/cost assumptions. This dataset injects ACE's symbol and direction, so it cannot answer universe discovery, direction selection, contract selection, or management. The next testable variant is a new immutable candidate tape that independently enumerates ACE's traded universe and both directions at each decision time; do not add reinforcement learning before that prerequisite exists.
