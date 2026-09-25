# ACE observed-alert timing experiment: methodology

## Question and boundary

This experiment asks a narrow, auditable question: among predeclared, frozen same-symbol/session candidate times, can a small model rank the timing of source-cited ACE entry alerts better than chance? It does **not** claim to discover a tradable universe, choose direction, price options, recover ACE's discretionary management, estimate option P&L, or prove profitability.

An ACE entry alert is a positive. A quiet-window candidate is *unlabeled*, not a known negative: it may be an omitted setup, an unexported trade, or a valid opportunity ACE decided not to post. Thus all classification metrics describe observed-alert propensity, not the probability of a winning trade. This is the correct positive-unlabeled interpretation for the currently available export.

## Frozen inputs

| Artifact | Role | Time discipline |
| --- | --- | --- |
| `ace_setup_outcomes_20260923_02/source_alert_outcomes.jsonl` | Source-cited entry alerts and precomputed underlying structure | Features use completed minute bars before decision; forward returns are held out. |
| `ace_setup_outcomes_20260923_02/matched_controls.jsonl` | Same-symbol/session, same-direction no-nearby-alert candidates | Frozen before this experiment; not inferred from outcomes. |
| `ace_context_selectivity_20260923_01/candidate_context.jsonl` | Discord watchlist context | Conservative message creation/edit availability strictly before the candidate time. |

No option price, fill, quantity, claimed return, image-derived result, post-alert message, outcome, news field, or universe-membership field is a model input. The local news library is deliberately excluded because its article timestamp is bounded but its historical observation time is not certified.

## Model plan

Chronological ET session-date partitions are fixed at 60% train, 20% validation, 20% final test. The test partition is never used to select a model. Every model is a predeclared ablation:

1. `price_structure_logistic`: five directional completed-bar fields—5m/20m return, VWAP distance, recent volume ratio, and opening-range distance.
2. `price_time_watchlist_logistic`: the above plus circular time-of-day and four point-in-time Discord watchlist flags.
3. `price_time_watchlist_hgb`: a deliberately capacity-limited histogram gradient booster on exactly the second set, included only to test whether shallow non-linearity adds stable observed-label retrieval.
4. `price_time_watchlist_xgb`: a CPU-only, depth-two XGBoost ablation with 80 trees, low learning rate, high child-weight, row/column subsampling, and strong L1/L2 regularization. It has its own training-label permutation diagnostic; it is not a hyperparameter search.

Each logistic classifier uses balanced classes and fixed `C=1`; the booster has fixed small trees, strong L2 regularization, and inverse-frequency sample weights. No GPU is required or expected to change these results. A train-label permutation with the highest-capacity fixed model is a required falsification arm; a real model that does not exceed it has not established an association.

## Measurements

Primary measurements are held-out average precision, ROC AUC, and within-source-group top-1/mean reciprocal rank. The group ranking is deliberately harder to misinterpret: every observed alert is ranked only against the frozen candidates tied to its exact source-message group.

Separately and never as a training label, the experiment reports 15-minute and 60-minute directional *underlying* returns—the two short intraday horizons actually present in the frozen replay artifact. Per source group, it computes the alert's return minus the mean return of its matched candidates, a bootstrap 95% interval, and a two-sided approximate minimum detectable effect (MDE) at alpha 0.05 and 80% power. A wide MDE means a null result is underpowered—not evidence that no difference exists.

## Decision rules and next experiment

This phase is a pass only if its real primary selector materially exceeds its permutation diagnostic on final-test retrieval and the next independent candidate tape confirms a practically meaningful underlying outcome after predeclared delay/cost assumptions. It cannot directly graduate to paper trading.

The next necessary experiment is an immutable, timestamped candidate tape that independently enumerates the traded universe and both directions before ACE's alert. That tape enables separate tests of universe selection, direction selection, timing, contract ranking, and management. Reinforcement learning is premature until those state/action/outcome records exist; it cannot restore information absent from the Discord export.
