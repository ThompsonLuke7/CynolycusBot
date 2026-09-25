# ACE strategy replication: what ML can and cannot do next

## Bottom line

Do not start with a larger black-box model or reinforcement learning. The
export can support a staged replication program, but it does **not** yet
identify ACE's complete policy: all seven VIP exports are empty, only 120
source alerts are point-in-time eligible for the current intraday replay, and
the no-alert comparison times are **unlabeled**, not verified losing trades.

The appropriate design is three separate models/rules, each with a different
label and success criterion:

1. **Candidate selection / imitation:** “Would ACE alert this candidate now?”
2. **Market outcome:** “Does the underlying offer a favorable, risk-normalized
   move from this candidate now?”
3. **Management and expression:** “How should a paper position be exited and,
   only after valid BBO data, which liquid option expresses it?”

An alert-imitation score is not a profitability model. A profitable-market
score is not evidence that it reproduces ACE. Both must be evaluated apart.

## Current evidence and hard limits

- The 120 alert versus matched-control study found no aggregate selectivity for
  the present Intraday Structure gate, Discord watchlist state, or available
  timestamp-bounded news features.
- The SPY/non-SPY split is a hypothesis, not a finding to deploy: SPY is
  -13.0 percentage points on explicit confirmation (n=54, paired exact p=.265)
  and non-SPY is +12.1 points (n=66, p=.215).
- Free-channel watchlists are broad lists more often than executable level
  rules. They are available before only 16/120 source candidates in the
  seven-day feature window and do not distinguish alerts from controls.
- There is no historical option NBBO/BBO in the local store. Option trade bars
  are prints, not marks; do not train or evaluate option P&L from them.
- The retained historical universe is incomplete before September 2026, and
  broad historical one-minute underlying coverage is absent. A full-universe
  intraday discovery backtest would therefore be contaminated or fail closed.

## Best next ML: positive-unlabeled candidate ranking

Treat the 120 confirmed source alerts as **positive examples**. Treat
same-symbol/session non-alert moments only as **unlabeled**, not negatives:
they may be an unobserved VIP alert, a setup ACE chose not to publish, or an
ordinary non-trade. This is the positive-unlabeled (PU) problem, not normal
binary classification.

Use a two-stage, chronological experiment once point-in-time candidate logs
exist:

1. Freeze a candidate universe and create one row for every candidate/time
   considered, including abstentions.
2. Label only observed ACE alert availability times as positive. Preserve the
   publication channel/free-vs-VIP exposure as a labeling-propensity feature.
3. Start with regularized logistic regression or a GAM, then a shallow
   gradient-boosted tree. Calibrate probabilities only on the training period.
4. Use day-grouped, walk-forward splits and a final untouched period. Report
   precision/recall at the capacity actually traded (top 1–3/day), PR-AUC,
   calibration, and lift over same-time random and deterministic-rule baselines.
5. Run a permuted-label null and an ablation for every feature family. Stop if
   it does not beat the predeclared simple rule out of sample.

PU learning is appropriate precisely because ordinary “no Discord alert =
negative” training would be wrong. But its result depends on the labeling
mechanism; free alerts versus missing VIP alerts are selected, not random.
The SAR literature explicitly warns that this propensity must be modeled or
the model will be biased.

Sources: [Bekker & Davis, 2018 (SAR PU learning)](https://proceedings.mlr.press/v94/bekker18a/bekker18a.pdf), [He et al., 2026 (unknown PU labeling mechanism)](https://proceedings.mlr.press/v337/he26a.html).

## Features worth collecting

At every candidate time, persist only information then available:

- **Underlying microstructure:** returns at 1/5/15/30 minutes, VWAP distance
  and slope, opening-range state, relative volume versus same minute-of-day,
  ATR-normalized range extension, higher-timeframe trend, and gap state.
- **Cross-market context:** SPY/QQQ/IWM trend and volatility, sector relative
  strength, breadth, and scheduled-event state.
- **Catalysts:** original publication timestamp, observation timestamp, source,
  novelty, direct-ticker relation, and predeclared text score. Do not use a
  same-day aggregate that may include later headlines.
- **Candidate-universe facts:** exact snapshot ID, liquidity/price filters,
  rank, and why a ticker was excluded. This is required to test discovery
  without injecting ACE's ticker.
- **Discord evidence:** pre-alert watchlist/level state only. Alert text itself
  may label ACE's decision but cannot be an input to predicting that decision.

Split this data into at least SPY/ETF and single-name sleeves from the start;
the observed composition and replay results do not support one shared policy.

## Separate market-outcome model

The candidate model learns ACE's publishing behavior, which may include
lotteries, hedges, and discretionary context. Train a distinct model on market
labels, not posted winners:

- Directional MFE and MAE at 15, 30, 60, 120 minutes and session close,
  normalized by pre-candidate ATR.
- A barrier label such as “target before invalidation” with an explicit stop,
  horizon, and delay.
- Conditional expected utility after conservative equity/ETF costs; options
  remain excluded until valid quotes exist.

It should be a small, interpretable baseline first (logistic/GAM/tree). A
model qualifies only if it improves risk-adjusted, walk-forward paper results
over a frozen technical baseline, with time/regime/ticker robustness—not merely
alert AUC.

## Management: state machine first, ML second

The ledger already preserves adds, trims, exits and invalidations where they
are source-cited, but too few lifecycles have fully resolved management to
behavior-clone it reliably. Keep the existing deterministic state machine for
paper positions. Once a larger, complete action ledger exists, test a
supervised exit-hazard model or fitted-Q model against that state machine using
underlying R-unit outcomes.

Behavior cloning alone is unsafe for a sequential policy: an early error puts
the replica into states absent from the demonstrations. DAgger-style methods
address that distribution shift by gathering labels on the learner's own
states; we cannot do that retrospectively with ACE. Any imitation policy must
therefore be paper-shadowed and evaluated on its own generated states.

Source: [Ross, Gordon & Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html).

## Options: deterministic liquidity ranking, then evidence

Do not apply an ML model to stale option prints. The next option component is a
deterministic ranker over an observed chain: DTE band, strike distance/delta,
bid/ask width as a percentage of mid, quoted size, quote age, volume/OI, and
expected-move fit. It needs a historical or forward-captured BBO record for
both the called contract and alternatives.

Alpaca documents that its free indicative feed is not actual OPRA quotes;
OPRA is the consolidated BBO feed and requires subscription. Its historical
options documentation also says coverage begins in February 2024. Verify any
entitlement and quote freshness on a small sample before using it in P&L.

Source: [Alpaca historical option data](https://docs.alpaca.markets/us/docs/historical-option-data).

## Concrete sequence

1. **Complete the evidence dataset:** obtain a full export that actually
   includes VIP messages/attachments, or continue collecting free-channel data
   prospectively. Promote screenshot OCR only after targeted human review.
2. **Build an append-only candidate tape:** every 1–5 minutes record the
   universe snapshot, all candidate features, rank, abstention/entry decision,
   raw news availability fields, and later underlying labels.
3. **Capture execution data prospectively:** underlying quote plus option-chain
   bid/ask/sizes/quote timestamps for liquid SPY/QQQ and a limited single-name
   set. Paper trade with explicit delay/slippage rules.
4. **Run the three-model bake-off:** PU imitation ranker, independent market
   outcome model, deterministic state machine. Keep SPY/ETF and single-name
   sleeves separate. Require each to beat its baseline out of sample.
5. **Only then test learned management:** supervised hazard/fitted-Q first;
   reinforcement learning only if it beats imitation and deterministic policy
   on the same frozen paper-replay protocol.

This can eliminate the need to follow alerts in real time, but it cannot
reconstruct hidden VIP content or turn incomplete message history into the
trader's exact private decision process. The productive target is a validated,
independent paper strategy that shares measurable behaviors with ACE—not an
unverified claim of exact replication.
