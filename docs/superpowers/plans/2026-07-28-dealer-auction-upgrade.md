# Dealer Positioning and Auction Router Upgrade Plan

**Created:** 2026-07-28  
**Status:** Ready for implementation  
**Canonical modules:** `strategies/intraday_structure/` and
`strategies/dealer_positioning/`  
**Operating mode:** Research and paper only

## 1. Objective

Upgrade the existing Intraday Structure engine into one reproducible pipeline that:

1. Receives an upstream candidate.
2. Builds the point-in-time auction and dealer context.
3. Detects a named, path-dependent one-minute setup.
4. Emits an explicit router decision with entry, invalidation, targets, timing, and no-trade reason.
5. Records every decision and outcome in a replayable ledger.
6. Optionally uses ML to rank confirmed setups and choose an instrument.
7. Evaluates underlying behavior separately from option execution.

This is an extension of the current system, not a new strategy tree.

## 2. Corrected schedule

The first assessment mixed engineering time with the calendar time needed to accumulate forward
evidence. The code upgrade does not require months.

### Engineering estimate

| Workstream | Focused engineering time |
|---|---:|
| Reproducible capture and closed ledger | 0.5–1 day |
| Missing auction features and router schema | 0.5–1 day |
| Replay/ablation runner and reporting | 0.5 day |
| ML dataset/training/scoring scaffold | 0.5–1 day |
| Option quote adapter/paper evaluation glue | 0.5–1 day |
| **Rules-first MVP total** | **2–3 days** |
| **MVP plus ML/option scaffold** | **3–4 days** |

These estimates assume the existing architecture remains canonical and the purchased data arrives in
a documented, machine-readable format. Vendor-specific normalization can add time if schemas,
timestamps, or entitlements are unclear.

### Evidence calendar

- If suitable historical underlying, chain, and NBBO data is purchased, research can start as soon
  as normalization and source-fitness checks pass.
- If data is collected only going forward, calendar time—not development time—becomes the limiting
  factor.
- Live-capital readiness is not scheduled here. It depends on results rather than completion of
  code.

## 3. Non-negotiable design decisions

1. `strategies/intraday_structure` is the one canonical auction/playbook router.
2. Existing rankers propose candidates; the router does not rescan the whole universe.
3. Rules identify and timestamp a setup. ML ranks setup quality after the rule fires.
4. Dealer positioning remains a context/target feature until ablations prove incremental value.
5. Underlying setup quality and option-contract profitability are evaluated separately.
6. Every feature must be available at the exact router decision timestamp.
7. Historical option trade bars are prohibited as arbitrary-time marks.
8. No live-order behavior changes during this plan.

## 4. Target pipeline

```text
Upstream candidate
      |
      v
Point-in-time context builder
  - multi-day regime
  - overnight/session auction
  - market/sector state
  - dealer map
      |
      v
Rule-based setup detector
  - sweep/reclaim
  - structural rejection
  - breakout/hold
  - VWAP reclaim
  - V reversal
  - trend pullback
      |
      v
Router decision
  - setup and trigger state
  - invalidation and targets
  - expected time window
  - no-trade reason
      |
      +-------------------+
      |                   |
      v                   v
Underlying quality ML   Contract policy/ML
      |                   |
      +---------+---------+
                |
                v
       Paper intent and ledger
                |
                v
      Replay, ablation, reporting
```

## 5. Sprint 1 — Reproducibility spine

**Goal:** Make every live/paper setup reconstructable without relying on bounded in-memory state.

### 5.1 Add an append-only one-minute capture

Create:

- `strategies/intraday_structure/capture.py`
- `strategies/intraday_structure/tests/test_capture.py`

Modify:

- `strategies/intraday_structure/runner.py`
- `strategies/intraday_structure/config.py`
- `strategies/intraday_structure/config/intraday_structure_v1.json`

Persist, per event:

- symbol;
- event timestamp;
- received/available timestamp;
- OHLCV;
- trade count and bar VWAP when supplied;
- session classification;
- feed/source identifier;
- candidate IDs active for the symbol;
- capture/config schema versions.

Target layout:

```text
Data/inference/intraday_structure/capture/
  session_date=YYYY-MM-DD/
    bars.parquet
    candidates.jsonl
    router_decisions.jsonl
    manifest.json
```

Implementation requirements:

- Write append-only raw records during the session.
- Materialize/deduplicate parquet atomically after the session.
- Do not overwrite raw capture.
- Reject timestamp regressions and report duplicates/gaps.
- Capture SPY, QQQ, VIXY, and referenced sector ETFs alongside active candidates.
- Include extended-hours bars when the playbook configuration enables overnight context.

### 5.2 Add a real setup/outcome ledger

Create:

- `strategies/intraday_structure/ledger.py`
- `strategies/intraday_structure/tests/test_ledger.py`

One ledger record per setup must include:

- candidate and setup IDs;
- source signal time and availability time;
- context/router version;
- rule/setup type;
- trigger timestamp and entry-delay rule;
- intended and modeled entry;
- invalidation and target sequence;
- expected time window;
- dealer snapshot ID and calculation version;
- optional ML model/version/score;
- entry, exit, exit reason, MFE, MAE, realized R, and time to target;
- underlying outcome and option outcome as separate sections;
- data-quality warnings.

Target files:

```text
Data/inference/intraday_structure/setup_events.jsonl
Data/inference/intraday_structure/closed_setups.jsonl
```

Acceptance tests:

- A restart cannot produce a second close for the same setup.
- Every closed setup references an existing candidate and router decision.
- A confirmed setup either closes or is explicitly marked open/incomplete.
- Same-bar stop/target collisions remain conservative and documented.

### 5.3 Add a daily capture audit

Create:

- `strategies/intraday_structure/scripts/audit_capture.py`

Report:

- candidate count;
- bars per symbol/session;
- missing minute percentage;
- duplicates and timestamp regressions;
- dealer coverage and snapshot age;
- market/sector context coverage;
- confirmed/open/closed setup reconciliation;
- option-quote coverage where applicable.

**Sprint 1 completion gate:** one synthetic session and one retained live session can be materialized,
audited, replayed, and reconciled to the same decisions.

## 6. Sprint 2 — Auction context and explicit router

**Goal:** Encode the missing “acceptance versus rejection” language and expose one stable decision
contract.

### 6.1 Add causal auction-context features

Create:

- `strategies/intraday_structure/auction.py`
- `strategies/intraday_structure/tests/test_auction.py`

Add features:

#### Higher-timeframe context

- multi-day balance, trend, and transition classification;
- distance and position within the multi-day range;
- prior-day range expansion/compression;
- gap and overnight inventory direction.

#### Overnight and session structure

- overnight high/low;
- overnight VWAP;
- fraction of overnight volume above/below prior close;
- prior-session VAH, VAL, and POC;
- current opening range and developing session VWAP;
- equal-high/equal-low clusters with tolerance and touch count.

#### Level-interaction features

- first break timestamp;
- maximum excursion beyond the level in ATR and percentage units;
- seconds/bars spent beyond the level;
- closes and volume beyond the level;
- reclaim speed;
- value/POC migration beyond the level;
- retest count and retest hold/failure;
- return-from-level velocity.

Every rolling calculation must use only bars whose timestamps are at or before the decision.

### 6.2 Add a dedicated sweep/reclaim detector

Create:

- `strategies/intraday_structure/detectors/sweep_reclaim.py`

Modify:

- `strategies/intraday_structure/detectors/__init__.py`
- `strategies/intraday_structure/engine.py`
- `strategies/intraday_structure/models.py`

Required states:

```text
APPROACHING_LEVEL
SWEPT
RECLAIMED
RETESTING
CONFIRMED
FAILED_ACCEPTANCE
INVALIDATED
```

The detector must distinguish:

- fast sweep and reclaim;
- sustained acceptance beyond the level;
- reclaim without retest;
- reclaim followed by a successful retest;
- false reclaim and renewed acceptance.

Initial thresholds remain versioned hypotheses in configuration.

### 6.3 Add the router decision schema

Extend `models.py` with a typed `RouterDecision` containing:

- `decision_timestamp`;
- `context_regime`;
- `active_level` and `active_level_type`;
- `setup_type`;
- `trigger_state`;
- `direction`;
- `entry_reference`;
- `invalidation_level`;
- `target_sequence`;
- `expected_time_to_target_bars`;
- `rule_confidence`;
- `dealer_context`;
- `market_context`;
- `preferred_instrument`;
- `no_trade_reason`;
- `feature_version`;
- `router_version`.

Initial instrument values:

- `shares`;
- `long_call`;
- `long_put`;
- `debit_spread`;
- `observe_only`;
- `no_trade`.

Rules may recommend an instrument class, but only the option policy may choose a contract.

### 6.4 Complete dealer calculation metadata

Modify:

- `strategies/dealer_positioning/models.py`
- `strategies/dealer_positioning/levels.py`
- `strategies/dealer_positioning/scripts/capture_historical_snapshots.py`

Add:

- explicit calculation name/version;
- sign assumption;
- OI observation date when available;
- chain query and underlying quote timestamps;
- expiration scope;
- formula unit/normalization metadata.

Keep the current calculation unchanged as version `oi_call_plus_put_minus_v1`.

Add research-only variants later:

- alternative sign convention;
- unsigned absolute gamma concentration;
- hypothetical-spot gamma sweep/zero-gamma level.

Do not silently replace the existing live levels.

**Sprint 2 completion gate:** a replayed known sweep/reclaim path emits the same typed router decision
as live processing, and a sustained break emits `no_trade`/failed acceptance.

## 7. Sprint 3 — Replay and ablation harness

**Goal:** Make “does dealer positioning add value?” a one-command reproducible experiment.

Create:

- `strategies/intraday_structure/scripts/build_replay_dataset.py`
- `strategies/intraday_structure/scripts/run_router_ablation.py`
- `strategies/intraday_structure/tests/test_router_ablation.py`

Fixed arms:

1. Upstream candidate at the existing baseline entry.
2. Price/auction rules only.
3. Price/auction plus market/sector context.
4. Price/auction plus static dealer map.
5. Price/auction plus dealer-map changes.
6. Full rules plus ML score, once trained.

Keep the candidate set, dates, entry delay, costs, and label windows fixed across arms.

Report:

- event/trade count;
- target-before-invalidation rate;
- average and median R;
- MFE and MAE;
- time to target;
- false-break/failure rate;
- expectancy after modeled underlying costs;
- option quote coverage and option results separately;
- setup, regime, symbol, DTE, and time-of-day cohorts;
- dealer-data missing/stale cohorts;
- calibration by router/ML confidence.

Split protocol:

- chronological train/validation/test;
- purge overlapping label windows;
- embargo adjacent days when required;
- group confidence intervals/bootstrap by trading day or week;
- never tune on the final test.

**Sprint 3 completion gate:** one command rebuilds the dataset and produces a deterministic report
whose price-only and dealer arms use identical candidate events.

## 8. Sprint 4 — ML layer

## 8.1 Should ML be used?

Yes, if suitable historical point-in-time data is purchased. The correct use is:

```text
Rules define the setup and prevent the model from searching arbitrary noise.
ML estimates whether this particular setup is worth taking.
Dealer and market structure become explanatory features.
An independent policy selects or rejects the option contract.
```

This reduces the search space, improves interpretability, and produces natural ablations.

Do not begin with an end-to-end “buy call/put now” model over every minute and symbol. That design
would be dominated by negative examples, regime changes, execution labels, and selection leakage.

### 8.2 Model A — Setup quality

Create:

```text
strategies/intraday_structure/ml/
  __init__.py
  dataset.py
  train_setup_quality.py
  scorer.py
  metrics.py
  tests/
```

One training row per rule-fired setup, at the exact trigger/confirmation time.

Primary label:

```text
target_before_invalidation_within_H
```

Secondary labels:

- time to target;
- maximum favorable excursion;
- maximum adverse excursion;
- realized volatility over the planned holding window;
- immediate continuation versus chop;
- target 2 reached after target 1.

Candidate models:

1. Rules confidence alone.
2. Regularized logistic regression.
3. LightGBM or XGBoost.

The tree model is adopted only if it improves over both simpler baselines out of sample.

Feature groups:

- upstream candidate/ranker score and provenance;
- setup type and level type;
- auction interaction features;
- session/time-of-day;
- multi-day regime;
- SPY/QQQ/sector/breadth context;
- dealer map levels, normalized distances, concentration, and change features;
- liquidity and volatility;
- no contract outcome, future quote, or revised OI information.

Model output:

- calibrated probability of target before invalidation;
- expected time-to-target bucket;
- expected MFE/MAE quantiles;
- reason codes/feature contributions;
- explicit insufficient-data result.

### 8.3 Model B — Instrument/contract policy

Train only after Model A/rule events have point-in-time option quotes.

Inputs:

- Model A outputs;
- target and invalidation distance;
- DTE;
- delta and moneyness;
- bid/ask spread and quoted size when available;
- IV, IV rank/term structure/skew if available;
- implied move hurdle;
- OI and volume;
- time of day and time remaining;
- setup and dealer regime.

Possible outputs:

- reject option and use shares/observe;
- choose DTE bucket;
- choose delta/moneyness bucket;
- choose naked long versus debit spread;
- estimate probability that executable option return clears costs before invalidation/expiry.

Do not optimize raw option return using stale trade bars.

### 8.4 ML acceptance gate

Adopt a model only if it:

- improves log loss/Brier score and calibration over rules confidence and logistic regression;
- improves top-ranked setup expectancy on validation and untouched test;
- remains positive after realistic costs;
- works in multiple calendar/regime blocks rather than one symbol/week;
- has adequate setup counts per cohort;
- survives dealer-feature ablation;
- does not require features missing from live inference;
- reproduces the training feature calculation exactly in replay and live scoring.

If ML fails, the rules-first router remains a complete valid endpoint.

## 9. Historical data acquisition plan

Buying “historical options data” is not enough. The sample must prove that it contains the fields and
timestamps needed for the intended question.

### 9.1 Minimum useful package

For a narrow first universe such as SPY, QQQ, IWM, and 10–20 liquid equities:

#### Underlying

- one-minute or finer trades/bars;
- extended hours;
- corporate-action handling;
- exchange/vendor and availability timestamps.

#### Options chain/dealer features

- full contract identifier, strike, expiration, and call/put;
- point-in-time OI with its effective/observation date;
- bid, ask, and sizes;
- trade price and size if flow is studied;
- IV and Greeks, or sufficient inputs to reproduce them;
- underlying mark synchronized to the chain;
- timestamps and correction/cancel metadata.

#### Execution

- one-minute NBBO snapshots at minimum;
- trade plus contemporaneous NBBO if testing flow classification;
- quote condition and crossed/locked-market handling.

### 9.2 Vendor sample gate before purchase

Request a small sample covering:

- one volatile session;
- one quiet/pinning session;
- SPY plus one liquid equity;
- at least one 0DTE and one longer-dated expiration.

Reject the source unless the sample passes:

- timestamps and timezone are unambiguous;
- OI effective date is documented;
- bid/ask are populated at required decision times;
- underlying and option clocks can be aligned;
- contracts and corporate actions map correctly;
- option returns have economically correct directional correlation;
- spreads and identical-mark rates are plausible;
- entitlement covers the actual delivered fields, not merely the endpoint name.

### 9.3 Data tiers

| Tier | Use |
|---|---|
| Underlying 1m + daily/EOD chain | Rules and static dealer-map study |
| Intraday chain snapshots + OI/Greeks | Dealer map-change study |
| One-minute option NBBO | Contract policy and option P&L |
| Trades with contemporaneous NBBO | Flow/sweep classification |

Buy only the tier required for the registered experiment. True OPRA flow is not necessary for the
first dealer-context ablation.

## 10. Implementation order

### Day 1

1. Add append-only bar/candidate/router capture.
2. Add setup/outcome ledger and reconciliation.
3. Add capture audit and focused tests.
4. Run a synthetic end-to-end replay.

### Day 2

1. Add auction-context features.
2. Add sweep/reclaim detector.
3. Add typed router decision.
4. Add dealer calculation metadata.
5. Run detector/router causality tests.

### Day 3

1. Add replay dataset builder and fixed ablation runner.
2. Add ML dataset schema, logistic baseline, and tree-model trainer.
3. Add model registry/scorer with fail-closed version checks.
4. Produce a baseline report from currently available underlying data.

### Day 4, if historical option data is available

1. Normalize the vendor sample.
2. Run source-fitness checks.
3. Join quotes/chains strictly as-of.
4. Add contract-policy dataset and quote-based paper evaluation.
5. Run the first rules-versus-ML and price-versus-dealer experiment.

## 11. Definition of done

The upgrade is implemented when:

- one command builds a point-in-time replay dataset;
- live and replay generate the same router decisions for the same event stream;
- every confirmed setup has a reconciled open or closed ledger record;
- missing/stale dealer or option data produces explicit warnings/no-trade reasons;
- the fixed ablation report compares identical candidate events;
- ML training is chronological, reproducible, calibrated, and optional;
- option returns use executable quotes/marks;
- all focused tests pass;
- no live-order path was enabled or modified.

## 12. Immediate first implementation slice

Start with Sprint 1 and the router schema together:

1. Define the immutable `RouterDecision` and ledger schemas.
2. Persist candidate-scoped one-minute bars and router decisions.
3. Close existing running setups into a real outcome ledger.
4. Build the replay materializer and prove live/replay parity.

This slice unlocks every later rule, dealer, ML, and option experiment and can be completed without
waiting for purchased data.

