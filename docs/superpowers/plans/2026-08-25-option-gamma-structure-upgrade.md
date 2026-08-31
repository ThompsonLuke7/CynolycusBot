# Option Gamma Structure — Representation, Uncertainty, and Incremental Value

**Created:** 2026-08-25
**Status:** Phases 0 and 1 IMPLEMENTED 2026-08-26. Phase 2 registered, readout deferred (see `2026-08-26-gamma-structure-preregistration.md`).
**Canonical module:** `strategies/dealer_positioning/` (extension — **not** a new module)
**Operating mode:** Research and paper only
**Relationship to prior plans:** complements `2026-07-28-dealer-auction-upgrade.md` (auction router,
replay ledger). This plan covers *what gamma means and what we predict with it*. Do not merge the two.

---

## 1. Verdict

The proposal is directionally right and its core reframing is correct:

> GEX carries little information about μ and possibly meaningful information about σ and the tails.

That is a **target change, not a feature change**, and it is the single most valuable idea in the
document. Everything else is supporting work.

Three qualifications from reading the code:

1. **About half of the proposal already exists**, some of it in more rigorous form than proposed.
   Two of its best ideas — node-movement velocity and multi-scenario positioning — are already
   **built and orphaned**. Building them again would be the worst possible outcome.
2. **The binding constraint is data, not engineering.** 34 nightly cross-sections and 41 intraday
   days. No amount of feature engineering fixes that, and the proposal's own warning about
   multiple-comparison bias applies with full force at this sample size.
3. **This must not become a new module.** The 2026-07-28 gap assessment already concluded
   "do not create another parallel engine," and a second gamma pipeline would fork the one
   calculation four consumers depend on.

**On ML:** the user's instinct is right. Phases 0 and 1 are deterministic feature engineering with
no model. ML appears only in Phase 2, and there it is a *measurement instrument* — the thing that
answers "does gamma add anything over trailing RV and IV" — not a trading model.

---

## 2. What already exists

Audited against the code on 2026-08-25.

| # | Proposal | Status | Evidence |
|---|---|---|---|
| 1 | Keep walls/magnets, reframe as topology | **Naming only** | `levels.py:177-205` computes them; only the label is wrong |
| 2 | Uncertainty on every observation | **Missing** | partial inputs exist (`stale_days`, `print_coverage`, `option_data_age_days`) |
| 3 | Unsigned gamma structure separately | **Mostly exists** | `total_abs_gex`, `gamma_density_5pct/10pct`, `gex_concentration_index`, `gex_entropy`, `vacuum_above/below` in `_matrix_features` |
| 4 | Several positioning scenarios + dispersion | **Exists in research, not live** | `research/options_lab/gex_reconstruct.py` implements three labeled `oi_source` variants; no dispersion metric, no live path |
| 5 | IV sensitivity / wall stability | **Missing entirely** | no `iv_shock` / stability field anywhere |
| 6 | Node movement / velocity | **BUILT, ORPHANED** | `build_level_dynamics.py` already emits `wall_change_1d/3d`, `gamma_flip_velocity`, `callwall_velocity_3d`, `level_stability_days`, `distance_to_call_wall_atr` — **zero consumers** |
| 7 | Intraday chain snapshots | **DONE for 5 ETFs** | `DealerPositioningRunner`, 60s poll, SPY/QQQ/IWM/GLD/SLV, DTE 0-2, ±4% window — 13,481 SPY ladders over 41 days (06-12 → 08-25). No Δ features derived from any of it |
| 8 | Expiry buckets | **Partial** | nightly has 3 expiry *scopes*; `_per_dte_levels` emits only D0/D1; no term-structure features |
| 9 | Catalyst/regime gating | **ARCHITECTURE BUILT, DARK** | `policy/rules.py:334` applies a `dealer_regime` multiplier, but `adapt_dealer_state` has no caller, so every decision records `CONTEXT_DEALER_UNAVAILABLE` |
| 10 | Per-symbol confidence priors | **Missing** | no liquidity-tiered reliability anywhere |
| 11 | Change what you backtest (σ, tails) | **Missing entirely** | no realized-vol target in any dealer code path |
| 12 | Confound test vs trailing RV | **Missing entirely** | — |

**Summary:** items 1, 3, 6, 7, 9 are substantially done or built-and-unwired. Items 2, 5, 8, 10 are
real gaps. Items 11 and 12 are the ones that actually matter.

---

## 3. Findings that change the plan

### 3.1 The ladder's per-strike gamma columns are cross-expiry means

`build_gamma_ladder` aggregates `call_gamma=("gamma", "mean")` across the 0/1/2 DTE expiries at each
strike ([levels.py:87](../../strategies/dealer_positioning/levels.py#L87)). So `call_gamma` and
`put_gamma` at the same strike are **not comparable** and neither is "the gamma at that strike."

This is currently harmless — `call_gex`/`put_gex` are computed per contract *before* the groupby and
then summed, so the GEX aggregation is correct, and nothing downstream reads the gamma columns. It
becomes actively dangerous the moment someone builds a per-strike gamma or IV-sensitivity feature
off the ladder, which is exactly what proposals 3 and 5 ask for.

**Action:** rename to `call_gamma_mean_by_expiry` / `put_gamma_mean_by_expiry`, or drop them. Any new
per-strike gamma work reads contract rows, not the ladder.

### 3.2 Chain quality is good enough, with two bounded defects

Measured across 200 sampled SPY snapshots (12,122 strike rows):

- Deep-ITM calls carry `gamma = 0` on ~18% of strikes, but those hold **0.1% of call OI** — a
  rounding/vendor artifact with negligible GEX impact. Worth a counter, not a fix.
- IV is per-strike and well-formed (61 unique values per snapshot, call/put IV agreeing to ~4 decimals
  where both exist, consistent with put-call parity). **This is the prerequisite for proposal 5 and
  it holds** — IV-shock recomputation is viable on this feed.

No blocking data defect was found. This is worth stating explicitly given the 2026-07 options-routing
retraction: that failure was stale trade *prints*; this feed is a quote/greeks snapshot and the
GEX inputs check out.

### 3.3 The one module that trades on gamma is losing money

Dealer Ranker: **-$18,434 realized over 21 closes, 37% win rate, profit factor 0.46**, with 12 of 19
exits hitting the -50% option stop. That is a small sample and the loss is plausibly an *option
execution* problem rather than a *signal* problem — but it means the burden of proof for expanding
gamma's influence sits high, and it argues for the proposal's framing (context modifier) over any
expansion of gamma-driven selection.

### 3.4 Sample size is the binding constraint

| Path | Coverage | Effective independent n |
|---|---|---|
| Nightly cross-sections | 34 dates (07-02 → 08-25) | ~34 |
| Intraday SPY/QQQ/IWM/GLD/SLV | 41 days, 13.5k snapshots | ~41 days |

Panel row counts look large but are dominated by cross-sectional and serial correlation. An honest
incremental-value test on future RV needs materially more, and the proposal's own multiple-comparison
warning (2.3% vs 45.4% across 24 thresholds) is precisely the trap a 34-day sample invites.

### 3.5 Interaction mining is closed until ~2027

The 2026-07 confluence study certified **zero** cross-signal interactions, established a power floor
of roughly +7-10pp, and burned the test set until approximately 2027. Proposal 11's interaction
terms (`wall distance × order flow`, `vacuum width × breakout`, …) fall squarely inside that
exhausted space. **Do not run them as a discovery exercise.** One pre-registered pair, at most.

---

## 4. Design decisions

1. **Extend `strategies/dealer_positioning/`.** No new module, no second pipeline. Four consumers
   read the current output; forking it creates a parity break.
2. **Rename the concept, version the calculation.** Artifacts say `estimated_net_gex` with an
   explicit `sign_convention` tag, never bare `net_gex` presented as fact. This was already the
   2026-07-28 gap assessment's recommendation and was never executed.
3. **Separate the trustworthy from the inferred.** Unsigned topology (where gamma is) is high
   confidence. Signed exposure (who owns it) is low confidence. They travel as different fields with
   different confidence, never blended into one number.
4. **Confidence is a documented reliability weight in [0,1], not a probability.** The nervous system
   already forbids mapping uncalibrated values into probability fields.
5. **No new ML in Phase 0 or 1.** Phase 2's model exists to measure incremental value, and its
   output is a Δ metric, not a signal.
6. **Gamma stays a context/target modifier.** Nothing in this plan lets gamma structure select a
   name or set a direction.

---

## 5. Phase 0 — Representation (no new information, no validation required)

These change how existing quantities are named, grouped, and qualified. They are safe because they
add no claim.

### 0.1 Split signed from unsigned

**New:** `strategies/dealer_positioning/topology.py`

```
GammaTopology (unsigned, high confidence)
  absolute_gamma_by_strike, absolute_gamma_by_expiry_bucket
  gamma_density_near_spot (±1%, ±2.5%, ±5%)
  gamma_concentration, gamma_entropy
  gamma_voids (contiguous low-density spans, width + distance)
  distance_to_nearest_node, node_rank_within_window

EstimatedSignedExposure (low confidence)
  estimated_net_gex, sign_convention="oi_calls_long@1"
  estimated_call_wall, estimated_put_wall, estimated_gamma_flip
```

Most inputs already exist in `_matrix_features`; this regroups and relabels them. Keep the current
column names as deprecated aliases for one release — `gate.py`, `options.py`,
`build_dealer_rankings.py`, and the SPY export all read them.

### 0.2 Uncertainty block on every snapshot

**New:** `strategies/dealer_positioning/confidence.py`

| Field | Derived from |
|---|---|
| `structure_confidence` | strike-window coverage, OI coverage, strike-spacing regularity, zero-gamma row share |
| `sign_confidence` | per-symbol liquidity tier prior × cross-convention dispersion (0 until 0.4 lands) |
| `data_freshness` | existing `option_data_age_days` / `stale_days`, normalized |
| `chain_quality` | row counts: total, zero-gamma, missing-IV, repaired |

Per-symbol tier prior (proposal 10), as a config table, not a model:

```
SPY/QQQ  0.60    liquid ETF  0.45    mega-cap single name  0.30
normal equity 0.20    illiquid  0.10
```

These are **stated priors, not estimates**. Label them as such in the config docstring so nobody
later mistakes them for fitted values.

### 0.3 Expiry-bucket separation

Extend `_per_dte_levels` (currently D0/D1 only) to `{0, 1-2, 3-7, 8-30, 30+}` and derive
`zero_dte_gamma_share`, `short_gamma_share`, `weekly_gamma_share`, `gamma_term_slope`.

### 0.4 Rename the misleading ladder columns

Per §3.1.

**Verification:** existing dealer + intraday suites green; a golden-snapshot test proving the new
unsigned fields reproduce the current `_matrix_features` values bit-for-bit where they overlap.

---

## 6. Phase 1 — Wire the orphans and add the two real gaps

### 1.1 Consume `level_dynamics` (already built — proposal 6, zero new research)

`Data/dealer_positioning/level_dynamics/*.parquet` holds wall/flip velocity and stability features
that nothing reads. Join them into `build_dealer_rankings.py` and the SPY daytrader context export.
This is the cheapest item in the plan: the features, the tests, and the no-forward-fill gap handling
already exist.

### 1.2 Publish `DealerState` (already built — proposal 9)

`adapt_dealer_state` is written and tested with no caller, so the policy's `dealer_regime` multiplier
never fires. Wire it into the nightly capture, **carrying `structure_confidence`**, and have a
low-confidence state resolve to a neutral multiplier rather than a missing state. This is what turns
proposal 9's "gamma weight collapses during catalysts" from an idea into the existing policy
machinery — the regime and catalyst states are already published and already multiply.

### 1.3 IV sensitivity and structural stability (new — proposal 5)

Recompute levels under IV × {0.8, 1.0, 1.2} using the existing BSM engine (`research/options_lab`,
already used by `gex_reconstruct`), from **contract rows, not the ladder**. Store:

```
call_wall_stability, put_wall_stability, gamma_flip_stability
node_rank_stability (Spearman across the three surfaces)
estimated_net_gex_sensitivity (spread / base)
```

Cost is three extra evaluations of an existing function per snapshot. This is the best genuinely-new
idea in the proposal: it measures whether a level survives its own model assumptions, and it feeds
`structure_confidence` directly.

### 1.4 Intraday Δ features for the five ETFs (proposal 7's missing half)

**New:** `strategies/dealer_positioning/scripts/build_intraday_level_dynamics.py`

41 days × 13.5k snapshots are already on disk and unused. Derive, at 1/5/15/30-minute horizons:
Δ gamma density, Δ call/put wall, Δ flip estimate, Δ concentration, Δ 0DTE gamma share, Δ IV skew.
Consumer is the intraday structure engine and the SPY daytrader export.

**Verification:** each item ships with a leakage test asserting no snapshot later than the decision
timestamp is readable, matching the existing `DealerLevelSummaryOptionsProvider` contract.

---

## 7. Phase 2 — The incremental-value study (pre-registered, deferred readout)

This is the point of the whole plan. It must be written **before** any of it is run, following
`docs/superpowers/plans/2026-07-25-options-routing-preregistration.md`.

### Targets — dispersion and tails, never direction

**Primary (one, chosen in advance):** next-session realized volatility, close-to-close.

**Secondary (reported, not selected on):** absolute return, high-low range,
P(|move| > 1 ATR), level penetration vs rejection at the nearest node.

**Explicitly excluded:** signed return, direction probability, any μ target.

### Design — confound-controlled, as the proposal specifies

```
Model A (control):   future_RV ~ trailing_RV + ATM_IV + ATR + market_regime
Model B (treatment): Model A + gamma structure features
```

Report out-of-sample Δ only. A Δ AUC of 0.612 → 0.619 is a result worth having; the study is not
looking for more, and finding much more should raise suspicion rather than confidence.

### Guardrails against the failure modes the proposal names

- **One primary target, declared before the first fit.** No threshold search. The 2.3%-vs-45.4%
  arithmetic is the reason.
- **No interaction mining** (§3.5). One pre-registered pair at most.
- **Feature groups declared in advance:** unsigned topology / signed exposure / stability / velocity.
  Ablate by group, not by individual feature.
- **Minimum sample stated up front**, with a readout date rather than a rolling peek. At current
  accumulation the nightly panel reaches a defensible n in roughly Q1 2027; the intraday ETF panel
  gets there sooner because it accrues ~330 observations per symbol per day, but its effective n is
  bounded by day count, not row count.
- **Test set untouched.** Validation and walk-forward only.

### Honest expected outcome

Given the Cboe-sponsored SPX study's ~0.2pp annualized volatility effect on *reconstructed actual*
market-maker positions, the effect available from an OI-sign-assumption proxy on single names should
be assumed smaller. Plan for a small positive Δ or a null. **A null is a publishable, useful result
here** and should be recorded as such rather than re-specified into significance.

---

## 8. What I would not do

| Item | Why |
|---|---|
| Cboe participant-tagged data | Correctly identified as the real fix for sign uncertainty, but expensive institutional data. Revisit only if Phase 2 returns a positive Δ that sign confidence is visibly capping. |
| Interaction-term discovery | Confluence study certified zero interactions; test set burned to ~2027 |
| Any directional GEX signal | The proposal, the academic evidence, and the Reddit critique all agree |
| A new module or engine | 2026-07-28 gap assessment; four consumers already read one pipeline |
| Intraday chain capture for the equity universe | Cost/benefit is poor before Phase 2 reads out; the 5 ETFs are the test bed |
| Tuning a "best GEX threshold" | Named explicitly in the proposal as the trap |

---

## 9. Sequencing and the honest schedule

```
Phase 0  representation, uncertainty, expiry buckets       engineering-bound
Phase 1  wire orphans, IV stability, intraday deltas       engineering-bound
Phase 2  pre-registration written                          engineering-bound
         ...data accumulates...
Phase 2  readout                                           calendar-bound
```

Phases 0 and 1 are worth doing regardless of how Phase 2 lands: they fix a naming error that
misleads every downstream consumer, they light up two subsystems already built and paid for, and
they cost nothing in validation budget because they add no claim.

Phase 2 is calendar-bound and cannot be accelerated by writing more features. Resisting that
substitution is the main discipline this plan asks for.

## 10. Implementation record (2026-08-26)

Phases 0 and 1 shipped. What landed, against what this plan asked for:

| Item | Module | Note |
|---|---|---|
| 0.1 signed/unsigned split | `topology.py` | `GammaTopology` + `EstimatedSignedExposure`; sign convention named in every artifact |
| 0.2 uncertainty block | `confidence.py` | structure / sign / freshness weights; tier priors stated, not fitted |
| 0.3 expiry buckets | `levels.py` | `per_bucket_levels` + `term_structure`; `compute_gamma_levels` contract unchanged |
| 0.4 honest ladder names | `levels.py` | `call_gamma_mean_by_expiry` added; old names kept as aliases for the 13k archived files |
| 1.1 consume level dynamics | `level_dynamics_feed.py` | bounded carry + visible staleness; wired into `build_dealer_rankings` |
| 1.2 publish DealerState | `state_publication.py` | volatility regimes only; low confidence asserts nothing |
| 1.3 IV stability | `stability.py` | levels re-derived under IV x {0.8, 1.0, 1.2} from contract rows |
| 1.4 intraday deltas | `scripts/build_intraday_level_dynamics.py` | 1/5/15/30-min horizons, gap- and session-guarded |

New entry point `levels.compute_gamma_structure` returns all of it together;
`compute_gamma_levels` keeps its two-tuple shape, so the four existing consumers
were not touched.

**Two decisions changed during implementation, both away from what this plan said:**

1. **Low-confidence states assert no regime rather than a neutral one.** The plan
   said publish `NEUTRAL_GAMMA`. That multiplier is 1.0 while `UNKNOWN` is 0.75,
   so publishing neutral on a weak read would have *raised* position size
   relative to today's behavior. Weak snapshots now assert nothing.
2. **The regime mapping is volatility-only by construction.** `SHORT_GAMMA`,
   `PINNING`, `POSITIVE_GAMMA`, `NEUTRAL_GAMMA` — never the two acceleration
   members, which are directional claims this proxy has no standing to make.

**Two defects found and fixed during implementation:**

* The dynamics join silently returned all-nulls: pre-created placeholder columns
  collided in the merge and pandas suffixed the real values away. Caught by a
  test asserting a value, not a shape.
* State publication rejected 36 of 707 symbols by passing a frame-wide capture
  time to an adapter that validates per-row capture evidence. Rows carry their
  own capture time; all 707 adapt now.

## 11. Definition of done

- No artifact presents an inferred dealer sign as observed fact.
- Every gamma observation carries structure, sign, and freshness confidence.
- `level_dynamics` and `DealerState` have live consumers.
- Wall stability under IV shock is stored on every snapshot.
- A pre-registration exists with one primary target, a declared minimum sample, and a readout date.
- No claim of gamma-derived edge without an out-of-sample Δ over the RV/IV/ATR control.
