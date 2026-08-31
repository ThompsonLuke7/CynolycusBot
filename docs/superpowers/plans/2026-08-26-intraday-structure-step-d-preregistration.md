# Pre-Registration: Does the Intraday Structure layer add anything?

**Created:** 2026-08-26
**Status:** REGISTERED — no analysis has been run against these targets
**Implements:** Step D of `docs/TRADE_PLAN_ENGINE_ASSESSMENT_2026-08-25.md`
**Format follows:** `2026-08-26-gamma-structure-preregistration.md`, deliberately
**Readout:** D1 not before **~2026-11**; D2 gated on a collector that does not exist; D3 runnable once D1's prerequisites land

> Why this is written before the data exists. The closed-setup ledger
> (`ledger.py`, shipped 2026-08-26) starts empty and fills forward. That is
> exactly the window in which thresholds get chosen after the fact. Writing the
> targets down now, while the answer is unknowable, is the only moment this
> document can do its job.

---

## 0. Three studies, three units of observation

| | **D1 — conditional outcome** | **D2 — counterfactual ablation** | **D3 — does confirmation add to the call wall?** |
|---|---|---|---|
| Question | Within the shipped config, does level evidence predict setup outcome? | Would the same candidates have done worse without levels / without dealer context? | Does an intraday-structure confirmation improve on the call-wall edge Study A measured? |
| Unit | one **closed setup** | one **closed setup**, replayed per arm | one **call-wall touch** |
| Data | `closed_setups.jsonl` (live) | replayed 1m bars — **archive does not exist** | Study A event table × intraday-structure transitions |
| Sample today | 0 | 0 | 239 call-wall touches / 41 sessions |
| Binding limit | session clusters | the missing collector | whether the engine watches the 5 ETFs |
| Status | gated on sample | **blocked on Phase A collector** | runnable after a small wiring fix |

**D3 is the one that answers the question actually being asked.** Study A
established that price rejects at call walls +9.3pp over an ordinary strike, and
noted the level alone is not a trading edge — "a trader running at 90% is adding
selection on top." The intraday structure engine *is* a selection layer. D3 tests
whether that selection is worth anything on top of the level.

---

## 1. What is already known, and must not be re-litigated

From `research/gamma_levels/00_STUDY_A_FINDINGS.md` (confirmatory arm, passed):

* Call wall: **+9.3pp [+3.0, +15.3]**. Graduates.
* Put wall: +1.0pp [-4.5, +6.4]. **Nothing.**
* Magnet: +1.5pp [-3.2, +6.1]. **Nothing**, and was an artifact of classifying at arrival.
* Base rate at an ordinary strike: **61.6% rejection.** Any claim must clear it.
* SLV: **-11.2pp [-20.6, -1.8]** — significantly negative. Not tradeable on this study.

Two consequences are treated as settled inputs here, not as open questions:

1. **Strike classification must come from a snapshot ~10 minutes before arrival.**
   Classifying at arrival is circular — gamma peaks at the money, so the magnet is
   the nearest strike 31-70% of the time. Any feature this project derives from a
   level must respect the same rule.
2. **Put walls and magnets carry no measured information.** The intraday
   structure engine currently weights `options_put_wall` at 0.90 and magnets at
   0.70-0.78 — the put wall equal to the call wall and above prior-day high
   (0.82). That weighting is now known to be unsupported. Correcting it is a
   **wiring change proposed separately**, not a finding of this study.

---

## 2. D1 — conditional outcome study

### Hypothesis

**H1-D1.** Among closed setups, outcome depends on the evidence behind the
target level: setups whose level was backed by a call wall, or by more
independent mechanisms, reach target-before-invalidation more often.

**H0-D1.** Outcome is independent of level backing.

### Unit and outcome

Unit: one row of `closed_setups.jsonl`.

**Primary outcome, fixed now:**

```
hit_target = mfe_points >= abs(targets[0] - entry_price)
```

Computed from stored fields, not from state-machine bookkeeping — `terminal_state`
depends on archival ordering and would drift if the lifecycle changes.

**Secondary outcomes** (reported, never promoted to headline): `net_return`,
`realized_r_after_costs`, `mae_points / risk_points`.

### The three contrasts — these and no others

| # | Treated | Control |
|---|---|---|
| **C1** | `target_level_sources` contains `options_call_wall` | dealer data available, but no call wall on the level |
| **C2** | `len(target_level_sources) >= 3` | `len(target_level_sources) == 1` |
| **C3** | `dealer_plate_qualified` is true | dealer data available, not qualified |

**C1's control is the whole point.** Comparing call-wall-backed setups against
*all* other setups would compare option-covered liquid names against everything
else and credit the call wall for the liquidity. The control is restricted to
setups where a dealer snapshot was available and fresh (`options_source != "none"`
and no `dealer_summary_stale` warning), so the contrast is within one coverage
universe.

Each contrast is judged on its own pre-declared threshold. **The best of three is
not the headline.** If one passes and two fail, that is one pass and two failures.

### Conditioners — two, taken from Study A

`level_persistence` and `gamma_concentration`, the two that survived Study A's
exploratory split (+6.3pp and +7.1pp).

**Neither is currently recorded in the ledger.** They are a prerequisite: until
`closed_setups.jsonl` carries them, the conditioning analysis cannot run and must
not be improvised from a live snapshot at analysis time — that would reintroduce
exactly the circularity Study A had to correct for.

### Sample minimum

All three must hold before the confirmatory contrasts are run:

* **1,500 closed setups**, and
* **60 distinct sessions** (this is the binding constraint — inference clusters
  by session, so 1,500 setups over 10 sessions is 10 effective observations), and
* **at least 200 setups in the smaller arm of each contrast.**

The third exists to stop a 12-row treated group producing a headline.

### Inference

Session-clustered bootstrap, 2,000 draws, matching Study A so the two are
comparable. **Ticker-clustered errors reported as a sensitivity** — several
setups on one ticker in one session are plainly not independent.

### Decision rule, fixed in advance

| Outcome | Decision |
|---|---|
| Gap ≥ **+8pp** with a session-clustered interval excluding zero | The feature graduates to a level-strength weight or a confirmation gate. |
| 0 < gap < +8pp | Recorded as real but immaterial. **No wiring change.** Do not re-specify to chase it. |
| Gap ≤ 0 | H0 accepted, published as a null in `research/`. |

+8pp matches Study A's registered threshold on purpose: a level-derived feature
should have to clear the same bar in the engine that it cleared in the study.

### Out-of-sample requirement

Following Study A's own closing instruction: after the confirmatory run, the
identical analysis re-runs unchanged on subsequent sessions. **No wiring happens
until the effect holds out of sample.** A halving is a 60-session regime, not a
result.

---

## 3. D2 — counterfactual ablation (blocked, and honestly so)

The original step D was four arms: price-only, +session levels, +dealer, +market.
That is a *counterfactual* — the same candidates replayed under different
configurations — and it cannot be computed from a live ledger, which observes one
arm only.

It requires **candidate-scoped immutable 1-minute bar history**, the gap named in
`docs/DEALER_POSITIONING_GAP_ASSESSMENT_2026-07-28.md` §3 and still open on
2026-08-26. `Data/shared/bars/` holds 1d/1h/4h; there is no 1m archive for
candidate names.

**Registered prerequisite — the collector:**

* Append-only 1-minute OHLCV for every ticker holding an active candidate, plus
  SPY/QQQ/VIXY and sector ETFs, extended hours included.
* Event time, arrival time, and capture time recorded separately.
* A daily manifest binding candidates, bars, dealer snapshot version, config
  hash, and code revision.
* Daily quality report: missing bars, duplicates, session gaps, late bars, halts.

Until that exists, **D2 is not runnable and no partial version of it should be
reported.** Replaying today against bars reconstructed after the fact would be a
lookahead study wearing an ablation's clothes.

Arms, fixed now so they cannot be chosen later: **A** price-only ·
**B** A + session/auction levels · **C** B + dealer levels · **D** C + market/sector.
Primary readout is Δ(hit_target) and Δ(mean R) between adjacent arms, session-clustered.

---

## 4. D3 — does structural confirmation add to the call wall?

This is the study the wiring thesis rests on, and it is nearly free: both datasets
already exist.

### Hypothesis

**H1-D3.** Among call-wall touch events, those where the intraday structure engine
had a `CONFIRMED` setup in the corresponding direction reject more often than
those where it did not.

**H0-D3.** Structural confirmation adds nothing to the call-wall base rate of 69.9%.

### Design

* **Unit:** one call-wall touch event from `research/gamma_levels/data/study_a_events.parquet`
  (n=239 treated events, 41 sessions).
* **Treatment:** an intraday-structure setup in state `CONFIRMED` on the same
  symbol, in the direction consistent with rejection, with `state_entered_at`
  within **[touch − 15 min, touch + 5 min]**. The asymmetric window is deliberate:
  a confirmation *after* the outcome window opens is not available to a trader.
* **Outcome:** Study A's own rejection/penetration label, unchanged. It is not
  re-derived; re-deriving it would let this study drift from the one it builds on.
* **Primary statistic:** rejection rate | confirmed − rejection rate | not
  confirmed, among call-wall touches only, session-clustered.

### Prerequisites, both small

1. **The engine must watch the ETFs.** SPY, QQQ and IWM are in the top-50 ADV
   liquidity seed today; **GLD and SLV are not.** SPY and QQQ are also
   `context_symbols`, which does not exclude them from candidate evaluation.
   Add GLD and SLV to `manual_watchlist` so all five carry setups. (SLV is
   retained despite its negative Study A result — dropping it after seeing that
   result is exactly the selection this document exists to prevent.)
2. **Transitions must be joinable to touches.** `transitions.jsonl` carries
   symbol, state and timestamp, so the join is available now for sessions after
   the engine started (2026-07-21). The overlap with Study A's window
   (2026-06-12 → 2026-08-25) is therefore **partial**; effective n must be
   reported on the overlap, not on Study A's full 41 sessions.

### Sample minimum and power

Registered minimum: **120 call-wall touches inside the overlap window, with at
least 40 in the smaller arm, across at least 30 sessions.**

**Stated in advance:** with roughly 6 call-wall touches per session and a
confirmation rate that is currently unknown, the treated arm may stay small for
months. If the honest answer is "underpowered", that is the finding, and the
response is more sessions — not a wider matching window and not a lower threshold.

### Decision rule

| Outcome | Decision |
|---|---|
| ≥ +8pp, interval excluding zero | Confirmation adds to the level. Wire the call wall as a gated level-strength weight. |
| 0 < gap < +8pp | The engine is not adding to the level. Record; do not wire. |
| ≤ 0 | Confirmation **subtracts**. This is a real possible outcome and would say the engine's filters are removing the good touches. Publish it. |

The bottom row deserves emphasis. The engine declines ~47% of detector-cleared
setups and, on SPY, 99.8% of them for `invalidation_wider_than_max_atr`. A filter
that aggressive can easily be discarding the very events the level edge lives in.

---

## 5. Known ways this produces a false positive

Listed now so that finding one later is a check, not a rescue.

1. **Coverage confounding (D1-C1).** Call walls exist only where option data
   exists — bigger, more liquid names. Mitigated by restricting the control to
   dealer-covered setups; **not** eliminated, since coverage still correlates with
   size within that set.
2. **Selection on the engine's own gates.** The ledger contains only setups that
   passed R:R, runway, and detector confirmation. Every D1 result is conditional
   on that filter and must say so.
3. **Non-independence within ticker-session.** Up to nine setup types can run on
   one ticker simultaneously off one candidate. Session clustering is the primary
   defence; ticker clustering is reported alongside.
4. **One regime.** The ledger starts 2026-08 and Study A covers mid-2026. Any
   positive result is a single-regime result.
5. **Cost-assumption drift.** Rows stamp the spread/slippage they were priced
   under and `reporting.py` refuses to blend groups. If the config changes
   mid-sample, the affected rows are analysed separately or excluded — never
   silently pooled.
6. **The throughput change of 2026-08-26.** Fixing the setup-revival defect raised
   SPY closed setups roughly 15x on a fixed window. Rows written before and after
   that fix describe different engines. **The ledger's analysable sample starts at
   the first session after the fix is deployed**, and earlier rows are excluded.

---

## 6. What must be built before any of this runs

| Prerequisite | For | Size |
|---|---|---|
| `level_persistence` + `gamma_concentration` on ledger rows | D1 conditioners | small |
| GLD, SLV added to `manual_watchlist` | D3 coverage | trivial |
| Touch↔transition join script | D3 | small |
| Candidate-scoped 1m archive + manifest | D2 | **large — the real blocker** |

None of the analysis harness should be built before the sample approaches its
minimum, for the reason the gamma pre-registration gives: building it early
creates the temptation this document exists to remove.

---

## 7. Amendments

Any deviation from this document is recorded here with its date, above the
results, never folded silently into the method.

### Amendment 2026-08-27 — feasibility measured, minimums fixed, nothing run

Ran the arm-size checks for the Study A out-of-sample re-run and for D3. **No
outcome was read in either case.** Both fall short, and the two shortfalls are
different in kind.

**Study A out-of-sample re-run — 2 of ~20 sessions.**
Study A's window closes 2026-08-25. The ladder archive now reaches 2026-08-27,
so genuinely fresh sessions number **two**. Measured from Study A's own event
table, call-wall touches resolve at **5.8 per session**. Scaling its published
interval (±6.15pp on 239 resolved events):

| Target half-width | Resolved events | Sessions |
|---|---:|---:|
| ±15pp | 40 | ~7 |
| ±10pp | 90 | ~16 |
| **±8pp** | **141** | **~24** |
| ±6pp | 251 | ~43 |

Two sessions gives roughly ±27pp against a +9.3pp effect — uninformative.

**Registered minimum for the re-run, fixed now: 20 sessions and 120 resolved
call-wall events.** That buys about ±8.7pp: enough to separate the effect from
zero, and — stated plainly in advance — **not** enough to separate +9pp from
+5pp. Distinguishing "held" from "halved" needs ~60 sessions. The re-run
therefore answers "is it still there", not "is it still that big". Earliest
readout ~2026-10.

**D3 — the treated arm is 2 events, not 40.**

| | measured |
|---|---:|
| ETF confirmations in the transition log | 35 |
| call-wall arrivals in the overlap | 409 |
| ...resolved within 30 min | 138 |
| sessions in overlap | 21 |
| **treated (confirmation nearby)** | **2** |
| control | 136 |

The shortfall is not resolution or window length; it is that the engine almost
never has an opinion at the moment a call wall is touched. Over 21 sessions and
five ETFs it confirmed 35 times — 0.3 per symbol-session — and only twice within
15 minutes of a call-wall touch.

**This is a finding about the engine, not about the level.** Two signals that do
not co-occur cannot be combined, and no amount of analysis fixes that. The
diagnosis is the setup-revival defect fixed on 2026-08-26: 77 of the 392 ETF
transitions were `CLOSED`, and before the fix a closed setup under a live
candidate never came back.

**Consequence, registered:** D3 is re-checked only after the fix is deployed and
at least 20 fresh sessions have accumulated. The expected post-fix confirmation
rate is roughly an order of magnitude higher (a fixed 16-session SPY replay went
from 3 to 44 closed setups), which would put the treated arm in range — but that
is an extrapolation from one symbol under a synthetic candidate feed, and it is
recorded here as a prediction to be checked, not as a basis for acting.

`research/gamma_levels/d3_confirmation_overlap.py` performs this check and
**refuses to read outcomes below the registered minimum.**

### Amendment 2026-08-28 — sample clock, direction semantics, and a long-only engine

**1. Adding tickers does not reset the clock (correcting the 2026-08-27 note).**
The earlier amendment said a candidate-population change restarts the analysable
sample. That was too blunt. Outcomes are per setup and setups are per ticker, so
adding a ticker does not contaminate the tickers already accumulating — it only
starts a clock for the ticker added. **D1 is analysed within ticker**, and each
ticker contributes from its own first post-change session. Pooled numbers report
the per-ticker start dates alongside. Nothing accumulated before 2026-08-28 is
discarded.

**2. The direction filter was missing from the implementation.** §4 registers the
treatment as a confirmation "in the direction consistent with rejection", and
`d3_confirmation_overlap.py` was matching **any** direction. At a call wall —
resistance above spot — rejection is price turning back *down*, so the
rejection-consistent setup is a **short**. Counting longs as treated would score
the engine as correct when it had said the opposite of what happened. Fixed.

**Applying the registered filter, the treated arm is 0, not 2.** Both apparent
hits were long confirmations into resistance.

**3. Why it is zero: the engine is long-only by construction.**

| candidate feed | long | short |
|---|---:|---:|
| `dealer_level_map` | 77 | 19 |
| `meta_ranker` | 34 | 0 |
| `4h_swing` | 24 | 0 |
| `high_liquidity_universe` | 22 | **0** |
| `dealer_ranker` | 7 | 0 |
| `30m_swing` | 6 | 0 |
| `momentum_expansion` | 4 | 0 |

Six of seven feeds hard-code `Direction.LONG`. Across the whole transition log,
confirmations run **1,085 long to 174 short**. The `structural_rejection`
detector can fire either way, but it needs a short *candidate* to exist first,
and on the broad watchlist none ever does.

So D3's shortfall was never a sample-size problem waiting on time. **The engine
could not have produced the treated observation at all**, and waiting 17 months
would not have changed that.

**Collection change, made 2026-08-28:** the five Study A ETFs are seeded into
`manual_watchlist` **on both sides**, and `ManualCandidateFeed` re-seeds them
once per ET session — the previous helper was long-only and registered once per
*process*, so with a 1,440-minute TTL a hand-picked symbol expired after a day
and never returned. D3's clock starts at the first session after this is
deployed.

**Not changed, and flagged for a decision:** `LiquidityCandidateFeed` remains
long-only. Making it two-sided would double the broad candidate set against a
`candidate_limit` of 130 and a measured ~33 ms/bar/candidate, so it is a load
decision as much as a research one and is not made here.

### Consequence for wiring

The proposed order was: (1) re-run Study A out of sample, (2) run D3, (3) wire
conditionally. Steps 1 and 2 are both short of their minimums, so **step 3's
condition is not met and no weight is changed.** The put-wall (0.90) and magnet
(0.70-0.78) weights stay as they are for now, wrong as they look, because
changing them on one unreplicated 41-session sample is the same error in the
opposite direction.
