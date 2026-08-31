# "Level Interaction / Trade Plan Engine" proposal — assessment and plan

Date: 2026-08-25. Written in response to a proposal to add a Level Interaction /
Trade Plan Engine sitting between the rankers and Order Policy.

## Executive take

**Five of the proposal's seven items already exist and have been running live,
paper-only, for 27 sessions.** The module is `strategies/intraday_structure/`.
It is not a sketch: 25,294 transitions from 2026-07-21 to 2026-08-25, 1,159
confirmed setups, 240 target hits.

This is the second time a proposal of this exact shape has arrived. The first
was assessed on 2026-07-28 as the "Auction Playbook Engine"
(`docs/DEALER_POSITIONING_GAP_ASSESSMENT_2026-07-28.md`) and reached the same
conclusion: architecture ~70-80% present, validation data ~25-35% present.
Four weeks later the architecture gap has closed slightly and **the validation
gap has not moved at all.**

The proposal's closing diagnosis — "you've solved the what, he's good at the
where and when" — is half right. The repo already computes the where and when.
What the repo cannot do is **tell you whether its own where-and-when is any
good**, because the engine emits no closed-setup ledger. That is the binding
constraint, and no amount of new signal architecture relieves it.

### What the existing engine's own log says

Reconstructing lifecycles from `transitions.jsonl` (entry proxy = `spot` at
`CONFIRMED`, exit proxy = `spot` at the terminal state), 1,152 matched
lifecycles, outliers trimmed at |return| > 2% (n=1,135):

| metric | value |
|---|---|
| mean return per confirmed setup | **-0.0023%** |
| median | -0.046% |
| win rate | 40.0% |
| median holding time | 3 minutes |

Before commissions, spread, or slippage. Split by exit state, `TARGET_REACHED`
is +0.52% at 98% win and `INVALIDATED` is -0.06% at 23% win — that is the
definition of the two states, not evidence.

**This is not a performance claim in either direction.** The proxy has no entry
delay, no costs, no MFE/MAE, and uses a bar-close spot rather than a fill. It is
the strongest statement the current artifacts can support, and the honest
reading is: *the level-interaction layer runs correctly and has not yet
demonstrated an edge on the underlying.* Wiring it into Order Policy today would
be wiring in an unmeasured component.

## Item-by-item

| # | Proposal | Status | Where |
|---|---|---|---|
| 1 | Level-interaction execution engine (trigger/invalidation/targets, approach→test→hold→confirm→entry) | **Exists** | `strategies/intraday_structure/engine.py`, `detectors/`, 10-state lifecycle |
| 2 | Explicit SIDEWAYS / NO-TRADE classifier | **Partly — decided, never emitted** | see below |
| 3 | Level fusion + confidence score | **Exists** | `levels.py: cluster_levels()` — 12 level families, noisy-OR strength, `sources` list |
| 4 | Breakout-quality classifier `P(continuation)` | **Proxy exists, uncalibrated** | `runway.py: score_runway()` — 6 hand-weighted components |
| 5 | Target-ladder engine (path through liquidity) | **Exists** | `target_manager.py: evaluate_extension()`, obstacle list in `RunwayResult` |
| 6 | Premarket plan generator | **Missing** | — |
| 7 | Separate setup selection from option expression | **Missing at the seam** | selector exists (`core/nervous_system/execution/options/selector.py`) but is not driven by setup geometry |

### 1, 3, 5 — already built, do not rebuild

`cluster_levels()` already produces exactly the canonical object the proposal
asks for — price, strength, type, `sources` — merging session levels (prior
day H/L/C, premarket H/L, opening range, session VWAP, candidate-anchored
VWAP, intraday swings), liquidity zones with touch and rejection counts,
rolling volume-profile HVN/LVN, round numbers, and dealer levels (call/put
wall, gamma flip, magnets). Strength combines by noisy-OR, so a level backed by
four mechanisms does dominate a lone TA pivot — the proposal's point 3 is
already the implementation.

`score_runway()` already returns the next destination *plus the intervening
obstacles*, and `evaluate_extension()` walks the ladder outward. Reward:risk is
computed before confirmation and the setup is refused below 1.25.

The one thing worth noting: the target ladder is built **one rung at a time**
(`TargetPlan.targets` is a 1-tuple; rungs 2 and 3 only appear if price gets
there and the runway re-scores). The proposal's "226.50 → 229.20 → 231.80 →
235.00 published up front" is a presentation difference, and it matters for
item 6 and item 7 — you cannot size an option to a destination you have not
named yet.

### 2 — the real gap is surfacing, not modelling

The engine already abstains, heavily. From the live `state.json` (5,163
setups):

| abstention | count |
|---|---|
| `risk_or_target_plan_unavailable` | 684 |
| `reward_risk_below_threshold` | 327 |
| `runway_below_threshold` | 19 |
| **total blocked confirmations** | **1,030** |

Against 1,159 that passed — the engine refuses roughly 47% of setups that
cleared their detector, on structural grounds. But `engine.py:373-396` handles
each refusal with a bare `return` after appending to `setup.warnings`. Nothing
transitions, nothing is logged, nothing is emitted. **The no-trade decision is
already being made correctly and then thrown away.**

So item 2 is not "train a LONG/SHORT/SIDEWAYS classifier". It is: promote the
existing refusal into a first-class `no_trade_reason` on the signal contract,
and add the two context features the current rule set genuinely lacks —
compression/ATR contraction and failed-breakout count are computed but unused
for abstention; "trapped between nearby support and resistance" is directly
derivable from `cluster_levels()` output and is not currently asked.

### 4 — the classifier is premature, and the proposal has it backwards

`score_runway()` weights six components (distance 0.25, congestion 0.22, level
strength 0.13, trend 0.18, market 0.14, options 0.08). Those weights were
chosen by hand and have never been fit to anything. Replacing them with a model
requires labelled outcomes per confirmed setup — which is the ledger that does
not exist. **Do not build item 4 until item A below has produced data.** ML
here is the last step, not an enhancement to bolt on.

### 6 — genuinely missing and genuinely cheap

Everything a premarket plan needs already exists and is already computed by
09:30: `cluster_levels()` gives the levels, `runway.py` gives the destinations,
the dealer snapshot gives walls/flip, and the 4H rankers publish the candidate
set. There is no assembler and no artifact. This is the highest
value-per-hour item in the whole proposal, and it is a reporting job.

### 7 — the seam does not exist

`intraday_structure` has no order-submission interface at all (deliberate, and
`config.paper_only` is enforced at construction). Separately,
`core/nervous_system/execution/options/selector.py` is a strong deterministic
selector — quote fitness, spread/OI/volume gates, exact expiry payoff, bounded
loss, reason codes on every rejection. What it is *not* given is the setup's
geometry: expected move to target, expected time to target, or target-arrival
probability. It scores the chain, not the trade. The proposal's
`setup → expected move → holding period → arrival probability → strike/DTE`
chain is the correct shape and two of its four inputs do not exist yet.

## Bug found during this assessment

`engine.py:443 _expire_candidates(bar)` expires every TTL-lapsed candidate
across **all** tickers, but stamps each resulting `CLOSED` transition with the
**arriving bar's** price. Evidence from the live log: at
`2026-08-24T13:07:00Z`, `GS` (~$1,040), `APTV` (~$50) and `FIVE` (~$250) all
closed at `spot=133.45` — one arriving bar's price stamped onto three unrelated
setups. Earlier, `IRM` (~$125) closed at `spot=836.94`.

Consequence today: the transition log's `spot` field is wrong on ~140 rows —
cosmetic, because nothing prices off it. Consequence tomorrow: that is the
exact field a closed-setup ledger would use for the exit, so this must be fixed
*before* item A, not after. Fix: carry the setup's own last known bar close, or
emit no price on a non-price-driven close.

## STATUS — updated 2026-08-26

**A, B and C are implemented.** D, E and F remain gated on evidence, as planned.

| Step | State | What landed |
|---|---|---|
| A — closed-setup ledger | **done** | `strategies/intraday_structure/ledger.py`, `closed_setups.jsonl`, `main.py report`; expiry-price bug fixed |
| B — no-trade emission | **done** | `strategies/intraday_structure/regime.py`, `abstentions.jsonl`, `context_regime` + `no_trade_reason` on the signal contract |
| C — premarket plan | **done** | `strategies/intraday_structure/premarket.py`, `scripts/build_premarket_plan.py`, 09:00 ET server slot, dashboard panel |
| D2 prerequisite — 1m bar archive | **done 2026-08-27** | `bar_archive.py`, wired into the runner, on by default |
| D — frozen ablation | gated | needs 6-8 weeks of ledger; the price-only arm now exists as `scripts/run_intraday_structure_baseline.py` |
| E — calibration | gated | needs D |
| F — option expression | gated | needs D plus NBBO data |

Also landed, not in the original plan:

* `strategies/intraday_structure/reporting.py` — reads the ledger and says what
  happened; groups by cost assumption rather than blending, and flags any bucket
  under 30 setups.
* A content-keyed memo on the liquidity-zone level computation (92% of level
  cost, recomputed per bar per candidate and only its last row read). 61.7 →
  33.4 ms/bar, verified bit-identical.
* `replay._trade_frame` now calls the same `build_closed_setup_record` the live
  ledger uses, so a replay row and a live row for the same setup are identical.

### Four defects found while building

1. **Expiry price attribution.** `_expire_candidates` stamped the arriving bar's
   price on every ticker's TTL close. Fixed to price off the setup's own tape,
   with a new `spot_as_of` field recording when the price was seen.
2. **Abstention reasons collapsed.** `risk_or_target_plan_unavailable` covered
   two unrelated failures. In a real baseline run 1,472 of 1,474 refusals landed
   in that one bucket, which told you nothing. Split into
   `invalidation_wider_than_max_atr` and `no_causal_target_beyond_spot` — and
   the split immediately showed all 1,472 were the former.
3. **`trapped_between_levels` fired on 100% of names pre-open.** Not a threshold
   to tune: prior-day high and low bracket spot at roughly one daily ATR *by
   construction*, so the intraday test is a category error on daily bars. The
   test is now disabled pre-open rather than fudged, and a strength floor was
   added so a lone round number cannot act as a wall.
4. **The target ladder had no reachability cap.** Publishing three rungs exposed
   what one rung hid: a $4.61 name got a rung 99.6 ATR away. Capped at the
   `max_target_distance_atr` the dealer-plate policy already used.

### A fifth defect, found BY the new measurement — and the top next action

The one-year SPY baseline produced **5 closed setups over 251 sessions**, against
3 over 16 sessions in a three-week smoke test. That discrepancy is not noise, and
chasing it found a real constraint in the engine.

`engine.register_candidate` creates and revives setups inside a block that is
only reachable when `self.candidates.get(key) is None`. When a candidate for
that `(ticker, direction)` already exists, the function merges and returns early
— **before** the loop that would revive a `CLOSED` setup. The
`_cooldown_complete` / `failure_cooldown_bars` re-arm logic lives inside that
unreachable block.

So a setup that closes can only come back once its candidate has been evicted by
the 1,440-minute TTL. A candidate refreshed more often than that keeps its
setups permanently dark.

`LiquidityCandidateFeed` re-emits the same top-50 ADV names once per ET session,
which is right at that boundary, so the broad watchlist decays. Measured against
the live `state.json`:

| | count |
|---|---|
| setups under a still-live candidate | 630 |
| of those, `CLOSED` and unrevivable | **204 (32%)** |

They do not look dead: `_evaluate_candidate` bumps `updated_at` on every bar
*before* it checks for `CLOSED`, so a permanently inert setup keeps a fresh
timestamp. That is why this survived 27 live sessions unnoticed.

**Not fixed here, deliberately.** Reviving setups increases how much the engine
trades, which is a trading change and has to be a measured one — and it would
invalidate the baseline that just ran. But it is on the critical path: fewer
setups means the ledger accumulates more slowly, which pushes out step D. This
should be the next change, made on its own and measured.

### Update 2026-08-26 — defect 5 fixed, step D pre-registered

`register_candidate` now creates and revives setups on **every** registration
path, not only when the candidate key is absent (`_ensure_setups`). Measured on a
fixed 16-session SPY window: **3 → 44 closed setups (~15x)**, and the headline
bucket clears the n>=30 reporting floor for the first time. 7 regression tests,
5 of which fail against the old behaviour.

That 15x is the extreme case — one candidate held all year. Live candidates from
the rankers already churn and took the fresh-key path, so the live multiplier will
be smaller; the gain is concentrated in the persistent broad watchlist, which is
where the 204 dark setups were.

**Consequence for measurement, registered:** rows written before and after this
fix describe different engines. The analysable ledger sample starts at the first
session after deployment.

Step D is now pre-registered at
`docs/superpowers/plans/2026-08-26-intraday-structure-step-d-preregistration.md`,
in the gamma study's format. It splits three ways, and the split is the
substantive finding: **the four-arm ablation D2 as originally written cannot be
computed from a live ledger at all** — it is a counterfactual, and it needs the
candidate-scoped 1-minute archive that has been an open gap since 2026-07-28.
What the ledger supports is D1, an observational conditional-outcome study. D3,
the combination test against Study A's touch events, is the one that answers the
wiring question and is nearly free.

### Measurement status

The instrument works end-to-end on real data. What it can and cannot say today:

* **Forward:** the live ledger fills from the next terminal setup onward. This
  is the real measurement and it is now running.
* **Backward: not recoverable.** Only 357 of 5,163 setups in `state.json` still
  hold an entry price, and they are the ones whose `setup_id` was never reused —
  median holding 388 minutes against 3 minutes for the full population. That is
  a ~100x survivorship skew, so no backfill was built. The historical
  population is gone; this is precisely the cost of having had no ledger.
* **Sideways:** a price-only baseline over stored SPY 1-minute history gives an
  unbiased read now, at the cost of being one symbol with no ranker and no
  dealer context. One year (251 sessions, 97,512 bars, 55 min of compute)
  returned **n=5** — far below the 30 the report needs before it will let a
  number be quoted, and throughput-limited by the defect above rather than by
  the tape. Artifacts in `Data/analysis/intraday_structure_baseline/spy_1y/`.
  The run is worth keeping as the harness proof and as the price-only arm of
  step D; it is **not** a read on edge and must not be quoted as one.

## Plan

Ordered by dependency. A is a prerequisite for everything except F.

### A. Closed-setup ledger — the missing measurement instrument
**~2-3 days. Do this first.**

* Fix `_expire_candidates` to stop stamping a foreign ticker's price.
* New `strategies/intraday_structure/ledger.py`: one immutable append-only
  record per setup that reaches a terminal state, written from `SetupRecord`
  which already carries `entry_price`, `entry_time`, `max_favorable_excursion`,
  `max_adverse_excursion`, `targets`, `invalidation`, `runway_score`,
  `expected_reward_risk`, `confidence`, `evidence`.
* Fields the record must add beyond that: candidate `sources`, `setup_type`,
  the level `sources` list for the active level, the dealer-plate qualification,
  the market alignment at confirm, and the terminal reason.
* Model the fill honestly: 1-bar entry delay and `ReplayPolicy`'s existing
  spread/slippage assumptions, recorded as assumptions rather than baked in.
* Reconciliation test: every `CONFIRMED` transition produces exactly one ledger
  row, and restart cannot duplicate or drop one.

Output: `Data/inference/intraday_structure/closed_setups.jsonl`, matching the
convention the other four modules already use.

### B. Emit the no-trade decision
**~1 day. Independent of A, but pointless to analyse without A.**

* Add `no_trade_reason: str | None` and `context_regime: str` to
  `StructureSignal`.
* Replace the three bare `return`s in `engine.py:373-396` with a recorded
  abstention (state stays put; a `no_trade` event is appended).
* Add the missing context features to `features.py`: ATR contraction ratio
  (current ATR vs 20-bar median), failed-breakout / failed-breakdown counts
  (already produced by `add_liquidity_zone_features`, currently discarded), and
  a `trapped_between_levels` flag derived from `cluster_levels()` — nearest
  support and nearest resistance both within N×ATR.
* `context_regime ∈ {TRENDING_UP, TRENDING_DOWN, BALANCED, COMPRESSED}` from
  those, by rule. **No model.** The rule version is the baseline that a model
  would later have to beat.

### C. Premarket plan generator
**~2-3 days. Depends on nothing. Highest value-per-hour.**

* New `scripts/build_premarket_plan.py`, run from the combined server at
  ~09:00 ET alongside the existing pre-open flushes.
* For each candidate from the 4H rankers plus the dealer top-N: run
  `StructuralLevelProvider` over the overnight + prior-session bars, run
  `score_runway()` for both directions, and emit the full ladder up front
  (not one rung) with the obstacles between rungs.
* Output shape, per name: `trigger`, `invalidation`, `target_1..3`,
  `context_regime`, `no_trade_reason`, and the level `sources` behind the
  trigger. Plus a MARKET block for SPY/QQQ and an explicit AVOID list — the
  names that were candidates and abstained, *with the reason*, which is the
  part of the proposal that is actually novel for us.
* Two artifacts: a JSON under `Data/inference/intraday_structure/` and a
  rendered panel on the existing 8774 dashboard. Reuse
  `core/shared_plotting` for anything charted.
* Constraint: everything must be computable from data available before 09:30.
  Dealer snapshots are captured ~15:45 the prior day, which is causally fine
  for a premarket plan and must be labelled as prior-session.

### D. Freeze the ablation — does structure add anything?
**~1 week of work, gated on ~6-8 weeks of ledger from A.**

Once A has accumulated, run the comparison the 07-28 doc already specified and
which still has never been run:

1. price-only detectors,
2. + session/auction levels,
3. + dealer map,
4. + market/sector alignment.

Same candidates, same window, same cost assumptions. Report by setup type, by
regime, and by time of day. If (1) and (4) are indistinguishable, the level
fusion is decoration and the honest move is to say so.

**Do not tune thresholds against this until it has run once untouched.**

### E. Calibrate breakout quality — only if D is positive
**~1 week. Explicitly gated.**

Replace `score_runway`'s hand-weights with a fit on the ledger's realised
outcomes: target-before-stop as the label, the existing six components plus the
item-B context features as inputs. Interpretable baseline first (logistic on
the six components) before anything gradient-boosted. Keep the component
breakdown in the output — the current transparency is a feature and must
survive.

### F. Option expression bridge — last, and separately gated
**~2-4 weeks plus data. Do not start before D.**

The seam is: `SetupRecord` → expected move (target - entry), expected time to
target (from the ledger's realised holding times per setup type), arrival
probability (from E) → `selector.py`. The selector already handles everything
downstream of that.

The blocker is unchanged from 07-28 and from the retracted options study: there
is no historical option quote data fit to evaluate this. `AGENTS.md` requires
the derivative-vs-underlying correlation check before any option P&L is
computed, and the last time it was skipped an entire study was retracted. So F
requires either purchased NBBO history or 8-12 weeks of forward mark capture,
and it must not begin until D says the underlying signal is real.

### Not recommended

* **A new module.** `intraday_structure` is the canonical playbook router. The
  07-28 doc says do not create a parallel engine; that still holds.
* **Wiring intraday_structure to Order Policy now.** It is unmeasured. The
  gate is D, not a code change.
* **Item 4 as proposed (a breakout classifier now).** No labels exist.

## Sequencing summary

```
A ledger + expiry fix  ──┬─→ D ablation (needs 6-8wk data) ─→ E calibration ─→ F options
B no-trade emission    ──┘
C premarket plan       ──  independent, ship anytime
```

A + B + C is roughly one week of engineering and turns the existing engine from
unmeasurable into measurable, while shipping the one genuinely missing
user-facing artifact. Everything after that is gated on evidence rather than on
enthusiasm.

## Verification performed

* Read `strategies/intraday_structure/` (engine, detectors, levels, runway,
  target_manager, features, options, config, runner, models) and
  `core/nervous_system/execution/options/` (selector, payoff).
* Counted 25,294 live transitions across 27 sessions and reconstructed 1,152
  confirm→terminal lifecycles from `transitions.jsonl`.
* Counted abstention warnings across 5,163 setups in the live `state.json`.
* Confirmed no closed-setup ledger exists, no `no_trade_reason` exists, no
  premarket plan generator exists, and that `intraday_structure` has no
  order-submission path.
* Confirmed the 4H execution path (`core/live_4h_exec.py`) has no price-level
  trigger — it sizes to notional and submits a limit ladder, which is the
  behaviour the proposal correctly criticises.
