# Stage 0 — preregistration and substrate

2026-09-23. Plan: `docs/superpowers/plans/2026-09-23-rl-reintroduction-staged-plan.md`.
Scripts: `panel.py` (shared substrate), `00_substrate_check.py`. Output:
`data/stage0_calibration.json`, `data/atr_by_session.parquet`.

Everything below is fixed BEFORE Stage 1 runs. Where a number is measured rather than
chosen, it is a property of the data, not of an outcome.

---

## 1. Datasets

### 1.1 Decision panel v2 — rebuilt

`./.venv/bin/python scripts/horizon_thesis/build_decision_panel.py` →
`research/execution_quality/data/decision_panel_v2.parquet` (91s).

| property | value |
|---|---|
| rows (raw / CA-guarded) | 1,567,029 / **1,566,875** |
| corporate-action flagged | 154 rows (0.0098%) |
| decision bars | 1,739 |
| tickers | 1,080 |
| span (decision day) | 2022-11-14 .. **2026-05-13** |
| median cross-section per bar | 952 |
| outcome-column null rate | 0.0000% at h=5/10, rising to 0.0050% at h=30 |

### 1.2 Momentum training matrix — the gate's own population

`strategies/momentum_expansion/data/processed/training_matrix_4h.parquet`:
2,317,972 rows × 117 cols, 6,922 4H bars, **2020-09-08 .. 2026-05-14**, and all seven
candidate-gate features present at **0.0000% null**.

**Correction to the plan's §1.3 assumption.** This matrix holds **1,081 tickers**, not the
3,089 present as per-ticker files in `data/processed/features_4h/`. The momentum pool and
the meta universe are effectively the same ~1,080 names. So the plan's idea of measuring
retrieval on a "broad pre-gate universe" inside this matrix is not available — the broad
population has to come from the daily bar cache, and the gate can only be evaluated on
pool rows, because pool rows are all it ever sees. Stage 1 is restructured into two layers
accordingly (§4).

### 1.3 Daily bar cache — the population, and its limits

4,104 tickers, last bar 2026-09-22. **549 tickers (13.4%) have no bar within 30 days of the
cache's end** — i.e. the cache does retain delisted/halted names rather than being purely
survivors. It remains UNADJUSTED, so the corporate-action guard
(`research/execution_quality/data/corporate_action_flags.parquet`, rebuilt by
`scripts/rank_depth/apply_ca_guard.py` over all 4,104 files) is mandatory on every tail
metric.

---

## 2. The risk unit

`1R = K_RISK × ATR14%` with **K_RISK = 2.0**, ATR from the Wilder EWM (α=1/14) on the daily
cache, taken at the session **strictly before** the entry session. Constant, never fitted.
`panel.py` enforces `atr_pct ≥ 0.002` so 1R cannot become a rounding error. ATR joined on
**100.00%** of guarded panel rows; median ATR% = 3.13%.

Every reward and every headline metric in this series is in R units, unclipped.
`expansion_survival_score` is a within-bar **percentile-rank** composite and therefore
cannot express the right tail this series is about, so it is used only as an incumbent
score to compare against — never as a reward.

---

## 3. Splits, embargo, and the test-set budget

### 3.1 The split

Reused verbatim from `scripts/horizon_thesis/run_horizon_grid.py:199` (`split_bars`):
decision-bar fractions 0.60 / 0.78 with `EMBARGO_BARS = 70` on each side. On panel v2:

| slice | bars → dates | rows |
|---|---|---|
| train | ≤ **2024-12-16** | 915,820 |
| validation | **2025-02-10 .. 2025-08-07** | 233,642 |
| test | **2025-09-26 .. 2026-05-13** | 286,130 |

### 3.2 The test block is NOT fresh — measured, not assumed

`research/execution_quality/24_horizon_thesis_experiments.md:33` records the horizon grid's
test window as **683 bars, 2024-12-26 .. 2026-05-14**, which **fully contains** the block
above. `confluence_discovery_2026-07-07.md:55` consumed **2026-03-11 .. 2026-05-14** and
states plainly "Test set has been consumed — do not re-mine against it."

So there is no unread holdout inside this panel. Consequences, fixed now:

* **Stages 1 and 3 run on train+validation only.** Neither needs a test read: Stage 1
  measures retrieval (not a fitted quantity) and Stage 3's oracle headroom is an
  arithmetic property of price paths. Stage 3's *imitation* arm is graded on validation.
* **Stage 2 and later get at most one test read each**, and any positive test result is
  labelled **UNCERTIFIED** — a re-used block cannot certify anything.
* **The cheapest route to a clean holdout is named here:** the OOF matrix ends 2026-05-14
  while the daily cache runs to 2026-09-22. Refreshing the momentum walk-forward OOF build
  would create **~4.3 months of genuinely unread data** (roughly 180 decision bars). That
  is the certification window for anything this series produces, and it requires no new
  vendor and no new capability. Recommended before Stage 4.

---

## 4. Stage 1 structure (fixed)

Two retrieval layers, measured separately, because they fail for different reasons:

* **L1 — universe.** Is the name in the momentum pool at all (1,081 of 4,104)?
  **Point-in-time pool membership does not exist**: `core/shared_universe/universe.py:236`
  states snapshots only exist from the first build after 2026-09-10. L1 is therefore
  measured against the **current** pool, which is contaminated *in the pool's favour*
  (today's list includes names partly selected because they later did well). A poor L1
  result is therefore informative; a good one is not trustworthy. Stated on every L1 table.
* **L2 — candidate gate.** Among pool rows, does `momentum_candidate_mask` fire? Swept
  one-threshold-at-a-time and as a joint ladder over the seven
  `MOMENTUM_CANDIDATE_FILTER_CONFIG` values, from current to fully open.

Population liquidity is reported at several floors (none / $1M / $5M / $25M median 20-day
dollar volume) rather than one, so "missed tail" can be separated from "untradeable tail".

**The headline statistic is lift (precision ÷ base rate), never recall.** A rule firing on
X% of rows catches X% of events for free; `research/regime_coverage_2026-09-21/recall_precision.py`
found recall equal to firing rate to within 0.5pp on every row it tested. `recall − fires`
is reported alongside.

---

## 5. The right-tail event — calibrated, then frozen

Base rates on the CA-guarded panel, MFE from the first executable price (next session's
open, per `23_rank_depth_and_options.md` §0):

| candidate definition | base rate |
|---|---|
| `rmfe_5 ≥ 3R` | 0.98% |
| `rmfe_10 ≥ 3R` | 3.74% |
| **`rmfe_10 ≥ 4R`** | **1.44%** |
| `rmfe_10 ≥ 5R` | 0.65% |
| `rmfe_20 ≥ 4R` | 5.64% |
| `mfe_10 ≥ 25%` | **4.07%** |
| `mfe_10 ≥ 50%` | 0.73% |
| `rmfe_20` top 1% within bar | **0.95%** |
| `mfe_20 ≥ 15%` | 22.98% |
| `mfe_5 ≥ 10%` | 11.26% |

rmfe quantiles (p50 / p95 / p99 / p99.9): h=5 → 0.50 / 1.78 / 2.98 / 5.70;
h=10 → 0.76 / 2.72 / 4.45 / 8.57; h=20 → 1.17 / 4.16 / 6.81 / 13.50.

**Harness check.** `mfe_20 ≥ +15%` reads **22.98%** here against the **25.87%** that
`recall_precision.py` measured for +15% within 20 sessions on a 2,811-ticker universe. The
two agree to ~3pp across a different universe and a different entry convention, so the
label machinery reproduces the prior result.

### 5.1 Frozen definitions

| id | definition | base rate |
|---|---|---|
| **P1 (primary)** | `rmfe_10 ≥ 4R` | 1.44% |
| **P2** | `mfe_10 ≥ 25%` | 4.07% |
| **P3** | `rmfe_20` in the top 1% of its decision bar | 0.95% |

P1 is primary: risk-normalized (so a 3%-ATR name and a 10%-ATR name are judged on the same
scale), 10 sessions matches the modules' own 25-bar label horizon, and 1.44% is rare enough
that lift is measurable. P2 keeps an absolute, dollar-interpretable arm. P3 is
vol-drift-immune and directly comparable to the ranking studies.

### 5.2 Two of the thread's proposed thresholds are rejected as non-events

The thread suggested "+8% within 3 days" and "+4R within 5 days" among others. In this
universe `mfe_5 ≥ 10%` is **11.26%** of rows and `mfe_20 ≥ 15%` is **22.98%** — these are
not tails, they are the middle of the distribution, and a study built on them would
reproduce the 1.00x-lift null by construction. `rmfe_5 ≥ 4R` (0.35%) *is* a tail and is
retained as a robustness arm only, being thin.

---

## 6. Standing requirements for every later stage

1. **Permuted-LABEL null arm** on anything reporting top-k excess or selection advantage,
   per `25_matched_control_and_depth_null.md` §B.1 and `scripts/horizon_thesis/topk_depth_null.py`.
   A shuffled-*score* null is ~0 and proves only that the harness is clean.
2. **MDE reported with every null result** (AGENTS.md). An underpowered null is "not
   measurable here", never "no effect".
3. **Corporate-action guard applied**, and the answer stated with and without it whenever a
   tail metric is involved.
4. **Full R-unit distribution** (p1/p25/median/p75/p99, share > +3R) alongside every mean.
5. **A failed stage names its next testable variant** or states the hypothesis class is
   exhausted and why.
6. No stage writes to live configuration. `MOMENTUM_CANDIDATE_FILTER_CONFIG` and every other
   live parameter is read-only for this series.

---

## 7. Stage 0 status

**Complete.** Panel v2 rebuilt and verified; ATR cache built (100% join); house split
adopted and its bounds recorded; test-set contamination measured and bounded; the
population's survivorship exposure quantified; the tail event calibrated and frozen; two of
the thread's proposed thresholds rejected on measured base rates; one correction to the
plan's substrate assumption recorded (§1.2) and Stage 1 restructured into two layers.
