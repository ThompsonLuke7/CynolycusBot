# Segmented entry/exit policy study — does policy-by-segment beat one global number?

**Date:** 2026-07-19 · **Status:** research only, nothing wired live
**Scripts:** `scripts/capstone/exit_policy_segmentation.py` (Phases 1–2),
`exit_policy_entry_side.py` (Phase 3), `exit_policy_learned_admission.py` (Phase 4)
**Artifacts:** `research/capstone/segmentation/` (streams, trades, cohort map, all result CSVs)

## Question

The live 4H book applies one global exit policy to every top-10 entry across
Momentum, HTF, and Meta. Prior rounds produced a two-policy frontier —
g284 (harvest +7%, hz 60: best win rate/velocity) vs id4 (trim 16% @ +30%,
hz 53: best mean/tail capture) — from **aggregate** stats. Aggregates are
dominated by the majority (grinder) population, so this study re-evaluates the
frontier **per segment**, tests entry-side levers, and tests a learned policy,
to decide between: (a) ship one global policy, (b) ship policy-by-segment,
(c) no change justified.

## Test-set discipline (read this first)

The meta VAL/TEST split (VAL 2025-07-01→2026-01-15, TEST →2026-05-15) had
already been read 4× before this study. Protocol used here:

- **Nothing was re-searched on exit parameters.** The 4 policies
  (current-live, deployed, g284, id4) are frozen inputs; per-segment breakdowns
  of fixed policies are descriptive, not selection.
- Every **new** decision (Phase 3 admission variants, gate thresholds, Phase 4
  model/threshold) was selected on VAL with a **pre-stated rule written into the
  script before any test simulation**, and the scripts structurally only
  simulate TEST for the pre-registered winner + baseline (one read each).
- Cohort definitions use only pre-VAL data (bars strictly before 2025-07-01)
  or entry-time features, so segment membership is known at entry.
- Cumulative erosion of this window is still real. **Any ship decision should
  be confirmed on the genuinely untouched 2026-05-15→present window** (now ~2
  months of live-era data no study in this thread has ever read).

All entry streams are clean walk-forward OOF (meta rebuilt via
`build_meta_scored_from_oof.py`; momentum/HTF from their own `oof_preds.parquet`),
per leakage_audit §4.3.

## Phase 1 — Segmentation grid

**Tail-propensity cohorts** (the load-bearing axis): KMeans (k=3, seed 7, n_init 20)
on rank-transformed pre-VAL distribution-shape features per ticker
(95th-pct 60-bar forward MFE, median MFE, 4H vol, return skew), computed only
from bars before 2025-07-01; tickers with <250 pre-VAL bars → "young".
Raw-feature KMeans degenerates (a handful of bad-bar artifact tickers with 4H
vol ≈ 0.84 capture whole clusters); the rank transform fixes this and is the
documented deviation.

| cohort | tickers | check names (data-derived, not hand-picked) |
|---|---|---|
| grinder | 381 | AAPL |
| moderate | 243 | MU, WDC |
| explosive | 275 | AXTI, NVDA |
| young (<250 pre-val bars) | 29 | SNDK (spun off Feb 2025) |

The 2025–26 extreme movers fall out of the clustering as intended — and SNDK
landing in "young" is itself a finding: the single biggest live runner had **no
pre-history at all**, so no history-based tail model could have flagged it.

**Theme concentration** (validates the axis): explosive-share by theme —
space_satellites 92%, nuclear_uranium 70%, cloud_neocloud 69%, ai_apps_agents
60%, biotech_innovation 53%; vs datacenter_infra/electronics_components 0%,
construction_infrastructure 5%. Small caps are 67% explosive, mega caps 14%.
Tail propensity is real, persistent, theme-structured — but per-theme trade
cells are almost all n<30, so theme is **descriptive only**, not a policy lever.

**Other axes:** cap_tier (universe file), mom_xs_rank quintile +
dollar_vol_pctile tercile (entry-time matrix), regime from the matrix's own
entry-time features (riskoff = vix_high, bull = spy_trend>0 & not vix_high,
else chop). Regime trade counts: val 2475 bull / 205 riskoff / 38 chop
(**chop-val is underpowered — no conclusions drawn from it**); test 678 bull /
526 riskoff / 222 chop.

Cell sizes for every axis are in `segmentation/cell_sizes.csv`; every claim
below uses cells with n≥30 unless flagged.

## Phase 2 — Policy × segment matrix: the frontier is fractal

Across **97 cells** (axis × segment × module × window, n≥30, all 4 policies):

- **id4 has the higher mean-per-trade in 93/97 cells.**
- **g284 has the higher ret-per-bar in 97/97 and higher win rate in 97/97.**
- **deployed (live params) is dominated in every single cell** — there is no
  segment anywhere in which the live policy is the right choice.
- current-live (pure rebalance) is worst on mean everywhere, as before.

The g284↔id4 trade-off is *not* a majority-population artifact — it replicates
**inside** every cohort, including grinders (test grinder capture-of-MFE:
id4 0.70–0.75 vs g284 0.18–0.21; but g284 rpb 3–5× id4's). The frontier is a
property of the exit mechanic itself, not of the population mix.

**The only 4 reversal cells anti-replicate.** On VAL, id4's mean went negative
in riskoff regime (meta −0.7%, momentum −8.3%) and mom_xs_q2 — the seemingly
obvious "harvest fast in drawdowns" rule. On TEST, riskoff was id4's *best*
regime (mean +13.5% to +18.1%, n=98–168, beating g284's +3.2–4.2%), and q2
flipped positive too. A segment-conditional exit selected on val would have
**hurt** on test. This is the strongest single piece of evidence against
policy-by-segment exits.

**Hybrid cohort→policy maps don't help either.** All 16 {g284,id4}⁴ maps over
the tail cohorts were evaluated (pooled modules): they form a smooth
mean↔velocity interpolation; no hybrid dominates a pure policy on both axes,
and pure id4 has the highest total return on **both** windows (val 236.8,
test 118.5 summed-return units vs 88.3/46.1 for pure g284).

**Phase 2 verdict: (a), not (b).** One global exit policy, chosen by objective:
id4 if the objective is mean/total return and tail capture; g284 if the
objective is capital velocity/win rate under slot constraints. Segmentation
adds nothing robust to the exit side. (This also upgrades the prior "id4 ≥
deployed" claim: deployed is now shown dominated in every segment, both
windows, all three modules.)

## Phase 3 — Entry side

### 3a. Cohort-conditioned admission (extend top-10 → rank≤X for tail cohorts)

Val-selected winners (rule: max total return s.t. mean ≥ 0.8× base) extended
admission to rank≤25 for explosive(+young)(+moderate) names. Frozen test:

| module | base total | ext total | base mean | ext mean | avg book 6.2 → |
|---|---|---|---|---|---|
| momentum | 33.8 | 38.2 (+13%) | 12.0% | 9.6% | 12.7 |
| htf | 48.2 | 62.0 (+29%) | 10.7% | 8.9% | 16.0 |
| meta | 36.5 | 51.0 (+40%) | 9.6% | 8.8% | 13.9 |

Total return rises **less than the capital deployed** (book 2–2.6×) and
mean/trade + ret-per-bar fall in all three modules: this is dilution, not
runner capture. **Rejected.**

Decisively: **MU and WDC do not appear in any extension's added tickers.**
Rank audit over val+test: MU/WDC are inside the **top-40** on only 11–88 of
~800–930 bars per module (momentum: MU never better than rank 15, in top-40
just 25 bars). No admission-depth policy reaches names the ranker scores that
low. The runner-capture failure for mega-cap steady movers is a
**ranker/feature problem** (consistent with meta's gain concentration in
dollar_vol_pctile_252), unfixable at the admission or exit layer.

### 3b. Score-level gate — "must we fill every top-10 slot every 4H?" → No.

Gate: only admit rank≤10 entries whose score clears an absolute threshold set
from the VAL entering-score distribution; exit fixed g284. Val winner per
module (pre-stated rule: max rpb among gates keeping ≥40% of trades), single
frozen test read — **held in all three modules**:

| module | gate | test trades kept | test mean | test rpb |
|---|---|---|---|---|
| momentum | q50 | 250/479 (52%) | 2.02% → **2.44%** | 0.0191 → **0.0256** (+34%) |
| htf | q25 | 589/680 (87%) | 3.29% → 3.26% (≈flat) | 0.0162 → **0.0172** |
| meta | q50 | 388/599 (65%) | 2.34% → **2.75%** | 0.0151 → **0.0172** (+14%) |

Momentum's result is the headline: **cutting 48% of trades improves every
per-trade and per-bar metric**. Cost: gross total return falls with trade count
(momentum test total 9.7→6.1) — the gate is a capital-efficiency/turnover
lever, not a gross-return lever; it frees slots/capital rather than making
more money on the same book. The direction (mean AND rpb up at half the
turnover) held val→test in 3/3 modules, which is the strongest entry-side
result available from this window.

## Phase 4 — Learned admission policy: hypothesis confirmed, it fails

XGBRegressor (fixed small hyperparameters, no tuning), entry-time-safe matrix
features only, trained on val trades entered before 2025-12-10 (embargo: no
label window overlaps test), keep-top-80% threshold fixed from train
predictions, compared against the incumbent mom_xs_rank skip-20% rule at the
same keep fraction, exit fixed id4:

| module | val kept-mean (xgb vs rule vs none) | **test** kept-mean (xgb vs rule vs none) | test skipped-mean (xgb) |
|---|---|---|---|
| momentum | 17.4% vs 14.2% vs 11.3% | 11.3% vs 12.7% vs 12.0% | **+21.3%** |
| htf | 19.2% vs 15.4% vs 13.0% | 10.8% vs 10.9% vs 10.7% | −1.0% |
| meta | 15.5% vs 12.9% vs 10.5% | 9.6% vs 9.9% vs 9.6% | +9.4% |

In-sample the model looks spectacular (val skipped buckets −13% to −18%);
out of sample it is a no-op (htf, meta) or **actively harmful** (momentum: the
trades it skipped averaged +21.3%, including runners). The internal early→late
val checks already flagged it (Spearman −0.07 to +0.10). The simple
mom_xs_rank rule remains the only entry filter with positive out-of-sample
evidence (here: +0.2–0.7pp mean under id4; larger under deployed in the prior
study). **Do not deploy a learned admission model on this signal surface.**

## Final recommendation

**(a) global, not (b) segmented — with one entry-side addition, and one
explicitly unresolved item:**

1. **Exit: ship one global policy; segmentation is not justified.** Every
   robust cut says the g284↔id4 frontier is invariant across segments and the
   only segment-conditional rules val would suggest anti-replicate on test.
   The choice between them is an objective call: id4 for mean/tail capture
   (recommended, consistent with the prior cross-module round), g284 if slot
   velocity/win rate is the binding constraint. The currently-deployed policy
   is dominated in all 97 cells and should be replaced either way.
2. **Entry: the score-level gate (3b) is a validated candidate** — don't fill
   top-10 slots on weak-score bars (momentum q50 / meta q50 / htf q25
   thresholds). Held val→test in 3/3 modules; cuts turnover 13–48% while
   improving per-trade and per-bar quality; reduces gross exposure-time
   accordingly. Given this window's reuse, confirm on 2026-05-15→present
   (untouched) and/or paper-soak before wiring.
3. **Rejected by evidence:** cohort-extended admission (dilution),
   segment-conditional exits (anti-replicates), learned XGB admission
   (overfits, harms momentum).
4. **(c) — unresolved: mega-cap runner capture (MU/WDC).** They sit outside
   the top-40 ~95% of bars; no exit, admission-depth, gate, or learned layer
   can reach them. If capturing that population matters, the work is at the
   **ranker/feature level** (why steady mega-cap trends score low — plausibly
   the dollar_vol_pctile_252 concentration + short-horizon labels), which is a
   different study with a fresh holdout. The ≥500%-MFE base rate (3 tickers)
   still means no policy reliably pre-identifies the next Micron; the honest
   target is "don't structurally exclude the population", not "select for it".

## Sample-size / power notes

- All headline claims use cells n≥30; theme cells, chop-val (n=38),
  mega-cap, young-cohort-id4 (n=10–31), and mom_xs_q1/q2 cells are underpowered
  and used descriptively only.
- Phase 3/4 test reads: exactly one per pre-registered winner. This window has
  now been read 5×; treat any further mining of it as contaminated.
- All results are frictionless price-return simulations on 4H stock bars
  (no options premium path, no slippage/costs); relative comparisons are the
  deliverable, not absolute levels.

## Reproduce

```
PYTHONPATH=. .venv/bin/python scripts/capstone/build_meta_scored_from_oof.py
PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_segmentation.py
PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_entry_side.py
PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_learned_admission.py
```
