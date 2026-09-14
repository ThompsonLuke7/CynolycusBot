# Horizon thesis, continuation, trust gate, theme look-ahead — five experiments

2026-09-12. Scripts: `scripts/horizon_thesis/{run_horizon_grid,build_decision_panel,
rank_falsification,continuation_model,trust_gate,theme_lookahead_audit,null_tilt_check}.py`
Raw results: `research/execution_quality/data/{horizon_grid_mom,rank_falsification,
continuation_model,trust_gate,theme_lookahead_audit}.json`

All five run on the existing leak-free spines: momentum's 4H training matrix (500-ticker
subsample, full history each) and the Meta research matrix's walk-forward OOF `mom_score`
(2022-11 .. 2026-05, 1,739 decision bars). GPU used for the XGBoost fits.

## Protocol fixes made before anything was measured

* **The bake-off harness's embargo was not what it claimed.** `run_bakeoff.py` uses
  `Timedelta(hours=4 * EMBARGO_BARS)`, but 4H bars arrive twice a day (10:00 and 14:00 ET),
  so its "60 bars" is 10 calendar days ≈ 14 bars. Labels near the train boundary could see
  into validation. The new harness counts the embargo in decision bars (70, against a
  longest look-forward of 60).
* Corporate-action guard (`core/corporate_actions.py` flags, non-organic only) applied to
  every forward window; ATR taken from the bar before the decision; forward windows start at
  t+1; CIs are moving-block bootstraps (block = the eval horizon) because neighbouring bars'
  windows overlap.
* Feature audit: `earnings_in_fwd_window` is built from the earnings calendar (dates are
  normally announced weeks ahead, so it is a mild caveat, not a leak); Meta's `news_p_*` are
  out-of-fold on the train slice.

---

## 1. Horizon thesis — train on one horizon, grade on 10/20/30 days

Six training horizons (2/5/10/15/20/30 trading days) x two label families (forward MFE in
ATR; "clean" = MFE − MAE), plus the deployed 25-bar composite (`M0_current`), two
pre-specified stacks, and five permuted-label nulls. Identical features, rows, split and
hyperparameters. Winner chosen on VALIDATION, reported on TEST (683 bars,
2024-12-26 .. 2026-05-14).

**Result A — for ranking future MFE, horizon matters and the deployed label is half as good.**

| eval target | best by val | test rho | matched-horizon label | `M0_current` | nulls |
|---|---|---|---|---|---|
| MFE 10d | `mfe_10d` | **0.146** | 0.146 | 0.075 | −0.016 .. +0.013 |
| MFE 20d | `STACK_mfe_all` | **0.137** | 0.132 | 0.076 | −0.017 .. +0.014 |
| MFE 30d | `STACK_mfe_all` | **0.123** | 0.125 | 0.069 | −0.016 .. +0.018 |

Every MFE-trained label roughly doubles the deployed composite's rank correlation, and the
stack is at or near the top at 20-30 days. So the Label Horizon Paradox shows up in the weak
form: short (10d) and stacked labels rank long-horizon excursions as well as or better than
the matched 30d label. **Stacking works** — which was your instinct.

**Result B — and it does not transfer to money. This is the important one.**

Scored against the *tradeable* target (share return, next open to the close h sessions later)
every MFE-trained label goes NEGATIVE, while the deployed composite and the clean family stay
positive:

| eval target | `M0_current` | best clean | all `mfe_*` | best by val → its test rho |
|---|---|---|---|---|
| return 10d | **+0.036** | +0.047 (`clean_30d`) | −0.004 .. −0.015 | `mfe_10d` → **−0.014** |
| return 20d | **+0.040** | +0.047 (`clean_30d`) | −0.008 .. −0.020 | `mfe_20d` → **−0.018** |
| return 30d | **+0.038** | +0.034 (`clean_30d`) | −0.014 .. −0.025 | `mfe_30d` → **−0.014** |

Paired block-bootstrap on top-3 excess, MFE-trained minus deployed: −1.90pp (p=0.114) at 10d,
**−6.32pp (p=0.008)** at 20d, **−10.88pp (p=0.012)** at 30d; the stack is worse still
(−3.23 / −9.08 / −13.06pp).

*Why*, measured rather than assumed (`null_tilt_check.py`, §5): MFE is expressed in ATR, so the
label rewards moves that are large **relative to the name's own volatility**, and the fitted model
duly picks names *below* universe volatility (top-3 carries 0.86x universe ATR, 0.93x realised
vol). Raw share return has no such denominator and in a rising tape it pays the opposite tilt.
The label is volatility-normalised; the money is not.

**Result C — validation selection failed on the money target.** Val rho on returns is tiny
(0.03-0.04) and picked the MFE labels that then lost. Only the MFE-target rows support
honest val→test selection. Treat any "best label" chosen on return-rho as unselected.

**Answers to the two questions asked:**
* *Should the momentum ranking label change?* **Not to an MFE target** — that is a measurable
  downgrade on realised return. The `clean` (MFE − MAE) family is the only candidate that beats
  the deployed label on test-return rho, but it barely trains (early stopping at 3-33 rounds vs
  80-130 for MFE) and its val rho is ~0, so it cannot be selected honestly on this evidence.
  This **agrees with the 2026-08-31 bake-off**, which also concluded only Meta's label should change.
* *Is 2-6 weeks the right horizon?* For the *label*, a 10-day MFE target ranks 20-30-day
  excursions as well as the matched-horizon label, so the 25-bar label is not mis-horizoned
  in the way `09_label_horizon.md` assumed. The horizon problem is on the exit side, not the label side.

---

## 2. Continuation — "keep it only while it's gaining momentum"

Top-3 (and top-10) momentum entries, checkpoint at the close of session 5, target = return
from there to session 30. Policies compared on the same test entries (n=615 top-3).

| policy | kept | mean/trade | median | test-window verdict |
|---|---|---|---|---|
| hold_30 (the thesis, unmanaged) | 100% | **+16.92%** | −0.49% | best on mean |
| exit_5 (≈ today's behaviour) | 0% | +2.66% | +0.63% | best on median |
| rule: keep if up at day 5 | 51% | +8.66% | −4.72% | worse than holding |
| rule: keep if beating SPY | 50% | +8.66% | −4.58% | worse than holding |
| ML continuation model | 63% | +9.36% | −1.40% | worse than holding |

**Every "cut the losers early" variant lost to simply holding.** In the test window the rules
kept the *worse* half (kept +9.58% vs dropped +19.58% remaining return, permutation p=0.010).

**Correction after a robustness check — do not over-read that.** On the FULL 2022-2026 sample
the same split is not significant (Mann-Whitney p=0.69; medians +2.61% kept vs +1.33% dropped),
and the day-5→remaining rank correlation among top-3 picks flips sign by year (2022 −0.31,
2023 +0.06, 2024 −0.13, 2025 −0.09, 2026 −0.03). Across the full cross-section it is ~0
(+0.003), so this is a property of the ranker's own picks, not a market-wide reversal.
**Honest verdict: a day-5 progress check adds nothing. There is no evidence it helps, and the
strong "it actively hurts" version rests on one 8-month window.**

**On the thesis itself:** holding to 30 sessions beats exiting at 5 on the mean (+16.9% vs
+2.7%) and on a 5%-trimmed mean (+7.3% vs +1.9%), but *not* on the median (−0.5% vs +0.6%).
The top 5% of trades supply 74% of total return. That is exactly the right-tail shape you have
been describing — the edge is real but it lives entirely in the tail, so position sizing and
survivorship matter more than the average trade. (Levels are inflated by the survivor-shaped
bar cache; the policy *comparisons* share entries and are unaffected.)

---

## 3. Trust gate — a second model deciding whether to trade at all

Target: will today's top-3 beat its own scored universe over the next 10 (and 20) sessions?
Features: the PIT regime panel, cross-sectional shape (universe size, score dispersion, top-3
margin, crowding), and the ranker's own trailing record lagged by the full hold. Thresholds
fixed on train.

| hold | gate AUROC | always-on | on/off gate | three-level hybrid | gate − always-on |
|---|---|---|---|---|---|
| 10d | **0.429** (worse than chance) | +4.32% | +3.05% (63% exposure) | +1.76% (46%) | −1.27pp, p=0.662 |
| 20d | 0.571 | +9.38% | +4.11% (39%) | +4.16% (35%) | −5.28pp, p=0.324 |

**Null, and do not wire it.** The gate cannot forecast its own good days; gating and the
hybrid both reduce return without improving return-per-unit-risk (0.28 → 0.25 → 0.20 at 10d).
This reproduces the earlier retracted breadth-gate finding on a different target and a longer
sample. The hybrid you suggested is included and does not rescue it.

---

## 4. Theme look-ahead audit

**Churn:** theme membership is far less stable than the backfill assumes. Between the real
weekly PIT snapshots, 83-88% of tickers change primary theme in 6-9 days. That is not just
Claude renaming clusters — theme *names* overlap 85-90% across snapshots, but a ticker's
co-member set barely persists (Jaccard median **0.124 / 0.140**). The backfill labels
2022-11..2026-06 with ONE registry, and 93.9% of its last-date assignments already differ
from the 2026-09-08 PIT snapshot.

**Leak signature: not found.** If 2026 membership were leaking backwards, theme IC should be
strongest in the earliest years and decay toward the present. It does the opposite (mean |IC|
0.045 in 2022 → 0.065 in 2026), and tracks the genuinely walk-forward control (`mom_score`
0.056 → 0.076). So my look-ahead hypothesis is **not supported** by this test.

**But the theme block hurts anyway.** Same split, same target, features in vs out:
with-theme test rho **+0.0029**, no-theme **+0.0197**; the block contributes
**−0.0167 [−0.0222, −0.0111]**, distinguishable from noise. Three theme columns
(`theme_age_days`, `theme_newness_score`, `theme_days_since_refresh`) are 100% NaN.

**Verdict:** drop the theme block from the Meta retrain, or stabilise membership first. The
problem is instability/noise, not the leakage I suspected.

---

## 5. Falsification + factor audit — is the momentum ordering real?

`mom_score`'s top-3 excess over its own scored universe, 1,739 decision bars, 2022-11..2026-05,
with the score shuffled within bar as the null and cross-sectional factor residuals as the control:

| hold | top-3 excess | on factor residuals | NULL (shuffled score) | best one-line sort |
|---|---|---|---|---|
| 5d | +3.31% [2.05, 4.60] | +2.03% [1.21, 3.01] | **+0.01%** [−0.08, +0.12] | +1.73% (20d momentum) |
| 10d | +5.93% [3.80, 8.25] | +3.29% [1.77, 5.13] | **+0.00%** | +2.98% (60d momentum) |
| 15d | +9.27% [5.57, 13.19] | +5.24% [2.67, 8.36] | **+0.05%** | +4.51% |
| 20d | +11.99% [7.22, 17.51] | +6.68% [3.05, 11.10] | **+0.13%** | +4.89% |
| 30d | +18.81% [11.29, 26.23] | +10.14% [4.67, 16.30] | **+0.13%** | +7.48% |

* **The harness does not manufacture edge.** The shuffled-score null is ~0 at every hold, so the
  measurement itself is clean.
* **The ordering survives the factor test.** Removing past 20/60/120d return, realised vol, log
  dollar volume, 60d beta and distance-to-52w-high cross-sectionally leaves 54-61% of the excess,
  significant at every hold. It is **not** just a repackaged momentum factor.
* **It beats the cheap alternative** by roughly 2.2-2.5x at every hold (30d: +18.8% vs +7.5% for a
  60-day momentum sort), so the ML is earning its complexity on this measure.
* **Excess grows monotonically with hold** (+3.3% at 5 sessions → +18.8% at 30), which supports the
  2-6 week thesis on the *selection* side.

### The metric caveat this experiment exposed (`null_tilt_check.py`)

In the §1 grid, the permuted-label models scored ~0 rank correlation yet showed a **positive**
top-3 excess on raw returns (+2.5 to +8pp, some with p<0.05). Measured cause: a model fitted to
noise still produces a systematic cross-sectional tilt. Its top-3 carries **1.46x** universe ATR,
**1.62x** realised vol and **1.36x** beta, and that tilt alone earned **+4.79pp** on 20-day returns
in the test window — *more* than the genuinely trained `mfe_10d` model's +0.94pp.

Two consequences, and they are general:
1. **Never read a model-based top-3 excess on raw returns without a trained-on-noise control.** The
   shuffled-*score* null used above is ~0, but the shuffled-*label* null is not; only the second one
   catches this, and it is the relevant control whenever a model, not a raw signal, does the ranking.
2. Rank correlation was the trustworthy metric in the grid (null ≈ 0.00-0.02 against 0.12-0.15 real).

## 6. The 2026-07-20 exit switch, re-checked on the now-uncensored window

Context: on 2026-07-20 the shared `ExecPolicy` default (Momentum/HTF/Meta) moved to the "id4
tail-rider" shape — take-profit .20→.30, scale .50→.16, horizon 25→53 bars, stop .50→.39,
trail .35→off. The July confirmation attempt (`exit_policy_fresh_window_check.py`) was
**uninformative**, not merely weak: only 33-178 bars existed past the OOF cutoff against 25-60
bar horizons, so 35-97% of trades were right-censored.

**What changed:** the untouched window is now 2026-05-15 → 2026-09-10 (~326 bars), so censoring
is largely gone. Re-ran it, plus a new paired version (`scripts/horizon_thesis/exit_paired_check.py`)
that holds the ENTRIES fixed and varies only the exit rule — the source script compares trade sets
of different sizes (328 vs 251), which confounds the exit with re-entry timing.

**Unchanged caveat: this is still scored with the DEPLOYED boosters**, which have very likely seen
this window in training. It is a direction check. A clean test needs a walk-forward OOF extension,
i.e. a GPU retrain.

Paired, same entries, same bars:

| module | pair | per trade | per bar | holds (bars) |
|---|---|---|---|---|
| momentum (n=226) | id4 − deployed | −2.26pp [−7.94, +2.36] p=0.415 | **−0.426pp** [−1.077, +0.016] p=0.071 | 22.3 → 44.8 |
| momentum (n=255) | g284 − deployed | −4.44pp p=0.055 | **+1.152pp** [+0.495, +1.690] **p=0.001** | 21.5 → 19.4 |
| meta (n=391) | id4 − deployed | −2.60pp [−6.34, +0.49] p=0.112 | −0.196pp p=0.257 | 21.3 → 41.0 |
| meta (n=419) | g284 − deployed | −2.93pp p=0.083 | **+1.207pp** [+0.766, +1.637] **p=0.000** | 21.0 → 19.6 |

* **The switch is not confirmed and is mildly contradicted.** id4 earns less per trade than the
  config it replaced (not significant) while holding ~2x as long, so per unit of capital-time it is
  behind — marginally in momentum (p=0.071), not significantly in Meta.
* **The harvester shape (g284: +7% target, full exit, stop .59) wins decisively on return per bar**
  in both modules, at the cost of per-trade mean. This is the same trade-off the 2026-07-19 study
  recorded ("id4 wins mean 93/97, g284 wins rpb+win 97/97"); the fresh window favours the rpb side.
* Recommendation: **do not flip the default on this evidence alone** (leakage caveat, one 4-month
  window). It is now a concrete reason to spend the GPU retrain on extending the OOF, which is the
  only thing that settles it.

**Two data defects found while doing this:**
1. `strategies/multi_ticker_swing_htf/data/processed/features_4h.parquet` ends **2026-06-02** and
   was last written 2026-06-14 — three months stale, against momentum's 2026-09-10. HTF's row in
   the fresh-window check covers 33 bars and is uninformative for that reason, not censoring.
2. `strategies/momentum_expansion/config/momentum_config.py` lines 329-334 define
   `atr_trail_distance`, `score_decay_exit` and `trend_break_atr` **twice** in the same dict
   literal. The values are identical so behaviour is unaffected, but one edit to the first copy
   would be silently overridden by the second.

## 7. What are themes actually good for? (cohesion)

§4 answered "do theme FEATURES help a cross-sectional ranker" — they hurt it. That is not the
original hypothesis, which was: **a theme is a group that moves together, with a leader and
sympathy followers.** That is a claim about co-movement and lead-lag, and nothing before this
tested it.

Forward 20-session mean pairwise correlation, measured on the sessions strictly AFTER each
monthly snapshot (44 snapshots, ~3,130 theme-months). Forward, because the theme embedding
already blends text with a 60-day TRAILING co-movement vector — trailing cohesion is guaranteed
by construction and proves nothing.

| grouping | forward pairwise correlation | vs theme |
|---|---|---|
| **theme groups** | **0.388** | — |
| trailing-correlation clusters (same count, same sizes, no text) | 0.343 | −0.046 |
| sector, size-matched draw | 0.233 | −0.155 |
| sector, whole sector | 0.233 | −0.155 |
| size-matched random | 0.217 | −0.171 |

* **Themes are a real co-moving grouping.** They are far tighter than sector membership
  (+0.155) — so this is not "tech moves with tech".
* **But most of it is co-movement, not the taxonomy.** Clustering the same tickers on their
  trailing 60-day correlations — no news, no embeddings, no Claude — reaches 0.343, i.e. about
  **73% of the theme grouping's edge over random**. The news/LLM taxonomy adds an incremental
  **+0.046** (the remaining ~27%).
* Read honestly: the expensive half of the theme pipeline (collection, embedding, Claude
  labelling) is buying the last quarter of the grouping quality. A nightly correlation clustering
  would buy the first three quarters for almost nothing — and would not churn 83-88% a week,
  because it has no LLM naming step (§4).

### Leader -> follower, and the 50/50 split

Event: a theme member closes **+7% or more** in a day. Followers = its theme-mates. Everything is
entered at the OPEN of the next session, so nothing here counts the same-day sympathy move.
24,865 leader-events over 896 days. Excess is against that day's whole scored universe; CIs are
clustered on the event day.

| hold | theme-mates | sector-mates (size-matched, same leader) | shuffled themes | theme − sector (paired) |
|---|---|---|---|---|
| 1d | +0.066% | +0.006% | +0.003% | +0.122pp [−0.002, +0.291] |
| 3d | +0.176% | +0.012% | +0.027% | **+0.208pp** [+0.007, +0.496] |
| 5d | +0.301% | +0.018% | +0.035% | **+0.183pp** [+0.002, +0.474] |
| 10d | +0.786% | +0.101% | +0.039% | **+0.408pp** [+0.070, +1.034] |
| 20d | **+1.490%** | +0.223% | +0.042% | **+0.671pp** [+0.023, +1.636] |

**The sympathy-follower effect is real and it is a theme effect, not a sector effect.** Shuffled
theme labels give ~0 (so it is not group size or the event day), and same-sector mates of the same
leader capture only +0.22% of the +1.49% at 20 sessions. The paired theme−sector difference clears
zero from 3 sessions out.

The allocation actually proposed, on the same events:

| hold | 50/50 leader+followers | leader only | followers only | universe |
|---|---|---|---|---|
| 5d | +1.086% | **+1.374%** | +0.799% | +0.498% |
| 10d | +2.956% | **+3.678%** | +2.234% | +1.449% |
| 20d | +5.763% [+4.661, +6.870] | **+7.203%** | +4.323% | +2.833% |

* **The 50/50 split works** — it beats the universe by ~2.9pp at 20 sessions.
* **But leader-weighting beat it at every horizon.** The follower sleeve earns its place
  (+4.32% vs +2.83% universe) yet dilutes the leader (+7.20%). On these numbers the split should
  tilt toward the leader, not sit at 50/50 — with the caveat that leader-only is a single name per
  event, so its concentration risk is not priced in an average-of-events table.

**The caveat that decides whether this is tradeable.** All of it is measured on the BACKFILLED
membership — one registry applied to all history, therefore unnaturally stable. The live PIT
snapshots churn 83-88% a week (§4). So this is evidence that *a stable theme grouping* has a
tradeable leader/follower structure; it is NOT yet evidence that today's live theme assignment
does. Also: overlapping 20-session windows across 896 days mean the CIs are optimistic (the
day-clustering handles within-day, not across-day overlap), and these are universe-relative
excesses with no costs or slippage.

**So themes are good for something — just not what they are currently wired into.** They are a
grouping/propagation signal, not a cross-sectional ranking feature (§4: −0.0167 rho when fed to
Meta). The two uses are independent and the evidence points opposite ways.

## Limits that apply to all of it

* One test window per experiment (grid test is 2024-12 .. 2026-05; continuation/gate test is
  ~8 months). Nothing here is walk-forward across many regimes.
* The bar caches are survivor-shaped and unadjusted; the corporate-action guard removes
  0.006-0.010% of rows, and extreme tails survive (max 30-day return in the panel is +2,641%).
  Absolute levels are inflated; within-bar comparisons are not.
* 500-ticker subsample for the grid (the box has a 19GB ceiling and a prior OOM).
* The grid's `earnings_in_fwd_window` caveat above applies equally to every candidate.
