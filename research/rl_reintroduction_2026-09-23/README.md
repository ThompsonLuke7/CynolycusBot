# Reinforcement learning, reintroduced — all six stages, run

2026-09-23. Plan: `docs/superpowers/plans/2026-09-23-rl-reintroduction-staged-plan.md`.
Source of the idea: `source_chatgpt_thread.md` (the user's "Simplify RL Trading" thread).
Preregistration: `00_preregistration.md` — written before any stage ran, and not amended after.

**Nothing in this series changed live configuration. The test block was never read by any
stage.**

---

## The one-paragraph answer

RL is not the missing ingredient, and the reason is now measured rather than guessed. Across
six stages, **every learned layer failed against its own control** — selection (Stage 2),
exit (Stages 3 and 4, on two independent estimators and three different controls), entry
timing (Stage 5), and sizing (Stage 6). What the series found instead is that **the largest
loss in the system is a configuration choice, not a missing model**: the live exit rule returns
+0.091R per trade where simply holding 30 sessions returns +0.662R, and two separate RL
estimators independently converged on "hold to the horizon" as the optimal policy. The
supporting positives are all properties of things that already exist — the deployed momentum
score concentrates right-tail events 8.8x and its Kelly fraction rises 6x monotonically across
deciles; the ATR risk unit already absorbs all the predictable dispersion. And one of my own
plan's central claims turned out to be wrong: the options hurdle is ~6% of underlying move, not
~35%, because I had forgotten to divide by leverage.

---

## Stage-by-stage

| stage | question | verdict |
|---|---|---|
| **0** | substrate, split, tail event | Complete. Panel v2 rebuilt (1,566,875 rows). Test block shown **not fresh**; fresh holdout named. Tail event frozen; 2 of the thread's thresholds rejected as non-events. |
| **1** | does the funnel see the right tail? | **No, and loosening it doesn't help.** Gate lift 1.06x, best of 11 settings 1.14x, joint ladders *below* 1.0. MDE 0.016–0.052. |
| **2** | is ENTER/PASS decidable? | **By the incumbent, yes (+0.441R, 8.8x tail capture). By anything I fit, no** — 2 of 3 arms lose to their own permuted-label null. **C1 retracted.** |
| **3** | how much is a perfect exit worth? | **2.15R over the live rule, 0.57R over hold-30 — and none of it is learnable.** Imitation's +0.740R gain is fully reproduced by its permuted-label twin. |
| **4** | does fitted-Q beat that? | **No** (mom −0.107R, htf +0.067R vs best fixed, CIs straddle zero) — **but the state is load-bearing**: offset-only FQI is 1.13R worse. |
| **5** | is there an entry-timing edge? | **No at daily granularity.** Every fixed delay ≈ 0. "Wait for the pullback" is **−0.588R**. The learned policy beats its noise twin (+0.295R) by correctly learning "enter now". |
| **6** | can sizing be learned? | **No.** Linear objective reduces to Stage 2's null; dispersion after ATR-normalisation is unpredictable (R² −0.022 vs null +0.004). **Score-proportional sizing is supported**: Kelly f\* 0.02→0.12 across deciles. |

Findings docs: `01_…` `02_…` `03_…` `04_…` `05_…` `06_…_findings.md`.
Shared substrate: `panel.py` (+ 6 tests in `tests/test_panel.py`).

---

## What is actionable, in order of measured size

1. **Validate the underlying-referenced option stop — the biggest known leak, fix already
   shipped but unproven.** `underlying_stop_atr = 1.5` went in on 2026-08-18 and its own
   docstring says "NOT yet paper-validated". It replaced a −39% *premium* stop that
   `research/daily_live_reports/underlying_vs_premium_stop.md` measured firing at a **median
   −3.1% underlying move**, with 18 of 42 stops firing while the underlying was down <2%
   (−$43,944 realised; 13 of those 18 traded back above entry within 40 4H bars).
   **CORRECTED:** my earlier "exit geometry ~0.57R/trade" actionable modelled
   `momentum_config`'s RISK block, which does NOT bind the live path (`live/runner.py:516`
   "Replaces MomentumOptionPolicy execution"). The live rule is
   `core.live_4h_exec.ExitPolicy`: no trail, 53-bar horizon, +30%/16% trim, 1.5-ATR underlying
   stop. So that 0.57R is the value of a change the repo already made. See the correction block
   at the top of `03_exit_oracle_findings.md`. What survives is **1.58R of headroom over the
   best fixed policy**, and the sharper open question: is a 53-bar horizon long enough, and is
   1.5 ATR the right width?
2. **Gate coverage — a pipeline hole nobody had counted.** The candidate gate is evaluated on
   only **80.2%** of pool-ticker days, and the P1 tail rate is **1.101% on evaluated rows vs
   1.507% across all pool rows** — the tail is ~37% denser on the rows the feature pipeline has
   no entry for. Diagnosable with no modelling risk.
3. **Do the option book and the top-of-ranking book hold the same names?** Momentum's top-3
   are median **$0.85M** daily dollar volume, ATR 11.18% (3.1x universe), beta 2.05.
   **REFINED:** the gates that matter are the LIVE ones —
   `options_exec.ROUTE_MIN_OPEN_INTEREST = 500` and `ROUTE_MIN_VOLUME = 100`, enforced in
   `route_option_or_shares` — and a name that fails them is **routed to shares, not dropped**.
   (The identical thresholds in `momentum_option_policy.py` are on the dead path; citing those
   was the same mistake as #1.) So the hypothesis is not "we can't trade our best ideas" but
   the sharper one: **the option book is systematically populated by the more liquid, lower-
   ranked names while the illiquid top picks go to equity** — which would be the wrong way
   round, since within momentum equity returned +5.44% and options −31.50% on the same signals.
   Directly measurable from the route + `oi`/`volume` fields already in
   `Data/inference/*/live_signal_audit.jsonl`.
4. **Score-proportional sizing** — 6x Kelly spread across deciles, monotone, bottom decile
   unprofitable at any fraction.
5. **Rule out "wait for the pullback"** — measured at −0.588R on 434 candidates.
6. **Refresh the momentum walk-forward OOF build.** It ends 2026-05-14 while bars run to
   2026-09-22; refreshing yields **~4.3 months (~180 bars) of never-read data**, which is the
   only clean certification window available and needs no new vendor.

---

## Corrections this series made to its own inputs

Recorded because they are the kind of thing that otherwise propagates:

* **C1 retracted.** The plan claimed a 4–8pp edge "cannot pay a 35% round-trip hurdle". The
  hurdle is denominated in *premium*; an underlying edge must be multiplied by elasticity
  first. At the picks' 129% vol, elasticity is 3.89x and breakeven is **6.07% of underlying
  move**. The route is open, not closed. (`02_decidability_findings.md` §D.)
* **The momentum training matrix holds 1,081 tickers, not 3,089** — the 3,089 are per-ticker
  feature files. Stage 1 was restructured into two layers because of it.
* **The "coverage loss" was initially inflated** by a population/matrix window mismatch
  (2018-10 vs 2020-09). Bounding it moved the number 78.6% → 80.2%, so the hole is real.
* **Two of the thread's proposed tail thresholds are not tails** in this universe:
  `mfe_5 ≥ 10%` fires on 11.26% of rows and `mfe_20 ≥ 15%` on 22.98%.

## Where a control changed the answer

Three times, which is the point of having them:

| stage | uncontrolled headline | after its control |
|---|---|---|
| 2 | "LightGBM finds +0.275R of top-3 excess, p<0.0001" | that *was* the permuted-label null; real net −0.268R |
| 3 | "a supervised exit policy recovers 57% of a perfect exit OOS" | permuted-label twin recovered more; net −0.083R |
| 4 | "fitted-Q reaches 1.97R against a 2.08R fixed benchmark" | true, and the offset-only control shows the state is genuinely used (+1.13R) — the honest reading is "it rediscovered hold-30" |

---

## What would change the verdict

The pre-registered next variant, unchanged by any stage, is **features and resolution — not
optimisers**:

* **Forward realised volatility** as a feature. `forward_rv_predictability.py` measured forward
  20d RV at OOS R² **0.742** against 0.003 for forward return. It did not exist when this
  state was assembled, and Stage 6 §A shows why it might matter: the predictable part of
  dispersion is what the ATR unit already uses, so the question is whether a *forecast* beats
  a *trailing* estimate.
* **Decision-time IV**, shipped 2026-09-22 and accumulating. Prerequisite for anything that
  chooses an option structure rather than inheriting one.
* **Intraday resolution for entry timing.** Stage 5 is a daily test; the thread's design and the
  existing `Execution_Agent` operate on 1m bars. Prerequisite: the 1m cache spans
  2026-07-08..08-28 only — the exact gap that caused the stale-bar bug in
  `25_matched_control_and_depth_null.md` §A.2 — so it must be extended first, and the MDE
  computed before the run.
* **Portfolio-level sizing.** Stage 6 optimises independent trades; the live book is
  correlated. `research/portfolio_lab/` is where that belongs.
