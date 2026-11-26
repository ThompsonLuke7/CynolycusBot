# Regime experiments: where regime belongs, how fine-grained it must be, and what it is missing

**Pre-registered 2026-11-24 15:19 ET**, before any result below was computed. E0 has run; everything
else is designed, not run. Each experiment lists its own question, data, method, control,
outputs, decision rule and where it runs.

## 0. The architecture these experiments test

- **One shared market state, published by the nervous system.** It has two clocks:
  - the **daily regime**: today, the 7-signal vote (`signals/market_regime/regime_vote.py`), known
    at the prior close;
  - an **intraday session state**: trend vs range day, volatility expansion and event shocks,
    refreshed during the session. E5 decides how often.
- **Each module owns a regime profile:** how *its* order policy responds to the shared state, given
  its holding period and goal. The state is shared; the response is not. Illustrative only — E7
  decides the actual mappings:

  | Module | Style | What the regime may change |
  |---|---|---|
  | Intraday structure | 0DTE/1DTE, minutes–hours | side bias, hold/exit rule, size (0DTE always allowed) |
  | SPY daytrader | 10-min bars, intraday | long/short probability thresholds (already built, off) |
  | Momentum (4H) | multi-day breakouts | entry threshold, size, max hold |
  | HTF swing | multi-day swings | entry threshold, size |
  | Meta | confluence, multi-day | size (multipliers live since 2026-11-23), threshold |
  | Dealer ranker | positioning levels | side bias, size |

- **Models keep one simple target each.** The regime lives in the policy unless E6 shows a
  model's own target is predicted better with it.

## 1. Regime factor inventory: what we have, what is missing, how far back

| Family | What it captures | In the live vote? | Daily history available |
|---|---|---|---|
| F1 Trend | price direction and persistence | yes (QQQ fast, QQQ<SMA50, SPY EMA20/50) | S&P 1928, Nasdaq-100 1985 |
| F2 Volatility | realised and implied vol, VIX term structure | realised only (spy_rv20_z) | realised 1928; VIX 1990; VIX3M 2006 |
| F3 Credit / liquidity | spreads, funding stress | yes (HYG/IEI) | HYG 2007; **Baa spread 1986 and HY OAS 1996 need a free FRED key** |
| F4 Rates / policy | 10y level and change, curve, Fed path | **no** | yields 1962, 3m bill 1960; fed funds via FRED |
| F5 Dollar / commodities | USD stress, oil shocks | **no** | dollar 1971; oil 2000 (1986 via FRED) |
| F6 Breadth / participation / correlation | narrow vs broad market, everything-moves-together | yes (11-sector breadth) | sector SPDRs 1998; RSP 2003 |
| F7 Sentiment / positioning | news tone, put/call, dealer gamma, social | **no** | news 2026-05; dealer 2026-07 (85 days); short history only |
| F8 Event calendar | FOMC, CPI, NFP, earnings season, OpEx | **no** | in repo FOMC 2022+ (earlier dates are public) |
| F9 Reaction to news | "good news sold" (beats that fall) | **no** | earnings dates in repo; short history |

The user's AI example (strong earnings, falling prices, fearful narrative) is F9, and it is
measured from prices: how the market *reacts* to good news. It is the leading candidate for a
"why" factor that separates similar-looking states.

## 2. Experiments

### E0 Data foundation (done)
- **Script:** `e0_fetch_long_history.py`, local, about 1 minute.
- **What it fetches:** 24 Yahoo series (indexes, VIX, yields, dollar, oil, HYG/IEF, sector SPDRs,
  RSP). Six FRED series are added when `FRED_API_KEY` is set. The key is free, and FRED's keyless
  CSV endpoint is refused from this network.
- **Rules:** the session in progress is dropped; raw downloads are immutable (one dated folder per
  run); FRED releases carry an explicit availability lag.
- **Fitness checks (run 2026-11-24 15:17 ET):** all passed.
  - Crash-day returns match: 1987-10-19 −20.5%, 2008-10-15 −9.0%, 2020-03-16 −12.0%.
  - Nasdaq-100 vs Alpaca QQQ daily returns correlate 0.9987 (2,736 days).
  - Yield and VIX ranges are sane.
- **Output:** `Data/research/regime_40y/panel_20261124T1517.parquet` (1927-12-30..2026-11-23).

### E1 Does a regime state describe different futures? (40 years, local)
- **Question:** do the vote's states have different forward *distributions*, mainly drawdown and
  volatility, consistently across decades?
- **Data:** Nasdaq-100 1986–2026 (primary, the AI/tech tape) and S&P 500 1962–2026 (secondary).
- **States:** a long-history vote from the F1/F2/F4 inputs that exist for the whole span.
  - Votes: fast trend, below SMA50, EMA20<EMA50, RV20 z>1, VIX z>1 (1990+), curve 10y–3m inverted.
  - With FRED: Baa spread z>0 (1986+).
  - Its agreement with the live 7-vote on 2021+ is reported. Thresholds are the round values from
    the live vote and are not tuned.
- **Targets** (all from the close of t):
  - worst drawdown over the next 20 sessions (primary);
  - realised vol over the next 20 sessions;
  - 20-session return (reported; direction is the hardest target).
- **Method:** state-conditional distributions with a circular 20-session block bootstrap, run
  separately for each decade.
- **Control:** a circular-shift placebo of the state series (1,000 draws).
- **Decision:** a state definition is *useful* if forward drawdown or vol differ between states in
  at least 4 of 5 decades beyond the placebo's 95th percentile. Direction is not required.

### E2 A general regime policy on 40 years (local)
- **Question:** which index-level policy survives every crash in the sample? The crashes are 1987,
  1990, 2000–02, 2007–09, 2011, 2015–16, 2018, 2020, 2022, 2025 and 2026.
- **Books** on the index, deciding at the close and trading the next open, 2 bp per unit turnover:
  - buy-and-hold;
  - long/flat when the vote is 4 or more;
  - size by regime with the live multipliers (1.0 / 0.8 / 0.5 / 0.25);
  - 15% volatility target;
  - the classic SMA200 rule.
- **Reported:** CAGR, Sharpe, MDD, time invested, a per-crash table and per-decade results.
- **Control:** a circular-shift placebo of each state series.
- **Decision:** adopt as the default general policy if MDD improves in at least 3 of every 4
  crashes, Sharpe is not below buy-and-hold beyond its confidence interval, and the placebo
  percentile on MDD is above 90%.

### E3 Are we missing a factor? Factor completeness (40 years where coverage allows, local)
- **Question:** does any factor family add out-of-sample information about the primary target (a
  20-session drawdown of 5% or worse) beyond price and volatility alone? Is any family the one we
  are missing?
- **Method:**
  - Walk forward on an expanding window with a yearly refit, Nasdaq-100 1990–2026 (training from
    1986).
  - Model: logistic regression with the scaler fit on training data only; a shallow LightGBM as a
    secondary check.
  - Arms: a baseline of F1+F2 realised; the baseline plus each family alone; all families; and
    leave-one-family-out.
  - Families with short history are run on their own span only, and the span is stated.
- **Control:** for each family, its features block-permuted in 60-session blocks (100 draws). This
  gives the null for Δ log-loss and doubles as the MDE.
- **Episode audit:** for every drawdown of 10% or more, which families moved *before* the price vote
  flagged, and where the baseline's error was largest.
- **Decision:** a family is *crucial and missing* if it improves out-of-sample log-loss beyond the
  permuted control's 95th percentile in at least 2 of the last 3 decades. A family that helps only
  in specific episodes is labelled *context* (it explains, but does not forecast).

### E4 Context factors with short history (2024–2026, local, descriptive plus forward collection)
- **"Good news sold" (F9):**
  - Measure: each day, the mean 1- and 2-day return relative to SPY of tickers with a positive
    earnings surprise or a positive catalyst (FinBERT).
  - A negative value means buyers are exhausted.
  - Test the user's case: did AI-name beats get sold before the index vote flagged on 10-16? Repeat
    for 2024-07 and 2025-02 if the data reach.
- **Other short-history families:** news-tone balance, dealer gamma sign on SPY/QQQ (85 snapshots),
  and CBOE put/call.
- **Decision:** with this few episodes nothing is certified. A factor that led the vote in at least
  2 of 3 episodes enters the forward shadow (E8) as a logged candidate.

### E5 How fine-grained must the intraday regime be? (local)
- **Why intraday at all:**
  - scheduled events (FOMC 14:00 and the 14:30 press conference; CPI and NFP at 08:30, so they are
    known at the open);
  - unscheduled news (tariffs, geopolitics);
  - trend-day vs range-day identity, which is often visible by 10:30;
  - volatility expansion and breadth thrusts.
  
  None of these needs a minute-by-minute *macro* regime. They need a session state that can flip on
  evidence.
- **Data:** SPY/QQQ 1-minute SIP bars 2016–2026 (raw); VIXY 1-minute for intraday vol; FOMC dates
  2016–2021 added from the Fed's published schedule.
- **Targets** from checkpoint t (10:00, 10:30, 11:00, 12:00, 14:30): rest-of-day realised range and
  rest-of-day return sign.
- **Arms:**
  - (a) the prior-close daily regime only;
  - (b) plus a session state at t: return since open, side of VWAP, opening-range break,
    range-so-far / ATR, VIXY change;
  - (c) the session state refreshed every 30 minutes;
  - (d) every 5 minutes;
  - (e) event-triggered refresh (FOMC/CPI days).
- **Decision:** use the coarsest refresh that captures at least 80% of the information gain of
  (d). Add event-triggered refreshes only if FOMC/CPI-day flips predict the rest of the day beyond
  what the schedule alone gives.

### E6 Model vs policy: does the model itself need the regime? (Colab for training)
- **For each ML module** (Momentum, SPY daytrader, Meta), with the label fixed and the existing
  walk-forward folds, **validation folds only** (the Stage 2/3 test set stays sealed until about
  2027-06):
  - (A) base features only;
  - (B) base plus the vote inputs;
  - (C) separate models per regime state.
  
  Momentum, Meta and the 30m swing already carry the regime-table columns, so their arm (A) is the
  ablation that removes them.
- **Scoring:** within each regime (IC or AUC, plus calibration), against a block-shuffled regime
  control.
- **Decision:** regime stays out of the model unless (B) or (C) beats (A) within regimes by more
  than the shuffled control in the validation folds. Expected answer: (C) is starved, with about
  five bear spells in training.

### E7 Policy replay: does a regime-aware order policy beat the current one? (local)
- **The same signals, run through two policies:** the current one and a regime-aware one. The
  rules are fixed before replay:
  - **Intraday structure:** extend `research/regime_switch_2026-11-20/08` to the full regime state.
    In RISK_OFF or CRISIS, shorts hold to their structural target or EOD and longs exit on the
    current fast rule. Otherwise the current policy applies.
  - **SPY daytrader:** its built-in regime thresholds on its out-of-fold signal frame
    (`phase4_signal_frame.parquet`), on vs off.
  - **Momentum / HTF / Meta:** threshold and size by regime on validation backtests
    (`family_backtest.py`) and the decision-state archive.
- **Control:** the same rules on block-shuffled regime days. The regime must beat a random
  assignment of the same rules.
- **Decision:** a module adopts its regime profile in paper only if it beats both the current
  policy and the shuffled control on validation. That leads to E8, not straight to live.

### E8 Forward shadow (live paper, ongoing)
- **What it logs:** every live decision is recorded under the current policy and every candidate
  regime profile (a counterfactual ledger), plus the E4 candidate factors.
- **Read date:** set when E7 finishes. The reading is the later of 40 sessions or 10 sessions in
  RISK_OFF/CRISIS, with the MDE stated at the read.

## 3. Order, compute and what is needed from the user

| Step | Runs where | Rough time | Needs |
|---|---|---|---|
| E0 | local | 1 min | done; **a free FRED API key** adds credit before 2007 and macro |
| E1, E2 | local | minutes | E0 |
| E3 | local | ~30 min | E0 (+ FRED) |
| E5 | local | ~1 h (1-minute fetch + replay) | Alpaca SIP |
| E7 | local | ~1 h | E5 for intraday; existing OOF frames |
| E6 | Colab | per module | E7 result first, so training is spent only where policy alone falls short |
| E4, E8 | local, ongoing | – | decision archive, shadow ledger |

Each script is written as it is run, with its pre-registered spec in the docstring, and results go
in this folder's README.

## 4. Addendum, pre-registered 2026-11-25 ~00:47 ET (after E1–E5, before any result below)

Times in this section are approximate (between `date` readings of 00:45 and 01:02 ET). The order
is exact: each block was written before its script first ran.

The user asked whether E3's null means VIX, credit, the curve and term structure "don't matter",
whether E3 was run correctly, and whether a trained model should determine the regime. Two
experiments answer that. Neither changes E1–E5's recorded verdicts.

### E3b Was E3's null an artifact of how it was tested? (local)
E3 asked one narrow question: does a family add information beyond price and volatility, about one
target (Nasdaq-100 20-session drawdown ≤ −5%), with one linear model. Each variant below relaxes
one of those choices:
- **Standalone vs incremental.** Each family *alone* vs climatology (the training base rate), next
  to base+family vs base. A family can be informative on its own and still add nothing beyond price
  and vol, because price and vol already carry the same information. **Overlap:** the out-of-sample
  correlation between the family-only and base-only predicted probabilities.
- **Horizons and tails:**
  - T1: Nasdaq-100, 20-session drawdown ≤ −5% (E3's target);
  - T2: Nasdaq-100, 60 sessions ≤ −10%;
  - T3: Nasdaq-100, 120 sessions ≤ −15%;
  - T4: Nasdaq-100, 20 sessions ≤ −10% (crash tail);
  - T5: S&P 500, 20 sessions ≤ −5% (base features on the S&P itself);
  - T6: S&P 500, 60 sessions ≤ −10%.
  - The purge equals the horizon, and permutation blocks are max(60, horizon).
- **Slow features,** the ones the literature uses at long horizons, as new families. E3's families
  are left unchanged.
  - `rates_slow`: share of the last 252 sessions with the curve inverted, and the 250-session curve
    change.
  - `credit_slow`: 120-session change in the Baa spread.
- **Models:**
  - logistic regression with features clipped to the training window's 1st/99th percentiles
    (primary, since raw levels were shown to break it), with 100 permutation draws;
  - LightGBM (E3's fixed parameters), with its own permutation null of 20 draws. E3 had no
    LightGBM null.
- **Decision, per target and family:** "adds beyond price and vol" if the clipped logistic beats its
  permuted 95th percentile in ≥ 2 of the last 3 decades. LightGBM is reported beside it. "Matters
  on its own" if the family-alone model beats its own permuted 95th percentile over all years.
  Power: a 120-session target has about 20 independent outcomes per decade, so its MDE is stated
  and a null there means "not measurable".

### E9 A trained regime model vs the hand vote (local)
- **Models,** all walk-forward with a yearly refit on an expanding window:
  - **HMM-R:** a Gaussian hidden Markov model on Nasdaq-100 daily log returns (switching mean and
    variance), K = 2 and K = 3, 1986+.
  - **HMM-M:** a diagonal Gaussian HMM, K = 3, on daily Nasdaq-100 return and the 5-session changes
    in log VIX, the Baa spread and the 10y−3m curve, 1991+.
  - The states use FILTERED probabilities only (forward algorithm, data ≤ t). Never smoothed
    probabilities or Viterbi over the sample, which would leak the future. States are ordered by
    their training return variance. EM uses 5 random starts at the first fit, then warm starts.
  - **SUP:** a supervised regime score. LightGBM and clipped logistic regression on base plus every
    long-span family plus sector correlation, predicting E3's T1 and T2. The score is the predicted
    probability.
- **States and books** (the same execution as E2: decide at the close of t, trade at the close of
  t+1, 2 bp, bills on cash):
  - HMM: weight = 1 − 0.75·P(highest-variance state) (range 0.25–1, as the live multipliers);
    long/flat = flat when that P > 0.5.
  - SUP: the score's percentile within the training window maps to 1.0 (< 80th), 0.8 (80–90th),
    0.5 (90–97th) and 0.25 (≥ 97th). These are about the vote's state frequencies. Long/flat = flat
    above the 80th percentile.
  - Benchmarks on the same out-of-sample span: buy-and-hold, the long-history vote with the live
    multipliers, vote long/flat, and the 15% vol target.
- **Description:** E1's eta² by decade for each model's 3-bucket state, with E1's placebo.
- **Decision:** a trained model is PREFERRED over the hand vote if its sized book (i) has a Sharpe
  above the vote book's with the 95% CI of the difference above zero, or (ii) beats the vote book's
  crash MDD in ≥ 3 of 4 out-of-sample crashes with a Sharpe-difference CI not below zero.
  Separately, it must beat the 15% vol target on the same terms to be worth its complexity. A tie
  keeps the vote, because it is simpler and auditable.
- **E9 specification details,** fixed 2026-11-25 ~00:52 ET before any E9 result:
  - **Primary variants:** HMM-R K = 2 (the standard Markov-switching model on returns) and SUP
    LightGBM on T1 without sector correlation (supervised, out-of-sample span from 1995). The
    others are secondary: HMM-R K = 3, HMM-M K = 3, SUP logistic on T1, SUP LightGBM and logistic on
    T2, and SUP with sector correlation (span from 2004). With about 8 variants, one secondary
    passing at 95% by chance is likely (about 1 − 0.95⁸ ≈ 34%), so a secondary pass alone does
    not displace the vote.
  - **SUP percentile cutoffs** come from the model's in-sample scores on its own training window.

### E7-M Momentum policy replay on Stage 3 validation, pre-registered 2026-11-25 ~00:56 ET (before running)
- **Data:** Stage 3 out-of-fold scores of the momentum slot model L2_20__pit
  (`stage3_results/oof_L2_20__pit.parquet`), aligned to the Stage 3 matrix by
  `eval_stage2.load_panel`.
  - Validation bars only, via `eval_stage2.partitions`. Bars on or after 2026-05-15 are dropped
    before any statistic, so the sealed test is never read.
- **Book:** at every 4H bar, go long the top 10 by score. Each trade earns its 20-session `fret_20`
  minus the matrix cost.
  - These are absolute returns, not SPY-relative, because the live module is long-only and
    unhedged.
  - The regime is the long-history Nasdaq vote from the prior session's close.
  - The live 7-vote (2021-10+) is a secondary check.
- **Policies:**
  - P0 current: constant size.
  - P1: the live multipliers by state (1 / 1 / 0.8 / 0.5 / 0.25).
  - P2: no entries when the vote is ≥ 4.
- **Primary metric:** return per unit of exposure, R = Σ w·r / Σ w. A regime profile helps only if
  it cuts the bars that later lose, which raises R above P0's.
- **Control:** circular shifts of the regime series (≥ 60 sessions, 1,000 draws). This preserves
  how often each state occurs and how long it lasts. The bootstrap CI uses Stage 3's blocks,
  ceil(20/5) = 4-week blocks of bars.
- **Decision:** a profile goes to the E8 paper shadow only if R(P) − R(P0) > 0 and that
  improvement is at or above the shuffled control's 95th percentile. It never goes straight to live.
- **Out of scope tonight, and why:**
  - HTF has no validated model (Stage 3 made no pick).
  - Meta and the 30m swing have no out-of-fold scores here.
  - The SPY daytrader needs its own execution engine.
  - The intraday structure hold rule was designed on the only ledger that exists (08/09), so a
    replay there would be in-sample. Its out-of-sample test is the forward data from 11-23 on.

### E4 specifics, pre-registered 2026-11-25 ~00:59 ET (before running)
The earnings history turned out longer than planned. Surprises go back to 1998 for 2,646 tickers;
daily SIP returns start in 2016. So E4 runs on 2016-02..2026-11 instead of 2024–2026.
- **Reports:** rows with a finite `surprise_pct`, dated before the last complete session.
  - Beat: surprise ≥ +5%. Miss: surprise ≤ −5%.
  - **Reaction:** close(d+1)/close(d−1) − 1 minus SPY's return over the same window. This covers
    both before-open and after-close reports. It is known at the close of d+1.
  - **Filters:** prior close ≥ $5. Reactions beyond ±40% are dropped (corporate actions or data
    errors) and counted.
  - **Returns:** split-adjusted SIP research bars through 2026-09-24, then live-cache daily
    returns. Returns are spliced, not price levels.
  - The universe is today's calendar tickers, so it is survivor-shaped. Stated, not fixed.
- **Daily factors at the close of t,** from reports known by t in the last 60 sessions:
  - GNS (good news sold): the mean beat reaction, needing ≥ 30 beats;
  - BNB: the mean miss reaction, needing ≥ 15 misses;
  - GNS_tech: GNS for tickers assigned to XLK, needing ≥ 10 beats (secondary).
- **(a) Episode lead (the plan's decision):**
  - For each Nasdaq-100 drawdown ≥ 10% since 2016-06, the price votes flag on the first day 3 of
    4 are bearish (as in E3).
  - GNS *leads* if it is < 0 on ≥ 10 of the 20 sessions before that flag. The base rate is the
    share of all days with GNS < 0.
  - The plan's three named episodes are 2024-07, 2025-02 and 2026-10. GNS goes to the E8 shadow
    if it leads in ≥ 2 of them. All other episodes are reported too.
- **(b) Predictive (secondary):**
  - GNS and BNB added to E3's base model for T1 (Nasdaq 20d ≤ −5%), clipped logistic, walk-forward.
  - The first test year is 2021 (4 training years), so only the 2020s can be scored. 100
    block-permutation draws give the null and MDE.
  - With one decade at most this is reported, not certified.
- **E4 follow-up, 2026-11-25 ~01:01 ET, after the first run (labelled):**
  - The pre-registered sign rule was degenerate. GNS was < 0 on 0.1% of days (beats beat SPY by
    +2.0% on average), and BNB is negative on nearly every day.
  - The follow-up uses a relative rule. GNS (or BNB) is "weak" when it is below its own trailing
    252-session 20th percentile (BNB: below its 20th percentile, meaning misses are punished
    unusually hard). It leads if weak on ≥ 10 of the 20 sessions before the flag. The base rate is
    the share of all days weak.
  - The predictive gain is also split by year.
- **E4 data-validity fix, 2026-11-25 ~01:10 ET:**
  - The shared calendar's reported EPS ends 2026-09-24. After that the window only shrank, so
    Oct–Nov 2026 values were a stale, dwindling sample.
  - Factors are now NaN when their newest report is more than 10 sessions old. Episodes with fewer
    than 15 evaluable pre-flag days are "not evaluable".

## 5. Addendum, pre-registered 2026-11-25 21:31 ET (before any result below)

The user asked four things:
- is too much (old) data hurting us in "unprecedented times";
- should the 20-day target adapt to conditions;
- predicted vs actual regime, and can a VIX "curl" predict a real spike;
- does fading VIX spikes (short vol / buy equity) pay.

Context for the reader: the E2/E9 books already re-decide every day. Only the E1/E3 *labels* are
fixed 20-day windows.

### E10 Less data: rolling windows vs all history (local)
- **Models:** E3's clipped logistic regression on the base features, and on base + long-span
  families. LightGBM on base + long-span families. HMM-R K=2 sizing (E9).
- **Targets:** T1 (Nasdaq 20d ≤ −5%) and T2 (60d ≤ −10%).
- **Windows,** for each yearly refit:
  - expanding (all history; the current design);
  - rolling 10, 5 and 3 years (the most recent years only).
  - The purge is as in E3b.
- **Metric:** out-of-sample log-loss improvement of each rolling window over expanding, by decade,
  with a 60-session block-bootstrap CI over all years. For the HMM: book Sharpe and MDD
  (E2 execution).
- **Decision:** "less history helps" if a rolling window beats expanding in ≥ 2 of the last 3
  decades AND its all-years CI is above 0. The 2020s cell answers the "unprecedented times"
  question on its own and is reported separately.
- The *feature* side of "less data" is already answered by E3 (adding families hurt) and E9 (the
  7-input vote beat the 30-feature model). It is not re-run.

### E11 Adaptive horizon: first-passage targets vs fixed 20 days (local)
- **Targets** (Nasdaq-100, from the close of t; the purge is 60):
  - T1 fixed: 20d path MDD ≤ −5%.
  - **FP5 (primary):** first passage at fixed ±5% with at most 60 sessions. 1 if −5% is touched
    before +5%. This adapts to conditions: it resolves in days in a fast tape and in weeks in a
    calm one.
  - FPσ (secondary): barriers at ±1·RV20·√20 (vol-scaled), max 60.
  - Days-to-resolution are reported for each.
- **Models:** clipped logistic regression and LightGBM on base + long-span families, walk-forward
  (E3 spans).
- **Books:** E9's out-of-fold-cutoff mapping (the follow-up method) for each target's score, on the
  same span.
- **Decision:** FP5 is preferred over T1 if its book beats T1's (Sharpe-difference CI above 0, or
  ≥ 3 of 4 crash wins with the CI not below 0). Predictability (log-loss vs climatology) is
  reported beside it.

### E12 VIX: predicted vs actual, spike prediction, and fading spikes (local)
- **E12a, predicted vs actual regime:**
  - Predicted turbulent:
    - long-history vote ≥ 4;
    - HMM P(turbulent) > 0.5;
    - live vote ≥ 4 (2021+).
  - Actual turbulent (ex post): next-20-session realised vol above the trailing-252 75th
    percentile of RV20. A second definition: a next-20-session drawdown ≤ −5%.
  - Report precision P(actual | predicted), recall and the base rate, by decade. This is the
    "60% vs 50%" view. It is descriptive.
- **E12b, can a VIX spike be predicted?**
  - Target S5: the max VIX over the next 5 sessions ≥ 1.25 × VIX_t. Secondary S10: the max over 10
    sessions ≥ 1.40 × VIX_t.
  - Features: E3 base; VIX "curl" (1d and 5d log change, VIX z, VRP); credit 20d change; sector
    correlation; VIX/VIX3M (2007+, on its own span).
  - Walk-forward clipped logistic regression and LightGBM. Spans: VIX 1991+ and sectors 1999+.
  - Metrics:
    - log-loss vs climatology;
    - top-decile precision vs the base rate, with a block-bootstrap CI;
    - "curl adds" = the curl features block-permuted (100 draws) inside the full model.
  - **Decision:** spikes are "predictable" if the model beats climatology AND its top-decile
    precision exceeds the base rate with a CI above 0, in ≥ 2 of 3 decades (1990s are included
    where they exist).
- **E12c, fading spikes** (entries at the close of t+1 after a signal at the close of t):
  - **Signals:**
    - M1 "spike": VIX_t / min(VIX over t−10..t) ≥ 1.5.
    - M2 "spike, then the first down day": M1 true within the last 5 sessions, VIX_t < VIX_{t−1},
      and VIX_t ≥ 1.3 × that 10-day min. M2 responds to the turn rather than predicting it.
    - After an entry, signals are ignored for the holding period (no overlapping trades).
  - **Trades:**
    - (i) long the S&P 500 (index, 1990+) or Nasdaq-100;
    - (ii) short vol = −1× the daily VIXY return minus a 1.35%/yr fee (SVIX-like), 2011+.
  - **Holding periods:** 1, 5, 10 and 20 sessions. Report mean, median, hit rate, worst trade and
    the trade list.
  - **Control:** random entries with the same count and holding, avoiding signal-window overlap
    (1,000 draws), plus the unconditional always-short-vol carry book.
  - **Decision:** a fade is an E8 shadow candidate if its mean AND hit rate beat the random-entry
    control's 95th percentile at the pre-registered primary holding (10 sessions), on both
    instruments where they exist. Its worst trade is always reported, and the decision is made at
    a size where that worst trade is survivable. This is not a live recommendation.
