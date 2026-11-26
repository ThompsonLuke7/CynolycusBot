# Regime experiments: results

Design and pre-registered rules: [PLAN.md](PLAN.md). Each script's docstring holds its spec. Any
change made after a first run is dated and labelled in that docstring and below. All results are
research on daily or 5-minute index data. None is a module backtest, and none changes live trading.

| Step | Status (2026-11-24) | Verdict |
|---|---|---|
| E0 data | re-run 21:25 ET with the FRED key | complete except HY OAS before 2023 (FRED license) |
| E1 states vs futures | run | **fails the pre-registered rule on Nasdaq-100 (3/5 decades)**; passes on S&P (6/7) |
| E2 general policy | run | **all four books pass the rule**; the live multipliers and a 15% vol target are the strongest |
| E3 missing factor | run (+ clipped-feature variant, sector split) | **no family adds beyond price/vol at 20d** (not "they don't matter": see E3b); VIX, dollar, sector correlation = 2020s-only context |
| E5 intraday granularity | run (+ FOMC-flag follow-up) | **30-minute refresh + calendar flags**; direction unpredictable at any clock |
| E3b audit of E3's null (2026-11-25) | run, 6 targets | factors **do** carry information alone; at 20d price/vol already holds it; at 60–120d the price/vol baseline breaks and VIX/credit beat it |
| E9 trained regime model | run, 12 variants | HMM on returns beats the vote but only equals a 15% vol target; supervised drawdown models lose to the vote |
| E7-M momentum policy replay | run (validation only) | de-risking momentum entries in RISK_OFF is **backwards** (worse than random) |
| E4 "good news sold" | re-run on refreshed earnings (2026-11-25) | **confirmed: tech beats sold in Nov (−0.2% vs SPY); GNS led 2 of 3 named episodes → E8 shadow candidate** |
| E8 forward shadow | logger built, first row 2026-11-24 | **wired into `nightly_data_readiness.sh` (post-stamp, non-fatal) 2026-11-25** |
| E3c known-answer controls | run | test detects corr-0.11 signals; false-positive rate 5% (nominal): the nulls are real |
| E10 less history (2026-11-25 evening) | run | **all history beats rolling 10/5/3-year windows everywhere, including the 2020s** |
| E11 adaptive horizon | run | first-passage targets are *less* predictable (direction); daily re-deciding books already adapt |
| E12 VIX | run | regime calls hit 60–72% vs 42% base; VIX spikes not predictable (curl adds nothing); fading spikes ≈ carry with −63% tails |
| E6, other modules' E7 | not run | E6 is Colab; E7 for HTF/Meta/30m/SPY/intraday lacks clean validation inputs (see E7-M) |

## E0 addendum: FRED (run 2026-11-24 21:25 ET, `panel_20261124T2125.parquet`)
- **Added:**
  - Baa−10y spread from 1986 (the pre-2007 credit factor);
  - 10y−3m curve from 1982;
  - fed funds from 1954;
  - initial claims from 1967 (weekly, 5-day availability lag);
  - NFCI from 1971 (latest vintage, *not* point-in-time, so it is not used).
- **ICE BofA HY OAS on FRED only starts 2023-11.** The vendor license limits FRED to 3 years, so
  the Baa spread is the only credit series before HYG (2007).
- **Cross-check:** FRED's 10y−3m and Yahoo's ^TNX−^IRX correlate 0.994.
- **Carry rule (documented in `regime_common.py`):** yields, the dollar and WTI are carried
  forward at most 5 sessions on bond or NYMEX holidays while stocks trade. WTI needed this on 15
  half-day sessions in 2000–2016. Nothing else is filled.

## E1: does a regime state describe different futures? (`e1_state_futures.py`)

The long-history vote (`regime_common.py`) mirrors the live 7-vote. The five price and volatility
votes are kept as they are. VIX z>1, curve inversion and Baa z>0 replace breadth, risk appetite and
HYG/IEI, which have no long history. Targets run from the close of t: the 20-session path max
drawdown, 20-session realised vol and the 20-session return. The statistic is eta² (the share of
the target's variance explained by the 5-state label). The null is a circular shift of the state
series within each decade (1,000 draws).

| | 1980s | 1990s | 2000s | 2010s | 2020s | Rule |
|---|---|---|---|---|---|---|
| Nasdaq-100 (primary), drawdown / vol eta² | 0.03 / 0.06 | 0.02 / 0.03 | **0.18 / 0.30** | **0.06 / 0.22** | **0.09 / 0.28** | 3 of 5 pass (4 needed): **not useful** |
| S&P 500 (secondary, 1960s–2020s) | – | – | – | – | – | 6 of 7 pass (6 needed): useful; fails the 1980s |

Bold means the result beats the placebo's 95th percentile.

Nasdaq-100 forward 20-session outcomes by state, 2020s (95% block-bootstrap CI):

| State | Share of days | Mean path MDD | P(MDD ≤ −5%) | Realised vol | Mean 20d return |
|---|---|---|---|---|---|
| RISK_ON | 59% | −5.0% [−5.9, −4.3] | 38% | 20% | +0.7% |
| NEUTRAL | 23% | −4.9% | 35% | 21% | +2.3% |
| DETERIORATING | 6% | −6.2% | 45% | 27% | +2.9% |
| RISK_OFF | 11% | −8.4% [−10.3, −6.7] | 76% | 38% | +3.2% |

- **The two failed Nasdaq decades fail for different reasons.**
  - The 1980s cell is underpowered: it has only 1986–89, and its placebo 95th percentile is
    eta² 0.08, about three times later decades', so even the 2010s' effect would not register.
    That is "not measurable here", not "no effect".
  - The 1990s is a real miss with adequate power (placebo 85th/92nd percentile). In the 1990s tech
    tape, bearish states were followed by only slightly worse drawdowns (−7.1% vs −5.4%).
- **Since 2000 the state describes risk consistently on both indexes.** RISK_OFF roughly doubles
  the chance of a 5% drawdown in the next 20 sessions.
- **It describes risk, not direction.** Forward 20-day returns are *higher* in RISK_OFF than in
  RISK_ON in every decade except the 2000s. This matches the earlier finding that shorting on the
  vote loses.
- **Beyond trailing volatility, it adds to forward vol but not to forward drawdown.** Diagnostic
  (a): after removing log trailing RV20, the state still explains forward vol in 2000s–2020s but
  not forward drawdown. So the vote's drawdown information is mostly volatility clustering.
- **The comparators are weaker.** RV-only and SMA50-only states are weaker than the vote in most
  decades; RV-only fails the 2000s on Nasdaq.
- **The long-history vote vs the live vote** (live built in memory with `build_daily_regime()`,
  2021-10..2026-11):
  - the BEAR flag (≥4) agrees on 90% of days, and vote counts correlate 0.89;
  - the 5-state labels agree on only 56%;
  - the long-history vote is milder (BEAR 22% vs 30% of days). On the live vote's 111 CRISIS days
    (mostly 2022) it reads RISK_OFF.
  - It is also slower. In the current selloff it reached BEAR on 11-05, against 10-16 for the live
    vote, because breadth, appetite and HYG/IEI (live-only inputs) flagged first.
- **Next testable variant:** drop the curve vote. Inversions last months to years (1998, 2000,
  2006–07, 2019, 2022–24) and cannot describe a 20-session future. Rerun E1 on the 1990s with only
  the price and volatility votes. Not run.

## E2: a general regime policy on 40 years (`e2_general_policy.py`)

The books trade the index itself. Each weight is decided at the close of t and traded at the close
of t+1 (the plan said the next open; see below). Cash earns the 3-month bill, and every unit of
turnover costs 2 bp.

**Nasdaq-100, 1986-10..2026-11** (price index):

| Book | CAGR | Sharpe | MDD | Invested | Sharpe − B&H, 95% CI | Crash wins vs B&H | vs matched exposure | MDD placebo pct |
|---|---|---|---|---|---|---|---|---|
| Buy and hold | 13.9% | 0.51 | −83% | 100% | – | – | – | – |
| Long/flat, vote ≥ 4 | 14.0% | 0.63 | −64% | 81% | [−0.08, +0.33] | 11/11 | 11/11 | 96 |
| **Live multipliers** (1/1/0.8/0.5/0.25) | **15.1%** | 0.62 | −73% | 93% | **[+0.03, +0.18]** | 11/11 | 10/11 | 99 |
| **15% vol target** | 12.8% | **0.67** | **−40%** | 76% | **[+0.03, +0.28]** | 11/11 | 11/11 | 100 |
| SMA200 | 12.1% | 0.55 | −39% | 76% | [−0.17, +0.27] | 11/11 | 9/11 | 100 |

**S&P 500, 1962–2026:** every book passes. SMA200 has the best Sharpe (0.40 vs 0.27) and MDD (−22%
vs −57%).
**S&P total return, 1988–2026 (dividends):**
- the defensive books give up 0.3–1.8 points of CAGR (9.4–10.9% vs 11.2%);
- they gain Sharpe (0.57–0.59 vs 0.52) and about 10–33 points of MDD;
- every book still passes the rule.

Crash drawdowns, Nasdaq-100 (peak to trough of each crash):

| Crash | Index | Long/flat v≥4 | Live multipliers | Vol target 15% | SMA200 |
|---|---|---|---|---|---|
| 1987 | −40% | −4% | −23% | −22% | −25% |
| 2000–02 | −83% | −63% | −72% | −40% | −37% |
| 2007–09 | −54% | −18% | −40% | −26% | −16% |
| 2020 | −28% | −8% | −18% | −14% | −21% |
| 2022 | −36% | −29% | −35% | −20% | −18% |
| 2025 | −23% | −8% | −13% | −15% | −9% |
| 2026 (ongoing, 10-06..11-20) | −20% | −15% | −17% | −14% | −17% |

- **The pre-registered rule passes all four books** on Nasdaq-100, S&P 500 and S&P total return,
  so it does not pick one. The tie-breakers are below.
  - **The live multipliers** are the only book with *higher* CAGR than buy-and-hold together with
    a Sharpe CI above zero. They survive the matched-exposure control in 10 of 11 crashes. This is
    index-level support for the 2026-11-23 adoption, using the long-history vote in place of the
    live one.
  - **The 15% vol target** is the strongest risk reducer: MDD −40% vs −83%, a Sharpe CI above zero,
    and 11/11 crashes vs matched exposure. It costs about a point of CAGR on Nasdaq.
  - **Long/flat on the vote** has the best crash record against matched exposure (11/11 on both
    indexes and 10/10 with dividends), but its Sharpe CI includes zero.
- **The value is concentrated in bear decades.** In the 1990s bull, every book's Sharpe is at or
  below buy-and-hold's (Nasdaq 0.89–1.10 vs 1.09). Most of the gain comes in the 2000s and 2020s.
- **This crash was protected less.** In 2026, long/flat saved 5 points and the multipliers 3,
  because the long-history vote flagged on 11-05. The live vote flagged on 10-16.
- **Changes from the plan:**
  - *Execution, changed before the first run.* Yahoo index opens before 2000 (Nasdaq) and 2008
    (S&P) are copies of the prior close, so trades go at the next close. Where opens are real,
    next-open and next-close agree within 0.1 Sharpe (`e2_*_next_open.csv`).
  - *Crash windows, a bug fixed after the first run.* The peak was taken first, which turned the
    Nasdaq "2020 crash" into a 3-day June dip and dropped 2025. The trough is now found first and
    the peak before it. Only the crash table changed.
  - *Matched-exposure control, added after the first run* because every book passed. It is a
    constant weight equal to the book's mean exposure. By construction its Sharpe equals
    buy-and-hold's, so only its crash and MDD comparisons are informative.
- **Limits:**
  - These are index books. Whether a module's own book gains from the same overlay is E7.
  - The long-history vote is a proxy for the live one (see E1).
  - Index returns are price-only except in the S&P total-return check.
- **Next testable variant:** vol target × live multiplier combined, and the same books on the
  *live* 7-vote over 2021–2026 (short, MDE to be stated). Not run.

## E5: how fine-grained must the intraday state be? (`e5_intraday_granularity.py`)

- **Data:** SPY and QQQ 5-minute SIP bars, 2016-02..2026-11-20, dividend-adjusted (every input is a
  within-session ratio, so the factor cancels; 2,711 and 2,704 full sessions;
  half days excluded), plus VIXY 5-minute bars (split-adjusted; 0.1–0.2% of decision bars carry the
  last VIXY print).
- **FOMC calendar:** 2016–2021 from federalreserve.gov, 2022+ from the repo. Checked in the data:
  on 86 FOMC days the 14:00–14:05 SPY bar is a median 4.3× that day's typical 5-minute range.
- **Decisions:** every 5-minute bar close from 10:00 to 15:30. This was clarified before running:
  the plan's checkpoints sit on the 30-minute grid, where a 30-minute and a 5-minute refresh are
  identical.
- **Method:** test years 2018–2026, ~149k decisions per symbol. The gain is each arm's loss
  reduction over the daily-only arm (a); 95% CIs come from a whole-session bootstrap.

| Refresh of the session state | SPY range gain (share of 5-min) | QQQ range gain (share) | Direction gain, SPY / QQQ (millinats) |
|---|---|---|---|
| (b) once, at 10:00 | 37% [26, 49] | 41% [28, 56] | −1.6 / +0.9 |
| **(c) every 30 minutes** | **91% [74, 111]** | **92% [76, 113]** | −2.5 / +0.7 |
| (d) every 5 minutes | 100% (MSE −23% vs daily-only) | 100% (−22%) | −2.6 / +0.7 |
| (e) once + FOMC 14:05/14:35 + shock-bar refreshes | 63% [51, 77] | 60% [48, 78] | −2.4 / +0.3 |

- **Range and volatility: refresh every 30 minutes.** It is the coarsest clock that reaches 80% of
  the 5-minute refresh's gain (point estimate 91–92%; the lower CI bound is 74–76%). Its share
  grows through the day, from 83% before 11:00 to 93% after 14:30. A state computed once at 10:00
  captures only about 40%. Event-triggered refreshes on top of it reach about 60% and do not
  replace a clock.
- **Direction: no refresh rate helps.** The daily-only log-loss is 0.694 (a coin flip is 0.693),
  and no arm beats it. Every direction gain lies within ±2.5 millinats of zero (CIs about ±2.5),
  so any direction information is below this test's MDE. The rule's "pick b" for QQQ direction is
  a ratio of two zeros, not a result.
- **Events: do not add event-triggered refreshes; publish the calendar instead.** On FOMC afternoons
  (70 test days), the schedule flag alone beats the event-refreshed state: range gain +0.50
  [+0.34, +0.65] over the once-a-day state, against −0.25 [−0.36, −0.15] for refresh vs schedule.
  - *Follow-up, labelled.* The FOMC flag on top of the 30-minute state still helps: +0.42
    [+0.30, +0.53] on FOMC afternoons for SPY and +0.45 for QQQ, and +19/+16 millinats across all
    rows.
  - *Recommended session state:* a 30-minute refresh plus calendar flags that are known in
    advance.
- **Limits:**
  - Index ETFs only.
  - The targets are rest-of-day range and direction, not a module's P&L. Whether a module's
    policy gains from the state is E7.
  - Linear models only. A non-linear model might find direction information that these did not.
- **Next testable variant:**
  - a trend-day vs range-day target (rest-of-day |return| / range), which is what the intraday
    structure module's hold rule needs;
  - CPI/NFP-day flags in the same design;
  - neither run yet.

## E3: are we missing a factor? (`e3_factor_completeness.py`)

- **Target:** a Nasdaq-100 20-session path drawdown of 5% or worse (base rate 34–42%, depending on
  the span).
- **Method:** walk-forward with a yearly refit and a 20-session purge. The model is logistic
  regression with the scaler fit on training rows only; LightGBM is a secondary check. Each family
  is tested on its own span: baseline vs baseline+family on identical rows.
- **Baseline:** price trend plus realised vol. It already beats climatology clearly, at log-loss
  0.609 vs 0.684.
- **Control:** each family's rows block-permuted in 60-session blocks (100 draws). Its 95th
  percentile is the null and the MDE, about +3 to +7 millinats per decade, which is roughly a
  tenth of the baseline's skill.

Out-of-sample improvement over the baseline (millinats; * = beats the permuted 95th percentile):

| Family | Span | 1990s | 2000s | 2010s | 2020s | Rule (2 of the last 3 decades) |
|---|---|---|---|---|---|---|
| VIX level, z, VRP, 5d change | 1991+ | −68 | −5 | −5 | **+22*** | 1/3: context |
| VIX3M/VIX term | 2007+ | – | – | −2 | −6 | 0/2 |
| Credit (Baa spread) | 1987+ | −9 | −7 | −2 | −5 | 0/3 |
| Rates and curve | 1987+ | −34 | −1 | −6 | 0 | 0/3 |
| Dollar | 1987+ | −17 | −3 | −4 | **+15*** | 1/3: context |
| Oil | 2001+ | – | −51 | −2 | −4 | 0/3 |
| Nasdaq/S&P and Russell/S&P relative | 1988+ | −11 | −2 | +1 | +4 | 0/3 |
| **Sectors: breadth + mean pairwise correlation** | 1999+ | – | **+16*** | −2 | **+14*** | 2/3 raw; **1/3 clipped**: not robust |
| Initial claims | 1987+ | 0 | −2 | −1 | −159 | 0/3 |
| *split, post-hoc:* sector breadth only | 1999+ | – | −2 | 0 | −2 | 0/3 |
| *split, post-hoc:* sector correlation only | 1999+ | – | **+18*** | −2 | **+17*** | 2/3 raw; 1/3 clipped (2000s −4) |

- **The raw-feature run passes one family, and the clipped variant removes it.**
  - **Raw features:** only "sectors" passes, and all of its signal is sector correlation (the mean
    pairwise 60-day correlation of the 9 sector SPDRs: everything moving together). Breadth alone
    adds nothing; the live vote already has breadth but not correlation.
  - **Clipped features** (the next variant, `E3_WINSOR=1`; same rule and seeds): sector
    correlation's 2000s pass disappears (−4 vs a +2 null), leaving 1 of 3, which is not crucial.
    The raw 2000s pass was 2004 alone (+136 millinats), the first test year, scored by a model
    trained only on 1999–2003, so it was an out-of-range artifact.
- **Verdict: no family is robustly crucial and missing.** (Read with E3b: on their own several of
  these families do forecast drawdowns; E3 only asks whether they add beyond price and vol at 20
  days, and at 60–120 days that is the wrong comparison.) At an MDE of about 3–7 millinats per
  decade, no daily macro or cross-asset family adds 20-day drawdown information beyond price and
  vol in two or more decades.
- **The 2020s are the exception, which makes these context.**
  - VIX (+19*), the dollar (+14*) and sector correlation (+17*) each beat their nulls in the 2020s,
    with raw and clipped features alike.
  - With clipping, all families together help in the 2020s (+26) and hurt in every earlier decade.
    LightGBM agrees (+21 in the 2020s).
  - Whether that is a new structure or two episodes (2020, 2022) cannot be told in-sample. These
    three go into the forward shadow (E8) as logged candidates, not into the vote.
  - In the episode audit, the correlation arm's lift before the price votes flagged was 2–7× its
    all-days average (+0.025) in 7 of 12 episodes since 2015, and +0.185 before the current one
    (flag 11-05). Its per-year gains are concentrated: 14 of 23 years are positive, and the top
    three years carry 126% of the net gain.
- **Credit, rates, curve, oil, claims and term structure add nothing measurable** in any recent
  decade, raw or clipped.
  - The credit null matters, because the HYG/IEI vote is live. It says the Baa spread's
    information about 20-day drawdowns is already in price and vol. It does not say the live
    credit vote is useless as a state description.
- **Why raw levels broke the linear model:**
  - The 2020 claims spike cost −159 millinats in the 2020s; clipped, it costs −14.
  - The 1990s VIX loss (−68 raw, −61 clipped) comes from fitting on only four years.
  - The all-families arm is the clearest case: in the 2020s it goes from −185 raw to +26 clipped.
- **Next testable variant:**
  - Split the 2020s gains by year, to tell structure from two episodes. This is cheap and needs
    no new data.
  - The short-history families (E4: "good news sold", news tone, dealer gamma, put/call).
  - The hypothesis class "long-history macro family beats price and vol before 2020" is
    exhausted at this MDE.

## E3b: do VIX, credit, the curve and term structure matter, or did E3 test them wrong? (`e3b_null_audit.py`, 2026-11-25)

E3 asked one narrow question: does a factor family add information *beyond price and volatility*,
about one target (a 20-session Nasdaq drawdown ≥ 5%), with a linear model. E3b relaxes each choice:

- the family **alone** vs the base rate;
- six targets: 20, 60 and 120 sessions, a −10% tail, and the S&P 500;
- slow features (curve inversion share, 120-day credit change);
- clipped logistic regression (100 permutations) and LightGBM with its own null (20).

It reproduces E3's clipped run exactly (45 cells, difference 0.0).

Family alone vs the base rate (millinats; * = beats both the base rate and its own permuted 95th
percentile), next to the price/vol baseline:

| Target | price/vol baseline | VIX | credit (Baa) | credit, slow | VIX term | sector breadth | curve + rates | curve, slow |
|---|---|---|---|---|---|---|---|---|
| Nasdaq 20d ≤ −5% | **+84** | **+54*** | +12* | +9* | +22* | +24* | −3 | −33 |
| Nasdaq 20d ≤ −10% (tail) | +29 | +30* | +16* | +13* | 0 | +10* | −39 | −70 |
| Nasdaq 60d ≤ −10% | **−23** | **+28*** | +17* | **+27*** | −12 | +9* | +6* | −33 |
| Nasdaq 120d ≤ −15% | **−164** | +15* | +4* | +11* | −1 | +20* | −37 | −286 |
| S&P 20d ≤ −5% | +55 | **+79*** (beats price/vol alone; overlap 0.89) | +14* | +13* | +49* | +25* | −27 | −46 |
| S&P 60d ≤ −10% | **−17** | +13* | **+30*** | +18* | +5* | +21* | −88 | −14 |

- **Adding beyond price/vol is rare and not consistent.** Of 72 family × target tests, 4 pass with
  the linear model and 4 with LightGBM.
  - **Sector correlation** accounts for 3 of the 4 linear passes (Nasdaq tail, S&P 20d, S&P 60d).
    But each target passes in a different pair of decades (S&P: 2000s and 2010s; Nasdaq: 2010s or
    2020s), the signs flip elsewhere, and LightGBM disagrees in most cells. That instability is
    what chance and regime-specific behaviour look like. It stays a watch item.
  - **LightGBM finds credit adding at 20 days on both indexes** (2 of 3 decades). That rests on only
    20 null draws, so it is the one lead worth a full-power test.

- **Yes, they matter, as risk barometers.**
  - On their own, VIX, credit spreads, VIX term structure and sector breadth forecast future
    drawdowns out of sample, beyond their permutation nulls, on almost every target.
  - That is the sense in which practitioners are right to watch them.
- **At a 20-day horizon, price and realised volatility already carry the same information.**
  - VIX's out-of-sample forecasts correlate 0.66 with the price/vol model's, and term
    structure's 0.59. VIX is largely a market forecast of the volatility that recent prices
    already show.
  - So adding them on top gives nothing, which is all E3 measured.
  - LightGBM, with interactions, finds credit adding beyond price/vol in 2 of 3 decades at 20
    days. That rests on only 20 null draws, so it is the one lead worth a proper test.
- **At 60–120 days the price/vol model breaks, and credit and VIX do better than it.**
  - The baseline is *worse than the base rate* there: −23 and −164 millinats. Trained on the
    1990s bull, its 2000s scores were −193 to −500.
  - In that decade VIX alone (+65/+75) and credit (+74; slow credit +91 / +69) caught the
    2000–02 and 2008 bears.
  - So "beyond price/vol" is the wrong test at long horizons, and E3's null understated these
    factors there.
  - Their value is concentrated in the 2000s credit cycle: little in the 2010s and nothing in the
    2020s. That is why 2008 dominates the conversation about credit.
- **The yield curve is the exception: it hurt at every horizon tested.**
  - Its famous record is about *recessions* 6–18 months out, not equity drawdowns.
  - In this sample its longest signal (inverted 2022-11..2024-12) came during a bull market.
  - With about five inversion episodes in 40 years, no test here can certify or reject it for
    recessions.
- **Why so many people discuss them:**
  1. **They explain moves as they happen.** Spreads widen and VIX jumps *with* a selloff. That is
     real and useful for describing the state, but it is coincident, not a forecast.
  2. **They matter at the horizons and for the assets people care about:**
     - credit and the curve for the business cycle (months to years);
     - rates for sector rotation;
     - the dollar for multinational earnings;
     - oil for energy and inflation.
     Our targets are index drawdowns over weeks.
  3. **Memorable episodes:** 2008 for credit, 2000 and 2007 for the curve. The false alarms are
     forgotten.
  4. **Few independent episodes.** A cycle-length factor gives a handful of signals in 40 years,
     which is too few to prove or disprove statistically. That leaves room for strong opinions in
     both directions.

## E3c: does the test machinery work? Known-answer controls (`e3c_controls.py`)

- **Positive controls:** a synthetic family built from the *future* label plus noise, to measure
  power.
- **Negative controls:** 20 pure-noise families, to measure false positives.
- Both run through E3's exact walk-forward, null and rule.

| Control | Correlation with the label | Gain over price/vol | Decades passed (of 3) | Verdict |
|---|---|---|---|---|
| Positive, s = 0.05 | 0.11 | +5.1 millinats | 3 | detected |
| Positive, s = 0.1 | 0.21 | +19.1 | 3 | detected |
| Positive, s = 0.2 | 0.35 | +42.4 | 3 | detected |
| Positive, s = 0.4 | 0.61 | +190.3 | 3 | detected |
| 20 noise families | – | – | 6.7% of decade tests | 1 of 20 false positive (nominal 5%) |

- **The test is not blind:** it detects a weak signal (correlation 0.11, +5 millinats) that price
  and vol do not already contain.
- **The test is not trigger-happy:** its false-positive rate is at the nominal 5%.
- **So the E3/E3b nulls are real "adds nothing beyond price/vol" results, not an insensitive test.**
  Chance alone predicts about 3.6 passes among E3b's 72 linear tests; there were 4.
- **One caveat:** the positive controls carry information orthogonal to price and vol by
  construction. A real factor that *overlaps* price/vol (VIX: 0.66–0.89) is correctly not credited
  with the overlapping part.

## E9: a trained regime model vs the hand vote (`e9_regime_model.py`, 2026-11-25)

Twelve walk-forward variants were tested on Nasdaq-100. The rules were pre-registered in PLAN.md §4.

- **The HMMs** use filtered probabilities only (data up to t, with parameters fit before the test
  year). Smoothed or Viterbi paths, which see the future, are never used.
- **The supervised models** output the probability of a 20- or 60-session drawdown, from every
  long-span factor family.
- **The books** use E2's execution.
- **The primaries,** fixed before the run, are HMM-R K=2 and SUP-LightGBM-T1.

| Model (out-of-sample span) | CAGR | Sharpe | MDD | Turnover/yr | Sharpe − vote, 95% CI | Crash wins vs vote | vs 15% vol target |
|---|---|---|---|---|---|---|---|
| Buy and hold (1991+) | 14.6% | 0.55 | −83% | – | – | – | – |
| Vote × live multipliers (1991+) | 15.4% | 0.64 | −73% | 4 | – | – | – |
| 15% vol target (1991+) | 13.0% | 0.71 | −40% | 4 | – | – | – |
| **HMM on returns, 2 states, sized** (primary) | 14.0% | **0.72** | **−38%** | 11 | [−0.04, +0.21] | **7/9** | Sharpe CI [−0.06, +0.08]; 3/9 crashes |
| HMM, 2 states, long/flat | 14.2% | 0.75 | −32% | 12 | [−0.13, +0.36] | 8/9 | 7/9, CI [−0.14, +0.21] |
| HMM, 3 states | 15.5% | 0.66 | −69% | 4 | [−0.07, +0.10] | 4/9 | 0/9 |
| HMM, multivariate (VIX, Baa, curve; 1995+) | 10.6% | 0.49 | −77% | 16 | [−0.28, −0.02] | 4/9 | worse |
| **Supervised LightGBM, 20d** (primary, 1995+) | 13.7% | 0.54 | −80% | 3 | [−0.20, +0.00] | 1/9 | worse |
| Other supervised variants (5, plus 4 follow-ups) | 11–15% | 0.47–0.62 | −44..−81% | – | all include or sit below 0 | 0–5/9 | none |

- **By the pre-registered rule, the HMM on returns is preferred over the hand vote:**
  - it beats the vote in 7 of 9 out-of-sample crashes without a worse Sharpe;
  - but it does **not** beat the 15% vol target: Sharpe 0.72 vs 0.71, and 3 of 9 crashes;
  - **so it is not worth its complexity.**
  - The reason: the model learned a calm state (about 1.0% daily SD) and a turbulent one (about
    2.3–3.0%). Sizing on P(turbulent) *is* volatility targeting, with more turnover.
  - Its long/flat book had the best drawdown found (−32%), but it is a secondary book and not
    decisive.
- **The supervised regime models lose to the hand vote,** with the same features and walk-forward
  as E3.
  - Their crash drawdowns are close to buy-and-hold's.
  - Predicting *risk* is not the same as improving *risk-adjusted return*. E3 shows the drawdown
    probability is forecastable: log-loss 0.61 against 0.68 for the base rate. But the high-risk
    states it flags also carry higher expected returns (E1: forward returns are higher in
    RISK_OFF). Cutting exposure there gives up the rebound as well as the loss. A volatility
    target cuts exposure in proportion to risk, which is what improves Sharpe.
- **Adding VIX, credit and the curve to the HMM made it worse:** 16 trades a year and a Sharpe of
  0.49. The 5-day changes in those series flip the state far more often than the regime changes.
- **Fixes and follow-ups (dated in the script):**
  - A name-prefix bug first scored the supervised primary on 2004+ only. It is fixed; the table
    is the 1995+ result.
  - In-sample percentile cutoffs left the supervised books 94–99% invested. Cutoffs from
    out-of-fold training scores (follow-up) raised de-risking but not the result (Sharpe 0.57–0.62).
- **Next testable variant:**
  - the HMM state as a *vol-target input*, sizing by the model's forecast state volatility instead
    of trailing RV20;
  - a supervised target that is risk-adjusted (forward Sharpe sign, or forward return ÷ forward
    vol) rather than drawdown.

## E7-M: momentum policy replay on Stage 3 validation (`e7_momentum_policy.py`)

- **Setup:**
  - Momentum slot model L2_20__pit; out-of-fold scores on 3,542 validation 4H bars, 2019-03..2026-04.
  - Every statistic stops before 2026-05-15 minus the embargo, so the sealed test was not read.
  - Each bar's book buys the top 10, and each trade earns its 20-session return net of costs.

| Vote at entry (long-history vote) | Bars | Mean trade return | SD | 10th percentile | Model selection (top 10 − all names) |
|---|---|---|---|---|---|
| RISK_ON | 2,040 | +0.33% | 12.9% | −12.8% | +0.48% |
| NEUTRAL | 812 | +1.39% | 12.4% | −11.4% | +0.29% |
| DETERIORATING | 250 | +1.73% | 10.5% | −12.3% | −0.42% |
| RISK_OFF | 422 | **+5.24%** | 16.0% | −13.4% | +2.70% |
| CRISIS | 18 | +2.8% | 23.9% | −32.3% | +4.9% |

| Profile | Return per unit exposure vs current | 95% CI | Shuffled-regime percentile | Verdict |
|---|---|---|---|---|
| Live multipliers by state | −0.27 pts (1.00% vs 1.27%) | [−0.66, +0.11] | 8th | **no**: worse than random |
| No entries when vote ≥ 4 | −0.64 pts | [−1.59, +0.23] | 9th | **no** |
| Live 7-vote, 2021-10+: multipliers / long-flat | −0.44 / −0.85 pts | – | 9th / 17th | **no** |

- **For momentum entries, the de-risking profile is backwards.**
  - Trades entered in RISK_OFF earned the most: +5.2% vs +0.3% in RISK_ON. Their left tail was
    hardly worse (10th percentile −13.4% vs −12.8%).
  - The model's own selection skill was also strongest there.
  - Cutting size by regime removed the best entries more often than a random regime assignment
    would (8th–9th percentile).
  - This is the module-level version of E1's finding: bearish states describe risk, and for a
    model buying relative strength they have been buying opportunities.
- **"The same state, a different response" is now measured, not assumed:**
  - for index exposure, cut size in RISK_OFF (E2);
  - for momentum entries, do *not* cut it.
- **The opposite profile was suggested by this data, so it is not tested on it.** The idea is to
  size *up* momentum entries in RISK_OFF. It goes into the E8 forward shadow, with a read date.
- **Out of scope tonight** (reasons in PLAN.md E7-M):
  - HTF (no validated model);
  - Meta and the 30m swing (no out-of-fold scores here);
  - the SPY daytrader (needs its execution engine);
  - intraday structure (its only ledger was used to design the rule; forward data since 11-23 is
    its test).

## E4: "good news sold" (`e4_good_news_sold.py`; re-run 2026-11-25 22:20 ET on the refreshed calendar)

- **Data:** 89,645 earnings reports, 2016-01..2026-11-24. The first run stopped at 2026-09-24,
  because the shared calendar's reported EPS had frozen there.
  - The union refresh on 2026-11-25 added 2,449 reports: the whole Oct–Nov season.
  - It lost no prior rows, and 99.92% of the 100,644 overlapping surprises agree.
- **Definitions:**
  - Reaction = the two-day move around the report minus SPY.
  - GNS = the mean reaction of beats (surprise ≥ +5%) over the last 60 sessions; GNS_tech = the same
    for XLK-assigned names.
  - BNB = the mean reaction of misses.
  - A factor is NaN when its newest report is more than 10 sessions old. That guard was added after
    the first run's stale-feed artifact, whose "record November BNB" is retracted.

| | 2016–2020 | 2021 | 2023–2025 | Sep–Oct 2026 | Nov 2026 |
|---|---|---|---|---|---|
| GNS, all beats vs SPY | +2.0 to +2.5% | +1.0% | +2.0 to +2.2% | **+0.9 to +1.2%** | +2.1% |
| GNS_tech, tech beats vs SPY | +2.1 to +2.7% | +1.2% | +2.0 to +2.5% | +1.0 to +1.6% | **+1.1% → +0.3% → −0.2%** |

- **Your AI-earnings observation is confirmed by the data.**
  - In Sep–Oct, beats overall earned half their normal reward.
  - In November, tech beats *underperformed SPY*, while beats elsewhere were rewarded more than
    normal. That is a rotation signature.
  - Examples: PLTR beat by 8% and fell 4.5% vs SPY; SMCI beat by 47% and fell 4.7%; NVDA beat by 8%
    and fell 2.0%; TSM beat and fell 3.2%.
- **Lead test** (relative rule fixed before the fresh data, so 2026-10 is out of sample; weak = below
  the trailing 1-year 20th percentile, about 25% of days):
  - GNS was weak on 15 of the 20 sessions before the 11-05 flag; GNS_tech on 14.
  - **GNS leads in 2 of 3 named episodes:** 2025-02 and 2026-10, but not 2024-07 (9 days, one short).
    That meets the shadow criterion.
  - Across all 10 episodes since 2018 it is 4 of 10, against a base rate of about 25%. Suggestive,
    not significant.
- **Predictive check (secondary, 2020s only):** +21.3 millinats against a +6.0 null. It is
  concentrated: 2022 +80, 2026 +56, 2024 −13.
- **Verdict: GNS and GNS_tech are now E8 shadow candidates.**
  - Logged nightly from 2026-11-25; the 11-24 row predates the refresh and is NaN.
  - Not certified: one decade, a survivor-shaped universe, and before-open/after-close timing
    unknown (the two-day window covers both).

## E8: forward shadow ledger (`e8_shadow_log.py`, started 2026-11-25)

- **What it logs:** one append-only row per completed session to
  `Data/research/regime_shadow/shadow_log.parquet`. A date already logged is never rewritten.
  - The live 7-vote, read from the nervous system's own daily-regime table.
  - The long-history vote.
  - HMM P(turbulent), with parameters frozen per calendar year in `hmm_r2_params_<year>.json`.
  - The 15% vol-target weight.
  - The candidates: sector correlation, VIX z, dollar, GNS/BNB (watch only; NaN while the
    earnings feed is stale).
  - Counterfactual weights per profile:
    - index live multipliers, vol target, HMM size, HMM long/flat;
    - **momentum size-up in RISK_OFF/CRISIS** (×1.5). E7-M suggested this, so it is tested only
      forward.
- **It is research only:** no orders, and nothing in live state is written.
- **First row, 2026-11-24:**
  - live vote 7/7 CRISIS; long-history vote RISK_OFF;
  - HMM P(turbulent) 0.99; vol-target weight 0.32;
  - sector correlation at the 94th percentile of its year; VIX z +2.2.
- **Wired 2026-11-25:** in the post-stamp section of `scripts/nightly_data_readiness.sh` (22:15 ET), after
  the 3/6 regime build so it carries that night's live vote. E0 runs first, then E8. It is non-fatal.
- **Read rule (PLAN.md E8):** read at the later of 40 logged sessions or 10 sessions in
  RISK_OFF/CRISIS, with the MDE stated. With nightly logging from 2026-11-25, 40 sessions is about
  2027-01-25.

## E10: is all that old data hurting us? Rolling windows vs all history (`e10_less_history.py`, 2026-11-25)

The same models were retrained every year on only the most recent 10, 5 or 3 years, instead of on
all history. Each window's gain over all history, in millinats (positive = less data is better):

| Model, target | Last 10 years | Last 5 | Last 3 | 2020s only, last 10 years |
|---|---|---|---|---|
| Logistic, price/vol, 20d | −5 [−20, +11] | −60 | −231 | −34 |
| Logistic, all families, 20d | −75 | −622 | −1,241 | −109 |
| LightGBM, all families, 20d | −47 | −91 | −139 | −96 |
| Logistic, price/vol, 60d | −36 [−103, +34] | −238 | −891 | −251 |
| LightGBM, all families, 60d | −82 | −185 | −278 | −134 |
| HMM sizing book, Sharpe (all history: 0.70) | 0.63 | 0.57 | 0.55 | – |

- **Less history is worse everywhere, including the 2020s.** No window passes. The shorter the
  window, the worse the result.
- **The reason:** crashes are rare. A model fit on 2017–2019 has seen one small correction and
  learns that dips are harmless. All history is what teaches it 1987, 2000, 2008 and 2020.
- **"More data isn't always better" is right about *features*, not about *history*.**
  - Adding feature families made models worse (E3).
  - The 7-input hand vote beat the 30-feature model (E9).
  - Keep many years of a *few* good inputs.
- **"Unprecedented times" is mostly not true in the data that matters here.** The 2026 selloff
  looks like past ones on price and volatility: the vote and HMM flagged it as they did before.
  What is new is the narrative (AI). Narrative effects show up as reaction-to-news (E4), not in
  the risk state.

## E12: VIX -- predicted vs actual regime, predicting spikes, and fading them (`e12_vix.py`)

### E12a Predicted vs actual regime (1991–2026; "actual" is measured after the fact)

| When it says turbulent... | Days flagged | Next 20d drawdown ≥ 5% happened | vs base rate | Next-month vol in top quartile | vs base |
|---|---|---|---|---|---|
| Long-history vote ≥ 4 | 18% | **64%** | 42% | **57%** | 27% |
| HMM P(turbulent) > 0.5 | 25% | **72%** | 42% | 44% | 27% |
| Live 7-vote ≥ 4 (2021+) | 29% | **60%** | 43% | 53% | 34% |

- **This is the "60% vs 50%" kind of edge, and it is real.** By decade, the HMM's drawdown
  precision is 60 / 86 / 60 / 63% against base rates of 46 / 56 / 25 / 42%. That is a lift in
  every decade.
- **It is a prediction of *risk*.** E1 and E7-M showed that forward *returns* are not lower in these
  states. That is why the right action is to size down index exposure, not to short.

### E12c Fading VIX spikes

- **Signals:**
  - M1 "spike": VIX ≥ 1.5 × its 10-day low.
  - M2 "spike, then the first down day".
- **Trades:** enter at the next close, with no overlapping trades.
- **Control:** random entries, same count and holding period.

| 10-session hold | Trades | Mean | Hit rate | Random-entry mean (median / 95th pct) | Random hit (median / 95th) | Worst trade |
|---|---|---|---|---|---|---|
| M1 → long S&P (1990+) | 66 | −0.3% | 52% | +0.4% / +1.0% | 61% / 70% | −9.7% |
| M2 → long S&P | 63 | +0.2% | 54% | +0.4% / +1.0% | 60% / 70% | −16.7% |
| M1 → short vol (−1× VIXY, 2011+) | 42 | +3.5% | 57% | +1.5% / +4.7% | 62% / 74% | −52% |
| M2 → short vol | 41 | +3.9% | 66% | +1.7% / +4.7% | 61% / 73% | **−63%** (2020-02-27) |
| Always short vol (carry), 2011+ | – | +14.2%/yr | – | – | – | MDD **−92%**; worst day −43% (2024-08-05) |

- **Buying the index after a VIX spike is no better than buying on a random day,** and its hit rate
  is lower.
- **Shorting vol after a spike earns more than a random day** (+3.5–3.9% vs +1.5% per 10 days),
  **but not beyond the random 95th percentile.** The tail is the whole story:
  - 2020-02-27 lost −63% in 10 days;
  - the current episode lost −25% from 11-09 to 11-23, as VIX kept rising after its first down
    day;
  - always-short lost 92% from peak to trough.
- **VIX does mean-revert,** but the timing of the reversion is not predictable enough to beat the
  contango carry that short vol earns on any random day. The trade is paid for bearing the crash,
  not for timing it.
- **Verdict:** no shadow candidate. If short vol is ever used, it should be small, sized for a −70%
  instrument move, and treated as carry, not as a spike-timing edge.

### E12b Can a VIX "curl" predict a real spike?

- **Target S5:** VIX at least 25% above today's level within the next 5 sessions (7.7% of days).
  S10 is the same with +40% within 10 sessions.
- **Features:** price/vol, the VIX curl (1d and 5d change, z, VRP), credit, sector correlation and
  VIX3M/VIX.
- **Models:** walk-forward logistic regression and LightGBM.

| | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|
| S5 model vs base rate, millinats (logistic / LightGBM) | −51 / −43 | −1 / −8 | −4 / −8 | +5 / +8 |
| S5 top-10% days: spike rate vs base rate | 2% vs 5% | 8% vs 5% | 10% vs 11% | **18–23% vs 11%** |

- **No.** By the pre-registered rule spikes are not predictable: 0 of 4 decades pass for the
  primary model, and at most 1 of 2 or 3 for the secondary variants.
- **The curl adds nothing.** Permuting it does not hurt, and removing it slightly *improves* the
  forecast. VIX rising today does not reliably mean a bigger spike is coming.
- **The 2020s are the one hint:** the top-scored days spiked about twice as often as normal
  (CIs barely clear 0), with VIX3M/VIX inversion helping. It is one decade, so it is a
  forward-shadow item, not a rule.
- **This is the "responding beats predicting" point in numbers.** The spike itself can't be called
  in advance. But once turbulence has started, it reliably persists (E12a: 60–86% precision), and
  responding to it (the vol target, E2/E9) is what has worked.

## E11: should the target adapt instead of a fixed 20 days? (`e11_adaptive_horizon.py`)

- **Targets:** fixed 20d ≤ −5% (T1) vs first passage: which comes first within 60 sessions, −5% or
  +5% (FP5), or a vol-scaled ±1σ (FPσ).
- **It adapts as intended:** FP5 resolves in a median of 15 sessions (10% within 4, 90% within 40),
  in 8 sessions in fast markets and 23 in calm ones.
- **Setup:** the same features, walk-forward, purge (60) and sizing map; out-of-sample span 1995+.

| Target | Skill vs base rate (logistic / LightGBM, millinats) | Sized book Sharpe (logistic / LightGBM) | Crash wins vs T1 |
|---|---|---|---|
| T1 fixed 20 days | +9 / **+77** | 0.55 / 0.56 | – |
| FP5 first passage ±5% | −195 / −70 | 0.56 / 0.54 | 2/9 |
| FPσ first passage ±1σ | −195 / −105 | 0.52 / 0.52 | 1/9 |
| (vote × live multipliers / 15% vol target, same span) | – | 0.63 / 0.72 | – |

- **The fixed horizon is not the problem, and the adaptive targets are worse.**
  - "Which way will it break first" is a *direction* question, and direction is the part that
    can't be predicted (E1, E5).
  - "How big will the swings be over the next month" is a *risk* question, and that part is
    predictable.
  - No adaptive target is preferred (Sharpe CIs straddle 0; 1–2 of 9 crashes).
- **The adaptivity you want is already in the policy.** The books re-decide every day, and the
  ones that respond directly to current conditions win: the vote at 0.63 and the vol target at
  0.72, against 0.52–0.56 for every supervised target.


## What this means for the regime architecture (updated 2026-11-25)

- **The regime describes risk, not direction.** That holds over 40 years (E1), intraday (E5) and
  inside the momentum module (E7-M). It belongs in each module's order policy, and the same state
  means different things to different modules:
  - **index exposure:** cut size in RISK_OFF. The live multipliers beat buy-and-hold over 40 years
    of crashes (E2).
  - **momentum entries:** do not cut. Entries made in RISK_OFF earned the most, and de-risking
    them was worse than random (E7-M).
- **The simplest strong risk overlay is a volatility target.**
  - A trained HMM rediscovers it and does not beat it (E9).
  - Supervised drawdown models lose to the hand vote (E9). Predicting risk is not predicting
    risk-adjusted return.
- **Keep the hand vote as the shared state; it is simple and auditable.**
  - Add a vol-target size channel beside it rather than replacing it with a model.
  - Next variant: the HMM's forecast state volatility as the vol-target input.
- **The factors people watch do matter, as risk barometers** (E3b).
  - At short horizons, price and vol already contain them.
  - At 60–120 days, credit and VIX beat price/vol models, mainly through the 2000s credit cycle.
  - Use them to *describe* the state and as long-horizon context, not as 20-day entry gates.
  - The yield curve hurt every equity target tested.
- **The intraday MarketState publisher should:**
  - refresh every 30 minutes;
  - carry calendar flags;
  - publish volatility and range, not a direction call (E5).
- **Forward evidence decides the rest** (E8 shadow, read about 2027-01-25):
  - momentum size-up in RISK_OFF;
  - the HMM and vol-target index books;
  - sector correlation, VIX, the dollar and GNS as watch items.
- **Earnings history (fixed 2026-11-25):**
  - `signals.events.earnings_calendar --refresh --union-history` keeps unfetched history.
  - Plain `--refresh` deletes rate-limited tickers and rows older than 40 quarters.
  - The weekly refresh now runs the union mode.
- **Answers to the 2026-11-25 evening questions:**
  - **Less history is worse** (E10). Keep all history and fewer inputs.
  - **Adaptive targets are worse than a fixed risk target** (E11). Adaptivity belongs in the daily
    re-decision, which the books already do.
  - **Predicting a VIX spike fails** (E12b). Responding to turbulence works (E12a precision 60–86%
    by decade; E2/E9 vol target).
  - **Fading spikes is carry with crash tails** (E12c). It is not a timing edge.
  - **"Good news sold" is real in this selloff** (E4, refreshed data). Tech beats underperformed
    SPY in November; GNS led the vote's flag. It is now an E8 shadow candidate.

## Reproduce

```
.venv/bin/python research/regime_experiments_2026-11-24/e0_fetch_long_history.py         # ~1 min, network
PYTHONPATH=. .venv/bin/python research/regime_experiments_2026-11-24/e1_state_futures.py  # ~20 s
PYTHONPATH=. .venv/bin/python research/regime_experiments_2026-11-24/e2_general_policy.py # ~10 s
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e3_factor_completeness.py   # ~3 min, 3 workers
E3_WINSOR=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e3_factor_completeness.py
PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e5_intraday_granularity.py  # ~2 min; fetches VIXY 5m once
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e3b_null_audit.py      # ~45 min, 5 workers
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e3c_controls.py        # ~15 min
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e9_regime_model.py     # ~6 min
PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e7_momentum_policy.py                    # ~30 s, needs stage3_results/
PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e4_good_news_sold.py                     # ~1.5 min
PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e8_shadow_log.py                         # nightly, after E0
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e10_less_history.py     # ~1 min
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e11_adaptive_horizon.py  # ~1 min
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e12_vix.py               # ~2 min; fetches VIXY/SVXY/SVIX once per day
```
Every script reads the newest `Data/research/regime_40y/panel_*.parquet` (these results:
`panel_20261124T2125`). Outputs go to `out/`. Seeds are fixed: 20261124 for E1–E5, 20261125 for
E3b, E4, E7-M and E9, and 20261126 for E3c.
