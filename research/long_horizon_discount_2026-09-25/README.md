# Long-horizon "discount + TMO pivot" feasibility (2026-09-25)

**Question.** Before building a 60-126 session (Roth) model: does the chart setup
"dips below the 200 SMA, TMO goes oversold, then turns up" earn excess return
over long holds, compared with matched controls?

**Script.** `01_setup_event_study.py` writes `01_results.txt` and `01_setup_uptrend_events.csv`.
- Universe: `shared_universe.csv` `type == "Stock"` (1,081 names), price >= $10, top 400 per date
  by trailing-60d median dollar volume. The ranking is point-in-time.
- Entry at the next session's open. Exit at the close H = 63/90/126 sessions later.
- Excess = the stock's return minus SPY's over the same window.
- One event per ticker per 20 sessions in each arm.
- Windows that contain a split-shaped gap are masked. A split-shaped gap is a price ratio
  <=0.7 or >=1.4 with dollar volume <2.5x its trailing median.
- CIs come from a bootstrap over signal months, because events cluster in time.
- TMO is a vectorised copy of `spy_intraday/.../custom_indicators.add_tmo` (14/5/3). Oversold is <= -9.

## Result: the setup has no measurable edge at long horizons

Excess vs SPY, arm minus all-days baseline, hold 90 sessions (all horizons are in `01_results.txt`):

| arm | n | Δ vs baseline [95% CI] |
|---|---|---|
| SETUP: dip<200 + oversold + TMO turn | 6,809 | −0.48% [−1.77, +0.77] |
| SETUP + rising 200 SMA (uptrend pullback) | 3,298 | +0.88% [−0.91, +2.65] |
| same discount, rising 200, **no** TMO turn (control) | 6,656 | +0.59% [−0.86, +2.01] |
| near 52w low + oversold + TMO turn | 3,686 | **−1.65% [−3.47, −0.05]** (−3.15% at 126d) |
| TMO oversold turn while **staying above** the 200 SMA | 2,716 | **+2.21% [+0.60, +3.87]** (+3.43% at 126d) |

- **TMO pivot adds ~0.3pp over simply "discounted in an uptrend."** That is not measurable here.
  The MDE is about 2-2.5pp per 90d, so this is "not measurable at this power", not proof of zero.
- **Buying near the 52-week low underperforms.** The deficit is significant and grows with the hold.
- **The one positive arm is a pullback that never broke the 200 SMA**, which is momentum's shape and
  not value's. It was found in a 7-arm × 3-horizon scan (multiple comparisons) and was not
  pre-registered, so it is a candidate to confirm and not a result.
- **The examples in the screenshots are survivors of hindsight.** Mega-caps in the uptrend-SETUP arm
  had 59 events (AMZN, GOOG/L, META, MSFT, NVDA, ORCL): mean +1.1% excess, median −0.7%.
  Over 2021-12..2022-01 the identical signal on META returned −34% twice.

## Biases, all of which flatter dip-buying (so the null is conservative)
- **Survivor-shaped bar cache and universe.** The universe is today's list of stocks, and the
  names that dipped and never recovered are mostly missing.
- **Volume field changes feed.** It is IEX-only until 2026Q3 and SIP afterwards (META median
  $ volume: ~$200M/day, then $10.6B). An absolute $-volume floor therefore selects a different
  universe in each era. This is why the study ranks per date.
- **No fundamentals were in the test.** The user's thesis is "good company + discount". This
  tested only the technical half.

## What exists for the fundamental half
- `signals/news/data/processed/ticker_earnings_calendar.parquet` has eps_estimate, reported_eps
  and surprise_pct per quarter. The history was wiped by the dedupe bug fixed 2026-09-25, so
  `earnings_calendar --refresh` must run first.
- The `signals/events/forward_guidance` guidance NLP covers only 20 mega-caps and 550 events
  (2024-01..2026-01), which is too narrow for a model.
- There is no point-in-time revenue/margin panel. SEC XBRL companyfacts (free, with a `filed`
  date = availability time) is the candidate source.

## Next testable variant
Test a quality × discount interaction. The quality filter is PIT revenue growth plus EPS-surprise
trend, with the latest quarter weighted highest. Compare quality+discount against quality-only and
discount-only at 63/126d, using this same harness and controls.

---

# Part 2: ML ranker + fundamentals (2026-09-25, same day)

**Data built** (fetchers under `signals/events/` and `scripts/research_data/`):
- Research bars: `Data/research/bars_1d_sip_adj`. 3,124 tickers, 2016+, SIP volume, split+div adjusted.
- SEC PIT fundamentals: 600,704 facts, 2,528 companies. Revenue coverage is ~72% of panel rows.
- Earnings calendar: restored, 101k reported quarters.
- 8-K Ex-99 press releases: fetching.

**Found:** `forward_guidance` "guidance" text is 8-K cover pages / 10-Q notes, not guidance.
**Found:** adjusted prices embed future reverse splits (FCEL 2019 reads ~$50 vs a real ~$0.20), so a
price floor or price feature leaks distress. Both were removed; the universe is the top 1,000 by dollar volume.

**Harness:** `02_build_panel.py` builds a weekly panel (509k rows, 1,820 tickers, 2016-12..2026-09).
`03_walkforward.py` trains yearly expanding folds 2019-2026, purged by H+5 sessions, with a LightGBM
rank-regression. `04_quality_gate.py` tests fundamentals used as a filter.

Top-20 excess over the liquid-universe mean, 6-month-block 95% CI (results in `03_results_h*.txt`, `04_results.txt`):

| arm | 126d | 63d | 126d CAGR / MDD (SPY 15.2%) |
|---|---|---|---|
| ML all features | +2.0% [−4.2, +8.1] | +0.5% [−3.1, +4.3] | 18.0% / −55.8% |
| ML technical only | +6.0% [+1.1, +11.3] | +1.3% [−2.1, +4.9] | 22.4% / −18.3% |
| ML fundamentals only | −0.5% [−5.2, +4.2] | −1.3% [−3.5, +0.9] | 12.8% / −52.4% |
| ML permuted-label null | +0.9% [−3.0, +5.3] | +0.4% [−2.1, +3.1] | 9.2% / −29.9% |
| **12-1 momentum (no ML)** | **+12.1% [+2.3, +22.2]** | **+5.9% [+0.5, +11.9]** | 28.2% / −37.6% |
| **fuzzy 200SMA(±5%) + TMO turn** | **+5.3% [+1.8, +9.0]** | **+2.6% [+0.7, +4.8]** | 13.4% / **−14.1%** |
| user's AMZN "compounder dip" | +1.4% [−3.3, +6.5] | +0.7% [−1.2, +2.7] | 17.0% / −34.4% |
| quality gate × momentum | +2.8% [−3.6, +9.4] | +2.5% [−0.8, +5.8] | 14.5% / −55.5% |
| quality gate × hand rule | +4.9% [+1.6, +8.3] | +2.5% [+0.9, +4.4] | 13.1% / −15.7% |

**Reading**
1. **ML did not beat simple rules.** Adding fundamentals made the ML ranker WORSE (drawdown −56%).
   Its top features are size (log_rev_ttm), 2-year drawdown and market-level features. Market-level
   features cannot rank names within a date. The full-universe IC is negative in every ML arm, so
   the tech-only top-20 edge is not trustworthy on its own.
2. **Fundamentals as defined (rev growth >10%, GM ≥40%, profitable/improving) added nothing,** whether
   as a filter or as model inputs, and they hurt momentum. Reported growth is priced in, and the gated
   set is the 2021-22 growth-crash cohort (worst year −11.7%).
3. **12-1 momentum is the strongest ranker, with large drawdowns.** The fuzzy 200-SMA+TMO rule is the
   most CONSISTENT arm: positive at both horizons, worst year −2.5%, MDD −14%. Its 2019-2020 years are
   out-of-sample relative to study 01. Its top-20 uses a momentum tiebreak, so it is effectively
   "pullback-holding-the-200, highest momentum first".
4. **The AMZN-shape rule is not a selection edge** vs the liquid universe. It matches the market at
   lower drawdown on 63d.

**Caveats:** survivor-shaped universe (today's tickers); 8 arms × 2 horizons (multiplicity); ~7 test
years, and 2021-22 dominates the losses; label windows overlap (block CI); portfolio CAGR is a single
rebalance phase.

**Next testable variants**
- (a) Momentum top decile × hand-rule entry: buy leaders on a pullback that holds the 200.
- (b) LambdaRank per date with market-level features removed and fewer features.
- (c) Guidance/press-release text features once the Ex-99 fetch completes.
- (d) Point-in-time delisted names for survivorship.

---

# Part 3: does anything beat SPY buy-and-hold? (2026-09-29/30)

Questions: (1) Does anything beat SPY buy-and-hold, and at which rebalance frequency
(monthly / quarterly / semiannual / annual)? (2) Part 2 variants (a) and (b). (3) Does the
quarterly earnings report predict the next quarter, for example "good report, but the stock tanked"?
(4) What did future 1- and 3-year winners look like beforehand?

## Corrections to Part 2 (read these first)
- **Part 2 drawdowns were understated.** Scripts 03/04 marked portfolios only at period ends, on one rebalance
  phase. 05 marks daily across every phase. Measured that way, the fuzzy 200-SMA + TMO rule's max drawdown is
  **−43% to −56%, not −14%**, and its Sharpe (0.66-0.77) is BELOW SPY's (0.87-0.92). The Part 2 claim that the
  rule was the "most consistent" is retracted. Momentum's drawdown is −50% to −71%, not −38%.
- **SEC fundamentals were delayed for ~18% of company-quarters.** `sec_fundamentals` chose the tag first
  and the filing second. So a quarter was dated to the first filing that used the preferred tag. NVDA's
  2017-18 10-Q revenue (tagged `Revenues`) was dated to the 2019 10-K, the first filing to use the
  ASC 606 tag. This is not lookahead, because the data arrived late rather than early, but it biased
  the fundamentals tests toward null. Fixed: the earliest filing now wins across tags
  (regression test `test_earliest_filing_wins_across_tags`). Quarters with a metric more than 120 days late fell
  from 17.6% to 8.5%. The rebuilt panel has identical rows, technical features and labels, so only the
  fundamentals columns changed (rev_yoy nulls 27%→21%). The v1 files are kept:
  `sec_quarterly_facts.v1_priority_tag.parquet`, `panel_weekly.v1.parquet`, and `*_v1.txt` results.
  On v2, 04 still shows no quality-gate edge (quality × momentum +1.6% [−4.8, +8.2] @126d; it was +2.8% on v1).
  The v2 re-run of 03 does not change the verdict. Fundamentals-only top-20 is −0.1% [−6.0, +5.8] @126d and
  +0.5% [−2.9, +4.1] @63d; all-features is +2.5% / +1.1%. Every CI includes zero, the permuted control sits at
  +1.9% / +0.4%, and 12-1 momentum stays at +12.1% / +5.9%. The tech-only arm is bit-identical to v1.

## 05: rotation backtest vs SPY (`05_rotation_backtest.py`, `05_results.txt`, chart `09_equity_monthly.png`)
Decisions 2019-01 to 2026-09 run on a weekly panel. Entry is the next open; top 20 equal weight; 20 bp round trip.
Marked daily. Every phase offset runs, and the median phase is reported. The rule arms are not fitted.
`lean_ml` uses walk-forward OOF scores. Arms were pre-registered in the script docstring. `high_vol`,
`mom_top300` and `core_satellite` were added after the smoke run and are labeled as such.

Monthly rebalance (quarterly is in brackets where it matters):

| arm | CAGR | active vs SPY /yr [95% CI] | Sharpe | MDD med / worst | beta | alpha/yr |
|---|---|---|---|---|---|---|
| SPY buy & hold | 16.9% | — | 0.92 | −34% | 1.00 | — |
| QQQ buy & hold | 22.8% | +6.0% [−0.2, +12.3] | 0.98 | −35% | 1.18 | +3.0% |
| RSP (equal-weight S&P) | 12.6% | −3.6% | 0.70 | −39% | 0.96 | −2.9% |
| **12-1 momentum, liquid-1000** | **38.3%** (q 31.5%) | **+27.1% [+6.2, +49.6]** | 0.90 | −50% / −55% | 1.57 | +17.4% |
| **12-1 momentum, top-300 liquid** | **44.0%** (q 35.6%) | **+29.6% [+9.3, +49.2]** | **1.03** | −52% / −58% | 1.51 | +20.6% |
| **70% SPY + 30% momentum** | **24.8%** (q 22.1%) | **+7.9% [+1.5, +14.9]** | **1.01** | **−35% / −35%** | 1.17 | +5.0% |
| momentum + SPY-200SMA cash filter | 18.9% | +10.2% [−8.1, +31.8] | 0.62 | −56% / −57% | 0.96 | |
| high-vol top 20 (control) | 23.9% | +24.5% [−2.0, +55.5] | 0.65 | −75% / −79% | 2.01 | +5.9% |
| fuzzy 200-SMA + TMO rule | 17.4% | +4.1% [−7.4, +16.1] | 0.66 | −46% / −56% | 1.17 | +1.1% |
| **(a) leader × pullback (pre-registered primary)** | 16.4% | +3.5% [−3.9, +11.6] | 0.67 | −38% / −43% | 1.15 | −0.9% |
| leader × dip, no TMO | 29.4% | +19.7% [+3.2, +38.7] | 0.79 | −45% / −51% | 1.45 | +10.6% |
| lean ML (06 OOF) | 22.3% | +15.1% [−11.6, +40.3] | 0.65 | −77% / −79% | 1.71 | +3.1% |
| lean ML, permuted labels (control) | 6.0% | −3.5% | 0.34 | −58% / −62% | 1.36 | |

Momentum CAGR by rebalance frequency, monthly / quarterly / semiannual / annual:
liquid-1000 38.3 / 31.5 / 27.9 / 26.7; top-300 44.0 / 35.6 / 33.2 / 24.7; 70/30 blend 24.8 / 22.1 / 20.8 / 20.6.

**Reading**
1. **Only momentum-based arms beat SPY with a CI that excludes zero.** The returns come mostly from
   beta and concentration. On the liquid-1000 the Sharpe equals SPY's. Only the top-300 version (1.03) and
   the 70/30 blend (1.01) improve return per unit of risk. The chart shows the price: liquid-1000 momentum
   sat 40-50% below its peak from early 2021 to early 2024.
2. **Faster rebalancing is better.** Monthly beats quarterly, which beats semiannual, which beats annual.
   The signal decays in weeks, so the quarterly earnings cadence is not the natural clock for it.
3. **(a) failed.** Adding a pullback/TMO entry to leaders gave SPY-like returns at a lower Sharpe. The MDE is
   ~8%/yr, so this is "not measurable at that power", but the point estimate is also far below plain momentum.
   Buying the leaders beat waiting to buy them on a dip.
4. **The trend filter hurt.** Momentum's 2021-22 crash happened while SPY was still above its 200 SMA, and
   the filter sat out the 2020 rebound.
5. **Momentum is not just a volatility tilt.** The high-vol control reached Sharpe 0.65 and MDD −75%, against
   momentum's 0.90 / −50%.
6. **Concentration.** 2020 (+158% on phase 0 monthly) dominates. Excluding 2020, phase-0 momentum compounds
   ~27%/yr vs SPY's ~17%. The picks are thematic bursts: 2021-02 held MARA/PLUG/NIO/GME, and today it holds
   SNDK/MU/WDC/STX/LITE/AMD (a single AI-hardware theme).
7. **Survivorship is unquantified.** The universe is today's tickers, so the 2020-21 bubble names that later
   delisted are missing from the momentum top 20. That flatters momentum. The equal-weight liquid-1000 is only
   ~1.2%/yr above RSP, and the top-300 version (fewer delistings) is stronger. Both argue against survivorship
   explaining everything. But the high-vol control's +6% alpha contradicts the long-run low-vol anomaly,
   which suggests high-vol names are flattered. 2019-26 was also an exceptional momentum/megacap era.
   Long-run long-only momentum premia are a few %/yr, not +17-20%.

## 06: lean ranker, variant (b) (`06_lean_ranker.py`, `06_results_h*.txt`)
Market-level features were removed, the 14 stock features were rank-normalized within date, and three arms were run:
LightGBM lambdarank, regression, and permuted-label lambdarank.

| arm | 126d top20 xs [CI] | IC | top-20 vol pct | 63d top20 xs [CI] |
|---|---|---|---|---|
| lambdarank | +7.5% [−2.3, +18.1] | −0.019 | 0.93 | +3.4% [−1.8, +8.6] |
| regression | +4.3% [+0.0, +8.6] | −0.001 | 0.69 | +2.5% [−0.1, +5.2] |
| permuted control | −0.4% [−2.7, +2.0] | +0.008 | 0.56 | −1.1% [−2.3, +0.2] |
| 12-1 momentum | +12.1% [+2.3, +22.2] | +0.033 | 0.88 | +5.9% [+0.5, +11.9] |

Lambdarank learned "buy the highest vol" (rv_63 carries 44% of the gain, IC is negative, and the portfolio MDD is −77%).
Regression is marginal. Neither beats momentum. On this panel and sample, the class "GBM on technical features" is
exhausted. Another attempt needs new information (v2 fundamentals, press-release text) or a vol-neutral label.

## 07: earnings beat/miss × price reaction (`07_earnings_reaction.py`, `07_results.txt`)
35,188 reported quarters, 1,551 liquid tickers, 2017-2026. Entry is the open 2 sessions after the report.
The number shown is excess vs SPY minus the same-month all-event mean, with event-month bootstrap CIs.

| group | share | 63d [CI] | t | 126d [CI] | t |
|---|---|---|---|---|---|
| beat + stock fell ("good report, tanked") | 20% | +0.61% [−0.08, +1.33] | +1.7 | +1.10% [+0.04, +2.15] | +2.0 |
| beat + stock rose | 28% | +0.34% [−0.12, +0.78] | +1.5 | +0.50% [−0.05, +1.10] | +1.7 |
| **miss + market shrugged (flat)** | 5% | **−1.49% [−2.25, −0.73]** | **−3.9** | **−2.77% [−4.00, −1.51]** | **−4.3** |
| inline + stock fell | 6% | −0.33% | −0.8 | −1.64% [−2.65, −0.59] | −3.1 |

- "Good report but it tanked" does not reliably recover. The +0.6-1.1% sits at t≈2 among 27 cells, 2020 carries
  it (+3.3%), and 2024 went the other way (−2.0%). Reaction quintiles are U-shaped (both extremes beat the
  middle), which is the signature of a volatility tilt, not a reversal.
- No long-side post-earnings drift survives in liquid names. The market prices the report within the
  reaction window, and it priced expectations before the report.
- The one strong effect is on the downside. A miss that the market shrugged off keeps underperforming
  (t −3.9 / −4.3), so it is an exclusion filter, not a buy signal.
- Caveat: Yahoo's consensus is not guaranteed to be point-in-time. The ±2% bands limit misclassification.

## 08: what future winners looked like beforehand (`08_winner_traits.py`, `08_results.txt`)
Liquid-1000 at each quarter start. Win/loss lift = P(top/bottom 5% forward return | trait) ÷ base rate.

| trait (1y horizon, v2 fundamentals) | win lift | loss lift |
|---|---|---|
| momentum top decile | 2.26 | 1.90 |
| revenue growth > 30% | 1.94 | 1.76 |
| high volatility | 2.58 | 2.68 |
| gross margin ≥ 50% | 1.25 | 1.07 |
| EPS beats avg > 5% (4q) | 1.15 | 0.97 |
| profitable | 0.72 | 0.69 |
| growth leader (rev>20%, GM≥50%, mom top 20%) | 2.62 | 1.69 |

- **Most "what winners share" traits are volatility.** They raise the chance of a big loss almost as much
  as a big win. Only gross margin, beat streaks and the growth-leader combination are asymmetric, and the combination
  covers 2.5% of names and swings 0.65-5.3× by year.
- **NVDA:** its best entry (2022-10, +1,398% over 3y) had almost no winner traits. It was near the 52w low, high vol,
  with +3% revenue growth and a 43% gross margin. In 2018-04 it had every trait and fell 11% the next year.
- **Multi-year ML is data-limited, not method-limited.** 2017-2026 contains ~3 non-overlapping 3-year windows. Even
  30 years of data would give only ~10, which is too few to validate a model at that horizon.

## Next testable variants (roadmap order)
1. **Survivorship.** Add point-in-time delisted names (Alpaca inactive assets + their bars), then re-run 05's momentum
   and 70/30 arms. This must happen before any capital or paper sleeve relies on momentum's numbers.
2. **Pre-register the miss+flat exclusion** on the momentum and 70/30 arms (drop names with a shrugged-off miss in the last 63 sessions).
3. **Guidance text (Ex-99)** once the fetch completes. Test tone/guidance-raise features against 07's reaction groups.
4. **Paper shadow of 70/30 monthly** in its own ledger, only after step 1 holds.

## 10: the year's top-10 winners in hindsight (`10_top10_winners.py`, `10_results.txt`, 2026-10-03)
Liquid-1000 at each prior year-end, first open to last close, 2019-2025.

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| perfect-foresight top 10 | +212% | +605% | +263% | +104% | +404% | +298% | +277% |
| SPY | +31% | +18% | +28% | −18% | +26% | +26% | +17% |
| median liquid stock | +30% | +10% | +20% | −19% | +13% | +10% | +8% |
| % of stocks beating SPY | 47% | 40% | 40% | 49% | 33% | 30% | 37% |
| P(10 random stocks beat SPY) | 45% | 49% | 29% | 53% | 32% | 16% | 42% |
| winners in Jan-1 momentum top 20 | 3/10 | 3/10 | 1/10 | 1/10 | 1/10 | 6/10 | 0/10 |
| winners held by monthly momentum at some point | 10/10 | 8/10 | 6/10 | 8/10 | 9/10 | 9/10 | 7/10 |
| avg share of the winner's run captured | 25% | 43% | 4% | 24% | 17% | 29% | −2% |

- Most stocks lose to the index in a typical year. Returns are skewed: a few huge winners lift the index,
  and the index owns all of them. That is why picking is hard.
- The eventual winners were NOT identifiable on Jan 1. Only 15/70 were in the momentum top 20. In 2023, 8 of
  10 were the prior year's WORST names (CVNA, MARA, COIN, AFRM rebounding from crashes). Nearly all were
  high-volatility, and so are the biggest losers (08: loss lift 2.68).
- Momentum works by riding the winners mid-run, not by predicting them. It held 57/70 at some point and
  captured ~20% of their run on average. Sometimes it buys the top (GME 2021 −37%, IREN 2025 −51%).
- The universe is survivor-shaped, so the median and random-pick figures are flattered. In reality the
  index's edge over stock-picking is larger than shown here.

---

# Part 4: the Part 3 next steps (2026-10-03)

## Breadth: top-20 vs top-50 vs top-100 momentum (`05 --tag breadth`, `05_results_breadth.txt`)
Today's-tickers universe, monthly rebalance (other frequencies are in the results file):

| arm | CAGR [phase min, max] | Sharpe (SPY 0.92) | MDD med / worst | alpha/yr |
|---|---|---|---|---|
| momentum top 20 | 38.3% [34.3, 43.1] | 0.90 | −50% / −55% | +17.4% |
| momentum top 50 | 34.8% [32.3, 35.4] | 0.93 | −43% / −48% | +13.0% |
| momentum top 100 | 26.4% [24.3, 29.4] | 0.85 | −40% / −41% | +6.5% |
| 70% SPY + 30% top 20 | 24.8% [23.7, 26.2] | 1.01 | −35% | +5.0% |
| 70% SPY + 30% top 50 | 23.2% [22.3, 23.4] | 1.01 | −35% | +3.8% |

More names do NOT raise return per unit of risk. The edge sits in the extreme tail and the names move
together as one theme, so adding names dilutes alpha about as fast as it cuts risk. Top 50 is the better-behaved
version: the drawdown is 7 points smaller and the result depends far less on the start date. Top 100 is
too diluted. The "many small independent bets" route that quant funds use needs independent signals or a long-short
book, not more names from one factor.

## Miss + flat exclusion (pre-registered in Part 3; `05 --arms ... mom_xmiss core_sat_xmiss`, `05_results_xmiss.txt`)
The rule skips any name whose last earnings report was a miss the market shrugged off (07) and whose reaction
window closed within the last 91 days. It flags 4.3% of universe rows and 3.3% of momentum picks.

| paired difference vs the unfiltered arm | monthly | quarterly |
|---|---|---|
| momentum top 20 | +1.83%/yr [+0.25, +3.36] | +0.61%/yr [−0.25, +1.18] |
| 70/30 blend | +0.56%/yr [+0.08, +1.04] | +0.22%/yr [−0.06, +0.41] |

The direction is as predicted, and the monthly CI excludes zero. The effect is small because few picks are flagged.
It is in-sample: 07 measured the effect on these same years. Keep it as an optional refinement and do not treat it
as a result.

## Guidance language, variant (c) (`11_guidance_text.py`, `11_results.txt`)
70,083 real press releases (8-K Ex-99) with exact SEC acceptance times. Regex labels: raised 12.2%, reaffirmed 7.0%,
lowered 2.6%, mixed 0.7%, none 77.5%. There are 25,951 liquid-1000 events, 2017-2026.
**The sanity check passed:** the announcement reaction is +1.48% for raised and −1.35% for lowered, so the labels do measure guidance.

| group | n | 63d adj [CI] | t | 126d adj [CI] | t |
|---|---|---|---|---|---|
| raised (all) | 4,021 | +0.15% [−0.52, +0.80] | +0.4 | +0.16% [−0.89, +1.11] | +0.3 |
| lowered (all) | 851 | +0.23% [−0.74, +1.38] | +0.4 | +1.03% [−0.87, +2.95] | +1.1 |
| **raised + stock fell** (the user's case) | 1,130 | +0.30% [−0.87, +1.50] | +0.5 | −0.68% [−2.24, +0.91] | −0.8 |
| raised + stock rose | 1,725 | +0.19% [−0.73, +1.13] | +0.4 | +0.30% [−0.95, +1.61] | +0.5 |

- Guidance direction is priced inside the two-session announcement window. No drift is left to trade afterwards, in
  either direction. "Good guidance but it tanked" does not recover (MDE ≈ 1.2% per quarter, so smaller effects
  are not measurable here).
- The regex has known noise ("Decreases 2024 Adjusted Operating Expense Guidance" is labeled lowered). That lowers
  power; it does not create a false null of this size, given the 2.8-point reaction spread.
- This hypothesis class (guidance DIRECTION → multi-month drift in liquid names) is exhausted at this power. The next
  variant would need the SIZE of the guidance surprise against consensus, which requires point-in-time estimates
  that we do not have.

## Survivorship: the point-in-time universe (`05 --pit`, `05_results_pit.txt`, `05_results_pit_delist0.txt`)
**How it was built** (`scripts/research_data/probe_symbol_history.py`, `build_pit_universe.py`, `02 --pit`):
- Alpaca's asset list omits most delisted names, but its bar endpoint still serves them. So every 1-4 letter symbol
  was probed with monthly bars: 23,091 symbols have traded since 2016.
- Daily SIP adjusted bars were fetched for every non-fund symbol whose best month averaged ≥ $5M/day (5,506 new symbols).
- Result: **8,158 stock securities vs 3,093 in today's cache; 3,426 stopped trading.** The weekly liquid-1000 is ranked
  among all of them. 2,276 securities were in the top 1,000 at some point (1,820 before). In 2019, 18% of each week's
  PIT top 1,000 was absent from the survivor top 1,000; in 2026 it is 2%.
- Data traps found and handled, each of which would have produced fake returns:
  - Alpaca keeps renamed companies under BOTH symbols (UTX and RTX). 834 aliases were dropped by matching daily trade counts.
  - It pads a dead ticker with zero-volume bars, which glues two companies together (FI = Frank's International at $3,
    then Fiserv at $130). Zero-volume bars are dropped and lives are split at gaps > 45 days.
  - New equity after bankruptcy and spin-off handoffs trade under the old ticker (OAS +19,893% in a day, BHVN −94.5%).
    Lives are split at one-day moves > 10x or < 0.12x.
  - **58 symbols in today's cache glued two companies/listings** (DOW, SE, MBLY, BTU, VAL, LB, ...). That contaminated
    Part 3's momentum signal for those names; the PIT bars fix it.
- Checks: every non-survivor name that entered the momentum top 20 was reviewed (125 names, no funds). No top-20 pick was
  ever held through a fake jump/drop. The equal-weight liquid-1000 now returns 12.8%/yr, next to RSP's 12.6% (it was 13.8%).

**Result, monthly rebalance, 2019-01 to 2026-09 (SPY 16.9%, Sharpe 0.92, MDD −34%):**

| arm | survivor universe (Part 3) | point-in-time | PIT active/yr [95% CI] | PIT Sharpe | PIT MDD med / worst |
|---|---|---|---|---|---|
| momentum top 20, liquid-1000 | 38.3% | **30.1%** | +21.0% [−0.1, +40.2] | 0.78 | −58% / −64% |
| momentum top 20, top-300 liquid | 44.0% | **39.7%** | +26.7% [+5.8, +46.0] | 0.95 | −53% / −57% |
| momentum top 50 | 34.8% | 29.1% | +16.8% [−1.4, +34.7] | 0.82 | −51% / −56% |
| 70% SPY + 30% momentum (liquid-1000) | 24.8% | **22.6%** | +6.1% [−0.2, +12.0] | 0.94 | −35% / −36% |
| 70% SPY + 30% momentum (top-300) † | — | **25.0%** | +7.9% [+1.7, +13.8] | 1.04 | −34% / −34% |
| high-vol top 20 (control) | 23.9% | **1.1%** | +1.8% [−25.0, +28.0] | 0.32 | −87% |
| fuzzy 200-SMA + TMO rule | 17.4% | 13.4% | +2.0% [−10.0, +14.3] | 0.55 | −51% / −61% |
| equal-weight liquid-1000 | 13.8% | 12.8% | −2.9% | 0.66 | −40% |

† Added after the first PIT run, so it is a post-hoc blend of two existing arms. Quarterly PIT: momentum 24.9%, top-300 31.7%,
70/30 20.3%, 70/30 top-300 22.1% (all CIs include zero at quarterly).

**Reading**
1. **Survivorship was worth ~8 points of CAGR for liquid-1000 momentum** (38% → 30%) and ~4 points for the top-300
   version. 2020 fell from +158% to +83%. The high-vol control collapsed from +24%/yr to +1%/yr, which confirms the diagnosis.
2. **The pre-specified momentum arm no longer clearly beats SPY.** It still returned more (30% vs 17%), but its CI
   touches zero, its Sharpe is below SPY's (0.78 vs 0.92), and its drawdown is −58% to −64%.
3. **Momentum among the 300 most liquid names survives**: +26.7%/yr [+5.8, +46.0], Sharpe 0.95. That arm was added
   in Part 3 because delistings are rare there, and that prediction held. It was not the primary arm, and one 7.7-year
   window decides it.
4. **The delisting terminal value does not matter.** Sending every distressed delisting to zero leaves the momentum
   arms unchanged to the decimal. Momentum never held a name at a distressed delisting, because those names had already
   crashed out of the leaders. The survivorship cost came from the crashes becoming visible, not from the delisting itself.
5. Remaining limits: a stock that moved to OTC keeps only its exchange history; 43 merger pairs are double counted on
   their overlap; smaller adjustment errors remain (NVS 2019-04-09); fund screening of unclassified symbols is manual.

## Next testable variants (roadmap order)
1. **Forward shadow** of the 70/30 blends (`12_shadow_ledger.py`): the only evidence that is not this one backtest window.
2. **A longer window.** Alpaca starts in 2016. A pre-2016 momentum check needs another vendor (CRSP-like, delisting-aware).
3. Guidance-surprise SIZE vs point-in-time consensus (needs an estimates source).

## Forward shadow (`12_shadow_ledger.py`, `shadow/ledger.csv`)
Research only: no orders and no account access. Arms were fixed on 2026-10-03, before any forward data existed:
`core_sat_300` (primary: 70% SPY + 30% top-20 momentum among the 300 most liquid), `core_sat_1000`, the two sleeves
alone, and SPY. First decision: **2026-10-02** (entry at the 2026-10-05 open). The top-300 sleeve holds
SNDK AXTI MRNA MU LITE AAOI WDC DELL STX AMD INTC MRVL TER BE COHR CIEN VLO HUT TWLO HPE, which is mostly one
AI-hardware theme. **Updated 2026-10-06:** the code moved to `scripts/shadow/momentum_shadow_ledger.py` (`12_shadow_ledger.py` is a
shim) and runs as stage 6 of `scripts/weekly_refresh.sh`. See Part 7 for the schedule.

---

# Part 5: can ML improve momentum inside the liquid-300? (2026-10-04)

**Question (user):** the surviving edge is "current momentum leaders, restricted to the few hundred most liquid
names". Can ML do better inside that universe?

## 13: ML vs plain momentum, liquid-300, point-in-time (`13_ml_top300.py`, `13_results*.txt`)
Walk-forward 2019-2026 with a 21-session label, to match the monthly rebalance. Inputs are 14 technical features as
within-date percentiles. Arms were fixed before the first run. Each holds the top 20 within the liquid-300 and is run
through 05's simulator. The verdict column is the PAIRED active return against plain `mom_top300`
(39.7% CAGR, Sharpe 0.95, MDD −53%).

| arm | rank label (pre-registered) | raw-return label (exploratory) | top-decile classifier (exploratory) |
|---|---|---|---|
| LightGBM trained on liquid-300 | 19.7%, **−20.1%/yr [−31.1, −8.4]** | 21.1%, −16.0% [−26.9, −5.3] | 24.2%, −10.7% [−24.1, +3.3] |
| LightGBM trained on liquid-1000 | 22.3%, −18.9% [−32.6, −5.1] | 23.7%, −13.5% [−23.3, −4.3] | 27.2%, −5.1% [−14.8, +6.4] |
| ridge, liquid-300 | 10.8%, −28.7% [−44.7, −13.9] | 10.9%, −25.9% [−39.8, −13.7] | 25.0%, −7.7% [−21.0, +4.5] |
| top-60 momentum leaders re-ranked by ML | 24.6%, −16.4% [−27.3, −4.6] | 31.5%, −7.9% [−16.0, −1.2] | 34.8%, −2.0% [−6.4, +2.2] |
| permuted-label control | 12.3% | 13.8% | 20.6% |

- **No ML arm beats plain momentum: 12 arms across 3 labels.** On the pre-registered rank label every arm is
  significantly WORSE. The ridge model is no better than the permuted control and puts a negative weight on 12-1 momentum.
- The top-decile classifier comes closest, but it does so by buying volatility: rv_63 carries 52% of the gain, its picks
  sit at the 92nd-94th vol percentile, and its drawdowns are −72% to −80%.
- Re-ranking the leaders with ML loses to simply taking the 20 strongest.

## 14: why (`14_momentum_profile.py`, `14_results.txt`)
Next-21-session return minus the liquid-300 mean, by momentum rank, 399 weekly dates:

| momentum rank | mean excess / month [CI] | median | P(next month in top decile) |
|---|---|---|---|
| 1-5 | **+4.01% [+1.68, +6.46]** | +0.07% | 29% |
| 6-10 | +2.14% [+0.38, +3.86] | +0.29% | 26% |
| 11-20 | +1.41% [+0.05, +2.65] | +0.22% | 20% |
| 21-40 | +0.33% [−0.45, +1.14] | −0.34% | 14% |
| 41-300 | −0.3% to +0.2% | −0.4% | 7-11% |

The edge is a right-tail effect in the extreme top ranks. The MEDIAN leader does not beat the group (hit rate
50%), but it is ~3x as likely to have a top-decile month. Across the whole liquid-300, momentum's IC is only +0.011. So a
model fitted to the cross-section has almost nothing to learn, and a model fitted to "big month" learns volatility,
which predicts big losses equally well. The momentum rank already IS the tail signal.

## 15: is 300 / 20 a lucky spot? (`15_liquidity_grid.py`, `15_results.txt`)
Top-N momentum among the L most liquid, monthly, point-in-time. CAGR / Sharpe / median MDD:

| L \ N | 5 | 10 | 20 | 30 |
|---|---|---|---|---|
| 100 | 33.4% / 0.79 / −78% | 33.1% / 0.84 / −63% | 26.6% / 0.80 / −49% | 23.2% / 0.79 / −45% |
| 200 | 46.0% / 0.92 / −71% | 40.0% / 0.91 / −66% | 33.5% / 0.88 / −58% | 27.3% / 0.81 / −49% |
| 300 | 52.8% / 0.98 / −73% | 43.3% / 0.94 / −66% | **39.7% / 0.95 / −53%** | 34.2% / 0.92 / −46% |
| 500 | 62.5% / 1.06 / −64% | 46.0% / 0.96 / −65% | 39.2% / 0.92 / −55% | 35.9% / 0.92 / −52% |
| 1000 | 26.6% / 0.68 / −72% | 33.0% / 0.79 / −61% | 30.1% / 0.78 / −58% | 31.3% / 0.82 / −57% |

**A plateau, not a spike.** L = 300-500 is flat. L = 100 is too narrow (few leaders to choose from) and L = 1000
admits the junk that crashes. The active-return CI excludes zero for L = 300 and 500 at every N. Fewer names raise the
return and deepen the drawdown at a roughly constant Sharpe, so N is a risk dial, not an edge.

## Verdict and next testable variant
The hypothesis class "ML on price-derived features to select inside the liquid universe" is exhausted at this sample:
12 arms, 3 labels, permuted controls, none above plain momentum. Selection is not where the remaining gain is.
1. **Size the sleeve, do not re-pick it.** The problem with momentum is its −53% drawdown. Test a volatility-managed
   sleeve (scale the momentum weight by target vol ÷ its trailing realized vol; the rest in SPY) against the fixed 70/30.
2. **Only new information can improve selection:** the miss+flat exclusion (Part 4, small but real), and signals not
   derived from price (options/dealer positioning, news/themes) once they have enough history. Track them in the shadow.

---

# Part 6: risk overlays from the existing system (2026-10-05)

**Question (user):** the sleeve works; risk is the problem. The system already has a lot of risk management. What transfers?

## 16: overlays on the top-300 sleeve (`16_risk_overlays.py`, `16_results.txt`)
Point-in-time, monthly, 4 phases. It reuses `portfolio_lab/covariance.ledoit_wolf_asof`, `sizing.correlation_penalty`,
`sizing.portfolio_vol_scale`, the 12% position cap, and the two stop shapes of `core/live_4h_exec`. Parameters were fixed
before the run. The overlay simulator is asserted equal to 05's on the no-overlay arm.

Daily-return mean / vol: **SPY +17.8% / 18.9%; sleeve +44.9% / 45.8%; correlation 0.62.** The sleeve earned 2.5x the
return at 2.4x the volatility, which is why the Sharpe ratios are nearly equal (0.95 vs 0.92).

| overlay (sleeve alone) | CAGR | Sharpe | MDD med / worst | vs plain sleeve /yr [95% CI] |
|---|---|---|---|---|
| none (plain sleeve) | 39.7% | 0.95 | −53% / −57% | — |
| correlation-penalised weights, 12% cap | 40.1% | 0.96 | −52% / −57% | +0.2% [−1.0, +1.1] |
| vol target 30% (78% invested on average) | 26.8% | 0.89 | −43% / −46% | −14.8% [−23.0, −6.6] |
| vol target 20% (56% invested) | 17.9% | 0.83 | −34% / −35% | −24.8% [−37.5, −12.2] |
| stop 10% from entry (36% of positions stopped) | 27.8% | 0.89 | −48% / −61% | −14.5% [−22.6, −7.1] |
| stop 20% from entry (15% stopped) | 33.5% | 0.92 | −55% / −63% | −6.2% [−12.3, −0.9] |
| trailing stop 25% from peak (15% stopped) | 33.6% | 0.91 | −54% / −58% | −6.3% [−11.0, −1.6] |
| cash while the sleeve is ≥ 25% below its peak | 19.7% | 0.67 | −54% / −57% | −19.9% [−29.8, −10.3] |

| overlay (70/30 blend) | CAGR | Sharpe | MDD | vs fixed 70/30 /yr [95% CI] |
|---|---|---|---|---|
| fixed 70/30 | 25.0% | 1.04 | −34% | — |
| correlation weights | 25.0% | 1.04 | −34% | +0.1% [−0.3, +0.4] |
| vol-managed sleeve share (10-50%) | 22.3% | 0.99 | −34% | −2.6% [−4.4, −0.5] |
| stop 20% on sleeve names | 23.7% | 1.03 | −35% | −1.2% [−2.4, −0.1] |
| all three together | 21.7% | 0.99 | −35% | −3.3% [−5.5, −1.1] |

- **No overlay helps, and stops, vol scaling and the drawdown breaker measurably hurt.** The paired CIs are tight and
  negative, so this is a real cost, not an underpowered null.
- **Stops do not even cut the drawdown.** Momentum's payoff is a right tail (Part 5): leaders are volatile, so a stop
  sells them before the big month and the rule re-buys them at the next rebalance.
- **The drawdown breaker is the worst rule.** It sat out the recoveries (2023: 0%, 2024: −3% vs +71%).
- **Vol scaling only slides down the same line.** Less sleeve means less return at an equal-or-lower Sharpe. In this
  window the high-vol periods (2020, 2025-26) were also the best ones for long-only momentum.
- Correlation weighting is neutral, because the 20 leaders are one cluster anyway.
- Caveats: cash earns 0% here (T-bills would add ~1%/yr to the partly-in-cash arms, not enough to flip any sign);
  stops are close-based daily; one 7.7-year window.

## The control that works: how much goes in the sleeve
Plain sleeve at X% of the account, the rest in SPY (added after the overlays failed; same two assets, different mix):

| sleeve share | 0% (SPY) | 10% | 20% | 30% | 40% | 50% | 70% | 100% |
|---|---|---|---|---|---|---|---|---|
| CAGR | 16.9% | 19.7% | 22.4% | 25.0% | 27.5% | 29.9% | 34.3% | 39.7% |
| Sharpe | 0.92 | 0.99 | 1.03 | 1.04 | 1.04 | 1.03 | 1.00 | 0.95 |
| MDD median / worst phase | −34% | −34% | −34% | −34% | −34% / −34% | −34% / −35% | −40% / −44% | −53% / −57% |

Up to ~50% in the sleeve the worst drawdown stays at SPY's −34%: the sleeve's own crash (2021-22) and SPY's (2020) fall
in different years. Sharpe is flat from 20% to 50%. Above 50% the sleeve's crash takes over.

## Verdict and next testable variant
The swing system's risk controls (stops, vol targeting, breakers) do not transfer to a monthly right-tail strategy.
The risk control for this sleeve is the allocation split, plus the per-name cap that equal weighting already gives
(5% of the sleeve = 1.5% of the account at a 30% share). The hypothesis class "per-position and vol-based overlays on
the sleeve" is exhausted on this window. What remains is evidence, not rules: the forward shadow (Part 4) and a
pre-2016 window from a delisting-aware vendor.

---

# Part 7: the last 12 months, and the shadow on the weekly schedule (2026-10-06)

## 17: the last 12 months, month by month (`17_last_12_months.py`, `17_results.txt`, `17_last_12_months.png`)
A BACKTEST replay on the point-in-time universe, on the live ledger's own 4-week grid: 13 decisions from 2025-10-03,
entries 2025-10-06 to 2026-09-08, marked to 2026-09-24, after 20 bp round-trip costs.

| | return | max drawdown | worst day |
|---|---|---|---|
| momentum sleeve, rotated every 4 weeks | **+32.4%** | −40.1% | −11.1% |
| month-0 leaders bought once and held | −15.1% | −47.6% | −9.8% |
| 50% SPY + 50% sleeve | +27.1% | −23.1% | −7.0% |
| 70% SPY + 30% sleeve | +23.0% | −15.7% | −5.3% |
| SPY buy & hold | +15.4% | −8.9% | −2.7% |

- **The rotation is the strategy.** Month 0's leaders (RGTI QBTS OKLO BE IONQ RKLB HOOD ...: the quantum / nuclear /
  retail-favourite theme) lost 15% held for the year. Re-picking every 4 weeks moved the sleeve into
  memory, storage and optics (MU, STX, SNDK, LITE, WDC), which supplied most of the gain. 63 names were held;
  4.8 of 20 changed per rebalance.
- **It was a rough ride.** Two periods lost more than 20% (Nov 2025 −22%, Jun-Jul 2026 −21%), and the sleeve was
  ~25% under water in November before it made anything. It beat SPY in 9 of 13 periods.
- **In a calm year for SPY the blend's drawdown is the sleeve's, not SPY's:** 70/30 fell 15.7% against SPY's 8.9%.
  The "blend keeps SPY's drawdown" result of Parts 4 and 6 is about 2020, when SPY itself fell 34%.
- **One year depends heavily on which weeks the rebalances fall.** The four possible weekly schedules give
  +21.4% / +23.3% / +31.6% / +32.4% over the same months (SPY +15.4% to +17.3%; drawdown ~−40% on all four).
  Quote the range, not one schedule.

## The shadow ledger on the weekly refresh (`scripts/shadow/momentum_shadow_ledger.py`, stage 6 of `scripts/weekly_refresh.sh`)
- **Schedule:** a decision on the last trading session of every 4th week, counted from the week of 2026-10-02
  (next: 2026-10-30, then 11-27, 12-24).
- **Any run day works.** A run records every scheduled decision that has closed and is missing from the ledger,
  using bars up to that decision date only. Friday evening, Saturday, Sunday, Monday before or after the open, or a
  week late all write the same row. A week counts as closed at 16:30 ET on its last session, so a Friday run before
  that records nothing and leaves the decision due for the next run. A missed period is caught up, never skipped.
- **Off-weeks** only re-mark the ledger (~32 tickers, ~6 s). A due week fetches the universe's daily bars (~15 min);
  only the 2 newest bar snapshots are kept (~83 MB each).
- **Research stage:** it runs last and does not change the weekly refresh's overall status; its exit code is in the
  stamp as `momentum_shadow_ledger`, and `--status` prints what is still owed with no network call.
- **Verified:** 23 unit tests (every run day Fri-Wed, Friday before the close, off-weeks, one and two missed periods,
  a holiday Friday, late recording cannot see later bars). The new code path reproduces the original 10-02 ledger
  row for row from the 10-03 bars, and again from a fresh 10-06 fetch (late recording). The orchestrator's
  exact call was rehearsed for an off-week.

---

# Part 8: which picks hurt, why whole months go red, and what can be done about it (2026-10-07)

**Questions (user, from the Part 7 chart):** why was BBAI picked three times while it fell; do pump-and-dump names get
in; are bad months a few bad stocks or everything at once; can we switch toward SPY or sell into strength so less is given back?

## 18: anatomy of the sleeve (`18_sleeve_anatomy.py`, `18_results.txt`). Diagnostic; 8,060 picks, 2019-2026, point-in-time
- **A pick is chosen on its PAST year, not on what it does next.** BBAI ranked 14th, 13th and 17th of 300 on trailing
  12-1 return (+257%, +311%, +207%; $1.49 → $7.19 in a year) at the three decisions it was held, and left when it fell
  to 33rd. One pick over 4 weeks: mean +3.4%, median +1.6%, **loses money 46% of the time**, worse than −20% in 10%,
  better than +20% in 15%. A losing streak lasts 1 period in 66% of cases, 2 in 24%, 3 or more in 10%.
- **Price features do not separate good picks from bad.** Thirds by volatility, distance above the 200 SMA, last-3-month
  return, smoothness or liquidity rank show no difference in mean (all CIs include zero). More extreme momentum is
  better (+1.9% [+0.03, +3.9] top third vs bottom third). High volatility brings more of BOTH tails (−20%: 16% vs 5%).
- **"Pump" picks are worse, and profitability splits them.** Pump = more than doubled in 3 months, or price above 2x its
  200-day average (13% of picks; 29% of picks in the last 12 months).

  | pick | n | mean | median | worse than −20% |
  |---|---|---|---|---|
  | pump, last filed quarter profitable | 366 | +3.8% | +0.7% | 17% |
  | pump, last filed quarter a net loss | 429 | **−1.0%** | **−9.1%** | 29% |
  | pump, no filing on record | 256 | −0.3% | −3.6% | 29% |
  | not pump, profitable | 3,824 | +3.8% | +2.2% | 5% |
  | not pump, loss-making | 1,703 | +4.3% | +1.4% | 13% |

  Loss-making pump minus every other pick: **−4.8% per period [−8.8, −1.2]** (194 dates). Profitability makes no
  difference outside the pump group. Last 12 months: loss-making pumps (OPEN, RGTI, QBTS, APLD, IREN, ...) −9.1% a
  period; profitable pumps (SNDK, MU, WDC, LITE, STX, ...) +7.6%.
- **Bad periods are everything at once.** The sleeve's period return and the share of its names that rose correlate
  0.88. In the worst 10% of periods only 7% of names rose and the median name lost 20.4%; removing each period's 3 worst
  names in hindsight still leaves −16.3% of −19.5%. In those periods SPY −4.6%, QQQ −6.3%, SMH −9.6%, ARKK −14.1%.
- **The common factor is the theme the leaders belong to.** Daily R²: SPY 0.38, SMH 0.47 over the full history; over the
  last 12 months **SMH 0.65** (beta 1.34) and SPY beta 3.3. The two all-red months of Part 7 were semiconductor/growth
  sell-offs that SPY barely registered (decision 2026-06-12: sleeve −21.3%, 0 of 20 names up, SMH −7.1%, SPY +0.3%;
  2025-10-31: −22.3%, 0 of 20 up, ARKK −12.9%, SPY −1.0%). The 20 names average 0.32 pairwise correlation (0.41 lately)
  against 0.25 for random liquid-300 names; 213 of the last 260 picks map to XLK.
- **Froth is a weak warning.** After the sleeve's trailing 3-month return was in its top fifth, the next period's edge
  over SPY averaged −0.4%; after the bottom fifth, +5.5% (rank correlation −0.10; ±0.20 is needed with ~100
  independent periods). Its drawdown, the picks' correlation and their volatility warn of nothing.

## 19: rules tested (`19_quality_and_timing.py`, `19_results.txt`). Point-in-time, monthly, 4 offsets, paired against the plain version
IN-SAMPLE: `no_pump`, `no_lossy_pump`, `b_trim`, `b_add` were prompted by 18 on this same history.

| sleeve alone | CAGR | Sharpe | MDD med / worst | vs plain /yr [95% CI] | last 12m |
|---|---|---|---|---|---|
| plain top 20 | 39.7% | 0.95 | −53% / −57% | — | +34.8% |
| drop pumps | 41.6% | 1.03 | −47% / −51% | +0.3% [−4.0, +4.4] | +20.2% |
| drop pumps with a reported loss | 41.0% | 1.00 | −56% / −57% | −0.2% [−5.6, +3.7] | +46.8% |
| drop the most volatile 10% | 29.9% | 0.93 | −34% / −35% | −11.9% [−24.1, −0.9] | +26.0% |
| drop names up >50% last month | 42.4% | 1.02 | −51% / −60% | +0.3% [−3.0, +3.2] | +14.7% |
| steady climbers only | 27.6% | 0.85 | −42% / −45% | −13.1% [−24.3, −2.6] | +54.6% |
| risk-adjusted momentum | 33.7% | 0.95 | −42% / −44% | −7.6% [−18.9, +2.6] | +31.3% |
| sector cap 30% | 34.8% | 0.94 | −45% / −51% | −5.7% [−14.8, +2.2] | +4.9% |

| 70/30 blend | CAGR | Sharpe | MDD | vs fixed 70/30 /yr [95% CI] | last 12m (MDD) |
|---|---|---|---|---|---|
| fixed 70/30, reset monthly | 25.0% | 1.04 | −34% | — | +24.2% (−15.7%) |
| never reset (drift) | 26.8% | 0.96 | −34% | +2.8% [−0.2, +5.6] | +27.4% (−28.7%) |
| sleeve 15% after a hot 3 months | 25.1% | 1.08 | −34% | −0.2% [−1.8, +1.3] | +30.2% (−13.4%) |
| sleeve 45% after a cold 3 months | 26.6% | 1.07 | −34% | **+1.3% [+0.2, +2.3]** | +29.2% (−15.7%) |
| both (band) | 26.5% | **1.10** | −34% | +1.0% [−1.1, +2.8] | +35.5% (−13.4%) |
| sleeve 15% while below its 100-day average | 22.5% | 0.99 | −34% | −2.1% [−3.5, −0.5] | +16.8% |
| take profit per name at +30% | 24.0% | 1.03 | −34% | −1.0% [−1.8, −0.4] | +23.5% |
| take profit per name at +50% | 25.0% | 1.05 | −34% | −0.1% [−0.8, +0.4] | +24.0% |
| drop pumps with a reported loss | 25.3% | 1.07 | −34% | +0.0% [−1.4, +1.1] | +26.7% |

**Reading**
1. **Selling strength works at the portfolio level and fails at the stock level.** Trimming the whole sleeve after its
   hottest 3 months and adding after its coldest is the only timing rule that does not cost return; per-name
   profit-taking, trend-following the sleeve, stops and breakers (Part 6) all do. The monthly reset to 70/30 is already
   a form of it: without the reset the sleeve drifts to 42% of the account and last year's drawdown is −28.7%, not −15.7%.
2. **The band's size is small and unproven.** Across 12 settings (look-back 42/63/126 days, top/bottom 20% or 30%, two
   share pairs) it is positive every time, +0.0% to +1.0% a year, Sharpe 1.06-1.11 against 1.04, and no CI excludes zero.
   The first setting tried is the best of the twelve, so expect the average (~+0.4%/yr). Last year it was hot before both
   −20% months (and before two good ones) and cold before two good ones.
3. **The loss-making-pump rule is real at the pick level and unmeasurable at the portfolio level.** It changes 1.1 of 20
   names per rebalance, so the expected gain (~1-2%/yr) sits inside a ±4.6%/yr CI. Dropping ALL pumps cuts the
   drawdown a little but throws out the year's biggest winners with the losers.
4. **Diversifying the sleeve costs return.** The sector cap and the volatility cap lower CAGR more than they lower risk;
   the edge is the concentrated theme.
5. **A red month cannot be avoided, only sized.** It is a theme-wide fall that the 20 leaders amplify about 3x.

## Next testable variants
1. Add two arms to the forward shadow, dated when added: the band (15% / 30% / 45% on the sleeve's trailing 3-month
   return against fixed thresholds frozen from 2019-2026) and the loss-making-pump exclusion (needs the latest SEC
   filing for each candidate). Forward data is the only clean test left for both.
2. Fundamentals for delisted names (SEC by CIK) would remove the "unknown" group and let the pump rule be tested fairly.

---

# Part 9: can a theme sell-off be sidestepped? Daily exits, puts, rotation (2026-10-07)

**Question (user):** the June 2026 drop looked overextended beforehand and broke down in its first days. Why hold for the
whole month? Could we exit, buy puts on the theme index, or rotate out? Parts 6 and 8 tested per-name stops and
month-end checks only; this tests daily portfolio/theme exits, hedges and rotation (`20_theme_mitigation.py`,
`20_results.txt`). 70/30 blend, point-in-time 2019-2026, 4 offsets, paired against the fixed 70/30 (25.0% CAGR,
Sharpe 1.04, MDD −34%). The theme index is the ETF (of 45) that best matches the 20-name basket over 60 days:
ARKQ 17% of decisions, IGV 14%, XLE 12%, QTUM 10%, BLOK 9%. "Extended" = that ETF more than 15% above its 100-day
average (24% of decisions).

| rule | CAGR | Sharpe | MDD | vs fixed 70/30 /yr [95% CI] | acted | last 12m (MDD) |
|---|---|---|---|---|---|---|
| fixed 70/30 | 25.0% | 1.04 | −34% | — | — | +24.2% (−15.7%) |
| leave when the sleeve is down 10% since entry | 24.1% | 1.05 | −34% | −1.2% [−2.7, +0.2] | 21% of periods | +20.3% (−14.2%) |
| leave on a theme-ETF breakdown | 16.8% | 0.85 | −35% | **−8.0% [−11.5, −3.6]** | 70% | +9.8% |
| the same, only if the theme was extended | 23.2% | 1.00 | −34% | −1.5% [−4.3, +1.2] | 18% | +24.0% (−15.0%) |
| leave when the sleeve is below its 20-day average | 17.9% | 0.91 | −35% | −6.8% [−11.0, −1.6] | 82% | +14.9% |
| puts on the theme ETF, every period (model) | 22.9% | **1.11** | **−28.5%** | −2.3% [−3.8, −0.2] | 100% | +14.9% (−13.0%) |
| puts only when extended (model) | 24.8% | 1.06 | −34% | −0.2% [−1.4, +1.0] | 24% | +23.1% (−15.2%) |
| short the theme ETF when extended (idealised) | 23.2% | 1.03 | −34% | −2.0% [−6.6, +1.8] | 24% | +10.5% |
| skip names whose theme ETF is below its 50-day average | 23.3% | 1.02 | −34% | −1.4% [−3.1, +0.1] | — | +22.4% |
| cap the theme at 6 names when extended | 25.5% | 1.06 | −34% | +0.3% [−0.0, +0.7] | — | +24.2% |

Sensitivity: exit at −8% −2.5% [−4.5, −0.6]; at −15% −1.3% [−2.8, −0.1]. Puts-when-extended with vol 20% lower
+0.5% [−0.5, +1.5]; 20% higher −0.9% [−2.4, +0.4].

**Reading**
1. **Leaving after the breakdown starts loses, because most breakdowns reverse inside the month.** The last 12
   months, sleeve return held vs with the −10% exit: Nov 2025 −22.3% → −9.5% and Jun 2026 −21.3% → −7.5% (the
   exit saved 13-14 points each time, as the user saw). But Mar 2026 +21.9% → −0.5% (it was down 13% in week one), Jul 2026
   +3.2% → −10.7% (down 21% mid-month) and Jan 2026 −3.3% → −6.2%. Net over the year: −11.9 points. Helped in 3
   periods, hurt in 3.
2. **"Overextended" is a weak warning, not a signal.** The theme was extended at 5 of the last 13 decisions: it then
   fell twice (−7%, −9%) and rose three times (+4%, +24%, +15%). Over the full history the sleeve's next 4 weeks average
   +1.8% after an extended decision against +4.0% otherwise, and lose more than 10% in 15% of cases against 10%.
   That supports holding less (Part 8's band), not leaving.
3. **Theme-level breakdown signals fire constantly** (70-82% of periods) and are the most expensive rules tested.
4. **Puts are the only thing that cut the worst drawdown** (−34% → −28.5%, because they paid in March 2020), at ~2.3% a
   year: fairly priced insurance in this model. Bought only when the theme is extended they are roughly free
   (−0.2%/yr) and protect little (−15.2% vs −15.7% last year). **These are Black-Scholes estimates, not quotes.**
   Real costs are higher where it matters: most theme ETFs chosen here (QTUM, ARKQ, BLOK) have thin option markets,
   so a real hedge would use SMH or QQQ puts and carry basis risk.
5. **Rotating out does not help.** Skipping names in a broken theme costs 1.4% a year; the cap rarely binds.

**Verdict:** the hypothesis class "act on a sell-off once it is visible" (exits and trend signals on names, the sleeve
or its theme, at monthly or daily frequency) is exhausted on this history: every form costs return. What survives is
decided BEFORE the drop: the sleeve's share of the account, the monthly reset, a smaller share when it is hot or
extended, and optionally puts as paid insurance.

**Next testable variant:** real option quotes. Capture nightly SMH/QQQ put prices in the existing IV-surface job, so
the put arms can be priced from quotes in the forward shadow instead of from a model.

---

# Part 10: puts on the single names, bought on the first red day or on bad news (2026-10-07)

**Question (user, with an NVDA chart):** instead of index puts, buy puts on the stocks themselves when they are
overextended or running out of momentum, on the first red day or on bad news. The drop "usually lasts a few days",
so the put should protect. Script `21_single_name_puts.py`, output `21_results.txt`. Liquid-300, point-in-time,
2019-01-04..2026-09-24. Signal on a daily close, entry at the next open.

"Extended" = close 15% or more above its 50-day average (a second definition in volatility units is in the results
file and agrees). Triggers, all requiring the name to be extended the day before:
- **first red day**: the day after a 20-day closing high closes down (also a version that needs a 3%+ fall),
- **fade**: the first close below the 10-day average after 10 or more sessions above it,
- **gap**: opens 3%+ below the prior close and closes down. This is the price proxy for bad news; the catalyst feed
  does not reach back to 2019.

The thresholds were fixed before the run and not tuned.

## A. What the shares do over the next 10 sessions (no option model)

| after | cases | mean | median | lower 10 sessions later | fell >10% | rose >10% | vs other extended names the same day [95% CI] |
|---|---|---|---|---|---|---|---|
| any day, any name | 578,755 | +0.61% | +0.56% | 46% | 7% | 8% | — |
| any day the name was extended | 39,415 | +0.83% | +0.23% | 49% | 14% | 16% | — |
| first red day | 8,470 | +0.75% | +0.24% | 49% | 12% | 14% | +0.12% [−0.19, +0.41] |
| first red day, down 3%+ | 2,293 | +0.94% | +0.23% | 49% | 18% | 20% | +0.32% [−0.33, +0.89] |
| fade | 2,961 | +0.86% | +0.38% | 48% | 12% | 14% | −0.04% [−0.37, +0.31] |
| gap down 3%+ | 1,953 | −0.24% | −1.67% | 55% | 28% | 22% | +0.15% [−1.07, +1.51] |
| gap down, no earnings report | 1,772 | −0.36% | −1.84% | 55% | 28% | 22% | +0.31% [−0.85, +1.56] |
| first red day, name held by the sleeve | 1,574 | +1.07% | +0.49% | 48% | 20% | 22% | +0.32% [−0.78, +1.23] |
| gap down, name held by the sleeve | 612 | +1.30% | −0.29% | 50% | 31% | 32% | +0.64% [−1.55, +2.92] |

## B. What a 2-week at-the-money put costs: real quotes

The nightly IV-surface capture holds closing bid/ask for 882 names on 11 sessions (2026-09-11..10-05). This is the
only real option pricing in the study.

| liquid-300 names by realized vol | implied vol | realized vol | implied / realized | half of the bid/ask, % of premium | premium, % of share price |
|---|---|---|---|---|---|
| under 30% | 25% | 25% | 1.00 | 12.9% | 1.8% |
| 30-50% | 34% | 37% | 0.90 | 9.9% | 2.5% |
| 50-80% | 51% | 66% | 0.78 | 7.4% | 3.7% |
| over 80% | 67% | 92% | 0.70 | 4.9% | 4.9% |
| extended names | 60% | 79% | 0.76 | 4.9% | 4.2% |
| the 20 sleeve names | 64% | 85% | 0.75 | 5.0% | 4.6% |

- **The implied/realized ratio is not constant**, so Part 9's single 0.82 multiple was too crude. This part prices
  with a fitted line, implied = 0.119 + 0.590 x realized (R² 0.78). A single name sits 0.89x to 1.11x that line
  (middle half of names), which is why the ±20% rows below matter.
- **Puts were about 5% dearer than the line on a gap-down day** (42 name-days), and no dearer than the day before on
  trigger days (−3% to −4%, 66 name-days). Small samples from one calm month.
- SPY, QQQ and SMH are not in the capture. Single names are.

## C. Does the put pay? Model premium against what the shares delivered

Payback = payoff per $1 of premium, 10-session at-the-money put. 1.00 is a fair price. A put needs about 1.10 to
cover the bid/ask both ways.

| put bought | premium | payoff | payback [95% CI] | after bid/ask |
|---|---|---|---|---|
| any day, any name | 2.97% | 2.57% | 0.87 [0.77, 0.97] | 0.78 |
| any day the name was extended | 4.30% | 3.88% | 0.90 [0.79, 1.03] | 0.82 |
| first red day | 3.95% | 3.53% | 0.89 [0.76, 1.04] | 0.81 |
| first red day, down 3%+ | 4.74% | 4.66% | 0.98 [0.84, 1.14] | 0.89 |
| fade | 3.87% | 3.44% | 0.89 [0.77, 1.04] | 0.81 |
| gap down 3%+, priced at the line | 5.73% | 6.61% | 1.15 [1.00, 1.34] | 1.05 |
| gap down 3%+, priced as quoted on gap days | 6.04% | 6.61% | 1.10 [0.95, 1.27] | 0.99 |
| any trigger, name held by the sleeve | 5.05% | 5.48% | 1.09 [0.90, 1.29] | 0.98 |

By year the gap put paid back 0.99 to 1.41 at the line (2019-2026, every year); the first-red-day put 0.66 to 1.31
(above 1.00 only in 2021 and 2022).

## D. On the portfolio: fixed 70/30 plus a put on each held name when its trigger fires

One put per share held, at the money, 10 sessions, one open put per name. Monthly, 4 offsets, paired against the
fixed 70/30. Model premiums plus 4.9% of premium each way.

| put on a held name when | CAGR | Sharpe | MDD | vs fixed 70/30 /yr [95% CI] | puts a year | premium a year (of the account) | last 12m (MDD) |
|---|---|---|---|---|---|---|---|
| none (fixed 70/30) | 25.0% | 1.04 | −33.8% | — | — | — | +24.2% (−15.7%) |
| first red day | 25.1% | 1.08 | −33.3% | −0.3% [−1.8, +1.2] | 104 | 8.8% | +27.2% (−13.5%) |
| first red day, down 3%+ | 24.8% | 1.06 | −33.7% | −0.4% [−1.4, +0.5] | 59 | 5.4% | +27.3% (−13.8%) |
| fade | 24.7% | 1.05 | −33.5% | −0.5% [−1.7, +0.7] | 60 | 4.7% | +22.7% (−15.9%) |
| gap down 3%+ | 25.8% | 1.09 | −33.0% | +0.4% [−0.6, +1.3] | 48 | 4.9% | +26.7% (−14.5%) |
| any of the three | 25.4% | 1.10 | −32.8% | −0.3% [−2.1, +1.7] | 121 | 10.4% | +28.5% (−12.6%) |
| **no trigger: a put whenever the name is extended** | 25.6% | 1.13 | −32.4% | −0.2% [−2.2, +1.9] | 151 | 13.3% | +26.0% (−13.2%) |
| any of the three, puts 20% cheaper | 28.0% | 1.19 | −32.8% | +1.8% [−0.1, +4.0] | 121 | 8.3% | +32.7% (−12.2%) |
| any of the three, puts 20% dearer | 22.8% | 1.01 | −32.9% | −2.3% [−4.3, −0.5] | 121 | 12.5% | +24.6% (−13.1%) |
| Part 9 reference: theme-ETF puts every period | 22.9% | 1.11 | −28.5% | −2.3% [−3.8, −0.2] | — | — | +14.9% (−13.0%) |

Last 12 months, "any of the three": the puts added +2.2% of the account in the Nov 2025 sell-off (sleeve −22%) and
+1.4% in June 2026 (sleeve −21%), and cost 0.9%, 0.6% and 1.5% in the three months the sleeve rose 22-24%.

## E. The NVDA chart

Every first red day off a 20-day closing high in NVDA over the last 12 months, with no "extended" filter: 18
occasions. NVDA was lower 10 sessions later in **14 of 18**, by 1.6% on average. A 10-session at-the-money put at
NVDA's own quoted price level cost 2.19% of the share price each time and paid back 2.33%; it made money on 9 of 18.

## Reading
1. **The chart is read correctly, and the pattern is too small to pay for a put.** On NVDA the first red day was
   followed by lower prices 14 times in 18, and the puts roughly broke even, because the typical fall (1.6%) is
   smaller than the premium (2.2%).
2. **Across all liquid names the first red day is a coin flip.** Lower 10 sessions later in 49% of 8,470 cases, with a
   +0.75% average, the same as any other day the name was extended. Names the sleeve holds do better still (+1.07%).
   The fade trigger is the same. As put triggers both are **rejected, not just unmeasured**: the top of the payback
   range (1.04) is below the ~1.10 needed after the bid/ask.
3. **Bad news is the one trigger with follow-through.** After a 3%+ gap down the name is lower 10 sessions later 55%
   of the time and falls more than 10% in 28% of cases. Priced as quoted on gap days the put pays back 1.10 before
   the bid/ask and 0.99 after: **break-even, so protection at no expected cost, not a profit**. This is unresolved,
   not rejected: the gap-day price rests on 42 quoted name-days.
4. **The trigger adds nothing over simply insuring extended names.** Buying a put whenever a held name is extended,
   with no trigger at all, scores the same or slightly better (Sharpe 1.13, MDD −32.4%).
5. **It does not cut the worst drawdown.** −33.8% → −32.8%. The worst drop (March 2020) started with nothing
   extended, so no puts were on. Index puts held all the time cut it to −28.5%, at about 2% a year.
6. **It tracks the portfolio better than index puts.** Last year the single-name puts took the 70/30 from +24.2% to
   +28.5% and the drawdown from −15.7% to −12.6%, while theme-ETF puts bought only when extended did nothing (+23.1%,
   −15.2%). The sleeve fell 21-22% in those months and the theme ETF only 7-9%.
7. **The answer depends on the price paid.** About 10% of the account goes out in premium each year and about 10%
   comes back. A 20% error in the put price moves the result from +1.8% to −2.3% a year, and one name's real price is
   routinely 10% off the fitted line.

**Limits.** Premiums in C and D are Black-Scholes at a line fitted to 11 sessions of quotes from one calm month.
Earnings dates inside a put's life are not priced (real puts cost more into a report). The gap is a price proxy for
bad news. One option contract covers 100 shares, so a 1:1 hedge needs a position of at least 100 shares per name;
at 1.5% of the account per name that is roughly a $330k account for a $50 stock. Drawdowns mark the puts at a constant vol.

**Verdict:** first-red-day and fade puts are rejected. Single-name puts are insurance at about a fair price that
fits the portfolio better than index puts but does not cover a crash from a non-extended start. The gap/bad-news
put is the only open question.

**Next testable variant:** price it from real quotes going forward. The nightly capture already holds bid/ask for
the single names, so add a put arm to the forward shadow: on each trigger on a held name, record the closing ask of
the 2-week at-the-money put and settle it at the bid or intrinsic value at expiry. Use the live catalyst feed for
"bad news" next to the gap proxy. Index puts still need SPY/QQQ/SMH added to the capture (Part 9).
