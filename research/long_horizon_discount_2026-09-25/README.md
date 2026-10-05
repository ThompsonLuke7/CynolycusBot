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
AI-hardware theme. Cadence: run `--snapshot` every 4 weeks after a Friday close, and `--mark` any time. It is not wired
to any scheduler. Each snapshot fetches ~620 days of bars (~15 min) into `Data/research/shadow_bars/{date}/`.
