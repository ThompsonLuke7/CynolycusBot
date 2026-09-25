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
