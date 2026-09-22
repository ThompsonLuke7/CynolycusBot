# Profit, prediction, and execution investigation

Research date: September 19, 2026. **Paper trading only.** Final calculations use `run3/` and `analysis/`; earlier runs preserve accounting/join development and are superseded.

**We can identify promising differences, but we have not identified a validated winner filter. Entry scores generally fail to distinguish profitable trades. Execution costs, instrument choice, position sizing, and accounting completeness matter substantially. The strongest historical feature split fails to repeat convincingly on newer open trades.**

## 1. What actually made money

“Overall” below means the broker-verifiable period **July 13–September 18**. The broker activity history contains a July 9 funding event, but its first retained trade is July 13. Local ledgers contain older records that cannot be reconciled to this order history; they are not silently added. “Last month” means **August 19–September 18 inclusive, ET**, classified by realization date. August is also reported separately.

| Realized results | Overall | Last month | Calendar August |
|---|---:|---:|---:|
| Profits from winning exits | $291,821.81 | $211,139.34 | $145,382.66 |
| Losses from losing exits | −$644,798.20 | −$308,948.13 | −$320,021.78 |
| **Net before fees** | **−$352,976.39** | **−$97,808.78** | **−$174,639.12** |
| Reported account fees | −$638.11 | −$327.57 | −$357.81 |
| Winning / losing / flat exits | 245 / 445 / 9 | 148 / 249 / 4 | 124 / 183 / 4 |
| Exit win rate | 35.1% | 36.9% | 39.9% |
| Profit factor | 0.45 | 0.68 | 0.45 |

An **exit** here is one broker sell order, with its partial fills aggregated, or one confirmed worthless expiration. It can be a profitable trim of a position that ultimately loses. Consequently, this table is appropriate for realized dollars, but its win rate is not a completed-position win rate. Fees include the account's entry/exit fees in each period; they are not allocated to individual lifecycle labels.

For outcome comparisons, I reconstructed **788 flat-to-flat symbol lifecycles: 619 closed and 169 reconstructed as still open**. Fifteen closed lifecycles involving exercise/basis transfers are excluded from the ordinary winner classifier. The remaining **604 completed lifecycles contain 162 winners, 433 losers, and 9 flat results**. In the last-month closure cohort, there are **348 completed lifecycles and 104 winners**. None of these counts treats partial fills as independent training examples.

Realized attribution, before fees:

| Module | Overall net | Last-month net |
|---|---:|---:|
| HTF Swing | −$15,163 | **+$28,485** |
| Meta Ranker | −$83,184 | −$1,488 |
| Momentum | −$78,662 | −$54,620 |
| Multiticker Swing 30m | −$77,491 | −$35,574 |
| Dealer | −$115,329 | −$47,862 |
| Intraday Structure | −$5,558 | −$5,558 |
| SPY Daytrader | −$875 | −$519 |
| Conflicting/shared ownership | +$26,376 | +$20,763 |
| Unattributed | −$3,090 | −$1,435 |

These are conservative research attribution buckets, not audited sleeve accounting. Exact entry IDs take precedence; causal plans and exact exit IDs are weaker fallbacks. Where a plan-based entry owner conflicts with an exit owner, the lifecycle stays in the conflicting bucket. **Do not allocate that bucket to whichever module makes its results look best.** Exercised shares can combine exposures from multiple modules.

The five largest completed winners were:

| Instrument | Reconstructed owner | Whole-lifecycle profit |
|---|---|---:|
| SNDK Aug 21 1200 call | HTF | $31,930 |
| DELL Sep 18 470 call | Conflicting Meta/HTF provenance | $18,840 |
| SMTC Sep 18 140 call | Dealer | $14,720 |
| CRWD Sep 18 187.5 call | HTF | $14,680 |
| ASST Sep 18 2.5 call | HTF | $8,192 |

DELL's $18,840 includes a $1,240 earlier trim and the previously omitted **$17,600 September 18 close**. The Momentum DELL 500-call lifecycle made $6,110, including an earlier trim and the $4,815 final close. Whole-trade and last-day profit are different quantities.

The top five completed winners account for **36.0% of completed-winner profits overall and 44.9% in the last-month closure cohort**; the top ten account for 49.0% and 61.0%. Removing the five worst completed losses still leaves that cohort negative: approximately −$276,020 overall and −$52,976 last month. The weakness is broader than a few bad outliers.

### Open gains and total account performance

The frozen September 18 20:05 ET broker snapshot contains **168 positions: 63 positive marks**, with $67,156 of unrealized gains, −$68,103 of unrealized losses, and **−$947 net unrealized P&L**. These are broker marks, not executable liquidation proceeds. The largest positive marks include INDP +$10,519, BE +$5,132, ABCL +$4,069, PSIG +$3,807, FWDI +$3,744, and CHPT +$3,545. [Every open position is exported](analysis/open_positions.csv).

Reported equity is $640,434 versus $1,000,000 recorded funding, a **−35.96% account change**. Relative to the August 18 closing snapshot of $777,468, the last-month equity change is **−$137,034, or −17.63%**. This differs from realized P&L because marks and inventory also change. The maximum drawdown among available daily account snapshots is approximately 43.1%; missing daily snapshots mean this is not a complete intraday drawdown series.

## 2. Does the model distinguish winners from similar-score losers?

Generally, **no reliable discrimination is visible in this selected, completed-trade sample**. Scores are compared within their own module and instrument type. They are not pooled across models, and ranking scores such as Meta's roughly 0.99 are not interpreted as 99% probabilities of profit.

An AUC of 0.50 means a randomly chosen winner and loser are ordered no better than chance. Higher is better. Overall results:

| Module / instrument | Scored nonflat trades | Winner AUC | Clustered 95% interval |
|---|---:|---:|---:|
| Momentum shares | 42 | 0.438 | 0.256–0.630 |
| HTF shares | 59 | 0.502 | 0.350–0.660 |
| Meta shares | 28 | 0.449 | 0.200–0.750 |
| Swing 30m options | 143 | 0.494 | 0.384–0.599 |
| Dealer options | 48 | 0.405 | 0.194–0.628 |
| SPY options | 105 | 0.484 | 0.315–0.629 |

Intervals resample tickers, or entry sessions for the single-ticker SPY module. Momentum options show AUC 0.692 overall and 0.844 last month, but those samples have only **three and two winners**, respectively. They are too small for a persuasive result; no confidence interval is presented. HTF options similarly have only four completed winners. Intraday Structure has no comparable archived entry model score in this joined dataset.

Last-month Swing 30m AUC improves to **0.596**, with a wide 0.404–0.764 interval. Its highest score tercile wins 50%, versus 33% and 28% for the lower two terciles, yet **all three terciles lose dollars**: −$8,487, −$12,089, and −$14,577, respectively. Better classification alone does not establish positive expectancy.

I also matched each winner to a distinct loser within the same module/instrument, within **14 entry-calendar days**, and within **0.25 within-cohort score standard deviations**. This produced **118 overall pairs and 65 last-month pairs**, with no reuse of a loser inside each cohort. The cohorts overlap and are not independent replications.

Examples from the last-month closure cohort:

| Winner versus loser | Entry score, winner / loser | P&L, winner / loser |
|---|---:|---:|
| FBRX / QTTB, Momentum shares | 0.418214 / 0.418057 | +$2,964 / −$1,159 |
| ANRO / CORZ, Momentum shares | 0.292230 / 0.292358 | +$455 / −$769 |
| GPN / USO calls, Swing 30m | 0.637542 / 0.637542 | +$150 / −$2,750 |
| BABA / GLD calls, Swing 30m | 0.561950 / 0.561950 | +$1,950 / −$1,120 |

FBRX and QTTB entered approximately eight minutes apart on July 14. Their nearly identical predictions preceded **+66.9% versus −52.1% whole-trade returns**. [All matched pairs](analysis/similar_score_pairs.csv) are available for individual review.

This is a test of **realized profitability discrimination**, not a claim that the models fail at their original training targets. Ranking, directional movement, maximum favorable excursion, and executable net profit are different objectives. The archived trade scores also represent a narrow, selected tail of the full ranking distribution.

## 3. Which features segregate the outcomes?

### Momentum shares: prior trend is the strongest historical lead

Among 41 completed trades with an uninterrupted prior-price window:

| Prior 20-session return | Trades | Wins | Win rate | Net P&L |
|---|---:|---:|---:|---:|
| Positive | 23 | 15 | 65.2% | +$4,520 |
| Zero or negative | 18 | 1 | 5.6% | −$22,721 |

Winners' median prior return was **+38.0%**, versus **−4.9%** for losers. Prior-trend AUC is 0.893; the exploratory Mann–Whitney result has BH-adjusted q≈0.0043 across 236 feature tests. Winners were also closer to their prior 20-session range highs: median range position 0.82 versus 0.51. These features were computed only through the session before the decision.

Within similar-score Momentum share pairs, the median winner-minus-loser prior trend difference remains approximately **38 percentage points**. However, after correcting across the **205 matched-feature comparisons**, **no feature survives**; the smallest adjusted q is 0.511. Unclustered exploratory p-values are not sufficient evidence for deployment.

Most importantly, this is a **maturity-selected historical cohort**: its entries are July 13–August 4, and 40 of its 41 feature-complete trades are also in the last-month closure cohort. Those two result tables are almost the same experiment. Newer still-open Momentum shares do not confirm the apparent rule: six nonpositive-prior-trend positions carry +$2,660 net marks, versus −$3,884 for 24 positive-prior-trend positions; both groups have negative median marked returns. Unequal holding periods and unrealized marks prevent treating this as a clean validation, but it materially weakens the proposed filter.

**Decision: retain prior trend/range location as a preregistered challenger, not a production veto.** The analogous HTF share split is also weaker and does not reproduce Momentum's win-rate separation.

### Intraday Structure: liquidity and premium are plausible, but confounded

Winners' median prior 20-session dollar volume is approximately $24.8 billion versus $1.49 billion for losers; their median option entry premium is approximately $1.86 versus $0.72. The liquidity separation survives the exploratory unclustered multiple-test adjustment (q≈0.028). Larger premiums can reduce percentage spread friction, but neither liquidity nor premium is itself proof of a tradable rule.

There are only **12 genuinely positive completed lifecycles**, concentrated in a handful of names: MU, MSTR, and NVDA contribute eight. Repeated names, direction, and session conditions confound these tests. Test a liquidity/premium budget with ticker/session controls and fresh quotes; do not add those names to a whitelist because they won this sample.

### No broad model-feature separator is established

The archived cross-model scores, rank, confirmation metrics, news score, and volume-context measures do not produce a robust, multiplicity-adjusted separator in the matched-score analysis. This does not prove that no useful combination exists. It says this dataset does not justify one yet. Full entry-time model vectors, bundle versions, and reliable inference wall clocks are not consistently archived; reconstructing all model features from today's matrices would risk revision and timing leakage.

## 4. Execution: precision, recall, and timing

### Trade precision is different from missed-opportunity recall

Completed-position profit precision, excluding exercise-transfer lifecycles, is 40.5% for Momentum shares, 45.8% for HTF shares, 39.5% for Swing 30m options, 16.4% for Intraday Structure, and 16.2% for SPY options. These reflect different policies, costs, and holding periods; they are not fair model competitions.

To examine recall, I evaluated **logged four-hour candidate decisions**, including those not joined to a trade, over fixed 1-, 5-, and 10-session underlying horizons. The common reference is the next session's open; the primary positive label below is a positive return to the fifth session's close. This is a gross underlying opportunity label, **not option profitability or an executable counterfactual**.

Last-month decision cohort, with a complete five-session outcome:

| Module | Candidates | Joined-trade precision | Joined-trade recall among positive candidates |
|---|---:|---:|---:|
| Momentum | 262 | 29.0% | 10.1% |
| HTF | 306 | 48.6% | 12.2% |
| Meta | 308 | 45.5% | 3.8% |
| Dealer | 92 | 41.4% | 29.3% |

**These recall values are restricted, conservative join-based diagnostics.** The population contains only logged selected candidates, repeated ticker decisions, and held names. Some entries have missing/ambiguous joins; many untraded candidates were already held, unfundable, illiquid, or intentionally declined. The values do not mean that the broker failed to execute the remaining percentage. We cannot measure full-universe recall, distinct-move recall, or complete unfilled-option profit recall from the retained records. The 30m/SPY/Intraday engines lack a comparable audited common candidate denominator in this study.

Within-decision rank controls also argue against “just trade the highest ranks”: last month Meta's top three candidates underperform ranks 4+ by an average 2.60 percentage points over five sessions, and HTF's by 2.47 points; Momentum's difference is +0.33 points. These are exploratory means across only 25–31 decisions, not significance-tested policy returns.

### Losing exits face substantially worse spread costs

The 30m module supplies the usable exact-order exit quote evidence. Restricting to quotes aged **0–60 seconds at the fill**:

| Cohort | Exit observations | Median quoted spread / mid | Median fill below mid |
|---|---:|---:|---:|
| Overall winners | 57 | 13.3% | 6.6% |
| Overall losers | 81 | 26.6% | 12.1% |
| Last-month winners | 20 | 12.6% | 7.3% |
| Last-month losers | 35 | 20.0% | 9.3% |

Across all 138 fresh observations, median exit cost is **8.4% of midpoint**. The aggregate midpoint-minus-fill difference is approximately **$60,182**; last month's 55 observations account for $22,316. **These amounts are not recoverable-profit estimates.** A midpoint is not guaranteed executable, and replacing an exit with a passive limit can create nonfills, adverse selection, or additional market loss.

At entry, 30m last-month winners and losers paid much more similar midpoint premiums: **4.0% versus 4.4%**. Their entry spreads were approximately 10.3% versus 11.2%. Four-hour entry quotes often lack trustworthy timestamps, so their nominal midpoint comparisons are secondary evidence. Expired-contract quotes and stale option trade bars are not used to reconstruct hypothetical option P&L.

The direction-to-instrument translation also loses opportunities. Among covered HTF option trades where the underlying moved in the call's favor, **7 of 10 still lost option money overall, and 5 of 8 last month**. For Intraday Structure, 16 of 23 favorable-direction observations lost money. These samples are small; spreads, theta, volatility, and timing are not separately identified. Nevertheless, getting the stock direction right is demonstrably insufficient.

### Faster broker fills are not the main winner/loser distinction

Typical first broker fills are fractions of a second after submission. In 30m, the overall medians are **0.093 seconds for winners and 0.091 seconds for losers**. Canceled ladder attempts are analyzed separately; the first filled rung is not the entire execution campaign.

The four-hour scheduling delays are much larger but generally shared by winners and losers: HTF's median signal-availability-to-submission delay is approximately **25.15 minutes in both groups**; Momentum's is approximately 32 minutes. Afternoon decisions can defer to the next session. The current study does not establish that eliminating these delays would turn the strategies profitable.

On sufficiently covered underlying paths, 30m winners surrendered a median **0.75 percentage points** from observed favorable excursion to exit, versus **2.22 points for losers**. Last-month values are 0.83 and 2.12 points. This supports investigating deterioration while holding, but it is an outcome-dependent diagnostic: the peak is known only in hindsight. It does not validate selling at that peak or widening/removing stops.

## 5. Additional architectural findings and prioritized next work

1. **Finish account-level accounting before optimizing module totals.** The raw fills reconcile to every filled order, but the account is not fully certified: reconstructed inventory contains **232 CMCO shares with $5,009.40 basis**, absent from the frozen broker position snapshot, with no corresponding fill/settlement in the downloaded activities. Reconstructed cash differs from the snapshot by **$4.78**. Preserve both discrepancies and use the existing reconciliation/quarantine architecture to resolve them; do not invent a CMCO sale or write off its cost without evidence. The ledger audit also finds 35 rows with missing order IDs, 38 whose orders are absent, and four referencing unfilled orders. Some absent orders are older than broker retention.

2. **Prioritize an option execution-cost experiment with a complete decision funnel.** Record signal availability, actual inference time, decision and submission times, every attempt/replacement, fresh bid/ask and size, fills, nonfills, and exit rationale. Evaluate a challenger against the current policy on comparable paper opportunities, including the P&L of failed or delayed exits. Estimate net economic value separately for shares and options. An entry spread filter alone cannot address spread widening as an option deteriorates.

3. **Treat model outputs as rankings until economic calibration is demonstrated.** Preserve the current named baselines. Evaluate price-direction, target magnitude, and realized net outcomes separately by module, route, DTE, liquidity, and maturity. A higher score threshold is not supported by these results; even the better recent 30m score bucket remains negative. No new model was fitted in this study, and no existing holdout was retuned.

4. **Run a narrow prospective Momentum-share challenger for prior trend/range position.** Freeze the zero prior-20-session-return split before collecting new outcomes; retain all blocked and accepted candidates. Compare at identical fixed horizons, with the same costs, finite-capital rules, and score/time controls. Track newer open positions to maturity. Do not call the overlapping overall/month results validation, and do not extend this filter to HTF or options without independent evidence.

5. **Audit exercise exposure and notional concentration.** The early SNDK share trade used $167,495 and lost $58,098. Its size amplified a prediction error. Current execution already defaults to approximately $5,000 per entry, so that sizing correction should not be proposed as new work. However, exercises can expand a small premium position into large share exposure: observed AAOI/GRAB/U equity lifecycles carried roughly $78,000–$94,000 of cumulative basis. Ten confirmed worthless expirations cost **$26,936**. Validate fill-confirmed expiry handling and exercise exposure through the existing risk/governance path, rather than sizing risk solely from option premium.

6. **Require stronger evidence before increasing Intraday/SPY capital.** Their completed-position profit precision is about 16%, with negative realized expectancy in this period. Liquidity and holding-time hypotheses warrant controlled forward evaluation; adding model complexity or a winner-name whitelist is not supported.

These recommendations follow the current roadmap's accounting, execution lifecycle, risk governance, and research-validity work. **No trading policy, live state, model, order, or risk limit was changed.**

## 6. Evidence quality and reproducibility

- Read-only paper-broker download: **2,916 orders, 2,394 FILL activities, 1,476 filled orders**, 1,277 fee activities, ten expirations, eight exercises with eight corresponding stock deliveries. Every filled-order quantity and VWAP notional agrees with the activity stream within the declared numeric tolerances. Funding is excluded from profit.
- Inventory uses chronological, within-lifecycle weighted-average cost. Each partial exit uses only basis accumulated before that exit. Exercise premiums transfer into delivered shares; exercise is not fabricated as a zero-price loss. This is research economic accounting, not a tax-lot report.
- Source files were snapshotted or hashed. Final data include **2,705 candidate decisions within the broker-verifiable period**, 499 scored completed ordinary lifecycles, 470 underlying endpoint pairs, and 322 observed paths meeting the coverage requirement before the ordinary-cohort exclusions.
- Features use the prior completed session. Four-hour left-labelled bars become available at their completion, not at their labels. One-minute closes are unavailable until one minute after the left label. The SPY score join uses an exact-order postfill policy snapshot; that is weaker provenance than a pre-submit immutable prediction and is identified in the export.
- Underlying historical one-minute bars and archived bars supply outcome diagnostics. Archive bars win duplicate symbol/minute overlaps; earliest archive arrival wins within archive duplicates (zero such duplicates observed). Paths require at least 80% expected regular-session minute coverage. Missing bars are not imputed. Observed excursion can miss a true peak, and archived bars can arrive late; outcome prices are not assumed to have been available to the strategy in real time.
- No option-bar marks were used. Actual option fill returns correlate approximately **+0.74** with direction-adjusted underlying endpoint returns in the covered sample. This is a sanity check, not an option valuation model. Large daily discontinuities conservatively invalidate affected derived features/outcomes rather than receive an invented split factor. Current historical bars may contain revisions; exact historical feature parity cannot be certified without entry-vector archives.
- Model scores are not comparable across modules or unversioned deployment changes. Open positions create substantial maturity bias. Matched samples are small; 236 unpaired and 205 paired feature tests receive separate BH adjustments. Unpaired/paired feature p-values are exploratory and not cluster-adjusted. SciPy warned that some small matched samples use a normal approximation; no matched-feature significance claim is made. Cluster bootstrap score intervals use 1,000 resamples with seed 1909 and are withheld with fewer than five minority outcomes.
- Six focused accounting/time tests and 18 artifact checks passed. Additional final checks verify ledger sums, exports, temporal ordering, and matched-pair constraints. The open-inventory and cash discrepancies remain explicitly unresolved; the research is not a clean whole-account reconciliation certificate.

### Reviewable artifacts

- [Four-panel findings figure](analysis/findings.png) · [PDF](analysis/findings.pdf)
- [Every profitable exit](analysis/all_profitable_exits.csv) · [Every losing exit](analysis/all_losing_exits.csv)
- [Every profitable ordinary lifecycle](analysis/all_profitable_lifecycles.csv) · [Other ordinary lifecycles](analysis/all_nonprofitable_lifecycles.csv)
- [All lifecycle features, including open/transferred positions](analysis/trades.csv)
- [Similar-score pairs](analysis/similar_score_pairs.csv) · [Feature tests](analysis/feature_separation.csv) · [Matched-feature tests](analysis/matched_feature_separation.csv)
- [Score discrimination](analysis/score_discrimination.csv) · [Score buckets](analysis/score_buckets.csv)
- [Execution comparisons](analysis/execution_comparison.csv) · [Fresh exit costs](analysis/fresh_exit_cost_summary.csv)
- [Candidate precision/recall](analysis/candidate_precision_recall.csv) · [Underlying translation](analysis/underlying_translation.csv)
- [Accounting checks](run3/checks.json) · [Ledger audit](run3/ledger_audit.csv)
- [Scripts](../../scripts/profit_investigation/) · [Reproduction notes](reproduction.md)
