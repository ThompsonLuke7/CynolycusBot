# Paper Account Post-Mortem, 2026-07-09 to 2026-11-02

Written 2026-11-03. Paper account only. All counterfactuals are **in-sample, on one
3.5-month path of one account**. They answer "what would have happened here", not
"what will work".

**Headline:** $1,000,000 deposited 07-09 (the day the user reset the paper
account; it is the only deposit) became $373,033 at the 11-02 21:05 ET snapshot
(**-62.7%**). Over 07-15 to 11-02, SPY was -0.4%, QQQ -5.1%, IWM -5.6%.

**Two equity series disagree.** The broker's daily portfolio history (and
`last_equity`) reads $305,066 at the 11-02 close (-69.5%). The two agreed within
a few $k until **08-14**, the expiry day of the four auto-exercised calls. On
that day the history steps down $76k against the live balance, and it has stayed
$50-90k below ever since. This write-up uses the live balance. It equals cash
plus positions exactly, and its day-to-day changes reconcile with every fill
(see Method). The history's step is unexplained on the broker side.

## Method (01_build_ledger.py)

For each pair of consecutive daily snapshots, each symbol's P&L is the change in
its market value plus its fill cash in that interval. The interval rows sum to
the account's equity change. Over all 73 intervals the residual is **$635 in
total** (largest single interval: $136).

- Fills come from the broker's full FILL history (4,207 fills). The local journal
  only starts 09-14.
- Auto-exercises (OPEXC) are added as pseudo-fills at the strike. The broker
  books them with `net_amount: 0`, and leaving them out had hidden $374k of
  stock purchases.
- Every symbol run from flat to flat is one **episode**: 1,309 in total.
- Each episode is attributed to a module by its **entry** order:
  1. a `pending-open` log line, or the 30m swing's submit line;
  2. otherwise, the module's own audited `order_plan`;
  3. otherwise, the runner's stdout block;
  4. otherwise, a closed-trade row for the symbol at the right time.
- **Unattributed: 114 small episodes, -$4.6k.**
- One mark artefact is corrected: ENHA's 10-29 snapshot priced the post-split
  share against the pre-split quantity (+$38.7k phantom). See
  `apply_split_to_claim.py`.

## Where the money went (02_attribution.py)

| Module | Equity | Options | Total | Episodes | Win % |
|---|---:|---:|---:|---:|---:|
| HTF swing | -110,189 | -52,996 | **-163,185** | 324 | 25 |
| Dealer ranker | -21,809 | -110,365 | **-132,174** | 120 | 26 |
| Momentum expansion | -32,090 | -89,671 | **-121,761** | 135 | 24 |
| Multi-ticker swing 30m | - | -107,102 | **-107,102** | 196 | 35 |
| Meta ranker | -68,385 | -15,661 | **-84,046** | 137 | 29 |
| Intraday structure | - | -10,900 | -10,900 | 200 | 16 |
| SPY daytrader | - | -1,236 | -1,236 | 82 | 18 |
| Unattributed | - | -4,579 | -4,579 | 114 | 16 |

- **Every module lost money.** The average episode returned **-12.7% of its
  cost**, and 24.5% of episodes won.
  - Equity episodes averaged -7.5% (31% win).
  - Option episodes averaged -16.2% (20% win).
- Options lost in every days-to-expiry bucket at entry:

  | DTE at entry | P&L | Mean return |
  |---|---:|---:|
  | 0-2 | -$38.7k | -11% |
  | 3-7 | -$59.3k | -18% |
  | 8-14 | -$61.2k | -32% |
  | 15-21 | -$75.1k | -18% |
  | 22-35 | -$124.4k | -21% |
  | 36+ | -$33.6k | -17% |

- **Market versus selection.**
  - The book's beta was about 2.65 to IWM (correlation 0.61) and about 2.2 to
    QQQ (0.75).
  - Against IWM, roughly **-$216k is market exposure** (beta × gross × IWM),
    and **-$409k is selection and costs**.
- **Leverage amplified the losses; it did not cause them.** Gross exposure went
  from 0.23x to 2.38x and was above 1x from 08-07 onward. Per dollar invested,
  returns were negative at every leverage level:

  | Gross leverage | Mean return per dollar invested, per interval |
  |---|---:|
  | below 1x | -1.10% |
  | 1-1.5x | -0.20% |
  | 1.5-2x | -0.93% |
  | 2x and above | -1.05% |

- **Losses are spread out.** The worst 10 episodes are 18% of the net loss, and
  the worst 100 are 71%.

## Q1. Were we overleveraged, and did the broker dump us?

**Overleveraged, yes. Dumped, no.**

- Only **2 of 1,193 sell orders ($3,160)** were not sent by our own modules.
  Both were expiring options that the broker closed at 15:30 on expiry day. There
  was never a margin-call liquidation.
- The damage that came from the account's own mechanics:
  - **Auto-exercise.** Eight options held in the money into expiry delivered
    **$374k of stock**. On 08-14 alone, GRAB, U, FIG and SMCI delivered $267k,
    30% of equity. The P&L on the delivered shares was small (+$2.6k, because
    they were sold the next morning), but the margin was consumed.
  - **Options held into expiry:** 24 episodes, -$37.5k. 16 expired worthless
    (-$32.5k).
  - **Sizing errors.** Meta bought **SNDK for $167k on 07-13** (-$49.9k, about
    33x the $5k size), and HTF bought AAOI for $89k (-$17.5k). A $10k-per-name
    cap would have saved +$57k, all of it in the first half.
  - **Unmanaged sessions.** Intervals containing a session with no server ran
    -$74k. That figure includes market moves, so not all of it was preventable.

## Q2. Was there bad news we didn't react to? (04_news_reaction.py)

**Not in a way that was measurable or robust.**

- Strongly negative FinBERT news on a held name is common. 593 of 1,250 episodes
  had at least one such record (negative minus positive ≥ 0.5) during the hold.
- The rule tested: exit at the next snapshot after the first such record. It was
  applied to all episodes, winners included. The placebo exits the same episodes
  at a random time inside their hold.

| Clock | Saved by the rule | Saved by random exit | News "skill" |
|---|---:|---:|---:|
| Pipeline (when we actually had the record) | +$181k | +$85k | +$96k |
| Same, excluding one batch (08-17 00:09) | +$92k | +$62k | **+$30k** over 470 episodes (about $64 each) |
| Real-time feed (timestamp + 15 min) | +$98k | +$152k | **-$54k** |

- 21% of the pipeline-clock triggers came from a single weekly collect, and the
  snapshot after it happened to fall just before 08-18, the worst day.
- **Buying into bad news** (a negative record available in the 72 h before
  entry) did no detectable harm: +$167 per episode (SE $137, minimum detectable
  difference about $384).
- **The pipeline cannot react intraday.** News is collected in nightly or weekly
  batches, so a negative record is available a median **16 hours** after its
  timestamp.

## Q3. Could connecting the dots have de-risked us? (03_counterfactuals.py, 03b_placebo.py)

Each rule decides at each entry, using only the counterfactual account and
regime data available at that time. The rule-free replay reproduces the actual
equity ($373,035 against $373,033).

**The placebo is the key result.** Because the per-trade edge was negative, any
rule that trades less loses less. Each rule is therefore compared with refusing
the **same number** of entries at random (30 seeds).

| Rule | Final equity | Saved vs actual | Random refusal of the same count (5-95%) | Beats random? |
|---|---:|---:|---:|---|
| Actual | 373,035 | - | - | - |
| Exits-only freeze once DD ≥ 10-15% | 903,260 | +530,227 | +462k .. +525k | **yes, marginally** (+$40k) |
| Exits-only freeze once DD ≥ 20% | 744,617 | +371,584 | +299k .. +411k | no |
| Kill switch (liquidate everything) at DD ≥ 15% | 848,610 | +475,577 | +441k .. +518k | no |
| No options (equity episodes only) | 765,053 | +392,019 | +286k .. +404k | no (+$54k, inside the band) |
| Gross cap 1.0x | 567,796 | +194,762 | +161k .. +276k | no |
| Gross cap 1.5x | 446,346 | +73,313 | +60k .. +151k | no |
| Freeze while IWM is below its 50-day average | 680,660 | +307,627 | +292k .. +383k | no |
| Freeze while breadth_z < -1 | 554,084 | +181,051 | +147k .. +268k | no |
| Freeze while spy_rv20_z > 1 | 373,035 | 0 (never fired) | - | - |
| Always short IWM at 2.65x gross (beta-neutral) | 589,267 | +216,233 | not a refusal rule | removes the market part |
| Gross 1x + $10k name cap + no options | 833,809 | +460,775 | +287k .. +408k | yes (driven by the name cap) |

- **Regime signals had little to work with.**
  - `spy_trend_state` read "uptrend" every day of the period.
  - `spy_rv20_z` never rose above 1.
  - `risk_appetite_z`, `liquidity_stress_z` and `credit_risk_z` were **frozen
    from 08-06**: the IEI bars had stopped (fixed 2026-11-03, see below).
- **The overlays mostly show no timing skill.** Gross caps, regime freezes, the
  kill switch and the options ban all fall inside the random band.
- **The one control that changes the outcome** is a performance kill switch.
  An exits-only freeze at a 10-15% live drawdown stops a strategy set whose live
  results contradict its research. Here it would have triggered in late July and
  ended at -9% instead of -62%. Its advantage over random is small: it works
  mostly because it stops early.

## Preventable versus not

- **Market beta, about -$216k.** Timing would not have avoided this. Holding
  less beta or hedging it would have.
- **Selection after costs, about -$409k. This is the root cause.** It is a model
  and strategy problem, not a risk-overlay problem. All seven modules had
  negative live expectancy, and options were worse than equities at every
  days-to-expiry bucket.
- **Specific operational losses, about -$100-160k, depending on overlap.**
  - Oversized SNDK and AAOI entries: -$67k.
  - Options held into expiry: -$37.5k.
  - Auto-exercise: no direct P&L, but it consumed margin.
  - Outage intervals: -$74k gross, which includes market moves.

## Salvage, applied 2026-11-03

- **The exits-only freeze is on** (`Data/runtime/trading_controls.json`).
  - Every buy that opens exposure is refused at the broker client.
  - Exits, stops and data collection continue. Every module still writes its
    `signal_decision` / `order_plan` audits, and the two-sleeve shadow tracker
    keeps running, so the forward record of what each module *would* have done
    keeps growing without spending.
- **Margin.** The cushion was 3.9% on 11-03 01:31: equity $371.8k against
  $357.2k maintenance. Exits widen it over time. A broad drop of about 1.7% in
  the long book would breach it. Trimming the most volatile low-priced names is
  the faster fix; that is your call.
- **When entries reopen, on this account or a new one,** the same gate enforces
  `max_gross_leverage` and `min_margin_cushion_pct`. Treat these as risk limits,
  not alpha: the placebo shows a cap does not create edge.

## Caveats

- One path and one account. Every threshold is in-sample, and results are
  reported as a grid.
- Entries the broker refused for buying power cannot be added back, which
  flatters rules that free buying power.
- Marks are daily, after the close. Exit costs are assumptions: 0.25% of market
  value for equities, 10% for options.
- Episodes are per symbol, because the broker nets shared names across modules.

## Next (roadmap)

Do not reopen entries on a new account until each module clears a forward
expectancy test on its own audited signals. The frozen period produces exactly
that record. Keep the option route closed until it clears its round-trip cost
hurdle.

## Files

- `00_fetch_broker_history.py`: raw broker FILL and non-trade history, read-only.
- `01_build_ledger.py`: the interval × symbol ledger, episodes and attribution.
- `02_attribution.py`: the descriptive breakdowns.
- `03_counterfactuals.py`, `03b_placebo.py`: rule replays and the random-refusal
  control.
- `04_news_reaction.py`: news exits and the bought-into-bad-news test.
- Outputs are in `Data/research/account_postmortem_20261103/`:
  `ledger_*.parquet`, `counterfactuals*.csv`, `news_exit_*.csv`, and the raw
  broker JSON (immutable).
