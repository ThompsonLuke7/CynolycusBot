# Rank depth, options, and where the sample runs out

2026-09-08. Supersedes the precision table in `22_selectivity_finding.md`.

## 0. Correction to 22

`22_selectivity_finding.md` reported HTF top-1 precision of 17.6% (3.5x base) and
momentum at 0.17x. Both were wrong, for one reason: forward MFE was measured from
the **prior close**, which credits a ranker with the gap between the decision bar
and the first price it could actually buy. HTF preferentially ranks names that
have already gapped, so it collected that gap as if it were foresight.

Recomputed from the next session's open -- the first executable price, the same
one the P&L below uses:

| module | base | p@1 | p@2 | p@3 | p@5 | p@10 |
|---|---|---|---|---|---|---|
| momentum_expansion | 5.0% | 7.5% | **10.4%** | 8.5% | 6.0% | 4.0% |
| meta_ranker | 5.0% | **7.9%** | 8.7% | 8.5% | 8.6% | 6.8% |
| multi_ticker_swing_htf | 5.0% | 4.4% | 5.9% | 3.9% | 4.7% | 4.3% |
| dealer_ranker | 5.0% | 0.0% | 0.0% | 0.0% | 1.0% | 1.8% |

HTF top-1 is *below* the base rate. Momentum, reported in 22 as having no signal
at any depth, has the highest precision at the top; the 0.17x figure was its
top-10, which is genuinely bad.

Scripts: `scripts/rank_depth/reconcile_precision.py`.

## 1. Realized share P&L by rank depth

68-72 decision bars per module, 2026-07-01..2026-08-28, all long. Entry = open of
the first session after the signal is available; exit = close H trading days
later. Reported as **excess over an equal-weight draw from the same eligible
universe on the same decision days** -- that control returned +0.52% (8d),
+0.76% (10d), +1.39% (15d), so the raw numbers are not being flattered by drift.
Intervals from a decision-day block bootstrap.

| module | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| meta (8d) | +1.56% | +0.42% | -0.79% | -1.92% | -2.44% |
| momentum (10d) | +0.54% | +2.66% | -0.94% | -2.89% | **-5.49%** * |
| swing_htf (10d) | -4.16% | -2.05% | -1.82% | -1.76% | -1.76% |
| dealer (10d) | +0.37% | +0.99% | +1.79% | +0.78% | +0.04% |

`*` p < 0.05. **No positive cell is significant.** The significant cells are all
negative: momentum top-10 (p<0.001 at 8/10/15d, -9.19% at 21d) and HTF top-1 at
15d/21d (-7.17%/-9.48%, p=0.012/0.007). Taking top-1 would not, on its own, have
made the system profitable.

Scripts: `share_grid.py`, `control.py`, `significance.py`, `combined_grid.py`.

## 2. But the ORDERING is real -- permutation test

The grid above scans ~240 cells, so a lone p<0.05 in it means nothing. The
decision actually rests on a narrower question with a null that removes every
other explanation:

> NULL: the module's rank ordering within a decision bar is arbitrary.

Shuffle ranks **within each bar** and recompute the top-k mean. Pick set,
decision days, universe drift and sector mix are all held fixed by construction;
only the ordering moves. 20,000 permutations.

| module | filter | hold | k | top-k | shuffled | edge | p |
|---|---|---|---|---|---|---|---|
| momentum | all | 10d | 2 | +3.42% | -4.46% | **+7.88pp** | <0.0001 |
| momentum | all | 15d | 2 | +1.78% | -5.63% | **+7.41pp** | <0.0001 |
| momentum | all | 10d | 1 | +1.31% | -4.46% | +5.76pp | 0.003 |
| meta | all | 15d | 1 | +3.90% | -1.67% | +5.56pp | 0.004 |
| meta | all | 8d | 1 | +2.08% | -1.92% | +4.00pp | 0.011 |
| swing_htf | all | 10d | 1 | -3.40% | -0.97% | -2.43pp | 0.911 |
| dealer | all | 10d | 3 | +2.79% | +1.65% | +1.14pp | 0.117 |

**Momentum's ordering is significant in 9 of 9 cells** (holds 8/10/15 x k 1/2/3),
most at p<=0.003 -- that survives any multiple-comparison correction. Meta is
significant in 5 of 9. HTF is significant in the *wrong direction* at k=1. Dealer
is null.

Both statements in sections 1 and 2 are true and they are not in conflict. The
pick set as a whole sits below the universe; the ordering lifts the top of it
back to roughly the universe. So cutting k is worth ~4-8pp per trade **relative
to the module's own picks**, which is a real improvement to the module -- and is
not the same thing as a demonstrated profitable strategy.

Script: `scripts/rank_depth/permutation.py`.

## 3. Momentum is not redundant with HTF

Believed ~90% correlated. Measured on the actual ranked picks:

| pair | overlap @10 | @3 | @1 |
|---|---|---|---|
| momentum vs swing_htf | **2.5%** | 3.6% | 3.1% |
| momentum vs meta | 5.7% | 1.6% | 1.6% |
| swing_htf vs meta | 22.3% | 10.6% | 0.0% |

They are separate books. Whatever was 90%-correlated, it was not these picks.

Momentum also has the widest score spread from rank 1 to rank 10 (29.8%
relative), and the steepest return decay (+1.31% at k=1 -> -7.44% at ranks 6-10).
Its ranking is informative and monotone; the take-10 policy is what destroys it.

**Meta is the opposite**: score spread rank 1 to rank 10 is 0.011 on a ~0.99 base
-- **1.1% relative**. Meta is saturated at the ceiling and barely distinguishes
its own top ten. Independent argument for the pending retrain.

The `df_1d=None` momentum defect recorded in earlier notes is **fixed**:
`live_feature_panel_4h.py:176` and `live/runner.py:396` both load daily bars.

Scripts: `overlap.py`, `profile_picks.py`, `score_vs_liquidity.py`.

## 4. dealer_ranker: it is the option wrapper

Realized, by module and route, from the broker fill stream:

| module | route | n | sum P&L | mean ret | win% | median hold |
|---|---|---|---|---|---|---|
| multi_ticker_swing_htf | equity | 22 | **+$7,879** | +0.13% | 59% | 43 d |
| momentum_expansion | equity | 6 | -$1,865 | -0.38% | 67% | 43 d |
| meta_ranker | equity | 9 | -$53,719 | -26.8% | 33% | 27 d |
| **dealer_ranker** | **option** | 27 | **-$69,916** | **-60.5%** | **11%** | **2 d** |
| momentum_expansion | option | 17 | -$37,882 | -46.6% | 6% | 3 d |
| multi_ticker_swing_htf | option | 17 | -$11,647 | -33.9% | 6% | 3 d |

dealer_ranker ranks **mega caps** -- $56 median price, $252M median daily dollar
volume, 4.2% median daily range -- and then buys short-dated options on them and
holds 2 days. A 4%-range name cannot travel far enough in 2 days to pay for
premium. Its *share* picks are mildly positive (+1.79% excess at top-3, 10d, not
significant). The wrapper is the entire loss.

meta's equity -26.8% is not a data artifact: SION really fell 91% and TENX 91%
inside the window, and the -39% premium stop (calibrated for option premium) is
being applied to share positions, where it is far too wide to protect anything.

## 5. Options at the top of the ranking

Real historical Alpaca option bars, ~30-60 DTE monthly calls, entry at the
option's open on the same session the share grid buys, exit at its close H days
later, **both dates required to have a real bar** (a missing bar is a day nothing
traded and its price would be a stale print). Costs = the measured live spread
for that underlying, charged once as a round trip.

Validation first, per the standing rule from the 2026-07 retraction:
corr(option return, underlying return) = **+0.63 / +0.84 / +0.79 / +0.84** at
5/8/10/15d; identical entry/exit prices 0-2.4%. **PASS** -- unlike the retracted
study, these prices track value.

Aggregate, top-3, all four modules:

| hold | n | underlying | option gross | option net | win% | median net |
|---|---|---|---|---|---|---|
| 5d | 83 | -5.83% | -15.1% | -36.3% | 12.0% | -47.7% |
| 8d | 79 | -7.71% | -32.7% | -49.7% | 10.1% | -59.2% |
| 10d | 83 | -6.20% | -21.9% | -42.6% | 16.9% | -65.3% |
| 15d | 76 | -1.71% | -12.8% | -36.5% | 22.4% | -59.9% |

The 15d row is the cleanest statement of the option problem. The underlying was
roughly flat (-1.71%) and the option still lost **-12.8% gross** -- that is the
premium decay, before any friction. The measured spread then took another
**23.6pp**. So the round-trip hurdle is roughly **35% of premium**, and the
ordering edge established in section 2 is 4-8pp of *underlying* return. It does
not come close.

There **is** two-sided convexity, exactly as expected: worst contract -98.0%,
best +383.3%, gross positive in 29% of cases. It is not enough.

### The one positive cell did not replicate -- RESOLVED 2026-09-08

swing_htf top-3 at 15d first read gross +75.1%, **net +21.0%**, 50% win, n=22 --
on the 2026-08-21 expiry alone, with LCID appearing three times and half the sum
in a single contract. It was recorded as one observation, not believed.

Two further expiry cycles were then priced through Schwab (section 6), and it
does not survive:

| expiry | n | gross | net | win% | median net |
|---|---|---|---|---|---|
| 2026-08-21 | 22 | +75.1% | **+21.0%** | 50.0% | -1.1% |
| 2026-09-18 | 27 | -55.7% | **-62.9%** | 14.8% | -87.3% |
| 2026-10-16 | 14 | +9.9% | **-18.9%** | 14.3% | -19.1% |
| **pooled** | **63** | **+4.6%** | **-23.8%** | **27.0%** | -- |

**Retracted.** With 944 contract-hold observations over 205 contracts and three
expiry cycles, validation still passing (corr +0.73..+0.88), every module, every
hold and every cycle is negative after measured cost:

| expiry | 5d | 8d | 10d | 15d |
|---|---|---|---|---|
| 2026-08-21 | -36.3% | -49.7% | -42.6% | -36.5% |
| 2026-09-18 | -32.8% | -40.7% | -38.9% | -56.5% |
| 2026-10-16 | -29.8% | -33.4% | -36.8% | -42.9% |

HTF remains the least bad and is the only module whose options are roughly
break-even GROSS (+4.6% pooled at top-3/15d) -- the convexity is real. The spread
is what takes it, which is the same conclusion section 5 reached, now on 3x the
data instead of one cycle.

Scripts: `option_grid.py`, `scripts/thesis_test/fetch_contract_paths.py`
(patched to carry rank and accept `--max-rank`).

## 6. Why there is only one expiry cycle -- and how to get more

Of 420 fetched contracts, **283 returned zero bars**: every 2026-09-18 and
2026-10-16 expiry. Not illiquidity -- a hard failure:

```
HTTP 403 {"message":"OPRA agreement is not signed"}
```

Expired contracts return full history; **currently-listed contracts are blocked**.
Signing the OPRA agreement in the Alpaca dashboard removes the cap.

**But Alpaca is not the only source, and the repo already had the other one.**
Schwab is the exact complement -- it serves LIVE contracts and returns nothing
for expired ones. Measured on LCID:

| contract | Alpaca | Schwab |
|---|---|---|
| LCID260821C00006000 (expired) | 37 bars | 0 candles, `empty=True` |
| LCID260918C00006000 (live) | **403 OPRA** | **45 candles** |

274 of the 290 blocked contracts (94%) were then priced through Schwab, which is
what turned one expiry cycle into three and retracted the finding above.

**Symbol-format trap.** Schwab requires the underlying root LEFT-JUSTIFIED IN SIX
CHARACTERS, space-padded: `'LCID  260918C00006000'` returns 45 candles,
`'LCID260918C00006000'` returns **HTTP 200 with an empty candle list**. The
unpadded form does not error -- it looks exactly like a contract that never
traded. `core/API/Schwab_API/option_bars.py::to_schwab_symbol` exists so this is
not rediscovered.

Two local assets change what is possible without waiting:

1. **`Data/options_history/bars/1Day/`** -- 5,280 contract-expiry files, **166
   expiry dates from 2024-02-02 to 2026-09-18**, 665 tickers, full daily OHLCV
   per contract, 738 MB, already on disk. Roughly 30 monthly cycles instead of 1.
2. **`meta_ranker_matrix_research.parquet`** -- 1.58M rows x 105 cols,
   **2022-11-14 to 2026-05-14**; the live matrix covers 2025-08-04 to 2026-09-08.

The binding constraint is neither prices nor features: it is that **ranked
signals only exist from 2026-07-01**, because they were reconstructed from live
order history. 63-72 decision bars is the whole record.

**Caveat that governs how the backfill must be done.** The live meta model was
trained 2026-06-20 on the research matrix, which ends 2026-05-14. Simply scoring
the current model back over 2022-2026 is *in-sample* and would measure
memorisation. The legitimate version is walk-forward: retrain per window
(`colab_competition.py` already does 18-month train / 4-month test) and keep only
each fold's out-of-sample predictions. That yields ~3 years of honest ranked
signals which can then be joined to the local option cache.

## 7. What this supports

1. **Cut k for momentum first, not meta.** Momentum's ordering is the only one
   significant in every cell without a post-hoc filter, worth ~+7.9pp per trade
   at k=2 versus its own picks. Meta second (5 of 9 cells). Do not cut k for HTF
   -- its ordering runs backwards at k=1.
2. **Stop routing dealer_ranker through options.** -60.5% mean over 27 trades on
   2-day holds of mega-cap premium; its share picks are not the problem.
3. **Do not scale options on this evidence.** ~35% round-trip hurdle versus a
   4-8pp ordering edge, now confirmed across three expiry cycles and 944
   observations. The one positive cell was retracted on replication.
4. **Sign the OPRA agreement** for a single-source path; meanwhile Schwab already
   covers live contracts and Alpaca covers expired ones, so option research is
   no longer blocked.
5. **Backfill ranked signals walk-forward.** The largest available increase in
   statistical power in the system: ~63 decision bars -> ~3 years.
6. **Remove the -39% premium stop from equity positions.** It is an option-premium
   number being applied to shares.

Nothing here has been wired into live code. Every number above is measurement.

---

# 8. Replication on 3.5 years (2026-09-08, added after sections 1-7)

Sections 1-2 rest on 67-72 decision bars. `mom_score` in the Meta research matrix
is momentum's **walk-forward out-of-fold** prediction with a 21-day embargo
(`build_meta_ranker_matrix.py:23,46-49`), which is the same ranking over
**1,739 decision bars, 2022-11-14 to 2026-05-13** -- 26x the sample, free, no
retraining. Joined to daily bars with `merge_asof(direction="forward")` on
decision_day + 1 day, so entry is the first session strictly after the bar.
`ret_10` present on 100% of rows.

## 8.1 The first read was contaminated

Raw, it reported top-1 excess of **+16.23%** at a 10-day hold (p<0.001). That is
~26x the live estimate and it contradicts momentum's own documented walk-forward
calibration (top-5/10 lift 1.06-1.12, "thin"). Checked before reporting:

| check | result |
|---|---|
| placebo (shuffle score within bar) | real +7.44%, shuffled **-0.15%** -- PASS |
| survivorship (matrix tickers missing bar files) | 0 of 1,080 -- no join-side loss |
| entry alignment | median lag 1 calendar day, entry always after the bar |
| **concentration** | **FAIL** |

The failure: **`Data/shared/bars/1d` carries unadjusted corporate actions.**
16 top-1 observations are WOLF around its 2025-09 Chapter 11 emergence, with
returns to **+2,189%**. They alone move 2025's top-1 mean to +47.3% against a
+5.5% median. This is the same defect class already guarded in live execution
(`live_4h_exec.py:140`, the TENX/SION 10:1 split) -- but nothing guards the
research cache.

## 8.2 Cleaned

| k | raw | \|ret\| <= 200% | winsorized 1/99 |
|---|---|---|---|
| 1 | +16.23% | +5.36% | +2.89% |
| 2 | +12.26% | +5.94% | +3.06% |
| 3 | +9.74% | +5.45% | +2.98% |
| 5 | +7.74% | +4.88% | +2.76% |
| 10 | +5.43% | +3.96% | +2.36% |

All still p<0.001 on 1,739 bars. But the depth GRADIENT -- the thing the top_n
change rests on -- shrinks from dramatic to modest, and the level is
tail-dependent: winsorizing halves it.

By year (200% cap, k=1): 2022 **-6.80%**, 2023 +3.17%, 2024 +8.55%, 2025 +8.29%,
2026 -0.68%. Not stationary. Both ends of the sample are negative.

## 8.3 What replicates, and what does not

Paired k-vs-k, 1,739 bars, 200% cap:

| pair | 5d | 10d | 15d |
|---|---|---|---|
| k=2 vs k=3 | +0.02 | +0.49* | +0.07 |
| k=2 vs k=10 | +0.84* | +1.98* | +1.90* |
| k=3 vs k=10 | +0.82* | +1.49* | +1.85* |
| k=1 vs k=2 | -0.24 | -0.40 | **-1.84*** |

**Replicates:** shallow beats top-10, at every hold, both samples. This is the
finding.

**Does not replicate:** the live sample's "k=2 beats k=3 at every hold"
(p=0.010/0.003/0.022) is a coin flip on 26x the data. Its +7.88pp magnitude does
not either -- the honest number is **+1.5 to +2.0pp** at a 10-day hold.

**New:** k=1 is *worse* than k=2 at 15 days (p<0.001). Do not go to 1.

## 8.4 The guard, and the corrected numbers

`core/corporate_actions.py` (+ `core/tests/test_corporate_actions.py`, 5 tests
including the real WOLF bars as a fixture). It FLAGS, it does not adjust prices
and does not label a flag "a 10:1 split" -- same reasoning as
`live_4h_exec._implausible_mark_move`: at these magnitudes a tolerance loose
enough to match a real split matches everything else, and a confident wrong
diagnosis is worse than a raw ratio someone can check against a real source.

The discriminator that matters, reported as a diagnostic rather than applied as a
silent filter:

* **real explosive move** (biotech readout, squeeze): price x3, volume x20 ->
  dollar volume way up. This is the tail a momentum study exists to measure and
  it must NOT be screened out.
* **recapitalisation** (WOLF): price x15, volume /13 -> dollar volume ~unchanged.
  Share count moved, value did not. WOLF's ratio is 1.18.

**Cache-wide scan:** 187 sessions across 148 tickers of 4,076 files; 75 have a
dollar-volume ratio under 3. The largest are bankruptcy emergences -- GPOR, CBL,
LINE, WW, BIOA.

**Corrected OOF table** (the earlier draft of this section used an ad-hoc
`|ret| <= 200%` cut, which also removed *genuine* large moves and so understated
the level; the guard drops only **0.006%** of observations):

| k | 5d | 10d | 15d |
|---|---|---|---|
| 1 | +3.66% | +6.71% | +8.84% |
| 2 | +3.53% | +6.59% | +9.99% |
| 3 | +3.31% | +5.97% | +9.30% |
| 10 | +2.21% | +4.29% | +6.46% |

| pair | 5d | 10d | 15d |
|---|---|---|---|
| k=2 vs k=3 | +0.22 | +0.62* | +0.68 |
| **k=3 vs k=10** | **+1.10*** | **+1.68*** | **+2.85*** |
| k=1 vs k=2 | +0.12 | +0.16 | -1.08 |

## 8.5 Why 0.006% contamination moved the headline 3x

| population | contamination rate | over-representation |
|---|---|---|
| all observations | 0.0062% | 1x |
| top-10 picks | 0.1495% | **24x** |
| top-3 picks | 0.4217% | **68x** |
| top-1 picks | 1.1501% | **186x** |

The ranker selects for extreme recent price action, and an unadjusted
recapitalisation is indistinguishable from that. Rare in the cache, concentrated
exactly where the study looks. This is the general lesson: **a defect's overall
rate does not bound its effect on a selective study.**

## 8.6 Action taken

`RANKING_CONFIG["top_n"]` set to **3** (not the 2 the live sample alone implied):
k=2 and k=3 are statistically indistinguishable in the large sample, the return
distribution is tail-driven, and k=1 is not better than k=2 anywhere.

## 8.7 Carried forward

**Levels are not trustworthy; differences are.** k=10 still shows +4.29% excess
at 10 days, which is implausibly large for a top-10 book. Of 4,076 bar files
exactly **2** end before 2026, so delisted names are largely absent and the OOF
universe churns 1-3 names a year against a real delisting rate of 4-8%. That
inflates every level. The k-vs-k comparisons are within-bar and unaffected by
which names survived to be in the cache, which is why only the differences were
used to set the parameter.

Getting a survivorship-clean universe is the next data-integrity item; it is a
bigger job than the corporate-action guard because it needs point-in-time
listing data the repo does not currently hold.

---

# 9. Survivorship, HTF, and live exposure (2026-09-10)

## 9.1 Correction to 8.7

8.7 said the k-vs-k differences are "unaffected by which names survived". That is
too strong. Section 8.5 demonstrated the mechanism that breaks it: a selective
ranker concentrates defects at the top (corporate actions 186x over-represented
at k=1). If top picks are also disproportionately the distressed names that later
delisted, pruning dead names removes more from the top than from the deep ranks
and inflates the difference. Differences are LESS exposed than levels, not immune.

## 9.2 By-year diagnostic

If pruning drives the gap, it should shrink toward the present (least time to be
pruned). 10-day hold, guarded:

| year | mom k3 vs k10 | htf k3 vs k10 |
|---|---|---|
| 2023 | +0.92 | -0.30 |
| 2024 | +0.90 | +1.03* |
| 2025 | **+4.78*** | +1.47* |
| 2026 (Jan-May) | **-1.76*** | **+2.94*** |

HTF's gap GROWS toward the present -- the opposite of the survivorship signature.
Momentum's is carried by 2025 and reverses significantly in early 2026, while the
live Jul-Aug 2026 spine shows +3.98pp (p=0.002). Momentum's depth effect is real
on the pool but not stationary.

## 9.3 HTF on 1,703 OOF bars -- contradicts the live study

| pair | 5d | 10d | 15d |
|---|---|---|---|
| k=1 vs k=2 | +0.29 | +0.76* | +1.13* |
| k=3 vs k=10 | +0.71* | +0.96* | +1.40* |

Monotone k1>k2>k3>k5>k10. Live (68 bars) had HTF top-1 significantly negative and
the ordering backwards. Candidate explanations, untested: the live universe is
~2,900 names including thin microcaps (live HTF top-1 median $6.79 / $6.7M ADV)
while the OOF pool is ~1,080 more established names; small-sample noise. No HTF
parameter change is justified while the two disagree.

## 9.4 Survivorship-clean universe -- feasibility

* Prices for dead companies ARE available: Alpaca returns bars for SIVB (to
  2023-03-09), BBBY, FRC, TUP up to their last trading day.
* The MEMBERSHIP list is what is missing. Alpaca's inactive assets (19,183; 2,867
  non-OTC) do not carry delisted names under their trading symbols (none of SIVB,
  BBBY, FRC, SBNY, WE, RAD, EXPR, GOEV, TUP, BIG present -- renamed on delisting,
  e.g. SIVB -> SIVBQ); a 60-symbol sample yielded 10 with 2022-26 history, mostly
  funds/ADRs, zero failure-shaped.
* The repo keeps no history of its own: `Data/**` is gitignored and
  `build_shared_universe` (`core/shared_universe/universe.py:146`) overwrites
  `shared_universe.csv` in place.

So a backward reconstruction needs an external point-in-time delisting source
(SEC EDGAR Form 25 is the free candidate; untested here). Going forward is
trivial: write a dated, immutable snapshot on every build.

## 9.5 Live exposure to the corporate-action defect

Live momentum reads the same unadjusted cache (`RAW_1D_DIR`). Checked every live
audit against 70 flags since 2026-05-01: one hit -- HTF targeted STI 27 days
after a 5x gap, and meta traded it. STI's dollar-volume ratio is 3,162 (volume up
624x), i.e. a real move with real participation, not a recapitalisation. No live
module ranked a recap-shaped name. The flag table also catches known leveraged
ETF reverse splits (TZA, TECS, SOXS 2026-07-15; MSTU 2026-08-24; ratios ~0.2).
Exposure is low in practice but unguarded.

---

# 10. Shipped: corporate-action screen and universe snapshots (2026-09-10)

## 10.1 Guard fix -- it could not see forward splits

The first `core/corporate_actions.py` flagged `|open/prev_close - 1| >= 300%`.
A down-gap can never exceed -100%, so it could not see a forward split at all.
The threshold is now on the RATIO in either direction (>= 4x or <= 1/4x). The
cache rescan went from 187 to **243 flags over 189 tickers**: 54 down-gaps it
had missed (TENX, SION, KLAC 2026-06-03, the Vanguard ETF splits 2026-04-21).

## 10.2 Organic vs not

`organic` = up-gap on >= 5x trailing median share volume. 161 up-gaps are
non-organic, 28 organic; the volume distribution is empty between 3.11 and 5,
so the threshold is not a fine-tuned value. Down-gaps are never organic (a
forward split multiplies share volume just as a real collapse does). Research
masks the 215 non-organic flags and keeps the 28 organic -- those are real returns.

Corrected OOF numbers with both directions masked (momentum, 10d): k=3 vs k=10
+1.66pp (was +1.68), 15d +2.82 (was +2.85). Down-gaps almost never sit inside a
top-10 momentum window, so the finding is unchanged.

## 10.3 Live entry screen

`core.live_4h_exec.corporate_action_screen` + `build_mixed_plan(...,
corporate_action_fn=)`: a NEW entry is refused when the shared 4H cache shows a
non-organic flag within the last `ENTRY_LOOKBACK_SESSIONS` = 20 sessions. Logged
to `contract_selection` with reason `corporate_action_suspect` and the raw
diagnostic, so it reaches every module's existing order_plan audit. Held
positions are untouched (the mark guard owns those). Applies to all four 4H
modules (meta, momentum, HTF, dealer) with no runner changes.

Verification on real 4H bars: TENX and SION flagged 2 sessions after their
2026-08-10 splits; WOLF 3 sessions after 2025-09-29; TENX as of 2026-08-07 is
clear (no look-ahead). Full live-universe scan today: 3,095 names screened at
14 ms each, **0 currently blocked** -- the recent 1d flags (LGCL, AIXI, HCWC,
CYCU, REAX, MSTU, BIAF) have no 4H file, i.e. they are outside what the 4H
modules can rank.

Known limitation: a vetoed name is not backfilled -- a top-3 book with one
vetoed name buys 2. Chosen over per-module ranking changes because the veto
fires rarely (0 today, 1 live hit in 2.5 months that would pass as organic) and
one choke point is easier to keep correct than four.

## 10.4 Universe snapshots

`build_shared_universe` now also writes
`Data/shared/universe/snapshots/shared_universe_<UTC>.csv.gz` in exclusive-create
mode (never overwritten). `load_universe_as_of(ts)` returns the latest snapshot
at or before `ts` and raises `LookupError` rather than fall back to today's list.
The first real snapshot lands on the next nightly build; history before
2026-09-10 still needs an external point-in-time source.
