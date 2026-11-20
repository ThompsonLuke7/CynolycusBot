# Stage 3 pre-registration: point-in-time history, neutral-score rule (rule v2)

Fixed 2026-11-08, **before the extended matrix exists**: no data for this design has been
built or looked at. It follows from `26_stage2_label_bakeoff_validation.md`:
- the v1 null was underpowered;
- dropping calendar features helped the top of the ranking (exploratory);
- the matrix is survivor-only and carries known feature leaks.

Anything changed after the matrix is built goes in the deviation log at the end, and every
result it touches is then exploratory.

## Why pre-register

v2 was chosen *after* looking at the 2022–2026 Stage 2 results. If the rule were chosen
again after seeing the new results, the new run would just be a second search, and its
best arm would look good by luck. Fixing the arms, metric and pass bar now makes the run a
test. That means one pick per slot, decided by a rule written before the data.

## Data (all research-only; the live cache is never written)

| item | rule |
|---|---|
| Securities | `Data/research/pit_universe/securities.parquet`, `kind == "stock"`. This covers delisted names (Alpaca serves them under the old symbol) and keeps reused tickers as separate lives (`SYMBOL~1`). |
| Hourly bars | Alpaca 1Hour, `feed=sip`, `adjustment=all`, 2016-01-01 to latest, via `fetch_intraday` (the same RTH filter as live, for feature parity). Stored in `Data/research/bars_1h_sip_adj/`. Each security keeps only bars inside its life window. Zero-volume bars are dropped. |
| 4H bars | `resample_1h_to_4h`, unchanged. |
| Daily bars | `Data/research/bars_1d_sip_adj_pit/`. |
| Context | SPY, QQQ, IWM, VIXY, TLT and sector ETFs, from the same feed and adjustment. |
| Universe per bar | Lagged close ≥ $1, and top 3,000 by trailing 20-session dollar volume, lagged one session. |
| Delisting | A security whose bars end inside a label window exits at its last close, and SPY is measured over the same truncated window. Rows are flagged `delisted_in_window`. |
| Warm-up | Features need 252 sessions of daily history, so the first usable rows are about 2017-01, and OOF starts about 2019-01. |

**Features (`pit_no_cal`, 82).** Start from the v1 `no_cal` set (93) and remove the 11
features that exist only for today's survivors. Their missing values would act as a
"this stock dies" flag, and the trainer's drop-missing rule would delete the dead names
again:
- `days_to_earnings`, `days_since_earnings`, `is_pre_earnings_3d`, `is_post_earnings_3d`,
  `earnings_in_fwd_window`: earnings dates exist for today's tickers only;
- `sector_id`, `rs_sector_5`, `rs_sector_20`: there is no sector mapping for dead names;
- `market_cap_bucket`: a single 2026 snapshot;
- `asset_type`, `is_etf`: the universe is stocks only.

`low_price_flag` stays: it is computed from the point-in-time close.

## Arms (6)

Each arm gets its own `__perm` twin, with labels shuffled within the bar:
- `L0__pit`, `L2_10__pit`, `L2_20__pit`;
- each arm's own `__perm` twin.

L0 is a baseline only; it is reported but cannot be picked. The training recipe is
unchanged from Stage 2:
- walk-forward with 24-month train and 6-month test folds, `embargo_days(h)`, the last 20%
  of the train window for early stopping;
- the same XGBoost parameters;
- seeds 42 and 43.

## Metric

**Neutral score:** within each bar, regress the percentile rank of the score on the
percentile ranks of `daily_atr_pct`, `beta_spy_60`, `adv_rank` and `px`, and keep the
residual (`eval_stage2._residualise`). This removes the volatility tilt that L2 rank IC
penalises.

**Primary metric:** within-bar Spearman IC of the neutral score against the slot's
yardstick, on bars with at least 50 pairs. Each bootstrap block spans ceil(h/5) weeks, and
the evaluator uses 1,000 draws.

## Partitions

- **Validation:** OOF bars before 2026-05-15, minus `embargo_days(h)`.
- **S_new:** OOF bars before 2022-04-01. No Stage 2 read has ever evaluated this period.
- **Test:** 2026-05-15 on. It is opened **once**, for the picked arm(s) only, and not
  before the h=20 labels have matured on at least 12 months of test bars (earliest
  2027-06-15). The test MDE is recorded before it is opened.

## Selection rule v2

- **Slots:**
  - momentum: yardstick L2_10;
  - htf: yardstick L2_20.
- **Candidates in both slots:** `L2_10__pit` and `L2_20__pit`. That makes 4 tests, with
  BH-FDR applied across all 4.

An arm qualifies for a slot only if **all** of these hold:
1. The 95% CI of its neutral IC on validation is above 0.
2. The 95% CI of (neutral IC − its perm twin's neutral IC) is above 0, with BH q ≤ 0.05
   across the 4 tests.
3. Its neutral IC is positive in a majority of validation folds.
4. Its neutral IC point estimate is positive on S_new.

The pick is the qualifier with the highest neutral IC. If nothing qualifies, there is no
pick: the test stays sealed, and the next step is a new pre-registration, not a re-cut of
this one.

**Confirmation:** the picked arm's neutral IC on the test window must have a 95% CI above
0. Otherwise the arm is "not confirmed". There is no re-picking after the test is opened.

**Reported, but not gates:**
- raw IC (the v1 metric);
- neutral top-3 excess, with corporate-action windows masked and returns winsorised at
  1/99 within the bar, gross and net of the assumed cost;
- results by year and by fold;
- the VOL row and `L0__pit`;
- a delist-zero sensitivity bound.

## Data-fitness gate (runs before anything is built)

A 50-security probe of the hourly fetch for 2016-01 to 2020-08: alive, delisted and
reused-ticker names, plus SPY and VIXY. The thresholds are fixed before the probe runs:

| check | pass |
|---|---|
| SIP entitlement for 2016+ hourly bars | no 403 or subscription error; data returned for delisted symbols |
| Volume is SIP, not IEX: Σ hourly RTH volume ÷ daily SIP volume, median | between 0.6 and 1.2 (IEX would be about 0.02–0.05) |
| Price agreement: hourly-derived session close vs daily close (both `all`-adjusted) | median \|diff\| < 0.5%; daily return correlation > 0.99 |
| Coverage: sessions with both 4H bars, for securities with daily dollar volume of $5M or more | ≥ 95% |
| Reused tickers | each life's hourly slice matches that life's daily bars |

**Gate result, 2026-11-08: PASS.** 48 securities plus SPY and VIXY; 27 are delisted.

| check | result | pass |
|---|---|---|
| Entitlement | 49/49 symbols returned, no errors | ✓ |
| Volume ratio | median 0.76; 0.66–0.87 on liquid names | ✓ |
| Close agreement | median 0.027% | ✓ |
| Return correlation | median 0.998 | ✓ |
| Both 4H bars, liquid names | median 99.5%, min 99.2% | ✓ |
| Reused tickers | every reused life (BBBY~1, WOLF~1 and new WOLF, LGCY~1, NCI~1, USAT~1, KALA~1, SYT~1) matches its own daily bars | ✓ |

- The volume ratio sits below 1 because the 09:30–10:00 half hour is missing and the
  daily bar includes the auctions.
- The worst close gaps are level offsets with return correlation ≥ 0.99 (VMRK 1.2%, ABEV
  0.8%, JOUT 0.7%). The cause is the `all`-adjustment snapshot: the daily set was fetched
  2026-10, the hourly 2026-11. Returns are unaffected.
- Illiquid names (DHX, $0.8M a day) correlate 0.96, because their hourly "close" is the
  last trade, not the closing auction.

The probe also found a defect, fixed before the full pull.
- alpaca-py 0.43 treats `limit` as a cap on the **total** number of bars, and it never
  exposes a next-page token. So `fetch_intraday(limit=10_000)` silently returned only the
  first ~10,000 raw bars, about 2.5 years of hourly data.
- The research fetcher now passes `limit=None`.
- Live callers of `fetch_intraday` are unchanged and still capped.

The full pull of 7,768 symbols started at 2026-11-08 03:06 ET.

## Known limits, accepted

- The hourly bars keep the live quirks: there is no 09:30–10:00 bar, and the 16:00 bar is
  an after-hours print. Kept for parity with live.
- `all` adjustment is a snapshot as of the fetch date.
- A stock that moved to OTC is marked at its last exchange close.

## Deviation log

| date | change | reason | results affected |
|---|---|---|---|
| 2026-11-08 | Securities whose adjusted OHLC includes a price ≤ 0 are excluded (JTAI hourly, LDTC~1 daily). | Found in a trial matrix (min px −4,557). Alpaca's `all` adjustment drives a few series negative, and those are not price histories. The rule uses prices only, never outcomes. | None: decided before any arm was trained. |
| 2026-11-08 | Data end is 2026-09-24, the median last daily bar of securities still trading. Live securities keep their hourly bars past that date, so windows near the end get exits. | The two daily fetch batches ended 09-24 and 10-02; the week between them would hold only part of the universe. | None: the test window is sealed. |

**Implementation notes, 2026-11-08 (not deviations).**
- **Code:**
  - build: `strategies/momentum_expansion/ablation/build_stage3_pit.py`;
  - evaluator: `eval_stage3.py`, which reuses `eval_stage2`;
  - labels: `stage2_labels.ticker_forward(exit_at_last_bar=True)`.
- **Universe ranking:** done among the rows present at each 4H timestamp.
- **Stray timestamps, 2.9% of rows:** these are illiquid names with no trade in a slot's first hour, so the bar carries a later timestamp. They get no targets and the trainer drops them, as in Stage 2.
- **Parity check:** a trial matrix on half the securities matched the live-cache Stage 2 matrix in 2023 on 628k shared rows.
  - price features: Spearman 0.975–0.998;
  - `fret_10`: 0.996; `L2_10`: 0.979;
  - volume features: about 0.86, the expected IEX-vs-SIP gap.
- **Trial label rates:**
  - y_L2: 20.2% positive;
  - delisting-clamped rows: 0.2% at h=10 and 0.4% at h=20;
  - rows with every feature: 96%.

**Full build, 2026-11-11 (no arm trained yet).**
- **Inputs:** hourly pull 7,768/7,768 symbols, 0 failed.
- **Securities:** 8,158 considered. Dropped: 2 with bad prices, 276 too short, 11 with no bars.
- **Matrix:** 15,247,752 rows, 7,586 securities, 2017-01-03 to 2026-09-24, 4.24 GB.
- `stage3_checks.json`: no problems.
  - 3,000 names on every full bar in every year;
  - stray-bar rows: 3.8%;
  - rows with every feature: 96.9%;
  - px min: $0.036;
  - y_L2_10 / y_L2_20 positive: 20.0%;
  - clamped rows: 0.20% (h=10) and 0.40% (h=20).
- **Point in time:** 2,948 of the 7,586 securities later stopped trading. Their share of universe rows falls from 34.7% in 2017 to 1.8% in 2026; the Stage 2 matrix had 3 of 2,888.

## Validation result (2026-11-19)

- **Momentum slot:** L2_20__pit qualifies, with neutral IC +0.0093 [+0.0023, +0.0166] and
  BH q = 0.028. The selection is frozen (hash 621d0acaa1cd).
- **HTF slot:** no pick (CI crosses 0; MDE 0.0157).
- **Final test:** still sealed; the 12-month gate is not met.
- **Caveats:** the result is regime-dependent (4 of 8 years positive), top-k is net negative,
  and live momentum daily features lag one session more than in training.
- Details: `28_stage3_validation.md`.
