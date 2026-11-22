# Regime switch: can the engine tell "still selling off" from "about to bounce"?

**Date:** 2026-11-20 · **Agent:** Claude · **Status:** investigation only; nothing is wired live

Triggered by the Oct–Nov 2026 tech selloff. QQQ fell about 18% from its 10-06 close (760, adjusted)
to about 610 intraday on 11-20. Over the same span SMH fell about 30%, NVDA about 40%, and SOXS
doubled between 10-28 and 11-05. The user asked whether
the system can objectively detect a down phase and then act on it: stop buying dips, short the weak
AI trade, rotate into what is still bid, and cut losers.

Scope: index and sector/theme ETFs, daily bars, 2016-10..2026-11-19 (Alpaca SIP, split+dividend
adjusted). No option is priced here, and no live module is evaluated. Both scripts are pre-registered
in their docstrings.

---

## 1. Why nothing in the repo captures this today

| Piece | State |
|---|---|
| `signals/market_regime/daily_regime.py` | Built nightly with trend, RV, breadth, credit and risk appetite. It flagged this selloff: `spy_trend_state` turned −1 on 11-05, `spy_rv20_z` ≈ 3, `credit_risk_z` −2.8 on 11-20. It feeds model *features* only. |
| Governed-path `MarketRegime` (`core/nervous_system`) | **Stub.** `adapt_market_row` hard-codes `MarketRegime.UNKNOWN`, so the RISK_OFF/CRISIS size multipliers never fire (every decision gets the flat UNKNOWN 0.5). No theme regime is classified either. |
| Live swing modules | Long calls first. 30m swing short entries are disabled (299 real put fills lost −$32.9k; single-name, mostly ≤2 DTE, `options_experiment/12`). Momentum is a long breakout ranker. HTF and Meta are long. Only intraday structure, the SPY daytrader and the dealer ranker can express a bearish view. |
| Account post-mortem 07–11 | Regime freezes did not beat random refusal, but `spy_trend_state` read "uptrend" every day of that window, so that study never saw a downtrend. |

## 2. Findings

### A. Trend state does not forecast index direction over 1–4 weeks (`01_regime_study.py`)

Four detectors were computed on the instrument itself: D0 fast (close < EMA20 and EMA20 falling), D1
close < SMA50, D2 EMA20 < EMA50 (the live table's formula) and D3 close < SMA200. Entries are at the
next open.

- **All days.** QQQ forward returns in the DOWN state were *not* lower than in UP, at 5, 10 and 20
  sessions, for any detector (DOWN minus UP: +0.05 to +1.5 pp, t ≤ 1.5). SPY and SMH show the same.
- **Dip buys in DOWN** (day ≤ −1.5%; QQQ shares) averaged +0.9% at 5d and +3% at 20d, with a 59–67% win
  rate. That is no worse than dip buys in UP. The cost is the path: the median adverse excursion is −4%
  to −6% at 20d, against −3% to −4% in UP. That path is what kills a long call.
- **Shorting the bounce in DOWN** (day ≥ +1%) returned roughly 0 at 5–10d and −1.7% to −2.3% at 20d. On
  SMH the 10d result was −0.7% to −1.0%. Bounces in down tapes are real and violent: QQQ rose 6.7% from
  11-05 to 11-11, and SOXS fell 44% over the same days.
- **Underlying books** (decide at the close, trade at the next open, 2 bp, marked daily, with a
  circular-shift placebo):

  | QQQ 2016-10..2026-11 | CAGR | Sharpe | MDD | Placebo Sharpe pct |
  |---|---|---|---|---|
  | Long always | 19.2% | 0.89 | −35% | – |
  | Long, flat when D1 DOWN | 12.4% | 0.88 | **−17%** | 85% (MDD: best of 1,000) |
  | Long, short when D1 DOWN | 2.8% | 0.23 | −36% | 30% |
  | Long, short when D2 DOWN | 3.2% | 0.25 | −34% | 13% |

  Going short on trend state had no timing skill on any instrument. Even in 2022 the long/short book
  lost 20–22% (QQQ −32.5%). Going flat is a **risk control**: it halved drawdown at equal Sharpe and
  gave up return. It is not alpha.

### B. Relative strength reverses in down tapes (`02_relative_strength_by_regime.py`)

The universe is 48 sector/theme/style ETFs (median 38 eligible per week). Each week the top 5 vs bottom 5
by 63-day return, held open to open.

| QQQ regime | Weeks | Long top5 / short bottom5 | Short bottom5 outright |
|---|---|---|---|
| D1 DOWN | 147 | −0.37%/wk (t −1.0) | −0.66%/wk (t −1.4) |
| D2 DOWN | 115 | −0.74%/wk (t −1.8) | −1.07%/wk (t −2.0) |
| D1 UP | 378 | +0.05%/wk | −0.18%/wk |

- **"Short the weak groups in a selloff" lost money over 2016–2026.** This is the documented
  momentum-crash pattern: in falling markets the most-beaten groups bounce hardest. The data rule out a
  hedged edge above about +0.33%/wk in D1 DOWN (the upper end of the 95% CI).
- **2026 is the exception so far:** long/short returned +29% (63-day) and +35% (21-day).
- **The obvious call was not visible in real time.** SOXX first entered the weakest-5 on 11-06.
  Before that the AI trade *was* the 63-day leader, which is why its unwind was so violent. The
  strong side on 11-06 was ARKG, XLE, XLP, IBB and XLV (healthcare and defensives), as the user
  observed.
- *Note:* the random-pair null percentile in `rs_by_regime.csv` (0.00 for DOWN) overstates
  significance. It holds no common factor risk, so rely on the t-stats.

### C. Detection speed (QQQ, every drawdown ≥ 10% since 2017)

| Detector | Median sessions to flag | Typical share of the drawdown already gone | Flips/yr | False flags/yr |
|---|---|---|---|---|
| D0 fast | 7 | 30–40% | 9.3 | 3.5 |
| D1 SMA50 | 8 | 40–60% | 7.7 | 2.7 |
| D2 EMA20/50 (live table) | 14–18 | 40–100% | 2.1 | 0.5 |
| D3 SMA200 | 17–20 | 40–100% | 2.2 | 0.0 |

This episode: QQQ D0/D1 flagged on 10-15/10-16 (QQQ 728/710, −4% to −7% from the peak). SMH flagged on
10-14. SPY's EMA20/50, the regime table, flagged on 11-05 at QQQ 647. Faster detection costs about
3 false alarms a year.

## 3. What this means

1. **Use regime for defense, not direction.** On every test here, "DOWN" means *wider outcomes*: bigger
   adverse excursions and violent bounces. It does not mean a predictable continuation. Shrink gross
   when D1 is DOWN; the index book halves drawdown. Going short on the same switch has no net edge
   (§6 shows where it wins and loses).

   *Corrected after user review, same day.* An earlier draft said "refuse new long calls while D1 is
   DOWN". That overstates the evidence. QQQ dip buys in DOWN states were profitable *as shares*
   (+3% at 20d, §2A). What fails is the expression: 30–60 DTE single-name calls, held through the
   larger adverse excursions without an exit. The personal record agrees (gitignored report).
2. **Losers, regime-independent.** In the personal record, long calls that reached −30% rarely
   recovered (report in `Data/personal/reports/`). A hard premium stop is supported in both regimes.
3. **The 2026 tape may be different** (relative strength +29% YTD), but one year cannot certify that.

## 4. Next testable variants (not run)

- **V1 Intraday continuation in high-vol regimes.** Does the first 30 minutes predict the last hour
  when `spy_rv20_z` > 1? This is the documented intraday-momentum effect, strongest on volatile days.
  It fits the "elevator down" observation, and its home is the SPY daytrader (already long/short).
  Needs 1m/5m bars: the 1h cache omits 09:30–10:00.
- **V2 Regime refusal for paper modules.** Apply "no new long calls while QQQ D1 DOWN" to the
  post-mortem ledger and the decision archive, against a random-refusal placebo. The IWM<SMA50
  version fell inside the band.
- **V3 Bearish expression where it is cheapest.** Compare QQQ/SMH put debit spreads with long puts in
  DOWN states on the OPRA cache (2024-02..2026-09). Run the derivative-vs-underlying correlation check
  first.
- **V4 Volatility-scaled gross.** Size by target vol / realized vol. It does not forecast direction,
  and it is well documented.
- **V5 Sentiment/catalyst conditioning** (market-wide FinBERT balance + breadth + credit). Only about
  two episodes have news coverage, so it can be descriptive only, not certified.

## 5. Caveats

- **Sample.** 2016–2026 is a strong bull market with V-shaped recoveries (2018Q4, 2020, 2025-04).
  2022 is the only grinding bear, and 2000–02 and 2008 are absent. Independent bear episodes number
  about five.
- **Universe.** The ETF list is today's (survivor-shaped). No costs in 02 (turnover is in `rs_weekly.csv`).
- **Partial session.** The 11-20 session is excluded (incomplete at fetch). The last 02 week is
  decided 11-06.

## Reproduce

```
.venv/bin/python -m scripts.research_data.fetch_research_bars_1d --tickers QQQ SPY SMH ... --out-dir Data/research/regime_switch_2026-11-20/bars_1d
.venv/bin/python research/regime_switch_2026-11-20/01_regime_study.py   # ~75 s
.venv/bin/python research/regime_switch_2026-11-20/02_relative_strength_by_regime.py
```
Outputs: `out/*.csv`, `out/meta.json` (bars fetched 2026-11-20 21:05 UTC; ETFs 21:11 UTC).

## 6. Follow-up (same day): where the short side wins and loses, and the user's intraday trade

### 6.1 Short runs, one by one (`04_short_leg_by_run.py`, QQQ, D1 close < SMA50)

Each run shorts at the open after the first DOWN close and covers at the open after the first UP close.

| Runs | Count | Win | Mean | Compounded |
|---|---|---|---|---|
| Inside a ≥10% dump | 28 | 36% | +0.3% | +7.7% |
| Outside any dump (false alarms) | 50 | 14% | −1.5% | **−53%** |

- **The short does catch the big grinding dumps:** 2018Q4 +9.1%, 2020 +3.6%, 2022 +7.3% / +13.2% /
  +3.8%, and 2025-02..05 +6.2%.
- **This selloff, in two runs:**
  - flagged 10-16, short from the 10-19 open, covered at the 10-29 open after the 10-28 close above
    the SMA50: +0.4%;
  - re-flagged 10-29, short from the 10-30 open: +3.7% through the 11-19 open (more through the
    11-20 close of ~609, which is outside the daily data).
- **The false alarms cost more than the dumps pay.** Even within 2026, before October, six short runs
  lost (−2.1, −2.3, −2.1, −0.6, −2.2, −5.0%) because those dips were bought.

### 6.2 Red open → short QQQ → cover an hour later (`03_red_open_intraday.py`; `03b` exploratory)

5-minute SIP bars, 2016-10..2026-11-20. Entry at 10:00 when the first half hour is ≤ −0.5% (03), or at
09:35 after a gap down ≤ −0.5% (03b). Exit one hour later.

| QQQ | n | Mean 1h | Win | Favorable ≥0.5% in the hour | Adverse ≥0.5% in the hour |
|---|---|---|---|---|---|
| Red first 30 min, D1 DOWN | 132 | −0.03% | 45% | 49% | (median adverse 0.41%) |
| Gap down, D1 DOWN | 185 | +0.03% | 53% | 53% | 53% |
| Gap down, this selloff | 9 | +0.29% | 56% | 44% | 78% |

- **No edge on average.** In down regimes the hour after a red open moves ≥0.5% *both* ways about
  half the time. That is volatility, not direction.
- **The favorable move the user sees is real, but so is the adverse one.** On 11-20 QQQ gapped −1.0%.
  A 09:35 short went 1.3% *against* first, then paid +0.7% at 10:35 and +3.9% by the close.
- **2026 leans the user's way** (red-open shorts on DOWN days: +0.24%, 57% win, n=14), but that is too
  few days to separate a regime change from luck.

### 6.3 Next tests of the user's rules (not run)

- **Forward paper shadow of the exact rules**, pre-registered with a fixed read date: (a) QQQ gap-down
  short when D1 is DOWN, out after 1h and at the close; (b) the D1 daily short with a stop on
  false-alarm runs.
- **Price the daily-flag short with real put marks** on the OPRA cache (2024-02..2026-09). Defined-risk
  puts cap the false-alarm loss at the premium, which is the version of the idea that could change
  the net.

## 7. Second follow-up (2026-11-22): a multi-signal regime vote, the intraday structure module, and 0DTE puts

The user asked for a regime built from several signals (not only the 50-day), for the intraday
structure module to go short when it turns bearish, and for QQQ 0DTE/1DTE puts over the last month or
two on 1-minute bars. All three scripts are pre-registered in their docstrings.

**The vote** (`05`, reused by `06`/`07`). Seven bearish votes, all known at the prior close:
- QQQ fast trend (close < EMA20, EMA20 falling) and QQQ close < SMA50;
- SPY EMA20 < EMA50, breadth_z < 0, risk_appetite_z < 0, credit_risk_z < 0 (HYG/IEI) and spy_rv20_z > 1.

BEAR = 4 or more. Thresholds are round values, none tuned. In this selloff the vote read BEAR from
the 10-16 close (4 votes), 6 votes from 11-05 and 7 from 11-17. Breadth stayed positive until 11-17
because defensives held up.

### 7.1 Vote vs the 50-day alone (`07_vote_vs_sma50.py`, QQQ daily, 2020-08..2026-11-19)

| Detector | Flips/yr | False flags/yr | Short runs in dumps | False-alarm runs | Long/flat Sharpe, MDD | Long/short CAGR (placebo pct) |
|---|---|---|---|---|---|---|
| Buy and hold | – | – | – | – | 0.75, −35% | 15.6% |
| SMA50 | 8.7 | 1.9 | 21 runs, +8.8% | 34 runs, −40% | 0.65, −17% | 0.1% (44%) |
| **Vote ≥ 4** | 5.2 | 1.1 | 16 runs, **+21%** | 17 runs, **−34%** | **0.81, −20%** | 7.6% (61%) |
| Vote ≥ 5 | 3.5 | 0.3 | 11 runs, −2% | 11 runs, −13% | 0.81, −28% | 10.0% (59%) |

- **The vote is better than the 50-day.** It has half the false alarms and flags real selloffs on the
  same day (2022-01, 2024-07, 2026-10-16), and as a defensive switch it beats buy-and-hold on Sharpe.
- **Shorting on it is still net negative** (−21% compounded over the short runs) and indistinguishable
  from random timing (placebo 61st percentile). It misses fast V-shaped dips (2020-09, 2021-02)
  entirely, so it is slower, not more predictive.

### 7.2 The intraday structure module, live paper ledger, by regime (`05_intraday_structure_by_regime.py`)

No backtest is needed here: the module logged 3,114 confirmed setups (678 tickers) from 2026-08-27 to
11-20 in real time, plus 249 usable paper option fills (0–4 DTE).

| Setups (underlying, net of 12 bp) | n | Gross mean | Net mean | Median hold |
|---|---|---|---|---|
| Longs, not BEAR | 1,368 | +0.02% | −0.10% | 4 min |
| Longs, BEAR | 835 | −0.02% | −0.14% | 3 min |
| Shorts, not BEAR | 576 | +0.04% | −0.08% | 5 min |
| Shorts, BEAR | 335 | −0.01% | −0.13% | 2 min |

Gating by regime (shorts only on BEAR days, longs otherwise) gave −0.104% per setup against −0.110%
ungated, which is 83rd percentile of a random gate.

| Real option fills | n | Mean premium return | Median | $ P&L |
|---|---|---|---|---|
| Puts, BEAR days | 24 | +12.5% | −4.5% | **+$2,915** |
| Puts, not BEAR | 71 | −13.3% | – | −$4,296 |
| Calls, BEAR days | 51 | −12.1% | −12.6% | −$3,987 |
| Calls, not BEAR | 101 | −11.1% | – | −$7,354 |

- **Direction is not the module's problem; holding time is.** The median setup lasts 2–5 minutes
  (exit "extension conditions deteriorated") with a median MFE of about zero, so the gross edge is
  zero on either side and the costs make it negative. A regime gate cannot help a 3-minute scalp
  ride a multi-day selloff.
- **The positive put cell is one day.** On 11-05, 0DTE puts on SPY, QQQ and IWM returned +297%, +108%
  and +92% ($3,784). The other 17 BEAR-day puts lost $652 combined; only one of them (CRWV, +11%)
  made money.
- **On BEAR days the module still took 2.5× as many longs as shorts** (835 vs 335), and its calls lost
  in every regime.
- **The module was dark on 12 of the 27 selloff sessions.** The 1-minute bar archive has no file for
  10-15..10-23, 11-03/04, 11-13, 11-16 or 11-19, so it missed the days the regime first flipped.
