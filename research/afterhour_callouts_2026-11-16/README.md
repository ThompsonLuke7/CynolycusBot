# AfterHour callout accuracy study (2026-11-16/17)

**Question.** Do the AfterHour accounts the user follows (RoyaltyTradez, Freeballer, Bobdog, PrismAnalyst, adioking,
Vivian, NewFishBigPond), plus the 20 most-followed accounts that posted in the last 90 days, make callouts that come
true more often than chance?

**Answer (retrospective, gross of costs).** No. Pooled over 4,879 scored calls the hit rate is 1-2 points *below*
each stock's own base rate (5d -1.3pp, 20d -2.3pp; MDE ~2pp), and the high-confidence subset (stated entries and
strike-bearing contracts) is no better (-1.5pp / -2.0pp). No account shows a significant positive edge; the only
|t|>2 results are negative (YungEmber, ThetaMaestro, Thegreek, x52x low-confidence) and none survives a multiple-testing correction
(~108 account x set x horizon tests). Strike calls finish past the strike 34% of the time vs 42% for a same-stock,
same-moneyness, same-tenor random-date control; stated price targets hit 61% vs a 70% control (ad hoc check on the
56 targets not already on the wrong side of the entry price; median target distance 2.9%).

## Data

* Source: AfterHour's unofficial public API (`https://api.afterhour.com`, unauthenticated GETs; endpoints from
  github.com/mphinance/aftermath `AFTERHOUR_API_SPEC.md`). AfterHour's Terms prohibit scraping; collection was
  single-threaded, >=1.2 s between requests, read-only, for personal research. Raw pulls stay local (gitignored).
* Collected 2026-11-17: full post history of 27 accounts (30,542 posts, 2023-11 .. 2026-11-17) and the full body +
  comments of every post that tags a ticker or has a cashtag (17,267 posts, 0 errors). "Top 20 by followers" =
  most-followed authors in a 90-day newest-first scan of the global feed (17,249 posts, 755 authors), excluding the
  7 named accounts; accounts silent for 90 days are therefore not included.
* Prices: Alpaca SIP daily bars, split-adjusted and raw (`raw/bars/`). Zero-volume padding rows are dropped (Alpaca
  pads reused symbols: `BTC` sat at 495.55 with volume 0 until the Grayscale mini trust listed 2024-07-31; after
  listing, corr(BTC, IBIT daily returns) = 0.999). A >10-calendar-day hole after entry ends the series (AKTS was
  delisted 2024-12 and the symbol reused 2026-01 - without this it showed a fake +21,700%). AfterHour itself maps
  `$BTC`/`$ETH` to the Grayscale mini-trust ETFs; DOGE, SPX, VIX, NQ and OTC names like MTPLF have no bars (150 calls).

## Method

1. **Extraction (no LLM; `scripts/afterhour/extract_calls.py`, schema `afterhour_callouts_rules_v1`).** Regex and
   lexicon rules over title + body, clause by clause. Tickers must be AfterHour-tagged securities (cashtag, tagged
   bare symbol, or a few company names); single-ticker posts let body sentences inherit the ticker. Tiers:
   A = stated entry (bought/added/entered/short/sold puts), B = option call (strike and/or side), C = directional
   opinion that also says something about the future. Excluded: hindsight/outcome wording, exits, questions,
   negated entries, third-party trades and flow alerts, analyst actions, covered calls, gain/loss-tagged posts.
   One call per account/ticker/direction/ET day; within a post, conflicting directions keep only the stronger tier.
2. **Audit (hand-labelled by Claude, frozen rules run_03, seed 4242, 60 calls).** Tier A 13/20 correct, strike-bearing
   tier B 8/8, bare "calls/puts" 3/12, tier C 12/20. => **high confidence** (A + strike contracts) ~75% precision,
   **low confidence** ~47%. A no-call sample (30 ticker posts) found ~8 missed calls (recall roughly 70-75%).
   Two earlier audit rounds (seeds 20261116, 777) were used to fix rules and are not used for the estimate.
   Wrong calls behave like coin flips, so a true edge is attenuated by roughly the precision (10pp -> ~7.5pp).
3. **Scoring (`scripts/afterhour/score_calls.py`, rules fixed before outcomes were inspected).** Entry = open of the
   first regular session starting after the post (same day only if posted before 09:30 ET). Hit at h sessions =
   close[entry+h-1]/open[entry] moves the called way. Control = the share of *all* h-session windows on the same
   symbol since 2023-01-01 that moved that way (expected hits = sum of per-call base rates), so long-only callers in
   a bull market get no credit for drift. Significance = t-stat clustered by entry week (calls in the same week are
   not independent; requires >=10 calls in >=5 weeks). MDE = 80% power, two-sided 5%. Strike calls: strict =
   expiry close past the strike, lenient = touched before expiry (sold puts: expired above the strike); control =
   same symbol, same strike/entry ratio and tenor from every start date. Levels are compared in the post-date share
   basis (split-adjusted bars rescaled at entry). Underlying only - no option P&L (option prints are not marks; see
   `research/options_experiment/10_RETRACTION_option_pnl_invalid.md`). Gross of costs.

## Results

Hit = share of calls that moved the called way; base = same-stock control; excess in percentage points; MDE = the
smallest excess detectable at 80% power (an excess smaller than the MDE is "not measurable here", not "no edge").
Calls on SPY itself have zero excess vs SPY by construction, so SPY-heavy accounts show ~0 in the last column.

### Pooled (all accounts)

| Account | Set | Calls | 5d hit | 5d base | 5d excess (pp) | t (wk) | MDE (pp) | 20d hit | 20d base | 20d excess (pp) | t (wk) | median 20d vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | ALL | 4879 | 48.9% | 50.1% | -1.3 | -1.02 | 2 | 48.5% | 50.8% | -2.3 | -1.55 | +0.0 |
| ALL | high | 2567 | 48.4% | 49.9% | -1.5 | -0.87 | 3 | 48.6% | 50.7% | -2.0 | -1.00 | -0.5 |
| ALL | low | 2312 | 49.3% | 50.4% | -1.0 | -0.83 | 3 | 48.4% | 51.0% | -2.6 | -2.00 | +0.0 |
| ALL | A | 2083 | 48.3% | 49.7% | -1.4 | -0.79 | 3 | 48.5% | 50.4% | -1.9 | -0.93 | -0.7 |
| ALL | B | 1341 | 48.7% | 50.4% | -1.6 | -1.05 | 4 | 46.4% | 50.9% | -4.5 | -2.48 | +0.0 |
| ALL | C | 1455 | 49.8% | 50.5% | -0.7 | -0.44 | 4 | 50.5% | 51.3% | -0.8 | -0.51 | +0.0 |

### Your 7 accounts

| Account | Set | Calls | 5d hit | 5d base | 5d excess (pp) | t (wk) | MDE (pp) | 20d hit | 20d base | 20d excess (pp) | t (wk) | median 20d vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RoyaltyTradez | ALL | 695 | 48.4% | 51.1% | -2.7 | -1.38 | 5 | 49.3% | 52.2% | -2.9 | -1.32 | +0.0 |
| RoyaltyTradez | high | 212 | 51.2% | 51.8% | -0.6 | -0.17 | 10 | 50.2% | 53.4% | -3.2 | -0.77 | +0.0 |
| RoyaltyTradez | low | 483 | 47.2% | 50.8% | -3.6 | -1.53 | 6 | 48.9% | 51.7% | -2.8 | -1.13 | +0.0 |
| Freeballer | ALL | 71 | 50.0% | 50.3% | -0.3 | -0.07 | 17 | 51.6% | 51.2% | +0.4 | 0.06 | +0.0 |
| Freeballer | high | 30 | 50.0% | 51.1% | -1.1 | -0.12 | 26 | 55.6% | 52.3% | +3.3 | 0.35 | +0.0 |
| Freeballer | low | 41 | 50.0% | 49.8% | +0.2 | 0.02 | 22 | 48.6% | 50.4% | -1.8 | -0.18 | +0.0 |
| Bobdog | ALL | 615 | 51.1% | 50.5% | +0.6 | 0.26 | 6 | 47.3% | 51.6% | -4.3 | -1.40 | -0.3 |
| Bobdog | high | 303 | 47.9% | 50.7% | -2.8 | -0.90 | 8 | 48.7% | 52.2% | -3.5 | -0.88 | +0.0 |
| Bobdog | low | 312 | 54.4% | 50.4% | +4.0 | 1.66 | 8 | 46.0% | 51.1% | -5.1 | -1.41 | -0.5 |
| PrismAnalyst | ALL | 25 | 56.5% | 52.9% | +3.6 | 0.31 | 29 | 40.0% | 54.6% | -14.6 | -1.38 | -1.0 |
| PrismAnalyst | high | 13 | 53.8% | 53.9% | -0.1 | -0.01 | 39 | 0.0% | 55.1% | -55.1 |  | -4.9 |
| PrismAnalyst | low | 12 | 60.0% | 51.6% | +8.4 | 0.42 | 44 | 66.7% | 54.3% | +12.3 |  | +4.3 |
| adioking | ALL | 127 | 51.6% | 50.7% | +0.9 | 0.15 | 12 | 55.7% | 50.5% | +5.2 | 0.71 | +1.3 |
| adioking | high | 74 | 56.2% | 50.4% | +5.7 | 0.93 | 16 | 59.2% | 50.0% | +9.2 | 1.08 | +1.4 |
| adioking | low | 53 | 45.3% | 51.0% | -5.7 | -0.65 | 19 | 50.0% | 51.2% | -1.2 | -0.14 | +1.1 |
| Vivian | ALL | 205 | 50.5% | 49.9% | +0.6 | 0.12 | 10 | 50.3% | 50.4% | -0.2 | -0.04 | -1.3 |
| Vivian | high | 58 | 44.8% | 47.9% | -3.1 | -0.41 | 18 | 43.1% | 47.0% | -3.9 | -0.56 | -3.5 |
| Vivian | low | 147 | 52.7% | 50.7% | +2.1 | 0.35 | 12 | 53.2% | 51.8% | +1.4 | 0.27 | -0.8 |
| NewFishBigPond | ALL | 368 | 45.0% | 46.0% | -1.1 | -0.37 | 7 | 40.2% | 44.3% | -4.1 | -1.45 | -6.2 |
| NewFishBigPond | high | 306 | 45.2% | 45.8% | -0.6 | -0.20 | 8 | 40.5% | 44.4% | -3.9 | -1.33 | -6.6 |
| NewFishBigPond | low | 62 | 43.5% | 46.8% | -3.3 | -0.49 | 18 | 39.0% | 43.9% | -4.9 | -0.78 | -4.2 |

### Top-20-by-followers accounts, high-confidence calls (accounts with >= 20 scored calls)

| Account | Set | Calls | 5d hit | 5d base | 5d excess (pp) | t (wk) | MDE (pp) | 20d hit | 20d base | 20d excess (pp) | t (wk) | median 20d vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mphinance | high | 349 | 49.6% | 49.8% | -0.3 | -0.08 | 7 | 51.9% | 50.4% | +1.5 | 0.52 | -1.0 |
| Rise | high | 187 | 48.1% | 50.5% | -2.3 | -0.56 | 10 | 55.1% | 51.1% | +3.9 | 0.68 | +0.0 |
| TRON | high | 156 | 48.7% | 47.3% | +1.4 | 0.31 | 11 | 45.2% | 47.8% | -2.6 | -0.53 | -1.5 |
| DocHollywood | high | 146 | 50.0% | 53.1% | -3.1 | -0.62 | 12 | 52.1% | 55.6% | -3.5 | -0.71 | -0.4 |
| x52x | high | 108 | 50.5% | 53.7% | -3.3 | -0.65 | 13 | 50.5% | 56.7% | -6.2 | -0.98 | -1.4 |
| trynamakeit | high | 97 | 47.4% | 52.7% | -5.3 | -0.96 | 14 | 62.0% | 55.1% | +6.9 | 1.36 | +0.0 |
| DayZero | high | 76 | 46.1% | 50.6% | -4.5 | -0.68 | 16 | 40.8% | 52.1% | -11.4 | -1.63 | -1.2 |
| YungEmber | high | 64 | 39.1% | 46.4% | -7.3 | -1.26 | 17 | 23.8% | 45.7% | -21.9 | -3.30 | -15.1 |
| zrosiez | high | 60 | 45.0% | 48.1% | -3.1 | -0.33 | 18 | 42.1% | 48.3% | -6.2 | -0.81 | -5.0 |
| WarrenBuffett | high | 56 | 57.1% | 50.0% | +7.2 | 1.02 | 19 | 57.1% | 49.7% | +7.4 | 1.07 | +1.7 |
| ALLOY | high | 55 | 47.3% | 53.8% | -6.5 | -0.90 | 19 | 61.8% | 56.9% | +4.9 | 0.84 | +0.0 |
| daren | high | 53 | 49.1% | 48.6% | +0.5 | 0.08 | 19 | 43.4% | 47.9% | -4.5 | -0.59 | -0.5 |
| ThetaMaestro | high | 52 | 44.2% | 51.2% | -7.0 | -0.99 | 19 | 32.7% | 53.4% | -20.7 | -2.85 | -0.3 |
| MrShorty | high | 42 | 52.4% | 45.5% | +6.9 | 0.98 | 22 | 51.2% | 44.0% | +7.2 | 0.83 | -4.5 |
| Thegreek | high | 29 | 37.9% | 52.0% | -14.1 | -2.31 | 26 | 51.7% | 54.2% | -2.5 | -0.27 | +0.0 |

### Strike calls vs control (same symbol, same strike/entry ratio, same days to expiry, random start dates 2023+)

| Account | Strike calls | Closed past strike at expiry | Control | Touched strike before expiry | Control  |
|---|---|---|---|---|---|
| ALL | 314 | 34.4% | 42.1% | 61.5% | 65.5% |
| NewFishBigPond | 65 | 41.5% | 46.0% | 78.5% | 80.0% |
| RoyaltyTradez | 45 | 31.1% | 33.4% | 60.0% | 60.4% |
| mphinance | 41 | 36.6% | 48.2% | 46.3% | 58.4% |
| Bobdog | 26 | 15.4% | 33.8% | 26.9% | 45.7% |
| x52x | 24 | 37.5% | 40.0% | 54.2% | 66.7% |
| daren | 14 | 21.4% | 37.4% | 78.6% | 62.7% |
| Vivian | 11 | 63.6% | 72.0% | 72.7% | 85.0% |
| DayZero | 11 | 27.3% | 54.5% | 90.9% | 81.0% |
| DocHollywood | 10 | 50.0% | 74.7% | 90.0% | 95.5% |
| Rise | 9 | 22.2% | 26.3% | 22.2% | 41.6% |
| TRON | 9 | 33.3% | 37.1% | 66.7% | 62.2% |
| adioking | 9 | 22.2% | 26.2% | 44.4% | 46.3% |
| YungEmber | 8 | 25.0% | 36.3% | 62.5% | 67.6% |

### Robustness slices

| Account | Calls | 5d hit | 5d base | 5d excess (pp) | t (wk) | MDE (pp) | 20d hit | 20d base | 20d excess (pp) | t (wk) | median 20d vs SPY |
|---|---|---|---|---|---|---|---|---|---|---|---|
| high, 2024 | 579 | 53.2% | 50.2% | +3.0 | 1.21 | 6 | 54.0% | 50.8% | +3.2 | 1.01 | +0.4 |
| high, 2025 | 1367 | 46.6% | 49.9% | -3.3 | -1.32 | 4 | 49.1% | 50.6% | -1.5 | -0.49 | -0.9 |
| high, 2026 | 621 | 48.1% | 49.7% | -1.5 | -0.51 | 6 | 41.8% | 50.6% | -8.7 | -2.41 | -2.1 |
| high, long calls | 2343 | 48.0% | 50.2% | -2.1 | -1.12 | 3 | 48.4% | 51.3% | -2.8 | -1.20 | -1.0 |
| high, short calls | 224 | 52.7% | 47.1% | +5.6 | 1.70 | 9 | 50.2% | 44.4% | +5.9 | 1.42 | +0.0 |
| all, long calls | 3942 | 48.7% | 51.0% | -2.4 | -1.31 | 2 | 49.0% | 52.8% | -3.8 | -1.75 | -0.5 |
| all, short calls | 937 | 49.7% | 46.3% | +3.5 | 1.55 | 5 | 46.6% | 42.8% | +3.8 | 1.41 | +0.0 |


## Reading the results

* **Pooled: a well-powered null.** With ~2pp MDE, a positive edge of more than ~2pp at 5d or 20d is ruled out for
  these accounts as a group (more than ~3pp after allowing for extraction precision). Point estimates are negative.
* **RoyaltyTradez** (695 calls): -2.7pp / -2.9pp, MDE 5pp - no detectable edge; an edge above ~+1-2pp is unlikely.
* **Bobdog** (615) and **NewFishBigPond** (368): no edge; NewFishBigPond's picks are small caps that fall more often
  than not (base 44-46%) and the median call trailed SPY by 6.6% over 20 sessions.
* **adioking**: the best point estimate (+9pp at 20d on 74 high-confidence calls) but t = 1.1 against an MDE of 16pp,
  2025 was 25% vs 49% base (12 calls), and spot checks show many 2026 "calls" are newsletter recaps of other people's flow.
* **Freeballer (71), PrismAnalyst (25, posting since 2026-08), Vivian (205, largely reposted tweets from
  @cantonmeow)**: too few calls or too weak to measure; "not measurable here", not "no skill".
* **By year** the sign flips (high-confidence 20d: 2024 +3.2pp, 2025 -1.5pp, 2026 -8.7pp) - no stable edge.
* **Lead, not a finding:** bearish calls (937 all-confidence) beat their base by +3.5pp at 5d (t 1.55) and +3.8pp at
  20d (t 1.41); longs (91% of calls) lag by 2-4pp. One slice of many, not significant.

## Limitations

* Rule extraction: ~75% / ~47% precision, ~70-75% recall; chart images (most RoyaltyTradez "Blueprints") are not read.
* Calls are scored on direction, strike and target over fixed horizons - not on the poster's own sizing, exits or
  option P&L. Many callers trade intraday or 0DTE; daily bars cannot score those.
* Survivorship: accounts were chosen because they are followed/popular *now*.
* Retrospective only; nothing here is a signal for the live system.

## Next testable variants

1. Bearish-call subset with a pre-registered rule and a held-out period (2026-11-17 onward), scored weekly.
2. Re-run on forward data only: collect new posts weekly and score them with the frozen run_03 rules (no rule changes
   after seeing outcomes).
3. If chart-based calls matter, OCR the post images (the repo already has OCR tooling in `scripts/discord_ledger/`).

## Reproduce

```
R=research/afterhour_callouts_2026-11-16
.venv/bin/python -m scripts.afterhour.collect --out $R/raw users <usernames...>
.venv/bin/python -m scripts.afterhour.collect --out $R/raw scan --days 90
.venv/bin/python -m scripts.afterhour.extract_calls --raw $R/raw select --out $R/raw/selection.jsonl
.venv/bin/python -m scripts.afterhour.collect --out $R/raw details --selection $R/raw/selection.jsonl
.venv/bin/python -m scripts.afterhour.extract_calls --raw $R/raw extract --out $R/run_03
.venv/bin/python -m scripts.afterhour.audit_sample --extract-dir $R/run_03 --out $R/run_03/audit --seed 4242 --misses 30
.venv/bin/python -m scripts.afterhour.score_calls --calls $R/run_03/calls.jsonl --bars-cache $R/raw/bars --out $R/run_03/scored --end 2026-11-17
.venv/bin/python -m pytest scripts/afterhour/tests -q
```
Outputs (local, gitignored): `run_03/calls.jsonl`, `run_03/scored/scored_calls.jsonl`, `run_03/scored/summary_by_user.csv`.
