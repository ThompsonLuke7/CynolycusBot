# Steps 6–9 (2026-09-24 evening) — quote fallback root cause, trigger attribution, the matrix gap, pool admission

Scripts: `06_quote_fallback_forensics.py`, `07_pool_admission.py`; ad-hoc checks noted inline.
Outputs: `data/quote_fallback_forensics.{csv,json}`, `data/pool_admission.json`.

---

## A. The underlying-quote fallback: mostly a timestamp bug, not missing data

Replayed every `premium fallback active` line (1,123 events, 09-17 to 09-24) against historical
IEX quotes and IEX/SIP 1-minute bars (`06_quote_fallback_forensics.py`).

| cause | events | share |
|---|---|---|
| **quote timestamp AFTER the pass's `now_et`** (negative age, rejected by `0 <= age`) | 1,033 | **92.0%** |
| quote would pass on the historical tape (most likely the live endpoint served a newer quote, so the same bug) | 73 | 6.5% |
| genuinely stale (> 60s) | 17 | **1.5%** |

**Root cause.** `scripts/live_risk_pass.py:210` takes `now_et` once, and the four modules then run
one after another (with state locks and broker calls in between), so the quotes are fetched 1–5s later. For a
liquid name the latest IEX quote is almost always newer than `now_et`: median "age" −2.0s
(p10 −3.1s). `core/risk_prices.py` requires `0 <= age`, so **the more liquid the name, the more often
it fails**. That's why CRWD and DELL top the list: CRWD had 3,045 IEX quotes in 10 minutes. It is not an IEX coverage
problem, and SIP would not fix it.

If age were measured against `now_et` with the quote taken at or before it, **98.4%** of these events pass
(median quote age 1.0s, p90 5.9s, p99 75.6s). The 17 genuinely stale events had quote ages of 60–120s,
were isolated single passes (the next 5-minute pass succeeded), and are DELL ×10, AXTI ×3, SMTC, MRNA.

**The user's two alternatives, measured on these events:**
* **Last-bar fallback.** Using the *latest IEX 1-minute bar* (not the 4H cache, which is up to ~4h old):
  the bar is the in-progress or just-finished minute at p90, **≤ 3.2 min old at p99, 11.7 min max**. Its close is
  within 2.9bp of SIP (median), 11.5bp at p90 and 32bp at p99, which is negligible against a 1.5-ATR stop (~6%).
  Using the 4H-cache close instead would mean a delay of up to one 4H bar.
* **Hold.** After the timestamp fix, a hold lasts **one 5-minute pass** in every one of the 17 residual cases
  (no two consecutive). The damage is bounded by the underlying's 5-minute move beyond the stop level,
  which is small against a 1.5-ATR stop (estimated, not simulated per position).

**The measured cost of the bug is smaller and less certain than 01_findings stated.** All 7 HTF
positions that fired the premium stop had the underlying **above** its stop at the moment (checked on SIP 1-min closes).
The correct rule would have held them. But 6 of the 7 (WULF, MRNA, PURR, SMCI, APLD, EIX, all expiring 09-18)
were 1-DTE, and `expiring_before_next_session` flattens them at 15:45 the same day. So their real cost
is the difference between a 09:33 and a 15:45 exit price, which can't be measured without option marks and could have either sign.
The −$18,971 is **not** the bug's cost. **HL (Oct expiry, stopped at −39% with the underlying 1% above its stop) is the
one clean instance.** The bug is still real and ongoing for any non-expiring position.

**APPLIED 2026-09-25 (user approved).** `core/risk_prices.py` now measures quote age at fetch time (5s skew tolerance), and falls back to the latest IEX 1-min bar close (≤ 300s old) before the premium stop. `CurrentUnderlyingPrices.source` records quote/bar/None for each symbol. New `AlpacaOptionsClient.get_latest_stock_bars`. Tests: `core/tests/test_risk_prices.py` (8). Smoke-tested against the live endpoint. The risk pass runs as a fresh subprocess, so no restart is needed. Original recommendation: in `CurrentUnderlyingPrices`, measure age against
the fetch time (take `now` after the request), or tolerate a few seconds of negative skew. That removes ~98% of
fallbacks. Optionally widen `max_age` to ~180s, and for anything left use the latest IEX 1-min bar
(or hold one pass) instead of the premium stop. No paid feed is needed.

---

## B. trigger_rule — the "0 of 96" was my join, not missing data

`05_sizing_and_triggers.py` joined trades to decisions with a nearest-2-day `merge_asof` and got
0 matches. An exact join on `(ticker, entry_bar == decision bar)` matches **88 of 97** momentum
closed trades (the 9 misses have a null `entry_bar`, legacy rows). The trigger was always on the
entry's `signal_audit.extra` in managed state and in the order audit; only the closed-trade row lacked it.

**Shipped:** `closed_trade_record` now writes `trigger_rule` from the entry state's signal audit
(`core/live_4h_exec.py`), null = unknown. Test added in
`core/tests/test_cross_module_closed_trade_ledger.py`.

Live outcome by trigger (fill-gain return per trade):

| route | trigger | n | mean | median | win |
|---|---|---|---|---|---|
| equity | break_body_prev_high | 35 | +5.4% | +2.9% | 51% |
| equity | pullback_continuation | 21 | +0.7% | −7.5% | 33% |
| option | break_body_prev_high | 19 | −28.0% | −50.0% | 21% |
| option | pullback_continuation | 13 | −47.2% | −53.8% | 8% |

Same sign as the Stage 5 prior (pullback −0.588R) on both routes, but **not measurable yet**:
equity difference −4.8pp, Mann-Whitney p=0.32, MDE at 80% power ≈ 22.6pp; option MDE ≈ 63pp.
An underpowered null, not evidence of no effect. Next testable variant: re-run at n≈150
momentum closes (the new field makes that a direct groupby), or an intra-sample test of
pullback vs breakout on the same ticker-weeks in the backtest.

---

## C. `feature_nan_in_span` = earnings features at the 90-day cap, dropped by `dropna(how="any")`

The 9.2% in-span gap is not a generic feature failure. In the June matrix, the last row before a
missing run has median `days_since_earnings` 87 (p90 = 90), and the first row after has median
`days_since_earnings` **0**. Runs have a median length of 5 sessions (p90 = 17).
`signals/events/earnings_calendar.add_earnings_features` sets any distance > `MAX_EARNINGS_DISTANCE_DAYS` (90) to **NaN**,
and `build_training_matrix` then drops the row (`expansion_labels.py:372`). Quarterly gaps are often
91–100 calendar days, so **the days immediately before an earnings report get deleted**. That
explains the 2.27x tail density: these are the earnings-move windows.

**The same drop happens live.** `inference/ranker.py:153` drops rows with any NaN model feature, and
`expansion_v1` uses `days_to_earnings`/`days_since_earnings`. Replaying the current calendar on the
2026-09-21 universe (2,703 names): **7.1% unscorable on 09-24, 6.8% on 09-10, 10.4% on 08-20**
(2.3% absent from the calendar entirely, the rest at the cap). Estimate from a replay, not read
from the live panel.

### C.1 A data-loss bug found on the way: the earnings calendar history has been destroyed

`ticker_earnings_calendar.parquet` has two writers: the historical builder
(`signals/events/earnings_calendar.py --refresh`) and the nightly news stage
(`signals/news/main.py --stage earnings-calendar`, run from `scripts/nightly_market_data.sh:305`).
The news stage concatenates its forward snapshot onto the file and runs
`drop_duplicates(["ticker", "snapshot_date"], keep="last")`. Historical report rows have
`snapshot_date = NaT`, so **every nightly run collapses each ticker's report history to one row**.
The file now holds exactly one historical row per ticker (1,099 tickers, max 1). Report dates start at 2026-03-23.

Consequences:
* **Rebuilding the training matrix now would silently drop almost all pre-2026 rows**:
  every historical `days_to/since_earnings` becomes NaN → `dropna(how="any")`. The current
  `features_4h.parquet` (09-24) already has both earnings columns 100% NaN. Decision #5 (matrix
  rebuild) is **blocked** until the calendar history is restored.
* Live scoring is only mildly affected (the forward snapshots still cover recent reports).

**FIXED 2026-09-25:** `merge_calendar_snapshot` / `merge_calendar_history` in `signals/events/earnings_calendar.py`: each writer replaces only its own row kind. `--refresh` also no longer drops the snapshot rows, and the news stage no longer swallows an unreadable prior file. Tests: `signals/events/tests/test_calendar_merge.py`. **History NOT yet restored**: that needs `python -m signals.events.earnings_calendar --refresh` (a yfinance sweep of the whole universe), which must run before any matrix rebuild.

---

## D. Pool admission — what admits a name, and do we miss the move?

`07_pool_admission.py` replays the three data rules of `score_universe` point-in-time on the
weekly (Sunday) cadence over 4,109 daily-bar tickers, 2021-01 → 2026-08 (4.09M rows, CA-masked).
Tail = 10-session MFE from next open ≥ 4 ATR. Candidate-pool membership is NOT modelled, and the
bar set is survivor-shaped (small illiquid names that later delisted are missing), which
**inflates the blocked groups' returns most**.

| status at the weekly rebuild | rows | tail rate | lift vs eligible | share of all tails | median fwd20 |
|---|---|---|---|---|---|
| eligible | 2.13M | 8.73% | 1.00x | 41.8% | +0.28% |
| blocked: ADV < $5M | 1.49M | 12.83% | 1.47x | 42.8% | +0.11% |
| blocked: < 200 days history | 0.44M | 14.20% | 1.63x | 14.1% | +0.45% |
| blocked: price < $1 / > $1000 | 0.02M | 27.85% | 3.19x | 1.4% | 0.00% |

**What admits names: dollar volume.** 4,917 of 5,962 admission episodes are ADV crossings,
1,025 are the 200-day history rule.

**Are they admitted after the move?** Yes, but only slightly. ADV admissions arrive after a median **+3.6%** 20-day
run against a +0.5% same-day baseline. **After admission they do not continue:** median fwd20
+0.20% vs +0.97% for already-eligible names the same day, and the tail rate matches the baseline
(8.68% vs 8.66%). By the time volume qualifies them, the edge is spent.

**Could we catch them earlier?** Loosening the ADV floor, rows added versus the current $5M rule:

| rule | added rows | added tails | tail lift vs eligible | median / mean fwd20 of added |
|---|---|---|---|---|
| ADV ≥ $3M | 367k | 37.0k | 1.15x | +0.30% / +1.17% |
| ADV ≥ $2M | 591k | 61.5k | 1.19x | +0.32% / +1.30% |
| ADV ≥ $1M | 880k | 94.2k | 1.23x | +0.28% / +1.32% |
| history ≥ 120d | 75k | 6.9k | 1.04x | +0.90% / +1.25% |
| history ≥ 60d | 103k | 9.9k | 1.10x | +0.72% / +1.15% |

Eligible baseline: median +0.28%, mean +0.68%. The tail rate among ADV-blocked names rises
steadily as liquidity falls ($3–5M 10.0% → <$0.5M 16.1%). "More tails" here is mostly
"smaller, more volatile names". The same median return and a higher mean make that a right-skew
bet, and survivorship pushes exactly those means upward.

**Reading.** The historical `not_in_universe_yet` 3.22x from step 3 was mostly a definitional
artifact. The matrix's per-ticker span starts at its first *feature* row (bar history plus warm-up),
not at an admission record, and point-in-time snapshots only exist since 2026-06-23 (momentum) and
2026-09-10 (shared). Measured directly, the admission lag is real but small, and there is no
post-admission continuation to capture. The only lever with a positive sign is a lower ADV
floor ($2–3M): +15–19% relative tail density and roughly 2x the mean fwd20. That's retrieval, not
edge, until the ranker is shown to pick well inside that slice.

**Next testable variant:** the momentum OOF predictions already cover pool tickers on dates when
their ADV was below $5M (the matrix is not ADV-gated). Score ranker top-k *inside* the $2–5M
slice against top-k in the eligible slice on the same dates, and include a noise-trained control
(see the top-k null-control memory). If top-k lift holds there, lowering the floor is worth a paper trial.

---

## E. Does admitting the $2–5M ADV band improve the momentum top-3? No.

`08_adv_band_topk.py`: walk-forward OOF scores of `expansion_v1`, the live candidate filter, and top-3 per
4H bar. 1,740 bars from 2022-11 to 2026-05, 172k core and 65k band candidate rows. Outcome = 25-bar
`fwd_close_return`. CIs from a 25-bar block bootstrap.

| | mean per pick | 95% CI |
|---|---|---|
| A: top-3 from core (live rule) | +3.42% | [+0.94, +5.97] |
| B: top-3 from core + band | +3.24% | [+0.25, +5.72] |
| **B − A, gross** | **−0.18pp** | [−1.63, +1.07] |
| **B − A, net of 20bp extra band cost** | **−0.26pp** | [−1.60, +0.95] |
| B − A on fwd_max_return | +1.81pp | [+0.63, +2.97] |
| ranker lift in core (top-3 − slice mean) | +2.42pp | [+0.39, +4.48] |
| ranker lift in core vs ATR-matched random | +1.54pp | [+0.11, +3.06] |
| ranker lift in band (top-3 − slice mean) | +0.72pp | [−1.46, +2.77] |
| ranker lift in band vs ATR-matched random | +0.62pp | [−0.92, +2.14] |

Band names would take 43% of picks and change the basket on 77% of bars, with **no gain in close-to-close
return** and a higher peak return, which means more volatility, not more expectancy. The model separates winners
significantly inside the core slice, including against a vol-matched control, but **not inside the band**. All of this holds
even though survivorship favours the band (band rows belong to names that later became liquid enough for today's
pool), so the null is conservative.

**Power:** the CI on B − A has a half-width of about 1.4pp, so the MDE at 80% power is about 2pp per 10-session hold. An
improvement smaller than ~2pp can't be ruled out, but the point estimate is negative. **Verdict: do not lower
the ADV floor for this model.** The hypothesis class "same model, wider pool" is closed. The remaining variant is a
model trained or calibrated on band rows (its ranking there is near-random), which only makes sense if a
band-specific edge shows up first. **Optional / Outside Roadmap.**
