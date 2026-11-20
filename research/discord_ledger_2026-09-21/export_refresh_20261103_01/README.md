# ACE export refresh 2026-11-03: successor account, disclosed method, free-alert scoring

All times Eastern. Research only. No live strategy, flag, model or order was changed.

## Source update

- The 14 root HTML exports were copied byte-for-byte into `raw_exports/` and version-merged against `export_refresh_20261010_02/source/messages.jsonl`. That yields 4,980 unique messages, 701 of them new (2026-10-12 to 2026-11-02).
- Two same-ID version changes are recorded: the `#🚀1k-100k-🚀` mention now renders as `#deleted-channel`, so that VIP channel was deleted. One prior message is absent from the new export and is retained.
- The seven VIP exports are still empty shells. The new VIP `#500-5k-challenge` channel was not exported.
- Old channels stopped: `🔥-ace-free-alerts` on 10-08, `⏳-ace-free-watchlists` on 10-05 and `1k-challenge` on 07-30. All new caller activity is in the new `ace-free-alerts` channel (1559060301955989554), `📈-post-ur-profits` and `free-chat`.

### ACE now posts as `Zach_ace`

The original ACE account (1079391263083733072) last posted 10-08 12:54 ET. `Zach_ace` (1557910141632782409) was created 10-08 20:18 ET per its snowflake and has posted 526 messages since 10-12. They split 362 in the alerts channel, 152 in profits and 12 in free-chat. Members address him as "Ace", and he runs the same alert and profit channels with the same `@everyone` style.

This alias is **inferred, not authenticated**. The legacy pipeline (`build`, `extract_alerts`, `ocr`, `attachments` priority) hard-codes the old ID and channel, so it would treat every new post as a member post. The new-era data is handled by `extract_ace_contract_alerts.py`, which records the alias on every row. The 538-record lifecycle ledger was **not** rebuilt.

72 successor posts are Discord **forwards**: 58 in profits and 14 in alerts. They include VIP results ("SPX CALLS 800%") and member praise ("My man Zach is goated", "Ace your plays are 🔥"). A forward's timestamp is only an upper bound on the original post time. Forwarded praise is member text, not a caller claim. IDs are in `analysis/successor_forwarded_message_ids.json`. None of the 20 extracted contract alerts is a forward.

### Images

The HTML exports reference images by **signed Discord CDN URLs**, not embedded files. This export's signatures expire 2026-11-03 13:31–2026-11-04 01:30 ET. After that, the HTML shows broken images and the files cannot be fetched again. Everything was preserved first:

- 1,624 of 1,625 unique attachment IDs are local in `attachments/`, hashed, with manifests `manifest.jsonl` and `manifest_topup.jsonl`. This includes all 230 successor attachments and member screenshots.
- The single exclusion is a 25 MB+ July screen-recording video, rejected by the downloader's safety cap.
- The downloader missed 248 forwarded-message attachments: DiscordChatExporter renders them under `.chatlog__forwarded-attachments`. The selector in `attachments.py` is fixed and tested, and the missed files were fetched from their freshest signed URLs.
- For future exports, enable DiscordChatExporter's asset download (CLI `--media --reuse-media`, GUI "Download assets") so images are bundled with the HTML. Otherwise, run the attachment fetch the same day.

## What the new posts disclose (recreation inputs)

Visual transcriptions are in `ace_successor_screenshot_evidence_20261103.json`. Verification status for each input:

| Input | Evidence | Status |
| --- | --- | --- |
| **Expected-move (EM) bands**: daily/weekly/monthly, "strongest S/R", fade/target them | 46 EM tables (Nov 2025 to Oct 2026), SPY/QQQ text levels, panel `D EM` | **Center reproduced exactly; width reproduced up to a per-table scale.** See "Image analysis and reproduction" below; that section supersedes the single-table "0.763" constant first reported here. |
| **GEX King / VEX King** per expiry | 10-26 watchlist card | **GEX King 4/4** testable cases equal argmax \|net GEX\| by strike for that expiry, from our own chain (`analysis/king_node_output.txt`). At n=4 this is not distinguishable from max call-OI strike. VEX King 3/4: GILD 132 unexplained. ZETA is not in our chain universe. |
| **"CYCLE ORB" panel**: cycle length/bias, days to due low, ORB break, SMA5/EMA21/SMA34/SMA55, divergence state on 5m/30m/2H/D/W, daily EM | VST daily chart (10-16) | The MA values match prior verification (Oct study). His "optimum entry 152.47" is the 0.5 fib and TP 174.52 the 1.272 extension. Cycle and divergence formulas remain private and unresolved. |
| **Fibonacci** 23.6/50/61.8/78.6 retracements and 1.272/1.618 extensions; Bollinger squeeze | VST chart, watchlist "Why" fields | Stated, not yet tested. |
| **Cycle timing windows**: SPY peak 10/23 ±3d, daily low 11/4 ±3d, weekly low 11/30 ±3w | 10-18 text | Prospective and checkable: SPY fell 10-29→10-30. One window is one observation, not a validation. |
| **VIX rule**: VIX ≥ +10% → "85%" up day within 2 sessions; ≤ −10% → 85% down day | 10-18 text | **Not an edge.** 2000–2026: 82.9% vs 79.8% base rate (+3.2pp, z 1.76, MDE ~5pp); 2-day return lift −0.1pp. The down leg runs the wrong way: 66.8% vs 70.5% base. The 85% is mostly the base rate (`analysis/vix_rule_output.txt`). |

## Free contract alerts scored on real option prints

`ace_successor_contract_alerts.jsonl` holds 20 premium-specified free alerts and 6 premium-less ideas, each with source message IDs and reply-linked management. `alert_scoring_02/` scores them on Alpaca 1-minute option trade prints with SIP underlying bars. `alert_scoring_01` is superseded: it scored a post-close BRK.B entry below its stop.

Rules were fixed before any outcome was seen:
- Entry: the first print ≥ 60 s after the post. Observed delays were 63–154 s, except BRK.B: posted 15:59 Friday, filled Monday.
- Rule A: stated stop vs +100%, whichever is touched first; the stop counts first on a same-bar tie.
- Rule B: stated stop, otherwise hold to expiry.
- Hold: expiry intrinsic from the underlying's last RTH minute.
- All returns are gross: no spread, no commission.

Fitness check: SPY/QQQ-put prints track the underlying, with sign-adjusted 1-minute correlation 0.77–0.95. Single names (0.25–0.62) and far-OTM QQQ calls (0.20) are noisy, so for those only the underlying-settled expiry result is reliable. Three unexpired contracts are censored because Alpaca returns 403 "OPRA agreement is not signed" for them: IWM 280p 11/4, IBM 265c 11/6 and MRK 155c 11/20. SPX and BRK.B are excluded from the summary. That leaves n = 15 across 9 sessions.

| Rule | Mean | Median | Win rate | Without best trade | Without the 10-29 index puts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hold to expiry | +124.8% | −100% | 27% | +38.7% | −68.9% |
| A: stop vs +100% | −0.9% (95% CI −35 to +34) | −15.6% | 27% | −8.1% | −6.6% |
| B: stated stop, else hold | +68.0% | −16.7% | 13% | −22.1% | −23.0% |

- **9 of 15 expired worthless.**
- The positive hold mean comes from three correlated index puts on one day (10-29, into the 10-29/30 selloff): QQQ 700p ×2 at +925% and +1,330%, SPY 765p at +443%.
- With his own stated stops, two of those three were stopped out (−18%, −16%) before the move.
- This is a descriptive audit of 15 bets that are clustered by regime, not a significance test. The MDE is not meaningful at this n.

### Marketing vs outcomes

- Recap cards and "%" posts use **Max Gain = peak premium after entry**, not realized P&L.
- The 10-30 "PREMIUM ALERTS RECAP" (11 VIP alerts, "+$55,730") omits the **three free call alerts posted that day**: QQQ 700c ×2 and SPY 758c 0DTE. All three expired worthless on the underlying close.
- The 11-02 card ("NO LOSSES", 10 plays, SPY 760p +360.71%, +$24,450 at 10 cons) contradicts his same-day text: 15 plays, SPY 760p +494%, +$41,260 at 10 cons.
- Silent outcomes include GOOGL 365c, GOOGL 352.5c "size for zero", VST 155c 0DTE (posted "NHOD CONS ITM" midday, expired worthless) and MRK. CRWV was openly closed on news, and the SPX lotto was openly called a miss.
- The free channel is now largely a sales funnel: 61 of 526 posts carry whop/promo/VIP wording, alongside VIP result forwards.

## Image analysis and reproduction (round 2)

**OCR coverage.** All 1,594 preserved images were OCR'd: 1,087 reused September readings matched by hash, 606 new, 0 failures. Output is `attachments/ocr.jsonl`, using `ocr.py --deps-dir` with an isolated RapidOCR install. `extract_image_evidence.py` classifies each image (`image_evidence.jsonl`):

| Author | Images | Quote pages | Positions | EM tables | CycleORB panels | Cards / recaps / GEX tool | Other (charts, misc.) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zach_ace | 230 | 126 | 20 | 10 | 12 | 2 / 3 / 1 | 56 |
| ACE (old) | 965 | 439 | 68 | 36 | 12 | 0 / 0 / 2 | 408 |
| FT, members | 399 | 2 | 57 | 0 | 0 | 0 | 340 |

- OCR of EM tables equals visual transcription on 64/64 rows.
- Key images were read visually: 12 panels, 2 watchlist cards, the rule card, 2 recaps and 5 EM tables. See `ace_cycle_panels_20261103.json`, `ace_rule_cards_20261103.json` and `ace_successor_screenshot_evidence_20261103.json`.
- An SPY chart legend names the tools: **"CycleORB Sig"** (the panel, plus BUY/SELL markers), **"Daddy Algo"** and a hidden **"LuxAlgo – Ult ORB"**. Neither CycleORB nor Daddy Algo is public under those names.

**What reproduces, from our own data.** Scripts and outputs are in `analysis/`.

| Element | Result | Evidence |
| --- | --- | --- |
| SMA5 / EMA21 / SMA34 / SMA55 (panel) | **40/40 within 0.15%** | `cycle_reverse_engineer_output.txt` |
| EM center | **= prior close**: panels 7/7 exact; 46 tables / 631 rows, 88% within 0.05% | `em_band_test_output.txt` |
| EM width | Within a table, the ratio of his width to nearest-expiry ATM IV × √T is nearly constant across names (CV 1–5%). Daily bands use the **nearest expiry** (e.g. Monday dailies), which removes earnings premium. The overall per-table scale **varies 0.58–1.03** across 7 testable tables (9/11 to 10/23 captures), so it is unexplained; candidates are his IV snapshot time or provider. A recreated band lands within roughly ±15–25%. | `em_reproduction_output.txt` |
| GEX King / VEX King | **GEX 9/12** as argmax \|net GEX\| per expiry (11/12 when total absolute gamma also counts); **VEX 9/12**. Chance ~5–10% per row. | `king_node_all_output.txt` |
| Rule-card trigger | **Trigger = prior-day high (calls) / low (puts)**: 6/6 within 0.05 or a whole-dollar round. Target 2 = daily EM edge. | `ace_rule_cards_20261103.json` |
| Cycle length / due low / bias | **Not reproduced.** The best pivot-spacing rule gets cycle length within ±1 on 5/10 panels, which is not clearly beyond chance after 450 variants. Due-low matches only 2/10. Hurst translation agrees with BULL/BEAR on 8/10 (suggestive only). Due-low counts down one per trading day, so it is a fixed projected date. | `cycle_reverse_engineer_output.txt` |
| Divergence states, ORB "Today" field | Not yet tested | — |

**Honesty checks.**
- **Receipts are real.** Robinhood quote screenshots match SIP underlying prices at the phone-clock minute (Pacific time) on 85/85 checkable cases; 5 apparent misses are OCR reading BRK.B as "B". Option marks match prints on 52/53. Median post delay is 2.6 min (`quote_authenticity.csv`).
- **Recap maxima are real prints.** 12/13 checkable claims are within ±10% of the observed post-entry high, and the 13th is understated (`recap_claims_check_output.txt`).
- The overstatement is therefore **metric and selection** (peak instead of exit, omitted losers), not fabricated numbers.

**First "does it work" test.** This uses his published EM levels on daily bars, 631 rows from 46 tables.
- 74% of final closes land inside the band, versus 68% for 1σ: implied vol overstates realized, which is the normal variance risk premium.
- At the edge, rejection given a touch is 50.5%, versus a −2.4pp bump relative to interpolated control levels at 0.75× and 1.25× (95% table-cluster CI −5.0 to +0.3pp). **Daily-close S/R at the EM edge is not supported**; an effect larger than ~3pp is ruled out at this resolution.
- Intraday touch-then-reverse, the way he trades the bands, is untested.

## Round 3 (2026-11-04): his own plans, cycle calls, and the "ACE emulator" hypothesis

The user pasted another assistant's read of the 10/06–10/07 screenshots. Its hypothesis: he keeps a short list of momentum/theme leaders, waits for an HTF trend plus a cycle-low window, buys a pullback into SMA5/EMA21 on short-timeframe confirmation, and exits at precomputed targets.
- Every screenshot it used is in this export (posts 10/06 05:01 to 10/08 13:36 UTC). Its GEV transcription is correct.
- I transcribed 32 underlying plans (Dec 2025 to Oct 2026, with entry/zone, fail and T1–T3) and 20 dated cycle-timing calls: `ace_plans_and_timing_20261104.json`.
- They were scored on daily bars by `analysis/plans_and_timing.py`, with outputs `plans_timing_output.txt`, `plans_timing_plans.csv` and `plans_timing_timing.csv`.

| Claim | Test | Result |
| --- | --- | --- |
| Entries sit at SMA5/EMA21 | Entry level vs his four MAs at the prior close | 10 of 15 at-or-below-close entries are within 1.5% of an MA (SMA5 8, EMA21 5). A random pullback level gets there 47% of the time (7.1 expected), because the MAs stack just under price. **Only weakly above chance.** |
| Candidates are exceptional-momentum names | Return percentile among the 1,000 most liquid names on the session before the post | 126/252-session returns: median **79th/81st** percentile; 14/29 and 16/29 in the top quintile (5.8 expected). 63-session: median **53rd**, 3 top-decile. They are **long-horizon leaders in a ~3-month consolidation**, not current momentum extremes. |
| Cycle "due low" timing is the missing piece | Pivot (lowest low within ±5 / ±10 sessions) inside his window, vs the share of same-length windows over that symbol's prior 500 sessions holding a pivot | k=5: **5/15 vs 5.8 expected (p=0.76)**; k=10: 3/14 vs 2.6. Panel due-low dates: ARM, COHR, GEV, AFRM, GOOGL all missed; CRDO hit at the window edge. The 10/23 "cycle peak" fell on a local *low*, and that window's highs were lower highs, so it cannot resolve as a hit. The 11/4 ±3 low is pending (10/30 low 750.36 is inside). **MDE ≈ +30pp over base, so this is "not measurable here", not "no skill".** |
| The setups throw fat right tails | Fill per the stated entry rule; T1 within 10 sessions vs the same name's prior-year base rate for a move that size | 24/32 filled; **T1 hit 4/16 vs 8.7 expected** (P(X≤4)=0.008 if independent, but six of the 16 fall in one October semis week). Median MFE +2.0%, median MAE −14.7%. Mean path return −12.0% vs SPY −0.6%. |
| Targets are fib extensions | Targets vs fib levels of trailing N-session (low, high), N 5..150, ±0.25% | GEV is exact: the chart draws 1.272/1.618 of the 40–50-session range, and he cites a "1.236% ext" for WMT. Across plans, 43/67 targets match vs **39.3 for jittered targets**, so no single rule is identifiable. Targets mix fib extensions, ATH, EM edge and King node. |
| BULL/BEAR bias is not direction | GEV shows 30d BEAR while above all MAs | Agrees with round 2: bias tracks cycle translation (8/10), a phase statistic, not trend. |

**Reading.**
- The chat's process description is accurate as a description of how he *presents* trades: a watchlist, precomputed levels, waiting for price to come to the level, then short-timeframe confirmation. INTC 10/06 is the clean example: EMA21 112.77, a 5m bull divergence, then the "cinema" post.
- The parts that would make it an edge are the parts that fail or can't be measured here:
  - his cycle calls hit at the base rate;
  - his plans underperformed their own base rates;
  - "near an MA" is barely above chance.
- The selection fact worth keeping is the 6–12-month-leader filter. CynolycusBot already measured something close to it: PIT top-liquid 12-month momentum beats SPY (long-horizon study).
- The state machine the chat proposes already exists as `strategies/intraday_structure`:
  - levels: prior-day high/low/close, opening range, premarket, VWAP, volume profile;
  - detectors: breakout, trend_pullback, vwap_reclaim;
  - dealer_plate for GEX.
- ACE inputs it lacks: daily MA level families (SMA5/EMA21/SMA34/SMA55), the D EM band, multi-timeframe RSI-style divergence (it only has a one-bar sign flip), and fib-extension targets.

## Rule-card backtest (2026-11-04): underlying, SIP 1-minute, 2022-01 → 2026-10

**Card geometry, reverse-engineered from the 10/23 card** (`ace_rule_cards_20261103.json`):
- Trigger = prior-day high (calls) / low (puts).
- T2 = EM edge (prior close ± h).
- **Stop = trigger ∓ 0.25h** (5/6 rows; CARR 0.21h).
- **T1 = 2R beyond the trigger** (5/6; XLV uses 1R because 2R would reach T2).
- "Scanner picks" all closed within 0.5h of their trigger.

The card is fully mechanical.

**Setup.**
- **Script:** `analysis/rule_card_backtest.py`. Bars come from `analysis/rule_card_fetch_1m.py` (SIP, split-adjusted, RTH, 15.5M minutes).
- **Universe:** 33 symbols, his 22 EM-table names (SPX excluded) plus 11 SPDR sector ETFs.
- **Entry rules:** candidates are within 0.5h of the trigger with ≥1R of room. Enter on a break in the first 30 min; skip gaps through the trigger.
- **Management:** half off at T1, stop to entry on the rest, rest exits at T2 or flat at the close.
- **Fills:** a buy-stop takes 1 tick of slippage, a stop takes 1 tick through, targets need a trade 1 tick through. Bar ordering is conservative; the optimistic bound is shown.
- **h proxy:** we have no IV history before 2026-09, so h = **0.966 × daily EWMA σ**. That scale is fit to 241 rows of his own daily EM tables, on levels and not outcomes. Per-table medians run 0.65–1.21, the same per-table drift as the IV fit.
- **Outputs:** `rule_card_backtest_01/backtest_output.txt`, `trades_primary.csv`, `summary.csv`.

| Variant | Trades | Mean R/trade [95% day-cluster CI] | Win | bps |
| --- | ---: | --- | ---: | ---: |
| **Primary** | 9,091 | **−0.122 [−0.159, −0.080]** | 33% | −7.1 |
| Frictionless (no tick slippage) | 9,091 | −0.030 [−0.069, +0.011] | 34% | −1.7 |
| Optimistic bar ordering | 9,091 | −0.032 [−0.074, +0.012] | 36% | −2.4 |
| Control: random minute, all 27,110 candidate sides | — | −0.056 [−0.083, −0.027] | 40% | — |
| First 15 min only | 7,680 | −0.151 | 32% | −9.1 |
| Any time of day | 14,158 | −0.092 | 35% | −4.8 |
| Stop on first 1-min close back through trigger | 9,091 | −0.140 | 11% | −8.0 |
| Top 3 calls + top 3 puts per day (closest to trigger) | 2,853 | −0.127 | 33% | −8.5 |
| No proximity filter | 11,455 | −0.135 | 36% | −7.4 |
| EM scale ×0.75 / ×1.25 | 7,368 / 10,434 | −0.234 / −0.058 | 30% / 37% | −10.0 / −4.0 |

- **Every year is negative** (2022 −0.04, 2023 −0.10, 2024 −0.15, 2025 −0.12, 2026 −0.22). So are calls, puts, ETFs, single stocks and each proximity bucket.
- **The trigger carries no direction.** T1 (2R) comes before the 1R stop on 30.6% of resolved trades; a driftless path gives 33.3%. Exits: 66% stop, 18% T2, 7% T1 then breakeven, 10% close.
- **Frictionless, it is flat;** the loss is fill cost. Median 1R is only 41 bps of price, so two ticks on a $20–$60 name cost 0.1–0.2R.
- **MDE ≈ 0.056R per trade** (80% power), so this is a well-powered null for the underlying: any edge above ~0.06R per trade is ruled out on this universe.
- **The 10/23 card itself:**
  - QQQ put triggered at 09:30 and was **stopped (−1.0R)**.
  - XLK never broke its trigger (low 181.55 vs 181.50) and rallied.
  - XLF and XLV never triggered.
  - CARR and ADI are outside the universe; ADI gapped past its trigger, so it was a skip.
- **Untested:** options on these trades. Convexity cannot rescue a zero-direction underlying signal once spread and theta are paid; see the options retraction.

## EM-edge fade / reaction (2026-11-05): underlying, SIP 1-minute

The design was pre-registered in the docstring of `analysis/em_edge_fade.py` before the first run; output is in `em_edge_fade_01/`.
- **Event:** first trade 1 tick through a band level.
- **Trade:** fade with a limit at the level, stop 0.25h beyond it, target the band center (prior close), flat at the close.
- **Controls:** the same trade at 0.75h and 1.25h levels.
- **Reaction metric:** move 30 and 60 minutes after the touch, in h units, positive = reversal.

| Cohort | Events at the edge | Fade R/event [95% CI] | 30-min reaction (h) | Edge minus neighbours |
| --- | ---: | --- | --- | --- |
| **C: proxy bands, 33 names, 2022–2026** (primary) | 18,140 | **−0.119 [−0.156, −0.085]** | −0.023 [−0.033, −0.013], i.e. slight continuation | −0.012R |
| A: his published daily tables | 100 (18 tables) | −0.176 [−0.502, +0.192] | +0.024 [−0.052, +0.106] | −0.323R |
| B: his published weekly tables | 126 (14 tables) | −0.043 [−0.220, +0.159] | +0.009 [−0.024, +0.048] | −0.007R |

- **Primary:**
  - The fade loses in every year (−0.07 to −0.17R), on both sides, in ETFs and in stocks, early or late in the session. MDE 0.05R.
  - The edge behaves the same as the 0.75h and 1.25h control levels, so the published level is not special.
  - After a touch, price drifts on away from the center by ~0.02h (≈4 bps over 30 min), below costs. Neither fading nor following the edge clears costs.
- The "target-first vs driftless" figures in the raw output ignore the session time limit, so only the control-level comparison is fair.
- Cohorts A and B are his exact levels but underpowered (MDE ~0.5R); they agree in sign with C.
- **Hypothesis class closed for the underlying:** his EM bands as intraday levels, both breakout (rule card) and fade, on this universe.
  - The one ACE input still untested is the **King-node magnet** (dealer GEX). It is the only one that is not a transform of price and volatility, and it is testable on dealer captures since 2026-09-11.

## Why followers post wins anyway (2026-11-05)

- **The channel is a funnel.**
  - 194 of the caller's 3,132 messages push VIP/premium/promo access, and 14 are price lists ($150/month to $1,000–1,300 lifetime). The moderator account FT accounts for 74 more.
  - We can see the marketing, not his account. VIP calls are unobserved.
- **Posted P&L is selected.**
  - Of 61 P&L screenshots with a readable total return (`analysis/pnl_screenshot_kinds.csv`), 57 show a gain and 4 a loss (93%).
  - His scored free alerts won 27%, with a median of −100%.
- **Many member "wins" are not trades.** 17 of 39 member screenshots are Robinhood **watchlist trackers** ("Cost when added", "Date added"), not positions. They show a hypothetical return with no money in, median +208%.
- **Peak-print bangers need no skill on 0DTE.**
  - Setup: `analysis/zero_dte_banger_rate.py`, output in `zero_dte_01/`. 459 sessions (2025-01..2026-10), with entries at 09:45, 11:00 and 13:00.
  - Each entry buys the ATM call/put (nearest dollar) and the call/put $2 out of the money, on expired SPY 0DTE contracts.
  - Sanity check: corr(option, SPY 1-min return) = +0.86 calls / −0.82 puts, with a median 390 bars per contract, so the prints are marks.

  | Entry 09:45, no signal | ATM call | ATM put | OTM call ($2) | OTM put ($2) |
  | --- | ---: | ---: | ---: | ---: |
  | Later prints ≥ 2× entry ("100% banger") | 42% | 44% | 45% | 45% |
  | Later prints ≥ 3× | 15% | 23% | 26% | 28% |
  | Expires ≤ 10% of entry | 49% | 55% | 70% | 76% |
  | Mean return held to expiry | −8% | −8% | −16% | −16% |

  - On **80% of sessions, the ATM call or the put doubles after 09:45**; both double on 6%.
  - Anyone who posts a call and a put (he routinely alerts both sides and calls one a hedge) and reports the peak has a "100% banger" four days in five, while holding loses money.
  - At 11:00 and 13:00, 34–43% still double at some point, and holding loses 11–49%.
- **A lottery book makes real winners.** A book of short-dated OTM options with median −100% and a mean carried by a few 5–10× hits produces some genuine big winners every week. With many followers, someone always has a real win to post. Our n=15 scored free alerts cannot tell such a book's mean from zero (CI −35..+34%).

## Checklist 3.2 gate (2026-11-06): exit-based scoring of every ACE option entry — FAILED

This is the gate in `research/year_two_experiment_checklist_2026-11-04.md` §3.2 ("ACE as a teacher policy").

**Alias.** The caller and channel IDs now live in `scripts/discord_ledger/identities.py`. Zach_ace shares the ACE *label*; lifecycle linking stays per author_id. The new alerts channel is read by `extract_alerts`. Covered by `tests/test_identities.py`.

**Ledger rebuild.**
- Steps: extract_alerts / challenge / context → `reconcile_export_refresh` (1,040 prior annotations carried over after source-version checks, plus 78 new text annotations; `reviewed_evidence_none_20261106.json` promotes no new image facts) → `build` → `ledger_01/`.
- Validation is all green: sources resolve, authors match, no future evidence, no fabricated fills.
- Result: 579 lifecycles; **249 ACE option entries** (223 ACE, 26 Zach_ace). Only 14 exits and 32 trims link to any of them.
- One reconcile fix: the preserved-attachment index is read only when a reviewed fact cites an image.

**Scoring** (`scripts/discord_ledger/score_ace_lifecycles.py`, rules in its docstring fixed before outcomes; output `exit_scoring_02/`).
- Entry: first option print ≥60 s after the alert was visible.
- Exit policies: stated exits (trim = half, exit = rest, remainder valued at expiry), hold to expiry, flat same session, and 2× limit or expiry.
- Control: the same contract bought at 20 random minutes of the same session, with identical exits.
- Fitness: corr(option, underlying 1-min returns) ≥0.5 over ≥30 minutes. Median corr is 0.62; 87 of 157 are fit.
- **First run superseded:** it valued expiry from the daily cache. That cache's `CVNA` file is **split-adjusted** for CVNA's 2026-05-08 5:1 split (Feb close $67.37 vs raw $336.62), while option strikes are raw, so it produced +120,000% fake winners. The 1d cache is therefore a mix of adjusted and unadjusted files, not uniformly unadjusted. It is kept as `exit_scoring_01_superseded_daily_cache_spot/`.
- The rerun uses the raw SIP last-minute close. Expiry values agree with the contract's own late prints on 117 of 121 checkable rows; the 4 mismatches sit at the strike.

| Policy (all 143 with an expiry outcome) | Win | Same-contract random-time control | Paired diff CI | Mean return [CI] | Median |
| --- | ---: | ---: | --- | --- | ---: |
| **His stated exits** (gate) | **16.1%** | 14.7% | −0.9..+3.9pp | **−24% [−60, +24]** | −100% |
| Hold to expiry | 11.2% | 10.9% | −0.2..+0.9pp | −32% [−73, +21] | −100% |
| Flat at the entry session's close | 29.4% | 33.4% | −11.2..+2.6pp | −7% [−35, +35] | −22% |
| +100% limit, else expiry | 37.8% | 39.6% | −7.3..+3.8pp | **−24% [−43, −6]** | −100% |

- **Gate: FAIL.** His entries don't beat a random minute on the same contract (timing ruled out above ~4pp), and the mean is not above zero. The checklist says stop, so the teacher dataset is not built.
- **No directional skill:** the underlying moved his way from entry to the close on 46.5% of 156 entries (mean −4 bps).
- **Return shape:** 71% of entries lose ≥90%; 11% return ≥+100%. The top five carry 58% of all gains, and the mean excluding them is −55%.
- **The fit-only subset** (prints track the underlying) agrees: stated-exit win 20.7% vs 18.7%, CI includes 0.
- **Zach_ace alone** (n=14) is +76% mean, 36% win. That is one week of index puts into the 10/30 drop (QQQ 700P, SPY 765P); the sample is too small and too clustered to separate from the old account.
- **Limits:**
  - 82 entries never name a full contract and can't be priced (direction of bias unknown).
  - Gross of spread, so real follower results are worse.
  - VIP alerts are unobserved.

## Validation

- Discord-ledger tests: see the LIVING_SUMMARY entry for the current count. New tests cover alert extraction, alert scoring and the forwarded-attachment fix.
- The fixed parser covers all 248 previously missed references on the real exports.
- Option prints were checked against the underlying per alert.
- Analysis scripts and their outputs were rerun from `analysis/` with identical results.

## Next testable variants

1. ~~Rule-card backtest~~: done 2026-11-04. Null on the underlying (−0.12R/trade after fills, −0.03R frictionless, MDE 0.06R). The breakout trigger shows no direction (T1-first 30.6% vs 33.3% driftless).
1b. **Next: intraday EM-edge reaction / fade.** T1-first slightly under the driftless rate, together with closes landing inside the band 74% of the time, points at mean reversion, not continuation.
   - Test first touch of his *published* EM edges (2025-11 → 2026-10; the 1-minute bars for his table names are already cached in `rule_card_backtest_01/bars_1m`) against control levels at 0.75h and 1.25h.
   - Pre-register: fade on touch, stop 0.25h beyond the edge, target the prior close.
1c. **Emulator selection on the rule:** rerun the card only on 6–12-month leaders, the selection his names actually show (top-quintile 126/252-session return, PIT). This needs bars for a PIT leader universe; the 33-name universe here is mostly ETFs and megacaps.
2. **King-node magnet**: test whether price is attracted to the argmax-|net GEX| strike into expiry, versus other high-OI strikes.
3. **Rescore the three censored alerts** after they expire, and score VIP contracts visible in recap cards. The recap cards are selection-biased; score them as a separate, labeled cohort.
4. ~~Add the successor alias to the legacy pipeline~~: done 2026-11-06 (`identities.py`, `ledger_01`). The 3.2 exit-based gate then failed (section above), which closes the ACE-as-teacher line.
5. **Run OCR over the remaining successor images** (most unreviewed), prioritizing EM tables, watchlist cards and recaps, then visually adjudicate them.

Reproduce:

```
stage_export_refresh --base export_refresh_20261010_02/source/messages.jsonl
attachments --include-member-images
extract_ace_contract_alerts
score_ace_contract_alerts
```

All four are under `scripts.discord_ledger`. Scripts under `analysis/` take the repo root or an output path.
