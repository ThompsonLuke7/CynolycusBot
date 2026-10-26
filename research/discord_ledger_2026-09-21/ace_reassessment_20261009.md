# ACE reproduction reassessment — 2026-10-09

The earlier result was too broad a conclusion from too narrow a test. It evaluated one simultaneous MA/pullback/VWAP/volume gate at sampled alert timestamps. It did not test waiting for a level, remembering the touch, scanning intervening candles, entering a position, or managing that same position through trims and exits. It therefore did not establish that the screenshot-derived approach failed.

The new continuous benchmark, the screenshot investigation and the existing-engine comparison are complete. A faithful reproduction has not been demonstrated, and the hypothesis family has not been exhausted.

## What the images establish

- All 16 displayed SMA5/EMA21/SMA34/SMA55 values for INTC, NBIS, ARM and ALAB match the local October 5 daily bars within 0.01. This is a useful, concrete validation of both price scale and average calculation.
- INTC's 111.37–113.00 zone contains its 112.63 EMA21. ARM's 298 level is close to its 297.21 SMA5. Those motivate separate level-selection families, rather than a rule that treats every nearby average equally.
- NBIS and ALAB show BEAR cycle labels alongside bullish planning. NBIS says a low is due tomorrow while its panel shows −31d. The private field's semantics cannot be replaced by a simple positive/negative countdown or BULL-only direction veto.
- The images contain different event types: premarket plans, underlying-zone tag notifications, option buy references, option mark updates and a “mostly out” reduction. A zone-hit notification is not a verified option fill; “mostly out” is not a full close. Each must be scored against its corresponding algorithm stage.
- Three pairs of images are duplicates. The visual transcription preserves image ordinals and leaves missing message IDs and absolute UTC publication times null. No dates or IDs were manufactured to make a match.

The October 6 SIP INTC path first touches the literal zone at 12:38 ET. That is consistent with the displayed 12:41 notification under an October 6/Eastern-clock interpretation, but that interpretation is not yet an authenticated message timestamp. ARM first touches the exact 298 level on October 7 at 09:57 ET. Its displayed 09:39 notification cannot be assigned to that first touch without resolving its date/timezone and availability. These are data-alignment questions, not a conclusion about the caller's honesty.

## What was run

The 222 original ACE lifecycles yield 201 distinct entry decisions with a usable direction and source underlying snapshot; eight duplicate decisions are consolidated and thirteen have unavailable direction/RTH evidence. The previous 120-alert cohort and chronological date boundaries stay fixed. The primary comparison partition contains 25 alerts across 12 sessions: 13 ETF alerts and 12 stock alerts. These historical test dates have already been inspected, so all new strategy variants are exploratory.

Historical SIP access succeeded through the existing connection. The capture contains 130,728 rows over 92 requested dates, including next trading sessions and October screenshot symbols, with no subscription or order changes. The dense replay scans 315 symbol/session panels and retains 24,360 of 24,534 possible five-minute candles (99.29%). Incomplete candles remain gaps. Daily features use only prior completed sessions; exact validated daily inputs are copied into the final run.

The new rule engine remembers a touched level for three candles, confirms a reaction, and then assigns invalidation, targets, trim state and trailing management to that one position. It scans both directions independently of ACE's publication time. Two-session paths remain censored if continuation data is unavailable. Partial-exit and full-exit matches require an already matched entry from that same algorithm position.

The existing Intraday Structure engine was also run with its current price detectors and management logic, both directions refreshed on a fixed 15-minute schedule. Broker execution and absent dealer context were disabled. Its opportunity paths can overlap and are not a portfolio/order simulation. Scoring uses bar completion and the first terminal transition after entry, rather than a later archival CLOSED timestamp.

## Entry timing on the same 25 historical comparison alerts

One-to-one matching requires the same symbol and direction; a signal cannot cover multiple source entries.

| Variant | Matches ±10 minutes | Matches ±15 minutes | Generated entries | Shifted-time recall ±10 minutes |
| --- | ---: | ---: | ---: | ---: |
| Previous simultaneous rule, continuous scan | 0/25 | 0/25 | 15 | 1.56% |
| Any-daily-MA touch/reaction, intraday management | 0/25 | 1/25 | 38 | 3.88% |
| Any-daily-MA touch/reaction, two-session management | 1/25 | 1/25 | 26 | 2.84% |
| SMA5-only touch/reaction, two-session management | 0/25 | 1/25 | 16 | 2.88% |
| EMA21-only touch/reaction, two-session management | 1/25 | 1/25 | 13 | 0.64% |
| Intraday VWAP/opening-range retest/reaction | 4/25 | 4/25 | 82 | 11.32% |
| Existing Intraday Structure price-only engine | 5/25 | 9/25 | 291 | 28.60% |
| Exact prepublished underlying entry-zone arm | Unscored for absent compatible plans | Unscored | 0 | Not estimable |

The retest rule's 16% recall is above its 11.32% shifted baseline, but its session-weighted lift interval is −9.82 to +28.50 percentage points, and its MDE is 28.74 points. The existing engine's 20% recall is below its 28.60% shifted baseline; its lift interval is −23.30 to +16.90 points, with MDE 30.58 points. Both comparisons are too small to establish a modest timing advantage. These MDEs refer to timing recall, not expected return.

The larger ±15-minute overlap in the existing engine comes with 291 opportunity entries. That frequency matters: a busy detector can land near many alerts incidentally. Absence from the incomplete Discord export is not proof that every extra signal was a bad trade, but it prevents interpreting those overlaps as faithful imitation.

On all 201 expanded decisions, the intraday retest rule matches 28 entries within ten minutes. It jointly matches two of 25 high/medium-confidence trim labels. The any-MA two-session rule jointly matches one of eight high/medium-confidence full-close labels. In the primary 25-alert comparison there are three high/medium trim labels and no high/medium full closes. The latter is unavailable exit evidence, not a measured zero exit win rate; low-confidence estimates are reported separately.

The exact-published-zone extractor found four unambiguous underlying plans. Two apparent entry prices were actually option premiums and were quarantined; three other multi-symbol/availability cases were retained as ambiguous. No compatible prepublished plan triggered on the scored panel. The zero-signal arm therefore does not reject precomputed entry zones.

## What the empty cells previously meant

The older outcome tables required both the source alert and a control to pass every gate. Zero qualifying pairs produced null cells even though the underlying bars existed. Its one-observation bootstrap interval was also degenerate, and a zero MDE for constant same-session daily states was a design artifact. The new reports use explicit counts/status reasons, full generated events, per-alert gate/position explanations and session-level uncertainty. No-signal policies have no estimated MDE.

## Scope still open

ARM, ALAB and COHR have no entry labels in the older export; NBIS has only one. An aggregate result from that cohort cannot settle the newer October AI/momentum basket. Independent SMA5/EMA21 scans of ARM, INTC, NBIS, ALAB, COHR and MU on October 6–8 are saved, including their generated management events and complete daily input snapshots. Their exact screenshot-time recall remains pending capture-date/display-timezone information.

The next testable variant is a typed chain: source-available watchlist → exact zone-tag notification → confirmation/entry → partial profit/stop update → terminal exit. Evaluate actual dated October plans/alerts against that chain, while separately testing candidate/theme selection. The current data supports further underlying research. It does not yet identify the custom cycle/divergence/expected-move calculation or demonstrate a strategy suitable for autonomous deployment.

## Artifacts and validation

- Final seven-rule replay: [report](ace_sequence_replay_20261009_05/report.md), [results](ace_sequence_replay_20261009_05/results.json), [per-alert audit](ace_sequence_replay_20261009_05/entry_match_audit.jsonl), [candle overlays](ace_sequence_replay_20261009_05/representative_entry_exit_overlays.png).
- Existing engine: [report](ace_existing_structure_20261009_01/report.md), [events](ace_existing_structure_20261009_01/algorithm_events.jsonl), [deterministic null rescoring](ace_existing_structure_20261009_01/deterministic_null_scoring.json). The rescoring fixes randomization-key ordering; generated paths and match counts are unchanged.
- Screenshots: [visual evidence transcription](ace_screenshot_evidence_20261009.json), [family-specific October case report](ace_screenshot_cases_20261009_01/report.md).
- Historical data: [immutable SIP capture manifest](ace_historical_sip_20261009_01/manifest.json). Prior IEX outputs and previous experiment runs were preserved.
- Reproducible commands are the CLI entry points in `scripts/discord_ledger/run_ace_sequence_replay.py`, `run_ace_existing_structure_replay.py`, `run_ace_screenshot_cases.py` and `fetch_ace_replay_bars.py`; the protocol is [recorded here](ace_sequence_methodology_20261009.md).

Verification: 61 Discord-ledger tests pass; compilation and diff checks pass; randomization results are identical under two different Python hash seeds; representative overlays were inspected. No live strategy, trading flag, broker order or model deployment was changed.
