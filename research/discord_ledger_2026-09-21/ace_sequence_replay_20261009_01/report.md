# ACE continuous sequence reproduction — 2026-10-09

This study evaluates continuously generated entries, trims and exits within ±1/2/3 five-minute candles of exported ACE events. All variants run on the same observed symbol/session panel in both directions. The historical comparison dates have been inspected before and these new hypotheses remain exploratory.

## Audit of the previous conclusion

The old program's state_machine function is a stateless simultaneous gate evaluated at sampled alert/control instants. It does not retain a touched level, scan intervening candles, generate a position, or test exits. Its empty outcome cells require both sides of a group to pass and do not mean the original bars are missing. It rejects 52% of test directions at the daily ordering gate and its later gates select zero confirmed source alerts. That is a result for one imposed proxy, not a test of all screenshot hypotheses.

Its one-observation bootstrap interval is degenerate and should not support an inference. Its constant daily-state zero MDE is also a design artifact: same-symbol/session controls share daily values. This rerun uses session-level uncertainty only where at least two independent sessions exist.

## Screenshot findings

All 16 displayed SMA5/EMA21/SMA34/SMA55 values match local October 5 daily bars within 0.01. INTC's entry zone contains EMA21 and ARM's 298 zone is close to SMA5. NBIS/ALAB have BEAR cycle labels alongside bullish planning. The screenshots' literal Due-low values conflict with a simple numerical countdown interpretation. Their message IDs, absolute publication dates and display timezone are missing, so they remain image-cited evidence rather than manufactured UTC labels.

## Data and protocol

- Original lifecycles: 222; expanded covered/deduplicated entry decisions: 201; primary frozen entries: 120.
- Symbol/session panels: 183; complete candles: 11870; omitted incomplete candles: 2368. No missing minutes are filled.
- Prior-session daily history is required and stale histories fail closed. Source-plan availability respects edits. Every modeled entry owns its invalidation, targets and management state; management cannot occur before its entry.
- Source entry times, symbols/directions and management labels are used only in scoring. The observed panel itself supplies candidate symbols/dates; full-universe/theme discovery is not assessed.
- One-to-one matching consumes each generated event at most once. Management matches additionally require a matching entry in the same modeled position. Unmatched generated signals mean no corresponding exported alert; the export is incomplete.

## Primary fixed comparison partition: entry timing

| Policy | Source entries | Algorithm entries | ±1 candle matches | ±2 candle matches | ±3 candle matches | ±2 recall | ±2 associated signal fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_snapshot | 25 | 16 | 0 | 0 | 0 | 0.0% | 0.0% |
| daily_touch_reaction | 25 | 37 | 0 | 0 | 1 | 0.0% | 0.0% |
| daily_touch_reaction_swing | 25 | 24 | 0 | 1 | 1 | 4.0% | 4.2% |
| intraday_retest_reaction | 25 | 70 | 3 | 5 | 5 | 20.0% | 7.1% |
| published_entry_zone | 25 | 0 | 0 | 0 | 0 | 0.0% | not estimable |

## Same-frequency shifted-time baseline (±2 candles)

| Policy | Recall | Shifted recall | Session-weighted lift (pp) | 95% session CI (pp) | MDE (pp) |
| --- | ---: | ---: | ---: | --- | ---: |
| strict_snapshot | 0.00 | 1.16 | -1.04 | [-2.4177083333333336, 0.0] | 1.80 |
| daily_touch_reaction | 0.00 | 3.32 | -2.73 | [-6.270833333333334, -0.3109375000000085] | 4.45 |
| daily_touch_reaction_swing | 4.00 | 2.60 | 0.55 | [-4.624999999999999, 6.709895833333325] | 8.37 |
| intraday_retest_reaction | 20.00 | 9.00 | 13.90 | [-4.251388888888889, 34.864583333333314] | 29.66 |
| published_entry_zone | 0.00 | 0.00 | 0.00 | [0.0, 0.0] | 0.00 |

## Entry and management confluence (primary test, ±2 candles)

| Policy | Management kind | High/medium source labels | Labels after matched entry | Matching full sequences | Status |
| --- | --- | ---: | ---: | ---: | --- |
| strict_snapshot | trim | 3 | 0 | 0 | measured |
| strict_snapshot | exit | 0 | 0 | 0 | no_exported_labels_in_partition |
| daily_touch_reaction | trim | 3 | 0 | 0 | measured |
| daily_touch_reaction | exit | 0 | 0 | 0 | no_exported_labels_in_partition |
| daily_touch_reaction_swing | trim | 3 | 1 | 0 | measured |
| daily_touch_reaction_swing | exit | 0 | 0 | 0 | no_exported_labels_in_partition |
| intraday_retest_reaction | trim | 3 | 0 | 0 | measured |
| intraday_retest_reaction | exit | 0 | 0 | 0 | no_exported_labels_in_partition |
| published_entry_zone | trim | 3 | 0 | 0 | measured |
| published_entry_zone | exit | 0 | 0 | 0 | no_exported_labels_in_partition |

## Expanded exported cohort, all dates (±2 candles)

| Policy | Source entries | Algorithm entries | Entry matches | Entry recall | High/medium trims: full matches/labels | High/medium exits: full matches/labels |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| strict_snapshot | 201 | 107 | 8 | 4.0% | 0/25 | 0/8 |
| daily_touch_reaction | 201 | 237 | 9 | 4.5% | 0/25 | 0/8 |
| daily_touch_reaction_swing | 201 | 198 | 10 | 5.0% | 0/25 | 0/8 |
| intraday_retest_reaction | 201 | 528 | 25 | 12.4% | 1/25 | 0/8 |
| published_entry_zone | 201 | 0 | 0 | 0.0% | 0/25 | 0/8 |

## Modeled underlying path diagnostics on primary test sessions

These are completed-candle price references. No ACE option return, quantity or broker execution is inferred; trims are timing events and do not change the single-unit return diagnostic.

| Policy | Closed/censored | Mean gross (%) | Gross-positive paths | Less 10bps (%) | Less 25bps (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| strict_snapshot | 16/0 | 0.040 | 0.562 | -0.060 | -0.210 |
| daily_touch_reaction | 37/0 | -0.255 | 0.351 | -0.355 | -0.505 |
| daily_touch_reaction_swing | 18/6 | -0.392 | 0.444 | -0.492 | -0.642 |
| intraday_retest_reaction | 70/0 | 0.118 | 0.329 | 0.018 | -0.132 |
| published_entry_zone | 0/0 | not estimable: no closed paths | not estimable: no closed paths | not estimable: no closed paths | not estimable: no closed paths |

## Why an alert was not reproduced

`entry_match_audit.jsonl` lists every primary alert for every policy, its nearest same-direction entry, ±2-candle status, data coverage and nearby gate states. `state_trace.jsonl` records arming/confirmation and missing/stale data. `management_labels.jsonl` preserves source IDs and confidence; missing exit evidence stays unlabelled. `results.json` includes unchanged train/validation/test dates, all tolerances and the low-confidence sensitivity.

## Next testable variant

Use the exact prepublished support/target zones and timestamped October entry-hit/trim/exit export with the existing Intraday Structure detector and exit policy. Those fields would distinguish cycle timing, level selection and discretionary confirmation. Keep candidate selection/theme tests separate until their historical observation coverage is verified. This study does not exhaust the screenshot-derived strategy family.
