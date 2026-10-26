# Updated ACE export: timestamp reconciliation and dated plan replay

Canonical outputs: [refreshed ledger](ledger_01/index.html), [ledger validation](ledger_01/validation.json), [dated replay report](dated_plan_replay_04/report.md), [full-session chart](dated_plan_replay_04/dated_notification_and_policy_paths.png).

## What changed

The user replaced the root exports and confirmed Eastern display time. All 13 current HTML files were copied byte-for-byte into `raw_exports/` before processing. The current exports contain 5,117 observations of 4,279 unique message IDs: 143 more than the prior source view. No prior IDs disappeared and no semantic same-ID changes were detected. Repeated observations and content hashes are preserved; rotating CDN URL signatures are not treated as edits. Seven VIP channel exports remain empty.

The refresh transfers all 1,021 previous annotations only after comparing every cited source version. It adds three text annotations and sixteen visually reviewed typed annotations: five charts, three zone notifications, one target notification, three option Buy references, three option marks, and one partial-exit claim. A duplicate MSTR extraction from the same source clause was retained in the rejection audit rather than counted twice. Twelve targeted original attachments were downloaded and hashed.

The refreshed ledger contains 538 lifecycle records: 373 entry claims/instructions, 159 orphan management records and six order intents. There are 380 ACE records and 158 FT records. These are not 538 distinct broker-filled, closed trades. Four new entry instructions/references are MSTR, two separate GEV contracts, and MU. Unknown fill quantities and realized P&L remain null.

New source IDs by channel: 52 free alerts, 60 free chat, 29 post-profit messages and two watchlist messages. The latter source material is preserved even where unreviewed images contain no extractable caller trade. The ledger's coverage file explicitly leaves 430 attachment-bearing messages and 282 trade-language messages needing further review across the whole corpus; this update is not a claim of exhaustive image extraction.

## Resolved dates and event types

All times below are Eastern. Creation timestamps come from Discord snowflakes, cross-checked against the export's displayed titles. The metadata footer says UTC-5, but these October dates are EDT (UTC-4). These are the available free-channel publication clocks, not independently recovered private-channel execution or fill clocks.

| Evidence | Publication | Source message ID |
| --- | --- | --- |
| ALAB chart plan | October 6, 01:25:49 | 1556900265440055336 |
| ARM chart plan | October 6, 01:26:32 | 1556900449167081553 |
| COHR chart plan | October 6, 01:26:54 | 1556900537985663009 |
| INTC chart plan | October 6, 01:27:22 | 1556900656491798558 |
| NBIS chart plan | October 6, 01:28:02 | 1556900823936802857 |
| ALAB TP1 notification | October 6, 09:41:15 | 1557024947400872000 |
| INTC zone-hit notification | October 6, 12:41:16 | 1557070249302565030 |
| MU Buy-reference image | October 7, 01:26:08 | 1557262734565969931 |
| ARM zone-hit notification | October 7, 09:39:02 | 1557386778673946758 |
| MU “mostly out” plus mark | October 7, 12:09:33 | 1557424653889175592 |
| NBIS zone-hit notification | October 8, 09:36:18 | 1557748479021809698 |

All five chart SMA5/EMA21/SMA34/SMA55 values match the frozen daily inputs through October 5 within 0.01. The previously cropped ARM and NBIS images now expose their prepublished entry levels: 298 and 231.88. The targets were also already published in the plans. The reviewed JSONL supplies per-field source IDs, source anchors, raw timestamps, reply IDs, edits and downloaded image hashes.

## What the dated test establishes

The observer scans the entire October 6–8 path after each plan, not just the notification's eventual session. It uses exact prepublished zones and only completed five-minute candles; incomplete candles are not filled. Source notification timestamps are used only for scoring.

INTC's first completed zone observation is October 6 at 12:40, versus the 12:41:16 notification. NBIS's is October 8 at 09:35, versus 09:36:18. ARM opened below 298 on October 7: the gap-aware threshold observation is 09:35, versus 09:39:02; an actual candle range did not cross back through 298 until the 09:55–10:00 candle. No 298 fill is inferred. ALAB's first completed TP1 observation is 09:35 on October 6, versus 09:41:15. Thus the gap-aware observer matches three of three zone notifications within one candle and the target within two. Gap handling was added after inspecting ARM: this is an exploratory explanation, not an untouched-test result.

This is conditional reproduction of notifications given ACE's plans, not independent generation of his candidate list, levels, confirmed entries, option contracts or management. Existing reaction/position policies were rerun separately: zero of three notification-clock overlaps within fifteen minutes for published-zone, SMA5-only and EMA21-only reactions; the intraday VWAP/ORB reaction overlaps one of three within ten minutes. These zone clocks are not actual entry labels. All modeled trims/exits refer to their own earlier modeled position entry. Selected-case MDE is not estimable; these counts are not a statistical strategy rejection. The prior historical timing benchmark had 28.7–30.6 percentage-point MDEs for its main policies.

MU's premarket image shows a 3.28 Buy reference, not an execution timestamp. Its later reviewed marks are 6.25, 7.73, 8.15 and 8.95. The nearby exported 10:41 image is not the user's 10:45 image showing 6.45; those remain separate. The paired COHR/NBIS option-mark screenshot also remains unmatched rather than being merged into a different underlying-price watchlist image. Expiry years in option annotations are calendar inferences from the literal M/D and dated source, not written contract years. MU's marks omit expiry, so their association with the earlier same-symbol/same-strike reference is a hypothesis, not a broker-confirmed contract match.

A hypothetical MU underlying follower entering at the first RTH minute open after the premarket reference and comparing against the next observed minute open after the partial-exit message gives gross moves of +6.04% (30/60-second exit-publication delay) and +6.17% (300 seconds). This is not ACE's entry/exit or option P&L, includes no costs, and assumes first-RTH entry rather than discovering a confirmation signal. No option trade-print valuation or fabricated bid/ask is used.

NBIS's October 6 “8% higher” post precedes its October 8 zone-hit post. It cannot be used as a realized winner for the later notification. Likewise, target notifications and “consider taking profits” suggestions are not executed exits. This prevents winner screenshots from being attached to the wrong later setup.

## Validation and remaining scope

Ledger validation resolves every field citation, author and channel, with unique event IDs, zero future/unknown evidence issues and null confirmed quantities/P&L. The raw source, daily-input and SIP-session hashes are verified. Focused regressions cover numbered export filenames, unknown authors, Unicode line separators, version preservation, causal candle availability, gap-vs-range semantics, partial exits, RTH delay benchmarks and plot coordinate parity.

The live server, candidate-tape flag, paper/live strategy configuration and broker orders were not changed. Intermediate failed or superseded staging/replay directories were preserved, not overwritten; `dated_plan_replay_04` is canonical.

The next testable variant is independent, point-in-time candidate/support planning (SMA5/EMA21 plus prior confirmed pivots), then separate gap/touch arming, confirmation and position-linked management on additional unseen dated examples. Private cycle, divergence and expected-move formulas remain unresolved, not disproven. Preserve the original historical split and use new examples for confirmation, not tuning the old test set.

Reproduction entry points are `stage_export_refresh`, `extract_alerts`, `extract_challenge`, `extract_context`, `reconcile_export_refresh`, `build`, and `run_ace_dated_plan_replay` under `scripts.discord_ledger`. Every rerun must use fresh staging and ledger/replay output directories.
