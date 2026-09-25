# ACE strategy reconstruction: first joined audit

This is a research-only extension of `run_curated_20260921_05`. The source HTML and canonical ledger were not changed. ACE and FT are not pooled. All joins are candidate links, not proof of a trading rule, broker fill, or profitable strategy.

**Correction after visual challenge:** The outcome table below counts only pairs with *text-parsed* entry and exit/trim prices. It does **not** count screenshot-based closes, and must not be used as closure coverage or trader win rate. See `../image_management_recovery_20260921_03/` and the correction in `../README.md`.

## Watchlist date and idea

`watchlist_entry_candidates.jsonl` searches *all* same-caller watchlist-channel messages, not just the 111 ACE / 10 FT watchlist annotations. A literal same-symbol mention must have become available in the prior seven days; edited text is only eligible after the end of its displayed edit minute. Multi-symbol posts can therefore match every explicitly mentioned symbol. The output retains full text, post/availability timestamps, message IDs, and confidence. This corrects the one-symbol-per-annotation limitation, but a watchlist-channel mention is still not necessarily a prospective idea.

| Caller | Entry lifecycles | Prior same-symbol mention | Prior message with idea-like wording |
| --- | ---: | ---: | ---: |
| ACE | 222 | 54 | 22 |
| FT | 147 | 0 | 0 |

The seven-day window and keyword taxonomy are research choices, not inferred trader rules. Example: multi-symbol message `1452668454292164700` says ACE wanted APP 830 calls and RDDT 240/245 calls; it precedes source-annotated entries `1452670301950316615` (APP 830 call) and `1452670654603067517` (RDDT 245 call). The matching symbols and timing are auditable, but the causal link remains a candidate.

## Underlying trigger: SPY pilot

`spy_trigger_pilot_20260921_02/snapshots.jsonl` freezes completed SPY one-minute bars before each ACE SPY entry's conservative availability time and samples five-minute comparison times on the *same 29 alert days*, excluding a 30-minute radius around any ACE SPY alert. There are 49 entry rows; 46 have a usable last completed minute and 1,638 same-day comparison rows. On those 46, 36 (78.3%) had a five-minute underlying move aligned with call/put direction; the per-alert same-day comparison expectation is 50.9%. This is an exploratory description, **not** a validated trigger, statistical significance test, return prediction, or evidence that the comparison times were non-trades. The export's VIP channels are empty; the local final-bar cache has unknown original feed/adjustment/publication time. The one-minute snapshot code fails closed on missing bars and does not use the bar in progress at alert time.

Candidate rule to test out of sample: directional five-minute SPY movement combined with a pre-alert watchlist/level state, with separate sleeves for momentum, reversals, hedges and 0DTE. A single momentum rule cannot explain all 49 alerts; compare opening-range, VWAP, dealer-level and catalyst states without using after-alert prices or text. Freeze rule definitions before an untouched future period.

## Contract selection and management

ACE's 222 entry-associated lifecycles include 219 annotated options; 147 have symbol, call/put, strike and expiry, 194 have a source-quoted premium, and 52 of 149 dated option entries expire on the alert date. These are **observed callout properties**, not liquidity rules. In the 46 usable SPY minute snapshots with a strike and call/put, the median strike distance in the out-of-money direction is 0.67% for 16 calls and 0.30% for 30 puts. The local retrospective bar feed and option identity annotation can be wrong; the high-tail outliers and hedge/lotto cases need separate review.

A testable contract ranker needs the chain as it existed at decision time: strike, expiry, bid, ask, displayed sizes, quote age, same-day traded volume and last-published open interest. Pre-register spread/depth/freshness gates and compare the called contract against *all eligible alternatives*, not just the selected one. Low quoted premium does not establish liquidity. The existing dealer/option selection code can supply implementation patterns, but this 0DTE study must stay research/paper-only until validated. For management, the current ledger conservatively links management to 34 ACE entry lifecycles; unresolved events must not be silently treated as held-to-expiry, full exits, or losses.

## Strict text-price outcome subset—not full closure coverage

`outcome_evidence.jsonl` distinguishes source-reported price changes from complete quantity cycles and broker-verified results. It rejects percentage phrases such as "AT 100%" (`1479510499426042018`) and "At 50%" (`1534201637000839348`) that the permissive event extractor had rendered as option prices. The canonical source events remain immutable; this is a stricter analytic gate.

| Caller | Entry lifecycles | Text-linked trim/exit lifecycle | Text-priced entry-to-trim/exit pairs | Complete text-claimed quantity cycles | Independently verified P&L |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACE | 222 | 32 | 6 | 1 | 0 |
| FT | 147 | 2 | 1 | 0 | 0 |

One illustrative ACE sequence: WMT 124 call 2/13, claimed buy 2 at 0.80 (`1467893759936037009`), claimed sale of 1 at 1.70 (`1467929148705083708`) and 1 at 1.95 (`1467958821623304334`). The quantity-weighted *reported-price* change is +128.125% gross before fees/spread. The first sale post includes a visually checked image of a broker-app "order executed" screen for one WMT 124 call at a 1.70 limit; the source image strengthens the sale evidence but is still not independent account access or a complete realized P&L statement. The lifecycle link remains low confidence. The other text-priced examples include a QQQ reported 0.37-to-0.25 loss (`1469372524998426686`, `1469376239323316235`) and partial or instruction-linked gains; unknown adds, quantities, contracts and selection of posted winners prevent a valid population win rate.

## Screenshot-close recovery after the text-only audit

The separate `../image_management_recovery_20260921_03/image_management_candidates.jsonl` pass found 57 image/close-language candidate rows across 56 messages. Thirty-eight rows match a unique preceding same-contract ACE entry, representing **26 distinct candidate entry lifecycles**; 31 uniquely linked rows display a numeric option price. These are not all confirmed exits: the wording includes partial sales and ambiguous "gone" posts, OCR can err, and the displayed price is usually a mark rather than an execution. Eleven rows have no eligible entry candidate and eight have multiple candidates. Therefore the six text pairs above were an undercount of visible lifecycle evidence, not evidence of only six closed trades.

User-supplied example: source alert `1548055480541384826` posted INTC 108 call 9/18 at 1.50 on September 11; source close-language post `1550160332549525574` on September 17 says "INTC CALLS GONE" and its image shows the same INTC 108 call at 3.50. This is a unique contract-level candidate link, visually checked. The displayed mark is **+133.3% versus the alert reference price**; it is not a proven sale fill. The image's **845.95% is explicitly "Today"**, so using it as return since the callout would be incorrect. The trade quantity, any adds/trims, and realized proceeds remain unresolved.

The seven visually reviewed member screenshots in `attachments/reviewed_follower_evidence_v4.md` show that some members reported participation at plausible prices; profit-posting selection bias makes them unusable as the denominator of a win-rate calculation. The additional `screenshot_return_claims.jsonl` indexes 58 OCR candidates showing a "Total return" line: ACE has 38 positive, 2 negative, 1 flat and 1 unparsed; FT has 16 positive. These are posted *snapshots*, often repeated views of an open position, **not** 58 distinct closed trades. I visually verified the negative ACE RIO and XLE open-position screens (`1466809051915616329`, `1469022904585945150`); the remaining OCR classifications are not visually confirmed. A screenshot with entry/exit prices is valid evidence of a *reported* price path, not automatically a validated complete trade. Subscriber count likewise does not measure trading profitability.

## Data-source decision

- Alpaca IEX historical **underlying** quotes worked for seven reviewed cases, but are a single-exchange feed. [Alpaca's historical-options documentation](https://docs.alpaca.markets/us/docs/historical-option-data) distinguishes free *indicative* quotes from subscribed OPRA BBO; the repo's tested dated option-quote route returned 404, while option bars/trades and [latest quotes](https://docs.alpaca.markets/us/reference/optionlatestquotes) exist. Neither bar prints nor an arbitrary latest snapshot reconstruct a historical spread.
- The local Schwab/Thinkorswim adapter has current chain/quote paths and daily option history, not a validated historical intraday option BBO archive. Its OAuth refresh token is documented as expired in the 2026-09-21 repo handoff, so no reauthentication or account query was attempted. No suitable connected data plugin was found in this session.
- Continue to use source-reported prices for a clearly labeled *claim audit*. To estimate what a delayed follower could execute historically, first obtain timestamped two-sided quotes and sizes for a small sample and verify entitlement, freshness, and option/underlying behavior. For future alerts, capture live chain/quote snapshots as append-only paper-research data.

## Modeling gate and review

Do **not** train a profitability classifier on 222 ACE entries and no reliable losing-trade denominator. A first imitation model can compare alert times against eligible same-symbol/session candidate times, but absence of a free-channel alert is only an *unobserved callout*, not a negative trade. Keep idea selection, underlying trigger, contract ranking, and management as separate models/rules. Group train/validation/test chronologically by day (and preferably market regime), hold out the latest period, and evaluate against simple frozen rules and a same-time random baseline. Profitability is a separate paper-execution evaluation with bid/ask, delay, sizing and costs; never use future outcome claims as entry features.

No user review of every message is necessary. The canonical run retains 235 unresolved link records rather than manufacturing certainty. Priority human checks now include the 26 screenshot-linked candidate lifecycles, the 22 keyword-linked idea candidates, and any unresolved link that would change a training label or complete-cycle count. A populated VIP export or broker transaction history would materially improve the denominator and management audit, but are not assumed to exist. No live orders or strategy changes were made.
