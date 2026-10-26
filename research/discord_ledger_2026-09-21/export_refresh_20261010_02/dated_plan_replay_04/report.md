# ACE dated October plan replay

The updated export resolves publication clocks in Eastern time. Plans were available before all scored notifications. The engine scans October 6–8 continuously; it is not given the notification date or clock. Chart levels are supplied by ACE, so this tests notification mechanics, not autonomous watchlist/level discovery.

| Event | ACE publication (ET) | First range-only observation (ET) | First gap-aware observation (ET) | Gap-aware timing error |
| --- | --- | --- | --- | ---: |
| INTC underlying_zone_tag_claim [1557070249302565030] | 2026-10-06 12:41:16 | 2026-10-06 12:40:00 | 2026-10-06 12:40:00 | -1.27 min |
| ARM underlying_zone_tag_claim [1557386778673946758] | 2026-10-07 09:39:02 | 2026-10-07 10:00:00 | 2026-10-07 09:35:00 | -4.05 min |
| NBIS underlying_zone_tag_claim [1557748479021809698] | 2026-10-08 09:36:18 | 2026-10-08 09:35:00 | 2026-10-08 09:35:00 | -1.32 min |
| ALAB underlying_target_tag_claim [1557024947400872000] | 2026-10-06 09:41:15 | 2026-10-06 09:35:00 | 2026-10-06 09:35:00 | -6.26 min |

The gap-aware observer matches all three zone notifications within one five-minute candle and ALAB's TP1 notification within two. Range-only misses ARM at the one/two/three-candle tolerance: the session opened below 298 and traded back through it later. No 298 fill is inferred. This gap-aware variant was added after inspecting that discrepancy and needs new examples, not a success claim based on these same cases.

## Entry and exit policy confluence

Zone touches are not confirmation entries. These unchanged reaction policies own their entry, stop, targets, trims and exits through a position ID. Below are overlaps with zone-notification clocks, not recall against actual option entry clocks.

| Reaction policy | All generated entries (six symbols, three sessions) | ±5m zone-clock overlaps | ±10m | ±15m |
| --- | ---: | ---: | ---: | ---: |
| published_entry_zone | 4 | 0/3 | 0/3 | 0/3 |
| sma5_touch_reaction_swing | 12 | 0/3 | 0/3 | 0/3 |
| ema21_touch_reaction_swing | 10 | 0/3 | 0/3 | 0/3 |
| intraday_retest_reaction | 51 | 0/3 | 1/3 | 1/3 |

This is a selected-case diagnostic, not an inferential null test: its MDE is not estimable, and zero overlaps do not reject the strategy. The earlier historical entry-timing benchmark's main policies had 28.7–30.6 percentage-point MDEs. New variants and population claims need additional independent cases.

No actual option entry-time label or full-close label is established for these October chains. MU's 01:26 buy reference is outside the regular option session and 'mostly out' is partial. Joint actual-entry/full-exit recall is explicitly unavailable, not zero. All generated management events pass the owning-position/prior-entry check.

## MU evidence and estimated underlying follower scenario

The export's 10:41 image shows 6.25, not the user's 10:45 image's 6.45; these are kept separate. Verified later images show 7.73, 8.15 and 8.95, with a 3.28 premarket Buy reference. Reference-to-mark changes are not realized returns. The 1000% text claim has a different/unresolved basis; the reviewed marks do not establish that return.

| Delay | Hypothetical underlying entry | Partial-exit-time price benchmark | Gross underlying change |
| --- | ---: | ---: | ---: |
| 30 sec | 1017.37 | 1078.83 | +6.04% |
| 60 sec | 1017.37 | 1078.83 | +6.04% |
| 300 sec | 1017.37 | 1080.13 | +6.17% |

This is an explicitly hypothetical first-RTH-entry/partial-publication-exit underlying benchmark, not ACE's realized trade or a tradable option fill estimate. Quantities, option execution, bid/ask, spread, cost, and realized P&L remain unestimated. Historical SIP OHLCV is used without missing-minute interpolation; vendor availability vintage is not independently certified.

## Next testable variant

Use the now-resolved prepublished plans as separate labels for candidate choice, support selection, threshold notification, confirmation entry and management. Test an independently generated SMA5/EMA21 plus prior-pivot support planner on new dated examples; retain gap-through as a separate armed state, not an assumed fill. Freeze selection before scoring. Do not impose BULL-only cycle or divergence vetoes from these conflicting panels. The private cycle/expected-move formulas remain unidentified; this hypothesis family is not exhausted.

[Full-session candles, exact source clocks and position-linked reaction signals](dated_notification_and_policy_paths.png). See the canonical JSONL for per-event source IDs, model position IDs, full state transitions and price references.
