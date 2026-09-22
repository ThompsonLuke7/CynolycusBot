# Source-cited callout process profile

This profiles the language and timing of the exported free-channel callouts. It does not validate performance or show what the caller knew privately. ACE and FT are separate authors. Counts are annotated events, not all trades made by either person.

## ACE

- Entry annotations: 223; complete option identities: 143; explicit 0DTE literals: 65.
- Most frequent literal symbols: [('SPY', 49), ('IWM', 15), ('SPX', 15), ('QQQ', 13), ('ORCL', 8), ('NFLX', 6), ('NDXP', 5), ('AAPL', 5)].
- Session timing (ET): {'midday_1200_1530': 98, 'outside_regular_session': 6, 'open_0930_1000': 40, 'morning_1000_1200': 73, 'close_1530_1600': 6}.
- Risk/tenor language: {'size_language': 76, 'lotto': 26, 'stop_language': 26, 'target_language': 11, 'swing_language': 8, 'explicit_high_risk': 2}.
- Setup language in alert: {'hedge': 3, 'dealer_or_expected_move': 3, 'flow': 1, 'breakout_or_level': 1, 'earnings_or_catalyst': 1}; in preceding same-caller/symbol context: {'dealer_or_expected_move': 22, 'breakout_or_level': 26, 'hedge': 6, 'earnings_or_catalyst': 1}.
- Management annotations: {'exit': 65, 'trim': 94, 'add': 13, 'stop_adjustment': 32, 'invalidation': 2}; entry lifecycles with at least one conservatively linked management event: 36.

## FT

- Entry annotations: 148; complete option identities: 108; explicit 0DTE literals: 59.
- Most frequent literal symbols: [('IWM', 21), ('SPY', 15), ('NVDA', 6), ('MSFT', 6), ('QQQ', 4), ('TSLA', 4), ('PLTR', 4), ('GLD', 3)].
- Session timing (ET): {'morning_1000_1200': 57, 'midday_1200_1530': 33, 'open_0930_1000': 54, 'close_1530_1600': 4}.
- Risk/tenor language: {'explicit_high_risk': 4, 'size_language': 2, 'stop_language': 16, 'target_language': 2, 'swing_language': 5, 'lotto': 5}.
- Setup language in alert: {'breakout_or_level': 1, 'earnings_or_catalyst': 1}; in preceding same-caller/symbol context: {'breakout_or_level': 10}.
- Management annotations: {'add': 4, 'invalidation': 1, 'trim': 3, 'exit': 8, 'stop_adjustment': 1}; entry lifecycles with at least one conservatively linked management event: 3.

## Reproduction hypotheses to test

1. Separate *idea/watchlist* from actionable entry. A quoted contract and an at-price idea are not automatically a buy; preserve the later explicit alert time.
2. For 0DTE/lotto callouts, test a low-premium directional-option sleeve only after reconstructing the underlying trigger and using executable quote/spread/size rules. Do not equate quoted premiums with available fills.
3. For swing callouts, encode literal breakout, high-of-day, target and stop language as candidate trigger/risk features. Link each feature only to messages available before entry.
4. Treat hedges, spreads and multi-leg messages as distinct strategies; they should not be mixed into a simple long-call/put result.
5. Run a matched-time, matched-symbol baseline and out-of-sample period before adding any rule to paper trading. No profitability conclusion is implied by follower counts or reported wins.

Each row in `signal_features.jsonl` lists the source message IDs and separates alert text from earlier eligible context. Keyword tags are search aids, not semantic proof; inspect the linked originals for candidate rules.
