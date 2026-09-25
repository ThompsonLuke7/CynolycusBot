# ACE point-in-time context selectivity

Research-only association study: ACE candidates are compared with frozen, same-symbol/session non-alert controls. Controls are weighted so every source alert contributes one total control weight.
No forward returns, source outcome claims, same-day news aggregates, option prices, or execution assumptions are inputs.

## Coverage

- Alerts: 120 (effective weight 120.0).
- Controls: 1231 (effective weight 120.0).
- Universe snapshots resolved: alerts 6, controls 58. Missing snapshots are left unresolved rather than substituted with today's universe.
- Discord features use conservative creation/edit availability. News uses article timestamps only and is not certified point-in-time because the local index has no historical observation/ingestion time.

## Selectivity

| Feature | ACE alerts | Controls | Difference |
| --- | ---: | ---: | ---: |
| watchlist_24h | 3.3% | 5.7% | -2.4% |
| watchlist_7d | 13.3% | 15.8% | -2.4% |
| watchlist_level_7d | 4.2% | 5.8% | -1.6% |
| watchlist_intention_or_instruction_7d | 4.2% | 4.2% | -0.1% |
| news_24h_has_article | 45.8% | 46.7% | -0.9% |
| in_eligible_universe | 100.0% | 100.0% | 0.0% |
| news_24h_article_count (mean) | 9.967 | 9.840 | 0.127 |
| news_24h_direct_catalyst_count (mean) | 4.633 | 4.620 | 0.014 |
| news_24h_max_record_catalyst_score (mean) | 0.358 | 0.362 | -0.004 |
| news_24h_mean_p_bullish (mean) | 0.264 | 0.265 | -0.001 |

## Inputs

- alerts: `research/discord_ledger_2026-09-21/ace_setup_outcomes_20260923_02/source_alert_outcomes.jsonl`.
- controls: `research/discord_ledger_2026-09-21/ace_setup_outcomes_20260923_02/matched_controls.jsonl`.
- context: `research/discord_ledger_2026-09-21/annotations/context_utc_20260923_01.jsonl`.
- messages: `research/discord_ledger_2026-09-21/source/messages.jsonl`.
- news: `Data/runtime/news_library_index.parquet`.
- universe_snapshots: `Data/shared/universe/snapshots`.

## Interpretation

This evaluates association with the observed ACE alert timing, not whether an independent system discovers the candidate universe, predicts profitability, or can execute options. Small and incomplete universe coverage is a data limitation, not a negative value.
