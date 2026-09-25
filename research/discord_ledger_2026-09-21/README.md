# Vault of Ace Discord research package

This package reconstructs the supplied Discord HTML export as source-cited evidence for researching callout behavior. It is not a broker statement, a performance audit, or a live-trading recommendation.

**Profitability-audit correction:** the earlier statement that only six ACE lifecycles have entry/exit evidence was a *text-priced-pair count*, not an audit of all screenshot-based closes. `image_management_recovery_20260921_03/` now indexes 38 uniquely contract-matched screenshot-management candidate rows across 26 distinct entry lifecycles (31 with displayed option prices). Some screenshots show order-fill confirmations; many show current marks rather than fills. The older six-pair count must not be presented as total closure coverage or a trader win rate. The ledger remains a conservative text-event reconstruction and needs image evidence promoted through review before any profitability result.

## Start here

- `run_curated_20260921_05/` is the canonical ledger run. Open `report.md`, then `index.html` for a searchable local view.
- `process_profile_20260921_04/process_profile.md` summarizes separately identified ACE and FT callout habits. `signal_features.jsonl` is the row-level, source-cited dataset for modeling/research.
- `attachments/reviewed_follower_evidence_v4.md` contains seven visually reviewed, same-contract member screenshot comparisons. They are evidence of reported member participation, not proof of causality or a representative follow-rate.
- `market_quotes/reviewed_iex_20260921_03/` holds read-only historical IEX *underlying* quotes around those seven alerts, including raw rows and 0/30/60/300-second snapshots. It contains no options quotes or inferred option fills.
- `data_feasibility.md` documents exact coverage and what the available data can/cannot establish.
- `strategy_audit_20260921_02/report.md` joins preceding multi-symbol watchlist-channel messages to entries, audits source-reported outcome coverage, and defines the separate rule-learning gates. Its JSONL files retain the row-level watchlist, outcome, and screenshot-snapshot evidence.
- `spy_trigger_pilot_20260921_02/` contains pre-alert SPY minute snapshots and same-day comparison times for a research-only directional-trigger pilot; comparison times are not verified losing trades.
- `image_management_recovery_20260921_03/` is the screenshot-close recovery index. It includes the INTC 108 call example and deliberately distinguishes marks, ambiguous "gone" wording, and OCR order-confirmation screens.
- `estimated_outcomes_20260922_03/report.md` is the estimate-based reconciliation: 11 selected ACE close-price references with evidence tiers, 13 more priced image entries without a defensible full-close reference, and unassigned loss posts. It is not an overall win rate or account P&L.
- `underlying_proxy_20260922_07/report.md` is the broad all-entry pass: 222 ACE entries, source-linked management anchors plus clearly separate expiry-horizon proxies, and an exploratory pre-alert comparison with existing VWAP/momentum/volume system features.
- `post_profits_results_20260922_01/report.md` indexes the already-normalized `📈-post-ur-profits` export as 69 ACE source-reported result claims and keeps member result language separate from ACE claims.
- `post_profits_results_20260922_04/report.md` adds the 174-message supplemental-export delta and indexes 91 ACE plus 15 FT source-reported result claims; `ace_ft_pooling_assessment_20260922_01.md` documents why ACE/FT management outcomes should remain separate while sharing a common feature vocabulary.
- `combined_outcome_coverage_20260922_01.md` gives the evidence-tier counts across all 369 ACE/FT entry lifecycles, including the separately reconstructed FT underlying-proxy outcomes.

## Evidence boundaries

- Raw normalized messages live in `source/messages.jsonl`; original HTML remains immutable in `VaultOfAceDiscordLogs/`.
- Every curated event and lifecycle field has `field_provenance` with message IDs. `unresolved_links.jsonl` intentionally retains ambiguous management instead of forcing a match.
- ACE (`1079391263083733072`) and FT (`720901855995101225`) are separate authors; do not pool them as one strategy.
- Seven VIP exports contain zero messages. This is an export limitation, not evidence of no VIP activity.
- The Discord footer says UTC-5, but message snowflakes and rendered timestamps establish `America/New_York` display behavior including DST. Edited messages preserve only their final exported version.
- Caller reference prices and visually reviewed screenshot prices are retained as reported claims. They are not assigned as generic follower fills. The historical IEX evidence is a single-exchange underlying quote feed, not OPRA options BBO.

## Reproduce

```bash
./.venv/bin/python -m scripts.discord_ledger.curate
./.venv/bin/python -m scripts.discord_ledger.build --out research/discord_ledger_2026-09-21/<fresh-run-directory>
./.venv/bin/python -m scripts.discord_ledger.strategy_patterns --run research/discord_ledger_2026-09-21/<fresh-run-directory> --out research/discord_ledger_2026-09-21/<fresh-profile-directory>
./.venv/bin/python -m scripts.discord_ledger.strategy_audit --run research/discord_ledger_2026-09-21/run_curated_20260921_05 --out research/discord_ledger_2026-09-21/<fresh-audit-directory>
./.venv/bin/python -m scripts.discord_ledger.spy_trigger_pilot --run research/discord_ledger_2026-09-21/run_curated_20260921_05 --out research/discord_ledger_2026-09-21/<fresh-trigger-directory>
./.venv/bin/python -m scripts.discord_ledger.index_screenshot_returns --ocr research/discord_ledger_2026-09-21/attachments/ocr.jsonl --out research/discord_ledger_2026-09-21/<fresh-audit-directory>/screenshot_return_claims.jsonl
./.venv/bin/python -m scripts.discord_ledger.recover_image_management --study research/discord_ledger_2026-09-21 --run research/discord_ledger_2026-09-21/run_curated_20260921_05 --out research/discord_ledger_2026-09-21/<fresh-recovery-directory>
./.venv/bin/python -m scripts.discord_ledger.reconcile_estimated_outcomes --study research/discord_ledger_2026-09-21 --run research/discord_ledger_2026-09-21/run_curated_20260921_05 --strategy-audit research/discord_ledger_2026-09-21/strategy_audit_20260921_02 --image-recovery research/discord_ledger_2026-09-21/image_management_recovery_20260921_03 --out research/discord_ledger_2026-09-21/<fresh-estimate-directory>
./.venv/bin/python -m pytest scripts/discord_ledger/tests -q
```

The dedicated screenshot review and historical-quote fetchers use explicitly versioned input/output paths. Their manifests retain hashes but never credentials.
