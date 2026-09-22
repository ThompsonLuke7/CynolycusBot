# Vault of Ace Discord research package

This package reconstructs the supplied Discord HTML export as source-cited evidence for researching callout behavior. It is not a broker statement, a performance audit, or a live-trading recommendation.

## Start here

- `run_curated_20260921_05/` is the canonical ledger run. Open `report.md`, then `index.html` for a searchable local view.
- `process_profile_20260921_04/process_profile.md` summarizes separately identified ACE and FT callout habits. `signal_features.jsonl` is the row-level, source-cited dataset for modeling/research.
- `attachments/reviewed_follower_evidence_v4.md` contains seven visually reviewed, same-contract member screenshot comparisons. They are evidence of reported member participation, not proof of causality or a representative follow-rate.
- `market_quotes/reviewed_iex_20260921_03/` holds read-only historical IEX *underlying* quotes around those seven alerts, including raw rows and 0/30/60/300-second snapshots. It contains no options quotes or inferred option fills.
- `data_feasibility.md` documents exact coverage and what the available data can/cannot establish.

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
./.venv/bin/python -m pytest scripts/discord_ledger/tests -q
```

The dedicated screenshot review and historical-quote fetchers use explicitly versioned input/output paths. Their manifests retain hashes but never credentials.
