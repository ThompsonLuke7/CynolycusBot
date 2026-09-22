# Vault of Ace: auditable managed-trade evidence ledger

Research reconstruction of the supplied export. Trader messages are unverified evidence; this is not a broker statement or a validated performance backtest.

- Archived messages: **3,962**; source-derived events: **726**.
- Entry-associated lifecycles: **365**; order/orphan records: **145**.
- Caller entry counts: `{'FT': 143, 'ACE': 222}`. ACE and FT are separate traders.
- Ambiguous/unmatched links: **212**; provenance/time issues: **0**.
- Message coverage: `{'no_extracted_trade_event': 1993, 'attachment_requires_review': 437, 'unannotated_trade_language_review_required': 361, 'annotated': 594, 'other_author_preserved_not_caller_trade': 577}`.

## Scope and evidence limits

All seven VIP exports contain zero messages. Empty exports do not prove no VIP trading occurred. The five populated free channels span October 2025–September 2026. Image-only content and unresolved management remain visible in coverage and attachment reports; no missing trade is treated as a breakeven.

The footer timezone is fixed UTC-5, not a daylight-saving timezone. Discord snowflake creation times provide millisecond UTC chronology and are checked against rendered timestamps. Edited messages preserve their final visible version and edit marker; the original text/history is unavailable. Final edited text is eligible only after the conservative end of its edit minute.

Every ledger/event field has `field_provenance` with source message IDs and its interpretation basis. High confidence means the extraction/link is clear, not that the trader's claim is verified. Prices remain reported references; fill quantities and P&L stay null. CSVs contain JSON provenance cells; the JSONL files are canonical.

## Point-in-time research and follower estimates

Each lifecycle has a research row and a readable case entry. Prior same-caller/symbol context is restricted to the preceding 30 days and eligible message versions. Later recaps, corrections and outcomes do not improve entry knowledge. Local historical market observations, where available, are clearly identified as retrospectively fetched and cannot certify original publication/revision history.

Follower scenarios explicitly model 30-, 60- and 300-second delays from evidence availability. A numeric option fill requires the identified contract's contemporaneous executable quotes, sizes and session eligibility. Trade bars and profit screenshots do not provide that evidence. Missing estimates are null with per-trade reasons, never replaced by alert prices or synthetic option marks.

## Files

- `managed_trades.jsonl` / `.csv`: conservative lifecycle ledger, including unresolved management records.
- `events.jsonl` / `.csv`: every annotated trade, management, watchlist and performance claim.
- `trade_research.jsonl`: per-lifecycle as-of context and follower scenarios.
- `trade_cases.md`: source-linked chronology and research for every lifecycle.
- `unresolved_links.jsonl`, `message_coverage.csv`, `validation.json`: uncertainty and audit coverage.
- `index.html`: local searchable ledger; `manifest.json`: input/output hashes and methodology version.

## Method sources

- [Discord snowflake format](https://docs.discord.com/developers/reference#snowflakes): message creation timestamp decoding.
- [Alpaca historical options data](https://docs.alpaca.markets/us/docs/historical-option-data): feed distinctions; indicative quotes are not actual OPRA quotes.

See the parent directory's `data_feasibility.md` and annotation/attachment review notes for detailed coverage findings.
