# Context extraction review

- Input: `research/discord_ledger_2026-09-21/export_refresh_20261103_01/source/messages.jsonl`.
- Scoped messages read: 2661 (channels: free-chat=249, free-watchlists=1282, post-ur-profits=1130).
- Caller messages considered: 1753 (ACE=1570, FT=183).
- Candidate messages classified: 375; context rows written: 651.
- Event counts: add=2, entry=4, exit=2, invalidation=1, performance_claim=254, stop_adjustment=37, trim=1, unknown_management=229, watchlist=121.
- Unresolved attachment/contract notes: 229.

## Interpretation constraints

- ACE and FT are separate callers. Other members' messages are outside this file even where they describe a trade.
- A performance claim remains a claim; it is not an exit, fill, return verification, or lifecycle link.
- Conditional breakout, support, stop, or target language is stored as watchlist context, never as a filled order.
- Options require ticker, call/put, strike, and expiry before a claimed entry/exit can be lifecycle-eligible. Missing terms remain null and are recorded as unresolved management context.
- Attachment URLs are evidence references only. No image was fetched or OCRed; image-only messages retain an explicit unresolved note.
- `field_sources` in every JSONL row maps each non-null substantive field to its originating message ID. Future messages are not used to enrich entry context.
