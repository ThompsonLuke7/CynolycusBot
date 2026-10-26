# Context extraction review

- Input: `research/discord_ledger_2026-09-21/export_refresh_20261010_02/source/messages.jsonl`.
- Scoped messages read: 2323 (channels: free-chat=174, free-watchlists=1282, post-ur-profits=867).
- Caller messages considered: 1589 (ACE=1406, FT=183).
- Candidate messages classified: 353; context rows written: 613.
- Event counts: add=2, entry=4, exit=2, invalidation=1, performance_claim=237, stop_adjustment=35, trim=1, unknown_management=212, watchlist=119.
- Unresolved attachment/contract notes: 212.

## Interpretation constraints

- ACE and FT are separate callers. Other members' messages are outside this file even where they describe a trade.
- A performance claim remains a claim; it is not an exit, fill, return verification, or lifecycle link.
- Conditional breakout, support, stop, or target language is stored as watchlist context, never as a filled order.
- Options require ticker, call/put, strike, and expiry before a claimed entry/exit can be lifecycle-eligible. Missing terms remain null and are recorded as unresolved management context.
- Attachment URLs are evidence references only. No image was fetched or OCRed; image-only messages retain an explicit unresolved note.
- `field_sources` in every JSONL row maps each non-null substantive field to its originating message ID. Future messages are not used to enrich entry context.
