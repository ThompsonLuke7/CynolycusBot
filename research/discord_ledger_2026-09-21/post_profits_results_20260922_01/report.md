# `📈-post-ur-profits` result-claim index

The supplied export `THE VAULT OF ACE - FREE TRADING SECTION - 📈-post-ur-profits [1416374095456632912].html` was already present in the normalized source package: **664 messages**, including **362 from ACE**. It is now also indexed as a dedicated results-claim dataset rather than relying on the generic event extractor.

## Extracted evidence

- **69 ACE result-claim rows:** 66 positive, 2 negative, and 1 without a signed percentage.
- **21 claims** match exactly one prior ACE entry candidate in the export, representing **13 distinct entry lifecycles**. Of those claims, 20 are positive and one is negative.
- **10 claims** have multiple compatible preceding entries and **38** have no prior matching entry in the supplied free-channel export. They remain claims, not forced links.
- **18 member posts** containing result language are retained separately. They establish only member-reported experience, not that an ACE alert caused it or that it is representative.

The parser creates one row per visible contract leg in a recap. For example, a multi-line recap such as AAPL 310C and SPX 7685P produces two correctly scoped rows rather than assigning AAPL's strike to SPX. Rows retain the source message ID, exact claim text, explicitly reported prices/percentages when present, and any candidate lifecycle ID.

`ace_result_claims.jsonl` is an evidence index—not broker P&L, a complete inventory of winning/losing trades, or a follower-execution study. The overwhelmingly positive claim mix is expected for a channel named “post your profits” and must not be used as a win rate.
