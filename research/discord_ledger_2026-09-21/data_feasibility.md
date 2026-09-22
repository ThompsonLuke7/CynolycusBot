# Discord managed-trade ledger: local market-data feasibility

**Assessment date:** 2026-09-21 (read-only local inventory).  This is a
fitness assessment, not a P&L result and not proof that any Discord alert was
tradeable.

## Decision

The export supports an auditable **message ledger**, source-reported option
prices and screenshots, and retrospective underlying context. These can support
process reconstruction and reported-execution comparisons. A market-based,
delay-specific option follower-fill estimate still needs historical two-sided
quotes (bid, ask, sizes and quote timestamp), which are not locally retained.
Do not substitute an option trade bar, OHLC, VWAP, or last trade for a mark.

Use the new `scripts.discord_ledger.market_context.lookup_context` only for
strictly pre-alert completed *underlying* bar context, with its explicit
non-certified-PIT limitation.  `follower_scenarios(..., instrument_type="option")`
intentionally leaves a numeric market-fill price null until such quotes are
acquired; it does not discard source-reported prices or member screenshots.

## Discord export facts

`VaultOfAceDiscordLogs/` contains 12 HTML exports and 3,962
`.chatlog__message-container` elements.  BeautifulSoup extraction of the five
non-empty free channels yields this coverage:

| Channel | Messages | First message title | Last message title |
| --- | ---: | --- | --- |
| `1k-challenge` | 664 | 2026-01-29 5:16 PM | 2026-07-30 9:35 AM |
| `free-chat` | 114 | 2026-07-02 7:14 PM | 2026-09-18 3:35 PM |
| `ace-free-watchlists` | 1,280 | 2025-11-28 9:21 AM | 2026-09-07 5:50 PM |
| `post-ur-profits` | 664 | 2025-10-22 9:42 AM | 2026-09-18 5:38 PM |
| `ace-free-alerts` | 1,240 | 2025-10-20 10:52 AM | 2026-09-20 8:49 PM |

All seven VIP HTML files have zero message containers/timestamps.  They are an
empty-export limitation, not evidence of zero VIP activity.  A simple text
pattern finds 388 candidate option strings across 96 apparent underlyings
(e.g. SPY, IWM, TSLA, ORCL, AAPL, MSFT, NVDA); this is a discovery aid only,
not a parsed trade record.

The HTML timestamp `title` strings contain a calendar time but no UTC offset.
The Discord message snowflake supplies the creation UTC time, and all 3,962
rendered titles match its minute after interpreting display time as
`America/New_York`; 1,545 summer titles would disagree under fixed UTC-5.
The footer's literal `UTC-5` is retained rather than treated as reliable
year-round. The export retains only the final edited version, not edit history.
For edited text, the builder uses the conservative end of the displayed edit
minute as availability, or unknown if that cannot be resolved.

Reproduce the message counts and span without downloading attachments:

```bash
./.venv/bin/python -c 'from bs4 import BeautifulSoup; from pathlib import Path; [print(f.name, len(BeautifulSoup(f.read_text(encoding="utf-8"), "html.parser").select(".chatlog__message-container"))) for f in sorted(Path("VaultOfAceDiscordLogs").glob("*.html"))]'
```

## Underlying bars: usable only as retrospective context

`Data/shared/bars/` holds 4,096 daily, 3,095 hourly, 3,095 derived 4-hour,
and 16 context Parquet files.  Representative daily coverage is broad: SPY,
IWM, TSLA, AAPL, MSFT, NVDA, QQQ, and AMD run from 2020-07-27 to 2026-09-18;
later listings begin later (PLTR 2020-09-30, COIN 2021-04-14, HOOD
2021-07-29).  The schema is `symbol,timestamp,open,high,low,close,volume,
trade_count,vwap`; timestamps are UTC.

The only broad cached 1-minute underlying file is SPY:
`Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet`, 235,685 rows from
2024-04-01 13:30:00Z to 2026-09-18 19:59:00Z.  Generic raw 1-minute files are
very sparse and episode-specific: for example AAPL has only 390 rows on
2026-08-25; most common alert symbols inspected (IWM, TSLA, ORCL, MSFT, NVDA,
QQQ, AMZN, RKLB, MSTR, HOOD, COIN, AMD) have no generic local 1-minute file.
Thus historical sub-day context/follower proxies will usually fail closed;
daily context is widely available.

The cache files themselves carry no feed, adjustment, retrieval-time, or
vendor-publication metadata.  The *current builder configuration* requests
split-adjusted SIP bars (`strategies/momentum_expansion/config/momentum_config.py`),
while the generic `fetch_intraday` helper defaults to raw/IEX unless its caller
overrides it.  That is not enough to label any pre-existing Parquet with a
verified feed/adjustment.  Source metadata emitted by `lookup_context` therefore
reports both as unknown and includes SHA-256, exact path, row count, and
timestamp span.

Daily source timestamps are bar starts (often 04:00Z), not proof that the
daily session was complete then.  The helper maps daily bars to the matching
16:00 America/New_York regular-session close and excludes same-session bars
before that close.  It uses minute-bar end time (`start + 1 minute`) and
requires the expected immediately preceding completed minute; any gap blocks
the lookup rather than carrying a stale close forward.  These rules prevent
within-bar lookahead, but cannot turn a retrospectively fetched final cache
into certified PIT data.

## Option data: bars/trades, not executable historical quotes

`Data/options_history/bars/` contains:

| Timeframe | Contract-expiry Parquet files | Underlyings | Expiry-file span |
| --- | ---: | ---: | --- |
| `1Day` | 5,280 | 665 | 2024-02-02 to 2026-09-18 |
| `30Min` | 5,108 | 676 | 2025-05-30 to 2026-09-18 |
| `1Min` | 288 | 143 | 2026-05-01 to 2026-06-18 |

The final column describes **contract expiry names**, not continuous bar
coverage.  A representative 1-minute file has only 26 rows for a single
contract and schema `osi_symbol,c,h,l,n,o,t,v,vw`; daily and 30-minute files
have the same trade-bar fields.  Sidecars such as
`*.parquet.meta.json` record the requested/fetched time intervals per OCC
symbol--they do not contain bid, ask, quote condition, quote size, or a
vendor quote timestamp.

This matches the cache implementation's documented endpoint check:
`research/options_lab/chain_cache.py` says Alpaca historical
`/v1beta1/options/quotes` returned 404, whereas bars/trades were cached.  The
live client can call latest option-quote/snapshot endpoints
(`core/API/Alpaca_API/options/options_api.py`), and the forward-only SPY mark
capture writes source quote timestamps when an active position exists
(`strategies/spy_intraday/Policy/option_mark_capture.py`).  No historical mark
files were found under `Data/inference`; the only `quote_mode_proxy_*` files
are explicitly proxy artifacts, not quotes.

An account-scoped probe in this study returned HTTP 200 for historical SPY
IEX quotes and HTTP 404 for the dated historical option-quotes route tested;
this does not establish that every equity symbol/date is covered or that no
other vendor has historical option BBO. Actual historical OPRA entitlement was
**not verified** in this read-only assessment.
Code comments document that currently listed Alpaca option bars have produced
an OPRA-agreement 403 and that Schwab daily candles complement expired/live
coverage, but neither establishes that a historical quote entitlement is
available now.  A successful endpoint response must not be treated as quote
fitness.  If a data tier is later authorized, first retain and audit a small
sample of bid/ask timestamps, quote freshness, and option-return correlation
to the underlying before computing any option P&L.

## Recommended ledger fields and allowed outputs

For every message-derived trade/event, keep `message_id`, channel filename,
raw timestamp title, inferred display timezone, snowflake `timestamp_utc`,
conservative `available_at_utc`, parsed symbol/contract confidence, raw
text hash, and source-file hash.  For underlying context keep returned source
hash/path and label `certified_point_in_time=false`.

Allowed now: message lifecycle audit; candidate-symbol coverage audit;
underlying daily/minute return and volume context; and clearly labeled
underlying delayed-bar **proxies** where the exact completed minute exists.

Source-reported option entry/exit prices and screenshot average costs can be
catalogued as claims. The missing item is a market-based, delay-specific
option price, spread, fill-rate or P&L estimate. The existing 2026-07
retraction applies directly: option trade bars are prints, and an illiquid
contract's last print is not a mark.
