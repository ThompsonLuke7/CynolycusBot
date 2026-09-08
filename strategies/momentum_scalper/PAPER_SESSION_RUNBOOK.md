# Momentum Scalper Paper-Session Runbook

## Current gate

Do not enable paper submission until this is green:

```bash
python -m strategies.momentum_scalper.live.readiness --json
```

The preflight is read-only.  It must report `ready_for_paper_submission: true`.

## Required before tomorrow's session

1. In Alpaca, enable the paid **Algo Trader Plus** market-data subscription
   under **Plans & Features**, then rerun the preflight.  The current account
   can read IEX but is denied recent SIP quotes, so it cannot observe the full
   premarket market or execute the quote-aware strategy safely.
2. Configure a dedicated QA-paper runtime using the required fields in
   `.env.example`: `CYNOLYCUS_ENVIRONMENT=QA_PAPER`, GCS journal bucket, Cloud
   SQL instance/DSN, explicit paper account identity, and secret binding.
   Keep `CYNOLYCUS_SUBMIT_ENABLED=false` until the preflight is green and the
   session operator deliberately enables the controlled paper-submit run.
3. Provide live, timestamped sources for the premarket universe, float, and
   material catalyst.  The existing Polygon downloader is historical/on-demand
   and cannot satisfy this gate by itself.
4. Start data observation early enough to validate fresh SIP quotes, trades,
   and halt statuses for each selected candidate.  `MomentumSIPMarketFeed` is
   the data-only bridge for that selected list.

## Hard limits for the first controlled paper run

- Paper account and paper endpoint only; no live order route exists.
- One concurrent position, at most three entries, max 1R risk per entry, and a
  3R daily stop, as defined in `configs/momentum_scalper_v1.json`.
- Extended-hours entries remain DAY limit orders only.
- If SIP, catalyst, float, quote freshness, or halt status is unavailable,
  reject the candidate; do not substitute IEX, stale data, or a fabricated
  spread.

## Evidence to retain

Keep raw market events with vendor and receipt timestamps, scanner decisions,
order intents, broker acknowledgements/fills, and end-of-session
reconciliation.  A single green session is an operational check, not
performance validation.
