# Forward Guidance Earnings Trader

V1 event-level pipeline for finding post-earnings opportunities where the
market reaction disagrees with historically bullish forward guidance traits.

The package mirrors the existing trading repo style:

- Alpaca market data comes through `API.Alpaca_API.market_data.fetch_intraday`.
- Event caches live under `events/forward_guidance/data/raw/{ticker}/{earnings_date}/`.
- Features, labels, training matrices, models, inference output, and backtests
  are cached locally as parquet/JSON.
- The dashboard is read-only; V1 does not submit orders.

## Event CSV

Minimum columns:

```csv
ticker,earnings_date,report_time
NVDA,2026-02-25,AMC
```

Optional columns include `sector`, `sector_etf`, `cik`, `fiscal_period`,
`source_url`, and `available_at`.

## Common Commands

```bash
python -m events.forward_guidance.main --stage discover-events --start 2025-01-01 --end 2026-02-01 --discovery-source sec --limit 10
python -m events.forward_guidance.main --stage ingest --events-csv events.csv --limit 3
python -m events.forward_guidance.main --stage fetch-market --events-csv events.csv --limit 3
python -m events.forward_guidance.main --stage features --events-csv events.csv
python -m events.forward_guidance.models.train --model-kind xgboost
python -m events.forward_guidance.inference.daily --events-csv today_events.csv
python -m UI.forward_guidance_dashboard
```

The `discover-events` stage writes:

- `events/forward_guidance/data/processed/discovered_earnings_events.csv`
- `events/forward_guidance/data/processed/earnings_events.parquet`

Then `ingest`, `fetch-market`, and `features` can run without `--events-csv`.

Discovery sources:

- `--discovery-source sec`: official SEC submissions, best for historical backfills.
- `--discovery-source yfinance`: optional Yahoo/yfinance earnings calendar, best for upcoming/recent scheduled dates.
- `--discovery-source both`: combines and deduplicates both.

By default, discovery skips ETFs/funds from the reusable swing universe because
they do not have operating-company earnings calls. Use `--include-funds` only
when you explicitly want to inspect fund filings.

Embedding and FinBERT features are optional and CPU-only:

```bash
python -m events.forward_guidance.main --stage features --events-csv events.csv --embeddings --finbert
```

Install optional NLP/model packages only when needed:

```bash
pip install transformers sentence-transformers lightgbm
```
