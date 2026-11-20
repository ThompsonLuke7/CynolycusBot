"""Fetch the paper account's full FILL history and non-trade activities (read-only).

The local fill journal (Data/inference/broker_reconciliation/paper/fills.jsonl)
starts 2026-09-14; the account was funded 2026-07-09. Raw broker JSON is written
once to Data/research/account_postmortem_20261103/ and never modified.

    PYTHONPATH=. .venv/bin/python research/account_postmortem_2026-11-03/00_fetch_broker_history.py
"""
from __future__ import annotations

import json
from pathlib import Path

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

OUT = Path("Data/research/account_postmortem_20261103")
NON_TRADE = ["OPEXP", "OPEXC", "MA", "DIV", "FEE", "JNLC", "REORG", "NC", "SC", "OPASN"]


def fetch_all(client: AlpacaOptionsClient, **params) -> list[dict]:
    rows, token = [], None
    while True:
        page = client.get_account_activities(direction="asc", page_size=100, page_token=token, **params) or []
        rows.extend(page)
        if len(page) < 100:
            return rows
        token = page[-1]["id"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = AlpacaOptionsClient(env_file=".env#PAPER")
    fills = fetch_all(client, activity_types="FILL", after="2026-07-01")
    (OUT / "broker_fills_20260701_20261103.json").write_text(json.dumps(fills))
    other = []
    for kind in NON_TRADE:
        other.extend(fetch_all(client, activity_types=kind, after="2026-07-01"))
    (OUT / "broker_nontrade_20260701_20261103.json").write_text(json.dumps(other))
    print(f"fills {len(fills)} ({fills[0]['transaction_time']} .. {fills[-1]['transaction_time']}); non-trade {len(other)}")


if __name__ == "__main__":
    main()
