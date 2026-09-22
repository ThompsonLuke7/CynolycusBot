#!/usr/bin/env python3
"""Read-only Alpaca FILL reconciliation; exits non-zero when reporting is unsafe."""
from __future__ import annotations

import argparse

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.broker_fill_reconciliation import reconcile_account


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env#PAPER")
    parser.add_argument("--account-label", default="paper")
    parser.add_argument("--after", help="ISO-8601 lower bound (defaults to last 7 days)")
    parser.add_argument("--until", help="ISO-8601 upper bound")
    args = parser.parse_args()
    certificate = reconcile_account(
        AlpacaOptionsClient(env_file=args.env_file), account_label=args.account_label,
        after=args.after, until=args.until,
    )
    print("broker reconciliation", certificate["status"],
          f"fills={certificate['fill_count']}",
          f"unattributed={len(certificate['unattributed_fills'])}",
          f"mismatches={len(certificate['ledger_mismatches'])}")
    return 0 if certificate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
