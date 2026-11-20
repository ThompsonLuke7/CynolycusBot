"""Show or change the account-wide entry gate (core/trading_controls.py).

    PYTHONPATH=. .venv/bin/python scripts/trading_controls.py status
    PYTHONPATH=. .venv/bin/python scripts/trading_controls.py freeze --reason "..."
    PYTHONPATH=. .venv/bin/python scripts/trading_controls.py open [--max-gross 1.5] [--min-cushion 0.15]

Takes effect on the next order; no restart. ``status`` also reads the paper
account (read-only) so the limits can be compared with where the account is.
"""
from __future__ import annotations

import argparse
import datetime as dt

from core.trading_controls import controls_path, load_controls, write_controls


def _status(env_file: str) -> None:
    ctl = load_controls()
    print(f"file:    {controls_path()} ({'present' if controls_path().exists() else 'absent'})")
    print(f"entries: {'FROZEN' if ctl.entries_frozen else 'open'}{f' - {ctl.reason}' if ctl.reason else ''}")
    print(f"limits:  max_gross_leverage={ctl.max_gross_leverage}  min_margin_cushion_pct={ctl.min_margin_cushion_pct}")
    try:
        from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

        acct = AlpacaOptionsClient(env_file=env_file).get_account()
        equity = float(acct["equity"])
        gross = float(acct.get("long_market_value") or 0) + abs(float(acct.get("short_market_value") or 0))
        maint = float(acct.get("maintenance_margin") or 0)
        print(
            f"account: equity ${equity:,.0f}  gross {gross / equity:.2f}x  "
            f"maintenance ${maint:,.0f}  cushion {(equity - maint) / equity:.1%}  "
            f"buying_power ${float(acct.get('buying_power') or 0):,.0f}"
        )
    except Exception as exc:  # noqa: BLE001 - status must still print the file
        print(f"account: unreadable ({type(exc).__name__}: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=".env#PAPER")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    freeze = sub.add_parser("freeze", help="refuse every order that opens exposure")
    freeze.add_argument("--reason", required=True)
    opened = sub.add_parser("open", help="allow entries, optionally under account limits")
    opened.add_argument("--reason", default="")
    for p in (freeze, opened):
        p.add_argument("--max-gross", type=float, default=None, help="refuse entries at/above this gross leverage")
        p.add_argument("--min-cushion", type=float, default=None, help="refuse entries below this (equity-maintenance)/equity")
    args = parser.parse_args()

    if args.command != "status":
        path = write_controls(
            entries="frozen" if args.command == "freeze" else "open",
            reason=args.reason,
            set_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            max_gross_leverage=args.max_gross,
            min_margin_cushion_pct=args.min_cushion,
        )
        print(f"wrote {path}")
    _status(args.env_file)


if __name__ == "__main__":
    main()
