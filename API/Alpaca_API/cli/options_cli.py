from __future__ import annotations

import argparse
import json
from typing import Any

from ..options.options_api import AlpacaOptionsClient


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _add_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", default=".env", help="Path to .env with Alpaca creds.")
    parser.add_argument("--trading-base-url", default=None, help="Override trading base URL.")
    parser.add_argument("--data-base-url", default=None, help="Override data base URL.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")


def _build_client(args: argparse.Namespace) -> AlpacaOptionsClient:
    return AlpacaOptionsClient(
        env_file=args.env_file,
        trading_base_url=args.trading_base_url,
        data_base_url=args.data_base_url,
        timeout_sec=args.timeout,
    )


def _contracts_cmd(args: argparse.Namespace) -> None:
    client = _build_client(args)
    params = {
        "underlying_symbol": args.symbol,
        "expiration_date": args.expiration_date,
        "type": args.type,
        "strike_price_gte": args.strike_gte,
        "strike_price_lte": args.strike_lte,
        "limit": args.limit,
        "page_token": args.page_token,
    }
    resp = client.get_option_contracts(**{k: v for k, v in params.items() if v is not None})
    _print_json(resp)


def _quotes_cmd(args: argparse.Namespace) -> None:
    client = _build_client(args)
    params = {}
    if args.symbols:
        params["symbols"] = args.symbols
    elif args.symbol:
        params["symbols"] = args.symbol
    resp = client.get_option_quotes(**params)
    _print_json(resp)


def _order_cmd(args: argparse.Namespace) -> None:
    client = _build_client(args)
    resp = client.submit_option_order(
        symbol=args.symbol,
        qty=args.qty,
        side=args.side,
        order_type=args.order_type,
        time_in_force=args.time_in_force,
        limit_price=args.limit_price,
        stop_price=args.stop_price,
    )
    _print_json(resp)


def _format_cmd(args: argparse.Namespace) -> None:
    client = _build_client(args)
    symbol = client.format_option_symbol(
        underlying=args.underlying,
        expiration=args.expiration,
        call_put=args.call_put,
        strike=args.strike,
    )
    print(symbol)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alpaca options CLI (contracts, quotes, orders)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_contracts = sub.add_parser("contracts", help="Fetch option contracts.")
    _add_client_args(p_contracts)
    p_contracts.add_argument("--symbol", required=True, help="Underlying symbol, e.g. SPY")
    p_contracts.add_argument("--expiration-date", required=False, help="YYYY-MM-DD")
    p_contracts.add_argument("--type", required=False, choices=["call", "put"], help="Option type")
    p_contracts.add_argument("--strike-gte", type=float, default=None)
    p_contracts.add_argument("--strike-lte", type=float, default=None)
    p_contracts.add_argument("--limit", type=int, default=None)
    p_contracts.add_argument("--page-token", default=None)
    p_contracts.set_defaults(func=_contracts_cmd)

    p_quotes = sub.add_parser("quotes", help="Fetch option quotes.")
    _add_client_args(p_quotes)
    p_quotes.add_argument("--symbols", default=None, help="Comma-separated option symbols")
    p_quotes.add_argument("--symbol", default=None, help="Single option symbol")
    p_quotes.set_defaults(func=_quotes_cmd)

    p_order = sub.add_parser("order", help="Submit an option order.")
    _add_client_args(p_order)
    p_order.add_argument("--symbol", required=True, help="Option symbol, e.g. SPY240216C00475000")
    p_order.add_argument("--qty", type=int, required=True)
    p_order.add_argument("--side", required=True, choices=["buy", "sell"])
    p_order.add_argument("--order-type", default="market", choices=["market", "limit", "stop", "stop_limit"])
    p_order.add_argument("--time-in-force", default="day", choices=["day", "gtc", "opg", "cls", "ioc", "fok"])
    p_order.add_argument("--limit-price", type=float, default=None)
    p_order.add_argument("--stop-price", type=float, default=None)
    p_order.set_defaults(func=_order_cmd)

    p_format = sub.add_parser("format", help="Format an OCC option symbol.")
    _add_client_args(p_format)
    p_format.add_argument("--underlying", required=True)
    p_format.add_argument("--expiration", required=True, help="YYYYMMDD or YYMMDD")
    p_format.add_argument("--call-put", required=True, choices=["C", "P", "c", "p"])
    p_format.add_argument("--strike", type=float, required=True)
    p_format.set_defaults(func=_format_cmd)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
