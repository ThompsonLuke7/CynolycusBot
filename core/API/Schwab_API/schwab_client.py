# schwab_client.py

import argparse
import json
from pathlib import Path
from datetime import datetime

from schwab.auth import client_from_manual_flow, client_from_token_file
from schwab.orders.equities import equity_buy_market, equity_sell_market
import httpx

# === FILL THESE IN ===
API_KEY = "DodBaa8AR1twJGwT5srKqFUdl6SepUnCwck38IGFlxc8QVt9"          # App Key from Schwab dev portal
APP_SECRET = "GAq7IMKHL5AiIJrZunI13WP7z7nGgAy0USgUxB0zZA3hTUP260kAJidKxjsAETBj"    # App Secret from Schwab dev portal
CALLBACK_URL = "https://127.0.0.1"     # EXACTLY matches your app's callback URL
TOKEN_PATH = Path(__file__).resolve().with_name("schwab_token.json")       # where schwab-py will store the token


def _backup_existing_token() -> Path | None:
    if not TOKEN_PATH.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = TOKEN_PATH.with_name(f"{TOKEN_PATH.stem}.invalid_{stamp}{TOKEN_PATH.suffix}")
    TOKEN_PATH.replace(backup_path)
    return backup_path


def _create_or_load_client(force_manual: bool = False):
    """
    Uses schwab-py's helpers to either:
    - Create a new token via a manual browser flow (first run), or
    - Load & refresh an existing token from disk (later runs).
    """
    if force_manual:
        backup_path = _backup_existing_token()
        if backup_path is not None:
            print(f"Backed up existing token file to: {backup_path}")
        client = client_from_manual_flow(
            api_key=API_KEY,
            app_secret=APP_SECRET,
            callback_url=CALLBACK_URL,
            token_path=str(TOKEN_PATH)
        )
    elif TOKEN_PATH.exists():
        # Reuse existing token file; schwab-py will refresh as needed
        client = client_from_token_file(
            token_path=str(TOKEN_PATH),
            api_key=API_KEY,
            app_secret=APP_SECRET
        )
    else:
        # First-time login: walks you through copy-paste OAuth flow in terminal
        client = client_from_manual_flow(
            api_key=API_KEY,
            app_secret=APP_SECRET,
            callback_url=CALLBACK_URL,
            token_path=str(TOKEN_PATH)
        )

    return client


class SchwabClient:
    def __init__(self, force_manual: bool = False):
        # This is a schwab.client.Client under the hood
        self.client = _create_or_load_client(force_manual=force_manual)

    # ================================
    # GET ACCOUNT HASHES
    # ================================
    def get_account_numbers(self):
        resp = self.client.get_account_numbers()
        # Raise nice error if Schwab returns anything other than 200
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Error getting account numbers: {e.response.status_code} {e.response.text}"
            ) from e

        return resp.json()

    # ================================
    # GET QUOTES
    # ================================
    def get_quotes(self, symbols):
        # schwab-py accepts list or string; normalize to list
        if isinstance(symbols, str):
            symbols = [symbols]

        resp = self.client.get_quotes(symbols)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Error getting quotes: {e.response.status_code} {e.response.text}"
            ) from e

        return resp.json()

    def get_stream_account_id(self, preferred_hash: str | None = None) -> str:
        accounts = self.get_account_numbers()
        if not isinstance(accounts, list) or not accounts:
            raise RuntimeError("No Schwab accounts available for stream authentication.")

        def _extract_account_number(item: object) -> str | None:
            if not isinstance(item, dict):
                return None
            raw = item.get("accountNumber") or item.get("accountId")
            if raw is None:
                return None
            text = str(raw).strip()
            return text or None

        preferred_hash_text = str(preferred_hash).strip() if preferred_hash is not None else ""
        if preferred_hash_text:
            for item in accounts:
                if not isinstance(item, dict):
                    continue
                if str(item.get("hashValue", "")).strip() != preferred_hash_text:
                    continue
                account_number = _extract_account_number(item)
                if account_number is not None:
                    return account_number

        for item in accounts:
            account_number = _extract_account_number(item)
            if account_number is not None:
                return account_number

        raise RuntimeError("Schwab accounts response did not include a usable account number.")

    # ================================
    # PLACE SIMPLE MARKET ORDER
    # ================================
    def place_order_market(self, account_hash, symbol, quantity, instruction="BUY"):
        """
        instruction = "BUY" or "SELL"
        quantity    = integer number of shares
        Uses schwab-py's equity order templates under the hood.
        """
        if instruction.upper() == "BUY":
            order_spec = equity_buy_market(symbol, quantity)
        elif instruction.upper() == "SELL":
            order_spec = equity_sell_market(symbol, quantity)
        else:
            raise ValueError("instruction must be 'BUY' or 'SELL'")

        resp = self.client.place_order(account_hash, order_spec)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Order failed: {e.response.status_code} {e.response.text}"
            ) from e

        # Schwab includes order location / id in headers
        return {
            "status": "submitted",
            "status_code": resp.status_code,
            "location": resp.headers.get("Location", "N/A"),
        }


# Quick smoke test if you run this file directly:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/load a Schwab client and optionally force fresh auth.")
    parser.add_argument(
        "--reauth",
        action="store_true",
        help="Ignore the current token file, back it up, and force a fresh browser login flow.",
    )
    args = parser.parse_args()

    c = SchwabClient(force_manual=args.reauth)
    accounts = c.get_account_numbers()
    print("Accounts:")
    print(json.dumps(accounts, indent=2))

    quotes = c.get_quotes(["SPY", "NVDA"])
    print("\nQuotes:")
    print(json.dumps(quotes, indent=2))
