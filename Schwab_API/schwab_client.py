# schwab_client.py

import os
import json

from schwab.auth import client_from_manual_flow, client_from_token_file
from schwab.orders.equities import equity_buy_market, equity_sell_market
import httpx

# === FILL THESE IN ===
API_KEY = "DodBaa8AR1twJGwT5srKqFUdl6SepUnCwck38IGFlxc8QVt9"          # App Key from Schwab dev portal
APP_SECRET = "GAq7IMKHL5AiIJrZunI13WP7z7nGgAy0USgUxB0zZA3hTUP260kAJidKxjsAETBj"    # App Secret from Schwab dev portal
CALLBACK_URL = "https://127.0.0.1"     # EXACTLY matches your app's callback URL
TOKEN_PATH = "schwab_token.json"       # where schwab-py will store the token


def _create_or_load_client():
    """
    Uses schwab-py's helpers to either:
    - Create a new token via a manual browser flow (first run), or
    - Load & refresh an existing token from disk (later runs).
    """
    if os.path.exists(TOKEN_PATH):
        # Reuse existing token file; schwab-py will refresh as needed
        client = client_from_token_file(
            token_path=TOKEN_PATH,
            api_key=API_KEY,
            app_secret=APP_SECRET
        )
    else:
        # First-time login: walks you through copy-paste OAuth flow in terminal
        client = client_from_manual_flow(
            api_key=API_KEY,
            app_secret=APP_SECRET,
            callback_url=CALLBACK_URL,
            token_path=TOKEN_PATH
        )

    return client


class SchwabClient:
    def __init__(self):
        # This is a schwab.client.Client under the hood
        self.client = _create_or_load_client()

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
    c = SchwabClient()
    accounts = c.get_account_numbers()
    print("Accounts:")
    print(json.dumps(accounts, indent=2))

    quotes = c.get_quotes(["SPY", "NVDA"])
    print("\nQuotes:")
    print(json.dumps(quotes, indent=2))
